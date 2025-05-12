#!/usr/bin/env python3
"""
Evaluation script for labeling.
This script loads test data, generates labels and explanations,
and evaluates the performance of the labeling system.
"""

from src.prompt_generator import PromptGenerator
from src.llm_prompter import LLMPrompter
import os
import sys
import time
import pandas as pd
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_dataset(file_path):
    """
    Load the test dataset from an Excel file.
    
    Parameters:
    - file_path (str): Path to the Excel file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the test data.
    """
    print(f"Loading dataset from {file_path}")

    try:
        df = pd.read_excel(file_path)

        # Print dataset info
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")

        # Preview
        print("\nPreview of first 3 rows:")
        print(df.head(3))

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using default columns for testing...")

        # Create a minimal test DataFrame if loading fails
        test_df = pd.DataFrame({
            'text': [
                "081111 090241 18 warn dfs.fs dataset: unexpected error trying to delete block blk_6566051927569845875",
                "081111 090159 18 warn dfs.fs dataset: unexpected error trying to delete block blk_7534533660812792996",
                "081112 120530 15 error network.connect: connection timed out while attempting to reach host 192.168.1.5"
            ],
            'LineId': [0, 1, 2],
            'anomaly_score': [1.0, 1.0, 1.0]
        })

        print(
            f"Created test DataFrame: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        return test_df


def preprocess_dataset(df):
    """
    Preprocess the dataset to match the expected format for the labeler.
    
    Parameters:
    - df (pd.DataFrame): Original DataFrame.
    
    Returns:
    - pl.DataFrame: Preprocessed Polars DataFrame ready for the labeler.
    """
    print("\nPreprocessing dataset...")

    # Copy to avoid modifying the original
    processed_df = df.copy()

    # Ensure required columns exist
    if 'text' not in processed_df.columns and 'm_message' in processed_df.columns:
        processed_df['text'] = processed_df['m_message']

    if 'LineId' not in processed_df.columns:
        processed_df['LineId'] = processed_df.index

    # Add anomaly score if not present (set all to 1.0 to evaluate the entire dataset)
    if 'anomaly_score' not in processed_df.columns:
        if 'pred_ano_proba' in processed_df.columns:
            processed_df['anomaly_score'] = processed_df['pred_ano_proba']
        else:
            print("Setting all anomaly scores to 1.0 to evaluate the entire dataset...")
            # Set all scores to 1.0 to ensure all records are evaluated
            processed_df['anomaly_score'] = 1.0

    # Set lexical context if not present
    if 'context_ids_ref' not in processed_df.columns:
        # For simplicity, each log is its own context
        processed_df['context_ids_ref'] = processed_df['LineId'].astype(str)

    # Handle label column naming - check for 'labels' column and rename to 'label'
    if 'label' not in processed_df.columns:
        if 'labels' in processed_df.columns:
            print("Found 'labels' column, renaming to 'label' for consistency")
            processed_df['label'] = processed_df['labels']
        elif 'anomaly_type' in processed_df.columns:
            processed_df['label'] = processed_df['anomaly_type']

    # Convert label to lowercase for consistency
    if 'label' in processed_df.columns:
        processed_df['label'] = processed_df['label'].str.lower()

    # Convert pandas DataFrame to polars DataFrame
    try:
        print("Converting pandas DataFrame to polars DataFrame...")
        polars_df = pl.from_pandas(processed_df)
        print(
            f"Preprocessing complete. Polars DataFrame shape: {polars_df.shape}")
        return polars_df
    except Exception as e:
        print(f"Error converting to polars DataFrame: {e}")
        raise


def run_evaluation(df, anomaly_threshold=0.0, sample_size=None):
    """
    Run the evaluation process on the dataset.
    Generates labels and compares to ground truth in the 'label' column.
    
    Parameters:
    - df (pl.DataFrame): Preprocessed Polars DataFrame.
    - anomaly_threshold (float): Threshold for anomaly detection.
    - sample_size (int, optional): Number of rows to sample for quicker evaluation.
    
    Returns:
    - dict: Evaluation results and metrics.
    """
    print(f"\nRunning evaluation with anomaly threshold: {anomaly_threshold}")

    # Sample the dataset if requested
    if sample_size and sample_size < df.shape[0]:
        print(f"Sampling {sample_size} records for evaluation...")
        df_sample = df.sample(sample_size, seed=42)
    else:
        df_sample = df

    # Initialize the components
    start_time = time.time()
    prompter = LLMPrompter()
    generator = PromptGenerator()

    # Step 1: Generate label prompts
    print("\nStep 1: Generating label prompts...")
    df_with_prompts = generator.generateLabelPrompts(
        anomaly_threshold, df_sample)
    print(
        f"Generated label prompts for {df_with_prompts.filter(pl.col('anomaly_label').is_not_null()).shape[0]} logs")

    # Step 2: Get labels from LLM
    print("\nStep 2: Getting labels from LLM...")
    final_df = prompter.getLabelResponses(df_with_prompts)
    print(
        f"Generated labels for {final_df.filter(pl.col('anomaly_result').is_not_null()).shape[0]} logs")

    # Processing time
    processing_time = time.time() - start_time
    print(f"\nEvaluation completed in {processing_time:.2f} seconds")

    # Verify we have a 'label' column for evaluation
    if 'label' not in final_df.columns:
        print("Warning: No 'label' column found for evaluation.")
        if 'labels' in df.columns:
            print("Found 'labels' column in original dataset. This should have been converted to 'label' during preprocessing.")
    else:
        print(f"Found 'label' column for evaluation. Now computing metrics...")

    # Compute metrics
    results = compute_metrics(final_df)
    results['processing_time'] = processing_time
    results['final_df'] = final_df

    return results


def compute_metrics(df):
    """
    Compute evaluation metrics.
    
    Parameters:
    - df (pl.DataFrame): Polars DataFrame with predicted labels.
    
    Returns:
    - dict: Dictionary of evaluation metrics.
    """
    print("\nComputing evaluation metrics...")

    # Filter rows with predictions
    predicted_rows = df.filter(pl.col('anomaly_result').is_not_null())
    print(f"Found {predicted_rows.shape[0]} rows with predictions")

    # If no predictions, return empty metrics
    if predicted_rows.shape[0] == 0:
        print("Warning: No predictions found.")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'predictions_count': 0}

    # Convert to Pandas for sklearn compatibility
    pred_df = predicted_rows.to_pandas()

    # Check if label column exists
    if 'label' not in pred_df.columns:
        print("Warning: No true labels found for comparison.")
        # Return basic metrics without comparison
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'predictions_count': len(pred_df),
            'class_distribution': pred_df['anomaly_result'].value_counts().to_dict()
        }

    # Get true and predicted labels
    y_true = pred_df['label'].tolist()
    y_pred = pred_df['anomaly_result'].tolist()

    # Print some examples to verify
    print("\nSample comparisons (True vs Predicted):")
    for i in range(min(5, len(y_true))):
        print(f"  {y_true[i]} vs {y_pred[i]}")

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Compute confusion matrix
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Compute per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Create per-class metrics dictionary
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': per_class_precision[i],
            'recall': per_class_recall[i],
            'f1': per_class_f1[i],
            'count': list(y_true).count(label)
        }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClass distribution:")
    for label, metrics in per_class_metrics.items():
        print(f"  {label}: {metrics['count']} items, F1={metrics['f1']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': labels,
        'per_class_metrics': per_class_metrics,
        'predictions_count': len(y_pred),
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, labels, save_path=None):
    """
    Plot and save the confusion matrix.
    
    Parameters:
    - cm (numpy.ndarray): Confusion matrix.
    - labels (list): Label names.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_metrics(metrics, save_path=None):
    """
    Plot per-class precision, recall, and F1 scores.
    
    Parameters:
    - metrics (dict): Metrics dictionary from compute_metrics.
    - save_path (str, optional): Path to save the plot.
    """
    # Check if we have per-class metrics
    if 'per_class_metrics' not in metrics:
        print("No per-class metrics available for plotting.")
        return

    per_class = metrics['per_class_metrics']
    labels = list(per_class.keys())

    precision = [per_class[label]['precision'] for label in labels]
    recall = [per_class[label]['recall'] for label in labels]
    f1 = [per_class[label]['f1'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()


def save_results(results, output_dir='evals/results'):
    """
    Save the evaluation results and plots.
    
    Parameters:
    - results (dict): Results from evaluation.
    - output_dir (str): Directory to save results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix plot
    if 'confusion_matrix' in results and 'labels' in results:
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            results['confusion_matrix'], results['labels'], cm_path)

    # Save metrics plot
    if 'per_class_metrics' in results:
        metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
        plot_metrics(results, metrics_path)

    # Save detailed results to CSV
    if 'final_df' in results:
        df = results['final_df']
        # Filter rows with predictions
        predicted_rows = df.filter(pl.col('anomaly_result').is_not_null())

        # Convert to pandas for easier saving
        results_df = predicted_rows.to_pandas()

        # Add 'correct' column if we have true labels
        if 'label' in results_df.columns:
            results_df['correct'] = results_df['label'] == results_df['anomaly_result']

        # Select columns for the results CSV - removing explanation columns
        columns_to_save = ['LineId', 'text', 'anomaly_score', 'anomaly_result']
        if 'label' in results_df.columns:
            columns_to_save.extend(['label', 'correct'])

        columns_to_save = [
            col for col in columns_to_save if col in results_df.columns]

        results_df[columns_to_save].to_csv(os.path.join(
            output_dir, 'detailed_results.csv'), index=False)
        print(
            f"Detailed results saved to {os.path.join(output_dir, 'detailed_results.csv')}")

    # Save summary metrics to text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Labeling Evaluation Results\n")
        f.write("===========================================\n\n")

        if 'class_distribution' in results:
            f.write("Classification Distribution (no true labels available):\n")
            for label, count in results['class_distribution'].items():
                f.write(
                    f"  {label}: {count} ({count/results['predictions_count']*100:.1f}%)\n")
        else:
            f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Precision: {results.get('precision', 'N/A'):.4f}\n")
            f.write(f"Recall: {results.get('recall', 'N/A'):.4f}\n")
            f.write(f"F1 Score: {results.get('f1', 'N/A'):.4f}\n")

        f.write(
            f"Processing Time: {results.get('processing_time', 'N/A'):.2f} seconds\n")
        f.write(
            f"Number of Predictions: {results.get('predictions_count', 'N/A')}\n\n")

        if 'per_class_metrics' in results:
            f.write("Per-Class Metrics:\n")
            for label, metrics in results['per_class_metrics'].items():
                f.write(f"  {label}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"    Count: {metrics['count']}\n\n")

    print(
        f"Summary metrics saved to {os.path.join(output_dir, 'summary.txt')}")

    # Save summary metrics to text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Labeling Evaluation Results\n")
        f.write("===========================================\n\n")
        f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"Precision: {results.get('precision', 'N/A'):.4f}\n")
        f.write(f"Recall: {results.get('recall', 'N/A'):.4f}\n")
        f.write(f"F1 Score: {results.get('f1', 'N/A'):.4f}\n")
        f.write(
            f"Processing Time: {results.get('processing_time', 'N/A'):.2f} seconds\n")
        f.write(
            f"Number of Predictions: {results.get('predictions_count', 'N/A')}\n\n")

        f.write("Per-Class Metrics:\n")
        for label, metrics in results.get('per_class_metrics', {}).items():
            f.write(f"  {label}:\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"    Count: {metrics['count']}\n\n")

    print(
        f"Summary metrics saved to {os.path.join(output_dir, 'summary.txt')}")


def main():
    data_dir = os.path.join('evals', 'data')
    dataset_path = os.path.join(data_dir, 'fc_public_test.xlsx')
    output_dir = os.path.join('evals', 'results')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("Labeling Evaluation")
    print("=" * 50)

    # Load and preprocess the dataset
    try:
        df = load_dataset(dataset_path)
        processed_df = preprocess_dataset(df)

        # Run evaluation
        print("\nStarting evaluation pipeline...")
        # Setting threshold to 0.0 to evaluate the entire dataset
        results = run_evaluation(processed_df, anomaly_threshold=0.0)

        # Save results
        save_results(results, output_dir)

        print("\nEvaluation completed successfully!")
        print(f"Results saved to {output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
