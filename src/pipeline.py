#!/usr/bin/env python3
import os
import polars as pl
import argparse
from dotenv import load_dotenv

from src.contextselection import ContextSelection
from src.filecontextselection import FileContextSelection
from src.prompt_generator import PromptGenerator
from src.llm_prompter import LLMPrompter
from src.utils.prompt_utils import clean_prompts, format_csv_for_presentation, determine_label, generate_explanation

def run_pipeline(df, **kwargs):
    """Run anomaly detection and explanation pipeline"""
    if kwargs["verbose"]:
        print(f"Running pipeline with threshold {kwargs['threshold']}...\n")
    
    # Context Selection
    if kwargs["verbose"]:
        print("=== Context Selection ===")
    
    try:
        if kwargs["anomaly_level"] == 'file':
            file_context_selector = FileContextSelection(file_path=kwargs["input"], filetype=kwargs["log_type"], anomaly_detection_method=kwargs["ad_method"])
            df_with_context = file_context_selector.get_context()

        elif kwargs["anomaly_level"] == 'line':
            context_selection = ContextSelection(
                selection_strategy="semantic", 
                df_path=None,
                column_name=kwargs["context_column"], 
                top_k_far=3, 
                top_k_near=2,
                drop_duplicates=False,
                verbose=kwargs["verbose"],
            )
            
            context_selection.set_in_memory_df(df)
            df_with_context = context_selection.getLexicalContext(df)
            
            if kwargs["verbose"]:
                print(f"Context selection complete.")
                anomalies = df_with_context.filter(pl.col('anomaly_score') >= kwargs["threshold"])
                print(f"Found {anomalies.shape[0]} anomalies with score >= {kwargs['threshold']}")
    
    except Exception as e:
        df_with_context = df.clone()
        
        for row in df.filter(pl.col('anomaly_score') >= kwargs["threshold"]).iter_rows(named=True):
            line_id = row['LineId']
            
            context_start = max(1, line_id - 5)
            context_end = min(df.shape[0], line_id + 5)
            
            df_with_context = df_with_context.with_columns(
                pl.when(pl.col('LineId') == line_id)
                .then(pl.lit(str(line_id)))
                .otherwise(pl.col('lexical_context'))
                .alias('lexical_context')
            )
            
            for ctx_id in range(context_start, context_end + 1):
                if ctx_id != line_id:
                    df_with_context = df_with_context.with_columns(
                        pl.when(pl.col('LineId') == ctx_id)
                        .then(pl.lit(str(line_id)))
                        .otherwise(pl.col('lexical_context'))
                        .alias('lexical_context')
                    )
    
    # Prompt Generation for Labels
    if kwargs["verbose"]:
        print("\n=== Prompt Generation for Labels ===")
    
    try:
        prompt_generator = PromptGenerator()
        df_with_prompts = prompt_generator.generateLabelPrompts(kwargs["threshold"], df_with_context)
        
        if kwargs["verbose"]:
            prompts_count = df_with_prompts.filter(pl.col('anomaly_label').is_not_null()).shape[0]
            print(f"Generated {prompts_count} label prompts")
    
    except Exception as e:
        return df_with_context
    
    if kwargs["test_mode"]:
        if kwargs["verbose"]:
            print("Test mode: Simulating LLM responses")
            print("\n=== LLM Prediction of Labels ===")
            # LLM Prediction of Labels            
        labels = []
        for row in df_with_prompts.iter_rows(named=True):
            if row.get('anomaly_label') is not None:
                message = row.get('message', row.get('m_message', ''))
                labels.append(determine_label(message, row.get('component', '')))
            else:
                labels.append(None)
        
        df_with_labels = df_with_prompts.with_columns(
            pl.Series(labels).alias('anomaly_result')
        )
        # Prompt Generation for Explanations
        if kwargs["verbose"]:
            print("\n=== Prompt Generation for Explanations ===")
        df_with_explanations = prompt_generator.generateExplanationPrompts(kwargs["threshold"], df_with_labels)
        
        # LLM Generation of Explanations
        if kwargs["verbose"]:
            print("\n=== LLM Prediction of Explanations ===")
        explanations = []
        for row in df_with_explanations.iter_rows(named=True):
            if row.get('explanation_prompt') is not None:
                label = row.get('anomaly_result', 'application')
                node = row.get('node', 'unknown node')
                component = row.get('component', 'unknown component')
                explanations.append(generate_explanation(label, node, component))
            else:
                explanations.append(None)
        
        final_df = df_with_explanations.with_columns(
            pl.Series(explanations).alias('explanation_result')
        )
    else:
        llm_prompter = LLMPrompter()
        
        if kwargs["verbose"]:
            print("Getting label responses from LLM...")
        df_with_labels = llm_prompter.getLabelResponses(df_with_prompts)
        if kwargs["verbose"]:
            print("\n=== Prompt Generation for Explanations ===")
        if kwargs["anomaly_level"] == 'file':
            df_with_explanations = prompt_generator.generate_file_explanation_prompt(filename=kwargs["input"], df_context=df_with_labels)
        else:
            df_with_explanations = prompt_generator.generateExplanationPrompts(kwargs["threshold"], df_with_labels)
        
        if kwargs["verbose"]:
            print("Getting explanation responses from LLM...")
        if kwargs["anomaly_level"] == 'file':
            final_df = llm_prompter.get_file_explanation_response(df_with_explanations)
        elif kwargs["anomaly_level"] == 'line':
            final_df = llm_prompter.getExplanationResponses(df_with_explanations)
    
    # Rename columns for the final output format
    final_df = final_df.with_columns(
        pl.col('anomaly_result').alias('anomaly_label_final'),
        pl.col('explanation_result').alias('anomaly_explanation')
    )
    
    if kwargs["clean_results"]:
        if kwargs["verbose"]:
            print("\n=== Cleaning Prompts ===")
        final_cleaned_df = clean_prompts(final_df)
        return final_cleaned_df
    
    return final_df

def prepare_data(df, args, verbose=False):
    """Prepare data for pipeline by mapping columns and ensuring correct formats"""
    column_map = {
        'pred_ano_proba': 'anomaly_score',
        'row_nr': 'LineId',
        'm_message': 'message',
        'label': 'true_label'  
    }
    
    if args.score_column and args.score_column != 'anomaly_score':
        column_map[args.score_column] = 'anomaly_score'
    if args.id_column and args.id_column != 'LineId':
        column_map[args.id_column] = 'LineId'
    if args.message_column and args.message_column != 'message':
        column_map[args.message_column] = 'message'
    
    for src, dst in column_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename({src: dst})
    
    # Process dash values in label column
    if 'label' in df.columns:
        df = df.with_columns(
            pl.when(pl.col('label') == "-")
            .then(pl.lit("normal"))
            .otherwise(pl.col('label'))
            .alias('label')
        )
    
    if 'true_label' in df.columns:
        df = df.with_columns(
            pl.when(pl.col('true_label') == "-")
            .then(pl.lit("normal"))
            .otherwise(pl.col('true_label'))
            .alias('true_label')
        )
    
    if 'lexical_context' not in df.columns:
        df = df.with_columns(pl.lit(None).alias('lexical_context'))
    
    if 'anomaly_score' not in df.columns:
        if 'anomaly' in df.columns:
            df = df.with_columns(
                pl.when(pl.col('anomaly') == True)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .alias('anomaly_score')
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias('anomaly_score'))
    
    try:
        df = df.with_columns(
            pl.when(pl.col('anomaly_score').cast(pl.Utf8) == 'true')
            .then(pl.lit(1.0))
            .when(pl.col('anomaly_score').cast(pl.Utf8) == 'false')
            .then(pl.lit(0.0))
            .otherwise(pl.col('anomaly_score'))
            .alias('anomaly_score')
        )
    except:
        pass
    
    if args.csv_input and os.path.exists(args.csv_input):
        try:
            anomaly_df = pl.read_csv(args.csv_input)
            
            id_col = None
            for col in ['row_nr', 'LineId']:
                if col in anomaly_df.columns:
                    id_col = col
                    break
            
            if id_col is not None:
                anomaly_ids = set(anomaly_df[id_col].to_list())
                df = df.with_columns(
                    pl.when(pl.col('LineId').is_in(anomaly_ids))
                    .then(pl.lit(1.0))
                    .otherwise(pl.col('anomaly_score'))
                    .alias('anomaly_score')
                )
        except Exception:
            pass
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Log Anomaly Explainer Pipeline')
    parser.add_argument('--input', type=str, default='src/data/bgl-demo-1.parquet', 
                        help='Input Parquet file path')
    parser.add_argument('--csv-input', type=str, help='Optional CSV file with known anomalies')
    parser.add_argument('--threshold', type=float, default=0.79, help='Anomaly score threshold')
    parser.add_argument('--score-column', type=str, default='pred_ano_proba', help='Anomaly score column name')
    parser.add_argument('--id-column', type=str, default='row_nr', help='ID column name')
    parser.add_argument('--message-column', type=str, default='m_message', help='Message column name')
    parser.add_argument('--context-column', type=str, default='e_message_normalized', help='Column name to be used for context selection')
    parser.add_argument('--log-type', choices=['BGL', 'LO2', 'mixed'], default='BGL')
    parser.add_argument('--anomaly-level', choices=['line', 'file'], default='line', help='Anomaly level, used for context selection')
    parser.add_argument('--ad-method', choices=['LOF', 'IF'], default='LOF', help='Anomaly detection method')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    parser.add_argument('--clean_results', action='store_true', help='Clean the prompt')
    parser.add_argument('--output', type=str, default='anomaly_results.csv', help='Output file path')
    args = parser.parse_args()
    
    load_dotenv()
    
    if not args.test_mode and not os.getenv('OPENROUTER_API_KEY'):
        print("API key not found in environment, using test mode")
        args.test = True
    
    try:
        print(f"Loading data from {args.input}")
        df = pl.read_parquet(args.input)
        print(f"Loaded {df.shape[0]} logs with {df.shape[1]} columns\n")
        
        df = prepare_data(df, args, verbose=args.verbose)
        
        # Show anomaly score distribution
        print("=== Anomaly Score Distribution ===")
        thresholds = [0.1, 0.5, 0.7, 0.75, 0.79, 0.8, 0.85, 0.9, 0.95, 1.0]
        for t in thresholds:
            count = df.filter(pl.col('anomaly_score') >= t).shape[0]
            percent = (count / df.shape[0]) * 100
            print(f"  Score >= {t:.2f}: {count} rows ({percent:.2f}%)")
    
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        return
    
    result_df = run_pipeline(
        df, 
        **args.__dict__,
    )
    
    format_csv_for_presentation(result_df, args.output)
    
    anomalies = result_df.filter(pl.col('anomaly_score') >= args.threshold)
    print(f"=== Pipeline Complete ===")
    print(f"Total logs: {result_df.shape[0]}")
    print(f"Anomalies: {anomalies.shape[0]}")
    print(f"Results saved to: {args.output}")
    
    return result_df

if __name__ == "__main__":
    main()