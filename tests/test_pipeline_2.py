#!/usr/bin/env python3
import os
import polars as pl
import argparse
from dotenv import load_dotenv

from src.contextselection import ContextSelection
from src.prompt_generator import PromptGenerator
from src.llm_prompter import LLMPrompter

from src.utils.log_generator import generate_supercomputer_logs
from src.utils.prompt_utils import clean_prompts, format_csv_for_presentation, determine_label, generate_explanation

def run_pipeline(df, threshold=0.8, verbose=True, test_mode=True, clean_results=True):
    """Run the complete anomaly detection pipeline"""
    if verbose:
        print(f"Running pipeline with threshold {threshold}...")
        print(f"Input DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 1: Context Selection
    if verbose:
        print("\n=== Context Selection ===")
    
    try:
        context_selection = ContextSelection(
            selection_strategy="semantic", 
            df_path=None,
            column_name="e_message_normalized", 
            top_k_far=3, 
            top_k_near=2,
            drop_duplicates=False
        )
        
        context_selection.set_in_memory_df(df)
        df_with_context = context_selection.get_context(df)
        
        if verbose:
            print(f"Context selection complete.")
            anomalies = df_with_context.filter(pl.col('anomaly_score') >= threshold)
            print(f"Found {anomalies.shape[0]} anomalies with score >= {threshold}")
    
    except Exception as e:
        print(f"Error during context selection: {e}")
        print("Using simplified context selection...")
        
        # Simplified context: just use nearby logs
        df_with_context = df.clone()
        
        # Add context_ids for anomalies
        for row in df.filter(pl.col('anomaly_score') >= threshold).iter_rows(named=True):
            line_id = row['LineId']
            
            # Find nearby rows (up to 5 before and after)
            context_start = max(1, line_id - 5)
            context_end = min(df.shape[0], line_id + 5)
            
            # Update context_ids_ref for the anomaly and context lines
            df_with_context = df_with_context.with_columns(
                pl.when(pl.col('LineId') == line_id)
                .then(pl.lit(str(line_id)))
                .otherwise(pl.col('context_ids_ref'))
                .alias('context_ids_ref')
            )
            
            for ctx_id in range(context_start, context_end + 1):
                if ctx_id != line_id:  # Skip the anomaly itself
                    df_with_context = df_with_context.with_columns(
                        pl.when(pl.col('LineId') == ctx_id)
                        .then(pl.lit(str(line_id)))
                        .otherwise(pl.col('context_ids_ref'))
                        .alias('context_ids_ref')
                    )
    
    # Step 2: Prompt Generation
    if verbose:
        print("")
        print("=== Prompt Generation ===")
    
    try:
        prompt_generator = PromptGenerator()
        df_with_prompts = prompt_generator.generateLabelPrompts(threshold, df_with_context)
        
        if verbose:
            prompts_count = df_with_prompts.filter(pl.col('anomaly_label').is_not_null()).shape[0]
            print(f"Generated {prompts_count} label prompts")
    
    except Exception as e:
        print(f"Error during prompt generation: {e}")
        return df_with_context
    
    # Step 3: LLM Prompting
    if verbose:
        print("")
        print("=== LLM Integration ===")
    
    if test_mode:
        if verbose:
            print("")
            print("Test mode: Simulating LLM responses")
        
        # Simulate labels based on log content
        labels = []
        for row in df_with_prompts.iter_rows(named=True):
            if row.get('anomaly_label') is not None:
                labels.append(determine_label(row['message'], row.get('component', '')))
            else:
                labels.append(None)
        
        df_with_labels = df_with_prompts.with_columns(
            pl.Series(labels).alias('anomaly_result')
        )
        
        # Generate explanation prompts
        try:
            df_with_explanations = prompt_generator.generateExplanationPrompts(threshold, df_with_labels)
        except Exception as e:
            print(f"Error generating explanation prompts: {e}")
            df_with_explanations = df_with_labels.with_columns(
                pl.when(pl.col('anomaly_result').is_not_null())
                .then(pl.lit("Please explain this anomaly."))
                .otherwise(pl.lit(None))
                .alias('explanation_prompt')
            )
        
        # Generate explanations
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
        # Real mode with API calls
        try:
            llm_prompter = LLMPrompter()
            
            if verbose:
                print("Getting label responses from LLM...")
            df_with_labels = llm_prompter.getLabelResponses(df_with_prompts)
            
            df_with_explanations = prompt_generator.generateExplanationPrompts(threshold, df_with_labels)
            
            if verbose:
                print("Getting explanation responses from LLM...")
            final_df = llm_prompter.getExplanationResponses(df_with_explanations)
            
        except Exception as e:
            print(f"Error during LLM processing: {e}")
            print("Falling back to test mode")
            return run_pipeline(df, threshold, verbose, test_mode=True, clean_results=clean_results)
    
    # Step 4: Clean up prompts if requested
    if clean_results:
        if verbose:
            print("")
            print("=== Cleaning Prompts ===")
        final_cleaned_df = clean_prompts(final_df)
        
        if verbose:
            print("Prompt cleaning complete")
        
        return final_cleaned_df
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Log Anomaly Explainer Pipeline')
    parser.add_argument('--logs', type=int, default=50)
    parser.add_argument('--anomaly-ratio', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--log-type', choices=['mixed', 'hadoop', 'bgl'], default='mixed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', action='store_true', help='Run in test mode without API calls')
    parser.add_argument('--no-clean', action='store_true', help='Do not clean prompts')
    parser.add_argument('--output', type=str, default='anomaly_results.csv', help='Output CSV file path')
    args = parser.parse_args()
    
    load_dotenv()
    
    if not args.test and not os.getenv('OPENROUTER_API_KEY'):
        print("API KEY not found in env, using test mode")
        args.test = True
    
    # Generate synthetic data
    df = generate_supercomputer_logs(
        num_logs=args.logs, 
        anomaly_ratio=args.anomaly_ratio,
        log_type=args.log_type
    )
    
    # Run the pipeline
    result_df = run_pipeline(
        df, 
        threshold=args.threshold, 
        verbose=args.verbose, 
        test_mode=args.test,
        clean_results=not args.no_clean
    )
    
    # Format and save results
    df_for_presentation = format_csv_for_presentation(result_df, args.output)
    
    # Print summary
    anomalies = result_df.filter(pl.col('anomaly_score') >= args.threshold)
    print("")
    print(f"=== Pipeline Complete ===")
    print(f"Total logs: {result_df.shape[0]}\n")
    print(f"Anomalies: {anomalies.shape[0]}\n")

    print(f"Complete results saved to: {args.output}")
    
    return result_df

if __name__ == "__main__":
    main()