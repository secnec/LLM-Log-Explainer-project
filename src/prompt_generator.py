from utils.prompts import DEFAULT_EXPLANATION_PROMPT, DEFAULT_LABEL_PROMPT
import polars as pl

class PromptGenerator:
    def __init__(self):
        # Default prompt templates
        self.explanation_prompt = DEFAULT_EXPLANATION_PROMPT
        self.label_prompt = DEFAULT_LABEL_PROMPT

    def generateLabelPrompts(self, threshold, df, prompt_template=None):
        """
        Generates prompts for getting labels and adds them as a new column in the DataFrame.
        Prompts are generated only for lines with an anomaly score greater than or equal to the threshold.

        Parameters:
        - threshold (float): The anomaly score threshold for identifying anomalous lines.
        - df (pl.DataFrame): Polars DataFrame containing log data with columns including 'anomaly_score',
                             'LineId', 'lexical_context', and optionally other headers.
        - prompt_template (str, optional): Custom prompt template to use. Defaults to self.label_prompt.

        Returns:
        - df (pl.DataFrame): Updated DataFrame with a new 'anomaly_label' column for anomalous lines containing prompts.
        """
        headers = df.columns

        # Initialize the 'anomaly_label' column with null
        df = df.with_columns(pl.lit(None).alias('anomaly_label'))

        # Filter rows where anomaly score meets or exceeds the threshold
        anomaly_rows = df.filter(pl.col('anomaly_score') >= threshold)

        # Use default prompt if none provided
        if prompt_template is None:
            prompt_template = self.label_prompt

        # Process each anomalous row
        for row in anomaly_rows.iter_rows(named=True):
            idx = row['LineId']  # Using LineId as identifier; adjust if Polars needs index differently
            anomaly_lineid = row['LineId']

            # Format the log line as semicolon-separated key-value pairs
            log_str = "; ".join([f"{header}: {row[header]}" for header in headers])

            # Get context lines (if any) based on lexical_context
            context_df = df.filter(pl.col('lexical_context') == anomaly_lineid)
            context_str = "No context lines available." if context_df.is_empty() else "\n".join(
                ["; ".join([f"{header}: {context_row[header]}" for header in headers]) 
                 for context_row in context_df.iter_rows(named=True)]
            )
            # Populate the prompt with the log line and context
            prompt = prompt_template.format(
                log_str=log_str,
                context_str=context_str
            )

            # Update the DataFrame with the prompt
            df = df.with_columns(
                pl.when(pl.col('LineId') == idx)
                .then(pl.lit(prompt))
                .otherwise(pl.col('anomaly_label'))
                .alias('anomaly_label')
            )

        return df

    def generateExplanationPrompts(self, threshold, df, prompt_template=None):
        """
        Generates prompts for getting explanations from an LLM and adds them as a new column in the DataFrame.
        Prompts are generated only for lines with an anomaly score greater than or equal to the threshold.

        Parameters:
        - threshold (float): The anomaly score threshold for identifying anomalous lines.
        - df (pl.DataFrame): Polars DataFrame containing log data with columns including 'anomaly_score',
                             'LineId', 'lexical_context', 'anomaly_label' (prompts from generateLabelPrompts),
                             and optionally 'anomaly_label' with actual labels (external).
        - prompt_template (str, optional): Custom prompt template to use. Defaults to self.explanation_prompt.

        Returns:
        - df (pl.DataFrame): Updated DataFrame with a new 'explanation_prompt' column for anomalous lines.
        """
        headers = df.columns

        # Define fixed few-shot examples
        example_normal = ""
        explanation_normal = ""
        example_anomalous = ""
        explanation_anomalous = ""

        # Initialize the 'explanation_prompt' column with null
        df = df.with_columns(pl.lit(None).alias('explanation_prompt'))

        # Filter rows where anomaly score meets or exceeds the threshold
        anomaly_rows = df.filter(pl.col('anomaly_score') >= threshold)

        # Ensure label prompts are generated if not already present
        if 'anomaly_label' not in df.columns or anomaly_rows['anomaly_label'].is_null().all():
            df = self.generateLabelPrompts(threshold, df)
            anomaly_rows = df.filter(pl.col('anomaly_score') >= threshold)

        if prompt_template is None:
            prompt_template = self.explanation_prompt

        # Process each anomalous row
        for row in anomaly_rows.iter_rows(named=True):
            idx = row['LineId']  # Using LineId as identifier
            anomaly_lineid = row['LineId']

            anomalous_log_str = "; ".join([f"{header}: {row[header]}" for header in headers])

            context_df = df.filter(pl.col('lexical_context') == anomaly_lineid)
            context_str = "No context lines available." if context_df.is_empty() else "\n".join(
                ["; ".join([f"{header}: {context_row[header]}" for header in headers]) 
                 for context_row in context_df.iter_rows(named=True)]
            )

            # Use 'anomaly_label' for label_str if it’s an actual label (not a prompt), else default
            label_str = "Anomaly Label: unknown"  # Default
            if 'anomaly_label' in df.columns and row['anomaly_label'] is not None:
                label_str = f"Anomaly Label: {row['anomaly_label']}" if not row['anomaly_label'].startswith("Below are examples") else "Anomaly Label: unknown"

            prompt = prompt_template.format(
                example_normal=example_normal,
                explanation_normal=explanation_normal,
                example_anomalous=example_anomalous,
                explanation_anomalous=explanation_anomalous,
                anomalous_log_str=anomalous_log_str,
                label_str=label_str,
                context_str=context_str
            )

            # Update the DataFrame with the prompt
            df = df.with_columns(
                pl.when(pl.col('LineId') == idx)
                .then(pl.lit(prompt))
                .otherwise(pl.col('explanation_prompt'))
                .alias('explanation_prompt')
            )

        return df

    def getContextLines(self, df):
        # Placeholder for context line retrieval
        pass