class PromptGenerator:
    def __init__(self):
        pass

    def generateLabelPrompts(self, treshold, df):
        #Generates prompts for getting labels from LLM, and adds this as a new column into the DF. Only generates them for lines whose anomaly score is larger or equal to treshold. 
        pass


    def generateExplanationPrompts(self, threshold, df):
        """
        Generates prompts for getting explanations from an LLM and adds them as a new column in the DataFrame.
        Prompts are generated only for lines with an anomaly score greater than or equal to the threshold.

        Parameters:
        - threshold (float): The anomaly score threshold for identifying anomalous lines.
        - df (pd.DataFrame): DataFrame containing log data with columns including 'anomaly_score',
                             'LineId', 'lexical_context', and optionally 'anomaly_label'.

        Returns:
        - df (pd.DataFrame): Updated DataFrame with a new 'explanation_prompt' column for anomalous lines.
        """

        # Get all column headers
        headers = df.columns.tolist()

        # Define fixed few-shot examples
        example_normal = ""
        explanation_normal = ""
        example_anomalous = ""
        explanation_anomalous = ""

        # Initialize 'explanation_prompt' column
        df['explanation_prompt'] = None

        # Filter anomalous rows
        anomaly_rows = df[df['anomaly_score'] >= threshold]

        # Check if anomaly labels are needed but not present
        if 'anomaly_label' not in df.columns or anomaly_rows['anomaly_label'].isna().all():
            # Call generateLabelPrompts to potentially populate labels
            df = self.generateLabelPrompts(threshold, df)
        # Note: If generateLabelPrompts is not implemented or doesn't populate 'anomaly_label',
        # the label will be omitted from prompts where it's missing

        for idx, row in anomaly_rows.iterrows():
            anomaly_lineid = row['LineId']

            # Format the anomalous log line
            anomalous_log_str = "; ".join([f"{header}: {row[header]}" for header in headers])

            # Retrieve context lines using 'lexical_context'
            context_df = df[df['lexical_context'] == anomaly_lineid]
            context_str = "No context lines available." if context_df.empty else "\n".join(
                ["; ".join([f"{header}: {context_row[header]}" for header in headers]) for _, context_row in context_df.iterrows()]
            )

            # Include anomaly label if present
            label_str = ""
            if 'anomaly_label' in df.columns and not pd.isna(row['anomaly_label']):
                label_str = f"Anomaly Label: {row['anomaly_label']}"

            # Construct the prompt with few-shot examples
            prompt = f"""
Below are examples of log lines with explanations.

Example 1:
Log Line: {example_normal}
Explanation: {explanation_normal}

Example 2:
Log Line: {example_anomalous}
Explanation: {explanation_anomalous}

Now, explain the following anomalous log line, considering its context:

Anomalous Log Line: {anomalous_log_str}
{label_str}
Context Lines:
{context_str}

Provide a natural language explanation for why this log line is considered anomalous, including possible root causes, and explain the label if provided.
"""
            # Assign the prompt to the DataFrame
            df.at[idx, 'explanation_prompt'] = prompt

        return df

