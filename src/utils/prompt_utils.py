import polars as pl
import csv
import random

def clean_prompts(df):
    """Clean up prompt columns after LLM responses have been generated"""
    
    # Check if we have results
    has_label_results = 'anomaly_result' in df.columns
    has_explanation_results = 'explanation_result' in df.columns
    
    # Clone the dataframe to avoid modifying the original
    cleaned_df = df.clone()
    
    # If we have label results, clean up the prompts
    if has_label_results and 'anomaly_label' in df.columns:
        cleaned_df = cleaned_df.with_columns(
            pl.col('anomaly_result').alias('anomaly_label_final')
        )
        cleaned_df = cleaned_df.drop('anomaly_label')
        if 'anomaly_result' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop('anomaly_result')
    
    # Similarly for explanation results
    if has_explanation_results and 'explanation_prompt' in df.columns:
        cleaned_df = cleaned_df.with_columns(
            pl.col('explanation_result').alias('anomaly_explanation')
        )
        cleaned_df = cleaned_df.drop('explanation_prompt')
        if 'explanation_result' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop('explanation_result')
    
    return cleaned_df

def format_csv_for_presentation(df, output_file):
    """Format CSV for better presentation"""
    # Get columns in the desired order
    columns = [
        'LineId', 'timestamp', 'node', 'level', 'component', 'message', 
        'anomaly_score'
    ]
    
    # Add result columns if they exist
    if 'anomaly_label_final' in df.columns:
        columns.append('anomaly_label_final')
    elif 'anomaly_result' in df.columns:
        columns.append('anomaly_result')
        
    if 'anomaly_explanation' in df.columns:
        columns.append('anomaly_explanation')
    elif 'explanation_result' in df.columns:
        columns.append('explanation_result')
    
    # Select only the columns we want, in the right order
    selected_df = df.select([col for col in columns if col in df.columns])
    
    # Convert to pandas for better CSV formatting
    pdf = selected_df.to_pandas()
    
    # Round anomaly scores
    if 'anomaly_score' in pdf.columns:
        pdf['anomaly_score'] = pdf['anomaly_score'].round(3)
    
    # Write to CSV
    pdf.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"Formatted results saved to {output_file}")
    return pdf

def determine_label(message, component=""):
    """Determine the anomaly label based on log content"""
    message = message.lower()
    component = component.lower()
    
    if "memory" in message or "out of memory" in message or "ecc error" in message:
        return "memory"
    
    elif "database" in message or "namenode" in component or "datanode" in component:
        return "database" if "database" in message else "application"
    
    elif "authentication" in message or "permission denied" in message or "login" in message:
        return "authentication"
    
    elif ("network" in message or "network" in component or "connection" in message or 
          "timeout" in message or "link" in message):
        return "network"
    
    elif ("file" in message or "io" in message or "io" in component or 
          "block" in message or "filesystem" in component):
        return "io"
    
    else:
        return "application"

def generate_explanation(label, node="unknown", component="unknown"):
    """Generate a realistic explanation for testing"""
    explanation_templates = {
        "memory": [
            "Memory issue detected in {component}. This type of error typically impacts system stability and is often caused by hardware failures or resource exhaustion.",
            "Critical memory anomaly detected on {node}. This suggests memory allocation failures or insufficient capacity, requiring immediate attention.",
            "Memory subsystem error in {component}. This is likely caused by hardware degradation or memory leaks in the application."
        ],
        "database": [
            "Database-related anomaly on {node}. This indicates connectivity or query execution problems that may impact data availability.",
            "The database subsystem is reporting errors in {component}. This suggests resource contention or configuration issues.",
            "Database operation failed. This may result in data inconsistency or application failures if not addressed quickly."
        ],
        "authentication": [
            "Authentication failure detected for {component}. This could potentially indicate security concerns or expired credentials.",
            "Security-related anomaly detected on {node}. This requires investigation to ensure system security and proper access control.",
            "Authentication subsystem error. This might be caused by misconfiguration or token synchronization issues."
        ],
        "network": [
            "Network connectivity issue between {node} and the cluster. This may impact inter-node communication and job completion.",
            "Communication failure in the network layer for {component}. This could affect data transfer and service availability.",
            "Network anomaly detected. This is typically caused by hardware issues or network congestion."
        ],
        "io": [
            "I/O anomaly on {node} related to {component}. This may impact data access and job performance.",
            "Storage subsystem reporting errors. This suggests possible hardware issues or capacity limitations.",
            "File system error detected. This could lead to data access problems or job failures if not addressed."
        ],
        "application": [
            "Application-level anomaly in {component}. This may impact specific workloads or services running on {node}.",
            "Software exception in the application layer. This might affect job completion or service availability.",
            "Application process error detected. This requires attention to restore normal operation."
        ]
    }
    
    template = random.choice(explanation_templates.get(label, explanation_templates["application"]))
    return template.format(node=node, component=component)