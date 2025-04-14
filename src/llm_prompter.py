import os
import requests
import json
import pandas as pd
import polars as pl
from dotenv import load_dotenv

class LLMPrompter:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise Exception("Failed to load OPENROUTER_API_KEY from environment")
        
        self.model = "openai/gpt-4o-mini"   
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://localhost"
        }
        
        print(f"Initialized LLMPrompter with OpenRouter API (model: {self.model})")
    
    def set_model(self, model_name):
        """Change the model being used"""
        self.model = model_name
        print(f"Changed model to: {self.model}")

    def call_llm(self, prompt, max_tokens=800):
        """Make a direct API call to OpenRouter"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                print(f"Response: {response.text[:100]}")
                return f"API Error: {response.status_code}"
                    
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"
        
    def getExplanationResponses(self, df):
        """
        Gets explanations from the LLM for each prompt in the 'explanation_prompt' column,
        and writes them into a new 'explanation_result' column.

        Parameters:
        - df (pl.DataFrame): Polars DataFrame containing log data with an 'explanation_prompt' column.

        Returns:
        - df (pl.DataFrame): Updated DataFrame with a new 'explanation_result' column.
        """
        # Filter rows with non-null explanation prompts
        rows_with_prompts = df.filter(pl.col('explanation_prompt').is_not_null())
        
        # List to store generated explanations
        explanations = []
        line_ids = []
        
        print(f"Generating explanations for {rows_with_prompts.shape[0]} prompts...")
        
        for row in rows_with_prompts.iter_rows(named=True):
            line_id = row['LineId']
            prompt = row['explanation_prompt']
            
            try:
                # Call the LLM with the prompt
                explanation = self.call_llm(prompt, max_tokens=300)
                
            except Exception as e:
                explanation = f"Error: {str(e)}"
                
            explanations.append(explanation)
            line_ids.append(line_id)
        
        # Create a temporary DataFrame with LineId and explanations
        if explanations:
            explanation_df = pl.DataFrame({
                'LineId': line_ids,
                'explanation_result': explanations
            })
            
            # Join the temporary DataFrame with the original DataFrame on 'LineId'
            df = df.join(explanation_df, on='LineId', how='left')
            print(f"Added {len(explanations)} explanations to the DataFrame")
        else:
            # If no explanations were generated, just add an empty column
            df = df.with_columns(pl.lit(None).alias('explanation_result'))
            print("No explanations were generated")
        
        return df
    
    def getLabelResponses(self, df):
        """
        Gets labels from the LLM for each prompt in the 'anomaly_label' column,
        and writes them into a new 'anomaly_result' column.
        
        Parameters:
        - df (pl.DataFrame): Polars DataFrame containing an 'anomaly_label' column with prompts.
        
        Returns:
        - df (pl.DataFrame): Updated DataFrame with a new 'anomaly_result' column.
        """
        # Initialize the 'anomaly_result' column with null values
        df = df.with_columns(pl.lit(None).alias('anomaly_result'))
        
        # Valid label categories
        valid_labels = ['application', 'authentication', 'io', 'memory', 'network', 'other']
        
        # Filter rows that have a prompt in the anomaly_label column
        rows_with_prompts = df.filter(pl.col('anomaly_label').is_not_null())
        
        print(f"Generating labels for {rows_with_prompts.shape[0]} prompts...")
        
        # Process each row with a prompt
        for row in rows_with_prompts.iter_rows(named=True):
            idx = row['LineId']
            prompt = row['anomaly_label']
            
            try:
                # Add instructions to the prompt to ensure we get a valid label
                label_prompt = f"""
                {prompt}
                
                Based on the log message and context above, classify this anomaly into exactly ONE of these categories:
                - application
                - authentication
                - io
                - memory
                - network
                - other
                
                Reply with ONLY the category name, nothing else.
                """
                
                # Call the LLM with the prompt
                response = self.call_llm(label_prompt, max_tokens=50)
                
                # Extract the label from the response - look for any of the valid labels
                response_text = response.strip().lower()
                label = next((l for l in valid_labels if l in response_text), 'other')
                
                # Update the DataFrame with the label
                df = df.with_columns(
                    pl.when(pl.col('LineId') == idx)
                    .then(pl.lit(label))
                    .otherwise(pl.col('anomaly_result'))
                    .alias('anomaly_result')
                )
                
            except Exception as e:
                print(f"Error processing LineId {idx}: {str(e)}")
                # Default to 'other' in case of errors
                df = df.with_columns(
                    pl.when(pl.col('LineId') == idx)
                    .then(pl.lit('other'))
                    .otherwise(pl.col('anomaly_result'))
                    .alias('anomaly_result')
                )
       
        return df

    def get_file_explanation_response(self, file_explanation_prompt: str) -> str | None:
        """
        Gets a single explanation response from the LLM for a file-level prompt.
        """
        try:
            print("Sending combined file prompt to LLM...")
            # Adjust max_tokens if needed for combined identify/explain
            response = self.lm(file_explanation_prompt, max_tokens=800) # Potentially longer response
            # Handle response list/string
            if isinstance(response, list) and response: return response[0].strip()
            if isinstance(response, str): return response.strip()
            return f"LLM returned no valid response. Raw: {response}"
        except Exception as e:
            print(f"Error getting file explanation from LLM: {e}")
            return f"Error during LLM communication: {str(e)}"
