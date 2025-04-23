from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import polars as pl
import pandas as pd
import numpy as np
from functools import cached_property
from typing import Union, List

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.verbose = verbose

    def embed(self, text: Union[str, List[str], pl.Series]) -> np.ndarray:
        if isinstance(text, pl.Series):
            text = text.to_list()
        elif isinstance(text, str):
            text = [text]
        return self.model.encode(text, show_progress_bar=self.verbose) 

class ContextSelection:
    def __init__(self, selection_strategy: str, df_path: str, column_name: str, top_k_far: int, top_k_near: int, **kwargs):
        self.selection_strategy = selection_strategy
        self.df_path = df_path
        self.column_name = column_name
        self.top_k_far = top_k_far
        self.top_k_near = top_k_near
        self.drop_duplicates = kwargs.get('drop_duplicates', False)
        self.limit_date = kwargs.get('limit_date', False)
        self.embedder_model = kwargs.get('embedder_model', EMBEDDER_MODEL)
        self.embedder = Embedder(self.embedder_model, verbose=kwargs.get('verbose', False))
        self.in_memory_df = None

    def set_in_memory_df(self, df):
        """Set an in-memory DataFrame for test mode"""
        self.in_memory_df = df
    
    @cached_property
    def context_df(self) -> pd.DataFrame:
        if self.in_memory_df is not None:
            # Use in-memory DataFrame for testing
            if isinstance(self.in_memory_df, pl.DataFrame):
                df = self.in_memory_df.to_pandas()
            else:
                df = self.in_memory_df
        elif self.df_path:
            # Load from file path
            df = pd.read_parquet(self.df_path)
        else:
            raise ValueError("No DataFrame or file path provided")
            
        if self.limit_date and 'date' in df.columns:
            df = df[df['date'] == df['date'].max()]
        
        if "context_ids" not in df.columns:
            df["context_ids"] = None
            
        print(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns")
        return df.sort_values(by='row_nr') if 'row_nr' in df.columns else df
    
    def process_anomalies(self):
        """Process all anomalies in the DataFrame"""
        if "anomaly" not in self.context_df.columns:
            self.context_df["anomaly"] = self.context_df.get("anomaly_score", 0) >= 0.8
            
        anomalies = self.context_df[self.context_df["anomaly"] == True].copy()
        for _, anomaly in anomalies.iterrows():
            line_number = anomaly.get('row_nr', anomaly.get('LineId'))
            log_line = anomaly[self.column_name]
            self.process_single_anomaly(line_number, log_line)
        
        return self.context_df
    
    def process_single_anomaly(self, line_number, log_line):
        """Process a single anomaly and find its context"""
        id_col = 'row_nr' if 'row_nr' in self.context_df.columns else 'LineId'
        self.context_df.loc[self.context_df[id_col] == line_number, 'context_ids'] = str(line_number)
        self.get_near_context(line_number)
        if self.selection_strategy == "semantic":
            self.get_far_context(line_number, log_line)
    
    def get_near_context(self, line_number):
        """Get temporally adjacent context (nearby log lines)"""
        id_col = 'row_nr' if 'row_nr' in self.context_df.columns else 'LineId'
        idx = self.context_df.index[self.context_df[id_col] == line_number].tolist()
        if not idx:
            return
            
        idx = idx[0]
        start_idx = max(0, idx - self.top_k_near // 2)
        end_idx = min(len(self.context_df) - 1, idx + self.top_k_near // 2)
        
        # Add context to rows before the anomaly
        self.context_df.iloc[start_idx:idx, self.context_df.columns.get_loc('context_ids')] = (
            self.context_df.iloc[start_idx:idx]['context_ids'].fillna('').astype(str) + 
            self.context_df.iloc[start_idx:idx]['context_ids'].apply(
                lambda x: f',{line_number}' if x else f'{line_number}'
            )
        )
        
        # Add context to rows after the anomaly
        self.context_df.iloc[idx+1:end_idx+1, self.context_df.columns.get_loc('context_ids')] = (
            self.context_df.iloc[idx+1:end_idx+1]['context_ids'].fillna('').astype(str) + 
            self.context_df.iloc[idx+1:end_idx+1]['context_ids'].apply(
                lambda x: f',{line_number}' if x else f'{line_number}'
            )
        )
        
    def get_far_context(self, line_number, log_line):
        """Get semantically similar context (logs with similar content)"""
        id_col = 'row_nr' if 'row_nr' in self.context_df.columns else 'LineId'
        reduced_df = self.context_df[self.context_df[id_col] < line_number]
        reduced_df = reduced_df[reduced_df[self.column_name].notnull()]
        
        if reduced_df.empty:
            return
            
        similarity_matrix = self.semantic_matrix(reduced_df, log_line)
        top_k_far = similarity_matrix.nlargest(min(self.top_k_far, len(similarity_matrix)), 'similarity')
        
        for idx in top_k_far.index:
            curr = self.context_df.loc[idx, 'context_ids']
            self.context_df.loc[idx, 'context_ids'] = f"{curr},{line_number}" if curr else f"{line_number}"
    
    def semantic_matrix(self, reduced_df, log_line) -> pd.DataFrame:
        """Calculate semantic similarity between log_line and entries in reduced_df"""
        if len(reduced_df) == 0:
            return pd.DataFrame(columns=['similarity'])
            
        try:
            # Use sentence transformers for better semantic similarity
            context_df_embeddings = self.embedder.embed(reduced_df[self.column_name].values)
            log_line_embedding = self.embedder.embed(log_line)
            cosine_similarities = cosine_similarity(log_line_embedding.reshape(1, -1), context_df_embeddings)[0]
        except Exception as e:
            print(f"Error calculating embeddings: {e}")
            # Fall back to simpler TF-IDF with CountVectorizer
            vectorizer = CountVectorizer().fit([log_line] + reduced_df[self.column_name].tolist())
            log_vector = vectorizer.transform([log_line])
            context_vectors = vectorizer.transform(reduced_df[self.column_name].values)
            cosine_similarities = cosine_similarity(log_vector, context_vectors)[0]
        
        similarity_df = reduced_df.copy()
        similarity_df['similarity'] = cosine_similarities
        return similarity_df

    def getLexicalContext(self, provided_df=None):
        """
        Calculate lexical context for log entries.
        
        Args:
            provided_df: Optional DataFrame to use instead of loading from file
            
        Returns:
            DataFrame with lexical_context column added
        """
        if provided_df is not None:
            self.set_in_memory_df(provided_df)
            
        df = self.context_df
        
        if self.drop_duplicates and self.column_name in df.columns:
            df = df.drop_duplicates(subset=[self.column_name])
            
        # Process all rows with anomaly_score >= 0.8 (or other threshold)
        if "anomaly" not in df.columns and "anomaly_score" in df.columns:
            df["anomaly"] = df["anomaly_score"] >= 0.8
            
        # Process all anomalies to create context connections
        result_df = self.process_anomalies()
        
        # Convert string context_ids to references to LineId in lexical_context column
        result_df['lexical_context'] = result_df['context_ids']
        
        # Return as a polars DataFrame if input was polars
        if provided_df is not None and isinstance(provided_df, pl.DataFrame):
            return pl.from_pandas(result_df)
        
        return result_df