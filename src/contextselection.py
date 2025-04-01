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
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, text: Union[str, List[str], pl.Series]) -> np.ndarray:
        if isinstance(text, pl.Series):
            text = text.to_list()
        elif isinstance(text, str):
            text = [text]
        return self.model.encode(text, show_progress_bar = True) 

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
        self.embedder = Embedder(self.embedder_model)
    
    @cached_property
    def context_df(self) -> pd.DataFrame:
        df = pd.read_parquet(self.df_path)
        if self.limit_date:
            df = df[df['date'] == df['date'].max()]
        df["context_ids"] = None
        print(f" The dataframe has {df.shape[0]} rows and {df.shape[1]} columns")
        return df.sort_values(by='row_nr') if 'row_nr' in df.columns else df
    
    def process_anomalies(self):
        anomalies = self.context_df[self.context_df["anomaly"] == True].copy()
        for _, anomaly in anomalies.iterrows():
            line_number = anomaly['row_nr']
            log_line = anomaly[self.column_name]
            self.process_single_anomaly(line_number, log_line)
            self.context_df.to_csv("first_try.csv")
            break
        return self.context_df
    
    def process_single_anomaly(self, line_number, log_line):
        self.context_df.loc[self.context_df['row_nr'] == line_number, 'context_ids'] = str(line_number)
        self.get_near_context(line_number)
        if self.selection_strategy == "semantic":
            self.get_far_context(line_number, log_line)
    
    def get_near_context(self, line_number):
        idx = self.context_df.index[self.context_df['row_nr'] == line_number].tolist()[0]
        start_idx = max(0, idx - self.top_k_near // 2)
        end_idx = min(len(self.context_df) - 1, idx + self.top_k_near // 2)
        self.context_df.iloc[start_idx:idx + 1, self.context_df.columns.get_loc('context_ids')] = self.context_df.iloc[start_idx:idx + 1]['context_ids'].fillna('').astype(str) + f',{line_number}'
        self.context_df.iloc[idx:end_idx + 1, self.context_df.columns.get_loc('context_ids')] = self.context_df.iloc[idx:end_idx + 1]['context_ids'].fillna('').astype(str) + f',{line_number}'
        
    def get_far_context(self, line_number, log_line):
        reduced_df = self.context_df[self.context_df['row_nr'] < line_number]
        reduced_df = reduced_df[reduced_df[self.column_name].notnull()]
        similarity_matrix = self.semantic_matrix(reduced_df, log_line)
        top_k_far = similarity_matrix.nlargest(self.top_k_far, 'similarity')
        self.context_df.loc[top_k_far.index, 'context_ids'] = self.context_df.loc[top_k_far.index, 'context_ids'].astype(str) + f',{line_number}'
    
    def semantic_matrix(self, reduced_df, log_line) -> pd.DataFrame:
        context_df_embeddings = self.embedder.embed(reduced_df[self.column_name].values)
        log_line_embedding = self.embedder.embed(log_line)
        cosine_similarities = cosine_similarity(log_line_embedding, context_df_embeddings)
        similarity_df = reduced_df.copy()
        similarity_df['similarity'] = cosine_similarities[0]
        return similarity_df

    def getLexicalContext(self) -> pd.DataFrame:
        df = self.context_df.drop_duplicates(subset=[self.column_name]) if self.drop_duplicates else self.context_df
        # Will be deleted after testing
        df = df.sample(1000) if df.shape[0] > 1000 else df
        context_df_embeddings = self.embedder.embed(df[self.column_name].values)
        cosine_similarities = cosine_similarity(context_df_embeddings)
        print(cosine_similarities)
        
        # Get the indices of the top similar rows (excluding self)
        top_similar_indices = np.argsort(-cosine_similarities, axis=1)[:, 1:]  # Exclude self (diagonal)
        top_k_far_indices = top_similar_indices[:, :self.top_k_far]
        # Map indices to "row_nr" values
        row_nr_values = df["row_nr"].values  # Convert the "row_nr" column to a NumPy array
        df["context_ids"] = [list(row_nr_values[indices]) for indices in top_k_far_indices]

        return df


if __name__ == "__main__":
    context_selection = ContextSelection("semantic", "filtered_data_one_month.parquet", "e_message_normalized", 5, 5)
    print(context_selection.getLexicalContext())
