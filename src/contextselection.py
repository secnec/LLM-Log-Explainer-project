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
        return self.model.encode(text) 

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
        print("inside near")
        idx = self.context_df.index[self.context_df['row_nr'] == line_number].tolist()[0]
       
        start_idx = max(0, idx - self.top_k_near // 2)
        end_idx = min(len(self.context_df) - 1, idx + self.top_k_near // 2)
        print(idx)
        print(start_idx)
        print(end_idx)
        print(self.context_df.iloc[start_idx:idx + 1]['context_ids'].shape)  # idx+1 to include idx
        print(self.context_df.iloc[idx:end_idx + 1]['context_ids'].shape)  # end_idx+1 to include end_idx
        self.context_df.iloc[start_idx:idx + 1, self.context_df.columns.get_loc('context_ids')] = self.context_df.iloc[start_idx:idx + 1]['context_ids'].fillna('').astype(str) + f',{line_number}'
        self.context_df.iloc[idx:end_idx + 1, self.context_df.columns.get_loc('context_ids')] = self.context_df.iloc[idx:end_idx + 1]['context_ids'].fillna('').astype(str) + f',{line_number}'
        
    def get_far_context(self, line_number, log_line):
        print("inside far")
        reduced_df = self.context_df[self.context_df['row_nr'] < line_number]
        reduced_df = reduced_df[reduced_df[self.column_name].notnull()]
        similarity_matrix = self.semantic_matrix(reduced_df, log_line)
        top_k_far = similarity_matrix.nlargest(self.top_k_far, 'similarity')
        print(top_k_far)
        self.context_df.loc[top_k_far.index, 'context_ids'] = self.context_df.loc[top_k_far.index, 'context_ids'].astype(str) + f',{line_number}'
    
    def semantic_matrix(self, reduced_df, log_line) -> pd.DataFrame:
        context_df_embeddings = self.embedder.embed(reduced_df[self.column_name].values)
        log_line_embedding = self.embedder.embed(log_line)
        cosine_similarities = cosine_similarity(log_line_embedding, context_df_embeddings)
        similarity_df = reduced_df.copy()
        similarity_df['similarity'] = cosine_similarities[0]
        return similarity_df


if __name__ == "__main__":
    context_selection = ContextSelection("semantic", "filtered_data_one_month.parquet", "e_message_normalized", 5, 5)
    context_selection.process_anomalies()
