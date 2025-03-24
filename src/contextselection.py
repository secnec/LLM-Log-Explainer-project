from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import polars as pl
import pandas as pd
from functools import cached_property
from typing import Union, List

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code = True)

    def embed(self, text: Union[str, List[str], pl.Series]) -> np.ndarray:
        if isinstance(text, pl.Series):
            text = text.to_list()
        elif isinstance(text, str):
            text = [text]
        return self.model.encode(text) 
    
class ContextSelection:
    def __init__(self, selection_strategy: str, df_path: str, column_name: str, top_k_far: int, top_k_near: int, **kwargs):
        self.selection_strategy = selection_strategy # Can be "scores", "semantic", "lexical", "hybrid"
        self.df_path = df_path
        self.column_name = column_name
        self.top_k_far = top_k_far
        self.top_k_near = top_k_near
        self.line_number = kwargs.get('line_number', None)
        self.drop_duplicates = kwargs.get('drop_duplicates', False)
        self.limit_date = kwargs.get('limit_date', False)
        if self.line_number is not None:
            self.line_number = int(self.line_number)
        self.log_line = kwargs.get('log_line', None)
        self.embedder_model = kwargs.get('embedder_model', EMBEDDER_MODEL)
        self.embedder = Embedder(self.embedder_model)
        self.line_index = None

    @cached_property
    def context_df(self) -> pd.DataFrame:
        """
        Load the context dataframes from the file path.
        """
        df = pd.read_parquet(self.df_path)
        
        if self.limit_date:
            # get the date from the line_number
            date = df.loc[df['row_nr'] == self.line_number, 'date'].values[0]
            df = df[df['date'] == date]
        df["context_ids"] = None
        if 'row_nr' in df.columns:
            return df.sort_values(by='row_nr')
        else:
            return df
    
    def get_line_index(self):
        if not self.line_number:
            raise ValueError("Line number is not provided")
        else:
            # Reset the index first, then apply the filter
            reset_df = self.context_df.reset_index()
            line_idx = reset_df.loc[
                reset_df['row_nr'] == int(self.line_number), 'index'
            ].values[0]
            return line_idx
        
    def remove_duplicates(self) -> pd.DataFrame:
        self.line_index = self.get_line_index()
        duplicates = self.context_df.duplicated(subset=self.column_name)
        # Ensure the specific index is retained
        df_cleaned = self.context_df.loc[
            (~duplicates) | (self.context_df.index == self.line_index)
        ]
        # reset line indices
        df_cleaned.reset_index(drop=True, inplace=True)
        # recomputing the line index
        self.line_index = self.get_line_index()    
        return df_cleaned
    
    def get_near_context(self) -> pd.DataFrame:
        """
        return the top_k_near before and after the log_line 
        """
        if self.drop_duplicates:
            self.context_df = self.remove_duplicates()   
        
        # Get indices for rows before and after
        start_idx = max(0, self.line_index - self.top_k_near//2)
        end_idx = min(len(self.context_df) - 1, self.line_index + self.top_k_near//2)
        print(start_idx, end_idx)
        print("self.line_index is ", self.line_index)
        # Add the line_index to the context column of rows from start idex to line_index and to end_index
        self.context_df.loc[start_idx:self.line_index, 'context_ids'] = self.line_number
        self.context_df.loc[self.line_index:end_idx, 'context_ids'] = self.line_number
        return self.context_df
    
    def get_far_context(self) -> pd.DataFrame:
        """
        return the top_k_far before the log_line 
        """
        # select only rows before the log_line
        reduced_df = self.context_df[self.context_df['row_nr'] < self.line_number]
        reduced_df = reduced_df[reduced_df[self.column_name].notnull()]
        if not self.log_line:
            self.log_line = self.context_df[self.context_df['row_nr'] == int(self.line_number)]
            self.log_line = self.log_line[self.column_name].values[0]
        similarity_matrix = self.semantic_matrix(reduced_df)
        top_k_far = similarity_matrix.nlargest(self.top_k_far, 'similarity')
        # locate these lines  in context_df then assign context_ids of to the line_number
        self.context_df.loc[top_k_far.index, 'context_ids'] = self.line_number
        return top_k_far
    
    def lexical_matrix(self) -> pd.Series:
        """
        Select the context for the log line based on the lexical similarity between the log line and the context.
        """
        # Create a CountVectorizer to convert text to numerical features
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.context_df[self.column_name])
        log_line_vector = vectorizer.transform([self.log_line])
        cosine_similarities = cosine_similarity(log_line_vector, X)
        return self.context_df.iloc[cosine_similarities.argmax()]

    def semantic_matrix(self, reduced_df) -> pd.DataFrame:
        """
        Select the context for the log line based on the semantic similarity between the log line and the context.
        """
        context_df_embeddings = self.embedder.embed(reduced_df[self.column_name].values)
        
        log_line_embedding = self.embedder.embed(self.log_line)
        cosine_similarities = cosine_similarity(log_line_embedding, context_df_embeddings)
        
        similarity_df = reduced_df.copy()
        similarity_df['similarity'] = cosine_similarities[0]
        
        return similarity_df
    
    def getLexicalContex(self, df):
        #Gets lexical context for each anomaly in the DF, and adds a new column in the DF to mark it. So if the context for anomaly in row 5 are rows 1,3,7,8 then the "lexical_context"-column has "5" for rows 1,3,7,8
        pass

    def getSemanticContext(self, df):
        #Gets lexical context for each anomaly in the DF, and adds a new column in the DF the same way as getLexicalContex
        near_context = self.get_near_context()        
        if self.selection_strategy == "semantic":
            self.get_far_context()
        
        return self.context_df