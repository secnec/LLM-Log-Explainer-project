from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd


from utils.helpers import load_yaml_config  


config = load_yaml_config('config.yaml')
EMBEDDER_MODEL = config['embedder_model']

class ContextSelector:
    def __init__(self, selection_strategy: str, context_df: str, log_line: str, column_name: str, top_k: int, **kwargs):
        """
        Initialize the context selector with a selection strategy.
        """
        self.selection_strategy = selection_strategy # Can be "scores", "similarity", "hybrid"
        self.context_df = context_df
        self.log_line = log_line
        self.column_name = column_name
        self.top_k = top_k
        self.embedder_model = kwargs.get('embedder_model', EMBEDDER_MODEL)

    def load_context_df(self) -> pd.DataFrame:
        """
        Load the context dataframe from the file path.
        """
        return pd.read_csv(self.context_df)

    def lexical_matrix(self) -> str:
        """
        Select the context for the log line based on the lexical similarity between the log line and the context.
        """
        # Create a CountVectorizer to convert text to numerical features
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.context_df[self.column_name])
        log_line_vector = vectorizer.transform([self.log_line])
        cosine_similarities = cosine_similarity(log_line_vector, X)
        return self.context_df.iloc[cosine_similarities.argmax()]

    def semantic_matrix(self) -> str:
        """
        Select the context for the log line based on the semantic similarity between the log line and the context.
        """
        model = SentenceTransformer(self.embedder_model)
        context_df_embeddings = model.encode(self.context_df[self.column_name])
        log_line_embedding = model.encode([self.log_line])
        cosine_similarities = cosine_similarity(log_line_embedding, context_df_embeddings)
        return self.context_df.iloc[cosine_similarities.argmax()]

    def hybrid_matrix(self) -> str:
        """
        Select the context for the log line based on the hybrid selection strategy.
        """
        pass

    def select_context(self) -> str:
        """
        Select the context for the log line based on the selection strategy.
        """
        pass

