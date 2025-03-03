from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

from utils.helpers import load_yaml_config

config = load_yaml_config('config.yaml')
EMBEDDER_MODEL = config['embedder_model']


class ContextSelector:
    def __init__(self, selection_strategy: str,
                 min_score: int = None,
                 top_k: int = None,
                 window: tuple[int, int] = None,
                 **kwargs):
        """
        :param selection_strategy: lexical, semantic, score or window
        :param min_score: Specifying a minimum similarity or anomaly score for the lines to be returned.
        :param top_k: Specifying how many similar/anomalous lines to return.
        :param window: Specifying how many lines before and after the most anomalous line should be returned.
        """
        self.selection_strategies = {
            "lexical": self._lexical_matrix,
            "semantic": self._semantic_matrix,
            "score": self._score_based,
            "window": self._window_based,
        }

        if selection_strategy not in self.selection_strategies:
            raise Exception(f"Invalid selection strategy: {selection_strategy}")
        self.selection_strategy = selection_strategy

        self.min_score = min_score
        self.top_k = top_k
        self.window = window

        if selection_strategy == "window" and window is None:
            raise Exception("The window must be specified for the window selection strategy.")

        if selection_strategy != "window" and self.min_score is None and self.top_k is None:
            raise Exception("At least one of min_score or top_k must be specified")

        self.embedder_model = kwargs.get('embedder_model', EMBEDDER_MODEL)

    def transform(self, context_df: pd.DataFrame,
                  line_number: int = None,
                  text_column: str = None,
                  score_column: str = None):
        """
        Returns the context based on the specifications.
        :param context_df: The dataframe containing the log.
        :param line_number Number of the anomalous line.
        :param text_column: The column containing the logs.
        :param score_column: The column containing the anomaly scores.
        """
        kwargs = {
            "line_number": line_number,
            "text_column": text_column,
            "score_column": score_column,
        }

        if self.selection_strategy == "window":
            return self._window_based(context_df, **kwargs)

        scores = self.selection_strategies[self.selection_strategy](context_df, **kwargs)

        filtering = (scores >= self.min_score) if self.min_score is not None else np.ones_like(scores)
        filter_ids = np.argwhere(filtering)
        sorting_ids = np.flip(np.argsort(scores))

        matches = [line_id for line_id in sorting_ids if line_id in filter_ids]

        top_k = self.top_k if self.top_k is not None else len(matches)
        sorted_indexes = list(sorted(matches[:top_k]))  # according to line number in log
        return context_df.iloc[sorted_indexes][text_column]

    def _lexical_matrix(self, context_df: pd.DataFrame, **kwargs) -> Iterable[float]:
        """
        Selects the context for the log line based on the lexical similarity between the log line and the context.
        """
        # Create a CountVectorizer to convert text to numerical features
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(context_df[kwargs["text_column"]])

        log_line = context_df[kwargs["text_column"]].iloc[kwargs["line_number"]]
        log_line_vector = vectorizer.transform([log_line])
        cosine_similarities = cosine_similarity(log_line_vector, X)[0]
        return cosine_similarities

    def _semantic_matrix(self, context_df: pd.DataFrame, **kwargs) -> Iterable[float]:
        """
        Selects the context for the log line based on the semantic similarity between the log line and the context.
        """
        model = SentenceTransformer(self.embedder_model)
        context_df_embeddings = model.encode(context_df[kwargs["text_column"]])

        log_line = context_df[kwargs["text_column"]].iloc[kwargs["line_number"]]
        log_line_embedding = model.encode([log_line])
        cosine_similarities = cosine_similarity(log_line_embedding, context_df_embeddings)[0]
        return cosine_similarities

    def _score_based(self, context_df: pd.DataFrame, **kwargs) -> Iterable[float]:
        """
        Selects the context for the log line based on the anomaly scores.
        """
        return context_df[kwargs["score_column"]]

    def _window_based(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Finds the line with the highest anomaly score and returns the specified number of lines before and after it.
        """
        scores = self._score_based(context_df, **kwargs)
        most_anomalous_line = np.argmax(scores)
        split = (most_anomalous_line - self.window[0], most_anomalous_line + self.window[1] + 1)

        if split[0] < 0:
            split = (0, split[1])

        if split[1] > len(context_df):
            split = (split[0], len(context_df))

        return context_df.iloc[split[0]:split[1]][kwargs["text_column"]]
