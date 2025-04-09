from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List
import polars as pl

# from utils.helpers import singleton

# @singleton
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
    

if __name__ == "__main__":
    embedder = Embedder("all-MiniLM-L6-v2")
    print(embedder.embed("Hello, world!"))

