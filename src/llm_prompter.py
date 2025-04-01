import os
import pandas as pd
import polars as pl

class LLMPrompter:
    def __init__(self):
        self.lm = None

    def getExplanationResponses(self, df):
        #Gets the responses from the LLM for each prompt in the explanation prompts -column, and writes them into a new explanations-column
        pass

    def getLabelResponses(self, df):
        #Gets the labels from the LLM for each prompt in the label prompts -column, and writes them into a new labels-column
        pass

