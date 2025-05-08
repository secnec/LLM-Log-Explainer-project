import polars as pl
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prompt_generator import PromptGenerator

def create_test_dataframe():
    """Helper function to create test DataFrame"""
    return pl.DataFrame({
        'LineId': [1, 2, 3],
        'anomaly_score': [0.1, 0.8, 0.9],
        'context_ids_ref': [None, 1, 1],
        'log_message': ['Normal', 'Error 1', 'Error 2']
    })

def test_generate_label_prompts_adds_column():
    df = create_test_dataframe()
    pg = PromptGenerator()
    result = pg.generateLabelPrompts(threshold=0.7, df=df)
    assert 'anomaly_label' in result.columns, "anomaly_label column not added"

def test_generate_label_prompts_populates_above_threshold():
    df = create_test_dataframe()
    pg = PromptGenerator()
    result = pg.generateLabelPrompts(threshold=0.7, df=df)
    anomalous_count = result.filter(pl.col('anomaly_score') >= 0.7)['anomaly_label'].is_not_null().sum()
    assert anomalous_count == 2, f"Expected 2 prompts, got {anomalous_count}"
    assert result.filter(pl.col('anomaly_score') < 0.7)['anomaly_label'].is_null().all(), "Non-anomalous rows should have null"

def test_generate_explanation_prompts_adds_column():
    df = create_test_dataframe()
    pg = PromptGenerator()
    result = pg.generateExplanationPrompts(threshold=0.7, df=df)
    assert 'explanation_prompt' in result.columns, "explanation_prompt column not added"

def test_generate_explanation_prompts_triggers_label_prompts():
    df = create_test_dataframe()
    pg = PromptGenerator()
    result = pg.generateExplanationPrompts(threshold=0.7, df=df)
    assert 'anomaly_label' in result.columns, "anomaly_label not generated when missing"
    anomalous_label_count = result.filter(pl.col('anomaly_score') >= 0.7)['anomaly_label'].is_not_null().sum()
    assert anomalous_label_count == 2, f"Expected 2 label prompts, got {anomalous_label_count}"

def test_generate_explanation_prompts_populates_above_threshold():
    df = create_test_dataframe()
    pg = PromptGenerator()
    result = pg.generateExplanationPrompts(threshold=0.7, df=df)
    explanation_count = result.filter(pl.col('anomaly_score') >= 0.7)['explanation_prompt'].is_not_null().sum()
    assert explanation_count == 2, f"Expected 2 explanations, got {explanation_count}"
    assert result.filter(pl.col('anomaly_score') < 0.7)['explanation_prompt'].is_null().all(), "Non-anomalous rows should have null"