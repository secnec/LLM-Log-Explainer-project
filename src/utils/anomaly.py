import dspy
from typing import Literal

class AnomalyLabeler(dspy.Signature):
    """Label the anomaly based on the text."""
    text: str = dspy.InputField()
    label: Literal['application', 'authentication', 'io', 'memory', 'network', 'other', ] = dspy.OutputField()