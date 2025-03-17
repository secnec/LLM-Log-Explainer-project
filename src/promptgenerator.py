class PromptGenerator:
    def __init__(self):
        pass

    def generateLabelPrompts(self, treshold, df):
        #Generates prompts for getting labels from LLM, and adds this as a new column into the DF. Only generates them for lines whose anomaly score is larger or equal to treshold.
        pass

    def generateExplanationPrompts(self, treshold, df):
        #Generates prompts for getting explanations from LLM, and adds this as a new column into the DF. Calls generateLabelPrompts if needed. Only generates them for lines whose anomaly score is larger or equal to treshold.
        pass

    def getContexLines(self,df):
        # Gets the context lines for an anomaly from the DF, and modifies them into a useful format (Semicolon-separated text, for example)
        pass
