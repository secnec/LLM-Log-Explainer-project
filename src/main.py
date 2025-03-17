from contextselection import ContextSelection
from promptgenerator import PromptGenerator
from llmprompter import LLMPrompter

def importdata():
    pass

def main():
    df = importdata()

    cs = ContextSelection()
    pg = PromptGenerator()
    prompter = LLMPrompter()

    df = cs.getLexicalContex(df)
    df = pg.generateExplanationPrompts(treshold=0.9, df=df)
    df = prompter.getExplanationResponses(df)

if __name__ == "__main__":
    main()