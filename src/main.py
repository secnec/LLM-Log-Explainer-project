from contextselection import ContextSelection
from prompt_generator import PromptGenerator
from llm_prompter import LLMPrompter

def import_data():
    pass

def main():
    df = import_data()

    cs = ContextSelection()
    pg = PromptGenerator()
    prompter = LLMPrompter()

    df = cs.getLexicalContext(df)
    df = pg.generateExplanationPrompts(threshold=0.9, df=df)
    df = prompter.getExplanationResponses(df)

if __name__ == "__main__":
    main()