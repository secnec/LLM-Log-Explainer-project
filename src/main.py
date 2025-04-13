from contextselection import ContextSelection
from prompt_generator import PromptGenerator
from llm_prompter import LLMPrompter
from filecontextselection import FileContextSelection

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

    # For file anomaly explanation
    # file_context_selector = FileContextSelection(file_path=file_path, filetype=file_type, anomaly_detection_method=ad_method)
    # selected_context_df = file_context_selector.get_context()
    # combined_prompt = pg.generate_file_explanation_prompt(filename=file_path, df_context=selected_context_df)
    # file_explanation_result = prompter.get_file_explanation_response(combined_prompt)


if __name__ == "__main__":
    main()