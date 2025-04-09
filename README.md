# 🧠  LLM-Log-Explainer-project

This repository implements a modular pipeline for generating natural language explanations for software log anomalies using a Large Language Model (LLM). The process involves selecting relevant context, creating prompts, and retrieving LLM-based explanations.

---

## 📁 Project Structure

```
.
├── contextselection.py        # Context selection module
├── prompt_generator.py        # Prompt generation logic
├── llm_prompter.py            # LLM interaction module
├── main.py                    # Pipeline entry point
```

---

## 🚀 Pipeline Overview

The pipeline consists of three main components:

1. **Context Selection** (`ContextSelection`)  
   Selects meaningful lexical or semantic context from the dataframe. This context is passed to the LLM to help in generating relevant explanations.

2. **Prompt Generation** (`PromptGenerator`)  
   Generates prompts based on the selected context and a given threshold.

3. **LLM Prompter** (`LLMPrompter`)  
   Interacts with an LLM to retrieve explanation responses for each prompt.

---

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/explanation-pipeline.git
   cd explanation-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenRouter key to use the LLM to .env file:
    ```bash
    OPENROUTER_API_KEY=your_api_key_here
    ```

---

## 🧪 Example Usage

To run the pipeline:

```bash
python main.py
```

---

## 🛠️ To Do

- [ ] Add Context Selection module for sequence based anomalies
- [ ] Improve the generated explanations using Prompt Engineering

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
