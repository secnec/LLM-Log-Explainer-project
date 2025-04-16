# 🧠  LLM-Log-Explainer-project

This repository implements a modular pipeline for generating natural language explanations for software log anomalies using a Large Language Model (LLM). The process involves selecting relevant context, creating prompts, predicting labels, and generating LLM-based explanations.

---

## 📁 Project Structure

```
.
├──src
   ├── contextselection.py        # Line-level Anomaly Context Selection 
   ├── filecontextselection.py    # File-level Anomaly Context Selection
   ├── prompt_generator.py        # Prompt generation Module
   ├── llm_prompter.py            # Labels and Explanations generation Module
   ├── pipeline.py                # Main pipeline implementation
   ├── helpers.py                 # Utility functions
   ├── utils
      ├── prompts.py              # Prompt templates and utility functions
      ├── ....                    # Other utility functions
├──notebooks 
   ├── ....                       # Jupyter notebooks for exploration and testing 
├──data                           # Sample Data Files
```

---

## 🚀 Pipeline Overview

The pipeline consists of the following steps:

1. **Data Preparation**  (`prepare_data`)
   The input data is loaded from a Parquet file and prepared for processing. This includes renaming columns, handling missing values, and ensuring the correct format for anomaly scores and labels. If a CSV file with known anomalies is provided, it is used to update the anomaly scores in the dataset.

2. **Context Selection** (`ContextSelection` and `FileContextSelection`)  
   This step selects meaningful lexical or semantic context for each anomaly. The context is used to provide additional information to the LLM for generating explanations. 
   If the anomaly is at line-level, The `ContextSelection` module selects the most similar lines from the log file based on the semantic similarity of the log messages. The goal here is to extend the context.
   If the anomaly is at file-level, the `FileContextSelection` module selects the chunks of the log that most probably contains the anomaly. The goal here is to shrink the context.

3. **Prompt Generation** (`PromptGenerator`)  
   Prompts are generated based on the selected context and a given anomaly score threshold. These prompts are used to query the LLM for anomaly labels and explanations.
   The prompts are built using the components in the `src/utils/prompts.py` file.

4. **LLM Integration** (`LLMPrompter`)  
   - In **test mode**, the pipeline simulates LLM responses by generating labels and explanations using predefined logic.
   - In **production mode**, the pipeline interacts with an LLM (via OpenRouter API) to predict anomaly labels and generate explanations for the prompts constructed previously.

5. **Result Cleaning**  
   The generated prompts and responses are cleaned to ensure consistency and readability. This step is optional and will  not be performed unless the flgs `--clean-results` is used.

6. **Output Formatting**  
   The final results, including anomaly labels and explanations, are saved to a CSV file. The output is formatted for easy presentation and analysis.

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

3. Add your OpenRouter key to use the LLM to `.env` file:
    ```bash
    OPENROUTER_API_KEY=YOUR_API_KEY
    ```

---

## 🧪 Example Usage

To run the pipeline with the demo data from the BGL dataset, please use the following command:

```bash
python -m src.pipeline --input data/bgl-demo/bgl-demo-1.parquet --threshold 0.91 --verbose --clean-results 
```

## 🛠️ To Do

- [ ] Add Explanations Evaluation module
- [ ] Add Labels Evaluation module

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
