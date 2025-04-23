# 🧠 LLM-Log-Explainer-Project

This repository implements a modular pipeline for generating natural language explanations for software log anomalies using a Large Language Model (LLM). The process involves selecting relevant context, creating prompts, predicting labels, and generating LLM-based explanations.

---

## 📁 Project Structure

```
.
├── src
   ├── contextselection.py        # Line-level Anomaly Context Selection 
   ├── filecontextselection.py    # File-level Anomaly Context Selection
   ├── prompt_generator.py        # Prompt Generation Module
   ├── llm_prompter.py            # Labels and Explanations Generation Module
   ├── pipeline.py                # Main Pipeline Implementation
   ├── helpers.py                 # Utility Functions
   ├── utils
      ├── prompts.py              # Prompt Templates and Utility Functions
      ├── ....                    # Other Utility Functions
├── notebooks 
   ├── ....                       # Jupyter Notebooks for Exploration and Testing 
├── data                           # Sample Data Files
```

---

## 🚀 Pipeline Overview

The pipeline consists of the following steps:

1. **Data Preparation** (`prepare_data`)  
   The input data is loaded from a Parquet file and prepared for processing. This includes renaming columns, handling missing values, and ensuring the correct format for anomaly scores and labels. If a CSV file with known anomalies is provided, it is used to update the anomaly scores in the dataset.

2. **Context Selection** (`ContextSelection` and `FileContextSelection`)  
   This step selects meaningful lexical or semantic context for each anomaly. The context is used to provide additional information to the LLM for generating explanations.  
   - If the anomaly is at the line level, the `ContextSelection` module selects the most similar lines from the log file based on the semantic similarity of the log messages. The goal here is to extend the context.  
   - If the anomaly is at the file level, the `FileContextSelection` module selects the chunks of the log that most likely contain the anomaly. The goal here is to shrink the context.

3. **Prompt Generation** (`PromptGenerator`)  
   Prompts are generated based on the selected context and a given anomaly score threshold. These prompts are used to query the LLM for anomaly labels and explanations.  
   The prompts are built using the components in the `src/utils/prompts.py` file.

4. **LLM Integration** (`LLMPrompter`)  
   - In **test mode**, the pipeline simulates LLM responses by generating labels and explanations using predefined logic.  
   - In **production mode**, the pipeline interacts with an LLM (via the OpenRouter API) to predict anomaly labels and generate explanations for the prompts constructed previously.

5. **Result Cleaning**  
   The generated prompts and responses are cleaned to ensure consistency and readability. This step is optional and will not be performed unless the flag `--clean-results` is used.

6. **Output Formatting**  
   The final results, including anomaly labels and explanations, are saved to a CSV file. The output is formatted for easy presentation and analysis.

---

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/secnec/LLM-Log-Explainer-project.git
   cd LLM-Log-Explainer-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenRouter key to the `.env` file:
   ```bash
   OPENROUTER_API_KEY=YOUR_API_KEY
   ```

---

## 🛠️ Pipeline Parameters

### Context Selection Parameters:
   - `--context-selection-strategy`: Defines how the context is selected. Can be one of [`semantic`, `lexical`, `hybrid`]. Default: `semantic`.  
     - If `semantic`, it uses the `embedder-model` model to compute the sentence embeddings of the log lines, then using the cosine similarity, it finds the top similar lines to the log anomaly line.  
     - If `lexical`, it uses the Count Vectorizer to find the most similar line.  
     - If `hybrid`, it will use both semantic and lexical context.
   - `--embedder-model`: The name of the embedder model to be used for semantic context selection. Default: `sentence-transformers/all-MiniLM-L6-v2`.
   - `--top-k-near`: The number of lines in total before and after the log anomaly to be used as a near context. Default: `6` (so 3 lines before and 3 lines after the log anomaly line will be used as a context).
   - `--top-k-far`: The number of the most similar lines to the log anomaly to be used as far context for both semantic and lexical context. Default:`5`.

### Input Data Parameters:
   - `--input`: The path to the input Parquet file containing the log data. Default: `src/data/bgl-demo-1.parquet`.
   - `--score-column`: The name of the column containing the anomaly scores. Default: `pred_ano_proba`.
   - `--id-column`: The name of the column containing the unique identifiers for each log entry. Default: `row_nr`.
   - `--message-column`: The name of the column containing the log messages. Default: `m_message`.
   - `--context-column`: The name of the column to be used for context selection, can be either the raw log message or the normalized message. Default: `e_message_normalized`.
   - `--log-type`: The type of log data being processed. Can be one of [`BGL`, `LO2`, `mixed`]. Default: `BGL`.
   - `--anomaly-level`: The level of anomaly detection. Can be either `line` or `file`. This determines how the context is selected. Default: `line`.
   - `--ad-method`: The anomaly detection method used. Can be either `LOF` or `IF`. Default: `LOF`.

### Other Parameters:
   - `--threshold`: The threshold for anomaly detection. If the anomaly score is above this threshold, the LLM will be called to generate the labels and explanations. Default: `0.79`.
   - `--verbose`: If set, the pipeline will provide detailed output during execution. Default: `False` (not set).
   - `--test-mode`: If set, the pipeline will run in test mode, simulating LLM responses. Real calls will not be performed. Default: `False` (not set).
   - `--clean-results`: If set, the pipeline will clean the generated LLM explanations and labels to ensure unified results. Default: `False` (not set).
   - `--output`: The path to the output CSV file where the results will be saved. Default: `anomaly_results.csv`.

## 🧪 Example Usage

To run the pipeline with the demo data from the BGL dataset, use the following command:

```bash
python -m src.pipeline --input data/bgl-demo/bgl-demo-1.parquet --threshold 0.91 --verbose --clean-results
```

---

## ✍️ Evaluation

### Evaluation of the LLM-generated Explanations
   The LLM-generated explanations were evaluated using Human Evaluation, and an LLM-as-a-judge evaluation method. Each anomaly was evaluated based on the following criteria:
   -  Correct Identification of the error: whether the LLM correctly identified the error and could explain it in a human-readable way.
   -  Relevance: The ability of the LLM to explain how this anomaly deviates from the normal behavior of the system.
   -  Completeness: The explanation should be complete and provide enough information about the root causes and how this impacts the system.
   -  Language and Plausability: how well the explanation is written and if it makes sense.
   
   The LLM used in the evaluation was gpt-4o mini from the ChatGPT UI. The judge LLM was also asked to provide a score out of 10 to evaluate how well the explanations were.
   With the highest score given by the LLM-judge being 10/10 and the lowest being 2/10, the average score was 5.8/10.
   The evaluation results are accessible via this [link](https://docs.google.com/spreadsheets/d/1DXa9TYVUxpWdIfeqK1H4wSHIXnWOW2DWcshtsEBu_84/edit?usp=sharing)

### Evaluation of the LLM-generated Labels

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
