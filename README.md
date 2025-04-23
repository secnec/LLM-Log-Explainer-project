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

## Pipeline Parameters

### Context Selection Parameters:
   - --context-selection-strategy: Defines how the context is selected. Can be one of ["semantic", "lexical", "hybrid"]. If "semantic", it uses the "embedder-model" model to compute the sentence embeddings of the log lines then using the cosine similarity it finds the top similar lines to the log anomaly line. If "lexical", it uses the Count Vectoriser to find the most similar line. If hybrid, it will use both semantic and lexical context.
   - --embedder-model: The name of the embedder model to be used for semantic context selection. Default is "sentence-transformers/all-MiniLM-L6-v2".
   - --top-k-near: The number of lines before and after the log anomaly to be used as a near context.
   - --top-k-far: The number of the most similar lines to the log anomaly to be used as far context for both semantic and lexical context

### Input Data Parameters:
   - --input: The path to the input Parquet file containing the log data.
   - --score-column: The name of the column containing the anomaly scores.
   - --id-column: The name of the column containing the unique identifiers for each log entry.
   - --message-column: The name of the column containing the log messages.
   - --context-column: The name of the column to be used for context selection, can be either the raw log message or the normalized message.
   - --log-type: The type of log data being processed. Can be one of ["BGL", "LO2", "mixed"].
   - --anomaly-level: The level of anomaly detection. Can be either "line" or "file". This determines how the context is selected.
   - --ad-method: The anomaly detection method used. Can be either "LOF" or "IF".

### Other Parameters:
   - --threshold: The threshold for anomaly detection. If the anomaly score is above this threshold than the LLM will be called to generate the labels and explanations.
   - --verbose: If set, the pipeline will provide detailed output during execution.
   - --test-mode: If set, the pipeline will run in test mode, simulating LLM responses, real calls will not be performed.
   - --clean-results: If set, the pipeline will clean the generated LLM explanations and labels to ensure unified results.
   - --output: The path to the output CSV file where the results will be saved.


## 🧪 Example Usage

To run the pipeline with the demo data from the BGL dataset, use the following command:

```bash
python -m src.pipeline --input data/bgl-demo/bgl-demo-1.parquet --threshold 0.91 --verbose --clean-results
```

---

## 🛠️ To Do

- [ ] Add Explanations Evaluation Module.
- [ ] Add Labels Evaluation Module.

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
