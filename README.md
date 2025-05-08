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
├── evals                          # Evaluation Scripts and Results
   ├── label_evals.py             # Script for Evaluating Label Predictions
   ├── data                       # Evaluation Datasets
   ├── results                    # Evaluation Results
```

---

## 🚀 Pipeline Overview

The pipeline consists of the following steps:

1. **Data Preparation** (`prepare_data`)  
   The expected format of the input data for line-anomaly processing is the output from LogLead. The data should have the columns created by the enhancers, and the anomaly_score -column, with the scores from the anomaly detection. This column should be created with auc_roc=True, so that it includes the scores.
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

The dependencies of this project are compatible with python 3.10.

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

## 🏷️ Log Anomaly Labeling 

### Labeling Framework

The labeling component of this project classifies log anomalies into six distinct categories:

- **Application**: Issues within the application logic, such as null pointer exceptions, unhandled errors, or application crashes
- **Authentication**: Security-related issues like failed login attempts, invalid credentials, or access control problems
- **IO**: Input/output related problems, including file system errors, disk access failures, or permission issues 
- **Memory**: Memory allocation problems, out-of-memory errors, or memory leak indicators
- **Network**: Communication issues such as connection timeouts, unreachable hosts, or network interface failures
- **Other**: Anomalies that don't clearly fit into the above categories

### Label Selection Methodology

Our labeling approach is based on the work from the research paper ["Towards Automated Log-Based Anomaly Detection Through Natural Language-Guided Machine Learning"](https://arxiv.org/pdf/2308.11526) by Hasan et al. (2023), which presents a framework for using natural language processing to enhance log-based anomaly detection. 
### Test Dataset

The evaluation was performed using a labeled dataset of 406 log lines, sourced from the [learning-representations-on-logs-for-aiops](https://github.com/Pranjal-Gupta2/learning-representations-on-logs-for-aiops) repository provided by the authors of the paper. 
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

The LLM-generated labels were evaluated using a test dataset of 406 labeled log lines from the [learning-representations-on-logs-for-aiops](https://github.com/Pranjal-Gupta2/learning-representations-on-logs-for-aiops) repository. The evaluation assessed:

- **Overall Accuracy**: 87.71%
- **Precision**: 90.99%
- **Recall**: 87.71%
- **F1 Score**: 88.33%

Performance varied across different anomaly categories:
- **IO**: Highest performance with 97.67% F1 score, perfect recall and 95.45% precision
- **Other**: Strong performance with 91.42% F1 score (largest category with 303 instances)
- **Application**: Good performance with 83.02% F1 score
- **Memory**: Perfect recall but moderate precision, resulting in 75.00% F1 score
- **Authentication**: Perfect recall with lower precision, 72.73% F1 score
- **Network**: Perfect recall but lower precision at 57.35%, resulting in 72.90% F1 score
- **Device**: Challenges with this rare category (only 2 instances), with 0% detection rate

The evaluation was completed in 468.16 seconds for 407 predictions.
## 📬 Contact

For questions or collaborations, feel free to open an issue or reach out.
