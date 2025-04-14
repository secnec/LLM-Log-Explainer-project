# utils/prompts.py

DEFAULT_EXPLANATION_PROMPT = """
You're a high-performance computing specialist analyzing supercomputer log anomalies. Explain why this log entry is anomalous.

NORMAL LOG EXAMPLE:
Sequence ID: 1719688444__correct,
Label: correct,
Anomaly Score: 73.48012061095164,

Log Messages:
<init> - normalised = /oauth2/token
getConfigStream - Config loaded from externalized folder for primary.jks in /config
getPrivateKey - filename = primary.jks key = selfsigned
debug - validate( "Basic MTM0Mzg0MmUtYzU3ZC00ZDVhLWIyOTUtZjY5MzM0YTVjNjlkOnlFb1ladmdKU3otZ1R6eF9kYVJhVFE=", "Basic MTM0Mzg0MmUtYzU3ZC00ZDVhLWIyOTUtZjY5MzM0YTVjNjlkOnlFb1ladmdKU3otZ1R6eF9kYVJhVFE=", authorization)
getPrivateKey - filename = primary.jks key = selfsigned
handleAuthorizationCode - code = OstfU9uASoy7PIX6vU7X4Q...
<init> - normalised = /oauth2/token
<init> - normalised = /oauth2/token
<init> - normalised = /oauth2/token

EXPLANATION: This sequence follows the expected OAuth2 authorization flow with successful config loading, key retrieval, validation, and authorization code handling. The low anomaly score confirms typical behavior.

ANOMALOUS LOG EXAMPLE:
Sequence ID: 1719756844__invalid_code_verifier_format_PKCE_400,
Label: invalid_code_verifier_format_PKCE_400,
Anomaly Score: 273.50457034572565,
Log Messages:
<init> - normalised = /oauth2/token
debug - validate( "Basic NjM3MjI5OGUtYzQxNC00YzRlLWE4MDUtZGJiNDE5N2ZhYTA0OkxjM2NkcnU3UzhLR0lmd3hWODFidGc=", "Basic NjM3MjI5OGUtYzQxNC00YzRlLWE4MDUtZGJiNDE5N2ZhYTA0OkxjM2NkcnU3UzhLR0lmd3hWODFidGc=", authorization)
getConfigStream - Config loaded from externalized folder for primary.jks in /config
getConfigStream - Config loaded from externalized folder for primary.jks in /config
<init> - path = /oauth2/token, base path is set to: null
<init> - path = /oauth2/token, base path is set to: null
<init> - path =/oauth2/token
debug - validate( "Basic NjM3MjI5OGUtYzQxNC00YzRlLWE4MDUtZGJiNDE5N2ZhYTA0OkxjM2NkcnU3UzhLR0lmd3hWODFidGc=", "Basic NjM3MjI5OGUtYzQxNC00YzRlLWE4MDUtZGJiNDE5N2ZhYTA0OkxjM2NkcnU3UzhLR0lmd3hWODFidGc=", authorization)

EXPLANATION: This sequence is anomalous due to an invalid code verifier format in the PKCE flow. The repeated getConfigStream entries, multiple null base path initializations, and high anomaly score (273.5) indicate a client-side request failure.

Analyze this anomalous log entry and context:
<anomalous_log>
{anomalous_log_str}
</anomalous_log>
<label>
{label_str}
</label>
<context>
{context_str}
</context>
Provide a clear, technical explanation of the anomaly, focusing on:

1. What specific error or issue is occurring
2. How it deviates from normal operation
3. Likely root cause
4. Potential system impact

Be direct and specific without unnecessary text.
"""


DEFAULT_LABEL_PROMPT = """

You're a high-performance computing expert specializing in supercomputer log analysis. Classify the log entry into exactly one category.
Categories:

- application: Software app issues, service failures, component errors
- authentication: Login problems, credential validation, access control
- io: File system errors, data stream issues, storage problems
- memory: Memory allocation failures, leaks, buffer issues
- network: Connection problems, packet issues, network service errors
- other: General system messages that don't fit other categories

Examples:

Example 1:
Log: "java.io.io exception: failed on local exception: java.io.io exception: couldn't set up io streams; host details : local host is: "minint-fnanli5/127.0.0.1"; destination host is: "msra-sa-41":8030;"
Reasoning: This log shows a Java IO exception specifically mentioning "couldn't set up io streams" between hosts. This is clearly related to input/output operations failing rather than network connectivity itself.
Label: io

Example 2:
Log: "dec 11 21:21:19 labsz sshd[3466]: pam_unix(sshd): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=103.99.0.122 user=ftp"
Reasoning: This log shows an SSH daemon (sshd) reporting an authentication failure. The message contains "pam_unix(sshd): authentication failure" which explicitly indicates a problem with user credentials or login process.
Label: authentication

Example 3:
Log: "081111 030115 18071 info dfs. data node$ block receiver: receiving empty packet for block blk_7717782362699139185"
Reasoning: This log indicates a network-related issue where a data node is receiving an empty packet for a specific block. This suggests a problem with data transmission over the network rather than an IO or application error.
Label: network

Example 4:
Log: "373800 node-147 action error 1085979770 1 halt (command 2992) error: couldn\047t connect to console (state = refused)"
Reasoning: This log shows an application action error where a command to halt node-147 failed because it couldn't connect to the console. This is related to application functionality rather than pure network connectivity.
Label: application

Example 5:
Log: "- 1131577111 2005.11.09 cn3 nov 9 14:58:31 cn3/cn3 kernel: ext3 fs on sda1, internal journal"
Reasoning: This log shows a system kernel message about the ext3 filesystem. It's an informational message rather than an error and doesn't fit clearly into the other categories.
Label: other

Here's the log and context to analyze:
<log entry>
Log Line: {log_str}
</log entry>
<context>

Context Lines:
{context_str}
</context>

Think through the following steps:

1. Identify what component or service is mentioned in the log
2. Determine what operation was being attempted
3. Analyze what specific error or issue occurred
4. Match the issue to the most appropriate category

Respond with ONLY one category name - no explanation, no additional text.
"""

# NEW: Combined Prompt for File Anomaly Identification & Explanation
DEFAULT_FILE_PROMPT = """
You are an expert log analysis assistant tasked with analyzing a log file snippet flagged as anomalous because it contains at least one erroneous log line.

**File Information:**
- Filename: `{filename}`
- **Potential Error Hint:** The filename suggests the primary error might be related to: `{error_type}`.

**Log Snippet & Task:**
Below is a sequence of log lines extracted from the file. Your task is to **identify the specific line(s)** in this snippet that are most likely the anomaly (or directly related to it) AND **provide a detailed explanation** for *why* they are considered anomalous, referencing the filename hint and log content.

**Log Snippet:**
(Format: (Score: Anomaly Score) Msg: Log Message Snippet)
{formatted_log_lines}

**Instructions:**
1.  **Prioritize Filename Hint:** Use the potential error type (`{error_type}`) as your primary guide.
2.  **Analyze Chronological Snippet:** Read through the ordered lines to understand the context.
3.  **Identify & Explain Anomalous Line(s):**
    *   Pinpoint the line(s) most likely related to the `{error_type}` or showing clear errors/high scores.
    *   For **each** identified line:
        *   Quote the **first ~30 characters** of its message content (starting after "Msg: ") for reference.
        *   Provide a **detailed explanation** covering:
            *   Why this specific line is anomalous or relevant to the suspected error (`{error_type}`).
            *   How its content (keywords, errors, status codes) supports this.
            *   How its anomaly score (`Score: ...`) relates, if significant.
            *   Comparison to expected normal behavior (implicitly or explicitly).
            *   Potential root causes if evident.
4.  **Address Missing Error Message (If Applicable):** If the *exact* error message matching the filename (e.g., a line explicitly stating "401 Unauthorized") is *NOT* present in this snippet, **state this explicitly**. Then, explain how the snippet *still supports* the conclusion that the filename's error occurred (e.g., "Although the exact 401 error message isn't shown in this snippet, the presence of stack traces [lines X, Y] with high anomaly scores strongly suggests they resulted from the 401 error indicated by the filename.").
5.  **Summarize:** Conclude with a concise summary of the identified anomaly and its nature within the file context.

**Output Format:**
-   Filename Anomaly Indication: [State the error suggested by the filename]
-   Analysis and Explanation:
    -   Line starting with "(Score: ...) Msg: [First ~30 chars...]": [Detailed explanation for this line...]
    -   Line starting with "(Score: ...) Msg: [First ~30 chars...]": [Detailed explanation for this line...]
    -   [If applicable]: Explicit statement that the direct error message is missing but the snippet provides related evidence (like stack traces).
-   Summary: [Concise summary of the overall anomaly found/inferred in the file snippet]
"""