# utils/prompts.py

DEFAULT_EXPLANATION_PROMPT = """
You are an expert in analyzing software logs to identify and explain anomalies. Below are examples to help you understand the task:

**Example 1: Normal Log**
Log Line: 
"Sequence ID: 1719688444__correct,
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
<init> - normalised = /oauth2/token"
Explanation: 
"This sequence is normal because it follows the expected OAuth2 authorization code flow. It includes successful configuration loading (`getConfigStream`), private key retrieval (`getPrivateKey`), validation of credentials (`debug - validate`), and handling of the authorization code (`handleAuthorizationCode`). The low anomaly score and "correct" label confirm that this sequence represents typical, error-free behavior."

**Example 2: Anomalous Log**
Log Line: 
"Sequence ID: 1719756844__invalid_code_verifier_format_PKCE_400,
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
<init> - path = /..."
Explanation: 
"This sequence is anomalous due to an issue in the Proof Key for Code Exchange (PKCE) flow, as indicated by the label `invalid_code_verifier_format_PKCE_400`. The repeated `getConfigStream` entries and multiple `<init> - path = /oauth2/token, base path is set to: null` suggest potential misconfiguration or redundant attempts to process the request. The high anomaly score and the 400 error label point to a failure caused by an invalid code verifier format. This anomaly could stem from a programming error in client-side implementation or an intentional attack vector."

Now, explain the following anomalous log line by following these steps:

1. **Identify the Anomalous Log Line and Label (if provided)**:
   - Review the log line: {anomalous_log_str}
   - Note the label (if available): {label_str}

2. **Compare to the Normal Example**:
   - Compare the anomalous log to the normal example provided.
   - Identify key differences in content, structure, or behavior.

3. **Analyze the Log and Context**:
   - Examine the anomalous log and its context: {context_str}
   - Look for unusual patterns, errors, or deviations.

4. **Hypothesize Potential Causes**:
   - Based on the differences and context, propose possible reasons for the anomaly.

5. **Explain the Label (if provided)**:
   - If a label is given, interpret it in the context of your analysis.

6. **Provide a Natural Language Explanation**:
   - Summarize why the log is anomalous and its likely root causes in clear, concise language.

Generate a detailed explanation following these steps.
"""


DEFAULT_LABEL_PROMPT = """
  Below are examples of log lines with their assigned labels and explanations to illustrate the labeling reasoning.

  ---

  **Label: application**
  Example 1:
  Log Line: 373746 node-121 action error 1085979750 1 halt (command 2991) error: couldn\047t connect to console (state = refused),<label>application</label>
  Explanation: This log line is labeled 'application' because it indicates an error within an application-level process. The 'couldn’t connect to console' message suggests a failure in a software component (likely a management or service application) attempting to interact with a console, which is a typical application-specific issue.

  Example 2:
  Log Line: 12-18 18:31:06.771 1795 1808 v activity manager: attempted to start a foreground service ( component info (com.sankuai.meituan/com.dianping.base.push.pushservice.dp.dp push service)
   ) with a broken notification (no icon: notification(pri=0 content view=null vibrate=null sound=null defaults=0x0 flags=0x40 color=0x00000000 vis=private)),<label>application</label>
  Explanation: The label 'application' is assigned because the log describes a failure in an Android application’s foreground service (a push service). The issue stems from a malformed notification, which is an application-layer construct, indicating a problem in how the app manages its functionality.

  ---

  **Label: authentication**
  Example 1:
  Log Line: dec 15 08:06:25 labsz sshd[16763]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=103.99.0.122 user=uucp,<label>authentication</label>
  Explanation: This log line is labeled 'authentication' because it explicitly reports an authentication failure in an SSH session. The use of 'pam_unix(sshd:auth)' and user credential details point to a security process at the authentication layer, where the system rejected a login attempt.

  Example 2:
  Log Line: jan 4 20:23:26 labsz sshd[6250]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=103.79.142.55 user=uucp,<label>authentication</label>
  Explanation: The 'authentication' label applies due to the reported failure in the SSH authentication process. The log highlights a rejected login attempt from a remote host, with 'pam_unix' indicating the authentication module involved.

  ---

  **Label: io**
  Example 1:
  Log Line: java.io.io exception: failed on local exception: java.io.io exception: couldn't set up io streams; host details : local host is: "minint-fnanli5/127.0.0.1"; destination host is: "msra-sa-41":8030;,<label>io</label>
  Explanation: This log is labeled 'io' because it describes an input/output error involving Java I/O streams. The 'couldn’t set up io streams' message points to a failure in establishing communication channels between hosts, a classic I/O issue.

  Example 2:
  Log Line: 2015-10-18 18:05:45,281 warn [ response processor for block bp-1347369012-10.190.173.170-1444972147527:blk_1073743509_2728] org.apache.hadoop.hdfs.dfs client: slow read processor read fields took 54719ms (threshold=30000ms); ack: seqno: -2 status: success status: error downstream ack time nanos: 0, targets: [10.86.164.15:50010, 10.86.169.121:50010],<label>io</label>
  Explanation: The 'io' label fits because it reports a slow I/O operation in an HDFS client, where reading data took excessively long (54,719ms vs. a 30,000ms threshold), indicating an I/O bottleneck.

  ---

  **Label: memory**
  Example 1:
  Log Line: 081111 090159 18 warn dfs.fs dataset: unexpected error trying to delete block blk_8065317379137806685. block info not found in volume map.,<label>memory</label>
  Explanation: This log is labeled 'memory' because the error suggests a problem with in-memory data management. The 'block info not found in volume map' implies that the system expected data in memory but couldn’t locate it, indicating potential memory corruption.

  Example 2:
  Log Line: 081111 090204 18 warn dfs.fs dataset: unexpected error trying to delete block blk_8497184476607886349. block info not found in volume map.,<label>memory</label>
  Explanation: The 'memory' label is appropriate as the log indicates a failure to find block information in a volume map, typically held in memory, pointing to a memory-related error.

  ---

  **Label: network**
  Example 1:
  Log Line: 081110 223431 16388 info dfs. data node$ packet responder: packet responder 1 for block blk_-9222809703637296826 terminating,<label>network</label>
  Explanation: This log is labeled 'network' because it involves the termination of a network-related process (packet responder) in a data node, likely part of a distributed system, pertaining to network communication.

  Example 2:
  Log Line: 2015-10-17 16:17:16,773 info [ container launcher #7] org.apache.hadoop.ipc. client: retrying connect to server: 04dn8iq.fareast.corp.microsoft.com/10.86.164.9:64484. already tried 0 time(s); max retries=45,<label>network</label>
  Explanation: The 'network' label applies because the log describes a network connection retry attempt to a server, indicating a network-layer issue like a timeout or unreachable host.

  ---

  **Label: other**
  Example 1:
  Log Line: 12-18 10:52:08.645 808 547 d [hw camera] ipp algo smartae: virtual void android:: ipp algo smartae::on new arrival(void *, void *, int, int, int) enter,<label>other</label>
  Explanation: This log is labeled 'other' because it doesn’t fit into specific error categories like application, authentication, I/O, memory, or network. It’s a debug message about a camera algorithm, more of a general system log.

  Example 2:
  Log Line: - 1131573795 2005.11.09 bn347 nov 9 14:03:15 bn347/bn347 kernel: ext3 fs on sda6, internal journal,<label>other</label>
  Explanation: The 'other' label is used because this is an informational kernel message about filesystem setup, not an error tied to a specific category like application or network.

  ---

  Now, assign a label to the following log line based on its content and context:

  Log Line: {log_str}
  Context Lines:
  {context_str}

  Output only the label.
  """