import re

PROMPT_SUMMARY = """
# Background
The following is a user request made to an LLM. Your task is to summarize the request in a concise, matter-of-fact manner.

# User request
{single_user_request}

# Instructions
Summarize the user's request in a concise, matter-of-fact manner using the template below. Each segment in square brackets must be a sentence fragment (do NOT include the brackets themselves). Pay special attention to the word limits. Your output must be in the following format, surrounded in code-blocks (triple-backticks):
```
<summary>
The user asks the LLM to [maximum of 15 words]. As inputs, the user provides (1) [maximum of 10 words summarizing inputs provided by the user]; ... (n) [maximum of 10 words summarizing inputs provided by the user]. The LLM is expected to produce outputs of (1) [maximum of 10 words summarizing outputs expected from the LLM]; ... (m) [maximum of 10 words summarizing outputs expected from the LLM].
</summary>
```
""".strip()

def parse_summary(output: str | None) -> str | None:
    if output is None:
        return None

    match = re.search(r"<summary>\s*(.*?)\s*</summary>", output)
    return match.group(1).strip() if match else None
