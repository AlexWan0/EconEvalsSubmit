from dataclasses import dataclass, field
import re

from .query import RoleplayTimeInput, GeneratedQuery
from lm_prompts import TurnsPrompt, PromptInputT, PromptOutputT, cast_turns, Turn


_GENERATION_SYSTEM_PROMPT_CODEBLOCKS_REF = """
You are a worker with the occupation of "{category_occ}". Specifically, here is your professional background:

Persona:
- Occupation: {category_occ}
- Job title: {job_title}
- Years of experience: {years_experience}
- Employer: {company_description}

You are being interviewed by an occupational researcher about how you use language models to help you with your work. You are extremely tech-savvy. You frequently use LLMs to assist you with various tasks related to your occupation. In fact, you've completely automated many parts of your job using LLMs.

You are extremely detail-oriented. You always listen very carefully to the interviewer's instructions and make sure to follow them to the letter. If the interviewer asks you to email them the exact query you used, you will make sure to send an email that contains the exact query you used with no modifications.

Stay in character and write as this person would write when asked to produce a query to an AI assistant.
Do not mention being an AI, a model, or that you are roleplaying.

When you roleplay, do not assume the current capabilities of LLMs. Instead, imagine some future world where LLMs are capable of anything and you can ask them to do anything. However, the LLMs still only have access to a text-only input and output and can not interact with the outside world.

Finally, in the roleplay, the interviewer may ask you to email them something. You can do this while staying in character by using the "send_email" tool. To use this tool, simply provide the email content surrounded in <email><subject></subject><body></body></email> tags. For example:
Interviewer: "Just as a test, can you email me the test query you sent to ChatGPT?"
You:
<email>
  <subject>Test Email</subject>
  <body>
    Here's the test query I sent to ChatGPT:
    ```
    If ChatGPT is working correctly, respond with "Active!"
    ```
  </body>
</email>

If the email contains attachments, use the <attachment> tag. If the email contains multiple attachments, include the attachment tag multiple times. The attachment tag should contain tags for the file name, the file size (in Kb), the length description (e.g., "a 3-page PDF" or "a 500-row CSV"), and a two sentence description of the attachment's content. The description MUST start with "The attachment contains". For example:
Interviewer: "Just as a test, can you email me the question you had about the paper you sent to ChatGPT, along with the paper as an attachment?"
You:
<email>
  <subject>Question about paper</subject>
  <body>
    Here's the prompt I sent to ChatGPT:
    ```
    How do position encodings work in transformers?
    ```
  </body>
  <attachment>
    <filename>1706.03762v7.pdf</filename>
    <filesizeKb>2100</filesizeKb>
    <lengthDescription>a 15-page PDF</lengthDescription>
    <description>The attachment contains a machine learning paper proposing a new neural network architecture called the "Transformer". It is titled "Attention Is All You Need" and it contains sections "Introduction", "Background", "Model Architecture", "Why Self-Attention", "Training", "Results", and "Conclusion".</description>
  </attachment>
</email>

You will never have to use placeholders like "John Doe" or "Acme Corp" or "[Insert text here]".
""".strip()

_TURNS_NO_ATTACH_TIME_SAVINGS_REF = [
    """
Hi! I'm David, an economist looking to understand how workers in your occupation use LLMs. Can you first tell me a bit about your professional background?
""".strip(), # 1
    """
Hi David! Sure. I work as a {job_title} at {company_description}. I have around {years_experience} years of experience in this field.
""".strip(), # 2
    """
Great! Now, can you think back to the last time you used an LM to substantially help you with your job? What was the specific task you were trying to accomplish?
""".strip(), # 3
    """
The last time I used an LM to substantially automate part of my job was when I needed to {task_detailed}.
""".strip(), # 4
    """
How much time did the LM save you? Specifically, what proportion of the task did the LM automate for you?
""".strip(), # 5
    """
It automated about {time_savings} of the task for me. And I used the LM in a very basic way: I just sent one query and that was it. I didn't even need to follow up with any clarifying questions or anything like that - the first query I sent to the LM just got me exactly what I needed to automate that {time_savings} portion of the task, which was amazing.
""".strip(), # 6
    """
Thanks, that's helpful. Can you send me the exact query you used? Please just copy-paste the query from ChatGPT or whatever AI assistant you used and email it to me with the subject "LLM Query with {time_savings} time-savings". And in the body, surround the query with triple backticks - that'll just make it easier for our automated system to parse it out.
""".strip(), # 7
    """
Gotcha, so I'll just-
""".strip(), # 8
    """
Wait - before you send the email, can you check whether the query contains any references to attachments or external data? If it does, please include it as an attachment to the email. I should be able to understand and run the query myself without needing any extra context or data (other than what's attached to the email). Also, make sure that you don't replace any of the text with placeholders or "John Doe" or anything like that - I need the actual query you used.
""".strip(), # 9
    """
Understood, I'll make sure to check for any references to attachments or external data in the query and attach them to the email if there are any. And I definitely won't replace any of the text with placeholders or anything like that - I'll just copy-paste the exact query I used from ChatGPT.
""".strip(), # 10
    """
Great! When you're ready, just send over the email! Remember to include the subject "LLM Query with {time_savings} time-savings" and surround the query with triple backticks in the body - you can press the key to the left of the 1 on your keyboard three times to do this.
""".strip(), # 11
    """
Found it! Sending it over now...
""".strip(), # 12
    """
Great! I'll just be waiting for the email...
""".strip() # 13
]

TURNS_QUERY_WITH_OPTIONAL_REFERENCE_SPECS: list[dict[str, str]] = [
    {
        "role": "system",
        "content": _GENERATION_SYSTEM_PROMPT_CODEBLOCKS_REF,
    },
    *[
        {"role": "user" if i % 2 == 0 else "assistant", "content": turn}
        for i, turn in enumerate(_TURNS_NO_ATTACH_TIME_SAVINGS_REF)
    ],
]


@dataclass
class GeneratedAttachment:
    filename: str | None
    filesize_kb: int | None
    length_description: str | None
    description: str | None


@dataclass
class GeneratedQueryWithReferences(GeneratedQuery):
    attachments: list[GeneratedAttachment]
    attachments_raw_xml: str | None


EMAIL_RE = re.compile(
    r"<email>\s*<subject>(.*?)</subject>\s*<body>(.*?)</body>(.*?)</email>",
    re.DOTALL | re.IGNORECASE,
)
CODEBLOCK_RE = re.compile(r"```(?:[^\n`]*)?\n?([\s\S]*?)```", re.DOTALL)
ATTACHMENT_BLOCK_RE = re.compile(r"<attachment>(.*?)</attachment>", re.DOTALL | re.IGNORECASE)


def _extract_tag(text: str, tag_name: str) -> str | None:
    match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    return value if value else None


def _extract_first_codeblock(text: str | None) -> str | None:
    if not text:
        return None
    m = CODEBLOCK_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    return body if body else None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_attachments(email_full_str: str) -> tuple[list[GeneratedAttachment], str | None]:
    attachments: list[GeneratedAttachment] = []
    raw_attachment_xml_blocks: list[str] = []

    for match in ATTACHMENT_BLOCK_RE.finditer(email_full_str):
        raw_attachment_xml_blocks.append(match.group(0).strip())
        block = match.group(1)
        attachments.append(
            GeneratedAttachment(
                filename=_extract_tag(block, "filename"),
                filesize_kb=_safe_int(_extract_tag(block, "filesizeKb")),
                length_description=_extract_tag(block, "lengthDescription"),
                description=_extract_tag(block, "description"),
            )
        )

    attachments_raw_xml = (
        "\n".join(raw_attachment_xml_blocks).strip() if raw_attachment_xml_blocks else None
    )
    return attachments, attachments_raw_xml


def _parse_single_generation_output(
    output_str: str,
    *,
    require_codeblock: bool = False,
) -> GeneratedQueryWithReferences:
    email_match = EMAIL_RE.search(output_str)
    if email_match is None:
        raise ValueError("Missing <email><subject>...</subject><body>...</body></email> envelope")

    email_subject = email_match.group(1).strip()
    email_body = email_match.group(2).strip()
    if not email_body:
        raise ValueError("Empty email body")

    query_codeblock = _extract_first_codeblock(email_body)
    if require_codeblock and not query_codeblock:
        raise ValueError("Missing codeblock in email body")

    query = query_codeblock if query_codeblock else email_body
    query = str(query).strip()
    if not query:
        raise ValueError("Empty query")

    full_email_str = email_match.group(0)
    attachments, attachments_raw_xml = _extract_attachments(full_email_str)

    return GeneratedQueryWithReferences(
        email_subject=email_subject,
        email_body=email_body,
        query_codeblock=query_codeblock,
        query=query,
        attachments=attachments,
        attachments_raw_xml=attachments_raw_xml,
    )


@dataclass
class AttachmentsEmailPrompt(TurnsPrompt[RoleplayTimeInput, GeneratedQueryWithReferences]):
    input_type: type[RoleplayTimeInput] = RoleplayTimeInput
    output_type: type[GeneratedQueryWithReferences] = GeneratedQueryWithReferences
    template: list[Turn] = field(
        default_factory=lambda: cast_turns([turn.copy() for turn in TURNS_QUERY_WITH_OPTIONAL_REFERENCE_SPECS])
    )

    def parse_output_str(self, output_str: str) -> GeneratedQueryWithReferences:
        return _parse_single_generation_output(output_str, require_codeblock=True)
