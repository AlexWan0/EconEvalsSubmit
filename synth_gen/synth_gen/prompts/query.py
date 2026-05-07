from dataclasses import dataclass, field
import re

from lm_prompts import TurnsPrompt, PromptInputT, PromptOutputT, cast_turns, Turn


@dataclass
class RoleplayTimeInput:
    category_occ: str
    task_detailed: str
    time_savings: str
    job_title: str
    years_experience: str
    company_description: str

@dataclass
class GeneratedQuery:
    email_subject: str
    email_body: str
    query_codeblock: str | None
    query: str

_GENERATION_SYSTEM_PROMPT_CODEBLOCKS = """
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

You will never have to use placeholders like "John Doe" or "Acme Corp" or "[Insert text here]".
""".strip()

_TURNS_NO_ATTACH_TIME_SAVINGS = [
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
Wait - before you send the email, can you check whether the query contains any references to attachments or external data? In other words, can you check to make sure that whatever you send me is completely self-contained. I should be able to understand and run the query myself without needing any extra context or data. Also, make sure that you don't replace any of the text with placeholders or "John Doe" or anything like that - I need the actual query you used.
""".strip(), # 9
    """
Hmmm... yep it doesn't reference any attachments or external data. It's definitely completely self-contained and you could totally run it yourself without needing any extra context or data. And I definitely won't replace any of the text with placeholders or anything like that - I'll just copy-paste the exact query I used from ChatGPT.
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

TURNS_NO_ATTACH_TIME_SAVINGS: list[dict[str, str]] = [
    {
        "role": "system",
        "content": _GENERATION_SYSTEM_PROMPT_CODEBLOCKS,
    },
    *[
        {"role": "user" if i % 2 == 0 else "assistant", "content": turn}
        for i, turn in enumerate(_TURNS_NO_ATTACH_TIME_SAVINGS)
    ]
]


EMAIL_RE = re.compile(
    r"<email>\s*<subject>(.*?)</subject>\s*<body>(.*?)</body>\s*</email>",
    re.DOTALL | re.IGNORECASE,
)
CODEBLOCK_RE = re.compile(r"```(?:[^\n`]*)?\n?([\s\S]*?)```", re.DOTALL)


def _extract_email(text: str | None) -> tuple[str, str] | None:
    if not text:
        return None
    m = EMAIL_RE.search(text)
    if not m:
        return None
    subject = m.group(1).strip()
    body = m.group(2).strip()
    return subject, body


def _extract_first_codeblock(text: str | None) -> str | None:
    if not text:
        return None
    m = CODEBLOCK_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    return body if body else None


def _parse_single_generation_output(
    output_str: str,
    *,
    require_codeblock: bool = False,
) -> GeneratedQuery:
    email_subject_body = _extract_email(output_str)
    if email_subject_body is None:
        raise ValueError("Missing <email><subject>...</subject><body>...</body></email> envelope")

    email_subject, email_body = email_subject_body
    email_subject = str(email_subject).strip()
    email_body = str(email_body).strip()
    if not email_body:
        raise ValueError("Empty email body")

    query_codeblock = _extract_first_codeblock(email_body)
    if require_codeblock and not query_codeblock:
        raise ValueError("Missing codeblock in email body")

    query = query_codeblock if query_codeblock else email_body
    query = str(query).strip()
    if not query:
        raise ValueError("Empty query")

    return GeneratedQuery(
        email_subject=email_subject,
        email_body=email_body,
        query_codeblock=query_codeblock,
        query=query,
    )

@dataclass
class BasicEmailPrompt(TurnsPrompt[RoleplayTimeInput, GeneratedQuery]):
    input_type: type[RoleplayTimeInput] = RoleplayTimeInput
    output_type: type[GeneratedQuery] = GeneratedQuery
    template: list[Turn] = field(
        default_factory=lambda: cast_turns([turn.copy() for turn in TURNS_NO_ATTACH_TIME_SAVINGS])
    )

    def parse_output_str(self, output_str: str) -> GeneratedQuery:
        return _parse_single_generation_output(
            output_str,
            require_codeblock=True,
        )
