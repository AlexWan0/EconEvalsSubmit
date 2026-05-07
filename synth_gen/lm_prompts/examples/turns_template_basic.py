from dataclasses import dataclass
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_prompts import Turn, TurnsPrompt, cast_turns


@dataclass
class InterviewInput:
    occupation: str
    task: str


@dataclass
class InterviewOutput:
    email_body: str
    query: str


@dataclass
class InterviewPrompt(TurnsPrompt[InterviewInput, InterviewOutput]):
    require_codeblock: bool = True

    def parse_output_str(self, output_str: str) -> InterviewOutput:
        email_match = re.search(
            r"<email>\s*<subject>.*?</subject>\s*<body>(.*?)</body>\s*</email>",
            output_str,
            re.DOTALL | re.IGNORECASE,
        )
        if not email_match:
            raise ValueError("Missing <email> envelope.")
        email_body = email_match.group(1).strip()
        code_match = re.search(r"```(?:[^\n`]*)?\n?([\s\S]*?)```", email_body, re.DOTALL)
        query = code_match.group(1).strip() if code_match else email_body
        if self.require_codeblock and not code_match:
            raise ValueError("Missing required codeblock.")
        return InterviewOutput(email_body=email_body, query=query)


def dummy_lm(turns: list[Turn]) -> str:
    _ = turns
    return (
        "<email><subject>LLM Query</subject><body>"
        "```"
        "Create a project timeline with milestones."
        "```"
        "</body></email>"
    )


if __name__ == "__main__":
    prompt = InterviewPrompt(
        input_type=InterviewInput,
        output_type=InterviewOutput,
        template=cast_turns([
            {"role": "system", "content": "You are a {occupation}."},
            {"role": "user", "content": "What query did you use for task: {task}?"},
        ]),
    )

    rendered_turns = prompt.render(
        InterviewInput(
            occupation="project manager",
            task="prepare a launch readiness plan",
        )
    )
    print("Rendered turns:\n", rendered_turns)

    raw_output = dummy_lm(rendered_turns)
    print("\nRaw LM output:\n", raw_output)
    print("\nvalidate_output:", prompt.validate_output(raw_output))
    print("parse_output:", prompt.parse_output(raw_output, verbose=True))
