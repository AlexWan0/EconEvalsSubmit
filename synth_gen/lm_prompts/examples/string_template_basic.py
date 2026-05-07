from dataclasses import dataclass
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_prompts import StringPrompt


@dataclass
class SupportInput:
    customer_name: str
    issue: str


@dataclass
class SupportOutput:
    email_body: str
    severity: str


@dataclass
class SupportReplyPrompt(StringPrompt[SupportInput, SupportOutput]):
    def parse_output_str(self, output_str: str) -> SupportOutput:
        body_match = re.search(r"<email_body>(.*?)</email_body>", output_str, re.DOTALL)
        severity_match = re.search(r"<severity>(.*?)</severity>", output_str, re.DOTALL)
        if not body_match or not severity_match:
            raise ValueError("Missing required <email_body>/<severity> tags.")
        return SupportOutput(
            email_body=body_match.group(1).strip(),
            severity=severity_match.group(1).strip(),
        )


def dummy_lm(prompt: str) -> str:
    return (
        "<email_body>"
        "Thanks for reporting this. We are actively looking into the issue."
        "</email_body>"
        "<severity>medium</severity>"
    )


if __name__ == "__main__":
    prompt = SupportReplyPrompt(
        input_type=SupportInput,
        output_type=SupportOutput,
        template=(
            "Write a short support reply to {customer_name} about {issue}. "
            "Output tags: <email_body>...</email_body><severity>...</severity>"
        ),
    )

    rendered = prompt.render(
        SupportInput(
            customer_name="Taylor",
            issue="the dashboard export failed",
        )
    )
    print("Rendered prompt:\n", rendered)

    raw_output = dummy_lm(rendered)
    print("\nRaw LM output:\n", raw_output)

    parsed = prompt.parse_output(raw_output, verbose=True)
    print("\nParsed object:\n", parsed)
