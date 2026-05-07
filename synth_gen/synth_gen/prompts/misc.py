from dataclasses import dataclass, field
import re

from lm_prompts import TurnsPrompt, StringPrompt, PromptInputT, PromptOutputT, cast_turns, Turn


# Prompt for rewriting dialogue turns
@dataclass
class RewriteInput:
    user_turn: str
    assistant_turn: str

@dataclass
class RewriteOutput:
    revised_assistant_turn: str

_REWRITE_SYSTEM_PROMPT = """
You are editing a draft assistant reply to sound natural and human-like. The current response was created through template-filling, so it sounds a bit off. Preseve the meaning but make it sound natural.
You will be provided the user's request and the assistant's draft reply. Edit the assistant's draft reply to sound like a natural human-like response to the user's request. Make sure not to alter the meaning.
Return only the revised assistant reply.
""".strip()

REWRITE_PROMPT = [
    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Previous user turn:\n"
            "{user_turn}\n\n"
            "Assistant draft reply:\n"
            "{assistant_turn}"
        ),
    },
]

@dataclass
class RewritePrompt(TurnsPrompt[RewriteInput, RewriteOutput]):
    input_type: type[RewriteInput] = RewriteInput
    output_type: type[RewriteOutput] = RewriteOutput
    template: list[Turn] = field(default_factory=lambda: cast_turns([turn.copy() for turn in REWRITE_PROMPT]))

    def parse_output_str(self, output_str: str) -> RewriteOutput:
        return RewriteOutput(revised_assistant_turn=output_str.strip())


# Prompt for rewriting queries to be more neutral and concise
POSTPROCESS_QUERY_PROMPT = """
# Background
You will be provided a prompt made to an LM. Rewrite this prompt following the instructions below.

# Rewrite Instructions
- Rewrite this prompt in a neutral, matter-of-fact tone
- The rewritten prompt must not use fluff words like "write a *comprehensive* report", "create *procurement-ready* descriptions", "I need a *complete* budget planning and allocation package for our purchasing department
- Remove anything that would not be directly useful for completing the task itself. For example: you must remove explicit mentions of things being "the output should be self-contained", you must also remove explicit mentions of automation (e.g., "the result should automate 25% of my work")

# LM Prompt to Rewrite
```
{query}
```

# Output Format
Output ONLY the rewritten prompt. Do not output anything except the rewritten prompt. Do not surround the rewritten prompt in backticks.
""".strip()
