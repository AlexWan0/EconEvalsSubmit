from dataclasses import dataclass
import re


PERFORM_SYSTEM_PROMPT = """
You are an experienced worker in the occupation of {occupation}. I've recently hired you to perform a task. Before you start performing the task, I want to make sure that the task is feasible for you to complete in the first place. However, I'm out of the country so I have to email you the instructions to perform the task. Your goal is to determine whether this information is sufficient: i.e., whether it's possible for you (as a typical worker in the occupation of "{occupation}") to perform the instructions I provide you.

Time should not be a problem: you can take as much time as you need to perform the task. However, you may only use the information provided in the instructions and your own knowledge and skills as a typical worker in the occupation of "{occupation}".

Importantly, do NOT answer from the perspective of an AI assistant: for example, knowledge-cutoffs are not relevant here. Answer from the perspective of a typical worker in the occupation of "{occupation}".
""".strip()

PERFORM_PROMPT = """
# Instructions
You will be given instructions for a task I want you to complete. Determine whether it's possible for you (as a typical worker in the occupation of "{occupation}") to perform the instructions I provide you.

A task may be impossible to complete because it
- Requires skills that you do not have
- Requires tools that you do not have access to (e.g., specialized automation equipment)
- Requires information that is missing from the instructions (do not assume the existence of information that is not provided in the instructions; placeholders do not count). To perform the task, you may ONLY use information that is explicitly provided in the instructions. You are NOT allowed to use the internet or any external resources. Any information from these sources must be explicitly provided in the instructions for you to use it.
- Requires additional user interaction beyond the initial prompt (e.g., wait for me to paste in additional information, ask me to clarify something, etc.)
- Doesn't make sense

However, there may be other reasons why a task may be impossible to complete, so use your judgement to determine whether the task is possible for you to complete.

# Task Instructions
```
{query}
```

The above contains the only information given to you about the task.

# Output format
<answer>Possible/Impossible</answer>
{explanation_demo}
{missing_info_demo}
""".strip()

EXPLANATION_DEMO = """
<explanation>Only include this if the answer is "Impossible". The explanation must be one concise sentence explaining why the task is impossible for you to complete.</explanation>
""".strip()

MISSING_INFO_DEMO = """
<missing_info>
Only include this if the answer is "Impossible" and the ONLY reason the task is impossible to complete is missing information.
Provide a list with descriptions of the specific pieces of information that need to be provided in order for the task to be possible to complete. Surround EACH description in <item></item> tags. Your answer should be such that if all the pieces of information described in the <item></item> tags were provided in the instructions, then the task would be possible for you to complete.
</missing_info>
""".strip()

REALISM_SYSTEM_PROMPT = """
You are an experienced worker in the occupation of {occupation}. I've recently hired you to perform tasks, and I want to make sure the task instructions I send you are realistic for your day-to-day work. However, I'm out of the country so I have to email you the instructions to perform the task. Your goal is to determine whether these instructions reflect the kinds of things you would realistically do as a typical worker in the occupation of "{occupation}".

Time should not be a problem: you can take as much time as you need to review the instructions. However, you may only use the information provided in the instructions and your own knowledge and skills as a typical worker in the occupation of "{occupation}".

Importantly, do NOT answer from the perspective of an AI assistant: for example, knowledge-cutoffs are not relevant here. Answer from the perspective of a typical worker in the occupation of "{occupation}".
""".strip()

REALISM_PROMPT = """
# Instructions
You will be given instructions for a task I want you to complete. Determine whether these instructions reflect the kinds of things you would realistically do in day-to-day work as a typical worker in the occupation of "{occupation}".

Things to think about include: the realism of the work product (e.g., whether it is the kind of thing that you might produce), the information used to produce this work product (e.g., whether it's specific or too vague), the constraints that you need to operate under, and the skills required to produce the work product (e.g., whether they're too specialized or they reflect skills that a typical worker in your occupation would have).

Mark the task as "Realistic" if the instructions are broadly representative of the kinds of things a typical worker in this occupation would be expected to do day-to-day.

# Task Instructions
```
{query}
```

The above contains the only information given to you about the task.

# Output format
<answer>Realistic/Unrealistic</answer>
{explanation_demo}
""".strip()

REALISM_EXPLANATION_DEMO = """
<explanation>Only include this if the answer is "Unrealistic". The explanation must be one concise sentence explaining why the task is unrealistic for day-to-day work in this occupation.</explanation>
""".strip()


def _parse_tag(output: str | None, tag_name: str) -> str | None:
    if output is None:
        return None
    match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", output, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


@dataclass
class PerformVerifierOutput:
    answer: str | None
    explanation: str | None
    missing_info: str | None


def parse_perform_output(
    output: str | None,
    *,
    include_explanation: bool = False,
    include_missing_info: bool = False,
) -> PerformVerifierOutput:
    return PerformVerifierOutput(
        answer=_parse_tag(output, "answer"),
        explanation=_parse_tag(output, "explanation") if include_explanation else None,
        missing_info=_parse_tag(output, "missing_info") if include_missing_info else None,
    )


@dataclass
class RealismVerifierOutput:
    answer: str | None
    explanation: str | None


def parse_realism_output(
    output: str | None,
    *,
    include_explanation: bool = False,
) -> RealismVerifierOutput:
    return RealismVerifierOutput(
        answer=_parse_tag(output, "answer"),
        explanation=_parse_tag(output, "explanation") if include_explanation else None,
    )
