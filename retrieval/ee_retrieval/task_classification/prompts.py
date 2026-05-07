from dataclasses import dataclass, field
from typing import Callable, Any


#: v1 prompt for task classification
prompt_v1 = """You are an economist analyzing occupational tasks that are at risk for automation by large language models. You have a large list of occupational tasks, however many of these involve e.g., interacting with the physical world, which doesn't make much sense for an LLM to do. You want to filter this list down to just the tasks that could *hypothetically* be completed with only a text input and text output. Don't worry about whether you think LLMs can complete the task yet. Just consider whether it's hypothetically possible for a virtual assistant to be able to complete the task with only a text input and text output.

First explain your reasoning in a single sentence in <thinking></thinking> tags. Then, answer either "Yes" or "No" for whether or not the task that could *hypothetically* be completed by a virtual assistant with only a text input and text output.

# Example
## Input
Repair and maintain gasoline engines used to power equipment such as portable saws, lawn mowers, generators, and compressors.

## Output
<thinking>Repairing and maintaining involves interacting with the physical world, so it is not possible.</thinking> <answer>No</answer>

# Example
## Input
Consult with clients to assess risks and to determine security requirements.

## Output
<thinking>Communicating with clients can be done through only a text interface.</thinking> <answer>Yes</answer>

# Target
## Input
{task_input}

## Output""".strip()

#: v2 prompt for task classification
prompt_v2 = """
# Background
You are an economist analyzing occupational tasks that are at risk for automation by AI. You have a large list of occupational tasks, some of which involve interacting with the physical world, which is not possible for current AI assistants. Your goal is to filter these out.

# Instructions
Determine whether the input task *only* involves interacting with the physical world or if substantial parts of it could hypothetically be done with a computer.

First, think about your answer in a single sentence using <thinking></thinking> tags. Then, using <answer></answer> tags, provide your final answer (either "Yes" or "No") for whether the task *only* involves interacting with the physical world.

# Example
## Input
Plant crops, trees, or other plants.

## Output
<thinking>This task solely involves interacting with the physical world through actions like digging, planting, and tending crops, with no substantial part that can be done by a computer alone.</thinking>
<answer>Yes</answer>

# Example
## Input
Consult with clients to assess risks and to determine security requirements.

## Output
<thinking>Consulting with clients to assess risk and determine security requirements relies on analysis and communication tasks that can be performed on the computer.</thinking>
<answer>No</answer>

# Target
## Input
{task_input}

## Output""".strip()

import re

def _parse_output(
        raw_text: str | None,
        tag: str = 'answer',
        labels: list[str] | None = ['Yes', 'No']
    ) -> str | None:

    if raw_text is None:
        return None

    pattern = rf'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>'
    m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    if labels is None:
        return m.group(1).strip()

    candidate = m.group(1).strip().lower()
    for label in labels:
        if label.strip().lower() == candidate:
            return label

    return None

@dataclass
class TaskPrompt:
    """Prompt/classification configuration for using LMs to filter out DWAs.
    
    Currently, this just means filtering out DWAs which, even hypothetically, could not be performed by LMs. However, we no longer perform this initial filtering step; we consider all DWAs as even DWAs that could not be performed by an LM alone can still be significantly assisted by an LM.
    """
    prompt: str
    keep_label: str
    parse_func: Callable[[str | None], str | None]
    model_kwargs: dict[str, Any] = field(default_factory = lambda: {
        'model_name': 'gpt-4.1',
        'max_tokens': 1024,
        'temperature': 0.5
    })

#: A mapping from version to :class:`TaskPrompt`.
#:
#: - ``'v1'`` to :data:`prompt_v1`
#: - ``'v2'`` to :data:`prompt_v2`
#:
#: See :class:`TaskPrompt` and the module :mod:`ee_retrieval.task_classification.prompts`.
TASK_CLASSIF_PROMPTS: dict[str, TaskPrompt] = {
    'v1': TaskPrompt(prompt_v1, 'Yes', _parse_output),
    'v2': TaskPrompt(prompt_v2, 'No', _parse_output),
}
