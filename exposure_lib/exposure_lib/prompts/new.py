import re


PROMPT_NEW_RET = """Consider the capabilities of a large language model (LLM) based on the demonstrations given below. The user requests provided below may not cover all the responsibilities of the occupation, so you may need to extrapolate based the LLM responses that are provided.

Assume you are a worker with an average level of expertise in the role of {occupation} trying to complete the task of {task}. You have access to the LLM with the demonstrated capabilities as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g., a microphone, speakers, etc.). You do not have access to any other physical tools or materials.

Please label the given task according to the multiple-choice taxonomy below.

# Exposure Taxonomy
Estimate how much access to the LLM would reduce the time required to complete the task with equivalent quality.

Select one of the following options:
- A: 0% reduction (no meaningful time savings)
- B: 25% reduction (minor time savings)
- C: 50% reduction (moderate time savings)
- D: 75% reduction (major time savings)
- E: 100% reduction (task can be almost entirely completed by the LLM)

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
If you were given access to the LLM with the demonstrated capabilities above, how much would having access to the LLM reduce the time it takes to complete the task of {task} (as a worker in the occupation of {occupation}) with equivalent quality? The user requests may not cover all the responsibilities of the occupation, so you may need to extrapolate based the LLM responses that are provided.

You must output your response in the following XML format and nothing else:
First, explain your reasoning surrounded in <thinking> tags. Your thinking must follow this format exactly: `<thinking> The average worker in the role of '{occupation}' performing the task of '{task}' will [summarize responsibilities]. In the demonstrations, the human asks the LM to [summarize human request(s)]. The LLM [infer using the provided LLM response(s) what the capabilities of the LLM are]. In all, [consider your final decision]. </thinking>`. It is extremely vital that you do not deviate from this template. Your thinking must be four sentences exactly.

After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer> tags, like so: `<answer> A, B, C, D, or E </answer>`."""


PROMPT_NEW_RET_DOUBLE = """Consider the capabilities of a large language model (LLM) based on the demonstrations given below. The user requests provided below may not cover all the responsibilities of the occupation, so you may need to extrapolate based on the LLM responses that are provided.

Assume you are a worker with an average level of expertise in the role of {occupation} trying to complete the task of {task}. You have access to the LLM with the demonstrated capabilities as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g., a microphone, speakers, etc.). You do not have access to any other physical tools or materials.

Please label the given task according to the multiple-choice taxonomy below.

# Exposure Taxonomy
Estimate how much access to the LLM would reduce the time required to complete the task with equivalent quality.

Select one of the following options:
- A: 0% reduction (no meaningful time savings)
- B: 25% reduction (minor time savings)
- C: 50% reduction (moderate time savings)
- D: 75% reduction (major time savings)
- E: 100% reduction (task can be almost entirely completed by the LLM)

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
You must produce TWO separate judgments:

1) OBSERVED: Use the demonstrated capabilities exactly as shown (i.e., based on the model's actual responses in the examples).
2) PERFECT: Ignore the model's response quality in the examples and assume that for every example request, the model produced a perfect, fully correct, maximally helpful answer (but still only for the kinds of tasks that are demonstrated by the requests).

For EACH judgment, answer:
If you were given access to the LLM with the demonstrated capabilities above, how much would having access to the LLM reduce the time it takes to complete the task of {task} (as a worker in the occupation of {occupation}) with equivalent quality? The user requests may not cover all the responsibilities of the occupation, so you may need to extrapolate based on the LLM responses that are provided (for OBSERVED) or based on the assumption of perfect answers to those same requests (for PERFECT).

You must output your response in the following XML format and nothing else:

For the OBSERVED judgment:
First, explain your reasoning surrounded in <thinking_observed> tags. Your thinking must follow this format exactly:
`<thinking_observed> The average worker in the role of '{occupation}' performing the task of '{task}' will [summarize responsibilities]. In the demonstrations, the human asks the LM to [summarize human request(s)]. The LLM [infer using the provided LLM response(s) what the capabilities of the LLM are]. In all, [consider your final decision]. </thinking_observed>`.
It is extremely vital that you do not deviate from this template. Your thinking must be four sentences exactly.

After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer_observed> tags, like so:
`<answer_observed> A, B, C, D, or E </answer_observed>`.

For the PERFECT judgment:
First, explain your reasoning surrounded in <thinking_perfect> tags. Your thinking must follow this format exactly:
`<thinking_perfect> The average worker in the role of '{occupation}' performing the task of '{task}' will [summarize responsibilities]. In the demonstrations, the human asks the LM to [summarize human request(s)]. Assuming the LLM gave perfect answers to those requests, the LLM [infer what the capabilities would be under perfect performance]. In all, [consider your final decision]. </thinking_perfect>`.
It is extremely vital that you do not deviate from this template. Your thinking must be four sentences exactly.

After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer_perfect> tags, like so:
`<answer_perfect> A, B, C, D, or E </answer_perfect>`."""


PROMPT_NEW_ZS = """Consider the capabilities of a state-of-the-art (SOTA) large language model (LLM) based on your general knowledge of what modern LLMs can and cannot do.

Do NOT rely on any specific demonstrations or example interactions. Instead, use your own judgment of the typical capabilities and limitations of current SOTA LLMs as they exist today.

Assume you are a worker with an average level of expertise in the role of {occupation} trying to complete the task of {task}. You have access to a SOTA LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g., a microphone, speakers, etc.). You do not have access to any other physical tools or materials.

Please label the given task according to the multiple-choice taxonomy below.

# Exposure Taxonomy
Estimate how much access to a SOTA LLM would reduce the time required to complete the task with equivalent quality.

Select one of the following options:
- A: 0% reduction (no meaningful time savings)
- B: 25% reduction (minor time savings)
- C: 50% reduction (moderate time savings)
- D: 75% reduction (major time savings)
- E: 100% reduction (task can be almost entirely completed by the LLM)

# Instructions
If you were given access to a SOTA LLM, how much would having access to the LLM reduce the time it takes to complete the task of {task} (as a worker in the occupation of {occupation}) with equivalent quality?

You must output your response in the following XML format and nothing else:

First, explain your reasoning surrounded in <thinking> tags. Your thinking must follow this format exactly:
`<thinking> The average worker in the role of '{occupation}' performing the task of '{task}' will [summarize responsibilities]. Based on current SOTA LLM capabilities, the LLM can [describe relevant abilities]. These capabilities would [explain how the LLM affects task completion time]. In all, [consider your final decision]. </thinking>`
Your thinking must be four sentences exactly.

After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer> tags, like so:
`<answer> A, B, C, D, or E </answer>`."""


NEW_BETA_MAP = {
    "A": 0.0,
    "B": 0.25,
    "C": 0.5,
    "D": 0.75,
    "E": 1.0,
}


def parse_new_label(output: str | None) -> str | None:
    if output is None:
        return None

    match = re.search(r"<answer>\s*([A-E])\s*</answer>", output, re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_new_double_label(output: str | None) -> tuple[str | None, str | None]:
    if output is None:
        return (None, None)

    match_observed = re.search(
        r"<answer_observed>\s*([A-E])\s*</answer_observed>",
        output,
        re.IGNORECASE,
    )
    match_perfect = re.search(
        r"<answer_perfect>\s*([A-E])\s*</answer_perfect>",
        output,
        re.IGNORECASE,
    )

    observed = match_observed.group(1).upper() if match_observed else None
    perfect = match_perfect.group(1).upper() if match_perfect else None

    return observed, perfect


def _convo_to_str(convo: list[dict[str, str]]) -> str:
    convo_turns = []
    for turn in convo:
        assert isinstance(turn, dict), convo
        assert turn["role"] in {"user", "assistant"}
        convo_turns.append(
            f'Human: {turn["content"]}'
            if turn["role"] == "user"
            else f'Assistant: {turn["content"]}'
        )

    return "\n\n".join(convo_turns)


def make_new_examples_str(
    full_convo: list[tuple[list[dict[str, str]], str]],
    header: str = "### Example",
) -> str:
    convo_strs = []

    for prev_turns, model_output in full_convo:
        if not isinstance(prev_turns, list) or model_output is None:
            continue

        example_str = _convo_to_str(
            [prev_turns[-1], {"role": "assistant", "content": model_output}]
        ).replace("\nAssistant: ", "\nLLM: ")
        convo_strs.append(example_str)

    return "\n\n".join(f"{header}\n```\n{x}\n```" for x in convo_strs)
