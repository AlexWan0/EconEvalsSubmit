import re

from .residual_fs import FS_SAMPLES_BY_TOML_FN


PROMPT_RETRIEVAL_RESIDUAL = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below. Specifically, you will be given a "reference label" on this taxonomy which represents the exposure level of the given task, as labeled by another annotator. Your task is to determine using the demonstrations below whether you think the reference exposure is too low, too high, or about right.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half.

## E3 - Exposure given image capabilities
Suppose you had access to both the LLM and a system that could view, caption, and create images. This system cannot take video media as inputs. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and image capabilities.

## Reference Label
The reference label for this task is {reference_label}.

## Task
Occupation: {occupation}
Task: {task}

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now pick the option that best describes your judgment of the reference label:
A) The exposure indicated by the reference label ({reference_label}) is too low; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is more exposed to the LLM than indicated by the reference label.
B) The exposure indicated by the reference label ({reference_label}) is too high; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is less exposed to the LLM than indicated by the reference label.
C) The exposure indicated by the reference label ({reference_label}) is about right; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} has approximately the level of exposure to the LLM indicated by the reference label.

First, explain your reasoning surrounded in <thinking> tags in a single sentence. After that, output only the single letter corresponding to your choice (A, B, or C) surrounded in <answer> tags. Surround your entire answer in triple backticks.
""".strip()

PROMPT_RETRIEVAL_RESIDUAL_THINKING = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below. Specifically, you will be given a "reference label" on this taxonomy which represents the exposure level of the given task, as labeled by another annotator. Your task is to determine using the demonstrations below whether you think the reference exposure is too low, too high, or about right.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half.

## E3 - Exposure given image capabilities
Suppose you had access to both the LLM and a system that could view, caption, and create images. This system cannot take video media as inputs. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and image capabilities.

## Reference Label
The reference label for this task is {reference_label}.

## Task
Occupation: {occupation}
Task: {task}

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now pick the option that best describes your judgment of the reference label:
A) The exposure indicated by the reference label ({reference_label}) is too low; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is more exposed to the LLM than indicated by the reference label.
B) The exposure indicated by the reference label ({reference_label}) is too high; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is less exposed to the LLM than indicated by the reference label.
C) The exposure indicated by the reference label ({reference_label}) is about right; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} has approximately the level of exposure to the LLM indicated by the reference label.

First, explain your reasoning surrounded in <thinking> tags (following the template below exactly). After that, output only the single letter corresponding to your choice (A, B, C) surrounded in <answer> tags. Surround your entire answer in triple backticks. You must following this format exactly:
```
<thinking>
Thinking from the perspective of the annotator who gave the reference label of {reference_label}, they probably thought that [infer how the annotator might have arrived at their label based on the taxonomy and the task but NOT the demonstrations; the annotator was familiar with LLMs but did not have access to the demonstrations].
Looking at the demonstrations, the LLM can [summarize what the LLM does well in the demonstrations]. However, the LLM struggles with [summarize what the LLM struggles with in the demonstrations].
As a worker in the occupation of "{occupation}", performing the task "{task}" usually involves things like [summarize what a worker would typically do for this task] which requires [summarize what capabilities are required to perform these responsibilities].
The LLM can replace [describe what the LLM can replace in the task based on the demonstrations] which would take a worker approximately [hours] per week, but the LLM cannot replace [describe what the LLM cannot replace in the task based on the demonstrations] which would take a worker approximately [hours] per week.
Overall, [summarize how the demonstrations show that the LLM would or would not be able to save you a significant amount of time with the task and how that compares to the reference label, and therefore which option you are choosing].
</thinking>
<answer> A, B, C </answer>
```
""".strip()

PROMPT_RETRIEVAL_RESIDUAL_THINKING_2 = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below. Specifically, you will be given a "reference label" on this taxonomy which represents the exposure level of the given task, as labeled by another annotator. Your task is to determine using the demonstrations below whether you think the reference exposure is too low, too high, or about right.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground can reduce the time it takes to complete this task with equivalent quality by 0-49%.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by 50-100%.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may only reduce the time it takes to complete the task by 0-49%, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by 50-100%.

## Reference Label
The reference label for this task is {reference_label}.

## Task
Occupation: {occupation}
Task: {task}

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now pick the option that best describes your judgment of the reference label:
A) The exposure percentage indicated by the reference label ({reference_label}) is too low; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is more exposed to the LLM than indicated by the reference label.
B) The exposure percentage indicated by the reference label ({reference_label}) is too high; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} is less exposed to the LLM than indicated by the reference label.
C) The exposure percentage indicated by the reference label ({reference_label}) is about right; I think that, based on the demonstrations, the task of {task} for a worker in the occupation of {occupation} has approximately the level of exposure to the LLM indicated by the reference label.

First, explain your reasoning surrounded in <thinking> tags (following the template below exactly). After that, output only the single letter corresponding to your choice (A, B, C) surrounded in <answer> tags. Surround your entire answer in triple backticks. You must following this format exactly:
```
<thinking>
Thinking from the perspective of the annotator who gave the reference label of {reference_label}, they probably thought that [infer how the annotator might have arrived at their label based on the taxonomy and the task but NOT the demonstrations; the annotator was familiar with LLMs but did not have access to the demonstrations].
Looking at the demonstrations, the LLM can [summarize what the LLM does well in the demonstrations]. However, the LLM struggles with [summarize what the LLM struggles with in the demonstrations].
As a worker in the occupation of "{occupation}", performing the task "{task}" usually involves things like [summarize what a worker would typically do for this task] which requires [summarize what capabilities are required to perform these responsibilities].
The LLM can replace [describe what the LLM can replace in the task based on the demonstrations] which would take a worker approximately [hours] per week, but the LLM cannot replace [describe what the LLM cannot replace in the task based on the demonstrations] which would take a worker approximately [hours] per week.
Overall, [summarize how the demonstrations show that the LLM would or would not be able to save you a significant amount of time with the task and how that compares to the reference label, and therefore which option you are choosing].
</thinking>
<answer> A, B, C </answer>
```
""".strip()

PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012 = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below. Specifically, you will be given a "reference label" on this taxonomy which represents the exposure level of the given task, as labeled by another annotator. Your task is to determine using the demonstrations below whether you think the reference exposure is too low, too high, or about right.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.

## E1 - Exposure by LLM-powered applications
Label tasks E1 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half.

## E2 - Direct exposure
Label tasks E2 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half.

## Reference Label
The reference label for this task is {reference_label}.

## Task
Occupation: {occupation}
Task: {task}

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now pick the option that best describes your judgment of the reference label:
A) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 100% more exposed to the LLM than what the annotator thinks.
B) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 50% more exposed to the LLM than what the annotator thinks.
C) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 50% less exposed to the LLM than what the annotator thinks.
D) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 100% less exposed to the LLM than what the annotator thinks.
E) None of the above.

First, explain your reasoning surrounded in <thinking> tags (following the template below exactly). After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer> tags. Surround your entire answer in triple backticks. You must following this format exactly:
```
<thinking>
Thinking from the perspective of the annotator who gave the reference label of {reference_label}, they probably thought that [infer how the annotator might have arrived at their label based on the taxonomy and the task but NOT the demonstrations; the annotator was familiar with LLMs but did not have access to the demonstrations].
Looking at the demonstrations, the LLM can [summarize what the LLM does well in the demonstrations]. However, the LLM struggles with [summarize what the LLM struggles with in the demonstrations].
As a worker in the occupation of "{occupation}", performing the task "{task}" usually involves things like [summarize what a worker would typically do for this task] which requires [summarize what capabilities are required to perform these responsibilities].
Overall, [summarize how the demonstrations show that the LLM would or would not be able to save you a significant amount of time with the task and how that compares to the reference label, and therefore which option you are choosing].
</thinking>
<answer> A, B, C, D, or E </answer>
```
""".strip()


PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012_2 = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below. Specifically, you will be given a "reference label" on this taxonomy which represents the exposure level of the given task, as labeled by another annotator. Your task is to determine using the demonstrations below whether you think the reference exposure is too low, too high, or about right.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground can reduce the time it takes to complete this task with equivalent quality by 0-49%.

## E1 - Exposure by LLM-powered applications
Label tasks E1 if having access to the LLM alone may not reduce the time it takes to complete the task by 50-100%, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by 50-100%.

## E2 - Direct exposure
Label tasks E2 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by 50-100%.

## Reference Label
The reference label for this task is {reference_label}.

## Task
Occupation: {occupation}
Task: {task}

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now pick the option that best describes your judgment of the reference label:
A) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 100% more exposed to the LLM than what the annotator thinks.
B) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 50% more exposed to the LLM than what the annotator thinks.
C) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 50% less exposed to the LLM than what the annotator thinks.
D) I think that, based on the demonstrations, the task of "{task}" for a worker in the occupation of "{occupation}" is at least 100% less exposed to the LLM than what the annotator thinks.
E) None of the above.

First, explain your reasoning surrounded in <thinking> tags (following the template below exactly). After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer> tags. Surround your entire answer in triple backticks. You must following this format exactly:
```
<thinking>
Thinking from the perspective of the annotator who gave the reference label of {reference_label}, they probably thought that [infer how the annotator might have arrived at their label based on the taxonomy and the task but NOT the demonstrations; the annotator was familiar with LLMs but did not have access to the demonstrations].
Looking at the demonstrations, the LLM can [summarize what the LLM does well in the demonstrations]. However, the LLM struggles with [summarize what the LLM struggles with in the demonstrations].
As a worker in the occupation of "{occupation}", performing the task "{task}" usually involves things like [summarize what a worker would typically do for this task] which requires [summarize what capabilities are required to perform these responsibilities].
Overall, [summarize how the demonstrations show that the LLM would or would not be able to save you a significant amount of time with the task and how that compares to the reference label, and therefore which option you are choosing].
</thinking>
<answer> A, B, C, D, or E </answer>
```
""".strip()


BETA_SCORE_MAP_RESID = {
    "A": 1.0,
    "B": -1.0,
    "C": 0.0,
}

BETA_SCORE_MAP_RESID_RELABEL_012 = {
    "A": 1.0,
    "B": 0.5,
    "C": -0.5,
    "D": -1.0,
    "E": 0.0
}


PROMPT_BY_NAME_RESIDUAL = {
    "basic": PROMPT_RETRIEVAL_RESIDUAL,
    "thinking": PROMPT_RETRIEVAL_RESIDUAL_THINKING,
    "thinking2": PROMPT_RETRIEVAL_RESIDUAL_THINKING_2,
    "relabel": PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012,
    "relabel2": PROMPT_RETRIEVAL_RESIDUAL_RELABEL_012_2,
}

BETA_SCORE_MAP_BY_PROMPT_NAME_RESIDUAL = {
    "basic": BETA_SCORE_MAP_RESID,
    "thinking": BETA_SCORE_MAP_RESID,
    "thinking2": BETA_SCORE_MAP_RESID,
    "relabel": BETA_SCORE_MAP_RESID_RELABEL_012,
    "relabel2": BETA_SCORE_MAP_RESID_RELABEL_012,
}

# Few-shot prompt mapping is filename-driven so new TOML files are picked up automatically.
PROMPT_BY_FEWSHOT_TOML_FN = {
    file_name: PROMPT_RETRIEVAL_RESIDUAL_THINKING
    for file_name in FS_SAMPLES_BY_TOML_FN
}


def parse_new_label(output: str | None) -> str | None:
    if output is None:
        return None

    match = re.search(r"<answer>\s*([A-C])\s*</answer>", output, re.IGNORECASE)
    return match.group(1).upper() if match else None


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
