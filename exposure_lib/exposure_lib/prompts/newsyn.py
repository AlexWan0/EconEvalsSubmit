import re


PROMPT_RETRIEVAL_MCQ_SYNTH = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below. Do NOT assume any capabilities beyond what is demonstrated in the examples.

You are a worker with an average level of expertise in the role of {occupation} trying to complete the task of {task}.

Assume that the LLM does not have any capabilities beyond what is demonstrated in the examples, even if you personally know that LLMs can do more than what is shown. For example, if the demonstrations only show the LLM answering questions, then you should assume that the LLM cannot perform other tasks such as writing code or generating images.

Please label the given task according to the multiple-choice taxonomy below and the demonstrated LLM capabilities.

# Target task
Occupation: {occupation}
Task: {task}

# Exposure Taxonomy
Estimate how much access to the LLM would reduce the time required to complete the task of "{task}" with equivalent quality.

Select one of the following options:
- A: 0-10% reduction (little to no time savings; I would still need to do almost all of the work)
- B: 11-30% reduction (small time savings; I would still need to complete most of the work myself)
- C: 31-60% reduction (moderate time savings; the LLM would reduce a meaningful portion of the work, but substantial effort would still be required)
- D: 61-85% reduction (large time savings; the LLM would complete most of the work, though I would still need to contribute some effort)
- E: 86-100% reduction (almost complete time savings; the LLM would handle nearly all of the work, and I would only need to make minimal contributions)

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
If you were given access to the LLM with the demonstrated capabilities above, how much would having access to the LLM reduce the time it takes to complete the task of {task} (as a worker in the occupation of {occupation}) with equivalent quality? 

You must output your response in the following XML format. First, explain your reasoning surrounded in <thinking> tags. After that, output only the single letter corresponding to your choice (A, B, C, D, or E) surrounded in <answer> tags. Surround your entire answer in triple backticks.

Follow this format exactly. You MUST NOT deviate from this template:
```
<thinking>
First, I should look carefully at the demonstrations. In the demonstrations, the human asks the LM to [summarize human request(s)]. In response, LLM [summarize what capabilities the LLM has demonstrated].

Next, I should check whether the LLM's response is correct and relevant to the human's request: [check for correctness and relevance]

Now, I should think about what I do day-to-day and how much time different things take. The average worker in the role of '{occupation}' performing the task of '{task}' will need to [summarize day-to-day responsibilities for this task].

If I had access to this LLM, I would no longer need to [summarize what the LLM would automate]. However, I would still need to [summarize what the human would still need to do]. Given this, I estimate that having access to the LLM would reduce the time required to complete the task by approximately [estimate percentage reduction]%, which corresponds to option [A/B/C/D/E].
</thinking>
<answer> A, B, C, D, or E </answer>
```
""".strip()


SYNTH_EXPOSURE_MAP = {
    "A": 0.05,
    "B": 0.205,
    "C": 0.455,
    "D": 0.73,
    "E": 0.93,
}


def parse_answer_label(output: str | None) -> str | None:
    if output is None:
        return None

    match = re.search(r"<answer>\s*([A-E])\s*</answer>", output, re.IGNORECASE)
    return match.group(1).upper() if match else None


def make_examples_str(
    full_convo: list[tuple[list[dict[str, str]], str]],
    header: str = "### Example",
) -> str:
    example_blocks = []

    for prev_turns, model_output in full_convo:
        if not isinstance(prev_turns, list) or model_output is None:
            continue

        user_turns = [turn for turn in prev_turns if turn.get("role") == "user"]
        if not user_turns:
            continue

        human_text = user_turns[-1]["content"]
        example_blocks.append(
            "\n".join(
                [
                    header,
                    "```",
                    "<human>",
                    str(human_text),
                    "</human>",
                    "",
                    "<LLM>",
                    str(model_output),
                    "</LLM>",
                    "```",
                ]
            )
        )

    return "\n\n".join(example_blocks)


PROMPT_NEWSYN_RET = PROMPT_RETRIEVAL_MCQ_SYNTH

NEWSYN_BETA_MAP = SYNTH_EXPOSURE_MAP

parse_newsyn_label = parse_answer_label
make_newsyn_examples_str = make_examples_str
