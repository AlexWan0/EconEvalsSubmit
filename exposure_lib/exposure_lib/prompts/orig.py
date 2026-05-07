import re


PROMPT_ORIG = """
Consider the capabilities of a large language model (LLM). This model can complete many tasks that can be formulated as having text input and text output where the context for the input can be captured in 2000 words. The model also cannot draw up-to-date facts (those from <1 year ago) unless they are captured in the input.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.
If a task requires a high degree of human interaction (for example, in person demonstrations) then it should be classified as E0.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half. This includes tasks that can be reduced to: - Writing and transforming text and code according to complex instructions, - Providing edits to existing text or code following specifications, - Writing code that can help perform a task that used to be done by hand, - Translating text between languages, - Summarizing medium-length documents, - Providing feedback on documents, - Answering questions about a document, or - Generating questions a user might want to ask about a document.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half. This software may include capabilities such as: - Summarizing documents longer than 2000 words and answering questions about those documents - Retrieving up-to-date facts from the Internet and using those facts in combination with the LLM capabilities - Searching over an organization's existing knowledge, data, or documents and retreiving information
Examples of software built on top of the LLM that may help complete worker activities include: - Software built for a home goods company that quickly processes and summarizes their up-to-date internal data in customized ways to inform product or marketing decisions - Software that is able to suggest live responses for customer service agents speaking to customers in their company's customer service interface - Software built for legal purposes that can quickly aggregate and summarize all previous cases in a particular legal area and write legal research memos tailored to the law firm's needs - Software specifically designed for teachers that allows them to input a grading rubric and upload the text files of all student essays and have the software output a letter grade for each essay - Software that retrieves up-to-date facts from the internet and uses the capabilities of the LLM to output news summaries in different languages

## E3 - Exposure given image capabilities
Suppose you had access to both the LLM and a system that could view, caption, and create images. This system cannot take video media as inputs. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and these image capabilities: - Reading text from PDFs, - Scanning images, or - Creating or editing digital images according to instructions.

## Annotation examples:
Occupation: Inspectors, Testers, Sorters, Samplers, and Weighers
Task: Adjust, clean, or repair products or processing equipment to correct defects found during inspections.
Label (E0/E1/E2/E3): E0
Explanation: The model does not have access to any kind of physicality, and more than half of the task (adjusting, cleaning and repairing equipment) described requires hands or other embodiment.

Occupation: Computer and Information Research Scientists
Task: Apply theoretical expertise and innovation to create or apply new technology, such as adapting principles for applying computers to new uses.
Label (E0/E1/E2/E3): E1
Explanation: The model can learn theoretical expertise during training as part of its general knowledge base, and the principles to adapt can be captured in the text input to the model.

Occupation: Allergists and Immunologists
Task: Prescribe medication such as antihistamines, antibiotics, and nasal, oral, topical, or inhaled glucocorticosteroids.
Label (E0/E1/E2/E3): E2
Explanation: The model can provide guesses for different diagnoses and write prescriptions and case notes. However, it still requires a human in the loop using their judgment and knowledge to make the final decision.

# Instructions
Now complete the following example:
Occupation: {occupation}
Task: {task}
Label (E0/E1/E2/E3):
Answer with the correct label and nothing else.
""".strip()


PROMPT_ZEROSHOT = """
Consider the capabilities of a large language model (LLM) with state-of-the-art capabilities.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half.

## E3 - Exposure given image capabilities
Suppose you had access to both the LLM and a system that could view, caption, and create images. This system cannot take video media as inputs. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and image capabilities.

# Instructions
Now complete the following example:
Occupation: {occupation}
Task: {task}
Label (E0/E1/E2/E3):
Answer with the correct label and nothing else.
""".strip()


PROMPT_RETRIEVAL = """
Consider the capabilities of a large language model (LLM) based on the demonstrations given below.
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have access to the LLM as well as any other existing software or computer hardware tools mentioned in the task. You also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to any other physical tools or materials.
Please label the given task according to the taxonomy below.

# E Exposure Taxonomy
## E0 - No exposure
Label tasks E0 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground cannot reduce the time it takes to complete this task with equivalent quality by half or more.

## E1 - Direct exposure
Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half.

## E2 - Exposure by LLM-powered applications
Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half.

## E3 - Exposure given image capabilities
Suppose you had access to both the LLM and a system that could view, caption, and create images. This system cannot take video media as inputs. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and image capabilities.

## Demonstration of LLM capabilities
{examples_strs}

# Instructions
Now complete the following example:
Occupation: {occupation}
Task: {task}
Label (E0/E1/E2/E3):
Answer with the correct label and nothing else.
""".strip()

BETA_SCORE_MAP_E = {
    "E0": 0.0,
    "E1": 1.0,
    "E2": 0.5,
    "E3": 0.5,
}


def parse_exposure_label(output: str | None) -> str | None:
    if output is None:
        return None

    output_str = output.strip().upper()
    answer_line = output_str.split("\n")[-1]
    match = re.search(r"\bE[0-3]\b", answer_line)

    return match.group(0) if match else None


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


def make_examples_str(
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


PROMPT_ORIG_RET = PROMPT_RETRIEVAL
PROMPT_ORIG_ZS = PROMPT_ZEROSHOT
PROMPT_ORIG_ZSORIG = PROMPT_ORIG

ORIG_BETA_MAP = BETA_SCORE_MAP_E

parse_orig_label = parse_exposure_label
make_orig_examples_str = make_examples_str
