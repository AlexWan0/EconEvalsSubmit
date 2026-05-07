from fast_openai import run_openai_str, RequestArgs, run_openai_embed
import numpy as np
from typing import Literal
import re
from diskcache import Cache
import logging
logger = logging.getLogger(__name__)


TASK_PROMPT = """What might workers in the following occupation performing the following task ask ChatGPT to help with?

{category}

Answer only with an example of a request for assistance with the above specified task directly (which includes requests that just ask the assistant to fully complete the task!) and not just *any* related task.

Also, in addition to the request, write out the full conversation the worker would have with ChatGPT. Format your answer as follows (just answer with the conversation, no backticks):
```
Human: the request for assistance

Assistant: assistant's response

Human: ...

Assistant: ...

etc.
```
""".strip()

TASK_LIST_PROMPT = """
You are an expert economist studying the economic impacts of large language models. Specifically, you are looking to study how workers currently use LLMs. What might workers in the following occupation performing the following task ask ChatGPT to do?

{category}

Answer with {n_trials} realistic examples of requests for assistance with the above specified task. These examples may include requests which just ask the assistant to complete the above task in full. Each example must be two sentences long. Make sure the occupation of the worker is clear in each of the examples.

Format your answer as follows:
```
<think>Use this space to plan out what examples to include.</think>
<item><n>1</n><answer>first example</answer></item>
<item><n>2</n><answer>second example</answer></item>
...
```
""".strip()

CONDITIONAL_TASK_PROMPT = """
Generate a realistic conversation between a user and ChatGPT where the user asks ChatGPT to perform the following task:
"{task}"

Format your answer as follows (just answer with the conversation, no backticks):
```
Human: the request for assistance

Assistant: assistant's response

Human: ...

Assistant: ...

etc.
```
""".strip()

vec_cache = Cache(directory='.cache/vec_cache')

def category_to_string(
        cat_data: dict[str, str],
        mode: Literal['headings', 'concat'] = 'headings'
    ) -> str:
    
    if mode == 'headings':
        return f'Task: {cat_data["task"]}\nOccupation: {cat_data["occupation"]}'
    
    elif mode == 'concat':
        return f'{cat_data["task"]}\n{cat_data["occupation"]}'
    
    raise ValueError(f'invalid mode: {mode}')

def extract_answers(xml_fragment: str) -> list[str]:
    contents = re.findall(r"<answer>(.*?)</answer>", xml_fragment, re.DOTALL)
    return [
        c.strip()
        for c in contents
    ]

def _log_to_file(text: str, fp: str = '.cache/embed-logs.txt'):
    with open(fp, 'a') as f_out:
        f_out.write(text + '\n')
        f_out.write('-' * 80 + '\n')

def embed_model_conditional_ensemble(
        cat_data: dict[str, str],
        task_prompt: str,
        convo_prompt: str,
        n_trials: int,
        task_model_name: str = 'gpt-4.1',
        convo_model_name: str = 'gpt-4.1-mini',
        temperature: float = 1.0,
        verbose: bool = False
    ) -> np.ndarray:

    task_model_input = task_prompt.format(
        category=category_to_string(cat_data),
        n_trials=n_trials
    )

    task_model_output = run_openai_str(
        [task_model_input],
        model_name=task_model_name,
        temperature=temperature,
        num_workers=1,
        max_tokens = 8192,
    )[0].output
    assert task_model_output is not None
    _log_to_file(task_model_input)
    _log_to_file(task_model_output)

    task_descrips = extract_answers(task_model_output)

    if verbose:
        logger.info(task_descrips)

    convo_model_inputs = [
        convo_prompt.format(task=t)
        for t in task_descrips
    ]
    
    convo_res = run_openai_str(
        convo_model_inputs,
        model_name=convo_model_name,
        temperature=temperature,
        num_workers=16,
        max_tokens = 8192,
        request_args=RequestArgs(
            post_timeout=5 * 60,
            total_timeout=5 * 60,
            num_retries=4
        )
    )

    convo_model_outputs: list[str] = []
    for inp, x in zip(convo_model_inputs, convo_res, strict=True):
        assert x.output is not None
        convo_model_outputs.append(x.output)

        if verbose:
            logger.debug(x.output)

        _log_to_file(inp)
        _log_to_file(x.output)

    embeds = run_openai_embed(
        convo_model_outputs,
        model_name='text-embedding-3-small',
        num_workers=16
    )
    embeds_vec = np.array([
        x.output
        for x in embeds
    ])

    embeds_avg = embeds_vec.mean(axis=0)

    return embeds_avg

@vec_cache.memoize()
def embed_model_ensemble(
        cat_data: dict[str, str],
        prompt: str,
        n_trials: int,
        model_name: str = 'gpt-4.1-mini',
        temperature: float = 1.0,
        verbose: bool = False
    ) -> np.ndarray:

    model_input = prompt.format(
        category=category_to_string(cat_data)
    )

    model_inputs = [model_input] * n_trials

    res = run_openai_str(
        model_inputs,
        model_name=model_name,
        temperature=temperature,
        num_workers=16,
        request_args=RequestArgs(
            use_cache=False
        ),
        max_tokens = 8192,
    )

    model_outputs: list[str] = []
    for x in res:
        assert x.output is not None
        model_outputs.append(x.output)

        if verbose:
            logger.debug(x.output)
    
    embeds = run_openai_embed(
        model_outputs,
        model_name='text-embedding-3-small',
        num_workers=16
    )
    embeds_vec = np.array([
        x.output
        for x in embeds
    ])

    embeds_avg = embeds_vec.mean(axis=0)
    logger.info(f"embeds_avg.shape: {embeds_avg.shape}")

    return embeds_avg

@vec_cache.memoize()
def embed_concat(
        cat_data: dict[str, str],
    ) -> np.ndarray:

    embeds = run_openai_embed(
        [
            category_to_string(cat_data, mode='concat')
        ],
        model_name='text-embedding-3-small',
        num_workers=1
    )

    return np.array(embeds[0].output)

