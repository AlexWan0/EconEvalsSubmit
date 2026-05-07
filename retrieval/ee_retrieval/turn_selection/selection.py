import typeguard
import pandas as pd
from fast_openai import run_openai_str, RequestArgs
import re
import logging
logger = logging.getLogger(__name__)

from ..conversations import add_instance_text_col
from ..utils import convo_to_str


TURN_PROMPT = """You are an expert ML scientist curating a high-quality dataset for evaluating AI assistants. You have collected several full conversations between humans and AI assistants and want to extract tasks that can be used for evaluation. The extracted tasks will take the form of turns (requests made by humans) in the conversation: you want to pick the best turn. Importantly, you also have a specific category of tasks that you're looking to collect instances for.

Following the instructions given below, determine the (human) turn to include in the dataset for a target conversation and category.

# Example
## Instructions
Given several turns of a conversation, determine the (human) turn that (1) is a specific, concrete, answerable request and (2) best fits with the target category. Before answering, explain your reasoning in two sentences in <thinking></thinking> tags. Both the human requests and the assistant responses will be enclosed in triple quotes (\"\"\"). Finally, give your answer in <answer></answer> tags (<answer>1, 2, 3, etc.</answer> corresponding to "Human turn 1, 2, 3, etc."). If there are multiple turns that are concrete, answerable requests and fit with the category, then just pick the best one.

## Input
### Conversation
Human turn 1: \"\"\"Hello\"\"\"

Assistant: \"\"\"Hello! How can I help you today? Is there something specific you would like to know or discuss? I'm here to provide information and answer any questions you may have.\"\"\"

Human turn 2: \"\"\"Can you write code?\"\"\"

Assistant: \"\"\"Yes, I can write code in a variety of programming languages, including Python, NAME_1, C++, and JavaScript. However, as a language model, I am not able to execute code or perform programming tasks in the same way that a human programmer can. My primary function is to generate natural language text based on the input I receive. I can provide assistance with understanding programming concepts and syntax, and can help with writing and debugging code, but I am not able to actually execute code. Is there something specific you would like to know or discuss related to programming?\"\"\"

Human turn 3: \"\"\"Write an example script in python to process a CSV file from a local machine, transform it to include new columns with initial default values, and upload the data to S3 to a dedicated bucket location with prefix set to "Transcripts/Analysis"
Do not worry about details such as file name, column names, values, ...etc...
Your example should be clear enough to allow readers to leverage it as an example for future similar use cases.\"\"\"

Assistant: \"\"\"Certainly! Here is an example script in Python that demonstrates how to process a CSV file from a local machine, transform it to include new columns with initial default values, and upload the data to S3 to a dedicated bucket location with prefix set to "Transcripts/Analysis":
```
import boto3
import csv

# Set up Boto3 client for Amazon S3
s3 = boto3.client('s3')
...
```\"\"\"

Human turn 4: \"\"\"continue\"\"\"

Assistant: \"\"\"script prints a message indicating that the data has been processed and uploaded to S3.

I hope this example is helpful! Let me know if you have any questions or if you would like further clarification on any of the steps.\"\"\"

### Target category
Create or modify data-processing scripts for file manipulation. For example, this involves writing or updating code to parse, transform, and combine datasets across formats (e.g., CSV, text, spreadsheets, and database tables), perform operations such as merging or joining files, extracting fields, pivoting or aggregating data, filtering or deduplicating records, and exporting results to specified outputs while optimizing runtime and memory usage. However, this does not include building full ETL/orchestration platforms, designing database schemas or infrastructure, or developing interactive analytics dashboards.

## Output
<thinking>Turns 1 and 2 just contain a greeting and a check for whether the assistant can perform a task, and turn 4 asks the assistant to continue with its answer to the previous request. Only turn 3 contains a concrete request which also fits under the target category, so the answer is 3.</thinking>
<answer>3</answer>

# Example
## Instructions
Given several turns of a conversation, determine the (human) turn that (1) is a specific, concrete, answerable request and (2) best fits with the target category. Before answering, explain your reasoning in two sentences in <thinking></thinking> tags. Both the human requests and the assistant responses will be enclosed in triple quotes (\"\"\"). Finally, give your answer in <answer></answer> tags (<answer>1, 2, 3, etc.</answer> corresponding to "Human turn 1, 2, 3, etc."). If there are multiple turns that are concrete, answerable requests and fit with the category, then just pick the best one.

## Input
### Conversation
Human turn 1: \"\"\"javascript check something before submit form, if something wrong, stop submit and stay in the page. how to do?\"\"\"

Assistant: \"\"\"You can accomplish this by binding a JavaScript function to the form's `onsubmit` event and returning `false` if there is an error. Here's an example:

HTML:
```html
<form id="myForm">
    
    <input value="Submit" type="submit">
</form>
```

JavaScript:
```javascript
function validateForm() {{
    // Check if something is wrong
    if (somethingIsWrong) {{
        // Display an error message or take any other necessary action
        
        // Prevent the form from submitting and staying on the page
        return false;
    }}
    
    // If everything is okay, allow the form to submit
    return true;
}}
```

In this example, replace `somethingIsWrong` with your specific condition for checking if there is an error. If `somethingIsWrong` evaluates to `true`, the function will return `false`, preventing the form from submitting and staying on the page. Otherwise, it will return `true` and allow the form to submit as usual.\"\"\"

Human turn 2: \"\"\"cannot stop if there is any error?\"\"\"

Assistant: \"\"\"If you want to prevent the form from submitting if there is any error, you can use the `event.preventDefault()` method in your JavaScript code. 

Here's an updated example:

HTML:
```html
<form id="myForm">
    
    <input value="Submit" type="submit">
</form>
```

JavaScript:
```javascript
document.getElementById('myForm').addEventListener('submit', function(event) {{
    // Check if something is wrong
    if (somethingIsWrong) {{
        // Display an error message or take any other necessary action
        
        // Prevent the form from submitting and staying on the page
        event.preventDefault();
    }}
}});
```

In this example, you will need to replace `somethingIsWrong` with your specific condition for checking if there is an error. If `somethingIsWrong` evaluates to `true`, the `event.preventDefault()` method will be called, preventing the form from submitting and staying on the page. Otherwise, the form will submit normally.\"\"\"

### Target category
Diagnose and resolve form submission and validation problems in web applications. For example, this involves troubleshooting input handling, client- and server-side checks, and the behavior of components such as checkboxes, dropdowns, buttons, and date pickers, as well as correcting issues in data processing after submission to ensure accurate storage and responses. However, this does not include designing new form workflows, building unrelated features, or conducting general usability research.

## Output
<thinking>Turn 1 is a concrete, answerable request about how to implement form validation to stop submission, but it is more about initial implementation than debugging. Turn 2 clearly describes a problem the user is facing with preventing form submission on error, making it a specific debugging request that fits the target category of fixing form submission and validation issues better.</thinking>
<answer>2</answer>

# Target
## Instructions
Given several turns of a conversation, determine the (human) turn that (1) is a specific, concrete, answerable request and (2) best fits with the target category. Before answering, explain your reasoning in two sentences in <thinking></thinking> tags. Both the human requests and the assistant responses will be enclosed in triple quotes (\"\"\"). Finally, give your answer in <answer></answer> tags (<answer>1, 2, 3, etc.</answer> corresponding to "Human turn 1, 2, 3, etc."). If there are multiple turns that are concrete, answerable requests and fit with the category, then just pick the best one.

## Input
### Conversation
{convo_turns}

### Target category
{category_info}"""


def _make_turn_model_input(row: pd.Series):
    '''
    Expects cols: category_data in each row
    '''
    convo = row['instance_data'].convo
    category_text = row['dwa_detailed']

    typeguard.check_type(convo, list[dict[str, str]])

    turns = []
    h_counter = 1
    
    for t_data in convo:
        text_content = t_data['content']

        if t_data['role'] == 'user':
            turns.append(
                f"Human turn {h_counter}: " + "\"\"\"" + text_content + "\"\"\""
            )
            h_counter += 1
        
        else:
            turns.append(
                "Assistant: " + "\"\"\"" + text_content + "\"\"\""
            )
    
    turns_string = '\n\n'.join(turns)

    return TURN_PROMPT.format(
        convo_turns=turns_string,
        category_info=category_text
    )

def _parse_output_tag(
        raw_text: str | None,
        tag: str = 'answer',
        labels: list[str] | None = ['Yes', 'No'],
        not_found_val: str = '',
    ) -> str:
    
        if raw_text is None:
            return not_found_val

        pattern = rf'<{re.escape(tag)}>(.*?)</{re.escape(tag)}>'
        m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if not m:
            return not_found_val

        if labels is None:
            return m.group(1).strip()

        candidate = m.group(1).strip().lower()
        for label in labels:
            if label.strip().lower() == candidate:
                return label

        return not_found_val

def _has_int_answer(model_output: str) -> bool:
    output = _parse_output_tag(model_output, labels=None, not_found_val='')

    if not output:
        return False
    
    try:
        parsed_int = int(output)
        return parsed_int > 0 # 1 indexed
    
    except ValueError:
        return False
    
def _get_convo_subset(row: pd.Series) -> list[dict[str, str]] | None:
    full_convo: list[dict[str, str]] = row['convo']
    picked_turn: int = row['picked_turn']

    typeguard.check_type(picked_turn, int)
    typeguard.check_type(full_convo, list[dict[str, str]])

    assert picked_turn != -1
    
    turns: list[dict[str, str]] = []
    h_counter = 1

    found_turn = False
    for t_data in full_convo:
        if t_data['role'] == 'user':
            turns.append(t_data)

            if h_counter == picked_turn:
                found_turn = True
                break

            h_counter += 1

        else:
            turns.append(t_data)

    if not found_turn:
        return None

    return turns

def get_turn_subset_selection(
        df: pd.DataFrame,
        dwa_summ_map: dict[str, str],
        dset_ptrs_path: str = 'data/chat_corpora/dataset_pointers.pkl'
    ) -> pd.DataFrame:
    """Given dataframe containing category-conversation pairs (most likely be only ones that are successfully classified by the pipeline), adds a columns for subsets of the original conversation where the last turn corresponds to the human request that best belongs to the category (the DWA).

    Args:
        df (pd.DataFrame): dataframe where each row is a category-conversation pair. Expects columns: 'instance_idx', 'dwa'
        dwa_summ_map (dict[str, str]): Mapping from original DWA Title to detailed DWA based on summarized adjacent task statements (see: ``summarization/tasks.py``)
        dset_ptrs_path (str): path for mapping from conversatoin ids (instance_idx) to datasets (see: conversations/dataset_ranges.py). Defaults to ``data/chat_corpora/dataset_pointers.pkl``
    
    Returns:
        pd.DataFrame: dataframe with additional columns `instance_text_subset`, `convo_subset` where the last turn is the turn picked to be a user request that best fits under the DWA.
    """

    add_instance_text_col(
        df,
        cols=('instance_text', 'instance_text_human_only'),
        dset_ptrs_path=dset_ptrs_path
    )

    df['dwa_detailed'] = df['dwa'].map(dwa_summ_map)
    assert not any(df['dwa_detailed'].isna())

    df['convo'] = df['instance_data'].apply(lambda x: x.convo)

    df['n_human_turns'] = df['convo'].apply(
        lambda full_convo: len([
            t_data
            for t_data in full_convo
            if t_data['role'] == 'user'
        ])
    )
    multiturn_df = df[df['n_human_turns'] > 1].copy()

    multiturn_df['_turns_model_input'] = multiturn_df.apply(_make_turn_model_input, axis=1)

    multiturn_df['_turns_model_output'] = run_openai_str(
        multiturn_df['_turns_model_input'],
        model_name='gpt-4.1-mini',
        max_tokens=2048,
        temperature=0.5,
        cache_flag='trial_1',
        num_workers=128,
        request_args=RequestArgs(hash_keys=True, num_retries=4),
        validation_function=_has_int_answer
    )

    answer_choices = [
        f'{i + 1}'
        for i in range(multiturn_df['n_human_turns'].max() + 1)
    ]
    multiturn_df['picked_turn_str'] = multiturn_df['_turns_model_output'].apply(
        lambda x: _parse_output_tag(
            x.output,
            tag='answer',
            labels=answer_choices,
            not_found_val=''
        )
    )
    multiturn_df['picked_turn'] = multiturn_df['picked_turn_str'].apply(
        lambda x: int(x) if x != '' else -1
    )

    turn_map = multiturn_df['picked_turn']
    assert len(turn_map) == sum(df['n_human_turns'] > 1), (len(turn_map), sum(df['n_human_turns'] > 1))
    def _map_turn(idx: int) -> int:
        if idx in turn_map:
            return turn_map[idx]

        return 1

    df['_full_idx'] = df.index
    df['picked_turn'] = df['_full_idx'].apply(_map_turn)

    # drop invalid turns
    orig_len = len(df)
    df = df[
        df['picked_turn'] != -1
    ]
    n_dropped = orig_len - len(df)

    logger.info(f"number of instances dropped b/c invalid turns: {n_dropped}")

    df['convo_subset'] = df.apply(
        _get_convo_subset, # type: ignore
        axis=1
    )

    for x in df['convo_subset']:
        assert x[-1]['role'] == 'user'

    df['instance_text_subset'] = df['convo_subset'].apply(
        convo_to_str
    )

    df_dedup = df.drop_duplicates(subset=['dwa', 'instance_idx']).copy()

    df_agg = df_dedup.groupby('dwa').agg({
        'dwa_detailed': 'first',
        'instance_idx': list,
        'convo_subset': list,
    }).reset_index()

    df_agg['n_instances'] = df_agg['instance_idx'].apply(len)

    df_agg['category_idx'] = df_agg.index
    
    df_dedup['category_idx'] = df_dedup['dwa'].map(df_agg.set_index('dwa')['category_idx'])

    return df_dedup
