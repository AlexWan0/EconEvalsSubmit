import re


def parse_single_tag(output: str | None, tag_name: str) -> str | None:
    if output is None:
        return None

    tag = re.escape(tag_name)
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", output, re.DOTALL)
    if match is None:
        return None

    value = match.group(1).strip()
    return value if value else None


def parse_multiple_tags(output: str | None, tag_name: str) -> list[str] | None:
    if output is None:
        return None

    tag = re.escape(tag_name)
    matches = re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}>", output, re.DOTALL)
    if not matches:
        return None

    values = [m.strip() for m in matches if m.strip()]
    return values or None


def parse_yes_no_item(item: str | None) -> bool | None:
    if item is None:
        return None

    item_norm = item.strip().lower()
    if item_norm == "yes":
        return True
    if item_norm == "no":
        return False
    return None


def parse_yes_no_items(output: str | None, tag_name: str = "item") -> list[bool | None] | None:
    parsed_items = parse_multiple_tags(output, tag_name)
    if parsed_items is None:
        return None
    return [parse_yes_no_item(x) for x in parsed_items]


def build_required_scoring_io(required_items: list[str]) -> tuple[str, str]:
    checklist_input = "\n".join(
        SCORING_PROMPT_ITEM_INPUT.format(id=i, checklist_question=q)
        for i, q in enumerate(required_items, start=1)
    )
    checklist_output = "\n".join(
        SCORING_PROMPT_ITEM_OUTPUT.format(id=i, checklist_question=q)
        for i, q in enumerate(required_items, start=1)
    )
    return checklist_input, checklist_output


CHECKLIST_GEN_PROMPT = """
# Background
I'm designing a checklist consisting of binary items to grade LLMs on. I want this checklist to be grounded on actual failures made by LLMs.

I will give you a prompt that I want a checklist for. I will also give you a rudimentary response from a small LLM. I want you to design checklist items that account for failures of the rudimentary response.

# Prompt to design checklist for
<prompt>
{query}
</prompt>

# Rudimentary LLM response
<llmResponse>
{query_out}
</llmResponse>

# Instructions
Based on the prompt and the rudimentary response, design a checklist that targets failures of the rudimentary response. The rudimentary response should score a 0 (no checklist items should be satisfied).

The checklist should consist of unambiguous, objective, and self-contained binary questions. Each question must start with "Does the response..." and end with a question mark. Because I ONLY want objective checklist items, avoid using fluff words like "comprehensive" or "detailed" as these are vague. Each binary question MUST be completely self-contained. It should be possible to understand the checklist item without looking at the original query.

You may denote each checklist item as "optional" (itemOptional) or "required" (itemRequired). Optional items may be present in a good response but may not be. Required items MUST be present in ALL good responses. You MUST include at least 1 and at most 5 optional checklist items. You MUST include at least 5 and at most 10 required checklist items.

When writing the checklist items you should use the rudimentary response as a reference, but it MUST NOT be specific to that response. I will use the checklist to grade other responses, so it should check for similar failure-modes.

Finally, make sure that your checklist covers different dimensions of the response quality. For example, if you already have a checklist item that checks to make sure a deliverable is included, you MUST NOT have any more checklist items that check for the presence of specific deliverables (you should instead have a single checklist item that checks for the presence of ALL deliverables and use the other checklist items to check for other failure modes).

Format your output as follows (your response must be surrounded in triple backticks as well):
<thinking>
[In three sentences or less, summarize ALL the failures of the rudimentary model's response to the prompt]
[In three sentences or less, organize failures into different mutually exclusive dimensions to make sure that the checklist items cover a diverse set of response attributes.]
</thinking>
<checklist>
    <itemOptional id="1">Does the response [...]?</itemOptional>
    [...]
    <itemOptional id="[1<=n<=5]">Does the response [...]?</itemOptional>

    <itemRequired id="1">Does the response [...]?</itemRequired>
    [...]
    <itemRequired id="[5<=m<=10]">Does the response [...]?</itemRequired>
</checklist>
```
""".strip()


CHECKLIST_GEN_PROMPT_NOMIN = """
# Background
I'm designing a checklist consisting of binary items to grade LLMs on. I want this checklist to be grounded on actual failures made by LLMs.

I will give you a prompt that I want a checklist for. I will also give you a rudimentary response from a small LLM. I want you to design checklist items that account for failures of the rudimentary response.

# Prompt to design checklist for
<prompt>
{query}
</prompt>

# Rudimentary LLM response
<llmResponse>
{query_out}
</llmResponse>

# Instructions
Based on the prompt and the rudimentary response, design a checklist that targets failures of the rudimentary response.

The checklist should consist of unambiguous, objective, and self-contained binary questions. Each question must start with "Does the response..." and end with a question mark. Because I ONLY want objective checklist items, avoid using fluff words like "comprehensive" or "detailed" as these are vague. Each binary question MUST be completely self-contained. It should be possible to understand the checklist item without looking at the original query.

You may denote each checklist item as "optional" (itemOptional) or "required" (itemRequired). Optional items are ones that may be present in a good response but may not be. Required items are ones that are always present in ALL good responses.

Include at MOST ten optional checklist items and at MOST ten required checklist items. If the rudimentary LLM response has no failures, then you MUST NOT include any *required* checklist items, but you MAY still include optional checklist items. The rudimentary response (above) should score a 0 on the required checklist items, but it may have a non-zero score on the optional checklist items. Remember: if there are no failures in the rudimentary response, then there MUST NOT be any required checklist items - required checklist items are meant to score responses on NECESSARY conditions for a good response.

When writing the checklist items you should use the rudimentary response as a reference, but it MUST NOT be specific to that response. I will use the checklist to grade other responses, so it should check for similar failure-modes.

Finally, make sure that your checklist covers different dimensions of the response quality. For example, if you already have a checklist item that checks to make sure a deliverable is included, you MUST NOT have any more checklist items that check for the presence of specific deliverables (you should instead have a single checklist item that checks for the presence of ALL deliverables and use the other checklist items to check for other failure modes).

Format your output as follows (your response must be surrounded in triple backticks as well):
<thinking>
[In three sentences or less, summarize ALL the failures of the rudimentary model's response to the prompt]
[In three sentences or less, organize failures into different mutually exclusive dimensions to make sure that the checklist items cover a diverse set of response attributes.]
</thinking>
<checklist>
    <itemOptional id="1">Does the response [...]?</itemOptional>
    [...]

    <itemRequired id="1">Does the response [...]?</itemRequired>
    [...]
</checklist>
OR (if there are no failures in the rudimentary response)
<checklist>
    <itemOptional id="1">Does the response [...]?</itemOptional>
    [...]
</checklist>
```
""".strip()


SCORING_PROMPT = """
# Background
You are an objective evaluator for LLM responses. You will be provided with a response from a model to evaluate and a checklist for required attributes of the response.

Your goal is to strictly grade the model's response against the provided checklist items.

# Inputs
## Grading Checklist
<checklist>
{checklist_input}
</checklist>

## Model Response to Evaluate
<response>
{query_out}
</response>

# Instructions
1. Read the Model Response carefully.
2. Iterate through every item in the Grading Checklist.
3. For each item, determine if the response satisfies the condition. Answer with either "Yes" or "No" for each item and nothing else.
4. You must be strict. If a required element is missing or incorrect, it must fail.

# Output Format
Provide your evaluation in the following XML format wrapped in triple backticks.
```
<evaluation>
{checklist_output}
</evaluation>
```
""".strip()


SCORING_PROMPT_ITEM_INPUT = """
<item id="{id}">{checklist_question}</item>
""".strip()


SCORING_PROMPT_ITEM_OUTPUT = """
<item id="{id}">[Either "Yes" or "No" for the question: "{checklist_question}"]</item>
""".strip()
