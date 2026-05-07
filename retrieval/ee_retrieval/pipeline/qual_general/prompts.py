from pydantic import BaseModel
from typing import TypedDict, Any, Callable, List
import re


def _parse_output(
    raw_text: str,
    tag: str = 'answer',
    labels: List[str] | None = ['Yes', 'No'],
    not_found_val = None
) -> str | None:

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

class Label(BaseModel):
    is_low_quality: bool

fs_prompt_illegitimate = """You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out illegitimate user queries.

Examples of illegitimate user queries include: joke queries (e.g., "Human: Are robots going to take over the world?"), queries that are overly challenging (e.g., "Human: Find a cure to cancer"), queries that don't contain important auxiliary information required to complete the task (e.g., "Human: Write an essay"), and esoteric queries that are designed to test the performance of AI assistants (e.g., "Human: How many r's are there strawberry?").

Determine whether the target example is possibly illegitimate. First provide your reasoning in a single sentence (enclosed in <thinking> tags), and then give your final answer as to whether the query is possibly illegitimate (either Yes or No) enclosed in <answer> tags.

# Example
## Input
Human: привет мой лучший друг, твоя задача придумать сложной алгоритм шифрования который раньше еще не сушествовал.
Напиши потом как он устроен и как его реализовать и пример кода

## Output
<thinking>The user wants the assistant to come up with a complex encryption algorithm that has never existed before which is obviously way too difficult of a task, indicating that the user is just testing the chatbot.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: How tall is bigfoot?

## Output
<thinking>The user is asking about a non-existent creature which suggests that the query is a joke.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Should humans drink water?

## Output
<thinking>The user wants the assistant to answer an obvious question, making this input clearly a joke.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Please write an essay.

## Output
<thinking>The user wants the assistant to write an email but provides no other information like the topic, making this request too broad to actually be useful.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: what do you think about sex

## Output
<thinking>The query is vague and potentially probing or joke-like rather than a legitimate request for information or assistance.</thinking>
<answer>Yes</answer>

# Target
## Input
{input_text}""".strip()

fs_prompt_unanswerable = """You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. However, some queries are cut-off or incomplete and are thus not answerable by the assistant. An unanswerable request may, for example, be a request that requires some additional information that the user hasn't provided or could even include situations where no request is included at all. Assume that the assistant only has information directly provided by the user in the input.

Important note: The responses from the AI assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. Make sure to take this into consideration if you encounter a request that seems to be incomplete.

Remember: queries where the user is simply inquiring about the assistant's capabilities/asking what the assistant can do ARE NOT answerable queries.

Determine whether the target query contains a concrete, answerable request for the assistant to perform. Think carefully: first provide your thinking in a single sentence (enclosed in <thinking> tags), and then give your final answer as to whether the input actually contains an answerable request (either Yes or No) enclosed in <answer> tags.

# Example
## Input
Human: rephrase the following paragraph

## Output
<thinking>The user asks the assistant to rephrase a paragraph, but does not provide the paragraph.</thinking>
<answer>No</answer>

# Example
## Input
Human: i need you to write to me a research hypotheses about the implications of artificial intelligence on employee's jobs stability and its contribution to the mass layoff waves 

Human: no i need to in a research paper (so i need the format of research paper )

Human: can you write each points of the above

## Output
<thinking>The user asks the assistant to write about certain points, which aren't included, but this is because they are referring to an assistant response which was omitted.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Am I tall enough to play professional basketball?

## Output
<thinking>The user asks a question that requires their height, which is not present in the input.</thinking>
<answer>No</answer>

# Example
## Input
Human: When was NAME_1 born?

## Output
<thinking>The user asks about a birth date, but the subject in question seems to be redacted.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you write Python?

## Output
<thinking>The user asks about the assistant's ability to code but does not actually provide a concrete request for a specific coding task.</thinking>
<answer>No</answer>

# Example
## Input
Human: You are an expert mathematics teacher with 20 years of experience. When I give you a topic, follow these steps:

1. **Introduction:** Greet me and state that your goal is to assess my understanding of the topic.
2. **Assessment:** Ask 3-4 open-ended questions to uncover my prior knowledge and any misconceptions.
3. **Explanation:** Summarize my responses, then deliver a concise, tailored walkthrough—definitions, key principles, examples, and common pitfalls.
4. **Adaptive Quiz:** Pose 5-7 questions, adjusting difficulty after each answer and providing immediate feedback.
5. **Wrap-up:** Recap what I've learned and suggest resources or exercises for further practice.

If you understand, respond with "I understand."

## Output
<thinking>The user asks the assistant to teach them mathematical topics, but does not actually provide a topic, except for a placeholder "I understand" response.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you please be my supply chain oracle and help me with on my queries ??

## Output
<thinking>The input contains a question, but it's more of a directive than a request for the assistant to perform.</thinking>
<answer>No</answer>

# Target
## Input
{input_text}""".strip()

fs_prompt_unanswerable_2 = """You are an expert machine learning scientist curating a high quality dataset of user queries to LLM AI assistants. However, some queries are cut-off, incomplete, or otherwise not answerable by the assistant. An unanswerable request may, for example, be a request that requires some additional information that the user hasn't provided, require capabilities that a text-based LLM does not have, or could even include situations where no request is included at all. Assume that the assistant (LLM) only has information directly provided by the user in the input.

Important note: The responses from the LLM assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. Make sure to take this into consideration if you encounter a request that seems to be incomplete.

Remember: queries where the user is simply inquiring about the assistant's capabilities/asking what the assistant can do ARE NOT answerable queries.

Determine whether the target query contains a concrete, answerable request for the assistant to perform. Think carefully: first provide your thinking in a single sentence (enclosed in <thinking> tags), and then give your final answer as to whether the input actually contains an answerable request (either Yes or No) enclosed in <answer> tags.

# Example
## Input
Human: rephrase the following paragraph

## Output
<thinking>The user asks the assistant to rephrase a paragraph, but does not provide the paragraph.</thinking>
<answer>No</answer>

# Example
## Input
Human: i need you to write to me a research hypotheses about the implications of artificial intelligence on employee's jobs stability and its contribution to the mass layoff waves 

Human: no i need to in a research paper (so i need the format of research paper )

Human: can you write each points of the above

## Output
<thinking>The user asks the assistant to write about certain points, which aren't included, but this is because they are referring to an assistant response which was omitted.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Am I tall enough to play professional basketball?

## Output
<thinking>The user asks a question that requires their height, which is not present in the input.</thinking>
<answer>No</answer>

# Example
## Input
Human: When was NAME_1 born?

## Output
<thinking>The user asks about a birth date, but the subject in question seems to be redacted.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you buy me some shoes that are size 14, green, and good for running? Make sure they come from a reputable brand. Also, my budget is $120.

## Output
<thinking>The user asks the assistant to perform a purchase which a text-based LLM alone can not perform.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you write Python?

## Output
<thinking>The user asks about the assistant's ability to code but does not actually provide a concrete request for a specific coding task.</thinking>
<answer>No</answer>

# Example
## Input
Human: You are an expert mathematics teacher with 20 years of experience. When I give you a topic, follow these steps:

1. **Introduction:** Greet me and state that your goal is to assess my understanding of the topic.
2. **Assessment:** Ask 3-4 open-ended questions to uncover my prior knowledge and any misconceptions.
3. **Explanation:** Summarize my responses, then deliver a concise, tailored walkthrough—definitions, key principles, examples, and common pitfalls.
4. **Adaptive Quiz:** Pose 5-7 questions, adjusting difficulty after each answer and providing immediate feedback.
5. **Wrap-up:** Recap what I've learned and suggest resources or exercises for further practice.

If you understand, respond with "I understand."

## Output
<thinking>The user asks the assistant to teach them mathematical topics, but does not actually provide a topic, except for a placeholder "I understand" response.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you please be my supply chain oracle and help me with on my queries ??

## Output
<thinking>The input contains a question, but it's more of a directive than a request for the assistant to perform.</thinking>
<answer>No</answer>

# Target
## Input
{input_text}""".strip()

fs_prompt_missing_artifact = """You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. You are trying to filter out low-quality queries from your dataset. Specifically, many requests ask the assistant to answer based on a piece of text. For example, for the task of translation the user needs to provide the text they want translated, for the task of summarization the user needs to provide the text they want summarized, etc. However, many queries end up not including the piece of text needed for the task.

You need to determine whether the given user query (1) involves a task where a piece of natural language text *provided in the input* is required to carry out the task (e.g., summarization, paraphrasing, translating, rating, etc.) and (2) is missing that piece of text (ignore this if the query does not involve a required text input).

In your answer, first provide your reasoning in three sentences (enclosed in <thinking> tags):
- The first sentence of your reasoning should determine whether or not the request involves using a piece of natural language text provided in the input. Start the first sentence of your reasoning with "The user wants the assistant to..." The second sentence should determine whether or not that text is missing (ignore this if the query does not involve a required text input). Also, if the query involves coding you should answer No.

- The responses from the AI assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. In the third sentence of your reasoning, provide an additional sentence of reasoning that checks whether the user is referring to a *previous* assistant response (if that's the case, then there is no missing text). Importantly, referring to *future* assistant responses does not count here.

After that, provide your answer for whether the query has a missing piece of text (answer either Yes or No, enclosed in <answer> tags). If the request does not involve a required text input then answer No.

# Example
## Input
Human: rephrase the following paragraph

## Output
<thinking>The user wants the assistant to rephrase a piece of text, which clearly is asking for a piece of natural language text as input. There is no paragraph provided, so the text to be rephrased is missing. Finally, there is only one turn, so it is not possible for the user to be referring to an assistant's response.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: write a paragraph about the history of the automobile

Human: rephrase that paragraph

## Output
<thinking>The user wants the assistant to rephrase a piece of text, which clearly is asking for a piece of natural language text as input. There is no paragraph provided, so the text to be rephrased is missing. However, in that request the user is referring to the response of a previous request so there is no missing text.</thinking>
<answer>No</answer>

# Example
## Input
Human: Puedes hablar en español?

Human: Puedes traducir textos del inglés al español?

Human: Te proporcionaré textos en inglés y lo traducirás al español

## Output
<thinking>The user wants the assistant to help with translation, which clearly is asking for a piece of natural language text as input (the text in the source language). There is no follow up queries with the text to be translated, so the text to be translated is missing. Finally, none of the user requests are referring to a previous assistant response.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Write a short blog post about the impacts of artificial intelligence on programmers. make it at least 400 words.

Human: less formal

## Output
<thinking>The user wants the assistant to adjust the formality of a piece of text, which is a style transformation task requiring an existing text input. There is no text input provided. However, they are clearly referring to the assistant's prior response.</thinking>
<answer>No</answer>

# Example
## Input
Human: When I send you my essay, please:

1. Confirm you're ready to receive it.
2. Assess it on a 1-10 scale for each of these dimensions:
   * Organization & Structure (logical flow, clear introduction and conclusion)
   * Argument & Content (clarity of thesis, depth of analysis, relevance of evidence)
   * Style & Tone (appropriateness for the intended audience, consistency)
   * Language & Mechanics (grammar, punctuation, word choice)
   * Originality & Engagement (fresh ideas, reader interest)

3. Provide:
   * A separate score (1-10) for each dimension above
   * An overall score (1-10)
   * A brief summary of the essay's main strengths
   * Two to three concrete suggestions for improvement in each area where it scored 7 or below
   * Examples or short excerpts (if helpful) to illustrate your feedback

Let me know when you're ready and I'll provide the essay for you to grade.

## Output
<thinking>The user wants the assistant to assess a piece of text (their essay) on multiple dimensions, which clearly is asking for a piece of natural language text as input. There is no essay provided, so the text to be assessed is missing. Finally, there is no previous assistant response that the user could be referring to.</thinking>
<answer>Yes</answer>

# Target
## Input
{input_text}""".strip()

fs_prompt_test = """You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out "evaluation" user queries.

"Evaluation" user queries are queries are asked to test the performance of a model rather than as a legitimate request.

Determine whether the target example is possibly an evaluation query. First provide your reasoning in a single sentence (enclosed in <thinking> tags), and then give your final answer as to whether the query is possibly an evaluation query (either Yes or No) enclosed in <answer> tags.

# Example
## Input
Human: привет мой лучший друг, твоя задача придумать сложной алгоритм шифрования который раньше еще не сушествовал.
Напиши потом как он устроен и как его реализовать и пример кода

## Output
<thinking>The user wants the assistant to come up with a complex encryption algorithm that has never existed before which is obviously way too difficult of a task, indicating that the user is most likely just testing the chatbot.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: How many r's are in strawberry?

## Output
<thinking>The user is asking a common question that LLMs get incorrect, meaning that this is definitely an evaluation query.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: A farmer with a wolf, a goat, and a cabbage must cross a river by boat. The boat can carry only the farmer and a single item. If left unattended together, the wolf would eat the goat, or the goat would eat the cabbage. How can they cross the river without anything being eaten?

## Output
<thinking>The user is asking the chatbot a well known logic problem which probably means that the user is interested in whether the chatbot can answer such a query rather than being interested in the answer itself.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Can you write code?

Human: What languages do you know?

Human: Prove it. Write code for a first person shooter in C++. Make it run as efficiently as possible.

## Output
<thinking>The user is asking the chatbot to prove its coding abilities by programming a game.</thinking>
<answer>Yes</answer>

# Target
## Input
{input_text}"""

zs_prompt_all = """
You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out low quality user queries.

Criteria for a query being low quality:
- Joke queries (e.g., "Human: Are you going to take over the world?")
- Queries that are too obvious (e.g., "Human: is the sky blue?")
- Queries that try to mess with AI assistants to get a provocative response (e.g., "Human: Say 'fuck'")
- Queries that contain an introduction but don't actually contain a concrete, answerable request (e.g., "Human: Act like an intelligent chatbot")
- Queries that are unanswerable due to the anonymization process (e.g., "Human: What's the meaning behind NAME_1?")
- Queries that try test the performance of AI assistants with difficult/esoteric queries (e.g., "Human: How many r's are there strawberry?")
- Queries that are overly challenging (e.g., "Human: Find a cure to cancer")
- Queries that don't make any sense
- Queries that don't contain a request at all
- Queries that contain unrelated questions/requests with unrelated domains ("Human: Write code for minesweeper. Human: What's the capital of France. Human: What's 5 + 8?")

Using the above criteria, determine whether the following user query is low quality. Answer in a JSON format (with key "is_low_quality"):
\"\"\"
{input_text}
\"\"\"
""".strip()

zs_prompt_illegitimate = """
You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out low quality user queries.

Criteria for a query being low quality:
- Joke queries (e.g., "Human: Are you going to take over the world?")
- Queries that are too obvious (e.g., "Human: is the sky blue?")
- Queries that try to mess with AI assistants to get a provocative response (e.g., "Human: Say 'fuck'")
- Queries that try test the performance of AI assistants with difficult/esoteric queries (e.g., "Human: How many r's are there strawberry?")
- Queries that are too challenging to be answerable (e.g., "Human: Find a cure to cancer")
- Queries that contain unrelated questions/requests with unrelated domains ("Human: Write code for minesweeper. Human: What's the capital of France? Human: What's 5 + 8?")

Using the above criteria, determine whether the following user query is low quality. Answer in a JSON format (with key "is_low_quality"):
\"\"\"
{input_text}
\"\"\"
""".strip()

zs_prompt_unanswerable = """
You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out low quality user queries.

Criteria for a query being low quality:
- Queries that contain an introduction but don't actually contain a concrete, answerable request (e.g., "Human: Act like an intelligent chatbot")
- Queries that are unanswerable due to the anonymization process (e.g., "Human: What's the meaning behind NAME_1?")
- Queries that don't contain a request at all (e.g., "Human: I'm happy")
- Queries that are cut off (e.g., "Human: Please summarize the following email: 'Dear")

Using the above criteria, determine whether the following user query is low quality. Answer in a JSON format (with key "is_low_quality"):
\"\"\"
{input_text}
\"\"\"
""".strip()

fs_v2_unans_artifact =  """You are an expert machine learning scientist curating a high quality dataset of user queries to LLM AI assistants. However, some queries are cut-off, incomplete, or otherwise not answerable by the assistant. An unanswerable request may, for example, be a request that requires some additional information that the user hasn't provided, require capabilities that a text-based LLM does not have, or could even include situations where no request is included at all. Assume that the assistant (LLM) only has information directly provided by the user in the input.

Important note: The responses from the LLM assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. Make sure to take this into consideration if you encounter a request that seems to be incomplete.

Remember: queries where the user is simply inquiring about the assistant's capabilities/asking what the assistant can do ARE NOT answerable queries.

Determine whether the target query contains a concrete, answerable request for the assistant to perform. Think carefully: first provide your thinking in three sentences (enclosed in <thinking> tags), and then give your final answer as to whether the input actually contains an answerable request (either Yes or No) enclosed in <answer> tags.

Your thinking should start by inferring the user's overall motivation (start the first sentence with "The user is talking to an AI assistant because they..."). You should then follow by listing the information that you need to answer the request (start this sentence with "The information that you would need to answer this request is").

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: i need you to write to me a research hypotheses about the implications of artificial intelligence on employee's jobs stability and its contribution to the mass layoff waves 

Human: no i need to in a research paper (so i need the format of research paper )

Human: can you write each points of the above

## Output
<thinking>The user is talking to an AI assistant because they want help formulating research hypotheses and structuring a research paper on AI's impact on job stability and mass layoffs. The information that you would need to answer this request is knowledge of AI's effects on employment, conventions for drafting research hypotheses and paper formats, and the specific points referenced in the earlier turns. Although those points aren't shown, an LLM can reasonably infer typical hypotheses and academic structure from its training data and context.</thinking>
<answer>Yes</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: Can you write Python?

## Output
<thinking>The user is talking to an AI assistant because they want help with Python. The information that you would need to answer this request is nothing because there is no request at all. The user asks about the assistant's ability to code in Python but does not actually provide a concrete request for a specific coding task.</thinking>
<answer>No</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: Tell me about the book "To Kill a Mockingbird".

Human: What are the main characters?

## Output
<thinking>The user is talking to an AI assistant because they want an overview of "To Kill a Mockingbird" and details about its main characters. The information that you would need to answer this request is knowledge of the novel's plot, setting, and character list. Although the text itself isn't provided, the work is widely known, so a language model can reasonably summarize its story and principal characters.</thinking>
<answer>Yes</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: What to do when check engine light is on

Human: Can you drive on highways when it's on?

Human: Can you tell me the best driving route to get from 100 West Street to

## Output
<thinking>The user is talking to an AI assistant because they want detailed navigation instructions from one address to another. The information that you would need to answer this request is the complete destination address and any routing preferences such as fastest route or avoiding tolls. Because the destination field is cut off, it's impossible to provide an accurate route.</thinking>
<answer>No</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: hi I got some test results. based on my chart please diagnose me

52 years 
Male 
178 cm 
85 kg 
Hypertension, Type 2 diabetes mellitus, 20 pack-year smoking history 
Metformin 500 mg BID, Lisinopril 10 mg QD 

 - Troponin I: 1.8 ng/mL (normal < 0.04 ng/mL) 
 - CK-MB: 25 U/L (normal 0-5 U/L) 
 - CBC and electrolytes within normal limits 

 - ST-segment elevation in leads V2-V5 

## Ouptut
<thinking>The user is talking to an AI assistant because they want a medical diagnosis based on supplied clinical data. The information that you would need to answer this request would be medical information about the patient. The user has provided this information. Additionally, the request is very clear. Providing a diganosis overall is very complete, but the assistant can still provide useful initial impressions coupled with a recommendation to see a doctor.</thinking>
<answer>Yes</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: summarize the text in the second pdf that I gave to you earlier (2.pdf)

## Output
<thinking>The user is talking to an AI assistant because they want a summary of a specific PDF document. The information that you would need to answer this request is the full contents of "2.pdf". Without access to the document's text, the assistant cannot determine its content. Because only the filename is provided and not the file itself, the request is incomplete and unanswerable as is.</thinking>
<answer>No</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: write an email wishing my friend happy birthday

## Output
<thinking>The user is talking to an AI assistant because they want help composing a birthday greeting email to send to a friend. The information that you would need to answer this request is the friend's name, the desired tone or style, and any personal anecdotes—though none are strictly required to craft a generic, heartfelt message. Drafting a birthday email is a well-defined task and a reasonable/useful email can be created with just the context provided, so this request is answerable.</thinking>
<answer>Yes</answer>

# Example
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
Human: You are an expert mathematics teacher with 20 years of experience. When I give you a topic, follow these steps:

1. **Introduction:** Greet me and state that your goal is to assess my understanding of the topic.
2. **Assessment:** Ask 3-4 open-ended questions to uncover my prior knowledge and any misconceptions.
3. **Explanation:** Summarize my responses, then deliver a concise, tailored walkthrough—definitions, key principles, examples, and common pitfalls.
4. **Adaptive Quiz:** Pose 5-7 questions, adjusting difficulty after each answer and providing immediate feedback.
5. **Wrap-up:** Recap what I've learned and suggest resources or exercises for further practice.

If you understand, respond with "I understand."

## Output
<thinking>The user is talking to an AI assistant because they want a structured lesson plan and interactive quiz on a mathematics topic. The information that you would need to answer this request is nothing, because the only concrete given in this turn is replying "I understand". The user has provided detailed instructional steps but hasn't given a topic or substantive content to teach—only a placeholder acknowledgment.</thinking>
<answer>No</answer>

# Target
## Task
Does the target query contains a concrete, answerable request for the assistant to perform with all the information required to answer it?

## Input
{input_text}

## Output""".strip()

fs_v2_illegit = """You are an expert machine learning scientist curating a high quality dataset of user queries to AI assistants. To do this, you need to filter out illegitimate user queries. Illegitimate user queries are queries that don't correspond to real requests for information.

Examples of illegitimate user queries include: joke queries (e.g., "Human: Are robots going to take over the world?"), queries that are overly challenging (e.g., "Human: Find a cure to cancer"), queries that don't contain important auxiliary information required to complete the task (e.g., "Human: Write an essay"), and esoteric queries that are designed to test the performance of AI assistants (e.g., "Human: How many r's are there strawberry?").

Important note: The responses from the LLM assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. Make sure to take this into consideration if you encounter a request that seems to be incomplete.

Determine whether ALL of the user requests are illegitimate. First provide your reasoning (enclosed in <thinking> tags), and then give your final answer as to whether all of the requests are illegitimate (either Yes or No) enclosed in <answer> tags.

# Example
## Input
Human: привет мой лучший друг, твоя задача придумать сложной алгоритм шифрования который раньше еще не сушествовал.
Напиши потом как он устроен и как его реализовать и пример кода

## Output
<thinking>The user wants the assistant to come up with a complex encryption algorithm that has never existed before which is obviously way too difficult of a task, indicating that the user is just testing the chatbot. This is the only request, so all of the requests are illegitimate.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: How tall is bigfoot?

## Output
<thinking>The user is asking about a non-existent creature which suggests that the query is a joke. This is the only request, so all of the requests are illegitimate.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: tell me abot the history of the industry revolution

Human: continue

Human: now summarize

## Output
<thinking>The user is asking the assistant to discuss the industrial revolution, to continue its response, and to summarize that discussion, all of which are real requests for information.</thinking>
<answer>No</answer>

# Example
## Input
Human: Should humans drink water?

## Output
<thinking>The user wants the assistant to answer an obvious question, making this input clearly a joke. This is the only request, so all of the requests are illegitimate.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: what do you think about sex

## Output
<thinking>The query is vague and potentially probing or joke-like rather than a legitimate request for information or assistance. This is the only request, so all of the requests are illegitimate.</thinking>
<answer>Yes</answer>

# Target
## Input
{input_text}

## Output""".strip()

fs_v2_unans_cap = """You are an expert machine learning scientist curating a high quality dataset of user queries to LLM AI assistants. However, some queries are cut-off, incomplete, or otherwise not answerable by the assistant. An unanswerable request may, for example, be a request that requires some additional information that the user hasn't provided, require capabilities that a text-based LLM does not have (e.g., physical embodiment, vision capabilities), or could even include situations where no request is included at all.

Important note: The responses from the LLM assistants are intentionally omitted. This means that you'll need to infer based on the conversation history whether the user may be referring to the omitted assistant's response. Make sure to take this into consideration if you encounter a request that seems to be incomplete.

Determine whether the target query contains a request that a text-based LLM could hypothetically perform (even if it would be difficult). Think carefully: first provide your thinking in three sentences (enclosed in <thinking> tags), and then give your final answer as to whether the input actually contains an answerable request (either Yes or No) enclosed in <answer> tags.

Your thinking should start by inferring the user's overall motivation (start the first sentence with "The user is talking to an AI assistant because they..."). You should then follow by listing the capabilities required to complete the requests.

# Example
## Input
Human: i need you to write to me a research hypotheses about the implications of artificial intelligence on employee's jobs stability and its contribution to the mass layoff waves 

Human: no i need to in a research paper (so i need the format of research paper )

Human: can you write each points of the above

## Output
<thinking>The user is talking to an AI assistant because they want the assistant to generate specific sections or points for a research paper based on a previous prompt. The capabilities and information required to complete the request include knowledge of the omitted assistant response (the list of hypotheses and paper format details) and the ability to format content according to research paper conventions. Therefore, this is a concrete, answerable request that a text-based LLM could hypothetically fulfill given the missing details.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Am I tall enough to play professional basketball?

## Output
<thinking>The user is talking to an AI assistant because they want to know if their height qualifies them for a professional basketball career. The capabilities and information required to complete the request include the user's actual height measurement and reference data on minimum or average height standards for professional basketball players. Therefore, the request is unanswerable as stated because the user's height is not provided.</thinking>
<answer>No</answer>

# Example
## Input
Human: When was NAME_1 born?

## Output
<thinking>The user is talking to an AI assistant because they want biographical information, specifically the birth date of a person. The capabilities and information required to complete the request include the actual identity or full name of the person referred to as NAME_1 and access to a reliable biographical source. Therefore, the request is unanswerable because the subject's identity is redacted.</thinking>
<answer>No</answer>

# Example
## Input
Human: Can you buy me some shoes that are size 14, green, and good for running? Make sure they come from a reputable brand. Also, my budget is $120.

## Output
<thinking>The user is talking to an AI assistant because they want the assistant to purchase shoes on their behalf that meet certain criteria. The capabilities and information required to complete the request include e-commerce transaction capabilities, access to inventory and pricing data, and the user's payment details. Therefore, this request is unanswerable by a text-based LLM, which cannot perform purchases.</thinking>
<answer>No</answer>

# Example
## Input
Human: Analyze the chord progression of "Superstition" by Stevie Wonder.

Human: What kind of sources did you use for such an analysis?

Human: Actually, did you listen to the original recording or reference published sheet music?

## Output
<thinking>The user is talking to an AI assistant because they want to understand the basis of the harmonic analysis. The capabilities and information required to complete the request include knowing whether the assistant can process audio recordings or read notated scores, and what training data or reference materials it relied upon. Therefore, this is a concrete, answerable request since a text-based LLM would explain that it didn't literally listen or read scores but drew on patterns in its training corpus.</thinking>
<answer>Yes</answer>

# Example
## Input
Human: Can you please be my supply chain oracle and help me with on my queries ??

## Output
<thinking>The user is talking to an AI assistant because they want ongoing assistance with supply chain questions. The capabilities and information required to complete the request include the specific supply chain queries or topics they wish to explore and domain expertise in supply chain management. Therefore, the request is unanswerable as stated because no concrete queries are provided.</thinking>
<answer>No</answer>

# Target
## Input
{input_text}

## Output""".strip()

class QualPromptConfig(TypedDict):
    """Configuration for user request quality classification prompt & output parsing.
    """
    name: str
    prompt: str
    model_args: dict[str, Any]
    is_struct: bool
    pred_extract: Callable[[Any], bool]

_label_extract = lambda x: x.is_low_quality
yes_to_bool = {'Yes': True, 'No': False, None: True}
yes_to_bool_opp = {'Yes': False, 'No': True, None: False}

_tag_extract = lambda x: yes_to_bool[_parse_output(x)]
_tag_extract_opp = lambda x: yes_to_bool_opp[_parse_output(x)]

QUAL_GENERAL_PROMPTS = [
    # Prompt(name='zs_all', prompt=zs_prompt_all, model_args={'model_name': 'gpt-4.1-mini', 'response_format': Label, 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.0}, is_struct=True, pred_extract=label_extract),
    # Prompt(name='zs_illegitimate', prompt=zs_prompt_illegitimate, model_args={'model_name': 'gpt-4.1-mini', 'response_format': Label, 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.0}, is_struct=True, pred_extract=label_extract),
    QualPromptConfig(name='zs_unanswerable', prompt=zs_prompt_unanswerable, model_args={'model_name': 'gpt-4.1-mini', 'response_format': Label, 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.0}, is_struct=True, pred_extract=_label_extract),
    QualPromptConfig(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'model_name': 'gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
    QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable, model_args={'model_name': 'gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
    QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'model_name': 'gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
    QualPromptConfig(name='fs_test', prompt=fs_prompt_test, model_args={'model_name': 'gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
]

PROMPT_MAP: dict[str, list[QualPromptConfig]] = {
    'v1': [
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
    ],
    'v2': [
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
    ],
    'v3': [
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract)
    ],
    'v4': [
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
    ],
    'v4_mini': [
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        QualPromptConfig(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'full_model_name': 'openai/gpt-4.1-mini', 'max_tokens': 4096, 'num_workers': 128, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract),
        QualPromptConfig(name='fs_unanswerable', prompt=fs_prompt_unanswerable_2, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=_tag_extract_opp),
        # Prompt(name='fs_missing_artifact', prompt=fs_prompt_missing_artifact, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=tag_extract),
        # Prompt(name='fs_illegitimate', prompt=fs_prompt_illegitimate, model_args={'full_model_name': 'google/gemini-2.0-flash', 'max_tokens': 4096, 'num_workers': 32, 'temperature': 0.5}, is_struct=False, pred_extract=tag_extract),
    ],
    'v5': [
         QualPromptConfig(
            name='fs_v2_unans_artifact',
            prompt=fs_v2_unans_artifact,
            model_args={
                'full_model_name': 'openai/gpt-4.1-mini',
                'max_tokens': 4096,
                'num_workers': 128,
                'temperature': 0.5
            },
            is_struct=False,
            pred_extract=_tag_extract_opp
        ),
         QualPromptConfig(
            name='fs_v2_unans_cap',
            prompt=fs_v2_unans_cap,
            model_args={
                'full_model_name': 'openai/gpt-4.1-mini',
                'max_tokens': 4096,
                'num_workers': 128,
                'temperature': 0.5
            },
            is_struct=False,
            pred_extract=_tag_extract_opp
        ),
         QualPromptConfig(
            name='fs_v2_illegit',
            prompt=fs_v2_illegit,
            model_args={
                'full_model_name': 'openai/gpt-4.1-mini',
                'max_tokens': 4096,
                'num_workers': 128,
                'temperature': 0.5
            },
            is_struct=False,
            pred_extract=_tag_extract
        ),
    ]
}
