FILTER_PROMPT_LIGHT = """
The following are one or more user requests from a conversation between a user and AI assistant (the assistant's responses are omitted). Each turn starts with the prefix "(index) Human:" and continues until the next "(index + 1) Human:" prefix (if there's only one, then that means there's only one turn). You are also given a description of an occupational task.

List all the indices corresponding to turns that are not related to the occupational task. Also list the indices corresponding to turns that don't contain a concrete request to the AI assistant and/or don't make sense.

Occupational task: {category_task}
User requests:
```
{numbered_instance_text_human_only}
```

First plan out your answer for two sentences in <thinking></thinking> tags. Then, in <answer></answer> tags, list the indices corresponding to turns that are *not* related to the occupational task ({category_task}) OR don't contain a concrete request to the AI assistant and/or don't make sense. Leave that space blank if no indices apply. Format your answer as a comma separated list of indices. Answer exactly with the format:
<thinking>Planning your answer for two sentences.</thinking>
<answer>
Comma separated list of indices. e.g., 1, 5, 6, 7. If no indices apply, just put nothing here.
</answer>
""".strip()

SUMMARIZE_PROMPT = """
The following are one or more user requests from a conversation between a user and AI assistant (the assistant's responses are omitted). Please summarize these requests.

First read the requests very carefully, then write a summary structured in the following way:
- In one or two sentences (depending on the number of requests made), summarize what the user wants the assistant to do. Start this with: "The user wants the assistant to...". Write very literally and make the summary as detailed as possible. You may reference short quotes (a couple words at most) from the original requests if necessary. Ignore any part of the request that does not correspond to an occupational task (e.g., introductions like "what model are you?").
- In one sentence, make an educated inference as to why the user is asking the AI assistant this (e.g., to help with homework, to learn, to write blog posts for their business, etc.). Answer with only the most likely motivation. Start this with: "The user might be asking this because...". Read the requests carefully before answering: it may be important to consider how the user makes the request, how the request is formatted, etc.
- In one sentence, make an educated inference as to the occupation of the person who asked the request and the industry that they work in. If it's not possible to make an inference, then specify that it's unknown. Start this with "The person asking this request probably works as a [occuaption] in the industry of [industry]" OR "The person asking this request is probably a student completing a [program] in [major/specialty/etc.]" OR "It's not clear the occupation of the person who asked this request".

Retain a matter-of-fact, neutral tone for your summary. Make sure to focus on the actual request(s) being asked rather than background information. Answer with nothing but the summary. Do not include anything other than the summary following the above requirements.
    
User request(s):
```
{instance_text_human_only}
```
""".strip()

CLASSIF_PROMPT = """
# Background
You are an economist looking to study how well AI assistants perform on occupational tasks. To do this, you want to build a benchmark. Specifically, you have a dataset of user requests made to AI assistants and you want to find samples that would accurately measure the performance of AI assistants on a target occupational responsibility.

# Instructions
Below is a target responsibility (which describes a category of related individual occupational tasks) and an occupation that you're studying (you should only consider the target responsibility as performed by the target occupation). Below is also a description of a request made to an AI assistant. Determine the amount of overlap between the request and the target responsibility as performed by the target occupation.

Target responsibility: {category_task}
Target occupation: {category_occ}

The Target responsibility may be shared by other occupations, but make sure you only consider the responsibility *as it would be performed by the Target occupation*.

Request made to AI assistant:
{summ_instance_text}

Which of the following statements best describes the request in relation to the Target responsibility as performed by the Target occupation?
(A) Part or all of the request replaces all or pretty much all of a real-world task that falls under the Target, specifically as performed by a worker in the occupation "{category_occ}".
(B) Part or all of the request replaces most (more than 80%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(C) Part or all of the request replaces a significant portion (more than 50%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(D) Part or all of the request replaces a non-negligible amount (more than 10%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(E) Part or all of the request replaces 10% or less of a real-world task that falls under the Target responsibilityspecifically as performed by a worker in the occupation "{category_occ}".
(F) The entire request is completely unrelated to the Target responsibility  as performed by a worker in the occupation "{category_occ}".

When answering this question, only consider the part of the request most relevant to the Target responsibility. You may ignore irrelevant parts of the request.

First, in at most four sentences carefully reason about the question in <thinking></thinking> tags. Then, put your answer in <answer></answer> tags (only put the letter here and nothing else). Make sure you follow this format exactly:
<thinking>Carefully reason about the question in at most four sentences. Follow this template exactly: "The real-world task most relevant to the request that falls under [target responsibility] and also would be performed by a worker in "{category_occ}" is [describe most relevant task]. The part of the request most relevant to this task is [describe most relevant part of request]. Considering only the most relevant part of the request in relation to the most relevant real-world task, [reasoning about the answer]. Therefore, [repeat the answer that you choose, e.g., part or all of the request ..."]"</thinking>
<answer>A, B, C, D, E, or F</answer>
""".strip()

CLASSIF_PROMPT_DIRECT = """
# Background
You are an LLM expert studying a dataset of user requests made to AI assistants. Specifically, you found a public dataset containing random, scraped user requests made to LLMs and you want to survey the tasks performed by the AI assistants.

# Instructions
Below is a target responsibility (which describes a category of related individual occupational tasks) and an occupation that you're studying (you should only consider the target responsibility as performed by the target occupation). Below is also a series of requests made to an LLM. Keep in mind that as the dataset is gathered from public requests it may contain nonsensical/joke/low-quality requests. Determine whether the requests made to the LLM contain a real-world request that replaces the target responsibility as performed by the target occupation.

Target responsibility: {category_task_detailed}
Target occupation: {category_occ}

The Target responsibility may be shared by other occupations, but make sure you only consider the responsibility *as it would be performed by the Target occupation*.

Requests made to an AI assistant LLM (Each request from the user starts with "Human:"; the assistant's responses are omitted):
```
{instance_text_human_only}
```

Which of the following statements best describes the requests to the LLM in relation to the Target responsibility as performed by the Target occupation?
(A) Part or all of the request to the LLM assistant replaces all or pretty much all of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(B) Part or all of the request to the LLM assistant replaces most (more than 80%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(C) Part or all of the request to the LLM assistant replaces a large portion (more than 50%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(D) Part or all of the request to the LLM assistant replaces a non-negligible amount (more than 10%) of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(E) Part or all of the request to the LLM assistant replaces 10% or less of a real-world task that falls under the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".
(F) The entire request is completely unrelated to the Target responsibility, specifically as performed by a worker in the occupation "{category_occ}".

Make sure to read the requests carefully as they may be misleading.

First, carefully reason about the question in <thinking></thinking> tags. Then, put your answer in <answer></answer> tags (only put the letter here and nothing else). Make sure you follow this format exactly:
<thinking>Follow this template exactly: "The real-world task most relevant to the request that falls under [target responsibility] and also would be performed by a worker in [occupation] is [describe most relevant task]. The part of the request to the LLM assistant most relevant to this task is [describe most relevant part of request]. Considering only the most relevant part of the request in relation to the most relevant real-world task, [reasoning about the answer]. Therefore, [repeat the answer that you choose, e.g., "part or all of the request ..."]."</thinking>
<answer>A, B, C, D, E, or F</answer>
""".strip()

DWA_SUMMARIZE_PROMPT = """
# Background
You are an LLM expert deeply familiar with how users prompt LLMs. You are trying to summarize user requests made to AI assistants in order to understand how and why people use LLMs.

# Instructions
The following are one or more user requests from a conversation between a user and AI assistant (the assistant's responses are omitted). Please summarize these requests.

First read the requests very carefully, then write a summary structured in the following way:
- In one sentence, succinctly extract the action the the user wants the assistant to perform. Focus on the task (e.g., summarization, debugging) rather than the body of the text (e.g., the text being summarized, the code being debugged)---this may mean that this summary is very short, but that's OK. The summary should also focus on the task that the user wants the assistant to most immediately perform, rather than tasks the assistant wants the assistant to perform in the future---this may mean that the summary is trivial (e.g., "acknowledge these instructions"), but that's OK as well. Furthermore, it's very important that the number of words you use that describe actions are greater than the number of words you use to describe non-actions. Start this with: "The action the user wants the assistant to perform is...".
- In one sentence, make an educated inference as to why the user is asking the AI assistant this (e.g., to help with homework, to learn, to write blog posts for their business, etc.). Answer with only the most likely motivation. Start this with: "The user might be asking this because...". Read the requests carefully before answering: it may be important to consider how the user makes the request, how the request is formatted, etc.
- In one sentence, make an educated inference as to the occupation of the person who asked the request and the industry that they work in. If it's not possible to make an inference, then specify that it's unknown. Start this with "The person asking this request probably works as a [occuaption] in the industry of [industry]" OR "The person asking this request is probably a student completing a [program] in [major/specialty/etc.]" OR "It's not clear the occupation of the person who asked this request".

Retain a matter-of-fact, neutral tone for your summary. Make sure to focus on the actual request(s) being asked rather than background information.

User request(s):
```
{instance_text_human_only}
```

# Output format
First, plan the contents of your summary using <thinking></thinking> tags. Then, provide your final answer in <answer></answer> tags. Follow this format exactly:
<thinking>
    <deconstruct>Overall, the user wants the assistant to [...]. In the [request/requests] provided, the user expects the assistant to output in the next [turn/turns] [...]. The user employs prompting techniques like [...].</deconstruct>
    <focus>I need to make sure that I'm looking at the right parts of the inputs. [An excerpt/Excerpts] relevant to the task, rather than the body of the text, [is/are] [...]. [An excerpt/Excerpts] relevant to the most immediate task, rather than future tasks, [is/are] [...].</focus>
</thinking>
<answer>
    [put your final answer here, which should be exactly three sentences]
</answer>
""".strip()

DWA_CLASSIF_PROMPT_TASK_ONLY = """
# Background
You are an economist looking to study how well AI assistants perform on occupational tasks. To do this, you want to build a benchmark. Specifically, you have a dataset of user requests made to AI assistants and you want to find samples that would accurately measure the performance of AI assistants on a target occupational responsibility.

# Instructions
Below is a target responsibility (which describes a category of related individual occupational tasks). Below is also a description of a request made to an AI assistant. Determine the amount of overlap between the request and the target responsibility.

Target responsibility: {category_task_detailed}

Request made to AI assistant:
{summ_instance_text}

Which of the following statements best describes the request in relation to the Target responsibility?
(A) Part or all of the request replaces all or pretty much all of a real-world task that falls under the Target responsibility.
(B) Part or all of the request replaces most (more than 80%) of a real-world task that falls under the Target responsibility.
(C) Part or all of the request replaces a significant part (more than 50%) of a real-world task that falls under the Target responsibility.
(D) Part or all of the request replaces a non-negligible amount (more than 10%) of a real-world task that falls under the Target responsibility.
(E) Part or all of the request replaces 10% or less of a real-world task that falls under the Target responsibility.
(F) The entire request is completely unrelated to the Target responsibility.

First, think about what the answer is using the <thinking></thinking> tags, following the given template. Then, put your final answer in <answer></answer> tags (only put the letter here and nothing else). Answer using only the information provided to you. Make sure you follow this format exactly:
<thinking>
- A concrete example of a real-world task that (1) definitely falls under and satisfies all aspects of [repeat the target responsibility] and (2) is relevant to the request is [describe most relevant task that also definitely is included in the target responsibility]. 
- The part of the request most relevant to this real-world task is [describe most relevant part of request to the real-world task].
- Considering only the most relevant part of the request in relation to the real-world task, [reasoning about the answer].
- Therefore, [the answer choice, e.g., part or all of the request ..."].
</thinking>
<answer>A, B, C, D, E, or F</answer>
""".strip()

IWA_CLASSIF_PROMPT_DIRECT = """
# Background
You are an LLM expert studying a dataset of user requests made to AI assistants. Specifically, you found a public dataset containing random, scraped user requests made to LLMs and you want to survey the tasks performed by the AI assistants.

# Task
Below is a series of user requests made to an AI assistant. Below is also a Target responsibility (which describes a category of related real-world occupational tasks) that you're studying. Keep in mind that as the user requests are sourced from a public dataset it may contain nonsensical/joke/low-quality requests. Determine whether the requests made to the LLM contain a request that replaces a real-world task that falls under the Target responsibility.

User requests made to an AI assistant (Each request from the user starts with "Human:"; the assistant's responses are omitted):
```
{instance_text_human_only}
```

Target responsibility: {iwa_detailed}

Which of the following statements best describes the requests to the AI assistant in relation to the Target responsibility?
(A) Part or all of the request replaces all or pretty much all of a real-world task that falls under the Target responsibility.
(B) Part or all of the request replaces most (at least 50%) of a real-world task that falls under the Target responsibility.
(C) Part or all of the request replaces some (at least 15%) of a real-world task that falls under the Target responsibility.
(D) Part or all of the request replaces a non-negligible amount (at least 5%) of a real-world task that falls under the Target responsibility.
(E) The request is topically related but does not replace a significant portion (less than 5%) of a real-world task that falls under the Target responsibility.
(F) The request is completely unrelated to the Target responsibility.

## Step-by-step instructions
To perform the task, you must carefully follow these step-by-step instructions:
1) Reread the requests above. Make sure to read the requests carefully as they may be misleading. Also, make sure to ignore requests that are completely irrelevant to the Target responsibility---you don't need to consider these when making your decision.
2) Reread the Target responsibility.
3) In <thinking></thinking> tags, reason about the answer in exactly three sentences. Make sure to only answer using only the information provided to you. Avoid speculation.
4) Put your final answer in <answer></answer> tags (only put the letter here and nothing else). 

## Output format
Make sure you follow this format exactly:
<thinking>
- In the part of the user requests most relevant to the Target responsibility, the user wants the assistant to [concise summary of the action the user wants the assistant to perform that's the most relevant to the Target responsibility].
- The real-world task most relevant to the user requests is [description of the most relevant real-world task to the user's request that also falls under the Target responsibility].
- Considering only the most relevant part of the request in relation to the most relevant real-world task, [reasoning about the answer].
</thinking>
<answer>A, B, C, D, E, or F</answer>
""".strip()

DWA_CLASSIF_PROMPT_DIRECT = """
# Background
You are an LLM expert studying a dataset of user requests made to AI assistants. Specifically, you found a public dataset containing random, scraped user requests made to LLMs and you want to survey the tasks performed by the AI assistants.

# Task
Below is a series of user requests made to an AI assistant. Below is also a Target responsibility (which describes a category of related real-world occupational tasks) that you're studying. Keep in mind that as the user requests are sourced from a public dataset it may contain nonsensical/joke/low-quality requests. Determine whether the requests made to the LLM contain a request that replaces a real-world task that falls under the Target responsibility.

User requests made to an AI assistant (Each request from the user starts with "Human:"; the assistant's responses are omitted):
```
{instance_text_human_only}
```

Target responsibility: {dwa_detailed}

Which of the following statements best describes the requests to the AI assistant in relation to the Target responsibility?
(A) Part or all of the request replaces all or pretty much all of a real-world task that falls under the Target responsibility.
(B) Part or all of the request replaces most (at least 50%) of a real-world task that falls under the Target responsibility.
(C) Part or all of the request replaces some (at least 10%) of a real-world task that falls under the Target responsibility.
(D) Part or all of the request replaces a non-negligible amount (at least 1%) of a real-world task that falls under the Target responsibility.
(E) The request is topically related but does not replace a meaningful portion (less than 1%) of a real-world task that falls under the Target responsibility.
(F) The request is completely unrelated to the Target responsibility.

## Step-by-step instructions
To perform the task, you must carefully follow these step-by-step instructions:
1) Reread the requests above. Make sure to read the requests carefully as they may be misleading. Also, make sure to ignore requests that are completely irrelevant to the Target responsibility---you don't need to consider these when making your decision.
2) Reread the Target responsibility.
3) In <thinking></thinking> tags, reason about the answer in exactly three sentences. Make sure to only answer using only the information provided to you. Avoid speculation.
4) Put your final answer in <answer></answer> tags (only put the letter here and nothing else). 

## Output format
Make sure you follow this format exactly:
<thinking>
- In the part of the user requests most relevant to the Target responsibility, the user wants the assistant to [concise summary of the action the user wants the assistant to perform that's the most relevant to the Target responsibility].
- The real-world task most relevant to the user requests is [description of the most relevant real-world task to the user's request that also falls under the Target responsibility].
- Considering only the most relevant part of the request in relation to the most relevant real-world task, [reasoning about the answer].
</thinking>
<answer>A, B, C, D, E, or F</answer>
""".strip()