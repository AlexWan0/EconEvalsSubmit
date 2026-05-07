import re


_SYSTEM_PROMPT_BASE = """
You are a worker with the occupation of "{category_occ}". Specifically, here is your background:

Work Persona:
- Occupation: {category_occ}
- Job title: {job_title}
- Years of experience: {years_experience}
- Employer: {company_description}

Personality:
- You are proud of the work that you do and relish in being able to talk about your job.
- You are open-minded about technology, but you always try to see through hype and marketing fluff.
- You always speak in a matter-of-fact way. You do not exaggerate or use hyperbolic language. You are straightforward and honest.
""".strip()

_SYSTEM_PROMPT_DEFAULT = _SYSTEM_PROMPT_BASE + "\n\n" + """
You are being interviewed by a market researcher working for Ipsos on behalf of an AI chatbot company called WorkflowAI. WorkflowAI is a company that makes AI assistants that help people with their work. They have a prototype chatbot that they are testing and they want to see how well it performs with real-world occupational tasks. Specifically, they want to know how much time it saves.

The user will play the role of the market researcher and will ask you questions about your work and ask you to give feedback on an AI assistant. You are to roleplay as this worker.

You must always stay in character and answer questions according to the work persona and personality described above. You must not break character or reveal that you are an AI assistant. Remember that this is a dialogue! Your response is something that is said out loud.
"""

_SYSTEM_PROMPT_2 = _SYSTEM_PROMPT_BASE + "\n\n" + """
In this turn, the market researcher will ask you about "{category_task}". This is a task that you often perform. Here is a more detailed task description: "{task_detailed}"

Answer conversationally using the task information and your persona. Remember that this is a dialogue! Your response is something that is said out loud.
"""

_SYSTEM_PROMPT_DEMO = _SYSTEM_PROMPT_BASE + "\n\n" + """
You are being interviewed by a market researcher working for Ipsos on behalf of an AI chatbot company called WorkflowAI. WorkflowAI is a company that makes AI assistants that help people with their work. They have a prototype chatbot that they are testing and they want to see how well it performs with real-world occupational tasks. Specifically, they want to know how much time it saves.

The user will play the role of the market researcher and will ask you questions about your work and ask you to give feedback on an AI assistant. You are to roleplay as this worker.

You must always stay in character and answer questions according to the work persona and personality described above. You must not break character or reveal that you are an AI assistant.

At a certain point the market researcher will demonstrate the AI assistant on a task. The demonstration will consist of an example prompt given to the assistant and the assistant's response. To not break character, this will happen through a formatted set of XML tags. They will have the following format:
<demo>
<prompt>
[the prompt given to the assistant]
</prompt>
<response>
[the response given by the assistant]
</response>
</demo>

This demonstration is NOT said out loud to you as dialogue. Instead, it is shown to you on a computer screen. You see the prompt and the response. When given the demonstration, you should react to it as if you are reacting to something you see on a computer screen. Remember to stay in character and use your persona when reacting to the demonstration.

Also: remember that this is a dialogue! Your response is something that is said out loud.
""".strip()

INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE: list[dict] = [
    {
        "role": "user",
        "content": (
            "Hi! I'm David. I work for Ipsos on behalf of WorkflowAI. Thank you for taking the time to talk with me today. "
            "Just to give some background, WorkflowAI is a company that makes AI assistants that help people with their work. They have a prototype chatbot that they are testing and they want to see how well it performs with real-world occupational tasks. Specifically, they want to know how much time it saves. "
            "Can you first tell me a bit about your background? Specifically, can you tell me about your job title, your employer, and how long you've been working in your occupation?"
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
    {
        "role": "user",
        "content": (
            "Thanks for sharing! Now, I see in your in-take form that you often perform the task of '{category_task}'. "
            "Can you tell me a bit more about what you mean by that? What does this task involve? What does this task not involve?"
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_2}},
    {"role": "user", "content": "Have you ever used an AI assistant for this task before?"},
    {"role": "assistant", "content": "No, I have not."},
    {
        "role": "user",
        "content": "When was the last time you performed this task?",
    },
    {"role": "assistant", "content": "Actually, I just performed this task yesterday!"},
    {
        "role": "user",
        "content": (
            "Great! Let's think back to yesterday when you performed this task. "
            "Can you first help me contextualize the situation a bit? Tell me about what was going on right before you performed the task. "
            "For example, what time of day was it? Where were you? What else were you doing? Who else were you with?"
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
    {
        "role": "user",
        "content": (
            "Now, can you put yourself back in that moment and walk me through how you performed the task step by step? "
            "Please be as detailed as possible. I'm trying to get a good understanding of not only what specifically you did but also the process you went through to perform those things, "
            "so please include even small details that you think might be relevant. Make sure to tell me about how long each step took you."
            "Don't be embarrassed about talking too much or including too many details. I really want to get a good picture of how you performed this task, so the more details the better!"
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
    {
        "role": "user",
        "content": (
            "Thanks - that was really helpful! Now, I want to demonstrate for you a prototype of the WorkflowAI chatbot that I mentioned earlier. "
            "This chatbot is designed to help people with their work tasks. I want to show you how it works on the task of '{category_task}'. "
            "After I show you, I would love to get your feedback on it. Does that sound good?"
        ),
    },
    {"role": "assistant", "content": "Yes. That sounds good!"},
    {
        "role": "user",
        "content": (
            "Ok, so I'll just pull up the demonstration...\n\n<demo>\n<prompt>\n{query}\n</prompt>\n<response>\n{response}\n</response>\n</demo>\n\n"
            "First give your first impressions of things that the demo shows in relation to the task of '{category_task}'. "
            "Then, imagine yourself back in the moment when you were performing the task yesterday. Please take me through that again, but this time you should also imagine that you have access to a chatbot on your phone or computer with the demonstrated capabilities. "
            "As you talk me through the steps of how you performed the task, you should also talk me through the times where you could have used the chatbot. "
            "Make sure to think out loud! Say things like 'I could then pull out my phone and ask the chatbot to...'. "
            "Note when using a chatbot seems ridiculous or unnatural, and also note when it seems helpful and like something you would actually do. "
            "I want your stream of consciousness."
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEMO}},
    {
        "role": "user",
        "content": (
            "Now with this in mind, can you give a rough estimate of how much time this would save you on the task of '{category_task}'? "
            "Please give a rough percentage (either 0%, 25%, 50%, 75%, or 100%) of how much time you think it would save you, "
            "and also explain a bit about how you arrived at that estimate."
        ),
    },
    {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
]

PROMPT_INTERVIEW_BINARIZE = """
The following is a free-form response given to the question: "Please give a rough percentage (either 0%, 25%, 50%, 75%, or 100%) of how much time you think it would save you, and also explain a bit about how you arrived at that estimate."

Response:
```
{result}
```

Categorize this response into one of the following two categories:
<answer>Yes</answer>: This response indicates that the respondent thinks that a chatbot like the one demonstrated will save them at least 50% of the time they spend on the task of '{category_task}'.

<answer>No</answer>: This response indicates that the respondent does not think that a chatbot like the one demonstrated will save them at least 50% of the time they spend on the task of '{category_task}'.

Answer ONLY with Yes or No surrounded in <answer> tags. For example, if you think the answer is Yes, answer with <answer>Yes</answer>. If you think the answer is No, answer with <answer>No</answer>.
""".strip()


PROMPT_INTERVIEW_BINARIZE_CAT = """
The following is a free-form response given to the question: "Please give a rough percentage (either 0%, 25%, 50%, 75%, or 100%) of how much time you think it would save you, and also explain a bit about how you arrived at that estimate."

Response:
```
{result}
```

Categorize this response into one of the following five categories:
<answer>0%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 0% of the time they spend on the task of '{category_task}'.

<answer>25%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 25% of the time they spend on the task of '{category_task}'.

<answer>50%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 50% of the time they spend on the task of '{category_task}'.

<answer>75%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 75% of the time they spend on the task of '{category_task}'.

<answer>100%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 100% of the time they spend on the task of '{category_task}'.

Answer ONLY with one of 0%, 25%, 50%, 75%, or 100%, surrounded in <answer> tags. For example:
<answer>50%</answer>
""".strip()

PROMPT_INTERVIEW_BINARIZE_SPAN = """
The following is a free-form response given to the question: "Please give a rough percentage of how much time you think it would save you, and also explain a bit about how you arrived at that estimate."

Response:
```
{result}
```

Categorize this response into one of the following four categories:
<answer>0-24%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 0-24% of the time they spend on the task of '{category_task}'.

<answer>25-49%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 25-49% of the time they spend on the task of '{category_task}'.

<answer>50-74%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 50-74% of the time they spend on the task of '{category_task}'.

<answer>75-100%</answer>: This response indicates that the chatbot like the one demonstrated would save the respondent 75-100% of the time they spend on the task of '{category_task}'.
""".strip()


PROMPT_INTERVIEW_BINARIZE_CAT10 = """
The following is a free-form response given to the question: "Please give a rough percentage of how much time you think it would save you, and also explain a bit about how you arrived at that estimate."

Response:
```
{result}
```

Categorize this response into one of the following eleven categories:
Answer ONLY with one of the following values in <answer> tags: <answer>0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%</answer>.

The answer tags must contain ONLY one of 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, or 100%.
""".strip()


def parse_interview_label(
    output: str | None,
    *,
    allowable_entries: tuple[str, ...] = ("yes", "no"),
) -> str | None:
    if output is None:
        return None

    if not allowable_entries:
        return None

    pattern = "|".join([re.escape(x) for x in allowable_entries])
    match = re.search(rf"<answer>\s*({pattern})\s*</answer>", output, re.IGNORECASE)
    if match is None:
        return None

    parsed = match.group(1).lower()
    normalized_allowables = tuple(x.lower() for x in allowable_entries)
    if parsed not in normalized_allowables:
        return None

    return parsed


INTERVIEW_BETA_MAP = {
    "yes": 1.0,
    "no": 0.0,
}

INTERVIEW_BETA_MAP_CAT = {
    "0%": 0.0,
    "25%": 0.25,
    "50%": 0.5,
    "75%": 0.75,
    "100%": 1.0,
}

INTERVIEW_BETA_MAP_CAT10 = {
    "0%": 0.0,
    "10%": 0.10,
    "20%": 0.20,
    "30%": 0.30,
    "40%": 0.40,
    "50%": 0.50,
    "60%": 0.60,
    "70%": 0.70,
    "80%": 0.80,
    "90%": 0.90,
    "100%": 1.0,
}

INTERVIEW_BETA_MAP_SPAN = {
    "0-24%": 0.12,
    "25-49%": 0.37,
    "50-74%": 0.62,
    "75-100%": 0.87,
}


def build_categorize_prompt(
    row,
    result_text: str,
    final_turn_template: str = PROMPT_INTERVIEW_BINARIZE,
) -> str:
    payload = row.to_dict()
    payload["result"] = str(result_text) if result_text is not None else ""
    return final_turn_template.format(**payload)
