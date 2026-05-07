import re

from .parsing_utils import (
    bucket_activity_task_match_score as bucket_activity_task_match_score,
    parse_activity_task_match_blocks as parse_activity_task_match_blocks,
    parse_interview_label as parse_interview_label,
    parse_minutes_from_activity_time as parse_minutes_from_activity_time,
)


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



PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS = """
The following is are a respondent's free-form answers about how much time a chatbot would save on an instance of a task they perform at work. The answers include (2) a breakdown of the activities or steps that the respondent executes while completing the task, and (2) estimates of how much time the respondent thinks a chatbot would save on each of those activities or steps.

You are a careful annotator that will compute the time savings from using an AI chatbot (out of the original time spent, without chatbot-assistance) for the task currently being scored. Your goal is to make sure that the breakdown of activities being used to compute the time savings are a good representation of the task currently being scored, rather than other tasks belonging to this occupation.

Occupation:
{category_occ}

Task currently being scored:
{category_task}

All O*NET tasks belonging to this occupation:
{occupation_tasks}

Respondent answer (breakdown of activities):
```
{_result_9}
```

Respondent answer (time-savings estimates for activities):
```
{_result_17}
```

# Instructions
1. Identify each distinct activity or step in the respondent answer that is part of completing the task. Make sure to identify ALL activities that the respondent mentioned *as part of completing the task*, not just the activities that the respondent discussed saving time on. For example, the respondent may first describe their "current process" then point out a couple activities that a chatbot could save them time on. You should identify the full list of activities mentioned in the respondent answer, including both the activities that the respondent discussed saving time on and the activities that they did not discuss saving time on but are still part of their process for completing the task. List the time-savings as zero for any activities that the respondent did not discuss saving time on, but that are still part of their process for completing the task.
2. For each activity, decide whether its estimated time savings should be attributed to the task currently being scored. First, read the task being scored carefully. Then, identify two other candidate tasks that *could* be related to the activity. Then, think about whether the activity is a good fit for the task being scored, in comparison to the other tasks. Finally, label “yes” if the activity is a good fit for this task and you believe its time savings can reasonably be used toward this task, as opposed to a different O*NET task for the occupation. Label “no” if the activity would be better attributed to another task from the list. Only label "yes" if the activity is most closely matched to the task currently being scored.
Ex) If the task currently being scored involves analyzing data or testing a system, and the activity is documenting results or writing notes/ a report, then this activity should be labeled "no" if there is another O*NET task that is a better fit for the activity.
Ex) If the task currently being scored involved executing a manual process, and the activity involves planning or organizing the steps to be taken, then this activity should be labeled "no" if there is another O*NET task that is a better fit for the activity.
Ex) If the task currently being scored involved developing or modify a process, and the activity involved communicating with a team member of client about that process, then this activity should be labeled "no" if there is another O*NET task that is a better fit for the activity.
3. For each activity, read the time_without_chatbot field for that activity and report that amount of time *spent* in *minutes*. This may involve converting from other time units (e.g., hours or seconds) into minutes. If the respondent specifies a range for the time spent, then you should use the mid-point of that range as the time-spent value to convert to minutes. Do NOT use edge-case time-spent values if the worker also specifies an average or more typical time-spent value: for example, if the respondent says "This takes me 2 to 3 hours, but usually it's more like 2 and a half hours", then you should use 2 and a half hours as the time-spent value to convert to minutes. It's also possible that there is not enough information to calculate a time-spent value in minutes for an activity. In that case, write "Missing information".
4. For each activity, read the time_saved_with_chatbot field for that activity and calculate the amount of time *saved*  in *minutes*. This may involve converting from other time units (e.g., hours or seconds) into minutes. This may also involve subtracting from the original task time. If the worker specifies a range for the time-savings, then you should use the mid-point of that range as the time-savings value to convert to minutes. If the summary includes multiple time-savings estimates, then you should use the most realistic, common, or typical time-savings value: for example, if the summary says "Using a chatbot would save 10 minutes if the activity is particularly difficult, but usually it would only save 5 minutes", then you should use 5 minutes as the time-savings value to convert to minutes. Do NOT use edge-case time-savings values if the worker also specifies an average or more typical time-savings value. It's also possible that there is not enough information to calculate a time-savings value in minutes for an activity. In that case, write "Missing information".

Output only the following XML structure:
<activity_task_matches>
  <activity>
    <activity_description>[description of the activity in exactly one sentence; it MUST start with "The respondent needs to..."]</activity_description>
    <task_currently_being_scored>[repeat the O*NET task currently being scored exactly: "{category_task}"]</task_currently_being_scored>
    <candidate_tasks>[semicolon-separated list of two OTHER O*NET tasks that *could* be related to this activity, pulled from the "All O*NET tasks belonging to this occupation:" list]</candidate_tasks>
    <thinking>[reasoning about whether this activity belongs to the task being scored versus the candidate tasks]</thinking>
    <belongs_to_task_being_scored>[yes or no]</belongs_to_task_being_scored>
    <original_time_minutes>[calculated time-spent for the activity (converted to minutes) from the time_without_chatbot field. Only include numeric values. Do not include any text or other characters. Write 0 if the respondent did not discuss the amount of time this activity takes. If the respondent specified a range in the summary use the mid-point. It's also possible that there is not enough information to calculate a time-spent value in minutes for an activity. In that case, write "Missing information".]</original_time_minutes>
    <reported_time_savings_minutes>[calculated time saved for the activity (converted to minutes) from the time_saved_with_chatbot field. Only include numeric values. Do not include any text or other characters. Write 0 if the respondent did not discuss saving time on this activity. If the respondent specified a range in the summary use the mid-point. It's also possible that there is not enough information to calculate a time-savings value in minutes for an activity. In that case, write "Missing information".]</reported_time_savings_minutes>
  </activity>
</activity_task_matches>

Repeat the <activity> block once for each activity. Use exactly the tag names shown above. Do not include activity descriptions, explanations, or any other free-text fields in the XML.
""".strip()


def pretty_print_result_for_prompt(result):
    """
    Format model-generated result text for prompt interpolation.

    Assumes result is plain text from an earlier model step. Returns stripped text
    with trailing whitespace removed and excessive blank lines collapsed.
    """
    if result is None:
        return ""

    lines = [line.rstrip() for line in str(result).strip().splitlines()]
    formatted = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", formatted)


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
    payload["result"] = pretty_print_result_for_prompt(result_text)
    return final_turn_template.format(**payload)


def build_activity_task_match_prompt(
    row,
    result_text,
    final_turn_template=PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS,
):
    """
    Build the v4 preprocessing prompt for activity-to-task matching.

    Assumes row includes occupation_tasks, a newline-delimited list of all task
    options for the occupation. Returns the prompt sent to the preprocessing LLM.
    """
    payload = row.to_dict()
    payload["result"] = pretty_print_result_for_prompt(result_text)
    return final_turn_template.format(**payload)
