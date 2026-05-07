from __future__ import annotations
from copy import deepcopy

from .interview import (
    PROMPT_INTERVIEW_BINARIZE,
    INTERVIEW_BETA_MAP,
    _SYSTEM_PROMPT_DEFAULT,
    _SYSTEM_PROMPT_2,
    _SYSTEM_PROMPT_DEMO,
    build_categorize_prompt,
    parse_interview_label,
)


INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE: list[dict] = [
    {
        "role": "user",
        "content": (
            "Hi! I'm David. I work for Ipsos on behalf of WorkflowAI. Thank you for taking the time to talk with me today. "
            "Just to give some background, WorkflowAI is a company that makes AI assistants that help people with their work. They have a prototype chatbot that they are testing and they want to see how well it performs with real-world occupational tasks. Specifically, they want to know how much time it saves. "
            "Can you first tell me a bit about your background? Specifically, can you tell me about your job title, your employer, and how long you've been working in your occupation?"
        ),
    },
    
    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_DEFAULT}, # system prompt used for this turn
    },

    {
        "role": "user",
        "content": "Thanks for sharing! Now, I see in your in-take form that you often perform the task of \'{category_task}\'. Can you tell me a bit more about what you mean by that? What does this task involve? What does this task not involve?"
    },

    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_2},
    },

    {
        "role": "user",
        "content": "Have you ever used an AI assistant for this task before?"
    },

    {
        "role": "assistant",
        "content": "No, I have not."
    },

    {
        "role": "user",
        "content": "When was the last time you performed this task?"
    },

    {
        "role": "assistant",
        "content": "Actually, I just performed this task yesterday!"
    },

    {
        "role": "user",
        "content": "Great! Let's think back to yesterday when you performed this task. Can you first help me contextualize the situation a bit? Tell me about what was going on right before you performed the task. For example, what time of day was it? Where were you? What else were you doing? Who else were you with?"
    },

    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_DEFAULT}
    },

    {
        "role": "user",
        "content": "Now, can you put yourself back in that moment and walk me through how you performed the task step by step? Please be as detailed as possible. I'm trying to get a good understanding of not only what specifically you did but also the process you went through to perform those things, so please include even small details that you think might be relevant. Make sure to tell me about how long each step took you."
        "Don't be embarrassed about talking too much or including too many details. I really want to get a good picture of how you performed this task, so the more details the better!"
    },

    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_DEFAULT}
    },

    {
        "role": "user",
        "content": "Thanks - that was really helpful! Now, I want to demonstrate for you a prototype of the WorkflowAI chatbot that I mentioned earlier. This chatbot is designed to help people with their work tasks. I want to show you how it works on the task of \'{category_task}\'. After I show you, I would love to get your feedback on it. Does that sound good?"
    },

    {
        "role": "assistant",
        "content": "Yes. That sounds good!"
    },

    {
        "role": "user",
        "content": "Ok, so I'll just pull up the demonstration...\n\n<demo>\n<prompt>\n{query}\n</prompt>\n<response>\n{response}\n</response>\n</demo>\n\nImagine yourself back in the moment when you were performing the task yesterday. Please take me through that again, but this time you should also imagine that you have access to a chatbot on your phone or computer with the demonstrated capabilities. As you talk me through the steps of how you performed the task, you should also talk me through the times where you could have used the chatbot. Make sure to think out loud! Say things like \'I could then pull out my phone and ask the chatbot to...\'. Note when using a chatbot seems ridiculous or unnatural, and also note when it seems helpful and like something you would actually do. I want your stream of consciousness."
    },

    # {
    #     "role": "user",
    #     "content": "Ok, so I'll just pull up the demonstration...\n\n<demo>\n<prompt>\n{query}\n</prompt>\n<response>\n{response}\n</response>\n</demo>\n\nPlease read it carefully and tell me what you think about the prompt and the response. If you had a chatbot that had this capability, what would it help with? What would it not help with? Would it make certain things easier? Would it save you time?"
    #     "Please only answer in relation to the task you discussed earlier of \'{category_task}\'. Even if the demonstration shows the chatbot being used for other tasks, please focus your feedback on how it would or would not be helpful for the task of \'{category_task}\'."
    # },

    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_DEMO}
    },

    {
        "role": "user",
        # "content": "Now with this in mind, can you give a rough estimate of how much time this would save you on the task you discussed? Again, I want you to verbalize your thought process as much as possible. Think about how much time different steps of the task took you, and which of those steps the chatbot would help with. Make sure to think back to YOUR previous answer for this, not what the task involves in general. After that, give a percentage estimate."
        "content": "Now with this in mind, can you give a rough estimate of how much time this would save you on the task you discussed? Again, I want you to verbalize your thought process as much as possible. Think about how much time different steps of the task took you, and which of those steps the chatbot would help with. Make sure to think back to YOUR previous answer for this, not what the task involves in general. After that, give a percentage estimate: 0%, 25%, 50%, 75%, or 100%."
        " Please think only about the task itself and not about other related tasks. Let me just remind you of your previous answer for what the task does and doesn't involve. You said '{_result_3}'."
    },

    {
        "role": "assistant",
        "content": {'system': _SYSTEM_PROMPT_DEFAULT}
    },
]

INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN: list[dict] = deepcopy(INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE)
INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE_SPAN[-2] = {
    "role": "user",
    "content": "Now with this in mind, can you give a rough estimate of how much time this would save you on the task you discussed? Again, I want you to verbalize your thought process as much as possible. Think about how much time different steps of the task took you, and which of those steps the chatbot would help with. Make sure to think back to YOUR previous answer for this, not what the task involves in general. After that, give a percentage estimate."
    " Please think only about the task itself and not about other related tasks. Let me just remind you of your previous answer for what the task does and doesn't involve. You said '{_result_3}'."
}


# INTERVIEW_BASE_PROMPT_MESSAGE_TEMPLATE: list[dict] = [
#     {
#         "role": "user",
#         "content": (
#             "Hi! I'm David. I work for Ipsos on behalf of WorkflowAI. Thank you for taking the time to talk with me today. "
#             "Just to give some background, WorkflowAI is a company that makes AI assistants that help people with their work. They have a prototype chatbot that they are testing and they want to see how well it performs with real-world occupational tasks. Specifically, they want to know how much time it saves. "
#             "Can you first tell me a bit about your background? Specifically, can you tell me about your job title, your employer, and how long you've been working in your occupation?"
#         ),
#     },
#     {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
#     {
#         "role": "user",
#         "content": (
#             "Thanks for sharing! Now, I see in your in-take form that you often perform the task of '{category_task}'. "
#             "Can you tell me a bit more about what you mean by that? What does this task involve? What does this task not involve?"
#         ),
#     },
#     {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_2}},
#     {"role": "user", "content": "Have you ever used an AI assistant for this task before?"},
#     {"role": "assistant", "content": "No, I have not."},
#     {"role": "user", "content": "When was the last time you performed this task?"},
#     {"role": "assistant", "content": "Actually, I just performed this task yesterday!"},
#     # {
#     #     "role": "user",
#     #     "content": (
#     #         "Great! Let's think back to yesterday when you performed this task. "
#     #         "Can you first help me contextualize the situation a bit? Tell me about what was going on right before you performed the task. "
#     #         "For example, what time of day was it? Where were you? What else were you doing? Who else were you with?"
#     #     ),
#     # },
#     # {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
#     {
#         "role": "user",
#         "content": (
#             "Now, can you put yourself back in that moment and walk me through how you performed the task step by step? "
#             "Please be as detailed as possible. I'm trying to get a good understanding of not only what specifically you did but also the process you went through to perform those things, "
#             "so please include even small details that you think might be relevant. Make sure to tell me about how long each step took you."
#             "Don't be embarrassed about talking too much or including too many details. I really want to get a good picture of how you performed this task, so the more details the better!"
#             "However, make sure to only talk about what you did for the task you mentioned in the beginning of the interview and not about other tasks that might be related but are not the focus of this interview."
#         ),
#     },
#     {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
#     {
#         "role": "user",
#         "content": (
#             "Thanks - that was really helpful! Now, I want to demonstrate for you a prototype of the WorkflowAI chatbot that I mentioned earlier. "
#             "This chatbot is designed to help people with their work tasks. I want to show you how it works on the task of '{category_task}'. "
#             "After I show you, I would love to get your feedback on it. Does that sound good?"
#         ),
#     },
#     {"role": "assistant", "content": "Yes. That sounds good!"},
#     {
#         "role": "user",
#         "content": (
#             "Ok, so I'll just pull up the demonstration...\n\n<demo>\n<prompt>\n{query}\n</prompt>\n<response>\n{response}\n</response>\n</demo>\n\n"
#             "Imagine yourself back in the moment when you were performing the task yesterday. Please take me through that again, but this time you should also imagine that you have access to a chatbot on your phone or computer with the demonstrated capabilities. "
#             "As you talk me through the steps of how you performed the task, you should also talk me through the times where you could have used the chatbot. "
#             "Make sure to think out loud! Say things like 'I could then pull out my phone and ask the chatbot to...'. "
#             "Note when using a chatbot seems ridiculous or unnatural, and also note when it seems helpful and like something you would actually do. I want your stream of consciousness."
#         ),
#     },
#     {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEMO}},
#     {
#         "role": "user",
#         "content": (
#             "Now with this in mind, can you give a rough estimate of how much time this would save you on the task you discussed? "
#             "Please think only about the task itself and not about other related tasks. Let me just remind you of your previous answer for what the task does and doesn't involve. You said '{_result_3}'."
#             "Again, I want you to verbalize your thought process as much as possible. "
#             "Think about how much time different steps of the task took you, and which of those steps the chatbot would help with. "
#             "Make sure to think back to YOUR previous answer for this, not what the task involves in general. "
#             "After that, give a percentage estimate: 0%, 25%, 50%, 75%, or 100%."
#         ),
#     },
#     {"role": "assistant", "content": {"system": _SYSTEM_PROMPT_DEFAULT}},
# ]
