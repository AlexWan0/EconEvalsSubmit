from typing import TypedDict, Literal


class Turn(TypedDict):
    """Type specification of a single turn in a conversation.

    In this repo, conversations are typically typed as `list[Turn]`.

    Args:
        role (Literal['user', 'assistant']): The speaker role in the turn.
        content (str): The text content of the message.
    """

    role: Literal['user', 'assistant']
    content: str

def convo_to_str(convo: list[Turn]) -> str:
    """Converts a conversation to a string.

    Format is ``Human: [turn text]\\n\\nAssistant: [turn text]\\n\\nHuman: [turn text] ...``

    Args:
        convo (list[Turn]): List of turns in the conversation

    Returns:
        str: Conversation as a string
    """

    convo_turns = []
    for turn in convo:
        assert isinstance(turn, dict), convo

        assert turn['role'] == 'user' or turn['role'] == 'assistant'

        convo_turns.append(
            f'Human: {turn["content"]}'
            if turn["role"] == "user" else f'Assistant: {turn["content"]}'
        )

    return '\n\n'.join(convo_turns)

def convo_to_str_human_only(convo: list[Turn]) -> str:
    """Converts a conversation to a string but only includes the human (user) turns

    Format is ``Human: [turn text]\\n\\nHuman: [turn text] ...``

    Args:
        convo (list[Turn]): List of turns in the conversation

    Returns:
        str: Conversation as a string
    """

    convo_turns = []
    for turn in convo:
        assert turn['role'] == 'user' or turn['role'] == 'assistant'

        if turn['role'] != 'user':
            continue

        convo_turns.append(
            f'Human: {turn["content"]}'
            if turn["role"] == "user" else f'Assistant: {turn["content"]}'
        )

    return '\n\n'.join(convo_turns)
