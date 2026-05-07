from typing import Literal, List
from typing_extensions import TypedDict

class Turn(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

Conversation = list[Turn]


def convo_to_string_nosystem(
        convo: Conversation
    ) -> str:

    return '\n\n'.join([
        ('User: ' + turn['content']) if turn['role'] == 'user' else ('Assistant: ' + turn['content'])
        for turn in convo
        if turn['role'] != 'system'
    ])
