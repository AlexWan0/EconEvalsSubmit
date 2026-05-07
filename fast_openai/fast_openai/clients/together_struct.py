from .openai_struct import OpenAIStructAPI
from .auth import TOGETHER_API_KEY


class TogetherStructAPI(OpenAIStructAPI):
    ENDPOINT = "https://api.together.xyz/v1/chat/completions"

    @property
    def api_key(self):
        return TOGETHER_API_KEY
