from .openai import OpenAIAPI
from .auth import TOGETHER_API_KEY


class TogetherAPI(OpenAIAPI):
    ENDPOINT = "https://api.together.xyz/v1/chat/completions"

    @property
    def api_key(self):
        return TOGETHER_API_KEY
