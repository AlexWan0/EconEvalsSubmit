from typing import TypedDict, Literal
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout

from .base import APIClient, _WrappedReturn, RequestArgs
from .auth import GOOGLE_GENAI_API_KEY
from .exceptions import RequestException, TokenLimitException, InvalidResponseError
from .token_tracking import oai_tokens_tracker
from .openai import BasicInput, Turn


class Content(TypedDict):
    pass

class TextContent(Content):
    text: str

class GoogleTurn(TypedDict):
    role: Literal['user', 'model']
    parts: list[Content]

_GOOGLE_ROLE_MAP: dict[Literal['user', 'assistant'], Literal['user', 'model']] = {'user': 'user', 'assistant': 'model'}
def to_google_format(orig: list[Turn]) -> list[GoogleTurn]:
    res: list[GoogleTurn] = []
    for turn in orig:
        text_content: TextContent = {'text': turn['content']}
        
        res.append({
            'role': _GOOGLE_ROLE_MAP[turn['role']],
            'parts': [
                text_content
            ]
        })
    
    return res

class GoogleGenAIAPI(APIClient[BasicInput, str]):
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    async def _call(
            self,
            payload: BasicInput,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[str]:

        if payload.logprobs or payload.top_logprobs is not None:
            raise ValueError("logprobs isn't supported for Google GenAI")

        full_endpoint = self.ENDPOINT.format(
            model_name=payload.model_name,
            api_key=GOOGLE_GENAI_API_KEY
        )

        request_body = {
            "contents": to_google_format(payload.model_input),
            "generationConfig": {
                "maxOutputTokens": payload.max_tokens,
                "temperature": payload.temperature
            }
        }

        headers = {
            "Content-Type": "application/json",
        }

        async with session.post(
            full_endpoint,
            headers=headers,
            json=request_body,
            timeout=ClientTimeout(total=request_args.post_timeout)
        ) as resp:
            resp_body = await resp.json()

            if resp.status != 200:
                raise RequestException(f"{resp.status}: {resp.headers}\n{resp_body}")

        cand = resp_body["candidates"][0]
        content = cand["content"]["parts"][0]["text"]

        usage = resp_body["usageMetadata"]
        total_tokens = usage["totalTokenCount"]
        prompt_tokens = usage["promptTokenCount"]
        completion_tokens = usage["candidatesTokenCount"]

        if not isinstance(content, str):
            raise InvalidResponseError(f"returned content is not a string: {content}")

        if not isinstance(total_tokens, int):
            raise InvalidResponseError(f"returned totalTokenCount is not an int: {total_tokens}")

        await oai_tokens_tracker.add_tokens(payload.model_name, total_tokens)

        # check if we've exceeded limits
        if oai_tokens_tracker.exceeded_model_limits():
            raise TokenLimitException(f"Exceeded token limit: current_model={payload.model_name}; token_tracker={oai_tokens_tracker.get_total_tokens_used()}")

        return _WrappedReturn(
            content=content,
            tokens_used=total_tokens,
            logprobs=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
