from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import json
from dataclasses import dataclass

from ..base import Serializable, JSONType
from .base import APIClient, _WrappedReturn, RequestArgs
from .logprobs import collect_openai_logprobs
from .auth import OPENAI_API_KEY
from .exceptions import RequestException, TokenLimitException, InvalidResponseError
from .token_tracking import oai_tokens_tracker
from .openai import Turn


StructType = TypeVar("StructType", bound=BaseModel)

@dataclass
class StructInput(Serializable, Generic[StructType]):
    model_input: list[Turn]
    response_format: Type[StructType]
    model_name: str
    max_tokens: int
    temperature: float
    logprobs: bool = False
    top_logprobs: int | None = None
    cache_flag: str = ''

    def to_json(self) -> JSONType:
        data = {
            'model_input': self.model_input,
            'response_format': self.response_format.model_json_schema(),
            'model_name': self.model_name,
            'max_tokens': str(self.max_tokens),
            'temperature': self.temperature,
            'logprobs': self.logprobs,
            'cache_flag': self.cache_flag
        }

        if self.top_logprobs is not None:
            data['top_logprobs'] = self.top_logprobs

        return data


from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai._types import NOT_GIVEN # type: ignore
from openai.lib._parsing._completions import parse_chat_completion, type_to_response_format_param # type: ignore

def _parser(
        raw_completion: ChatCompletion,
        response_format: Type[StructType]
    ) -> ParsedChatCompletion[StructType]:

    return parse_chat_completion(
        response_format=response_format,
        chat_completion=raw_completion,
        input_tools=NOT_GIVEN,
    )

class OpenAIStructAPI(APIClient[StructInput[StructType], StructType]):
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    @property
    def api_key(self):
        return OPENAI_API_KEY

    async def _call(
            self,
            payload: StructInput[StructType],
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[StructType]:

        request_body = {
            "model": payload.model_name,
            "messages": payload.model_input,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "response_format": type_to_response_format_param(payload.response_format),
            "logprobs": payload.logprobs,
        }

        if payload.top_logprobs is not None:
            request_body["top_logprobs"] = payload.top_logprobs

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with session.post(
            self.ENDPOINT,
            headers=headers,
            json=request_body,
            timeout=ClientTimeout(total=request_args.post_timeout)
        ) as resp:
            resp_body = await resp.json()

            if resp.status != 200:
                raise RequestException(f"{resp.status}: {resp.headers}\n{resp_body}")
        
        # json -> ChatCompletion
        response_compl = ChatCompletion.model_validate(resp_body)

        # ChatCompletion -> ParsedChatCompletion
        response = _parser(response_compl, payload.response_format)
        
        # get parsed response
        response_parsed = response.choices[0].message.parsed

        if response_parsed is None:
            raise InvalidResponseError("response.choices[0].message.parsed is None.")

        # get logprobs
        lp_result = None
        if payload.logprobs:
            lp_data = response.choices[0].logprobs
            if lp_data is None:
                raise InvalidResponseError("logprobs set to True but response.choices[0].logprobs is None")
            
            lp_resp_data = lp_data.content if lp_data.content is not None else lp_data.refusal

            if lp_resp_data is None:
                raise InvalidResponseError("logprobs set to True but both response.choices[0].logprobs.content and response.choices[0].logprobs.refusal are None")
        
            lp_result = collect_openai_logprobs(lp_resp_data)

        # update tokens tracker
        if response.usage is None:
            raise InvalidResponseError("response.usage is None.")

        used_tokens = response.usage.total_tokens
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        await oai_tokens_tracker.add_tokens(payload.model_name, used_tokens)

        # check if we've exceeded limits
        if oai_tokens_tracker.exceeded_model_limits():
            raise TokenLimitException(f"Exceeded token limit: current_model={payload.model_name}; token_tracker={oai_tokens_tracker.get_total_tokens_used()}")

        return _WrappedReturn(
            content=response_parsed,
            tokens_used=used_tokens,
            logprobs=lp_result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
