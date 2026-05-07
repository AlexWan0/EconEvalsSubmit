from typing import TypeVar, Generic, Type, TypedDict, Literal
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import json
from dataclasses import dataclass

from ..base import Serializable, JSONType
from .logprobs import collect_openai_logprobs
from .base import APIClient, _WrappedReturn, RequestArgs
from .auth import OPENAI_API_KEY
from .exceptions import RequestException, TokenLimitException, InvalidResponseError
from .token_tracking import oai_tokens_tracker
from ..utils import Turn, parse_model_with_params


@dataclass
class BasicInput(Serializable):
    model_input: list[Turn]
    model_name: str
    max_tokens: int
    temperature: float
    logprobs: bool = False
    top_logprobs: int | None = None
    cache_flag: str = ''

    def to_json(self) -> JSONType:
        data = {
            'model_input': self.model_input,
            'model_name': self.model_name,
            'max_tokens': str(self.max_tokens),
            'temperature': self.temperature,
            'logprobs': self.logprobs,
            'cache_flag': self.cache_flag
        }

        if self.top_logprobs is not None:
            data['top_logprobs'] = self.top_logprobs

        return data

def is_reasoning_model(name: str) -> bool:
    return name.startswith(("o1", "o3", "o4", "gpt-5"))

from openai.types.chat import ChatCompletion

class OpenAIAPI(APIClient[BasicInput, str]):
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    @property
    def api_key(self):
        return OPENAI_API_KEY

    async def _call(
            self,
            payload: BasicInput,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[str]:

        base_model, extra_params = parse_model_with_params(payload.model_name)
        is_reasoning = is_reasoning_model(base_model)

        request_body = {
            "model": base_model,
            "messages": payload.model_input,
            "temperature": payload.temperature,
            "logprobs": payload.logprobs,
        }

        # TODO: there should be a better way to handle this...
        if is_reasoning:
            request_body["max_completion_tokens"] = payload.max_tokens
        else:
            request_body["max_tokens"] = payload.max_tokens

        if payload.top_logprobs is not None:
            request_body["top_logprobs"] = payload.top_logprobs

        request_body.update(extra_params)

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
        response = ChatCompletion.model_validate(resp_body)

        # get parsed response
        response_str = response.choices[0].message.content
        
        if response_str is None:
            raise InvalidResponseError("response.choices[0].message.content is None.")

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
        await oai_tokens_tracker.add_tokens(base_model, used_tokens)

        # check if we've exceeded limits
        if oai_tokens_tracker.exceeded_model_limits():
            raise TokenLimitException(f"Exceeded token limit: current_model={payload.model_name}; token_tracker={oai_tokens_tracker.get_total_tokens_used()}")

        # try get num reasoning tokens
        compl_token_details = response.usage.completion_tokens_details
        reasoning_tokens = None
        if compl_token_details is not None:
            reasoning_tokens = compl_token_details.reasoning_tokens

        return _WrappedReturn(
            content=response_str,
            tokens_used=used_tokens,
            logprobs=lp_result,
            reasoning_tokens=reasoning_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
