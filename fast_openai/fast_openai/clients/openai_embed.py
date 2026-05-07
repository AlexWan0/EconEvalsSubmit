from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import json
from dataclasses import dataclass
import tiktoken
from tqdm import tqdm

from ..base import Serializable, JSONType
from .base import APIClient, _WrappedReturn, RequestArgs
from .auth import OPENAI_API_KEY
from .exceptions import RequestException, TokenLimitException, InvalidResponseError
from .token_tracking import oai_tokens_tracker


EmbedType = list[float]

@dataclass
class EmbedInput(Serializable):
    model_input: str
    model_name: str
    max_length: int = 8192 - 16 # max length minus some buffer

    def to_json(self) -> JSONType:
        return {
            'model_input': self.model_input,
            'model_name': self.model_name,
        }


class OpenAIEmbedAPI(APIClient[EmbedInput, EmbedType]):
    async def _call(
            self,
            payload: EmbedInput,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[EmbedType]:

        # truncate input
        encoding = tiktoken.encoding_for_model(payload.model_name)
        enc_model_input = encoding.encode(payload.model_input, disallowed_special=())

        if len(enc_model_input) > payload.max_length:
            trunc_model_input = encoding.decode(enc_model_input[:payload.max_length])
        else:
            trunc_model_input = payload.model_input

        # make request
        request_body = {
            "input": trunc_model_input,
            "model": payload.model_name,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        async with session.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=request_body,
            timeout=ClientTimeout(total=request_args.post_timeout)
        ) as resp:
            resp_body = await resp.json()

            if resp.status != 200:
                raise RequestException(f"{resp.status}: {resp_body}")
            
        # get parsed response
        response_val = resp_body["data"][0]["embedding"]

        # update tokens tracker
        used_tokens = resp_body["usage"]["total_tokens"]
        await oai_tokens_tracker.add_tokens(payload.model_name, used_tokens)

        # check if we've exceeded limits
        if oai_tokens_tracker.exceeded_model_limits():
            raise TokenLimitException(f"Exceeded token limit: current_model={payload.model_name}; token_tracker={oai_tokens_tracker.get_total_tokens_used()}")

        return _WrappedReturn(
            content=response_val,
            tokens_used=used_tokens
        )
