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


@dataclass
class ModerationInput(Serializable):
    model_input: str
    model_name: str

    def to_json(self) -> JSONType:
        return {
            'model_input': self.model_input,
            'model_name': self.model_name
        }

@dataclass
class ModerationCategories:
    sexual: bool
    hate: bool
    harassment: bool
    self_harm: bool
    sexual_minors: bool
    hate_threatening: bool
    violence_graphic: bool
    self_harm_intent: bool
    self_harm_instructions: bool
    harassment_threatening: bool
    violence: bool

@dataclass
class ModerationCategoriesScores:
    sexual: float
    hate: float
    harassment: float
    self_harm: float
    sexual_minors: float
    hate_threatening: float
    violence_graphic: float
    self_harm_intent: float
    self_harm_instructions: float
    harassment_threatening: float
    violence: float

@dataclass
class ModerationOutput:
    flagged: bool
    categories: ModerationCategories
    categories_scores: ModerationCategoriesScores

class OpenAIModerationAPI(APIClient[ModerationInput, ModerationOutput]):
    async def _call(
            self,
            payload: ModerationInput,
            session: ClientSession,
            request_args: RequestArgs
        ) -> _WrappedReturn[ModerationOutput]:

        # make request
        request_body = {
            "input": payload.model_input,
            "model": payload.model_name,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        async with session.post(
            "https://api.openai.com/v1/moderations",
            headers=headers,
            json=request_body,
            timeout=ClientTimeout(total=request_args.post_timeout)
        ) as resp:
            resp_body = await resp.json()

            if resp.status != 200:
                raise RequestException(f"{resp.status}: {resp_body}")

        first_result = resp_body["results"][0]

        flagged = first_result["flagged"]

        raw_cats = first_result["categories"]
        categories = ModerationCategories(
            sexual=raw_cats["sexual"],
            hate=raw_cats["hate"],
            harassment=raw_cats["harassment"],
            self_harm=raw_cats["self-harm"],
            sexual_minors=raw_cats["sexual/minors"],
            hate_threatening=raw_cats["hate/threatening"],
            violence_graphic=raw_cats["violence/graphic"],
            self_harm_intent=raw_cats["self-harm/intent"],
            self_harm_instructions=raw_cats["self-harm/instructions"],
            harassment_threatening=raw_cats["harassment/threatening"],
            violence=raw_cats["violence"]
        )

        raw_scores = first_result["category_scores"]
        categories_scores = ModerationCategoriesScores(
            sexual=raw_scores["sexual"],
            hate=raw_scores["hate"],
            harassment=raw_scores["harassment"],
            self_harm=raw_scores["self-harm"],
            sexual_minors=raw_scores["sexual/minors"],
            hate_threatening=raw_scores["hate/threatening"],
            violence_graphic=raw_scores["violence/graphic"],
            self_harm_intent=raw_scores["self-harm/intent"],
            self_harm_instructions=raw_scores["self-harm/instructions"],
            harassment_threatening=raw_scores["harassment/threatening"],
            violence=raw_scores["violence"]
        )

        output = ModerationOutput(
            flagged=flagged,
            categories=categories,
            categories_scores=categories_scores
        )

        return _WrappedReturn(
            content=output,
            tokens_used=0,
            logprobs=None
        )
