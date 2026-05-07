from typing import Any
from dataclasses import dataclass
from typeguard import typechecked, check_type, CollectionCheckStrategy
from glob import glob
import os
import json
import logging

from .base import Conversation
from ..utils import CheckTypesInit

logger = logging.getLogger(__name__)


@dataclass
class ModelAnswer(CheckTypesInit):
    uid: str
    ans_id: str
    model: str
    messages: Conversation
    metadata: dict[str, Any]
    tstamp: float | int

@dataclass
class ModelAnswerMap(CheckTypesInit):
    _data: dict[str, dict[str, ModelAnswer]]

    @classmethod
    def load_dir(cls, folder_path: str) -> 'ModelAnswerMap':
        filenames = glob(os.path.join(folder_path, "*.jsonl"))
        filenames.sort()
        model_answers: dict[str, dict[str, ModelAnswer]] = {}

        for filename in filenames:
            model_name = os.path.basename(filename).removesuffix('.jsonl')

            answer: dict[str, ModelAnswer] = {}
            with open(filename) as fin:
                for line in fin:
                    line = json.loads(line)

                    formatted_messages: Conversation = []

                    for t in line['messages']:
                        content = t['content']

                        if isinstance(content, dict):
                            content = content['answer']

                        assert isinstance(content, str)
                        assert t['role'] in {'user', 'assistant', 'system'}

                        formatted_messages.append(
                            {
                                'role': t['role'],
                                'content': content
                            }
                        )

                    line['messages'] = formatted_messages

                    answer[line["uid"]] = ModelAnswer(**line)
            
            model_answers[model_name] = answer

        return cls(
            _data=model_answers
        )

    def get_models(self) -> list[str]:
        return list(self._data.keys())
    
    def get_question_uids(self, model_name: str) -> list[str]:
        return list(self._data[model_name].keys())

    def get_response(self, model_name: str, question_uid: str) -> ModelAnswer:
        return self._data[model_name][question_uid]

@dataclass
class _LMSYSArenaQuestion(CheckTypesInit):
    uid: str
    category: str
    subcategory: str
    prompt: str

@dataclass
class QuestionData(CheckTypesInit):
    uid: str
    category: str
    subcategory: str
    convo: Conversation # for single turn this is just the human prompt

    def __post_init__(self):
        super().__post_init__()
        
        assert self.convo[-1]['role'] == 'user'

    @classmethod
    def load_auto(cls, fp: str) -> 'list[QuestionData]':
        '''
        Infers lmsys format vs multiturn format based on first item.
        '''
        with open(fp, 'r') as f_in:
            data = json.loads(next(iter(f_in)))

        assert isinstance(data, dict)

        if 'convo' in data:
            return cls.load_multi_turn(fp)
        
        elif 'prompt' in data:
            return cls.load_single_turn(fp)
        
        else:
            raise ValueError(f'could not infer question data format for item with keys {data.keys()}')

    @classmethod
    def load_single_turn(cls, fp: str) -> 'list[QuestionData]':
        '''
        For loading from the original lmsys format.
        '''
        logger.info('QuestionData: loading single turn')
        res: list[QuestionData] = []
        with open(fp, 'r') as f_in:
            for line in f_in:
                if len(line.strip()) == 0:
                    continue

                loaded = _LMSYSArenaQuestion(
                    **json.loads(line)
                )

                res.append(QuestionData(
                    uid=loaded.uid,
                    category=loaded.category,
                    subcategory=loaded.subcategory,
                    convo=[
                        {'role': 'user', 'content': loaded.prompt}
                    ]
                ))
    
        return res

    @classmethod
    def load_multi_turn(cls, fp: str) -> 'list[QuestionData]':
        logger.info('QuestionData: loading multi turn')
        res: list[QuestionData] = []
        with open(fp, 'r') as f_in:
            for line in f_in:
                if len(line.strip()) == 0:
                    continue

                loaded = cls(
                    **json.loads(line)
                )

                res.append(loaded)

        return res
