from typing import Optional, TypeVar, Type, ClassVar, Any
from dataclasses import dataclass
import yaml
from typeguard import typechecked, CollectionCheckStrategy

from .endpoint import EndpointConfig
from ..utils import CheckTypesInit


ArenaConfigType = TypeVar('ArenaConfigType', bound='ArenaConfig')

@dataclass
class ArenaConfig(CheckTypesInit):
    judge_endpoint: EndpointConfig
    regex_patterns: list[str]
    prompt_template: str
    model_list: list[str]
    bench_name: str

    reference: Optional[dict[str, Any]] = None

    expected_prompt_keys: ClassVar[frozenset[str]] = frozenset([
        'QUESTION',
        'ANSWER_A',
        'ANSWER_B',
    ])

    @classmethod
    def load_yaml(cls: Type[ArenaConfigType], fp: str, **kwargs: Any) -> ArenaConfigType:
        '''
        load arena config in configs/arena/...
        '''
        config_kwargs = {}
        with open(fp, "r") as f:
            config_kwargs = yaml.load(f, Loader=yaml.SafeLoader) or {}

        assert isinstance(config_kwargs, dict), type(config_kwargs)

        # Fill only missing keys from kwargs (do not override keys present in the file).
        for key, value in kwargs.items():
            if key not in config_kwargs:
                config_kwargs[key] = value
        
        config_kwargs['judge_endpoint'] = EndpointConfig(
            **config_kwargs['judge_endpoint']
        )

        return cls(
            **config_kwargs
        )

    def __post_init__(self):
        super().__post_init__()

        for key in self.expected_prompt_keys:
            assert key in self.prompt_template, f'{key} not in prompt_template'
        

@dataclass
class MultiturnArenaConfig(ArenaConfig):
    expected_prompt_keys: ClassVar[frozenset[str]] = frozenset([
        'QUESTION',
        'ANSWER_A',
        'ANSWER_B',
        'CONVERSATION_HISTORY'
    ])
