from typing import Optional, TypeVar, Type
from dataclasses import dataclass
import yaml
from fast_openai import RequestArgs

from ..utils import CheckTypesInit


DEFAULT_REQUEST_ARGS = RequestArgs(
    use_cache=True,
    hash_keys=True,
    num_retries=2,
    post_timeout=1 * 60,
    total_timeout=1 * 60,
    connect_timeout=10,
    retries_delay_start=1
)

EndpointConfigType = TypeVar('EndpointConfigType', bound='EndpointConfig')
EndpointMapType = dict[str, EndpointConfigType]

@dataclass
class EndpointConfig(CheckTypesInit):
    model_str: str
    max_tokens: int
    temperature: float
    parallel: int
    
    request_args: RequestArgs = DEFAULT_REQUEST_ARGS
    endpoints: Optional[list[dict]] = None # for self-hosted end-points I think

    @classmethod
    def load_yaml_map(cls: Type[EndpointConfigType], fp: str) -> EndpointMapType[EndpointConfigType]:
        with open(fp, "r") as f:
            endpoints_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

        assert isinstance(endpoints_kwargs, dict), f'incorrect type loaded: {type(endpoints_kwargs)}'

        loaded_map: EndpointMapType = {}
        for name, kwargs in endpoints_kwargs.items():
            loaded_map[name] = cls(
                **kwargs
            )

        return endpoints_kwargs
