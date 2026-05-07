from fast_openai.base import Result
from fast_openai.utils import Turn
from dataclasses import is_dataclass, fields
from typeguard import check_type, CollectionCheckStrategy
from typing import Any


def validate_dataclass(obj: Any):
    if not is_dataclass(obj):
        raise TypeError(f"{obj!r} is not a dataclass instance")

    for fld in fields(obj):
        value = getattr(obj, fld.name)
        try:
            check_type(
                value,
                fld.type,
                collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS
            )

        except TypeError as exc:
            raise TypeError(
                f"field {fld.name} expected type {fld.type!r}, but got value {value!r} of type {type(value)!r}"
            ) from exc

class CheckTypesInit():
    def __post_init__(self):
        validate_dataclass(self)

def run_dummy(model_inputs: list[list[Turn]], **kwargs) -> list[Result[str]]:
    dummy_outputs: list[str] = []

    for inp_convo in model_inputs:
        # convo_str = '\n\n'.join([f'{x["role"]}: {x["content"]}' for x in inp_convo])
        convo_str = str(inp_convo)

        sep = '\n' + '=*' * 80 + '\n'
        
        dummy_outputs.append(sep.join([
            f'{k} = {v}'
            for k, v in {
                'CONVERSATION': convo_str,
                **kwargs
            }.items()
        ]))

    return [
        Result(
            output=x,
            error=None,
            tokens_used=0,
            from_cache=False,
            logprobs=None
        )
        for x in dummy_outputs
    ]