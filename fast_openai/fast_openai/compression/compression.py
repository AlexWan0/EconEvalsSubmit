from cashews.commands import Command
from cashews.backends.interface import Backend
from typing import Any
import pickle
import gzip
from dataclasses import dataclass, field

from .hash_keys import sha256_hash


def compress_object(
        obj: Any,
        protocol: int = pickle.HIGHEST_PROTOCOL,
        **kwargs
    ) -> bytes:

    serialized = pickle.dumps(obj, protocol=protocol)
    compressed = gzip.compress(serialized, **kwargs)
    return compressed

def decompress_object(
        data: bytes,
        **kwargs
    ) -> Any:

    decompressed = gzip.decompress(data, **kwargs)
    obj = pickle.loads(decompressed)
    return obj

@dataclass
class Compressed:
    data: bytes
    decompress_kwargs: dict[str, Any] = field(default_factory=dict)

def parse_middleware_args(
        key: str,
        sep: str = ':'
    ) -> dict[str, str]:

    sep_idx = key.rfind(sep)
    suffix = key[sep_idx + 1:]

    result = {}
    for pair in suffix.split(','):
        k, v = pair.split('=')
    
        result[k] = v
    
    return result

def get_compression_middleware(
        # compress_keys: bool = False,
        # compress_values: bool = False,
        compress_kwargs: dict[str, Any] = {},
        decompress_kwargs: dict[str, Any] = {}
    ):

    async def compression_middleware(call, cmd: Command, backend: Backend, *args, **kwargs):
        # first hash the key if enabled
        # print('hash', kwargs, cmd)
        # no_hash_key = kwargs["key"].endswith("hash_keys=false") # python bools get converted to lowercase true/false
        # yes_hash_key = kwargs["key"].endswith("hash_keys=true")
        # assert no_hash_key or yes_hash_key, f'{kwargs["key"]} must end with hash_keys=true or hash_keys=false'
        if not isinstance(kwargs["key"], str):
            raise TypeError("kwargs['key'] must be a str")

        middleware_args = parse_middleware_args(kwargs["key"])
        # TODO: add exceptions here: validate keys & parsing

        compress_values = (middleware_args["compress_values"] == 'true')
        compress_keys = False # does not work yet

        if middleware_args["hash_keys"] == 'true':
            kwargs["key"] = sha256_hash(kwargs["key"])

        # then do compression stuff
        if cmd == Command.SET:
            # print('compress', kwargs, cmd)
            if compress_values:
                kwargs["value"] = Compressed(
                    data=compress_object(kwargs["value"], **compress_kwargs),
                    decompress_kwargs=decompress_kwargs
                )

            if compress_keys:
                kwargs["key"] = Compressed(
                    data=compress_object(kwargs["key"], **compress_kwargs),
                    decompress_kwargs=decompress_kwargs
                )

            return await call(*args, **kwargs)

        elif cmd == Command.GET:
            # print('decompress', kwargs, cmd)
            orig_key = kwargs["key"]
            
            # if we expect that cache keys are going to be compressed
            # we need to compress first
            if compress_keys:
                kwargs["key"] = Compressed(
                    data=compress_object(orig_key, **compress_kwargs),
                    decompress_kwargs=decompress_kwargs
                )

            # ...then retrieve
            retrieved_val = await call(*args, **kwargs)
            # print('decompress raw', retrieved_val, cmd)

            if isinstance(retrieved_val, Compressed):
                retrieved_val = decompress_object(
                    retrieved_val.data,
                    **retrieved_val.decompress_kwargs
                )
            
            # then undo that compression
            if compress_keys:
                kwargs["key"] = orig_key

            return retrieved_val

        return await call(*args, **kwargs)

    return compression_middleware
