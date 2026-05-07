import hashlib
import pickle
from typing import Any, Callable
from cashews.commands import Command
from cashews.backends.interface import Backend 


def sha256_hash(obj: Any) -> str:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(data, usedforsecurity=False).hexdigest()

# async def hash_keys_middleware(call: Callable, cmd: Command, backend: Backend, *args, **kwargs):
#     print('hash', kwargs, cmd)

#     no_hash_key = kwargs["key"].endswith("hash_keys=false") # python bools get converted to lowercase true/false
#     yes_hash_key = kwargs["key"].endswith("hash_keys=true")
#     assert no_hash_key or yes_hash_key, f'{kwargs["key"]} must end with hash_keys=true or hash_keys=false'

#     if no_hash_key:
#         return await call(*args, **kwargs)

#     kwargs["key"] = sha256_hash(kwargs["key"])

#     return await call(*args, **kwargs)
