from hashlib import sha256
from typing import Callable, Any, Optional
from functools import wraps
import pickle
import os
import zstandard as zstd
import inspect
import logging
import base64
logger = logging.getLogger(__name__)


MAX_ARG_SIZE = 24
MAX_ARGS_SIZE = 80
MAX_CODE_SIZE = 8
MAX_FN_SIZE = 128


def _concise_sha256(x: object) -> str:
    hash_bytes = sha256(pickle.dumps(x)).digest()
    return base64.urlsafe_b64encode(hash_bytes).decode(encoding='utf-8').rstrip('=')

def _short_ident(x: object) -> str:
    '''
    prefixed with "~" if it's the real value, otherwise it's a base64 encoded hash
    '''
    if isinstance(x, str) and len(x) <= (MAX_ARG_SIZE - 1):
        return f"~{x}"
    
    if (isinstance(x, int) or isinstance(x, bool)) and (len(str(x)) <= (MAX_ARG_SIZE - 1)):
        return f"~{x}"
    
    return _concise_sha256(x)[:MAX_ARG_SIZE]

def _args_str(func: Callable, args: tuple[Any], kwargs: dict[str, Any]) -> str:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)

    parts = []
    for name, value in bound.arguments.items():
        parts.append(f"{name}={_short_ident(value)}")
    args_str = "@".join(parts)

    if len(args_str) > MAX_ARGS_SIZE:
        return _concise_sha256(args_str.encode("utf-8"))[:MAX_ARGS_SIZE]

    return args_str

def _save_compressed(obj: object, fp: str, level: int = 3, threads: int = 4):
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    with zstd.open(fp, 'wb', cctx=cctx) as f_out:
        pickle.dump(obj, f_out)

def _load_compressed(fp: str) -> object:
    with zstd.open(fp, 'rb') as f_in:
        return pickle.load(f_in)

FuncType = Callable[..., Any]
def cached_func(
        cache_dir: str = '.cache/funcs',
        zstd_level: int = 3,
        zstd_threads: int = 4,
        fn_ident_override: Optional[str] = None
    ) -> Callable[[FuncType], FuncType]:
    """Decorator to cache function outputs. Keys are encoded in the file name.

    Cache key includes the function name, function code, and arguments. If function inputs/names are concise, it tries to transparently display them in the file name directly; otherwise, it hashes them.

    Args:
        cache_dir (str): Path of the folder to store the cached functions.
        zstd_level (int): Compression level to use.
        zstd_threads (int): Number of threads to use for compression.
        fn_ident_override (Optional[str]): If provided, this string is used as the file name for the cache, overriding the default generated identifier.
    """

    def _cached_dec(
            func: FuncType,
        ) -> FuncType:
        @wraps(func)
        def _wrap(*args, **kwargs) -> Any:
            logger.info(f'cached_func ({func.__name__}): making key')
            
            # if an override is provided, use it directly.
            if fn_ident_override is not None:
                fn_ident = fn_ident_override
            else:
                code_hash = _concise_sha256(inspect.getsource(func).encode())[:MAX_CODE_SIZE]
                fn_ident = f'{func.__module__}.{func.__name__}+{code_hash}+{_args_str(func, args, kwargs)}'
                if len(fn_ident) > MAX_FN_SIZE:
                    fn_ident = _concise_sha256(fn_ident.encode('utf-8'))[:MAX_FN_SIZE]

            # setup paths
            os.makedirs(cache_dir, exist_ok=True)
            fp = os.path.join(cache_dir, f'{fn_ident}.pkl.zst')

            if os.path.isfile(fp):
                logger.info(f'cached_func ({func.__name__}): found cached at {fp}')
                return _load_compressed(fp)
        
            res = func(*args, **kwargs)

            logger.info(f'cached_func ({func.__name__}): saving')
            _save_compressed(
                res,
                fp,
                level=zstd_level,
                threads=zstd_threads
            )

            logger.info(f'cached_func ({func.__name__}): saved to {fp}')

            return res
        
        return _wrap

    return _cached_dec