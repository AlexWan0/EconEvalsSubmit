import os
from cashews import Cache

from .compression import get_compression_middleware


CACHE_PATH = os.getenv('FAST_OPENAI_CACHE_DIR', '.cache/fast_openai_cashews')
CACHE_SIZE_LIMIT_GB = os.getenv('FAST_OPENAI_CACHE_SIZE_LIMIT_GB', 10000)
CACHE_N_SHARDS = os.getenv('FAST_OPENAI_CACHE_N_SHARDS', 16)

GB = 1_073_741_824

def get_cashews_cache(
        with_middlewares: bool = True
    ) -> Cache:

    cache = Cache()
    cache.setup(
        f"disk://?directory={CACHE_PATH}",
        size_limit=CACHE_SIZE_LIMIT_GB * GB,
        shards=CACHE_N_SHARDS,
        eviction_policy="least-recently-stored",
        middlewares=(
            get_compression_middleware(),
        ) if with_middlewares else tuple()
    )

    return cache

cache = get_cashews_cache()
