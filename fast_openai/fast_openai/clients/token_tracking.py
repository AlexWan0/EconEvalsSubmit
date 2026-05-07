import asyncio
import logging

logger = logging.getLogger(__name__)


class TokensTracker:
    def __init__(
            self,
            model_limits: dict[str, int] = {},
            default_limit: int = 200_000_000,
            milestone_interval: int = 10_000_000
        ):

        self.tokens_used: dict[str, int] = {}
        self.model_limits = model_limits
        self.default_limit = default_limit

        self.lock = asyncio.Lock()

        self.next_milestone: dict[str, int] = {}
        self.milestone_interval = milestone_interval

    async def add_tokens(self, model_name: str, tokens: int) -> dict[str, int]:
        async with oai_tokens_tracker.lock:
            if model_name not in self.tokens_used:
                self.tokens_used[model_name] = 0
                self.next_milestone[model_name] = self.milestone_interval
            
            self.tokens_used[model_name] += tokens

            curr_used = self.tokens_used[model_name]

            if curr_used >= self.next_milestone[model_name]:
                self.next_milestone[model_name] = ((curr_used // self.milestone_interval) + 1) * self.milestone_interval

                print(f'[TokensTracker] used {curr_used} tokens for model {model_name}')

        return self.tokens_used

    def display_tokens_used(self):
        print(self.tokens_used)
        logger.info(self.tokens_used)

    def get_total_tokens_used(self) -> int:
        return sum(self.tokens_used.values())
    
    def exceeded_model_limits(self) -> bool:
        for model_name, limit in self.model_limits.items():
            model_used_tokens = self.tokens_used.get(model_name, 0)
            if model_used_tokens > limit or model_used_tokens > self.default_limit:
                return True

        return False

oai_tokens_tracker = TokensTracker(
    model_limits={
        'gpt-4o-mini': 50_000_000_000,
        'gpt-4.1-mini': 1_000_000_000,
        'gpt-4.1-nano': 1_000_000_000,
        'gpt-4.1': 50_000_000,
        'gpt-4o': 50_000_000,
        'text-embedding-3-small': 10_000_000_000
    },
    default_limit=10_000_000_000
)
