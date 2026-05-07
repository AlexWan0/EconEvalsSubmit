from openai.types.chat import ChatCompletionTokenLogprob
from dataclasses import dataclass, field
    

# utils for storing logprobs from openai
@dataclass
class TopLogprobs():
    '''
    All lists must be the same size
    '''
    tokens: list[str] = field(default_factory=list)
    seq_bytes: list[list[int] | None] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class Logprobs():
    '''
    All lists must be the same size
    '''
    tokens: list[str] = field(default_factory=list)
    seq_bytes: list[list[int] | None] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[TopLogprobs | None] = field(default_factory=list)


def collect_openai_logprobs(data: list[ChatCompletionTokenLogprob]) -> Logprobs:
    result = Logprobs()

    # for each token in the response
    for lp_single in data:
        # retrieve the tokens/logprobs data for that token
        result.tokens.append(lp_single.token)
        result.seq_bytes.append(lp_single.bytes)
        result.logprobs.append(lp_single.logprob)
        
        # retrieve the top logprobs data for that token if it exists
        top_logprobs: TopLogprobs | None = None
        if len(lp_single.top_logprobs) > 0:
            top_logprobs = TopLogprobs()

            for lp_top in lp_single.top_logprobs:
                top_logprobs.tokens.append(lp_top.token)
                top_logprobs.seq_bytes.append(lp_top.bytes)
                top_logprobs.logprobs.append(lp_top.logprob)

        result.top_logprobs.append(top_logprobs)

    return result
