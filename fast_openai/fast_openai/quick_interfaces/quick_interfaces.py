from typing import Type
import asyncio
import os

# Note: anything that we'd need to import in quick_inferfaces, we should also
# make sure to put in the root __init__.py
# (except for stuff that's just used as types)
from ..base import Result
from ..sized_iterable import MaybeSizedIter
from ..clients import (
    StructType,
    OPENAI_EMBEDDING_LIMIT,
    CacheOnlyClient,
    OutputValidation,
    ValidationFunctionType,
    VALIDATION_FUNCTION_DEFAULT
)
from ..utils import Turn
from .. import (
    ResultsCollectorArgs,
    RequestArgs,
    OpenAIAPI,
    OpenAIStructAPI,
    OpenAIEmbedAPI,
    OpenAIModerationAPI,
    ModerationInput,
    ModerationOutput,
    TogetherAPI,
    TogetherStructAPI,
    GoogleGenAIAPI,
    StructInput,
    BasicInput,
    EmbedInput,
    ConcurrentRunner,
    map_maybe_sized,
    str_to_convo,
)


ConvoType = list[Turn]

def run_openai_struct(
        model_inputs: MaybeSizedIter[ConvoType],
        response_format: Type[StructType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False
    ) -> list[Result[StructType]]:

    def _make_input(ex_model_input: ConvoType):
        return StructInput(
            model_input=ex_model_input,
            response_format=response_format,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIStructAPI[response_format]() if not cache_only else CacheOnlyClient[StructInput[StructType], response_format](),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(runner.run_to_list(inputs))

    loop.close()

    return responses

def run_openai_struct_str(
        model_inputs: MaybeSizedIter[str],
        response_format: Type[StructType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False
    ) -> list[Result[StructType]]:

    return run_openai_struct(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        response_format=response_format,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        cache_only=cache_only
    )

def run_openai_struct_to_disk(
        model_inputs: MaybeSizedIter[ConvoType],
        response_format: Type[StructType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        output_dir: str = '.cache/struct_output'
    ) -> str:

    os.makedirs(output_dir, exist_ok=True)

    data_fp = os.path.join(output_dir, 'data.bin')
    pointer_fp = os.path.join(output_dir, 'pointer.pkl')

    def _make_input(ex_model_input: ConvoType):
        return StructInput(
            model_input=ex_model_input,
            response_format=response_format,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIStructAPI[response_format](),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(
        runner.run(
            inputs,
            ResultsCollectorArgs(
                how='disk',
                kwargs={
                    'file_path': data_fp
                }
            )
        )
    )

    loop.close()

    responses.save(pointer_fp)

    return pointer_fp

def run_openai_struct_str_to_disk(
        model_inputs: MaybeSizedIter[str],
        response_format: Type[StructType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        output_dir: str = '.cache/struct_output'
    ) -> str:

    return run_openai_struct_to_disk(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        response_format=response_format,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        output_dir=output_dir
    )

def run_openai(
        model_inputs: MaybeSizedIter[ConvoType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False,
        validation_function: ValidationFunctionType[str] = VALIDATION_FUNCTION_DEFAULT
    ) -> list[Result[str]]:

    def _make_input(ex_model_input: ConvoType):
        return BasicInput(
            model_input=ex_model_input,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIAPI() if not cache_only else CacheOnlyClient[BasicInput, str](),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(
        runner.run_to_list(
            inputs,
            output_validation=OutputValidation(
                verify_function=validation_function
            )
        )
    )

    loop.close()

    return responses

def run_openai_str(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False,
        validation_function: ValidationFunctionType[str] = VALIDATION_FUNCTION_DEFAULT
    ) -> list[Result[str]]:

    return run_openai(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        cache_only=cache_only,
        validation_function=validation_function
    )

def run_openai_to_disk(
        model_inputs: MaybeSizedIter[ConvoType],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        output_dir: str = '.cache/output',
        in_progress_fp: str | None = '.cache/output/pointer.pkl.incomplete',
        in_progress_interval: int = 100_000
    ) -> str:

    os.makedirs(output_dir, exist_ok=True)

    data_fp = os.path.join(output_dir, 'data.bin')
    pointer_fp = os.path.join(output_dir, 'pointer.pkl')

    def _make_input(ex_model_input: ConvoType):
        return BasicInput(
            model_input=ex_model_input,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIAPI(),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(
        runner.run(
            inputs,
            ResultsCollectorArgs(
                how='disk',
                kwargs={
                    'file_path': data_fp,
                    'in_progress_fp': in_progress_fp,
                    'in_progress_interval': in_progress_interval
                }
            )
        )
    )

    loop.close()

    responses.save(pointer_fp)

    return pointer_fp

def run_openai_str_to_disk(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        output_dir: str = '.cache/output',
        in_progress_fp: str | None = '.cache/output/pointer.pkl.incomplete',
        in_progress_interval: int = 100_000
    ) -> str:

    return run_openai_to_disk(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        output_dir=output_dir,
        in_progress_fp=in_progress_fp,
        in_progress_interval=in_progress_interval
    )

def run_openai_embed_to_disk(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'text-embedding-3-small',
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(
            rate_limit_rpm=OPENAI_EMBEDDING_LIMIT
        ),
        pbar_name: str = '{model_name}',
        output_dir: str = '.cache/embed_output',
        in_progress_fp: str | None = '.cache/embed_output/pointer.pkl.incomplete',
        in_progress_interval: int = 100_000
    ) -> str:

    os.makedirs(output_dir, exist_ok=True)

    data_fp = os.path.join(output_dir, 'data.bin')
    pointer_fp = os.path.join(output_dir, 'pointer.pkl')

    def _make_input(ex_model_input: str) -> EmbedInput:
        return EmbedInput(
            model_input=ex_model_input,
            model_name=model_name,
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIEmbedAPI(),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(
        runner.run(
            inputs,
            ResultsCollectorArgs(
                how='disk',
                kwargs={
                    'file_path': data_fp,
                    'in_progress_fp': in_progress_fp,
                    'in_progress_interval': in_progress_interval
                }
            )
        )
    )

    loop.close()

    responses.save(pointer_fp)

    return pointer_fp

def run_openai_moderation(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'omni-moderation-latest',
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(
            rate_limit_rpm=OPENAI_EMBEDDING_LIMIT
        ),
        pbar_name: str = '{model_name}',
    ) -> list[Result[ModerationOutput]]:

    def _make_input(ex_model_input: str) -> ModerationInput:
        return ModerationInput(
            model_input=ex_model_input,
            model_name=model_name,
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIModerationAPI(),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(runner.run_to_list(inputs))

    loop.close()

    return responses

def run_openai_embed(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'text-embedding-3-small',
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(
            rate_limit_rpm=OPENAI_EMBEDDING_LIMIT
        ),
        pbar_name: str = '{model_name}',
    ) -> list[Result[list[float]]]:

    def _make_input(ex_model_input: str) -> EmbedInput:
        return EmbedInput(
            model_input=ex_model_input,
            model_name=model_name,
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=OpenAIEmbedAPI(),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(runner.run_to_list(inputs))

    loop.close()

    return responses

def run_google(
        model_inputs: MaybeSizedIter[ConvoType],
        model_name: str = 'gemini-2.0-flash',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False,
        validation_function: ValidationFunctionType[str] = VALIDATION_FUNCTION_DEFAULT
    ) -> list[Result[str]]:

    def _make_input(ex_model_input: ConvoType):
        return BasicInput(
            model_input=ex_model_input,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=GoogleGenAIAPI() if not cache_only else CacheOnlyClient[BasicInput, str](),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(
        runner.run_to_list(
            inputs,
            output_validation=OutputValidation(
                verify_function=validation_function
            )
        )
    )

    loop.close()

    return responses

def run_google_str(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'gemini-2.0-flash',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False
    ) -> list[Result[str]]:

    return run_google(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        cache_only=cache_only
    )


def run_together(
        model_inputs: MaybeSizedIter[ConvoType],
        model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False
    ) -> list[Result[str]]:

    def _make_input(ex_model_input: ConvoType):
        return BasicInput(
            model_input=ex_model_input,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            cache_flag=cache_flag
        )
    
    inputs = map_maybe_sized(
        _make_input,
        model_inputs
    )

    runner = ConcurrentRunner(
        client=TogetherAPI() if not cache_only else CacheOnlyClient[BasicInput, str](),
        n_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name.format(model_name=model_name)
    )

    loop = asyncio.new_event_loop()

    responses = loop.run_until_complete(runner.run_to_list(inputs))

    loop.close()

    return responses

def run_together_str(
        model_inputs: MaybeSizedIter[str],
        model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False
    ) -> list[Result[str]]:

    return run_together(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        cache_only=cache_only
    )

def run_auto(
        model_inputs: MaybeSizedIter[ConvoType],
        full_model_name: str = 'openai/gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False,
        validation_function: ValidationFunctionType[str] = VALIDATION_FUNCTION_DEFAULT
    ) -> list[Result[str]]:

    # peel off first value to get provider
    parsed_model_name = full_model_name.split('/')
    provider = parsed_model_name[0]
    model_name = '/'.join(parsed_model_name[1:])

    if validation_function != VALIDATION_FUNCTION_DEFAULT:
        if provider not in ['openai', 'google']:
            raise NotImplementedError()

    if provider == 'openai':
        return run_openai(
            model_inputs=model_inputs,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            num_workers=num_workers,
            request_args=request_args,
            pbar_name=pbar_name,
            cache_flag=cache_flag,
            cache_only=cache_only,
            validation_function=validation_function
        )

    elif provider == 'google':
        return run_google(
            model_inputs=model_inputs,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            num_workers=num_workers,
            request_args=request_args,
            pbar_name=pbar_name,
            cache_flag=cache_flag,
            cache_only=cache_only,
            validation_function=validation_function
        )

    elif provider == 'together':
        return run_together(
            model_inputs=model_inputs,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            num_workers=num_workers,
            request_args=request_args,
            pbar_name=pbar_name,
            cache_flag=cache_flag,
            cache_only=cache_only
        )
    
    else:
        raise ValueError(f'{provider} (from {full_model_name}) is not a valid provider')

def run_auto_str(
        model_inputs: MaybeSizedIter[str],
        full_model_name: str = 'openai/gpt-4.1-mini',
        max_tokens: int = 1024,
        temperature: float = 0.5,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        num_workers: int = 64,
        request_args: RequestArgs = RequestArgs(),
        pbar_name: str = '{model_name}',
        cache_flag: str = '',
        cache_only: bool = False,
        validation_function: ValidationFunctionType[str] = VALIDATION_FUNCTION_DEFAULT
    ) -> list[Result[str]]:

    return run_auto(
        model_inputs=[str_to_convo(x) for x in model_inputs],
        full_model_name=full_model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        num_workers=num_workers,
        request_args=request_args,
        pbar_name=pbar_name,
        cache_flag=cache_flag,
        cache_only=cache_only,
        validation_function=validation_function
    )
