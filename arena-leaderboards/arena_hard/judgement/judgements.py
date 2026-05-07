from typing import Any, Optional, Callable, Literal
from typeguard import typechecked
from dataclasses import dataclass
import re
import json
import os
import logging

from ..configs import ArenaConfig, EndpointConfig, MultiturnArenaConfig
from ..dataloading import ModelAnswer, ModelAnswerMap, QuestionData, Conversation, convo_to_string_nosystem
from .judge_settings import JUDGE_SETTINGS
from fast_openai import run_auto
from ..utils import run_dummy, CheckTypesInit

logger = logging.getLogger(__name__)


@dataclass
class JudgePromptInput(CheckTypesInit):
    '''
    Args to make JudgeModelInput (i.e., the args to make the prompt)
    '''
    question: QuestionData # instance we're evaluating the model with
    target_answer: ModelAnswer
    baseline_answer: ModelAnswer

    arena_cfg: ArenaConfig

    reference: Optional[dict[str, Any]] = None

    @property
    def is_multiturn(self) -> bool:
        return isinstance(self.arena_cfg, MultiturnArenaConfig)

    def flipped(self) -> 'JudgePromptInput':
        return JudgePromptInput(
            question=self.question,
            target_answer=self.baseline_answer,
            baseline_answer=self.target_answer,
            arena_cfg=self.arena_cfg,
            reference=self.reference
        )

@dataclass
class JudgeModelInput(CheckTypesInit):
    '''
    Args for actual input to judge LM
    '''
    messages: Conversation
    endpoint_cfg: EndpointConfig
    
    output_regex_patterns: list[str] # for parsing the model output

    def __post_init__(self):
        super().__post_init__()

        assert self.messages[-1]['role'] == 'user'

@dataclass
class JudgeModelOutput(CheckTypesInit):
    '''
    Parsed model output from judge LM
    '''
    score: str | None
    judgment: dict[Literal["answer"], str]
    prompt: Conversation

def get_score(judgment: str, patterns) -> str | None:
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    
    return None


def run_judge_lm_batch(inputs: list[JudgeModelInput]) -> list[JudgeModelOutput | None]:
    endpoint_cfg = inputs[0].endpoint_cfg
    output_regex = inputs[0].output_regex_patterns

    if endpoint_cfg.model_str == 'dummy':
        results = run_dummy(
            model_inputs=[
                x.messages
                for x in inputs
            ], # type: ignore
            full_model_name=endpoint_cfg.model_str,
            max_tokens=endpoint_cfg.max_tokens,
            temperature=endpoint_cfg.temperature,
            request_args=endpoint_cfg.request_args,
            num_workers=endpoint_cfg.parallel
        )
    
    else:
        results = run_auto(
            model_inputs=[
                x.messages
                for x in inputs
            ], # type: ignore
            full_model_name=endpoint_cfg.model_str,
            max_tokens=endpoint_cfg.max_tokens,
            temperature=endpoint_cfg.temperature,
            request_args=endpoint_cfg.request_args,
            num_workers=endpoint_cfg.parallel
        )

    scores = [
        get_score(res.output, output_regex) if res.output is not None else None
        for res in results
    ]

    return [
        JudgeModelOutput(
            score=score,
            judgment={"answer": res.output},
            prompt=inp.messages
        ) if res.output is not None else None
        for inp, res, score in zip(inputs, results, scores, strict=True)
    ]


def make_judge_model_input(inp: JudgePromptInput) -> JudgeModelInput:
    convo = inp.question.convo # prompt/convo history we used to evalute the model with
    question_prompt = convo[-1]['content']
    
    # the models answers to the prompt
    target_answer_str = inp.target_answer.messages[-1]['content']
    baseline_answer_str = inp.baseline_answer.messages[-1]['content']

    assert isinstance(target_answer_str, str)
    assert isinstance(baseline_answer_str, str)

    # print(target_answer_str)
    # print(convo)
    # print(inp.target_answer.messages[:-1])

    # validate that the convo history is the same
    assert inp.target_answer.messages[:-1] == convo
    assert inp.baseline_answer.messages[:-1] == convo

    prompt_args = {
        "QUESTION": question_prompt,
        "ANSWER_A": baseline_answer_str,
        "ANSWER_B": target_answer_str
    }

    if inp.is_multiturn:
        prompt_args["CONVERSATION_HISTORY"] = convo_to_string_nosystem(convo[:-1])

    # if inp.reference is not None:
    #     prompt_args[f"REFERENCE"] = inp.reference["messages"][-1]["content"]
    if inp.reference is not None:
        raise NotImplementedError()


    judge_prompt_str = inp.arena_cfg.prompt_template.format(**prompt_args)
    question_cat = inp.question.category
    messages: Conversation = [
        {
            "role": "system",
            "content": JUDGE_SETTINGS[question_cat]["system_prompt"],
        },
        {
            "role": "user", 
            "content": judge_prompt_str,
        }
    ]

    return JudgeModelInput(
        messages=messages,
        endpoint_cfg=inp.arena_cfg.judge_endpoint,
        output_regex_patterns=inp.arena_cfg.regex_patterns
    )

def collate_outputs(
        inp: JudgePromptInput,
        res_regular: Optional[JudgeModelOutput],
        res_flipped: Optional[JudgeModelOutput]
    ) -> dict[str, Any]:

    answer = inp.target_answer
    baseline = inp.baseline_answer
    return {
        "uid": inp.question.uid,
        "category": inp.question.category,
        "judge": inp.arena_cfg.judge_endpoint.model_str,
        "model": answer.model,
        "baseline": baseline.model,
        "games": [
            res_regular.__dict__ if res_regular is not None else None,
            res_flipped.__dict__ if res_flipped is not None else None,
        ]
    }

def run_single(
        save_fp: str,
        model: str,
        baseline_model: str,
        arena_cfg: ArenaConfig,
        questions: list[QuestionData],
        model_answers: ModelAnswerMap,
    ) -> None | tuple[list[JudgeModelOutput | None], list[JudgeModelOutput | None]]:

    if os.path.isfile(save_fp):
        logger.info('%s already exists, skipping', save_fp)
        # os.remove(save_fp)
        return

    found_models = set(model_answers.get_models())
    assert model in found_models, f'no model answers found for {model}; found_models={found_models}'
    assert baseline_model in found_models, (
        f'no model answers found for baseline={baseline_model}; found_models={found_models}'
    )

    judge_inputs: list[JudgePromptInput] = []
    regular_inputs: list[JudgeModelInput] = []
    flipped_inputs: list[JudgeModelInput] = []

    found_uids = set(model_answers.get_question_uids(model))

    for question in questions:
        uid = question.uid

        if not uid in found_uids:
            logger.warning('%s answer to %s cannot be found', model, question.uid)
            continue
        
        target_answer = model_answers.get_response(model, uid)
        baseline_answer = model_answers.get_response(
            baseline_model,
            uid
        )

        judge_prompt_inp = JudgePromptInput(
            question=question,
            target_answer=target_answer,
            baseline_answer=baseline_answer,
            arena_cfg=arena_cfg,
            reference=None
        )

        judge_inputs.append(
            judge_prompt_inp
        )

        regular_inputs.append(
            make_judge_model_input(
                judge_prompt_inp
            )
        )

        flipped_inputs.append(
            make_judge_model_input(
                judge_prompt_inp.flipped()
            )
        )

    logger.info('running judging for %s', model)
    regular_outputs = run_judge_lm_batch(regular_inputs)
    flipped_outputs = run_judge_lm_batch(flipped_inputs)

    for (j_inp, out_reg, out_flip) in zip(judge_inputs, regular_outputs, flipped_outputs, strict=True):
        with open(save_fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                collate_outputs(j_inp, out_reg, out_flip),
                ensure_ascii=False
            ) + "\n")

    return (regular_outputs, flipped_outputs)

def run_all(
        arena_cfg: ArenaConfig,
        questions: list[QuestionData],
        model_answers: ModelAnswerMap,
        save_fp_map: dict[str, str],
        baseline_model: str,
    ):
    
    for model in arena_cfg.model_list:
        run_single(
            save_fp_map[model],
            model,
            baseline_model,
            arena_cfg,
            questions,
            model_answers
        )
