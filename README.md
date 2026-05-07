# Synthetic Data Generation
## Installation
In `synth_gen`, run: `pip install -e .`

## Reproduction
As inputs, we require the O*NET hierarchy of tasks & occupations. Because each task statement is underspecified, we also require that this is augmented with an extra Task_detailed column containing a more detailed description of the task (we do this by contrasting the task w/ other tasks belonging to the occupation).

To produce this run:
```
python scripts/make_detailed_tasks.py \
    --onet-version 30.2 \
    --output-path data/detailed_tasks_30_2.csv
```

To reproduce the synthetic data generation, run in `exposure_lib`:
```
python scripts/run_search.py \
    --tasks-path data/detailed_tasks_30_2.csv \
    --occupations-path "" \
    --output-path data/search_results_full_30_2.tar.gz \
    --included-respondents 2 \
    --time-savings 90%-100% 80% 70% 60% 50% 40% 30% 20% 10% 5% 1%
```

# Exposure prediction
## Installation
In `exposure_lib` run: `pip install -e .`

## Reproduction
The latest exposure prediction can be reproduced as follows, run in `synth_gen`:
```
python scripts/run_scoring_newsyn_ret_int.py \
  --input-path ../synth_gen/data/search_results_full_30_2.tar.gz \
  --response-llm-config-dir scripts/model_configs \
  --prompt-version v3_span \
  --output-path data/scored_interview_sampleocc3_30_2_span5_5mini_v3.pkl.zst \
  --interview-model-name openai/gpt-5-mini@reasoning_effort=low \
  --categorize-setting span5 \
  --occupations-path data/occupations_sample_3.txt \
  response_gpt_5_mini_reasoning_medium.json
```

which uses the output from the previous synthetic data generation: `search_results_full_30_2.tar.gz`.

