# MotiveMob

alternative transition sample generation + Motivation & State Predictor training & Inference.

Pipeline: take a real trajectory, swap one transition's **motivation** for a fake one, ask
"if the motivation were X, where would this person go and when?", generate candidates with the 
Transformer, pick one (by rule or by an LLM judge) as the counterfactual sample,
then fine-tune Llama on observed and alternative samples.

## Layout

```
MotiveMob/
├── seen_users/                   training users, one .pkl per user
├── unseen_users/                 held-out users, same format
├── data/
│   ├── global_loc_map.pkl        location name <-> id
│   ├── location_activity_map.pkl subcategory -> top-level activity category
│   ├── grid_map.pkl              lat/lng grid binning (grid_key2id, lat_min, lat_step, ...)
│   └── pos_map.pkl               "Category (lat, lng)" -> location id
├── transformer_predictor.py      ImprovedTabTransformer: structure, training, candidate generation
├── llm_judge.py                  LLM-as-judge: picks one candidate per transition
├── alternative_sample_generation.py  gen-data (candidates -> LLM training csv) + mix
├── implicit_world_modeling.py    LoRA fine-tune: state predictor (next state)
├── motivation_predictor_training.py  LoRA fine-tune: motivation predictor
├── inference.py                  constrained trajectory generation + post-processing
├── evaluation.py                 SD/SI/DARD/STVD metrics, real vs. generated trajectories
├── prompt_builder.py             shared prompt templates (state/motivation/judge)
├── utils.py                      shared training helpers (DDP, report parsing, 4-bit+LoRA loading)
├── location_utils.py             coordinate/distance/event parsing (used by inference.py)
└── test/                         held-out example inference CSVs
```

Every script's data-path flags default to paths under this directory (`./seen_users`,
`./data/...`) — pass `--pkl_dir`/`--pkl-dir`, `--global_loc_map`/`--global-loc-map`,
`--grid_map`/`--grid-map`, `--location_activity_map`/`--location-activity-map` explicitly if
your data lives elsewhere.

## Each user's .pkl

A 5-item list: `[trajectories, test_trajectories, motivations, report, report_with_covid_context]`.
`report_with_covid_context` is only present for users regenerated for the 2020-04–10 window;
code that expects 4 items still works (older pkls have no 5th element).

## Reproducing the pipeline

```bash
# 1. Train the candidate-generation model
python transformer_predictor.py \
    --pkl_dir seen_users --model_dir ../models/

# 2. Label alternative (counterfactual) transitions with an LLM judge
#    (transformer_predictor.py has no "predict" mode anymore - labeling is
#    always done here, by importing construct_df/collect_transformer_candidates
#    from transformer_predictor.py)
python llm_judge.py \
    --pkl-dir seen_users --transformer-ckpt ../models/transformer_improved_best.pt \
    --judge-model <llama_path> \
    --output-csv llm_judged.csv --output-parquet llm_judged.parquet
# (optional sharding for large runs: --num-shards N --shard-id i, then --merge-shards)

# 3. Build state-predictor training data (observed + alternative) and motivation-predictor data
python alternative_sample_generation.py --action gen-data --phase state \
    --pkl-dir seen_users --input_csv llm_judged.csv
python alternative_sample_generation.py --action gen-data --phase mot \
    --pkl-dir seen_users

# 4. Mix observed (positive) + alternative (negative) 1:1
python alternative_sample_generation.py --action mix
# -> state_training_data.csv

# 5. Fine-tune
python implicit_world_modeling.py \
    --csv state_training_data.csv --model <llama_path> \
    --save_path ft_out/phase1_next_state/llama_8B_phase1

python motivation_predictor_training.py \
    --csv motivation_training_data.csv --model <llama_path> \
    --phase1_lora None \
    --save_path ft_out/phase2_motivation/llama_8B_phase2

# 6. Inference (bring your own held-out csv_in with person/date/is_weekend/report/history columns)
python inference.py \
    --csv_in <your_inference_input.csv> --model <llama_path> \
    --phase1_lora ft_out/phase1_next_state/llama_8B_phase1 \
    --phase2_lora ft_out/phase2_motivation/llama_8B_phase2

# 7. Evaluate generated vs. real trajectories
python evaluation.py --input merged.csv
```

`--model` must be the same base model across steps 5–6 — a LoRA adapter only loads onto the
base it was trained from.

## Data privacy

`seen_users/` and `unseen_users/` use anonymized filenames (`seen_1.pkl`, `unseen_1.pkl`, ...).
