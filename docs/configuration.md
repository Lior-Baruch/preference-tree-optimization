# Configuration reference

A run is described by one YAML file (see [`configs/`](../configs/)). Every key is optional: a
missing key takes the default below, and the defaults reproduce the reported K = 5 run.
`configs/pto_k0.yaml` differs only in `lookahead.k: 0`; `configs/quicktest.yaml` shrinks
everything for a smoke test. Unknown keys are rejected, so a typo fails loudly instead of being
ignored.

```bash
pto train --config configs/pto_k5.yaml --dry-run       # validate + print the resolved config
pto train --config configs/pto_k5.yaml --iterations 2 --output-dir /tmp/runs
```

From Python, `pto.config.load_config(path, overrides)` accepts dotted overrides such as
`{"loop.num_iterations": 2, "lookahead.k": 0}`. The resolved config is saved as
`{output_dir}/{run_name}/config.yaml` at the start of a run.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `run_name` | `null` | Name of the run directory. `null` derives `pto_{oracle}_{policy}_K{k}_MCL{mcl}_M{m}_{greedy\|indep}`, e.g. `pto_Q1Q2_Llama32-1B_K5_MCL12_M8_greedy`. Letters, digits, `.`, `_` and `-` only. |
| `output_dir` | `runs` | Parent directory of all runs. |
| `seed` | `42` | Seeds `random`, `numpy`, `torch`, the per-iteration persona shuffle (`seed + n`), the patient API calls and DPO. |

## `models`

| Key | Default | Meaning |
|---|---|---|
| `policy` | `meta-llama/Llama-3.2-1B` | Hugging Face id or local path of the base policy (the LoRA adapter is trained on top). |
| `tokenizer` | `null` | Tokenizer id; `null` uses `policy`. The ChatML chat template is always installed. |
| `patient` | `gpt-4o-mini-2024-07-18` | Patient-simulator model (OpenAI chat completions). Also used for look-ahead patient turns. |
| `oracle` | `gpt-4o-mini-2024-07-18` | Training oracle; must support `response_format` with a strict JSON schema. |
| `api_base_url` | `null` | Base URL of an OpenAI-compatible server for the patient and the oracle (e.g. vLLM). `null` = api.openai.com. |
| `dtype` | `auto` | Base-model dtype: `auto` (bfloat16 on CUDA with bf16 support, float16 on other CUDA GPUs, float32 on CPU), `bfloat16`, `float16` or `float32`. |
| `load_in_4bit` | `false` | Load the base model in 4-bit NF4 (needs the `quant` extra, CUDA only). |

## `lora`

| Key | Default | Meaning |
|---|---|---|
| `r` | `16` | LoRA rank. |
| `alpha` | `16` | LoRA scaling α. |
| `dropout` | `0.05` | LoRA dropout. |
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]` | Modules that receive adapters (all attention and MLP projections of Llama). |

## `loop`

| Key | Default | Meaning |
|---|---|---|
| `num_iterations` | `10` | PTO iterations (generate, build trees, DPO, save). |
| `conversations_per_iteration` | `96` | Conversations per iteration, one persona each (at most 96; the personas are shuffled per iteration and the first N are used). |
| `conversation_length` | `49` | Utterances generated after the therapist's opening line (therapist and patient combined), so a finished conversation has up to `conversation_length + 1` utterances, as in the original code. Also caps greedy trunks. |
| `epochs_per_iteration` | `2` | DPO epochs over each iteration's pairs. |

## `generation`

Full-conversation generation (step 1 of every iteration, and the final evaluation pass).

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `64` | Conversations generated in parallel by the policy; also the chunk size for greedy branch sampling. During conversation generation it is halved automatically on out-of-memory; during branch sampling it is not: an out-of-memory chunk's branches are dropped and those trunks stop growing for the rest of the iteration, so on small GPUs set it low enough that `batch_size` candidates fit. |
| `therapist_temperature` | `0.9` | Policy sampling temperature. |
| `patient_temperature` | `0.7` | Patient-simulator temperature. |
| `max_input_tokens` | `2048` | Policy context window at generation time (oldest tokens are truncated). Also used during look-ahead. |
| `max_new_tokens` | `200` | Maximum tokens per therapist reply and per patient reply. |
| `patient_concurrency` | `96` | Maximum concurrent patient API calls (generation, look-ahead and greedy patient depths). |
| `patient_max_retries` | `3` | Attempts per patient call before it counts as failed. |
| `patient_backoff_s` | `1.0` | Back-off before the second attempt, in seconds; doubles each attempt. |
| `max_retries_without_progress` | `3` | Generation rounds without any progress before the remaining conversations are given up. |
| `stop_strings` | `["<\|im_end\|>", "<\|im_start\|>"]` | Stop generation at the first ChatML marker. The base model sees these markers as plain text and otherwise starts writing the patient's turn itself; keep them in sync with the chat template. |

## `oracle`

| Key | Default | Meaning |
|---|---|---|
| `questionnaires` | `[1, 2]` | Questionnaire ids that define the reward (mean of their mean item scores). Only flat-score questionnaires can be a reward: 1 (Q1), 2 (Q2), 3 (WAI-SR), 4 (CSQ-8), 6 (MI-SAT). |
| `temperature` | `0.0` | Oracle sampling temperature. |
| `max_retries` | `3` | Attempts per oracle call (malformed JSON, wrong id, item count or range, timeouts); back-off doubles each time. |
| `timeout_s` | `60` | Timeout of one oracle request, in seconds. |
| `concurrency` | `128` | Maximum concurrent oracle calls. |

## `lookahead`

| Key | Default | Meaning |
|---|---|---|
| `k` | `5` | Simulated utterances after each candidate before scoring (patient first, then alternating). `0` disables look-ahead and scores the reply in place. |
| `therapist_temperature` | `0.9` | Policy temperature for simulated therapist turns. |
| `patient_temperature` | `0.7` | Patient temperature for simulated patient turns. |
| `max_new_tokens` | `200` | Maximum tokens per simulated utterance. |
| `sub_batch_size` | `128` | Cap on the batched therapist `generate()` call during look-ahead (`null` = all active simulations at once); halved automatically on out-of-memory. |

## `tree`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `greedy` | `greedy`: grow one trunk per conversation, always appending the best candidate. `independent`: branch every eligible patient turn of the recorded conversations, without feedback. See [method.md](method.md#preference-trees). |
| `branches` | `8` | M, candidates sampled per branch point (≥ 2). |
| `tau` | `0.1` | Emit a pair only when best − worst > τ (strict). |
| `temperature` | `1.2` | Sampling temperature for the M candidates (higher than generation, for diversity). |
| `max_new_tokens` | `200` | Maximum tokens per candidate. |
| `min_conv_length` | `12` | MCL. Greedy: length of the trunk seed (must be even so it ends on a patient turn). Independent: minimum prefix length of a branch point. |
| `trunk_target_len` | `49` | Greedy trunks stop at `min(loop.conversation_length, trunk_target_len)` utterances; `null` = `loop.conversation_length`. Lower values build shallower trees faster, at the cost of fewer and earlier branch points. |

## `dpo`

| Key | Default | Meaning |
|---|---|---|
| `learning_rate` | `1.0e-5` | Peak learning rate (cosine schedule). |
| `beta` | `0.1` | DPO β. |
| `loss_type` | `sigmoid` | TRL DPO loss type (`sigmoid` = standard DPO). |
| `precompute_ref_log_probs` | `true` | Compute reference log-probs in a no-grad pass before training (same loss, less memory). |
| `per_device_train_batch_size` | `2` | Pairs per device per step. Keep small: DPO materialises logits over the full prompt + completion for a large vocabulary. |
| `per_device_eval_batch_size` | `4` | Eval batch size. |
| `gradient_accumulation_steps` | `8` | Effective batch = 2 × 8 = 16 pairs per optimizer step. |
| `gradient_checkpointing` | `true` | Recompute activations in the backward pass (large memory saving). |
| `warmup_ratio` | `0.01` | Warm-up steps as a fraction of the iteration's total steps. |
| `eval_split` | `0.05` | Fraction of *conversations* whose pairs form the eval set (grouped split, at least one conversation stays in training). |
| `max_prompt_length` | `2048` | Token budget of a training prompt; older turns are dropped to fit. |
| `max_completion_length` | `200` | Completion budget; the DPO sequence limit is prompt + completion. |
| `bf16` | `auto` | Mixed precision: `auto` enables bf16 only where CUDA supports it; `true` / `false` force it. |

## `logging`

| Key | Default | Meaning |
|---|---|---|
| `report_to` | `[tensorboard]` | Trainer loggers: any of `tensorboard`, `wandb` (needs the `tracking` extra; W&B reads `WANDB_API_KEY`), or `[]`. |
| `logging_steps` | `1` | Log every N optimizer steps. |
| `save_steps` | `10` | Write a DPO checkpoint every N optimizer steps (used to resume a partly trained iteration). |
| `save_total_limit` | `2` | Checkpoints kept per iteration (`null` = all). |
| `save_generations` | `true` | Write `iteration_n/eda/generations.jsonl`: every candidate of every branch point with its score, per-questionnaire sub-scores and role (chosen / rejected / neither). |
| `save_lookahead_transcripts` | `true` | Include each candidate's simulated look-ahead text in those records (the main size driver). |

## `hub`

| Key | Default | Meaning |
|---|---|---|
| `push` | `false` | Push the adapter to the Hugging Face Hub after every iteration (one commit per iteration). Needs `HF_TOKEN` with write access. |
| `repo_id` | `null` | Target repository, e.g. `your-name/pto-llama32-1b-k5`; required when `push` is true. |

## Validation rules

`pto train` (including `--dry-run`) rejects a config when, among others:

- `tree.mode` is `greedy` and `tree.min_conv_length` is odd;
- `oracle.questionnaires` contains anything outside 1, 2, 3, 4, 6;
- `tree.branches` < 2, `tree.tau` < 0, or `lookahead.k` < 0;
- `loop.conversation_length` < `tree.min_conv_length + 1` (no branch point could ever be reached);
- `dpo.eval_split` is not in (0, 1);
- `hub.push` is true without `hub.repo_id`.

## Environment variables

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | patient simulator and oracle (optional with a self-hosted `api_base_url`) |
| `HF_TOKEN` | gated base models such as Llama 3.2, and `hub.push` |
| `ANTHROPIC_API_KEY` | `pto evaluate --judge anthropic:...` |
| `WANDB_API_KEY` | `logging.report_to: [wandb]` |
| `WANDB_PROJECT` | optional W&B project name (default `pto`; the run group is the run name) |
