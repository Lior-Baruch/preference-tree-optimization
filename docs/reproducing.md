# Reproducing the reported runs

## The two runs

| Config | Look-ahead | Run name (derived) |
|---|---|---|
| [`configs/pto_k0.yaml`](../configs/pto_k0.yaml) | K = 0 | `pto_Q1Q2_Llama32-1B_K0_MCL12_M8_greedy` |
| [`configs/pto_k5.yaml`](../configs/pto_k5.yaml) | K = 5 | `pto_Q1Q2_Llama32-1B_K5_MCL12_M8_greedy` |

Everything else is shared: Llama-3.2-1B + LoRA (r = 16), gpt-4o-mini-2024-07-18 as patient and
oracle, Q1+Q2 reward, greedy trees with M = 8, τ = 0.1, MCL = 12, 96 conversations of up to 50
utterances (the therapist's opening line plus 49) per iteration, 10 iterations of 2 DPO epochs,
seed 42.

```bash
export OPENAI_API_KEY=... HF_TOKEN=...
pto train --config configs/pto_k0.yaml
pto train --config configs/pto_k5.yaml

# Evaluate every model state with both judges, then compare the arms at iteration 10
for n in $(seq 0 10); do
  for arm in K0 K5; do
    run=runs/pto_Q1Q2_Llama32-1B_${arm}_MCL12_M8_greedy
    pto evaluate --conversations $run/conversations/model_iter_$n \
        --judge openai:gpt-4o-mini-2024-07-18 --out scores/${arm}_iter${n}_gpt.csv
    pto evaluate --conversations $run/conversations/model_iter_$n \
        --judge anthropic:claude-haiku-4-5 --out scores/${arm}_iter${n}_haiku.csv
  done
done
pto compare scores/K0_iter10_gpt.csv scores/K5_iter10_gpt.csv --out k5_vs_k0_gpt.md
```

`pto generate --config C --adapter PATH --out DIR --seed S` produces a fresh conversation set for any
adapter. Its persona order and patient seed follow `S`; the training run used `S = seed + n + 1`
for `model_iter_{n}`, so pass that value to regenerate a set with the same persona order.

Exact numbers will not repeat: the patient and the oracle are sampled API models, the policy is
sampled at temperatures 0.9–1.2, and hosted model snapshots can change. The reported results are a
single run per configuration, so run-to-run variance is not part of the reported uncertainty.

## Hardware and compute

The reported runs used a single **A100 on Google Colab** with bfloat16 (no quantization).
GPU-hours below are wall-clock time on the training host, including the time the GPU waits for API
responses (which is what a GPU rental bills):

| | Generate | Build trees | DPO | **Total** | Per iteration |
|---|---|---|---|---|---|
| PTO, K = 0 | 1.32 h | 5.67 h | 1.13 h | **8.1 h** | 0.81 h |
| PTO, K = 5 | 1.37 h | 16.80 h | 1.51 h | **19.7 h** | 1.97 h |

Tree building (sampling M candidates, look-ahead, oracle scoring) dominates: 70% of the time at
K = 0 and 85% at K = 5. DPO itself is short.

## API usage and cost drivers

Training API calls over iterations 1–10 (evaluation-judge calls not included):

| | K = 0 | K = 5 | K5 / K0 |
|---|---|---|---|
| Candidates scored | 49,920 | 60,384 | 1.21 |
| Oracle calls | 99,622 | 121,806 | 1.22 |
| Oracle input (million characters) | 987.1 | 1,849.2 | 1.87 |
| Patient calls | 17,587 | 179,634 | 10.2 |
| **Total API calls** | **117,209** | **301,440** | **2.57** |

What drives the cost:

- **Oracle calls** ≈ candidates × number of reward questionnaires (2 for Q1+Q2). Candidates per
  iteration ≈ trunks × branch points × M, so M, MCL and the trunk length scale it linearly.
- **Oracle input size.** Each call sends the whole transcript, so cost grows with conversation
  length, and look-ahead adds K utterances to every candidate.
- **Look-ahead patient calls.** K = 5 adds three patient calls and two policy generations per
  candidate (2.6 patient calls per candidate on average, because simulations stop early when a
  session ends).
- **Evaluation.** `pto evaluate` makes one call per conversation and instrument (8 instruments,
  PCT reuses the MI-process coder), i.e. 768 calls per model state and judge.

## Practical gotchas

- **Stop strings are load-bearing.** The base model has never seen ChatML; `<|im_start|>` and
  `<|im_end|>` are plain text to it. Without `generation.stop_strings` it keeps writing the
  patient's side of the conversation, which derails the simulator. Keep the stop strings, the
  chat template (`pto.model.CHATML_TEMPLATE`) and `clean_completion` in sync if you change any of
  them.
- **Prompt-caching layout.** Oracle prompts put the instructions and rubric first and the
  transcript last, in a single user message. That keeps the long static part (and the
  conversation prefix shared by sibling candidates) at the front, where OpenAI's automatic prompt
  caching can reuse it. It is also part of the measurement instrument: reordering it changes
  scores, so the questionnaires are covered by byte-for-byte golden tests.
- **trl before torch.** On Blackwell GPUs (e.g. RTX 50-series), importing `torch` before `trl`
  can crash at CUDA initialisation. Every `pto` module that needs torch imports `pto._compat`
  first, which imports `trl` early (`import pto` itself stays light and does not); when scripting
  around the package, import `pto._compat` (or `trl`) before `torch`.
- **Memory on small GPUs.** An A100 runs the defaults comfortably. On a 12–16 GB card start by
  lowering `generation.batch_size` (e.g. to 16) and `lookahead.sub_batch_size` (e.g. to 32).
  Conversation generation and look-ahead halve these automatically on out-of-memory, but greedy
  branch sampling does not (an out-of-memory chunk's branches are dropped and those trunks
  freeze), so `generation.batch_size` must fit on its own. Keep `dpo.per_device_train_batch_size` small and
  `dpo.gradient_checkpointing` on: DPO logits over a 128k vocabulary are the largest tensor.
  `models.load_in_4bit` saves weight memory but slows generation.
- **Resume, don't restart.** Re-running the same command continues where a run stopped.
  Completion markers, in order: `conversations/model_iter_{n}/conversation_{i}.csv` (one per
  conversation), `iteration_{n}/pref_pairs/pairs.csv` (finished tree build; reused only if the tree settings
  recorded in `pairs.meta.json` match: mode, M, τ, MCL, K, conversation/trunk lengths,
  questionnaires, oracle model and seed; after changing sampling temperatures, token limits, the
  patient/policy model or `dpo.max_prompt_length`, delete `iteration_{n}/pref_pairs/` by hand),
  `_progress.json` (a tree build in progress), DPO checkpoints under `iteration_{n}/training/`,
  and `iteration_{n}/adapter` (iteration done). Delete a marker to force that stage to run again;
  to retrain an iteration, delete both `iteration_{n}/adapter` and `iteration_{n}/training/`
  (otherwise training resumes from the last checkpoint).
- **Colab.** VMs are ephemeral: point `output_dir` at persistent storage (for example a mounted
  Google Drive folder) so a disconnect costs minutes, not the run. If `peft` raises an
  `ImportError` about `torchao`, uninstall it (`pip uninstall -y torchao`); the project does not
  use it.
- **Gated model.** `meta-llama/Llama-3.2-1B` requires accepting the license on the Hugging Face Hub
  and an `HF_TOKEN`.
- **Judges disagree on scale.** On Q1+Q2, claude-haiku-4-5 scores the same conversations about
  1.2–1.6 points lower than gpt-4o-mini. Compare contrasts within a judge, never levels across judges.
