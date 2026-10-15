# Method

This page describes the algorithm as implemented in `src/pto/`. The paper
([arXiv 2608.12062](https://arxiv.org/abs/2608.12062)) introduced PTO with look-ahead; this
implementation is the iterative version used for the Llama-3.2-1B results in the main README.

## Setting and notation

| Symbol | Meaning | Reported runs |
|---|---|---|
| π<sub>θ</sub> | therapist policy: a base causal LM plus a LoRA adapter | Llama-3.2-1B, r = 16 |
| P | patient simulator: an API LLM with a persona system prompt | gpt-4o-mini |
| O | oracle: an API LLM that fills in questionnaires about a transcript | gpt-4o-mini |
| Q | questionnaires that define the reward | Q1 + Q2 |
| L | utterances generated after the therapist's opening line (`loop.conversation_length`); a conversation has up to L + 1 utterances | 49 |
| M | candidates sampled per branch point (`tree.branches`) | 8 |
| K | look-ahead depth in simulated utterances (`lookahead.k`) | 0 or 5 |
| τ | minimum score gap for a preference pair (`tree.tau`) | 0.1 |
| MCL | minimum conversation length before a branch point (`tree.min_conv_length`) | 12 |
| N | number of iterations (`loop.num_iterations`) | 10 |

**Personas.** The patient population is a full factorial grid of 96 personas (gender, age,
problem: smoking or obesity, problem duration, prior attempts, cooperation level). The grid order
defines `persona_id`. The therapist always opens the session with a fixed utterance; turns then
alternate. A session ends after the opening line plus L further utterances, or earlier when either
side writes `SESSION ENDED`.

**Reward.** For a transcript *x*, the oracle answers each questionnaire *q* ∈ Q with one integer per
item (strict JSON schema; questionnaire id, item count and range are validated, malformed answers
are retried). The reward is

```
r(x) = mean over q in Q of ( mean item score of q )        # Q1+Q2: average of two 1-5 means
```

If any questionnaire call for a transcript still fails after its retries, that transcript has no
score and the candidate is dropped (it can never become chosen or rejected).

## One iteration

Iteration *n* = 1, ..., N starts from policy π<sub>n−1</sub> and produces π<sub>n</sub>:

1. **Generate.** π<sub>n−1</sub> talks to P once per persona (`loop.conversations_per_iteration`
   conversations: the therapist's opening line plus up to L further utterances). The personas are
   shuffled with seed `seed + n`; the patient API calls use the same seed.
2. **Build preference pairs** with a greedy or an independent tree (below), using look-ahead depth K
   and the τ filter.
3. **Train** DPO for `loop.epochs_per_iteration` epochs on the pairs.
4. **Save** the adapter as `iteration_n/adapter`.

After the last iteration, one more generation pass (persona seed `seed + N + 1`) produces the
conversations of the final policy.

### Iteration-to-model-state mapping, and why conversations double as evaluation

The conversations generated in step 1 of iteration *n* come from π<sub>n−1</sub>. They are saved as
`conversations/model_iter_{n-1}/` and are used both as tree seeds and as the **evaluation set of
model state n−1**; there is no separate evaluation pass. The final pass fills
`model_iter_N/`. So `model_iter_0` is the base model, and `model_iter_n` is the policy after *n*
DPO updates, which is what the results call "iteration *n*". Each directory has a `manifest.json`
mapping the file index to `persona_id`, because the persona order is shuffled per iteration and
evaluation pairs model states by persona.

## Look-ahead

A candidate therapist reply *c* is judged by where the conversation goes next, not only by the
reply itself. Given the transcript *h* that ends on a patient turn and the candidate *c*:

```
x = h + c                                  # K = 0: score the reply in place
for step in 0 .. K-1:
    if step is even: x += P(x)             # patient reply (API, patient temperature)
    else:            x += pi(x)            # therapist reply (current policy, look-ahead temperature)
score(c) = r(x)
```

With K = 5 the tail is patient, therapist, patient, therapist, patient: three patient calls and
two policy generations per candidate. A simulation stops early, keeping the transcript so far, when
the patient call fails, either side writes `SESSION ENDED`, or the policy produces an empty reply.
All candidates of a step are simulated in lock-step: the patient calls run concurrently, and each
therapist step is one batched `generate()` call (sub-batched, and halved automatically on
out-of-memory). The oracle always sees the full, untruncated transcript.

## Preference trees

### Greedy mode (default, `tree.mode: greedy`)

Each conversation from step 1 that is longer than MCL utterances contributes a **trunk**: its first
MCL utterances, which end on a patient turn (MCL must be even because the therapist speaks first).
All trunks then grow in lock-step, alternating two kinds of depth:

- **Therapist depth (a branch point).** For every active trunk, sample M candidates from the policy
  (temperature `tree.temperature`), extend each with K look-ahead utterances and score it. Among the
  candidates that are non-empty and scored, take the best and the worst. The **best is always
  appended to the trunk**, which is what makes the tree greedy: the winner conditions the next
  branch point. A pair (prompt, best, worst) is emitted only when the τ filter passes and the
  training prompt fits its token budget.
- **Patient depth.** Every active trunk gets one fresh patient reply.

A trunk stops when it reaches `min(loop.conversation_length, tree.trunk_target_len)` utterances, or
freezes on `SESSION ENDED`, a failed patient call, or a branch point where no candidate could be
scored. The seeds reuse the opening of the evaluation conversations and then diverge from them;
the evaluation conversations themselves are never modified.

### Independent mode (`tree.mode: independent`)

Branch every patient turn of each recorded conversation whose prefix has at least MCL utterances
and is followed by a recorded therapist turn. For each such branch point, sample M candidates,
drop empty ones, score the rest (at least two are needed) and apply the τ filter. Nothing is fed
back: the recorded conversation continues as it was, whichever candidate won.

Greedy trees explore the states the improving policy actually reaches with its best replies;
independent trees are cheaper to reason about but keep revisiting the states of the unimproved
rollout.

### The τ filter

```
scored = [(score_m, m) for candidates m with non-empty text and a score]
if len(scored) >= 2 and max(scored) - min(scored) > tau:     # strict inequality
    emit (prompt, candidate[argmax], candidate[argmin])
```

Both modes use the same rule (`pto.tree.select_pair`). A branch point therefore produces **at most
one pair**, and none when the candidates are near-ties: with τ = 0.1 on a 1–5 scale, a pair is kept
only when the oracle actually distinguishes the two replies. If a whole iteration yields no pair,
training stops with an error that suggests lowering τ or MCL, or raising M.

### Why MCL exists: reward faithfulness on partial conversations

The questionnaires ask about a *whole session* (was it satisfying, did the patient feel understood,
did they learn something useful). Scoring a branch point means scoring a *partial* conversation.
Early in a session there is little for the oracle to judge, so the score of a short prefix is a
poor proxy for the score the finished conversation would receive. In the thesis analysis, the rank
agreement between the oracle's score of a partial conversation and its score of the full
conversation was barely above chance for the shortest prefixes and rose steadily with prefix
length. MCL therefore sets a floor: only branch points with at least MCL utterances of context
produce training signal. In greedy mode MCL is the seed length; in independent mode it filters
branch points. Look-ahead attacks the same problem from the other side by showing the oracle where
a reply leads.

## DPO update

Each pair is rendered as a ChatML prompt (therapist system prompt plus the conversation so far, from
the therapist's perspective) with the chosen and rejected replies as completions. The prompt is
capped at `dpo.max_prompt_length` tokens by **dropping the oldest turns** (the system message and the
most recent turns are always kept), matching the context the policy sees at generation time; a pair
whose most recent turn alone exceeds the budget is skipped. The DPO sequence limit is
`max_prompt_length + max_completion_length`, so truncation never cuts into a reply.

The loss is standard DPO (`dpo.loss_type: sigmoid`) with the iteration's starting policy as the
reference:

```
L = - log sigmoid( beta * [ log pi(y_w|x) / pi_ref(y_w|x)  -  log pi(y_l|x) / pi_ref(y_l|x) ] )
pi_ref = pi_{n-1} (frozen),   beta = 0.1
```

Reference log-probabilities are precomputed in a no-grad pass (equivalent for DPO since the
reference is frozen, and it saves memory). Pairs are split into train and eval sets **by
conversation** (`dpo.eval_split`), so no conversation leaks across the split. Each iteration starts a
fresh optimizer (cosine schedule with warmup) on top of the carried-over LoRA weights; only the
policy adapter is saved, not the temporary reference adapter that TRL creates. That reference
adapter is then deleted from the in-memory model, and the next iteration's trainer re-creates it
from the current weights (π<sub>n−1</sub>), so the reference is always the iteration-start policy.

## Generation details that matter

- **ChatML on a base model.** The policy is a base (not instruction-tuned) model, and the ChatML
  markers `<|im_start|>` / `<|im_end|>` are plain text to its tokenizer. Early in training it often
  "self-plays" by opening a new turn and writing the patient's lines. Generation therefore stops at
  the first of either marker, and any residue is cut from the decoded text (`clean_completion`).
  A reply that is empty after cleaning ends the conversation, is excluded from scoring, and freezes
  a look-ahead simulation.
- **Resume.** Every stage is restartable: conversations are written one CSV per conversation and
  skipped when present; `pairs.csv` marks a finished tree build (reloaded only when its sidecar
  config fingerprint matches); `_progress.json` checkpoints a build in progress after every
  conversation (independent) or depth (greedy); `iteration_n/adapter` marks a finished iteration;
  DPO checkpoints resume a partially trained iteration.

## Implementation notes

This package is a port of the research code behind the reported runs. The algorithms are unchanged;
the intentional differences are:

- **Packaging.** An installable package with relative imports, a YAML config, and the `pto` CLI
  (`train`, `generate`, `evaluate`, `compare`) instead of notebook globals.
- **Credentials from the environment** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`,
  `WANDB_API_KEY`); optional `models.api_base_url` for any OpenAI-compatible endpoint.
- **Output layout.** `{output_dir}/{run_name}/` holds the resolved `config.yaml`,
  `run_metadata.json`, `iteration_{n}/{adapter,training,pref_pairs,eda}/`, and the conversations in
  `conversations/model_iter_{n}/conversation_{i}.csv` with a `manifest.json` (`i -> persona_id`) so
  evaluation can pair model states by persona.
- **Run name.** Defaults to `pto_{oracle}_{policy}_K{k}_MCL{mcl}_M{m}_{greedy|indep}`, e.g.
  `pto_Q1Q2_Llama32-1B_K5_MCL12_M8_greedy`; it can be set explicitly.
- **Precision.** DPO `bf16` defaults to `auto` (on only where CUDA supports bf16) instead of
  always on, and the base-model dtype is configurable, with float32 on CPU (previously a float16
  fallback).
- **Import order.** `trl` is imported before `torch` (via `pto._compat`); the reverse order
  crashes at CUDA initialisation on Blackwell GPUs.
- **Optional tracking.** TensorBoard and W&B are optional and imported lazily; `report_to` defaults to
  `["tensorboard"]`.
- **Optional Hub push** (`hub.push`, `hub.repo_id`); nothing is pushed by default.
- **Stale-pair guard.** Reloading a finished `pairs.csv` now also checks a config fingerprint
  (`pairs.meta.json`), so pairs built under a different τ, M or K are rebuilt instead of reused.
- **Shared pair selection.** The τ filter is one helper, `select_pair(scores, tau)`, called by both
  tree modes; its behaviour (ties, empty and unscored candidates) is identical to the two inline
  copies it replaces.
- **Seeding.** `random`, `numpy` and `torch` are seeded inside `train()`.
- **Dead code removed:** unused helpers, duplicate async runners, and code paths that only served
  other experiments.
- **Small robustness changes.** When fewer than two conversations yield pairs, DPO runs without an
  eval split (`eval_strategy="no"`) instead of failing on an empty one. The `pairs.csv`/
  `_progress.json` fingerprint also covers the tree mode, K, the questionnaires and the oracle model,
  because the run name no longer has to encode them. `load_base_model` defaults to
  `trust_remote_code=False` (Llama needs no remote code). The oracle's async primitives are cached
  per event-loop object rather than per `id(loop)`, which could be reused after a loop closed.
  Invalid persona/therapist enum values raise `ValueError` instead of failing later.
  `build_truncated_training_prompt` supports only its default `drop_oldest` mode; the unused
  token-tail fallback was removed. TRL's reference adapter is deleted from the live model after
  each iteration's save: older peft versions silently overwrote it when the next trainer added it
  again, newer ones (0.20+) raise on the duplicate name, so the reference is unchanged. A missing
  API key now fails before the run directory is created.
