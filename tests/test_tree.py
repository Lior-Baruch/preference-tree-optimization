"""Preference-tree building blocks: pair selection, seeds, scoring, persistence and resume."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from pto.config import load_config
from pto.conversation import ConversationState, build_truncated_training_prompt
from pto.oracle import OracleAsyncPrimitives
from pto.personas import get_personas, get_therapist_prompts
from pto.tree import (
    PAIRS_FILENAME,
    PAIRS_META_FILENAME,
    _load_pref_progress,
    _reload_pairs_csv,
    _score_completions_batch_detailed,
    _slice_prefix_seeds,
    _write_pairs_csv,
    _write_pref_progress,
    extract_pref_pairs_from_conversations,
    grow_preference_trees_batch,
    load_completed_pairs,
    save_completed_pairs,
    select_pair,
)

# --- select_pair -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "tau", "expected"),
    [
        ([1.0, 3.0, 2.0], 0.1, (1, 0)),
        ([2.0, 4.5, 4.0, 1.5], 0.0, (1, 3)),
        # Strict boundary: a gap equal to tau is NOT enough.
        ([1.0, 1.5], 0.5, None),
        ([1.0, 1.5], 0.25, (1, 0)),
        ([3.0, 3.0, 3.0], 0.0, None),  # all tied never pairs, even with tau = 0
        # Ties: best is the LAST index among the maxima, worst the FIRST among the minima.
        ([1.0, 3.0, 3.0], 0.1, (2, 0)),
        ([1.0, 1.0, 3.0], 0.1, (2, 0)),
        ([2.0, 5.0, 1.0, 5.0, 1.0], 0.1, (3, 2)),
        # Unscored candidates (oracle failure / empty completion) are ignored.
        ([None, 1.0, None, 4.0], 0.1, (3, 1)),
        ([None, 3.0], 0.0, None),
        ([None, None], 0.0, None),
        ([], 0.0, None),
        ([5.0], 0.0, None),
    ],
)
def test_select_pair(scores, tau, expected) -> None:
    assert select_pair(scores, tau) == expected


# --- seeds -------------------------------------------------------------------------------------


def _state(
    idx: int, n: int, *, failed: bool = False, first: str = "therapist"
) -> ConversationState:
    other = "patient" if first == "therapist" else "therapist"
    turns = [
        {"role": first if j % 2 == 0 else other, "content": f"c{idx} u{j} text"} for j in range(n)
    ]
    return ConversationState(
        permutation_index=idx,
        conversation=[t["content"] for t in turns],
        messages_Patient_assist=[{"role": "system", "content": "p"}],
        messages_Therapist_assist=[{"role": "system", "content": "t"}],
        turns=turns,
        failed=failed,
        is_active=False,
    )


def test_slice_prefix_seeds() -> None:
    long_conv = _state(0, 9)
    states = [
        long_conv,
        _state(1, 4),  # exactly MCL: does not extend past it
        _state(2, 3),  # shorter than MCL
        _state(3, 9, failed=True),
        _state(4, 9, first="patient"),  # prefix would end on a therapist turn
    ]
    seeds = _slice_prefix_seeds(states, 4)
    assert [s.permutation_index for s in seeds] == [0]
    (seed,) = seeds
    assert seed.turns == long_conv.turns[:4]
    assert seed.conversation == long_conv.conversation[:4]
    assert seed.next_speaker == "therapist"
    assert seed.messages_Patient_assist == [] and seed.messages_Therapist_assist == []
    assert seed.is_active and not seed.failed

    # Copies: growing a seed never touches the recorded conversation.
    seed.turns.append({"role": "therapist", "content": "new"})
    seed.turns[0]["content"] = "mutated"
    seed.conversation.append("new")
    assert len(long_conv.turns) == 9 and long_conv.turns[0]["content"] == "c0 u0 text"
    assert len(long_conv.conversation) == 9


# --- pairs.csv / fingerprint -------------------------------------------------------------------

PAIRS = [
    {
        "prompt": "<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\n",
        "chosen": 'A reply, with "quotes"\nand a newline.',
        "rejected": "NA",
        "chosen_score": 4.25,
        "rejected_score": 2.0,
        "conversation_id": 3,
        "branch_depth": 12,
    },
    {
        "prompt": "p2",
        "chosen": "None",
        "rejected": "short",
        "chosen_score": 3.5,
        "rejected_score": 3.0,
        "conversation_id": 5,
        "branch_depth": 14,
    },
]


def test_pairs_csv_round_trip(tmp_path) -> None:
    path = str(tmp_path / "pref" / PAIRS_FILENAME)
    _write_pairs_csv(PAIRS, path)
    assert not os.path.exists(path + ".tmp")
    reloaded = _reload_pairs_csv(path)
    assert reloaded == PAIRS
    assert all(isinstance(r["chosen_score"], float) for r in reloaded)


def test_empty_pairs_csv_reloads_empty(tmp_path) -> None:
    path = str(tmp_path / PAIRS_FILENAME)
    _write_pairs_csv([], path)
    assert os.path.exists(path)
    assert _reload_pairs_csv(path) == []


def _cfg(**overrides):
    return load_config(None, overrides)


def test_completed_pairs_fingerprint(tmp_path) -> None:
    pref_dir = str(tmp_path / "pref_pairs")
    os.makedirs(pref_dir)
    cfg = _cfg()
    save_completed_pairs(PAIRS, pref_dir, cfg)
    assert os.path.exists(os.path.join(pref_dir, PAIRS_META_FILENAME))

    assert load_completed_pairs(pref_dir, cfg) == PAIRS
    # Iteration count and logging do not change the pairs a build produces.
    assert load_completed_pairs(pref_dir, _cfg(**{"loop.num_iterations": 3})) == PAIRS

    # A different tau must rebuild instead of silently reusing stale pairs.
    assert load_completed_pairs(pref_dir, _cfg(**{"tree.tau": 0.5})) is None
    assert not os.path.exists(os.path.join(pref_dir, PAIRS_FILENAME))
    assert load_completed_pairs(pref_dir, cfg) is None


@pytest.mark.parametrize(
    "changed",
    [{"tree.branches": 4}, {"lookahead.k": 0}, {"tree.min_conv_length": 10}, {"seed": 1}],
)
def test_fingerprint_covers_tree_settings(tmp_path, changed) -> None:
    pref_dir = str(tmp_path)
    save_completed_pairs(PAIRS, pref_dir, _cfg())
    assert load_completed_pairs(pref_dir, _cfg(**changed)) is None


def test_pairs_without_fingerprint_are_rebuilt(tmp_path) -> None:
    pref_dir = str(tmp_path)
    _write_pairs_csv(PAIRS, os.path.join(pref_dir, PAIRS_FILENAME))
    assert load_completed_pairs(pref_dir, _cfg()) is None


def test_missing_pairs_csv(tmp_path) -> None:
    assert load_completed_pairs(str(tmp_path), _cfg()) is None


# --- _progress.json ----------------------------------------------------------------------------


def test_progress_snapshot_guards(tmp_path) -> None:
    path = str(tmp_path / "_progress.json")
    cfg = _cfg()
    _write_pref_progress(
        path, mode="greedy", iteration=2, cfg=cfg, perm_ids=[3, 1], pairs=PAIRS,
        eda_records=[], depth=4,
    )
    snap = _load_pref_progress(path, cfg, 2, "greedy", [1, 3])
    assert snap is not None and snap["depth"] == 4 and snap["pairs"] == PAIRS

    assert _load_pref_progress(path, cfg, 3, "greedy", [1, 3]) is None  # iteration
    assert _load_pref_progress(path, cfg, 2, "independent", [1, 3]) is None  # mode
    assert _load_pref_progress(path, cfg, 2, "greedy", [1, 2, 3]) is None  # conversation set
    assert _load_pref_progress(path, _cfg(**{"tree.tau": 0.2}), 2, "greedy", [1, 3]) is None
    assert _load_pref_progress(None, cfg, 2, "greedy", [1, 3]) is None

    with open(path, "w", encoding="utf-8") as f:
        f.write('{"truncated": ')
    assert _load_pref_progress(path, cfg, 2, "greedy", [1, 3]) is None


# --- branch scoring ----------------------------------------------------------------------------


def _constant_oracle(schema_tools, per_qid: dict[int, int]):
    def answer(schema, prompt):
        qid = schema_tools.qid(schema)
        n = schema["properties"]["scores"]["minItems"]
        return {"questionnaire_id": qid, "scores": [per_qid[qid]] * n}

    return answer


def _score(client, oracle_cfg, lookahead_cfg, *, policy=None, tokenizer=None, n=2):
    primitives = OracleAsyncPrimitives(oracle_concurrency=8, lookahead_patient_concurrency=8)
    therapist_prompt, _ = get_therapist_prompts()
    transcript = "[THERAPIST]: Hello.\n\n[PATIENT]: I smoke too much."
    return asyncio.run(
        _score_completions_batch_detailed(
            client, oracle_cfg, lookahead_cfg, primitives,
            policy=policy, tokenizer=tokenizer,
            therapist_system_prompt=therapist_prompt,
            transcripts=[transcript] * n,
            completions=[f"Candidate {i}?" for i in range(n)],
            patient_system_prompts=[get_personas()[0]["patient_system_prompt"]] * n,
            questionnaire_ids=[1, 2],
        )
    ), transcript


def test_score_is_mean_over_questionnaires(
    fake_openai, schema_tools, make_oracle_cfg, make_lookahead_cfg, no_sleep
) -> None:
    client = fake_openai(oracle_fn=_constant_oracle(schema_tools, {1: 2, 2: 5}))
    details, transcript = _score(client, make_oracle_cfg(), make_lookahead_cfg(k=0))
    for d in details:
        assert d["score"] == pytest.approx(3.5)
        assert d["sub_scores"] == {"1": 2.0, "2": 5.0}
        assert d["success"] is True and d["retries"] == 0
        assert d["lookahead"] == {"k": 0, "realized_turns": 0, "ended_early": False, "tail": None}
    # K = 0: the oracle scores prefix + candidate directly.
    prompts = {c["messages"][0]["content"] for c in client.oracle_calls}
    assert any(f"{transcript}\n\n[THERAPIST]: Candidate 0?" in p for p in prompts)
    assert client.patient_calls == []


def test_score_none_when_any_questionnaire_fails(
    fake_openai, schema_tools, make_oracle_cfg, make_lookahead_cfg, no_sleep
) -> None:
    def oracle(schema, prompt):
        if schema_tools.qid(schema) == 2:
            return "not json"
        return _constant_oracle(schema_tools, {1: 4})(schema, prompt)

    client = fake_openai(oracle_fn=oracle)
    details, _ = _score(client, make_oracle_cfg(max_retries=2), make_lookahead_cfg(k=0), n=1)
    (d,) = details
    assert d["score"] is None and d["success"] is False
    assert d["sub_scores"] == {"1": 4.0}
    assert d["retries"] == 1


def test_lookahead_extends_before_scoring(
    fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model, tiny_tokenizer, no_sleep
) -> None:
    client = fake_openai()
    details, _ = _score(
        client, make_oracle_cfg(), make_lookahead_cfg(k=2, max_tokens=8),
        policy=make_tiny_model(), tokenizer=tiny_tokenizer, n=3,
    )
    assert len(client.patient_calls) == 3  # one simulated patient turn per candidate
    for d in details:
        la = d["lookahead"]
        assert la["k"] == 2
        assert la["tail"].startswith("\n\n[PATIENT]: ")
        # A degenerate (empty) therapist turn may freeze a sim after the patient turn.
        assert la["realized_turns"] in (1, 2)
        assert la["ended_early"] == (la["realized_turns"] < 2)
        assert d["score"] is not None


def test_lookahead_stops_when_patient_ends_session(
    fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model, tiny_tokenizer, no_sleep
) -> None:
    client = fake_openai(end_after_patient_turns=1)
    details, _ = _score(
        client, make_oracle_cfg(), make_lookahead_cfg(k=4, max_tokens=8),
        policy=make_tiny_model(), tokenizer=tiny_tokenizer, n=2,
    )
    for d in details:
        la = d["lookahead"]
        assert la["realized_turns"] == 1 and la["ended_early"] is True
        assert "SESSION ENDED" not in la["tail"]  # the marker is cut, the text before it kept


# --- tree builders on the tiny model -----------------------------------------------------------


def _tree_cfg(mode: str, mcl: int, length: int):
    return _cfg(**{
        "tree.mode": mode,
        "tree.min_conv_length": mcl,
        "tree.trunk_target_len": length,
        "tree.branches": 3,
        "tree.tau": 0.1,
        "tree.max_new_tokens": 8,
        "loop.conversation_length": length,
        "lookahead.k": 0,
        "generation.max_new_tokens": 8,
        "generation.batch_size": 8,
        "generation.patient_max_retries": 1,
    })


def _recorded_states(n_convs: int, n_utts: int) -> list[ConversationState]:
    return [_state(i, n_utts) for i in range(n_convs)]


def _build(coro_fn, *args, **kwargs):
    return asyncio.run(coro_fn(*args, **kwargs))


def test_greedy_growth_and_resume(
    tmp_path, fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model, tiny_tokenizer,
    no_sleep,
) -> None:
    cfg = _tree_cfg("greedy", mcl=4, length=8)
    therapist_prompt, _ = get_therapist_prompts()
    personas = get_personas()
    model = make_tiny_model()
    recorded = _recorded_states(2, 6)
    progress = str(tmp_path / "_progress.json")
    common = dict(
        policy=model, tokenizer=tiny_tokenizer, therapist_system_prompt=therapist_prompt,
        oracle_cfg=make_oracle_cfg(), lookahead_cfg=make_lookahead_cfg(k=0),
        cfg=cfg, iteration=1, progress_path=progress,
    )

    client = fake_openai()
    seeds = _slice_prefix_seeds(recorded, 4)
    pairs = _build(
        grow_preference_trees_batch, seeds, personas,
        primitives=OracleAsyncPrimitives(8, 8), client=client, **common,
    )
    assert pairs, "the hashed fake oracle should separate at least one branch point"
    for p in pairs:
        assert p["chosen_score"] - p["rejected_score"] > cfg.tree.tau
        assert p["chosen"] != p["rejected"]
        assert p["branch_depth"] in (4, 6)  # therapist turns at utterances 4 and 6
        assert p["conversation_id"] in (0, 1)
        assert p["prompt"].endswith("<|im_start|>assistant\n")
    # The first branch point's prompt is the seed prefix rendered for DPO.
    first = [p for p in pairs if p["branch_depth"] == 4]
    for p in first:
        seed_turns = recorded[p["conversation_id"]].turns[:4]
        assert p["prompt"] == build_truncated_training_prompt(
            seed_turns, therapist_prompt, tiny_tokenizer, cfg.dpo.max_prompt_length
        )
    # Trunks grew to the target length without touching the recorded conversations.
    assert all(len(s.conversation) == 8 for s in seeds)
    assert all(len(s.turns) == 6 for s in recorded)
    # M candidates per trunk per therapist depth, each scored on two questionnaires.
    assert len(client.oracle_calls) <= 2 * 2 * 3 * 2
    assert len(client.patient_calls) == 2 * 2

    snap = json.loads(open(progress, encoding="utf-8").read())
    assert snap["mode"] == "greedy" and snap["depth"] == 4

    # Resume from the snapshot: nothing left to grow, pairs are carried over, no API calls.
    again = fake_openai()
    resumed = _build(
        grow_preference_trees_batch, _slice_prefix_seeds(recorded, 4), personas,
        primitives=OracleAsyncPrimitives(8, 8), client=again, **common,
    )
    assert resumed == json.loads(json.dumps(pairs))
    assert again.calls == []


def test_independent_pairs_and_resume(
    tmp_path, fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model, tiny_tokenizer,
    no_sleep,
) -> None:
    cfg = _tree_cfg("independent", mcl=5, length=8)
    therapist_prompt, _ = get_therapist_prompts()
    personas = get_personas()
    recorded = _recorded_states(2, 8)
    progress = str(tmp_path / "_progress.json")
    common = dict(
        policy=make_tiny_model(), tokenizer=tiny_tokenizer,
        therapist_system_prompt=therapist_prompt, oracle_cfg=make_oracle_cfg(),
        lookahead_cfg=make_lookahead_cfg(k=0), cfg=cfg, iteration=1, progress_path=progress,
    )

    client = fake_openai()
    pairs = _build(
        extract_pref_pairs_from_conversations, recorded, personas,
        primitives=OracleAsyncPrimitives(8, 8), client=client, **common,
    )
    # Patient turns 1, 3, 5, 7: only 5 has >= MCL utterances AND a recorded therapist reply.
    assert pairs
    assert {p["branch_turn_index"] for p in pairs} <= {5}
    for p in pairs:
        assert p["chosen_score"] - p["rejected_score"] > cfg.tree.tau
        assert p["prompt"] == build_truncated_training_prompt(
            recorded[p["conversation_id"]].turns[:6], therapist_prompt, tiny_tokenizer,
            cfg.dpo.max_prompt_length,
        )
    assert client.patient_calls == []
    assert len(client.oracle_calls) <= 2 * 3 * 2

    snap = json.loads(open(progress, encoding="utf-8").read())
    assert snap["processed_conv_ids"] == [0, 1]

    again = fake_openai()
    resumed = _build(
        extract_pref_pairs_from_conversations, recorded, personas,
        primitives=OracleAsyncPrimitives(8, 8), client=again, **common,
    )
    assert resumed == json.loads(json.dumps(pairs))
    assert again.calls == []


def test_independent_with_lookahead(
    tmp_path, fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model, tiny_tokenizer,
    no_sleep,
) -> None:
    cfg = _tree_cfg("independent", mcl=5, length=8)
    therapist_prompt, _ = get_therapist_prompts()
    client = fake_openai()
    pairs = _build(
        extract_pref_pairs_from_conversations, _recorded_states(2, 8), get_personas(),
        primitives=OracleAsyncPrimitives(8, 8), client=client,
        policy=make_tiny_model(), tokenizer=tiny_tokenizer,
        therapist_system_prompt=therapist_prompt, oracle_cfg=make_oracle_cfg(),
        lookahead_cfg=make_lookahead_cfg(k=1, max_tokens=8), cfg=cfg, iteration=1,
        progress_path=str(tmp_path / "_progress.json"),
    )
    assert isinstance(pairs, list)
    # K = 1: every scored candidate got exactly one simulated patient turn before the oracle.
    n_q = len(cfg.oracle.questionnaires)
    assert len(client.patient_calls) == len(client.oracle_calls) // n_q > 0


class _Crash(Exception):
    """Simulated crash in the middle of a tree build."""


def _crash_on_call(real, n: int):
    calls = {"n": 0}

    async def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise _Crash
        return await real(*args, **kwargs)

    return wrapper


def test_greedy_resume_after_mid_build_crash(
    tmp_path, monkeypatch, fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model,
    tiny_tokenizer, no_sleep,
) -> None:
    import pto.tree as tree_mod

    cfg = _tree_cfg("greedy", mcl=4, length=10)
    therapist_prompt, _ = get_therapist_prompts()
    personas = get_personas()
    recorded = _recorded_states(3, 6)
    progress = tmp_path / "_progress.json"
    common = dict(
        policy=make_tiny_model(), tokenizer=tiny_tokenizer,
        therapist_system_prompt=therapist_prompt, oracle_cfg=make_oracle_cfg(),
        lookahead_cfg=make_lookahead_cfg(k=0), cfg=cfg, iteration=1, progress_path=str(progress),
        primitives=OracleAsyncPrimitives(8, 8),
    )

    real = tree_mod._grow_patient_depth
    monkeypatch.setattr(tree_mod, "_grow_patient_depth", _crash_on_call(real, 2))
    with pytest.raises(_Crash):
        _build(grow_preference_trees_batch, _slice_prefix_seeds(recorded, 4), personas,
               client=fake_openai(), **common)
    snap = json.loads(progress.read_text(encoding="utf-8"))
    assert snap["depth"] == 3
    assert snap["trunks"] and all(len(t["turns"]) == 7 for t in snap["trunks"])

    # Resume: trunks continue from the snapshot, carried pairs are kept as a prefix.
    monkeypatch.setattr(tree_mod, "_grow_patient_depth", real)
    seeds = _slice_prefix_seeds(recorded, 4)
    pairs = _build(grow_preference_trees_batch, seeds, personas, client=fake_openai(), **common)
    carried = snap["pairs"]
    assert carried and pairs[:len(carried)] == carried
    assert len(pairs) > len(carried)
    assert all(p["branch_depth"] == 8 for p in pairs[len(carried):])
    assert all(len(s.conversation) == 10 for s in seeds)


def test_independent_resume_after_mid_build_crash(
    tmp_path, monkeypatch, fake_openai, make_oracle_cfg, make_lookahead_cfg, make_tiny_model,
    tiny_tokenizer, no_sleep,
) -> None:
    import pto.tree as tree_mod

    cfg = _tree_cfg("independent", mcl=5, length=8)
    therapist_prompt, _ = get_therapist_prompts()
    personas = get_personas()
    recorded = _recorded_states(2, 8)
    progress = tmp_path / "_progress.json"
    common = dict(
        policy=make_tiny_model(), tokenizer=tiny_tokenizer,
        therapist_system_prompt=therapist_prompt, oracle_cfg=make_oracle_cfg(),
        lookahead_cfg=make_lookahead_cfg(k=0), cfg=cfg, iteration=1, progress_path=str(progress),
        primitives=OracleAsyncPrimitives(8, 8),
    )

    real = tree_mod.build_pref_pairs_for_conversation
    monkeypatch.setattr(tree_mod, "build_pref_pairs_for_conversation", _crash_on_call(real, 2))
    with pytest.raises(_Crash):
        _build(extract_pref_pairs_from_conversations, recorded, personas,
               client=fake_openai(), **common)
    snap = json.loads(progress.read_text(encoding="utf-8"))
    assert snap["processed_conv_ids"] == [0]
    carried = snap["pairs"]
    assert carried and all(p["conversation_id"] == 0 for p in carried)

    # Resume: conversation 0 is skipped, only conversation 1 is built.
    monkeypatch.setattr(tree_mod, "build_pref_pairs_for_conversation", real)
    pairs = _build(extract_pref_pairs_from_conversations, recorded, personas,
                   client=fake_openai(), **common)
    assert pairs[:len(carried)] == carried
    assert len(pairs) > len(carried)
    assert all(p["conversation_id"] == 1 for p in pairs[len(carried):])
