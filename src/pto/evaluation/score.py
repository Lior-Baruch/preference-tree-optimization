"""Score a directory of saved conversations with an LLM judge.

Input: ``conversation_{i}.csv`` files (columns ``role, conversation, session_ended_by,
session_ended_explanation``; therapist speaks first) plus an optional ``manifest.json`` mapping
``"<i>" -> persona_id``. Without a manifest, ``persona_id = i``.

Output: a tidy long table ``conversation_index, persona_id, instrument, value`` -- one row per
(conversation, instrument), written incrementally to ``out_csv`` so an interrupted run resumes
where it stopped (pairs already in the file are skipped).

Each instrument is one judge call per conversation and reports one headline column:

=========  ==============  =====================================================
Instrument Questionnaire   Reported value
=========  ==============  =====================================================
Q1         Q1 (5 items)    ``Q1_Mean``
Q2         Q2 (17 items)   ``Q2_Mean``
WAI-SR     WAI-SR          ``WAI_TotalMean``
CSQ-8      CSQ-8           ``CSQ8_Mean``
MI-SAT     MI-SAT          ``MI_Mean``
MITI       MITI            ``MITI_GlobalMean``
MICI       MICI            ``MICI_Rate`` (inconsistent behaviours per therapist turn;
                           lower is better)
PCT        MIPROC coder    CT / (CT + ST) over the patient utterance codes; NaN when
                           the coder found neither change nor sustain talk
=========  ==============  =====================================================

``Q1+Q2`` = mean(Q1, Q2) is added for every conversation that has both.

Failure semantics: a call that still fails after its retries is logged and left *absent* from
the table (so a re-run retries it); it is never recorded as zero. A PCT row with an empty value is
a valid "undefined" score, not a failure, and is not re-scored.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._async import run_async
from ..questionnaires import (
    CSQ8_LABELS,
    MI_SAT_LABELS,
    MICI_BEHAVIOR_LABELS,
    MICI_GLOBAL_LABELS,
    MIPROC_PT_CODES,
    MIPROC_TH_CODES,
    MITI_BEHAVIOR_LABELS,
    MITI_GLOBAL_LABELS,
    PCT_BEHAVIOR_LABELS,
    PCT_GLOBAL_LABELS,
    Q1_LABELS,
    Q2_LABELS,
    WAI_SR_LABELS,
    WAI_SR_SUBSCALES,
    QuestionnaireID,
    _count_patient_utterances,
    _count_therapist_utterances,
    get_prompt_eval_questionnaire,
    parse_json_response,
)
from .judges import (
    DEFAULT_OPENAI_JUDGE,
    EVAL_TEMPERATURE,
    JudgeSpec,
    call_judge_json,
    make_judge_client,
)

logger = logging.getLogger(__name__)


INSTRUMENTS = ["Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "MICI", "PCT"]
COMPOSITE = "Q1+Q2"
COLUMNS = ["conversation_index", "persona_id", "instrument", "value"]
DEFAULT_CONCURRENCY = 32

_CONV_FILE_RE = re.compile(r"^conversation_(\d+)\.csv$")


# --- Transcript ---


def reconstruct_conversation_text(utterances: list[Any]) -> str:
    """Convert an utterance list into the ``[THERAPIST]: ...`` / ``[PATIENT]: ...`` transcript.

    Roles alternate starting with the therapist; the ``role`` column of the CSV is ignored, which
    is how the reported scores were produced.
    """
    return "\n".join(
        f"[{'THERAPIST' if i % 2 == 0 else 'PATIENT'}]: {utt}" for i, utt in enumerate(utterances)
    )


def load_conversation_utterances(path: str | Path) -> list[Any]:
    """Read the utterance column of one conversation CSV.

    The default ``pandas.read_csv`` parsing is kept on purpose (an empty utterance reads as NaN
    and renders as ``nan`` in the transcript): changing it would change the judge prompt for such
    conversations and break comparability with the reported scores.
    """
    df = pd.read_csv(path)
    return df["conversation"].tolist()


def list_conversation_files(conv_dir: str | Path) -> list[tuple[int, Path]]:
    """``[(i, path)]`` for every ``conversation_{i}.csv`` in ``conv_dir``, sorted by ``i``."""
    out = []
    for p in Path(conv_dir).iterdir():
        m = _CONV_FILE_RE.match(p.name)
        if m and p.is_file():
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_manifest(conv_dir: str | Path) -> dict[int, int]:
    """Read ``manifest.json`` (``{"<i>": persona_id}``); empty when the file is absent."""
    path = Path(conv_dir) / "manifest.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): int(v) for k, v in raw.items()}


# --- Row builders (per-conversation aggregates of one questionnaire) ---

# Simple questionnaires: items keyed by label plus mean and total.
_SIMPLE_ROW_SPECS = {
    QuestionnaireID.Q1: (Q1_LABELS, "Q1_Mean", "Q1_Total"),
    QuestionnaireID.Q2: (Q2_LABELS, "Q2_Mean", "Q2_Total"),
    QuestionnaireID.CSQ8: (CSQ8_LABELS, "CSQ8_Mean", "CSQ8_Total"),
    QuestionnaireID.MI_SAT: (MI_SAT_LABELS, "MI_Mean", "MI_Total"),
}


def _build_simple_row(scores: dict, labels: list, mean_col: str, total_col: str) -> dict:
    row = {k: scores.get(k, np.nan) for k in labels}
    vals = [row[k] for k in labels]
    row[mean_col] = float(np.nanmean(vals))
    row[total_col] = float(np.nansum(vals))
    return row


def _build_wai_sr_row(scores: dict) -> dict:
    row = {k: scores.get(k, np.nan) for k in WAI_SR_LABELS}
    for sub, items in WAI_SR_SUBSCALES.items():
        row[f"{sub}_Mean"] = float(np.nanmean([row.get(i, np.nan) for i in items]))
    row["WAI_TotalMean"] = float(np.nanmean([row[k] for k in WAI_SR_LABELS]))
    row["WAI_TotalSum"] = float(np.nansum([row[k] for k in WAI_SR_LABELS]))
    return row


def _build_miti_row(scores: dict) -> dict:
    # MITI is asymmetric: globals get a mean, behaviours get a total.
    row: dict = {}
    for k in MITI_GLOBAL_LABELS:
        row[k] = scores.get(k, np.nan)
    row["MITI_GlobalMean"] = float(np.nanmean([row[k] for k in MITI_GLOBAL_LABELS]))
    for k in MITI_BEHAVIOR_LABELS:
        row[k] = scores.get(k, np.nan)
    bvals = [row[k] for k in MITI_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
    row["MITI_BehaviorTotal"] = float(np.nansum(bvals)) if bvals else np.nan
    return row


def _build_pct_row(scores: dict) -> dict:
    # PCT questionnaire: 3 globals get a mean; 3 patient-utterance counts get a total plus the
    # derived ChangeProp = CT / (CT + ST). The reported PCT comes from the MIPROC coder instead
    # (see pct_from_coder); this builder serves direct calls with questionnaire 8.
    row = {k: scores.get(k, np.nan) for k in PCT_GLOBAL_LABELS}
    row["PCT_GlobalMean"] = float(np.nanmean([row[k] for k in PCT_GLOBAL_LABELS]))
    for k in PCT_BEHAVIOR_LABELS:
        row[k] = scores.get(k, np.nan)
    bvals = [row[k] for k in PCT_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
    row["PCT_BehaviorTotal"] = float(np.nansum(bvals)) if bvals else np.nan
    ct, st = row.get("PCT_ChangeTalk", np.nan), row.get("PCT_SustainTalk", np.nan)
    denom = (0 if pd.isna(ct) else ct) + (0 if pd.isna(st) else st)
    row["PCT_ChangeProp"] = float(ct / denom) if denom > 0 else np.nan
    return row


def _build_mici_row(scores: dict, n_th_turns: int) -> dict:
    # MICI: severity global + MI-inconsistent behaviour counts (higher = worse).
    # Rate = total inconsistent behaviours per therapist turn.
    row = {k: scores.get(k, np.nan) for k in MICI_GLOBAL_LABELS}
    for k in MICI_BEHAVIOR_LABELS:
        row[k] = scores.get(k, np.nan)
    bvals = [row[k] for k in MICI_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
    total = float(np.nansum(bvals)) if bvals else np.nan
    row["MICI_BehaviorTotal"] = total
    row["MICI_Rate"] = float(total / n_th_turns) if n_th_turns else np.nan
    op = row.get("MICI_OverPraise", np.nan)
    row["MICI_OverPraiseRate"] = (
        float(op / n_th_turns) if (n_th_turns and not pd.isna(op)) else np.nan
    )
    return row


def _build_miproc_row(scores: dict) -> dict:
    """MIPROC (utterance-level MI process coder): per-code counts and rates for both speakers,
    MITI-style proficiency ratios, and the raw code strings as an audit payload.

    Every ratio defaults to ``0.0`` on a zero denominator (never NaN) because a row containing NaN
    is treated as incomplete. The reported PCT therefore must not be read from
    ``MIPROC_ChangeProp``: use :func:`pct_from_coder`, which is NaN when CT + ST == 0. The empty
    code-string payload uses a ``"-"`` sentinel for the same reason.
    """
    th = list(scores.get("MIPROC_TherapistCodes") or [])
    pt = list(scores.get("MIPROC_PatientCodes") or [])
    n_th, n_pt = len(th), len(pt)

    def _ratio(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    row: dict = {"MIPROC_NTh": n_th, "MIPROC_NPt": n_pt}
    th_n = {c: th.count(c) for c in MIPROC_TH_CODES}
    for c in MIPROC_TH_CODES:
        row[f"MIPROC_TH_{c}"] = th_n[c]
        row[f"MIPROC_TH_{c}_rate"] = _ratio(th_n[c], n_th)
    pt_n = {c: pt.count(c) for c in MIPROC_PT_CODES}
    for c in MIPROC_PT_CODES:
        row[f"MIPROC_PT_{c}"] = pt_n[c]
        row[f"MIPROC_PT_{c}_prop"] = _ratio(pt_n[c], n_pt)
    sr, cr, oq, cq = th_n["SR"], th_n["CR"], th_n["OQ"], th_n["CQ"]
    row["MIPROC_PctCR"] = _ratio(cr, sr + cr)
    row["MIPROC_RtoQ"] = _ratio(sr + cr, oq + cq)
    row["MIPROC_PctOQ"] = _ratio(oq, oq + cq)
    row["MIPROC_ChangeProp"] = _ratio(pt_n["CT"], pt_n["CT"] + pt_n["ST"])
    row["MIPROC_MIInconsistentRate"] = _ratio(th_n["PRA"] + th_n["PERS"] + th_n["CONF"], n_th)
    row["MIPROC_ThCodes"] = "|".join(th) or "-"
    row["MIPROC_PtCodes"] = "|".join(pt) or "-"
    return row


def _build_row(qid_enum: QuestionnaireID, scores: dict, conv_str: str = "") -> pd.DataFrame:
    """Dispatch to the row builder for ``qid_enum``; returns a single-row DataFrame.

    ``conv_str`` is used only by MICI, for the per-therapist-turn rate.
    """
    if qid_enum in _SIMPLE_ROW_SPECS:
        labels, mean_col, total_col = _SIMPLE_ROW_SPECS[qid_enum]
        return pd.DataFrame([_build_simple_row(scores, labels, mean_col, total_col)])
    if qid_enum == QuestionnaireID.WAI_SR:
        return pd.DataFrame([_build_wai_sr_row(scores)])
    if qid_enum == QuestionnaireID.MITI:
        return pd.DataFrame([_build_miti_row(scores)])
    if qid_enum == QuestionnaireID.PCT:
        return pd.DataFrame([_build_pct_row(scores)])
    if qid_enum == QuestionnaireID.MICI:
        return pd.DataFrame([_build_mici_row(scores, _count_therapist_utterances(conv_str))])
    if qid_enum == QuestionnaireID.MIPROC:
        return pd.DataFrame([_build_miproc_row(scores)])
    return pd.DataFrame([scores])


def pct_from_coder(row: Mapping[str, Any]) -> float:
    """Reported PCT for one conversation: CT / (CT + ST) over the MIPROC patient codes.

    ``row`` is a MIPROC row (any mapping with ``MIPROC_PT_CT`` / ``MIPROC_PT_ST``). NaN when the
    coder found neither change nor sustain talk, or when the counts are missing.
    """
    try:
        ct, st = float(row["MIPROC_PT_CT"]), float(row["MIPROC_PT_ST"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    den = ct + st
    return ct / den if den > 0 else float("nan")


def _column(name: str) -> Callable[[Mapping[str, Any]], float]:
    def get(row: Mapping[str, Any]) -> float:
        return float(row[name])

    return get


# instrument -> (judge call, reported value extracted from the built row)
INSTRUMENT_SPECS: dict[str, tuple[QuestionnaireID, Callable[[Mapping[str, Any]], float]]] = {
    "Q1": (QuestionnaireID.Q1, _column("Q1_Mean")),
    "Q2": (QuestionnaireID.Q2, _column("Q2_Mean")),
    "WAI-SR": (QuestionnaireID.WAI_SR, _column("WAI_TotalMean")),
    "CSQ-8": (QuestionnaireID.CSQ8, _column("CSQ8_Mean")),
    "MI-SAT": (QuestionnaireID.MI_SAT, _column("MI_Mean")),
    "MITI": (QuestionnaireID.MITI, _column("MITI_GlobalMean")),
    "MICI": (QuestionnaireID.MICI, _column("MICI_Rate")),
    "PCT": (QuestionnaireID.MIPROC, pct_from_coder),
}


# --- One conversation x one questionnaire ---


async def evaluate_conversation_with_judge(
    client: Any,
    judge: JudgeSpec,
    conversation: list[Any],
    questionnaire_id: QuestionnaireID | int,
) -> pd.DataFrame | None:
    """Score one conversation with one questionnaire; return a 1-row DataFrame or ``None``.

    ``None`` means the call or its validation failed after all retries (the error is logged).
    """
    conv_str = reconstruct_conversation_text(conversation)
    qid_enum = (
        questionnaire_id
        if isinstance(questionnaire_id, QuestionnaireID)
        else QuestionnaireID(questionnaire_id)
    )
    try:
        ed = get_prompt_eval_questionnaire(questionnaire=questionnaire_id, conversation=conv_str)
        resp = await call_judge_json(
            client,
            judge,
            ed["prompt"],
            ed["schema"],
            schema_name=f"questionnaire_{qid_enum.value}_evaluation",
        )
        result = parse_json_response(
            response_content=resp,
            questionnaire_id=questionnaire_id,
            labels=ed["labels"],
            # MIPROC length-validates its code arrays against the utterance counts (Claude cannot
            # enforce minItems/maxItems server-side); every other rubric ignores the argument.
            expected_counts=(
                _count_therapist_utterances(conv_str),
                _count_patient_utterances(conv_str),
            ),
        )
        return _build_row(qid_enum, result["scores_dict"], conv_str)
    except Exception as e:
        logger.warning("Judge %s failed on questionnaire %d: %s", judge, qid_enum.value, e)
        return None


async def evaluate_conversation(
    client: Any,
    conversation: list[Any],
    questionnaire_id: QuestionnaireID | int,
    model: str = DEFAULT_OPENAI_JUDGE,
    eval_temperature: float = EVAL_TEMPERATURE,
) -> pd.DataFrame | None:
    """OpenAI-judge shorthand for :func:`evaluate_conversation_with_judge` (seed 42)."""
    judge = JudgeSpec(provider="openai", model=model, temperature=eval_temperature)
    return await evaluate_conversation_with_judge(client, judge, conversation, questionnaire_id)


# --- Whole directory ---


def _normalize_instruments(instruments: Iterable[str] | None) -> list[str]:
    if instruments is None:
        return list(INSTRUMENTS)
    lookup = {name.upper(): name for name in INSTRUMENTS}
    out: list[str] = []
    for raw in instruments:
        name = raw.strip()
        if not name:
            continue
        if name.upper() in (COMPOSITE, "Q1Q2"):
            wanted = ["Q1", "Q2"]
        elif name.upper() in lookup:
            wanted = [lookup[name.upper()]]
        else:
            raise ValueError(
                f"Unknown instrument {raw!r}; choose from {INSTRUMENTS + [COMPOSITE]}."
            )
        out.extend(w for w in wanted if w not in out)
    if not out:
        raise ValueError("No instruments selected.")
    return out


def _format_value(value: float) -> str:
    return "" if value is None or math.isnan(value) else repr(float(value))


class _RowWriter:
    """Appends result rows to the output CSV, one short-lived append per row so that every
    finished call is on disk even if the run is interrupted."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(COLUMNS)

    def write(self, row: dict[str, Any]) -> None:
        if self._path is None:
            return
        with open(self._path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [
                    row["conversation_index"],
                    row["persona_id"],
                    row["instrument"],
                    _format_value(row["value"]),
                ]
            )


def _read_existing(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is not a score file (missing columns {missing}).")
    return [
        {
            "conversation_index": int(r.conversation_index),
            "persona_id": int(r.persona_id),
            "instrument": str(r.instrument),
            "value": float(r.value),
        }
        for r in df[COLUMNS].itertuples(index=False)
    ]


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df.astype({"conversation_index": int, "persona_id": int, "value": float})
    order = {name: i for i, name in enumerate([COMPOSITE, *INSTRUMENTS])}
    df["_order"] = df["instrument"].map(order).fillna(len(order))
    df = df.sort_values(["conversation_index", "_order", "instrument"], kind="stable")
    df = df.drop(columns="_order").reset_index(drop=True)
    return df.astype({"conversation_index": int, "persona_id": int, "value": float})


async def evaluate_run_async(
    conv_dir: str | Path,
    judge: JudgeSpec,
    instruments: Iterable[str] | None = None,
    out_csv: str | Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    client: Any = None,
) -> pd.DataFrame:
    """Score every conversation in ``conv_dir`` with ``judge``; see the module docstring.

    Args:
        conv_dir: Directory holding ``conversation_{i}.csv`` (and optionally ``manifest.json``).
        judge: Judge to call.
        instruments: Subset of :data:`INSTRUMENTS` (``"Q1+Q2"`` selects Q1 and Q2); all by default.
        out_csv: Tidy output CSV; appended to and used for resume. ``None`` keeps results in memory.
        concurrency: Maximum number of judge calls in flight.
        client: Async SDK client; built from environment credentials when ``None``.

    Returns:
        The tidy table (all rows of ``out_csv`` when given, including earlier runs). Pairs that
        failed are absent and listed in ``df.attrs["missing"]``.
    """
    conv_dir = Path(conv_dir)
    if not conv_dir.is_dir():
        raise FileNotFoundError(f"Conversation directory not found: {conv_dir}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    selected = _normalize_instruments(instruments)
    files = list_conversation_files(conv_dir)
    if not files:
        raise FileNotFoundError(f"No conversation_*.csv files in {conv_dir}")

    manifest = load_manifest(conv_dir)
    if manifest:
        absent = [i for i, _ in files if i not in manifest]
        if absent:
            logger.warning(
                "manifest.json has no persona for conversation(s) %s; using the file index",
                absent,
            )
    else:
        logger.info("No manifest.json in %s; persona_id = conversation index", conv_dir)
    persona_of = {i: manifest.get(i, i) for i, _ in files}

    out_path = Path(out_csv) if out_csv is not None else None
    rows = _read_existing(out_path)
    done = {(r["conversation_index"], r["instrument"]) for r in rows}

    todo = [(i, p, name) for i, p in files for name in selected if (i, name) not in done]
    logger.info(
        "Scoring %d conversation(s) x %d instrument(s) with %s: %d call(s) to make, %d done",
        len(files), len(selected), judge, len(todo), len(files) * len(selected) - len(todo),
    )

    writer = _RowWriter(out_path)
    missing: list[tuple[int, str]] = []
    if todo:
        if client is None:
            client = make_judge_client(judge)
        utterances = {i: load_conversation_utterances(p) for i, p in files}
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        completed = 0

        async def _one(i: int, name: str) -> None:
            nonlocal completed
            qid, extract = INSTRUMENT_SPECS[name]
            async with sem:
                rdf = await evaluate_conversation_with_judge(client, judge, utterances[i], qid)
            # A row with any NaN is incomplete (e.g. MICI on a conversation without therapist
            # turns) and is treated like a failed call.
            if rdf is None or rdf.isnull().values.any():
                logger.warning("No %s score for conversation %d", name, i)
                missing.append((i, name))
                return
            row = {
                "conversation_index": i,
                "persona_id": persona_of[i],
                "instrument": name,
                "value": extract(rdf.iloc[0].to_dict()),
            }
            async with lock:
                writer.write(row)
                rows.append(row)
                completed += 1
                if completed % 50 == 0:
                    logger.info("  %d/%d calls done", completed, len(todo))

        await asyncio.gather(*(_one(i, name) for i, _, name in todo))

    # Q1+Q2 composite for every conversation that has both components.
    values = {(r["conversation_index"], r["instrument"]): r for r in rows}
    for i in sorted({r["conversation_index"] for r in rows}):
        if (i, COMPOSITE) in values:
            continue
        q1, q2 = values.get((i, "Q1")), values.get((i, "Q2"))
        if q1 is None or q2 is None:
            continue
        row = {
            "conversation_index": i,
            "persona_id": q1["persona_id"],
            "instrument": COMPOSITE,
            "value": (q1["value"] + q2["value"]) / 2.0,
        }
        writer.write(row)
        rows.append(row)

    if missing:
        logger.warning(
            "%d (conversation, instrument) pair(s) failed and are missing; re-run to retry them",
            len(missing),
        )
    df = _to_frame(rows)
    df.attrs["missing"] = sorted(missing)
    return df


def evaluate_run(
    conv_dir: str | Path,
    judge: JudgeSpec,
    instruments: Iterable[str] | None = None,
    out_csv: str | Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    client: Any = None,
) -> pd.DataFrame:
    """Synchronous wrapper around :func:`evaluate_run_async` (same arguments and result)."""
    return run_async(
        evaluate_run_async(
            conv_dir,
            judge,
            instruments=instruments,
            out_csv=out_csv,
            concurrency=concurrency,
            client=client,
        )
    )
