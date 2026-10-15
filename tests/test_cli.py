"""Command-line interface: help, dry run, compare, and light imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pto.cli import main


def test_help_lists_subcommands(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("train", "generate", "evaluate", "compare"):
        assert command in out


@pytest.mark.parametrize("command", ["train", "generate", "evaluate", "compare"])
def test_subcommand_help(capsys, command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_train_dry_run(capsys, configs_dir: Path, tmp_path: Path) -> None:
    code = main([
        "train", "--config", str(configs_dir / "quicktest.yaml"),
        "--output-dir", str(tmp_path), "--iterations", "1", "--dry-run",
    ])
    assert code == 0
    out = capsys.readouterr().out
    resolved = yaml.safe_load(out)
    assert resolved["loop"]["num_iterations"] == 1
    assert resolved["output_dir"] == str(tmp_path)
    assert str(tmp_path / resolved["run_name"]) in out
    # A dry run never creates the run directory (no model, no API, no files).
    assert list(tmp_path.iterdir()) == []


def test_train_dry_run_invalid_config(capsys, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("tree:\n  branches: 1\n", encoding="utf-8")
    assert main(["train", "--config", str(bad), "--dry-run"]) != 0
    assert "branches" in capsys.readouterr().err


def _scores(values: np.ndarray, offset: int = 0) -> pd.DataFrame:
    rows = []
    for pid, v in enumerate(values):
        for inst, val in (("Q1", v), ("Q2", v + 0.25)):
            rows.append({"conversation_index": pid + offset, "persona_id": pid,
                         "instrument": inst, "value": val})
    return pd.DataFrame(rows)


def test_compare_writes_markdown(capsys, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(3.0, 0.4, 24)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _scores(base).to_csv(a, index=False)
    _scores(base + 0.6 + rng.normal(0, 0.1, 24), offset=100).to_csv(b, index=False)
    out_md = tmp_path / "out" / "table.md"
    assert main(["compare", str(a), str(b), "--out", str(out_md), "--n-boot", "300"]) == 0
    printed = capsys.readouterr().out
    assert "| Q1 " in printed and "| Q2 " in printed
    written = out_md.read_text(encoding="utf-8")
    assert "| Q1 " in written and "| Q2 " in written
    assert "delta" in written


def test_compare_without_scipy_names_the_extra(monkeypatch, capsys, tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(3.0, 0.4, 8)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _scores(base).to_csv(a, index=False)
    _scores(base + 0.5).to_csv(b, index=False)
    monkeypatch.setitem(sys.modules, "scipy", None)  # makes `from scipy import ...` fail
    assert main(["compare", str(a), str(b), "--n-boot", "50"]) == 1
    err = capsys.readouterr().err
    assert "pto compare: ImportError" in err and "eval" in err


def test_evaluate_cli(monkeypatch, capsys, tmp_path: Path, fake_openai,
                      conversation_dir_writer) -> None:
    import pto.clients

    monkeypatch.setattr(pto.clients, "make_openai_client", lambda *a, **k: fake_openai())
    utterances = ["Hi, what brings you?", "I smoke.", "Tell me more?", "It helps me relax."]
    d = conversation_dir_writer(tmp_path / "convs", [utterances] * 2, [3, 7])
    csv = tmp_path / "scores.csv"
    assert main([
        "evaluate", "--conversations", str(d), "--instruments", "q1, Q2", "--out", str(csv),
    ]) == 0
    assert {"Q1", "Q2", "Q1+Q2"} <= set(pd.read_csv(csv)["instrument"])
    assert "| Q1+Q2 " in capsys.readouterr().out

    assert main([
        "evaluate", "--conversations", str(d), "--instruments", "NOPE",
        "--out", str(tmp_path / "x.csv"),
    ]) == 1
    assert "NOPE" in capsys.readouterr().err


def test_train_without_api_key_fails_cleanly(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("logging:\n  report_to: []\n", encoding="utf-8")
    out_dir = tmp_path / "runs"
    assert main(["train", "--config", str(cfg_file), "--output-dir", str(out_dir)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("pto train: MissingAPIKeyError") and "OPENAI_API_KEY" in err
    assert "Traceback" not in err
    # The key is checked before the run directory is created.
    assert not out_dir.exists()


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )


def test_package_import_is_light() -> None:
    proc = _run_python(
        "import sys, pto, pto.cli, pto.config\n"
        "heavy = [m for m in ('torch', 'trl', 'transformers', 'openai', 'anthropic') "
        "if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "print(pto.__version__)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1.0.0"


def test_help_does_not_import_torch() -> None:
    proc = _run_python(
        "import sys\n"
        "from pto.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_python_m_pto_help() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-m", "pto", "--help"], capture_output=True, text=True, env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "train" in proc.stdout
