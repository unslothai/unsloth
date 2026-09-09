# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flags and probes that were parsed, printed or promised and then not acted on.

Four of a kind, all silent. `--grad-checkpoint` reached argparse and stopped there. The seed
was set after the LoRA adapters had already been initialised, so the one thing it was there to
make reproducible was the one thing it did not cover. `--shard-load` on a .bin checkpoint
failed with `unmaterialised tensors remain`, which names the symptom. And the fast-path probe
imported one symbol out of a group transformers imports together, so a node where the real
import fails could report a matching gate.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_plumbing", "studio/spark_cluster.py")


@pytest.fixture
def doctor():
    return _load("doctor_for_plumbing", "unsloth_cli/commands/doctor.py")


# ── --grad-checkpoint ───────────────────────────────────────────────────────────


def test_grad_checkpoint_reaches_the_stage_process(cluster, monkeypatch) -> None:
    """It parsed, and then the dispatch forwarded only --pipeline-args."""
    seen = {}
    monkeypatch.setattr(
        cluster,
        "_cmd_pipeline",
        lambda model, port, extra, run: seen.update(model = model, extra = extra) or 0,
    )
    assert cluster.main(["train", "--layer-split", "m", "--grad-checkpoint"]) == 0
    assert "--grad-checkpoint" in seen["extra"]


def test_it_is_not_duplicated_when_already_given(cluster, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        cluster, "_cmd_pipeline", lambda model, port, extra, run: seen.update(extra = extra) or 0
    )
    cluster.main(
        [
            "train",
            "--layer-split",
            "m",
            "--grad-checkpoint",
            "--pipeline-args",
            "--steps 4 --grad-checkpoint",
        ]
    )
    assert seen["extra"].count("--grad-checkpoint") == 1


def test_without_the_flag_nothing_is_added(cluster, monkeypatch) -> None:
    """No regression: a run that did not ask for it must not get it."""
    seen = {}
    monkeypatch.setattr(
        cluster, "_cmd_pipeline", lambda model, port, extra, run: seen.update(extra = extra) or 0
    )
    cluster.main(["train", "--layer-split", "m", "--pipeline-args", "--steps 4"])
    assert "--grad-checkpoint" not in seen["extra"]


def test_the_public_wrapper_exposes_the_option() -> None:
    """The pipeline supported it and the estimator recommends it; nothing offered it."""
    source = (REPO / "unsloth_cli" / "commands" / "spark.py").read_text(encoding = "utf-8")
    train = source.split('@spark_app.command("train")')[1].split("@spark_app.command")[0]
    assert '"--grad-checkpoint"' in train
    assert 'extra.append("--grad-checkpoint")' in train


# ── the seed ────────────────────────────────────────────────────────────────────


def test_the_seed_is_set_before_the_model_is_built() -> None:
    """`get_peft_model` initialises the adapter matrices when it is called, so a seed after
    it leaves the trained parameters different on every rank and every run."""
    source = (REPO / "studio" / "spark_pipeline.py").read_text(encoding = "utf-8")
    seeds = [i for i, line in enumerate(source.splitlines()) if "manual_seed(TRAIN_SEED)" in line]
    build = next(
        i for i, line in enumerate(source.splitlines()) if "model, cfg, _ = build_stage_model(" in line
    )
    peft = next(i for i, line in enumerate(source.splitlines()) if "get_peft_model(" in line)
    assert seeds, "the run no longer seeds at all"
    assert min(seeds) < build < peft, (seeds, build, peft)


# ── --shard-load on a .bin checkpoint ───────────────────────────────────────────


def test_a_bin_checkpoint_is_rejected_by_name(tmp_path) -> None:
    pipeline = _load("spark_pipeline_for_plumbing", "studio/spark_pipeline.py")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")

    class _Model:
        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

    with pytest.raises(RuntimeError) as excinfo:
        pipeline._materialise(_Model(), str(tmp_path), None, "cpu", None, lambda *a: None)
    message = str(excinfo.value)
    assert "safetensors" in message and ".bin" in message
    assert "unmaterialised" not in message


# ── the fast-path probe ─────────────────────────────────────────────────────────


def test_the_probe_imports_what_transformers_imports(doctor) -> None:
    """modeling_qwen3_next.py takes causal_conv1d_fn and causal_conv1d_update in ONE try, so
    a missing `_update` leaves both None; and it reaches the delta rule through
    `fla.ops.gated_delta_rule`, a different module from `fla.ops`."""
    # The runtime half is base64'd into the outer probe, so read it at the source.
    source = doctor.FASTPATH_RUNTIME_SOURCE
    assert "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" in source
    assert "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" in source
    assert "fused_recurrent_gated_delta_rule" in source
    assert "from fla.modules import FusedRMSNormGated" in source


def test_the_metadata_probe_still_imports_nothing(doctor) -> None:
    """No regression: `runtime = False` is the half that must not touch the GPU."""
    source = doctor.fastpath_probe_source(runtime = False)
    assert "from causal_conv1d import" not in source


def test_the_peer_probe_resolves_the_login_like_every_other_ssh(doctor, monkeypatch) -> None:
    """Without USER or USERNAME -- a service, a cron job, a container -- it used the literal
    account `nvidia` and reported UNKNOWN on a pair whose other SSH works."""
    monkeypatch.delenv("USER", raising = False)
    monkeypatch.delenv("USERNAME", raising = False)
    monkeypatch.setattr(doctor, "_ssh_login", lambda: "alice")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ssh")
    seen = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(argv, **kwargs):
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _run)
    doctor._run_probe_peer("192.168.200.13", "print(1)")
    assert "alice@192.168.200.13" in seen["argv"]


def test_parity_only_does_not_run_the_runtime_imports() -> None:
    """`--parity-only` is documented as "No GPU work, no NCCL run", and the runtime half of
    the fast-path probe imports torch and the native kernel packages on both nodes."""
    for relative in ("unsloth_cli/commands/doctor.py", "unsloth_cli/commands/spark.py"):
        source = (REPO / relative).read_text(encoding = "utf-8")
        assert "check_fastpath(peer_ip, runtime = deep or not parity_only)" in source, relative
