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
        i
        for i, line in enumerate(source.splitlines())
        if "model, cfg, _ = build_stage_model(" in line
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


# ── the peer launch ─────────────────────────────────────────────────────────────


def test_a_custom_studio_home_with_a_space_stays_one_shell_word(
    cluster, monkeypatch, tmp_path
) -> None:
    """Bare, `[ -f {act} ] && . {act}` split it into words, the peer venv was not activated,
    and the launch failed on a torchrun that a non-interactive SSH PATH does not have."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "my studio"))
    quoted = cluster.venv_activate_sh()
    assert quoted.startswith('"') and quoted.endswith('"')
    fragment = f"[ -f {quoted} ] && . {quoted}"
    # What the peer's shell sees, inside the `bash -c '...'` the callers build.
    assert subprocess.run(["bash", "-c", f"{fragment}; true"]).returncode == 0
    words = subprocess.run(
        ["bash", "-c", f"set -- {quoted}; echo $#"], capture_output = True, text = True
    )
    assert words.stdout.strip() == "1", words.stdout


def test_the_default_still_expands_home_on_the_peer(cluster, monkeypatch) -> None:
    """No regression: the default is deliberately `$HOME/...`, resolved on the PEER."""
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    quoted = cluster.venv_activate_sh()
    out = subprocess.run(
        ["bash", "-c", f"HOME=/somewhere; echo {quoted}"], capture_output = True, text = True
    )
    assert out.stdout.strip() == "/somewhere/.unsloth/studio/unsloth_studio/bin/activate"


def test_the_peer_stage_is_stopped_when_the_local_one_fails(cluster, monkeypatch) -> None:
    """It was launched under `setsid nohup` and abandoned, so it held its model and CUDA
    context until the distributed timeout and the next provision refused the peer as busy."""
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a"')
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None)
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(
        cluster, "stage_run_inputs", lambda command, ip, user, home: (command, [], [])
    )
    stopped = []
    monkeypatch.setattr(
        cluster,
        "stop_peer_by_pidfile",
        lambda ip, user, opts, pid_file: stopped.append(pid_file) or True,
    )

    class _Done:
        def __init__(self, rc):
            self.returncode = rc

    monkeypatch.setattr(cluster.subprocess, "run", lambda *a, **k: _Done(0 if "ssh" in a[0] else 1))
    plan = {"env": {}, "node0": "true", "node1": "true", "peer_ip": "192.0.2.7"}
    assert cluster.run_pipeline(plan) == 1
    assert stopped == [cluster._PEER_STAGE_PID]


def test_a_successful_run_leaves_the_peer_to_finish(cluster, monkeypatch) -> None:
    """Rank 1 writes its own stage under --save after the last step; killing it there would
    truncate the half of the checkpoint it owns."""
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a"')
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None)
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(
        cluster, "stage_run_inputs", lambda command, ip, user, home: (command, [], [])
    )
    stopped = []
    monkeypatch.setattr(
        cluster,
        "stop_peer_by_pidfile",
        lambda ip, user, opts, pid_file: stopped.append(pid_file) or True,
    )

    class _Done:
        returncode = 0

    monkeypatch.setattr(cluster.subprocess, "run", lambda *a, **k: _Done())
    plan = {"env": {}, "node0": "true", "node1": "true", "peer_ip": "192.0.2.7"}
    assert cluster.run_pipeline(plan) == 0
    assert stopped == []


def test_rank_1_is_pointed_at_the_inputs_that_were_staged(cluster, monkeypatch, tmp_path) -> None:
    """Both ranks got the same --data and --model, and rank 1 runs from its own $HOME with
    only the venv, the bundle and the kernel caches copied to it."""
    data = tmp_path / "rows.jsonl"
    data.write_text('{"q": "a", "a": "b"}\n', encoding = "utf-8")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    copied = []
    monkeypatch.setattr(
        cluster,
        "_rsync_to_peer",
        lambda local, remote, ip, user: copied.append((local, remote)) and None or None,
    )
    command = f"torchrun x --model {checkpoint} --data {data} --steps 4"
    rewritten, staged, failed = cluster.stage_run_inputs(command, "192.0.2.7", "u", "/home/bob")
    assert failed == []
    assert "/home/bob/.unsloth/spark_inputs/ckpt" in rewritten
    assert "/home/bob/.unsloth/spark_inputs/rows.jsonl" in rewritten
    assert str(tmp_path) not in rewritten
    assert [remote for _, remote in copied] == [
        "/home/bob/.unsloth/spark_inputs/ckpt",
        "/home/bob/.unsloth/spark_inputs/rows.jsonl",
    ]


def test_a_repo_id_is_staged_from_the_local_hf_cache(cluster, monkeypatch, tmp_path) -> None:
    """Cached only on the initiating Spark, the peer would refetch it at internet speed over
    a link that moves 444 MB/s, or fail."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cached = tmp_path / ".cache" / "huggingface" / "hub" / "models--org--m"
    cached.mkdir(parents = True)
    copied = []
    monkeypatch.setattr(
        cluster, "_rsync_to_peer", lambda local, remote, ip, user: copied.append(remote) and None
    )
    rewritten, staged, failed = cluster.stage_run_inputs(
        "torchrun x --model org/m", "192.0.2.7", "u", "/home/bob"
    )
    assert failed == []
    # The repo id is still the repo id; only the cache moved.
    assert "--model org/m" in rewritten
    assert copied == ["/home/bob/.cache/huggingface/hub/models--org--m"]


def test_a_model_that_is_neither_a_path_nor_cached_is_left_alone(
    cluster, monkeypatch, tmp_path
) -> None:
    """No regression: a repo id the peer can fetch for itself needs nothing staged."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cluster, "_rsync_to_peer", lambda *a: pytest.fail("nothing should have been copied")
    )
    rewritten, staged, failed = cluster.stage_run_inputs(
        "torchrun x --model org/m", "192.0.2.7", "u", "/home/bob"
    )
    assert (staged, failed) == ([], [])
    assert "--model org/m" in rewritten


def test_a_failed_stage_is_reported_and_not_silently_skipped(
    cluster, monkeypatch, tmp_path
) -> None:
    data = tmp_path / "rows.jsonl"
    data.write_text("{}\n", encoding = "utf-8")
    monkeypatch.setattr(cluster, "_rsync_to_peer", lambda *a: "permission denied")
    _, staged, failed = cluster.stage_run_inputs(
        f"torchrun x --data {data}", "192.0.2.7", "u", "/home/bob"
    )
    assert staged == [] and failed and "permission denied" in failed[0]


def test_the_peer_stages_are_collected_after_a_successful_run(
    cluster, monkeypatch, tmp_path
) -> None:
    """`spark merge` reads every stage from ONE local directory and says it needs no second
    Spark, while rank 1 wrote its stage on the peer."""
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a"')
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None)
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(
        cluster, "stage_run_inputs", lambda command, ip, user, home: (command, [], [])
    )
    collected = []
    monkeypatch.setattr(
        cluster,
        "collect_stage_outputs",
        lambda save, ip, user, home: collected.append((save, home)) and None,
    )

    class _Done:
        returncode = 0

    monkeypatch.setattr(cluster.subprocess, "run", lambda *a, **k: _Done())
    plan = {
        "env": {},
        "node0": f"torchrun x --save {tmp_path / 'out'}",
        "node1": "torchrun x",
        "peer_ip": "192.0.2.7",
    }
    assert cluster.run_pipeline(plan) == 0
    assert collected == [(str(tmp_path / "out"), "/home/peer")]


def test_a_run_without_save_collects_nothing(cluster, monkeypatch) -> None:
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate_sh", lambda: '"$HOME/a"')
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None)
    monkeypatch.setattr(cluster, "peer_home", lambda ip, user: "/home/peer")
    monkeypatch.setattr(
        cluster, "stage_run_inputs", lambda command, ip, user, home: (command, [], [])
    )
    monkeypatch.setattr(
        cluster, "collect_stage_outputs", lambda *a: pytest.fail("nothing to collect")
    )

    class _Done:
        returncode = 0

    monkeypatch.setattr(cluster.subprocess, "run", lambda *a, **k: _Done())
    plan = {"env": {}, "node0": "torchrun x", "node1": "torchrun x", "peer_ip": "192.0.2.7"}
    assert cluster.run_pipeline(plan) == 0


# ── what the legacy backend will not pretend to reproduce ───────────────────────


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_eager_attention_is_refused_on_the_legacy_backend() -> None:
    """Its forwards pass no attention_mask, and `eager_attention_forward` adds one only
    `if attention_mask is not None`, so the run would train bidirectionally."""
    pipeline = _load("spark_pipeline_for_attn", "studio/spark_pipeline.py")
    problem = pipeline.legacy_attention_problem(_Cfg(_attn_implementation = "eager"), 512)
    assert problem and "bidirectional" in problem


@pytest.mark.parametrize("impl", ["sdpa", "flash_attention_2", "flash_attention_3"])
def test_the_kernels_that_derive_causality_are_untouched(impl: str) -> None:
    """No regression: with attention_mask None these select the causal kernel themselves,
    which is why the legacy backend was correct for them and is kept as a control arm."""
    pipeline = _load("spark_pipeline_for_attn", "studio/spark_pipeline.py")
    assert pipeline.legacy_attention_problem(_Cfg(_attn_implementation = impl), 4096) is None


def test_a_sequence_past_the_sliding_window_is_refused() -> None:
    pipeline = _load("spark_pipeline_for_attn", "studio/spark_pipeline.py")
    cfg = _Cfg(_attn_implementation = "sdpa", sliding_window = 1024)
    assert pipeline.legacy_attention_problem(cfg, 2048)
    # At or below the window the two masks are the same matrix.
    assert pipeline.legacy_attention_problem(cfg, 1024) is None


def test_a_model_that_cannot_fit_the_split_still_needs_a_full_finetune_refusal() -> None:
    pipeline = _load("spark_pipeline_for_attn", "studio/spark_pipeline.py")
    assert pipeline.full_finetune_save_problem(True, "out", 2)
    # Single node saves a complete checkpoint, and LoRA merges.
    assert pipeline.full_finetune_save_problem(True, "out", 1) is None
    assert pipeline.full_finetune_save_problem(False, "out", 2) is None
    assert pipeline.full_finetune_save_problem(True, "", 2) is None


def test_the_chat_template_requirement_is_checked_before_the_model_is_built() -> None:
    """`apply_chat_template` raised on a base tokenizer, after both ranks had materialised."""
    source = (REPO / "studio" / "spark_pipeline.py").read_text(encoding = "utf-8")
    lines = source.splitlines()
    check = next(i for i, line in enumerate(lines) if 'getattr(tok, "chat_template", None)' in line)
    build = next(i for i, line in enumerate(lines) if "model, cfg, _ = build_stage_model(" in line)
    assert check < build, (check, build)


def test_the_emitted_serve_commands_quote_the_model_path() -> None:
    source = (REPO / "studio" / "spark_cluster.py").read_text(encoding = "utf-8")
    serve = source.split("def _cmd_serve(")[1].split("\ndef ")[0]
    assert "qmodel = shlex.quote(model)" in serve
    assert "-m {model}" not in serve, "a bare model path is still interpolated into a command"


def test_the_launch_records_the_pid_it_would_stop() -> None:
    source = (REPO / "studio" / "spark_cluster.py").read_text(encoding = "utf-8")
    launch = source.split("def run_pipeline(")[1].split("\ndef ")[0]
    assert "echo $! > " in launch and "_PEER_STAGE_PID" in launch


def test_parity_only_does_not_run_the_runtime_imports() -> None:
    """`--parity-only` is documented as "No GPU work, no NCCL run", and the runtime half of
    the fast-path probe imports torch and the native kernel packages on both nodes."""
    for relative in ("unsloth_cli/commands/doctor.py", "unsloth_cli/commands/spark.py"):
        source = (REPO / relative).read_text(encoding = "utf-8")
        assert "check_fastpath(peer_ip, runtime = deep or not parity_only)" in source, relative
