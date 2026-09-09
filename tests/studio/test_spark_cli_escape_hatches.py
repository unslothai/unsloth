# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Options a command documents have to reach the code that implements them.

Four separate escapes existed only in a backend: `merge(force=True)`, `provision --force`,
`serve --rpc-port`, and the RPC backfill's job count. Each was reachable from the module but
not from the command users are told to run, so the documented recovery was not available at
all. The fifth case is the mirror image: a default hard-coded to one architecture's module
names, in a code path that accepts any architecture with a decoder layer list.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def merge_mod():
    return _load("spark_merge_for_cli", "studio/spark_merge.py")


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_cli", "studio/spark_cluster.py")


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_cli", "studio/spark_pipeline.py")


def test_merge_force_reaches_the_merge_and_the_default_still_refuses(merge_mod, monkeypatch):
    plan = {
        "n_stages": 1,
        "stages": [{"path": "/tmp/stage0", "layers": [0, 1], "n_keys": 4}],
        "layers_covered": [0, 1],
        "ambiguous_keys": [],
        "problems": ["layers 2..7 are in no stage"],
        "ok": False,
    }
    monkeypatch.setattr(merge_mod, "plan_merge", lambda root: plan)
    seen = {}

    def _merge(root, out, force = False):
        seen["force"] = force
        return {"n_tensors": 4, "out": out}

    monkeypatch.setattr(merge_mod, "merge", _merge)

    # The safe default is unchanged: an incomplete stage set is still refused.
    assert merge_mod._cmd_merge("/tmp/run", "/tmp/out") == 1
    assert seen == {}

    assert merge_mod._cmd_merge("/tmp/run", "/tmp/out", force = True) == 0
    assert seen == {"force": True}


def test_the_module_parser_carries_force(merge_mod, monkeypatch):
    seen = {}

    def _cmd(root, out, dry_run, force = False):
        seen["force"] = force
        return 0

    monkeypatch.setattr(merge_mod, "_cmd_merge", _cmd)
    assert merge_mod.main(["/tmp/run", "--out", "/tmp/out", "--force"]) == 0
    assert seen["force"] is True


def test_the_public_merge_and_provision_commands_expose_force():
    source = (REPO / "unsloth_cli" / "commands" / "spark.py").read_text()
    merge_src = source.split('@spark_app.command("merge")')[1].split("@spark_app.command")[0]
    assert '"--force"' in merge_src
    assert "force = force" in merge_src

    prov_src = source.split('@spark_app.command("provision")')[1].split("@spark_app.command")[0]
    assert '"--force"' in prov_src
    assert 'argv.append("--force")' in prov_src


def test_serve_rpc_port_survives_the_delegated_parser(cluster, monkeypatch):
    seen = {}

    def _serve(model, port = 8080, rpc_port = None, ctx = 8192, engines = 2, slots = 16):
        seen.update(model = model, port = port, rpc_port = rpc_port)
        return 0

    monkeypatch.setattr(cluster, "_cmd_serve", _serve)
    # This is exactly what `unsloth spark serve --rpc-port 51000` appends; before the option
    # was registered, argparse exited 2 on it and _cmd_serve was never called.
    assert cluster.main(["serve", "--model", "m.gguf", "--rpc-port", "51000"]) == 0
    assert seen["rpc_port"] == 51000

    seen.clear()
    assert cluster.main(["serve", "--model", "m.gguf"]) == 0
    assert seen["rpc_port"] == cluster.RPC_DEFAULT_PORT


def test_the_rpc_backfill_does_not_expand_an_unassigned_ncpu():
    source = (REPO / "studio" / "setup.sh").read_text()
    body = source.split("_backfill_local_rpc_server() {")[1].split("\n}\n")[0]
    assert "$NCPU" not in body, "NCPU is assigned in the source-build section this path skips"
    assert "_llama_build_jobs" in body
    # And the assignment really is later than the call site, which is what makes it fatal.
    assert source.index("NCPU=$(_llama_build_jobs)") > source.index(
        '_backfill_local_rpc_server "$LLAMA_CPP_DIR"'
    )


def _model_with(names):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            for n in names:
                setattr(self, n, nn.Linear(2, 2))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

    return Model()


def test_lora_targets_are_read_off_the_model_not_assumed(pipeline):
    pytest.importorskip("torch")
    llama = _model_with(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Unchanged for the architectures that already worked, in the same order.
    assert pipeline.lora_target_modules(llama) == list(pipeline.LORA_TARGETS_LLAMA)

    neox = _model_with(["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"])
    targets = pipeline.lora_target_modules(neox)
    assert "query_key_value" in targets and "dense_h_to_4h" in targets
    assert not set(targets) & set(pipeline.LORA_TARGETS_LLAMA)


def test_the_peft_call_no_longer_hard_codes_llama_names(pipeline):
    source = (REPO / "studio" / "spark_pipeline.py").read_text()
    call = source.split("get_peft_model(")[1].split("\n    model.train()")[0]
    assert "lora_target_modules(model)" in call
    assert not re.search(r'"q_proj"', call)
