# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The consent scan must read the code a load can run: every model along an adapter chain (a base
with its own adapter_config.json is swapped by transformers) and a checkpoint's EAGLE-3 drafter
folder, which GGUF export converts under the same trust."""

import json

from utils.security import evaluate_remote_code_consent_for_targets

EVIL = "import subprocess\nsubprocess.Popen(['id'])\n"


def _write(folder, name, data):
    folder.mkdir(parents = True, exist_ok = True)
    (folder / name).write_text(
        data if isinstance(data, str) else json.dumps(data), encoding = "utf-8"
    )


def _code_model(folder):
    _write(
        folder, "config.json", {"model_type": "llama", "auto_map": {"AutoModel": "modeling_x.M"}}
    )
    _write(folder, "modeling_x.py", EVIL)


def test_code_two_adapter_hops_away_is_scanned(tmp_path):
    adapter, base, hidden = tmp_path / "adapter", tmp_path / "base", tmp_path / "hidden"
    _write(adapter, "adapter_config.json", {"base_model_name_or_path": str(base)})
    _write(base, "config.json", {"model_type": "llama"})
    _write(base, "adapter_config.json", {"base_model_name_or_path": str(hidden)})
    _code_model(hidden)

    # What the workers and the consent dialog pass: the adapter and its first base.
    decision = evaluate_remote_code_consent_for_targets(
        [str(adapter), str(base)], trust_remote_code = True
    )
    assert decision.has_remote_code is True
    assert decision.blocked is True


def test_a_plain_adapter_chain_without_code_stays_a_noop(tmp_path):
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    _write(adapter, "adapter_config.json", {"base_model_name_or_path": str(base)})
    _write(base, "config.json", {"model_type": "llama"})
    decision = evaluate_remote_code_consent_for_targets(
        [str(adapter), str(base)], trust_remote_code = True
    )
    assert decision.has_remote_code is False
    assert decision.blocked is False


def test_a_chain_loop_terminates(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "adapter_config.json", {"base_model_name_or_path": str(b)})
    _write(b, "adapter_config.json", {"base_model_name_or_path": str(a)})
    decision = evaluate_remote_code_consent_for_targets([str(a)], trust_remote_code = True)
    assert decision.blocked is False


def test_a_drafter_folders_code_is_scanned(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write(ckpt, "config.json", {"model_type": "llama"})
    _write(ckpt, "drafter.json", {"strategy": "eagle3"})
    drafter = ckpt / "eagle3"
    _write(drafter, "config.json", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    _write(
        drafter,
        "tokenizer_config.json",
        {"auto_map": {"AutoTokenizer": ["tokenization_x.T", None]}},
    )
    _write(drafter, "tokenization_x.py", EVIL)

    decision = evaluate_remote_code_consent_for_targets([str(ckpt)], trust_remote_code = True)
    assert decision.has_remote_code is True
    assert decision.blocked is True


def test_a_studio_trained_drafter_without_code_stays_a_noop(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write(ckpt, "config.json", {"model_type": "llama"})
    _write(ckpt / "eagle3", "config.json", {"speculators_model_type": "eagle3"})
    decision = evaluate_remote_code_consent_for_targets([str(ckpt)], trust_remote_code = True)
    assert decision.has_remote_code is False
    assert decision.blocked is False


def test_trust_off_reads_nothing(tmp_path, monkeypatch):
    import utils.security.consent as consent

    monkeypatch.setattr(
        consent,
        "_with_adapter_chains",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    decision = evaluate_remote_code_consent_for_targets([str(tmp_path)], trust_remote_code = False)
    assert decision.blocked is False


def _managed_workspace(monkeypatch, workspace):
    import os

    from utils.paths import storage_roots

    root = os.path.realpath(workspace)
    monkeypatch.setattr(
        storage_roots,
        "within_account",
        lambda path: os.path.realpath(path).startswith(root + os.sep),
    )


def test_a_managed_scan_does_not_follow_a_drafter_link_out_of_its_workspace(tmp_path, monkeypatch):
    workspace, other = tmp_path / "ws", tmp_path / "other"
    ckpt = workspace / "ckpt"
    _write(ckpt, "config.json", {"model_type": "llama"})
    _code_model(other)
    (ckpt / "eagle3").symlink_to(other, target_is_directory = True)
    _managed_workspace(monkeypatch, workspace)

    decision = evaluate_remote_code_consent_for_targets([str(ckpt)], trust_remote_code = True)
    assert decision.has_remote_code is False


def test_a_managed_scan_does_not_follow_a_chain_base_out_of_its_workspace(tmp_path, monkeypatch):
    workspace, other = tmp_path / "ws", tmp_path / "other"
    adapter, base = workspace / "adapter", workspace / "base"
    _write(adapter, "adapter_config.json", {"base_model_name_or_path": str(base)})
    _write(base, "config.json", {"model_type": "llama"})
    _write(base, "adapter_config.json", {"base_model_name_or_path": str(other)})
    _code_model(other)
    _managed_workspace(monkeypatch, workspace)

    decision = evaluate_remote_code_consent_for_targets(
        [str(adapter), str(base)], trust_remote_code = True
    )
    assert decision.has_remote_code is False
