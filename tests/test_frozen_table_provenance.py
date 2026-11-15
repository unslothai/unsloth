# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""frozen_tables.json provenance record: written on adapter save, checked against the base table on load."""

import contextlib
import json
import os
import shutil
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _frozen_table_parallel as ftp  # noqa: E402

pytestmark = pytest.mark.skipif(ftp.available() is not None, reason = str(ftp.available()))


@pytest.fixture(autouse = True)
def _restore_environ():
    # _base sets UNSLOTH_FROZEN_TABLE / UNSLOTH_ZOO_DISABLE_GPU_INIT for the model it builds; restore them so later
    # tests (and the subprocesses they spawn) see the session's own environment.
    saved = {k: os.environ.get(k) for k in ("UNSLOTH_FROZEN_TABLE", "UNSLOTH_ZOO_DISABLE_GPU_INIT")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _base(path, mode = "ram"):
    os.environ["UNSLOTH_FROZEN_TABLE"] = mode
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from transformers import AutoConfig, Qwen4ExpForCausalLM
    from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp

    patch_qwen4_exp()
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "eager"
    return Qwen4ExpForCausalLM.from_pretrained(path, config = cfg, dtype = torch.float32)


@pytest.fixture(scope = "module")
def checkpoints(tmp_path_factory):
    same = ftp.make_checkpoint(str(tmp_path_factory.mktemp("ckpt")))
    other = str(tmp_path_factory.mktemp("other"))
    for f in os.listdir(same):
        shutil.copy(os.path.join(same, f), os.path.join(other, f))
    # one byte of the table changed: same files, sizes and headers, different rows
    key, fname = sorted(ftp.table_files(other).items())[0]
    from unsloth_zoo.frozen_table_store import resolve_tensor_locations

    loc = resolve_tensor_locations(os.path.join(other, fname), names = [key])[key]
    with open(loc.path, "r+b") as f:
        f.seek(loc.offset + 5)
        b = f.read(1)
        f.seek(loc.offset + 5)
        f.write(bytes([b[0] ^ 0x5A]))
    return same, other


@pytest.fixture
def adapter(checkpoints, tmp_path):
    from peft import LoraConfig, get_peft_model

    provenance = ftp.load_module("frozen_table_provenance")
    assert provenance.install()
    model = get_peft_model(_base(checkpoints[0]), LoraConfig(r = 2, target_modules = ["o_proj"]))
    model.save_pretrained(str(tmp_path / "adapter"))
    return str(tmp_path / "adapter")


def test_adapter_save_writes_the_table_provenance(checkpoints, adapter):
    with open(os.path.join(adapter, "frozen_tables.json")) as f:
        record = json.load(f)
    (table,) = record["tables"].values()
    assert {t["name"] for t in table["tensors"]} == set(ftp.table_files(checkpoints[0]))
    assert all(len(t["sample_sha256"]) == 64 for t in table["tensors"])
    assert set(table["files"]) == {
        os.path.basename(f) for f in ftp.table_files(checkpoints[0]).values()
    }


@contextlib.contextmanager
def _no_provenance_warning():
    """Only the provenance warning is an error: transformers / PEFT may warn about their own deprecations."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message = r"Unsloth: .*frozen lookup tables")
        yield


def test_reload_on_the_same_tables_is_silent(checkpoints, adapter):
    from peft import PeftModel
    with _no_provenance_warning():
        PeftModel.from_pretrained(_base(checkpoints[0], mode = "disk"), adapter, torch_device = "cpu")


def test_reload_on_other_tables_warns_or_refuses(checkpoints, adapter, monkeypatch):
    from peft import PeftModel

    with pytest.warns(UserWarning, match = "other frozen lookup tables"):
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter, torch_device = "cpu")
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "contents differ"):
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter, torch_device = "cpu")
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "off")
    with _no_provenance_warning():
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter, torch_device = "cpu")


def test_reload_with_the_table_resident_is_not_a_mismatch(checkpoints, adapter, monkeypatch):
    # UNSLOTH_FROZEN_TABLE=gpu (or auto on a large GPU) loads the same table as an ordinary embedding, so no table
    # is bound to compare against: that is a residence change, not other base tables.
    from peft import PeftModel

    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with _no_provenance_warning():
        PeftModel.from_pretrained(_base(checkpoints[0], mode = "gpu"), adapter, torch_device = "cpu")
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "warn")
    with _no_provenance_warning():
        PeftModel.from_pretrained(_base(checkpoints[0], mode = "gpu"), adapter, torch_device = "cpu")
    # A saved table the model has no module for at all is still reported.
    provenance = ftp.load_module("frozen_table_provenance")
    with open(os.path.join(adapter, "frozen_tables.json")) as f:
        record = json.load(f)
    record["tables"] = {"model.no_such_layer.table": next(iter(record["tables"].values()))}
    other = os.path.join(os.path.dirname(adapter), "renamed")
    os.makedirs(other, exist_ok = True)
    with open(os.path.join(other, "frozen_tables.json"), "w") as f:
        json.dump(record, f)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "no such frozen table"):
        provenance.check_provenance(_base(checkpoints[0], mode = "gpu"), other)


def test_pipeline_stages_merge_their_tables():
    provenance = ftp.load_module("frozen_table_provenance")
    a = {"format": "unsloth_frozen_tables", "version": 1, "tables": {"model.layers.1.t": {"n": 1}}}
    b = {"format": "unsloth_frozen_tables", "version": 1, "tables": {"model.layers.4.t": {"n": 2}}}
    merged = provenance.merge_provenance([a, None, b])
    assert merged["tables"] == {"model.layers.1.t": {"n": 1}, "model.layers.4.t": {"n": 2}}
    assert provenance.merge_provenance([None, None]) is None


def _has_lora(model, name = None):
    return any(
        "lora_A" in n and (name is None or f".{name}." in n + ".") for n, _ in model.named_modules()
    )


def test_strict_load_adapter_refuses_before_touching_the_model(checkpoints, adapter, monkeypatch):
    """A refused adapter is never installed: the check runs before PEFT loads it."""
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_base(checkpoints[1]), LoraConfig(r = 2, target_modules = ["o_proj"]))
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "other frozen lookup tables"):
        model.load_adapter(adapter, adapter_name = "other")
    assert "other" not in model.peft_config and not _has_lora(model, "other")


def test_strict_from_pretrained_refuses_before_touching_the_base(checkpoints, adapter, monkeypatch):
    """PeftModel.from_pretrained injects the adapter's layers into the base in place: refused first."""
    from peft import PeftModel

    base = _base(checkpoints[1])
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "other frozen lookup tables"):
        PeftModel.from_pretrained(base, adapter, torch_device = "cpu")
    assert not _has_lora(base)


def test_named_adapters_carry_their_own_record(checkpoints, tmp_path):
    """Every saved adapter gets the record, also when adapters are passed positionally."""
    from peft import LoraConfig, get_peft_model

    provenance = ftp.load_module("frozen_table_provenance")
    assert provenance.install()
    model = get_peft_model(_base(checkpoints[0]), LoraConfig(r = 2, target_modules = ["o_proj"]))
    model.add_adapter("second", LoraConfig(r = 2, target_modules = ["o_proj"]))
    out = tmp_path / "both"
    model.save_pretrained(str(out))
    assert (out / "frozen_tables.json").is_file() and (
        out / "second" / "frozen_tables.json"
    ).is_file()
    only = tmp_path / "only"
    model.save_pretrained(str(only), True, ["second"])
    assert (only / "second" / "frozen_tables.json").is_file()
    assert not (only / "frozen_tables.json").exists()


def test_hub_adapter_is_checked_before_loading(checkpoints, adapter, monkeypatch):
    """Hub adapter: the record comes via the Hub cache and strict mode refuses before PEFT fetches."""
    import huggingface_hub
    from peft import PeftModel

    provenance = ftp.load_module("frozen_table_provenance")
    calls = []

    def download(repo_id, filename, **options):
        calls.append((repo_id, filename, options))
        assert filename == provenance.FILE_NAME
        return os.path.join(adapter, filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _base(checkpoints[1])
    with pytest.raises(ValueError, match = "other frozen lookup tables"):
        PeftModel.from_pretrained(
            base, "some-org/some-adapter", revision = "abc123", torch_device = "cpu"
        )
    assert calls == [("some-org/some-adapter", provenance.FILE_NAME, {"revision": "abc123"})]
    assert not _has_lora(base)


def test_hub_sidecar_absent_or_unreachable(monkeypatch):
    """No Hub record: no check; unanswerable: strict refuses, warn goes on; no tables: no Hub call."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    provenance = ftp.load_module("frozen_table_provenance")
    monkeypatch.setattr(provenance, "table_provenance", lambda model: {"tables": {}})
    raised = []

    def download(repo_id, filename, **options):
        raise raised[-1]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    raised.append(EntryNotFoundError("no such file"))
    assert provenance._adapter_dir(None, "org/adapter", {}) is None
    raised.append(LocalEntryNotFoundError("offline"))
    assert provenance._adapter_dir(None, "org/adapter", {}) is None
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "cannot read frozen_tables.json"):
        provenance._adapter_dir(None, "org/adapter", {})
    monkeypatch.setattr(provenance, "table_provenance", lambda model: None)
    raised.append(AssertionError("asked the Hub"))
    assert provenance._adapter_dir(None, "org/adapter", {}) is None


@pytest.mark.parametrize("record_at", ["root", "named"])
def test_native_resume_checks_every_adapter_before_reading(
    checkpoints, tmp_path, monkeypatch, record_at
):
    """native_load_model_checkpoint checks every adapter record before reading any weight."""
    from peft import LoraConfig, get_peft_model

    native = ftp.load_module("fsdp2_native")
    provenance = ftp.load_module("frozen_table_provenance")
    assert provenance.install()

    def two_adapters(path):
        torch.manual_seed(0)
        model = get_peft_model(_base(path), LoraConfig(r = 2, target_modules = ["o_proj"]))
        model.add_adapter("second", LoraConfig(r = 2, target_modules = ["o_proj"]))
        return model

    ckpt = tmp_path / "checkpoint-1"
    saved = two_adapters(checkpoints[0])
    with torch.no_grad():
        for n, p in saved.named_parameters():
            if "lora_" in n:
                p.fill_(0.5)
    saved.save_pretrained(str(ckpt))
    assert (ckpt / "second" / "frozen_tables.json").is_file()
    if record_at == "named":
        os.remove(
            ckpt / "frozen_tables.json"
        )  # only the named adapter's record is left to catch it
    model = two_adapters(checkpoints[1])
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "other frozen lookup tables"):
        native.native_load_model_checkpoint(model, str(ckpt))
    after = dict(model.named_parameters())
    assert all(torch.equal(after[n], v) for n, v in before.items())
