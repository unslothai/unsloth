# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""CPU tests for the transformers security fixes in unsloth/import_fixes.py:
CVE-2026-4372 (config.json `_attn_implementation_internal` picks a Hub kernel, fixed in 5.3.0),
CVE-2026-5241 (LightGlue config.json `trust_remote_code`, fixed in 5.5.0) and
CVE-2026-9856 (named chat template path traversal on save, fixed in 5.10.0).
Each fix must block its exploit on an affected transformers, leave benign loads and saves
unchanged, apply only once, and install nothing on a fixed transformers."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from packaging.version import Version  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.configuration_utils import PretrainedConfig  # noqa: E402
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: E402

try:
    from transformers.processing_utils import ProcessorMixin
except Exception:  # pragma: no cover
    ProcessorMixin = None

REPO = Path(__file__).resolve().parent.parent
TF_VERSION = Version(transformers.__version__.split("+")[0].split("rc")[0].split(".dev")[0])
KERNEL_AFFECTED = TF_VERSION < Version("5.3.0")
LIGHTGLUE_AFFECTED = Version("4.54.0") <= TF_VERSION < Version("5.5.0")
TEMPLATE_AFFECTED = TF_VERSION < Version("5.10.0")


def _load_import_fixes():
    # Loaded by path so the test needs neither a GPU nor unsloth_zoo; UNSLOTH_IMPORT_FIXES lets a
    # base-vs-head run point the same tests at another revision.
    path = os.environ.get("UNSLOTH_IMPORT_FIXES", str(REPO / "unsloth" / "import_fixes.py"))
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_security", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unwrapped(owner, name, flag):
    current = owner.__dict__.get(name)
    function = current.__func__ if isinstance(current, classmethod) else current
    while function is not None and getattr(function, flag, False):
        function = function.__wrapped__
    return classmethod(function) if isinstance(current, classmethod) else function


_TARGETS = [
    (PretrainedConfig, "from_dict", "_unsloth_patched_untrusted_config_fields"),
    (PreTrainedTokenizerBase, "save_pretrained", "_unsloth_patched_chat_template_names"),
]
if ProcessorMixin is not None:
    _TARGETS.append((ProcessorMixin, "save_pretrained", "_unsloth_patched_chat_template_names"))


@pytest.fixture
def unpatched():
    # conftest's `import unsloth` may already have installed the fixes, so strip them first and
    # put back exactly what was there afterwards.
    saved = [(owner, name, owner.__dict__.get(name)) for owner, name, _ in _TARGETS]
    for owner, name, flag in _TARGETS:
        setattr(owner, name, _unwrapped(owner, name, flag))
    yield
    for owner, name, original in saved:
        setattr(owner, name, original)


@pytest.fixture
def patched(unpatched):
    fixes = _load_import_fixes()
    fixes.fix_transformers_untrusted_config_fields()
    fixes.fix_transformers_chat_template_path_traversal()
    return fixes


@pytest.fixture
def kernel_fetches(monkeypatch):
    from transformers.integrations import hub_kernels

    requested = []

    def get_kernel(repo_id, *args, **kwargs):
        requested.append(repo_id)
        raise RuntimeError(f"blocked fetch of {repo_id}")

    # 4.x only binds get_kernel when `kernels` imports, hence raising = False.
    monkeypatch.setattr(hub_kernels, "get_kernel", get_kernel, raising = False)
    # Stands in for an installed `kernels` package, which the exploit needs.
    monkeypatch.setattr(hub_kernels, "_kernels_available", True, raising = False)
    return requested


def _tiny_llama(path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        max_position_embeddings = 32,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(path)
    return path


def _tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    backend = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "[UNK]").save_pretrained(path)
    return path


def _crafted_kernel_repo(tmp_path):
    repo = _tiny_llama(tmp_path / "kernel_repo")
    config_path = repo / "config.json"
    config = json.loads(config_path.read_text())
    config["_attn_implementation_internal"] = "attacker/evil-kernel"
    config_path.write_text(json.dumps(config))
    return repo


def _crafted_lightglue_repo(tmp_path):
    marker = tmp_path / "LIGHTGLUE_CODE_RAN"
    detector = tmp_path / "evil_detector"
    detector.mkdir()
    (detector / "configuration_evil.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
        "from transformers import PretrainedConfig\n"
        "class EvilConfig(PretrainedConfig):\n"
        "    model_type = 'evil_det'\n"
    )
    (detector / "config.json").write_text(
        json.dumps(
            {"model_type": "evil_det", "auto_map": {"AutoConfig": "configuration_evil.EvilConfig"}}
        )
    )
    repo = tmp_path / "lightglue_repo"
    repo.mkdir()
    (repo / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lightglue",
                "trust_remote_code": True,
                "keypoint_detector_config": {
                    "model_type": "evil_det",
                    "_name_or_path": str(detector),
                },
            }
        )
    )
    return repo, marker


def _crafted_template_repo(tmp_path):
    repo = _tiny_tokenizer(tmp_path / "tok_repo")
    config_path = repo / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = [
        {"name": "default", "template": "{{ messages }}"},
        {"name": "../../ESCAPED_TEMPLATE", "template": "written outside"},
    ]
    config_path.write_text(json.dumps(config))
    return repo


def _escaped_files(root):
    return [p for p in Path(root).rglob("ESCAPED_TEMPLATE*")]


# ---------------------------------------------------------------------------
# Wiring: the fixes run at `import unsloth`, on both the GPU and the MLX path.
# ---------------------------------------------------------------------------


def test_fixes_are_called_and_deleted_in_gpu_init():
    source = (REPO / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    for name in (
        "fix_transformers_untrusted_config_fields",
        "fix_transformers_chat_template_path_traversal",
    ):
        assert f"\n{name}()\n" in source, f"{name} is never called in _gpu_init.py"
        assert f"del {name}\n" in source, f"{name} is left bound on the unsloth namespace"


def test_fixes_are_called_on_the_mlx_path():
    source = (REPO / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    assert "fix_transformers_untrusted_config_fields as _fix_untrusted_config" in source
    assert "fix_transformers_chat_template_path_traversal as _fix_template_names" in source
    assert "_fix_untrusted_config()" in source and "_fix_template_names()" in source


# ---------------------------------------------------------------------------
# The exploits reproduce without the fix exactly where the advisories say.
# ---------------------------------------------------------------------------


def test_unpatched_kernel_field_reaches_the_kernel_fetch(unpatched, kernel_fetches, tmp_path):
    repo = _crafted_kernel_repo(tmp_path)
    try:
        AutoModelForCausalLM.from_pretrained(repo)
    except Exception:
        pass
    assert (kernel_fetches == ["attacker/evil-kernel"]) == KERNEL_AFFECTED


@pytest.mark.skipif(TF_VERSION < Version("4.54.0"), reason = "LightGlue was added in 4.54.0")
def test_unpatched_lightglue_runs_detector_code(unpatched, tmp_path):
    repo, marker = _crafted_lightglue_repo(tmp_path)
    try:
        AutoConfig.from_pretrained(repo)
    except Exception:
        pass
    assert marker.exists() == LIGHTGLUE_AFFECTED


def test_unpatched_template_name_escapes_save_dir(unpatched, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(_crafted_template_repo(tmp_path))
    save_dir = tmp_path / "out" / "a" / "saved"
    try:
        tokenizer.save_pretrained(save_dir)
    except ValueError:
        pass
    assert bool(_escaped_files(tmp_path / "out")) == TEMPLATE_AFFECTED


# ---------------------------------------------------------------------------
# With the fix: blocked, benign behaviour unchanged.
# ---------------------------------------------------------------------------


def test_kernel_field_from_config_json_is_dropped(patched, kernel_fetches, tmp_path):
    repo = _crafted_kernel_repo(tmp_path)
    config = AutoConfig.from_pretrained(repo)
    assert config.__dict__.get("_attn_implementation_internal") is None
    model = AutoModelForCausalLM.from_pretrained(repo)
    assert kernel_fetches == []
    assert model.config._attn_implementation in ("eager", "sdpa")


@pytest.mark.skipif(
    not KERNEL_AFFECTED, reason = "fix installs nothing here; upstream owns explicit kernel loading"
)
def test_explicit_kernel_attn_implementation_still_reaches_the_hub(
    patched, kernel_fetches, tmp_path
):
    repo = _tiny_llama(tmp_path / "plain")
    try:
        AutoModelForCausalLM.from_pretrained(
            repo, attn_implementation = "kernels-community/flash-attn"
        )
    except Exception:
        pass  # the stub refuses the fetch; what matters is that the caller's choice got there
    assert kernel_fetches == ["kernels-community/flash-attn"]


def test_explicit_attn_implementation_is_kept(patched, tmp_path):
    repo = _tiny_llama(tmp_path / "plain")
    assert (
        AutoConfig.from_pretrained(repo, attn_implementation = "eager")._attn_implementation
        == "eager"
    )
    model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation = "eager")
    assert model.config._attn_implementation == "eager"


@pytest.mark.skipif(TF_VERSION < Version("4.54.0"), reason = "LightGlue was added in 4.54.0")
def test_lightglue_config_cannot_opt_into_remote_code(patched, tmp_path):
    repo, marker = _crafted_lightglue_repo(tmp_path)
    try:
        AutoConfig.from_pretrained(repo)
    except Exception:
        pass
    assert not marker.exists()


@pytest.mark.skipif(TF_VERSION < Version("4.54.0"), reason = "LightGlue was added in 4.54.0")
def test_stock_lightglue_config_still_loads(patched, tmp_path):
    from transformers import LightGlueConfig

    repo = tmp_path / "stock_lightglue"
    LightGlueConfig().save_pretrained(repo)
    config = AutoConfig.from_pretrained(repo)
    assert config.keypoint_detector_config.model_type == "superpoint"


def test_traversing_template_name_is_refused_before_writing(patched, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(_crafted_template_repo(tmp_path))
    with pytest.raises(ValueError, match = "Invalid chat template name"):
        tokenizer.save_pretrained(tmp_path / "out" / "a" / "saved")
    assert _escaped_files(tmp_path / "out") == []


def test_stock_template_names_still_save_and_reload(patched, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(_tiny_tokenizer(tmp_path / "tok"))
    names = {"default": "{{ messages }}", "tool_use": "T", "rag": "R", "v2.1-x_y": "V"}
    tokenizer.chat_template = dict(names)
    save_dir = tmp_path / "saved"
    tokenizer.save_pretrained(save_dir)
    reloaded = AutoTokenizer.from_pretrained(save_dir)
    assert reloaded.chat_template == names


def test_single_string_template_still_saves(patched, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(_tiny_tokenizer(tmp_path / "tok"))
    tokenizer.chat_template = "{{ messages }}"
    tokenizer.save_pretrained(tmp_path / "saved")
    assert AutoTokenizer.from_pretrained(tmp_path / "saved").chat_template == "{{ messages }}"


def test_benign_model_roundtrip_is_unchanged(patched, tmp_path):
    repo = _tiny_llama(tmp_path / "plain")
    ids = torch.tensor([[1, 2, 3, 4]])
    model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation = "eager")
    model.save_pretrained(tmp_path / "resaved")
    again = AutoModelForCausalLM.from_pretrained(tmp_path / "resaved", attn_implementation = "eager")
    with torch.no_grad():
        assert torch.equal(model(ids).logits, again(ids).logits)


# ---------------------------------------------------------------------------
# Idempotent, and nothing installed on a transformers that carries the fix.
# ---------------------------------------------------------------------------


def _wrap_depth(owner, name, flag):
    current = owner.__dict__.get(name)
    function = current.__func__ if isinstance(current, classmethod) else current
    depth = 0
    while getattr(function, flag, False):
        depth += 1
        function = function.__wrapped__
    return depth


def test_double_apply_is_a_noop(patched):
    patched.fix_transformers_untrusted_config_fields()
    patched.fix_transformers_chat_template_path_traversal()
    expected_config = 1 if (KERNEL_AFFECTED or LIGHTGLUE_AFFECTED) else 0
    expected_save = 1 if TEMPLATE_AFFECTED else 0
    assert _wrap_depth(*_TARGETS[0]) == expected_config
    for owner, name, flag in _TARGETS[1:]:
        assert _wrap_depth(owner, name, flag) == expected_save


def test_fixed_transformers_is_left_untouched(unpatched):
    before = [owner.__dict__.get(name) for owner, name, _ in _TARGETS]
    fixes = _load_import_fixes()
    fixes.fix_transformers_untrusted_config_fields()
    fixes.fix_transformers_chat_template_path_traversal()
    after = [owner.__dict__.get(name) for owner, name, _ in _TARGETS]
    if not (KERNEL_AFFECTED or LIGHTGLUE_AFFECTED):
        assert after[0] is before[0]
    if not TEMPLATE_AFFECTED:
        assert all(a is b for a, b in zip(after[1:], before[1:]))


@pytest.mark.skipif(TF_VERSION < Version("4.54.0"), reason = "LightGlue was added in 4.54.0")
def test_lightglue_class_ignores_a_disguised_model_type(patched, tmp_path):
    # The file can claim any model_type; LightGlueConfig.from_pretrained still builds LightGlue.
    from transformers import LightGlueConfig

    repo, marker = _crafted_lightglue_repo(tmp_path)
    config = json.loads((repo / "config.json").read_text())
    config["model_type"] = "not_lightglue"
    (repo / "config.json").write_text(json.dumps(config))
    try:
        LightGlueConfig.from_pretrained(repo)
    except Exception:
        pass
    assert not marker.exists()


def test_template_names_checked_even_without_jinja_files(patched):
    # Processor save_pretrained writes named templates without consulting save_jinja_files.
    class Holder:
        chat_template = {"default": "x", "../../escape": "y"}

    with pytest.raises(ValueError, match = "Invalid chat template name"):
        patched._check_chat_template_names(Holder(), {"save_jinja_files": False})


def test_windows_drive_relative_template_name_is_refused(patched, monkeypatch):
    # On Windows `C:evil` joins onto a C: probe base unchanged but lands outside a save
    # directory on another drive. Simulate ntpath with a drive on the probe base.
    import ntpath
    import types

    win = types.SimpleNamespace(
        **{k: getattr(ntpath, k) for k in dir(ntpath) if not k.startswith("__")}
    )
    win.abspath = lambda p: ntpath.normpath(p if ntpath.splitdrive(p)[0] else "C:" + p)
    monkeypatch.setattr(patched, "os", types.SimpleNamespace(path = win, sep = "\\"))
    assert patched._chat_template_name_escapes("C:evil")
    assert patched._chat_template_name_escapes("..\\evil")
    for name in ("default", "tool_use", "rag", "v2.1-x_y"):
        assert not patched._chat_template_name_escapes(name)
