# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DeepSeek-OCR loads on the transformers 4.57.6 sidecar and generates through its own infer() (#8732)."""

from __future__ import annotations

import base64
import importlib
import importlib.machinery
import io
import json
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_STUBBED: list[str] = []


def _stub_if_missing(
    name,
    attrs = (),
    named_spec = False,
):
    """Same helper as test_chat_prompt_bos.py: core.inference.inference imports unsloth, trl and peft."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None) if named_spec else None
    module.__version__ = "0.0.0"
    module.__getattr__ = lambda _attr: MagicMock()
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


for _torchao in (
    "torchao",
    "torchao.prototype",
    "torchao.prototype.safetensors",
    "torchao.prototype.safetensors.safetensors_support",
    "torchao.prototype.safetensors.safetensors_utils",
    "torchao.quantization",
    "torchao.dtypes",
    "torchao.float8",
    "torchao.utils",
):
    _stub_if_missing(_torchao, named_spec = True)

_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("unsloth_zoo")
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))
_stub_if_missing("peft", ("PeftModel", "PeftModelForCausalLM"))

from core.inference.inference import InferenceBackend  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import utils.transformers_version as tv  # noqa: E402

_OCR2 = {
    "architectures": ["DeepseekOCR2ForCausalLM"],
    "model_type": "deepseek_vl_v2",
    "auto_map": {"AutoModel": "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"},
}


def _local_model(tmp_path: Path, config: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("architecture", ["DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM"])
def test_deepseek_ocr_routes_to_the_457_sidecar(tmp_path, monkeypatch, architecture):
    monkeypatch.setattr(tv.sys, "platform", "linux")
    path = _local_model(tmp_path, {**_OCR2, "architectures": [architecture]})
    assert tv.get_transformers_tier(path, probe = False) == "457"


def test_deepseek_vl2_shares_the_model_type_but_keeps_its_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.sys, "platform", "linux")
    path = _local_model(tmp_path, {**_OCR2, "architectures": ["DeepseekVLV2ForCausalLM"]})
    assert tv.get_transformers_tier(path, probe = False) != "457"


def test_macos_keeps_deepseek_ocr_on_mlx(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.sys, "platform", "darwin")
    assert tv.get_transformers_tier(_local_model(tmp_path, _OCR2), probe = False) != "457"


class _OcrModel:
    config = types.SimpleNamespace(architectures = ["DeepseekOCR2ForCausalLM"])

    def __init__(self):
        self.calls = []
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return "ids"

    def infer(self, tokenizer, **kwargs):
        self.calls.append((tokenizer, kwargs, Path(kwargs["image_file"]).is_file()))
        self.generate(max_new_tokens = 8192)  # what the real infer() hardcodes
        return "UNSLOTH 8732 OCR"


def _backend(model):
    backend = InferenceBackend.__new__(InferenceBackend)
    backend._generation_lock = threading.Lock()
    backend.active_model_name = "unsloth/DeepSeek-OCR-2"
    backend.models = {
        backend.active_model_name: {"model": model, "tokenizer": "tok", "is_vision": True}
    }
    return backend


def _generate(
    backend,
    image,
    text = "Free OCR.",
    **kwargs,
):
    return list(
        backend._generate_chat_response_inner(
            messages = [{"role": "user", "content": text}],
            image = image,
            **kwargs,
        )
    )


def _image():
    return pytest.importorskip("PIL.Image").new("RGB", (32, 32), "white")


def test_chat_runs_the_models_own_infer_on_the_attached_image():
    model = _OcrModel()

    assert _generate(_backend(model), _image()) == ["UNSLOTH 8732 OCR"]
    tokenizer, kwargs, image_was_written = model.calls[0]
    assert (tokenizer, kwargs["prompt"], kwargs["eval_mode"]) == ("tok", "<image>\nFree OCR.", True)
    assert image_was_written


def test_the_request_bounds_and_can_cancel_the_generate_infer_runs():
    model = _OcrModel()
    _generate(_backend(model), _image(), max_new_tokens = 300, cancel_event = threading.Event())

    assert model.generate_kwargs["max_new_tokens"] == 300
    assert model.generate_kwargs["stopping_criteria"] is not None
    assert "generate" not in vars(model), "the bound generate must not outlive the request"


def test_the_requested_adapter_state_is_applied_under_the_generation_lock():
    model = _OcrModel()
    backend = _backend(model)
    applied = []
    backend._apply_adapter_state = lambda state: applied.append(
        (state, backend._generation_lock.locked())
    )
    _generate(backend, _image(), _adapter_state = False)

    assert applied == [(False, True)]


def test_chat_without_an_image_gets_a_client_visible_error():
    with pytest.raises(ValueError, match = "attach one") as raised:
        _generate(_backend(_OcrModel()), image = None)
    assert (raised.value.public, raised.value.openai_param) == (True, "messages")


def test_deepseek_ocr_does_not_count_as_a_transformers_5_model(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.sys, "platform", "linux")
    assert tv.needs_transformers_5(_local_model(tmp_path, _OCR2)) is False


def test_the_request_stop_sequence_ends_the_ocr_text():
    assert _generate(_backend(_OcrModel()), _image(), stop = [" 8732"]) == ["UNSLOTH"]


def _page_b64():
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", (1700, 2200), "white").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_the_worker_raises_the_image_cap_only_for_deepseek_ocr():
    from core.inference import worker

    cmd = {"messages": [], "image_base64": _page_b64()}
    other = types.SimpleNamespace(
        config = types.SimpleNamespace(architectures = ["Qwen2VLForConditionalGeneration"])
    )

    def longest_side(model):
        return max(worker._generation_kwargs(_backend(model), cmd, None)["image"].size)

    assert longest_side(_OcrModel()) == 2048
    assert longest_side(other) == 800


def test_the_route_side_resize_uses_the_cap_the_worker_reported():
    Image = pytest.importorskip("PIL.Image")
    from core.inference.orchestrator import InferenceOrchestrator, _mirrored_model_entry

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    page = Image.new("RGB", (1700, 2200), "white")

    orchestrator.models = {"m": _mirrored_model_entry({"max_image_side": 2048}, "m")}
    assert max(orchestrator.resize_image(page).size) == 2048
    orchestrator.models = {"m": _mirrored_model_entry({}, "m")}
    assert max(orchestrator.resize_image(page).size) == 800


@pytest.mark.parametrize(
    ("architecture", "image_size"),
    [("DeepseekOCRForCausalLM", 640), ("DeepseekOCR2ForCausalLM", 768)],
)
def test_each_model_gets_its_own_gundam_tile_size(architecture, image_size):
    model = _OcrModel()
    model.config = types.SimpleNamespace(architectures = [architecture])
    _generate(_backend(model), _image())

    assert model.calls[0][1]["image_size"] == image_size
