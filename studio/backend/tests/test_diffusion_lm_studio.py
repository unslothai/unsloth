# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-diffusion LMs (LLaDA, LLaDA-MoE, Dream, Nemotron-Labs-Diffusion, LLaDA2, SDAR, DiffusionGemma) in
Studio: detection, LoRA targets, trainer choice, packing, export limits and chat dispatch. Every other model
keeps its existing path."""

from __future__ import annotations

import inspect
import json
import threading
from types import SimpleNamespace

import pytest

from utils.models import diffusion_lm
from utils.models.diffusion_lm import (
    DIFFUSION_LM_PROFILES,
    diffusion_lm_export_limits,
    diffusion_lm_profile,
    diffusion_lm_profile_for_model,
    export_refusal,
)

# Raw config.json fields of the published checkpoints (tiny copies carry the same ones).
DIFFUSION_CONFIGS = {
    "llada": {"model_type": "llada", "architectures": ["LLaDAModelLM"]},
    "llada_moe": {"model_type": "llada", "architectures": ["LLaDAMoEModelLM"]},
    "dream": {"model_type": "Dream", "architectures": ["DreamModel"]},
    "nemotron_labs_diffusion": {
        "model_type": "nemotron_labs_diffusion",
        "architectures": ["NemotronLabsDiffusionModel"],
    },
    "llada2": {"model_type": "llada2_moe", "architectures": ["LLaDA2MoeModelLM"]},
    "sdar": {"model_type": "sdar", "architectures": ["SDARForCausalLM"]},
    "diffusion_gemma": {
        "model_type": "diffusion_gemma4",
        "architectures": ["DiffusionGemma4ForBlockDiffusion"],
    },
}
AUTOREGRESSIVE_CONFIGS = [
    {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"]},
    {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]},
    # Autoregressive Nemotron shares the backbone name, never the diffusion profile.
    {"model_type": "nemotron", "architectures": ["NemotronForCausalLM"]},
    {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]},
]


@pytest.mark.parametrize("profile", sorted(DIFFUSION_CONFIGS))
def test_each_family_is_detected_from_raw_config(profile):
    assert diffusion_lm_profile(DIFFUSION_CONFIGS[profile]) == profile


@pytest.mark.parametrize("config", AUTOREGRESSIVE_CONFIGS, ids = lambda c: c["model_type"])
def test_autoregressive_models_are_not_diffusion(config):
    assert diffusion_lm_profile(config) is None


def test_legacy_diffusion_gemma_type_and_odd_inputs():
    assert diffusion_lm_profile({"model_type": "diffusion_gemma"}) == "diffusion_gemma"
    assert diffusion_lm_profile({"architectures": "SDARForCausalLM"}) == "sdar"
    assert diffusion_lm_profile(None) is None
    assert diffusion_lm_profile({}) is None


def test_mirror_matches_the_core_registry():
    """Same names, same model_types / architectures, same order as Unsloth's own profile registry."""
    profiles_mod = pytest.importorskip("unsloth.models.diffusion_profiles")
    core = [
        (p.name, tuple(p.model_types), tuple(p.architectures))
        for p in profiles_mod.diffusion_profiles()
    ]
    assert core == [(n, tuple(m), tuple(a)) for n, m, a in DIFFUSION_LM_PROFILES]
    # And the core resolver agrees with the mirror on every sample config.
    for name, cfg in DIFFUSION_CONFIGS.items():
        resolved = profiles_mod.resolve_diffusion_profile(SimpleNamespace(**cfg))
        assert resolved is not None and resolved.name == name


def _write_config(path, config):
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return path


def test_local_checkpoint_and_adapter_resolve_to_the_base(tmp_path):
    base = _write_config(tmp_path / "base", DIFFUSION_CONFIGS["sdar"])
    assert diffusion_lm_profile_for_model(str(base)) == "sdar"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding = "utf-8"
    )
    assert diffusion_lm_profile_for_model(str(adapter)) == "sdar"
    ar = _write_config(tmp_path / "ar", AUTOREGRESSIVE_CONFIGS[0])
    assert diffusion_lm_profile_for_model(str(ar)) is None
    # An unreadable folder is "not detected", never an exception.
    assert diffusion_lm_profile_for_model(str(tmp_path / "missing")) is None


def test_adapter_pointing_at_itself_does_not_loop(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(adapter)}), encoding = "utf-8"
    )
    assert diffusion_lm_profile_for_model(str(adapter)) is None


# ---- export ---------------------------------------------------------------------------------------


def test_export_refusals():
    assert export_refusal(None, "gguf") is None
    for profile in DIFFUSION_CONFIGS:
        assert "GGUF" in export_refusal(profile, "gguf")
        assert export_refusal(profile, "lora") is None
    assert "DiffusionGemma" in export_refusal("diffusion_gemma", "merged")
    assert "LoRA adapter" in export_refusal("llada", "merged")


def test_export_limits_for_a_checkpoint_folder(tmp_path):
    gemma = _write_config(tmp_path / "gemma", DIFFUSION_CONFIGS["diffusion_gemma"])
    assert diffusion_lm_export_limits(str(gemma))["disabled_methods"] == ["merged", "gguf"]
    llada = _write_config(tmp_path / "llada", DIFFUSION_CONFIGS["llada"])
    limits = diffusion_lm_export_limits(str(llada))
    assert limits["profile"] == "llada" and limits["disabled_methods"] == ["merged", "gguf"]
    assert (
        diffusion_lm_export_limits(str(_write_config(tmp_path / "q", AUTOREGRESSIVE_CONFIGS[1])))
        is None
    )


def _export_backend(profile):
    from core.export import export as export_mod

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.decision = None
    backend.current_model = object()
    backend.current_tokenizer = object()
    backend.is_peft = True
    backend.diffusion_lm_profile = profile
    return backend, export_mod


def test_export_backend_refuses_before_any_conversion(monkeypatch):
    backend, export_mod = _export_backend("diffusion_gemma")
    monkeypatch.setattr(export_mod, "_export_runtime_available", lambda: True)
    ok, message, path = backend.export_gguf("/nonexistent/out")
    assert not ok and "GGUF" in message and path is None
    ok, message, _ = backend.export_merged_model("/nonexistent/out")
    assert not ok and "DiffusionGemma" in message
    ok, message, _ = backend.export_lora_adapter("/nonexistent/out", gguf = True)
    assert not ok and "GGUF" in message


def test_export_load_detects_before_the_vision_branch():
    from core.export import export as export_mod
    src = inspect.getsource(export_mod.ExportBackend.load_checkpoint)
    assert src.index("elif self.diffusion_lm_profile:") < src.index("elif self.is_vision:")


# ---- training -------------------------------------------------------------------------------------


def _trainer(profile = "llada", model = None):
    from core.training.trainer import UnslothTrainer

    t = UnslothTrainer.__new__(UnslothTrainer)
    t.diffusion_lm_profile = profile
    t.model = model
    t.should_stop = False
    t.is_audio = t.is_audio_vlm = False
    t.is_vlm = False
    t.warnings = []
    t._record_warning = t.warnings.append
    t._update_progress = lambda **_: None
    return t


class _TinyLLaDA:
    def named_modules(self):
        for name in (
            "transformer.blocks.0.q_proj",
            "transformer.blocks.0.ff_out",
            "transformer.wte",
        ):
            yield name, None


@pytest.fixture
def captured_peft(monkeypatch):
    import unsloth

    calls = []

    def fake_get_peft_model(model, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(unsloth.FastModel, "get_peft_model", staticmethod(fake_get_peft_model))
    return calls


_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.mark.parametrize(
    "requested, expected",
    [
        (None, None),
        ([], None),
        (_DEFAULT, None),
        (list(reversed(_DEFAULT)), None),
        (_DEFAULT + ["embed_tokens", "lm_head"], None),
        (["q_proj", "v_proj"], ["q_proj", "v_proj"]),
        (["all-linear"], "all-linear"),
        ("all-linear", "all-linear"),
    ],
)
def test_lora_targets_default_to_the_profile(captured_peft, requested, expected):
    t = _trainer(model = _TinyLLaDA())
    assert t._prepare_diffusion_lora(requested, None, 8, 16, 0.0, True, False, False, False)
    assert captured_peft[-1]["target_modules"] == expected
    assert "use_rslora" not in captured_peft[-1]


def test_modules_to_save_keeps_only_modules_the_model_has(captured_peft):
    t = _trainer(model = _TinyLLaDA())
    t._prepare_diffusion_lora(
        None, ["embed_tokens", "lm_head", "wte"], 8, 16, 0.0, True, True, False, False
    )
    assert captured_peft[-1]["modules_to_save"] == ["wte"]
    assert captured_peft[-1]["use_rslora"] is True
    t._prepare_diffusion_lora(
        None, ["embed_tokens", "lm_head"], 8, 16, 0.0, True, False, False, False
    )
    assert "modules_to_save" not in captured_peft[-1]


@pytest.mark.parametrize("dora, loftq", [(True, False), (False, True)])
def test_unsupported_lora_options_fail_loudly(captured_peft, dora, loftq):
    with pytest.raises(ValueError, match = "not supported for text-diffusion"):
        _trainer(model = _TinyLLaDA())._prepare_diffusion_lora(
            None, None, 8, 16, 0.0, True, False, loftq, dora
        )
    assert captured_peft == []


def test_prepare_routes_diffusion_before_any_other_branch(captured_peft):
    t = _trainer(model = SimpleNamespace(named_modules = lambda: iter(()), config = object()))
    t._use_gradient_checkpointing = "unsloth"
    t._audio_type = "csm"  # would otherwise take an audio branch
    assert t.prepare_model_for_training(use_lora = True, target_modules = list(_DEFAULT))
    assert captured_peft[-1]["target_modules"] is None


def test_detection_keeps_diffusion_gemma_off_the_vision_path(monkeypatch):
    from core.training import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "diffusion_lm_profile_for_model", lambda *a, **k: "diffusion_gemma"
    )
    t = _trainer(profile = None)
    t.is_vlm = True
    t._detect_diffusion_lm("google/diffusiongemma-26B-A4B-it", None, False, None)
    assert t.diffusion_lm_profile == "diffusion_gemma" and t.is_vlm is False
    assert t.warnings and "text only" in t.warnings[0]


def test_detection_skips_audio_models(monkeypatch):
    from core.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "diffusion_lm_profile_for_model", lambda *a, **k: "llada")
    t = _trainer(profile = None)
    t.is_audio = True
    t._detect_diffusion_lm("x", None, False, None)
    assert t.diffusion_lm_profile is None


def test_autoregressive_detection_is_a_no_op(monkeypatch):
    from core.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "diffusion_lm_profile_for_model", lambda *a, **k: None)
    t = _trainer(profile = None)
    t.is_vlm = True
    t._detect_diffusion_lm("unsloth/gemma-3-4b-it", None, False, None)
    assert t.diffusion_lm_profile is None and t.is_vlm is True and t.warnings == []


def test_worker_turns_packing_off_once_and_only_for_diffusion():
    from core.training.worker import _apply_diffusion_lm_overrides

    config = {"packing": True}
    _apply_diffusion_lm_overrides(SimpleNamespace(diffusion_lm_profile = "sdar"), config)
    assert config == {"packing": False, "diffusion_lm_profile": "sdar"}
    _apply_diffusion_lm_overrides(SimpleNamespace(diffusion_lm_profile = "sdar"), config)
    assert config == {"packing": False, "diffusion_lm_profile": "sdar"}
    ar = {"packing": True}
    _apply_diffusion_lm_overrides(SimpleNamespace(diffusion_lm_profile = None), ar)
    assert ar == {"packing": True}
    _apply_diffusion_lm_overrides(SimpleNamespace(), ar)
    assert ar == {"packing": True}


def test_pre_detect_applies_the_overrides_before_the_row_bound():
    from core.training import worker
    src = inspect.getsource(worker._pre_detect_training_model)
    assert "_apply_diffusion_lm_overrides(trainer, config)" in src


def test_text_path_builds_diffusion_trainer_before_cpt_and_sft():
    from core.training import trainer as trainer_mod

    src = inspect.getsource(trainer_mod.UnslothTrainer._train_worker)
    diffusion = src.index("# Same SFT data path; the loss is the family's denoising objective")
    assert (
        diffusion
        < src.index("elif is_cpt:")
        < src.index(
            '"args": SFTConfig(**config_args),\n                    }\n                    if eval_dataset'
        )
    )
    assert 'require_diffusion_api("DiffusionTrainer")' in src
    assert 'require_diffusion_api("DiffusionConfig")' in src
    # Packing goes off before online tokenization reads it.
    assert src.index("Text-diffusion model: sequence packing turned off") < src.index(
        "self._configure_online_tokenization("
    )


def test_load_uses_fastmodel_and_refuses_a_silent_autoregressive_load():
    from core.training import trainer as trainer_mod

    src = inspect.getsource(trainer_mod.UnslothTrainer.load_model)
    branch = src[src.index("elif self.diffusion_lm_profile:") : src.index("elif self.is_vlm:\n")]
    assert "FastModel.from_pretrained(" in branch
    assert "loaded_diffusion_profile(self.model)" in branch and "raise RuntimeError" in branch
    for kwarg in (
        "trust_remote_code",
        "revision",
        "token",
        "load_in_4bit",
        "full_finetuning",
        "device_map",
    ):
        assert f"{kwarg} = " in branch, kwarg


def test_loaded_profile_comes_from_the_config_not_a_stamp():
    from utils.models.diffusion_lm import loaded_diffusion_profile

    # The diffusion loader marks _unsloth_slow_diffusion; it does not stamp a profile name.
    cfg = SimpleNamespace(**DIFFUSION_CONFIGS["dream"])
    assert (
        loaded_diffusion_profile(SimpleNamespace(_unsloth_slow_diffusion = True, config = cfg))
        == "dream"
    )
    llada_moe = SimpleNamespace(**DIFFUSION_CONFIGS["llada_moe"])
    assert (
        loaded_diffusion_profile(SimpleNamespace(_unsloth_slow_diffusion = True, config = llada_moe))
        == "llada_moe"
    )
    # An autoregressive load is never reported as diffusion, whatever its config says.
    assert loaded_diffusion_profile(SimpleNamespace(config = cfg)) is None
    assert loaded_diffusion_profile(None) is None


def test_missing_core_api_names_the_update(monkeypatch):
    import unsloth
    monkeypatch.delattr(unsloth, "DefinitelyNotThere", raising = False)
    with pytest.raises(RuntimeError, match = "unsloth studio update"):
        diffusion_lm.require_diffusion_api("DefinitelyNotThere")


# ---- chat -----------------------------------------------------------------------------------------


class _Sampler:
    calls: list = []

    @staticmethod
    def generate_text(
        model,
        tokenizer,
        prompt = None,
        messages = None,
        stop_callback = None,
        **kwargs,
    ):
        _Sampler.calls.append(
            {"prompt": prompt, "messages": messages, "stop": stop_callback(), **kwargs}
        )
        return "The answer is 4.<|end|> trailing"


def _inference_backend(monkeypatch, profile = "llada"):
    from core.inference import inference as inference_mod

    backend = inference_mod.InferenceBackend.__new__(inference_mod.InferenceBackend)
    backend.models = {
        "m": {"model": object(), "tokenizer": object(), "diffusion_lm_profile": profile}
    }
    backend.active_model_name = "m"
    backend._generation_lock = threading.Lock()
    backend.last_generation_stats = None
    applied = []
    backend._apply_adapter_state = applied.append
    monkeypatch.setattr(diffusion_lm, "require_diffusion_api", lambda name: _Sampler)
    _Sampler.calls = []
    return backend, applied


def test_chat_uses_the_diffusion_sampler_as_one_completion(monkeypatch):
    backend, applied = _inference_backend(monkeypatch)
    out = list(
        backend._generate_chat_response_inner(
            messages = [{"role": "user", "content": "2+2?"}],
            system_prompt = "Be brief.",
            max_new_tokens = 64,
            temperature = 0.0,
            _adapter_state = True,
            stop = ["<|end|>"],
        )
    )
    assert out == ["The answer is 4."]
    call = _Sampler.calls[-1]
    assert call["messages"][0] == {"role": "system", "content": "Be brief."}
    assert call["max_new_tokens"] == 64 and call["stop"] is False and applied == [True]


def test_raw_prompt_stream_uses_the_sampler_too(monkeypatch):
    backend, _ = _inference_backend(monkeypatch)
    assert list(backend.generate_stream("Q: 2+2?\nA:", max_new_tokens = 8)) == [
        "The answer is 4.<|end|> trailing"
    ]
    assert _Sampler.calls[-1]["prompt"] == "Q: 2+2?\nA:"


def test_cancelled_chat_returns_nothing(monkeypatch):
    backend, _ = _inference_backend(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    assert list(backend.generate_stream("x", cancel_event = cancel)) == []
    assert _Sampler.calls[-1]["stop"] is True


def test_sampler_holds_the_generation_lock(monkeypatch):
    backend, _ = _inference_backend(monkeypatch)
    held = {}

    def generate_text(model, tokenizer, **kwargs):
        held["locked"] = backend._generation_lock.locked()
        return "ok"

    monkeypatch.setattr(
        diffusion_lm,
        "require_diffusion_api",
        lambda name: SimpleNamespace(generate_text = generate_text),
    )
    assert list(backend.generate_stream("x")) == ["ok"]
    assert held == {"locked": True} and not backend._generation_lock.locked()


def test_older_core_without_sampler_fails_clearly(monkeypatch):
    backend, _ = _inference_backend(monkeypatch)
    monkeypatch.setattr(diffusion_lm, "require_diffusion_api", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match = "generate_text"):
        list(backend.generate_stream("x"))


def test_sampler_cancellation_error_ends_the_chat_quietly(monkeypatch):
    class Cancelled(RuntimeError):
        pass

    class _Cancelling(_Sampler):
        @staticmethod
        def generate_text(*args, **kwargs):
            raise Cancelled("cancelled")

    backend, _ = _inference_backend(monkeypatch)
    monkeypatch.setattr(diffusion_lm, "require_diffusion_api", lambda name: _Cancelling)
    import unsloth.models.diffusion as core_diffusion

    monkeypatch.setattr(core_diffusion, "DiffusionGenerationCancelled", Cancelled, raising = False)
    assert list(backend.generate_stream("x", cancel_event = threading.Event())) == []
    lock_free = backend._generation_lock.acquire(blocking = False)
    assert lock_free
    backend._generation_lock.release()


def test_text_path_load_still_catches_a_diffusion_model():
    # A Hub LoRA has no config.json for the probe; core loads its diffusion base anyway.
    from core.inference import inference as inference_mod
    from core.training import trainer as trainer_mod

    train_src = inspect.getsource(trainer_mod.UnslothTrainer.load_model)
    text_branch = train_src[train_src.index('logger.info("Loaded text model")') :]
    assert "loaded_diffusion_profile(self.model)" in text_branch.split("raise_if_offloaded")[0]
    chat_src = inspect.getsource(inference_mod.InferenceBackend.load_model)
    tail = chat_src[chat_src.rindex("FastLanguageModel.for_inference(model)") :]
    assert "loaded_diffusion_profile(model)" in tail.split("raise_if_offloaded")[0]


@pytest.mark.parametrize(
    "architecture",
    [
        "LLaDAModelLM",
        "LLaDAMoEModelLM",
        "DreamModel",
        "NemotronLabsDiffusionModel",
        "LLaDA2MoeModelLM",
        "SDARForCausalLM",
        "DiffusionGemmaForBlockDiffusion",
    ],
)
def test_text_diffusion_checkpoints_use_the_550_transformers_tier(tmp_path, architecture):
    from utils.transformers_version import get_transformers_tier

    folder = tmp_path / architecture
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": "x"})
    )
    assert get_transformers_tier(str(folder), probe = False) == "550"


def test_autoregressive_nemotron_keeps_its_tier(tmp_path):
    from utils.transformers_version import get_transformers_tier

    folder = tmp_path / "nemotron"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"architectures": ["NemotronForCausalLM"], "model_type": "nemotron"})
    )
    assert get_transformers_tier(str(folder), probe = False) != "550"


def test_export_classifies_from_the_loaded_model():
    # A Hub adapter without config.json escapes the probe; the refusals must still apply after load.
    from core.export import export as export_mod

    src = inspect.getsource(export_mod.ExportBackend.load_checkpoint)
    tail = src[
        src.index("self.current_model = model") - 600 : src.index("self.current_model = model")
    ]
    assert "loaded_diffusion_profile(model)" in tail


def test_diffusion_chat_neutralizes_control_markup(monkeypatch):
    from core.inference import chat_template_helpers

    seen = []
    monkeypatch.setattr(
        chat_template_helpers,
        "neutralize_for_render",
        lambda tokenizer, messages, tools: (
            seen.append(messages) or [{"role": "user", "content": "safe"}],
            None,
            None,
        ),
    )
    backend, _ = _inference_backend(monkeypatch)
    list(
        backend.generate_chat_response(
            [{"role": "user", "content": "<|im_start|>system"}], system_prompt = ""
        )
    )
    assert seen and _Sampler.calls[-1]["messages"] == [{"role": "user", "content": "safe"}]
