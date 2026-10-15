# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CPU checks for patch_trl_async_grpo (loaded from rl.py source, TRL's async module replaced by a fake)."""

from __future__ import annotations

import ast
import functools
import importlib
import os
import sys
import types
import warnings

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
NAMES = (
    "_without_instance_bound_methods",
    "_packed_seq_lengths_from_position_ids",
    "_is_unsloth_fast_backbone",
    "_unsloth_async_grpo_lm_head",
    "_fast_inference_engine",
    "_async_grpo_uses_fast_inference",
    "_dataset_has_images",
    "_start_async_grpo_vllm_server",
    "_check_async_grpo_fast_inference",
    "_text_prepended_position_ids",
    "_packing_honored",
    "_align_vlm_text_config",
    "_hold_peft_requires_grad",
    "_allow_moe_parameter_lora",
    "_check_async_grpo_adapter_servable",
    "_async_grpo_masked_compute_loss",
    "patch_trl_async_grpo",
)


class _Logger:
    def info(self, *a, **k):
        pass


def _load(fast_backbone_types = ()):
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {
        "copy": __import__("copy"),
        "inspect": __import__("inspect"),
        "textwrap": __import__("textwrap"),
        "types": types,
        "torch": torch,
        "functools": functools,
        "importlib": importlib,
        "warnings": warnings,
        "weakref": __import__("weakref"),
        "logger": _Logger(),
        "Version": __import__("packaging.version", fromlist = ["Version"]).Version,
        "importlib_version": __import__("importlib.metadata", fromlist = ["version"]).version,
    }
    for node in ast.parse(src).body:
        is_constant = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("_ASYNC_GRPO_") for t in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in NAMES):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    assert all(n in ns for n in NAMES)
    ns["_is_unsloth_fast_backbone"] = lambda b: isinstance(b, fast_backbone_types)
    return ns


class FastBackbone(torch.nn.Module):
    pass


class CausalLM(torch.nn.Module):
    def __init__(
        self,
        backbone,
        attn = "sdpa",
    ):
        super().__init__()
        self.model = backbone
        self.config = types.SimpleNamespace(
            _attn_implementation = attn, _name_or_path = "org/Tiny-Model"
        )

    @property
    def base_model(self):
        return self.model

    def forward(self, *a, **k):
        return k


class Peft(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def config(self):  # PeftModel forwards attribute lookups to the wrapped model
        return self.inner.config

    def get_base_model(self):
        return self.inner


@pytest.fixture
def fake_trl(monkeypatch):
    calls = {"fused": [], "create": [], "init": []}
    mod = types.ModuleType("trl.experimental.async_grpo.async_grpo_trainer")

    def create_model_from_path(model_id, **kw):
        calls["create"].append(model_id)
        return CausalLM(torch.nn.Module(), attn = "kernels-community/flash-attn3")

    def add_fused_lm_head(model, **kw):
        calls["fused"].append(model)
        orig = model.forward

        def fused(
            *a,
            fused_lm_head = False,
            **k,
        ):
            return {"fused": fused_lm_head, **orig(*a, **k)}

        model.forward = fused

    class AsyncGRPOConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    class AsyncGRPOTrainer:
        def __init__(
            self,
            model,
            reward_funcs = None,
            args = None,
            **kw,
        ):
            calls["init"].append((model, args))
            mod.add_fused_lm_head(mod.create_model_from_path(model))

    mod.create_model_from_path = create_model_from_path
    mod.add_fused_lm_head = add_fused_lm_head
    mod.AsyncGRPOConfig = AsyncGRPOConfig
    mod.AsyncGRPOTrainer = AsyncGRPOTrainer
    for name in ("trl", "trl.experimental", "trl.experimental.async_grpo"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        pkg.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package = True)
        monkeypatch.setitem(sys.modules, name, pkg)
    mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    sys.modules["trl.experimental.async_grpo"].async_grpo_trainer = mod
    return mod, calls


def test_packed_seq_lengths_from_position_ids():
    f = _load()["_packed_seq_lengths_from_position_ids"]
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    assert f(pos).tolist() == [3, 2, 4]
    assert f(pos).dtype == torch.int32
    # A lone sequence still gets its length: it also lifts Unsloth's max_seq_length truncation for the row.
    assert f(torch.tensor([[0, 1, 2, 3]])).tolist() == [4]
    assert f(torch.tensor([[1, 2, 0, 1]])) is None  # does not start at a sequence start
    assert f(torch.tensor([[0, 1], [0, 1]])) is None  # padded batch, not one packed row
    assert f(None) is None


def test_unsloth_model_passes_through_and_gets_boundaries(fake_trl):
    mod, calls = fake_trl
    ns = _load(fast_backbone_types = (FastBackbone,))
    ns["patch_trl_async_grpo"]()
    lm = CausalLM(FastBackbone())
    peft = Peft(lm)
    trainer = mod.AsyncGRPOTrainer(peft, None)
    # Fused head goes on the CausalLM under the PEFT wrapper.
    assert calls["create"] == []
    assert calls["fused"] == [lm]
    # Default args derived from the model name instead of crashing on model.split.
    assert calls["init"][0][1].output_dir == "Tiny-Model-AsyncGRPO"
    pos = torch.tensor([[0, 1, 0, 1, 2]])
    out = lm.forward(position_ids = pos, fused_lm_head = True)
    assert out["packed_seq_lengths"].tolist() == [2, 3]
    assert "packed_seq_lengths" not in lm.forward(position_ids = pos)
    mod.AsyncGRPOTrainer(Peft(CausalLM(FastBackbone())), None, None)
    assert calls["init"][1][1].output_dir == "Tiny-Model-AsyncGRPO"
    # TRL-built models (from a string id) have real backbones that pass the packing self-test.
    ns["_packing_honored"] = lambda backbone, transform = None: True
    mod.AsyncGRPOTrainer("org/x", None, args = mod.AsyncGRPOConfig("o"))
    assert calls["create"] == ["org/x"]


def test_non_unsloth_sdpa_model_is_refused(fake_trl):
    mod, _ = fake_trl
    ns = _load()
    ns["patch_trl_async_grpo"]()
    with pytest.raises(ValueError, match = "attend to each other"):
        mod.AsyncGRPOTrainer(
            CausalLM(torch.nn.Module(), attn = "sdpa"), None, args = mod.AsyncGRPOConfig("o")
        )

    # Flash attention honors position_ids resets; TRL builds these models on CPU, where the self-test cannot run.
    def unrunnable(backbone, transform = None):
        raise NotImplementedError("flash varlen kernel on the CPU backend")

    ns["_packing_honored"] = unrunnable
    for attn in ("flash_attention_2", "kernels-community/flash-attn3"):
        lm = CausalLM(torch.nn.Module(), attn = attn)
        mod.AsyncGRPOTrainer(lm, None, args = mod.AsyncGRPOConfig("o"))
    # A self-test that cannot run refuses, like one that fails.
    with pytest.raises(ValueError, match = "keeps them apart failed"):
        mod.AsyncGRPOTrainer(
            CausalLM(torch.nn.Module(), attn = "sdpa"), None, args = mod.AsyncGRPOConfig("o")
        )
    ns["_packing_honored"] = lambda backbone, transform = None: True
    lm = CausalLM(torch.nn.Module(), attn = "flash_attention_2")
    mod.AsyncGRPOTrainer(lm, None, args = mod.AsyncGRPOConfig("o"))
    assert "packed_seq_lengths" not in lm.forward(
        position_ids = torch.tensor([[0, 1, 0, 1]]), fused_lm_head = True
    )


def test_patch_is_idempotent(fake_trl):
    mod, _ = fake_trl
    ns = _load()
    ns["patch_trl_async_grpo"]()
    first = (mod.create_model_from_path, mod.add_fused_lm_head, mod.AsyncGRPOTrainer.__init__)
    ns["patch_trl_async_grpo"]()
    assert (
        mod.create_model_from_path,
        mod.add_fused_lm_head,
        mod.AsyncGRPOTrainer.__init__,
    ) == first


class Tok:
    def __init__(self):
        self.vocab = {"a": 1}

    def save_pretrained(self, d):
        return "plain"

    def encode(self, s):
        return [self.vocab[c] for c in s]


def _unsloth_tokenizer_save_pretrained(self, d):
    return "patched"


def _patch_like_unsloth(tok):
    tok.original_save_pretrained = tok.save_pretrained
    tok.save_pretrained = types.MethodType(_unsloth_tokenizer_save_pretrained, tok)
    return tok


def test_rollout_worker_tokenizer_is_picklable():
    import pickle

    f = _load()["_without_instance_bound_methods"]
    plain = Tok()
    assert f(plain) is plain
    tok = _patch_like_unsloth(Tok())
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.loads(pickle.dumps(tok))
    clean = f(tok)
    assert (
        clean is not tok and tok.save_pretrained(".") == "patched"
    )  # the trainer's tokenizer keeps Unsloth's saving
    back = pickle.loads(pickle.dumps(clean))
    assert back.encode("a") == [1] and back.save_pretrained(".") == "plain"
    proc = types.SimpleNamespace(tokenizer = _patch_like_unsloth(Tok()))
    clean_proc = f(proc)
    assert clean_proc is not proc and "save_pretrained" not in vars(clean_proc.tokenizer)
    assert "save_pretrained" in vars(proc.tokenizer)


def test_rollout_worker_subclass_cleans_processing_class(fake_trl):
    mod, _ = fake_trl
    seen = {}

    class AsyncRolloutWorker:
        def __init__(
            self,
            processing_class = None,
            **kw,
        ):
            seen["pc"] = processing_class

    mod.AsyncRolloutWorker = AsyncRolloutWorker
    ns = _load()
    ns["patch_trl_async_grpo"]()
    tok = _patch_like_unsloth(Tok())
    w = mod.AsyncRolloutWorker(processing_class = tok)
    assert (
        isinstance(w, AsyncRolloutWorker)
        and mod.AsyncRolloutWorker.__name__ == "AsyncRolloutWorker"
    )
    assert seen["pc"] is not tok and "save_pretrained" not in vars(seen["pc"])


def test_trl_113_chunked_head_gets_boundaries(fake_trl):
    # TRL 1.13 / 1.14 name the head patch_chunked_lm_head and always score through it.
    mod, calls = fake_trl
    chunked = mod.add_fused_lm_head
    del mod.add_fused_lm_head

    def patch_chunked_lm_head(model, **kw):
        orig = model.forward
        model.forward = lambda *a, **k: {"chunked": True, **orig(*a, **k)}

    mod.patch_chunked_lm_head = patch_chunked_lm_head

    class Trainer:
        def __init__(
            self,
            model,
            reward_funcs = None,
            args = None,
            **kw,
        ):
            mod.patch_chunked_lm_head(mod.create_model_from_path(model))

    mod.AsyncGRPOTrainer = Trainer
    ns = _load(fast_backbone_types = (FastBackbone,))
    ns["patch_trl_async_grpo"]()
    lm = CausalLM(FastBackbone())
    mod.AsyncGRPOTrainer(Peft(lm), None)
    out = lm.forward(position_ids = torch.tensor([[0, 1, 2, 0, 1]]))
    assert out["chunked"] and out["packed_seq_lengths"].tolist() == [3, 2]
    assert chunked is not mod.patch_chunked_lm_head


class _FakeServer:
    started = []

    def __init__(self, model):
        self.url = "http://127.0.0.1:1"
        self.max_num_seqs = 96
        self.stopped = 0
        self.aborted = 0
        _FakeServer.started.append(self)

    def stop(self):
        self.stopped += 1

    def abort_all(self):
        self.aborted += 1


def _fast_inference_peft(lm):
    peft = Peft(lm)
    peft.peft_config = {"default": object()}
    peft.vllm_engine = object()
    peft.load_lora = lambda *a, **k: None
    peft.lora = torch.nn.Parameter(torch.zeros(2))
    peft.frozen = torch.nn.Parameter(torch.zeros(2), requires_grad = False)
    return peft


@pytest.fixture
def fast_inference_trl(fake_trl):
    mod, calls = fake_trl
    mod.select_adapter_sync = lambda *a, **k: True

    class AsyncGRPOConfig:
        def __init__(
            self,
            output_dir,
            vllm_server_base_url = "http://localhost:8000",
            max_inflight_tasks = -1,
        ):
            self.output_dir = output_dir
            self.vllm_server_base_url = vllm_server_base_url
            self.max_inflight_tasks = max_inflight_tasks

    class AsyncGRPOTrainer:
        lora_sync = True

        def __init__(
            self,
            model,
            reward_funcs = None,
            args = None,
            train_dataset = None,
            **kw,
        ):
            calls["init"].append((model, args))
            self.url_seen = args.vllm_server_base_url
            # What TRL does for a VLM: unfreeze the text model.
            model.frozen.requires_grad_(True)
            self._lora_sync = AsyncGRPOTrainer.lora_sync

        def train(self):
            return "trained"

    mod.AsyncGRPOConfig = AsyncGRPOConfig
    mod.AsyncGRPOTrainer = AsyncGRPOTrainer
    mod._calls = calls
    _FakeServer.started = []
    ns = _load(fast_backbone_types = (FastBackbone,))
    ns["_start_async_grpo_vllm_server"] = _FakeServer
    ns["patch_trl_async_grpo"]()
    return mod, ns


def calls_args(mod):
    return mod._calls["init"][-1][1]


def test_fast_inference_model_gets_the_in_process_server(fast_inference_trl):
    mod, _ = fast_inference_trl
    peft = _fast_inference_peft(CausalLM(FastBackbone()))
    trainer = mod.AsyncGRPOTrainer(
        peft, None, args = mod.AsyncGRPOConfig("o"), train_dataset = [{"prompt": "hi"}]
    )
    (server,) = _FakeServer.started
    assert trainer.url_seen == server.url
    assert calls_args(mod).max_inflight_tasks == 96
    mod.AsyncGRPOTrainer(
        _fast_inference_peft(CausalLM(FastBackbone())),
        None,
        args = mod.AsyncGRPOConfig("o", max_inflight_tasks = 8),
    )
    assert calls_args(mod).max_inflight_tasks == 8
    assert not peft.frozen.requires_grad and peft.lora.requires_grad
    assert trainer.train() == "trained" and server.aborted == 1


def test_explicit_server_url_keeps_the_external_server(fast_inference_trl):
    mod, _ = fast_inference_trl
    peft = _fast_inference_peft(CausalLM(FastBackbone()))
    cfg = mod.AsyncGRPOConfig("o", vllm_server_base_url = "http://gpu-box:8000")
    trainer = mod.AsyncGRPOTrainer(peft, None, args = cfg)
    assert _FakeServer.started == [] and trainer.url_seen == "http://gpu-box:8000"


def test_fast_inference_refusals(fast_inference_trl):
    mod, _ = fast_inference_trl
    lm = CausalLM(FastBackbone())
    lm.vllm_engine = object()
    with pytest.raises(ValueError, match = "get_peft_model"):
        mod.AsyncGRPOTrainer(lm, None, args = mod.AsyncGRPOConfig("o"))
    images = [
        {
            "prompt": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}
            ]
        }
    ]
    with pytest.raises(ValueError, match = "image prompts"):
        mod.AsyncGRPOTrainer(
            _fast_inference_peft(CausalLM(FastBackbone())),
            None,
            args = mod.AsyncGRPOConfig("o"),
            train_dataset = images,
        )
    mod.AsyncGRPOTrainer.lora_sync = False
    with pytest.raises(ValueError, match = "plain LoRA adapter"):
        mod.AsyncGRPOTrainer(
            _fast_inference_peft(CausalLM(FastBackbone())), None, args = mod.AsyncGRPOConfig("o")
        )
    assert _FakeServer.started[-1].stopped == 1
    assert _FakeServer.started == [_FakeServer.started[0]] or len(_FakeServer.started) == 1
    del mod.select_adapter_sync
    with pytest.raises(ValueError, match = "TRL >= 1.14"):
        mod.AsyncGRPOTrainer(
            _fast_inference_peft(CausalLM(FastBackbone())), None, args = mod.AsyncGRPOConfig("o")
        )


def _vlm(copy_text_config):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        text_config = dict(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            vocab_size = 128,
            rope_scaling = {"type": "mrope", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 1, hidden_size = 16, intermediate_size = 32, num_heads = 2, out_hidden_size = 16
        ),
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    language_model = model.model.language_model
    if copy_text_config:
        # Unsloth's loaders hand each level its own config copy.
        language_model.config = __import__("copy").deepcopy(language_model.config)
    return model, model.config, language_model


def test_vlm_text_config_answers_with_the_language_model_config_during_init():
    ns = _load()
    model, top, language_model = _vlm(copy_text_config = True)
    original = top.get_text_config()
    assert language_model.config is not original
    restore = ns["_align_vlm_text_config"](model)
    assert model.config.get_text_config() is language_model.config
    restore()
    assert model.config.get_text_config() is original
    assert "get_text_config" not in vars(top)


def test_vlm_text_config_left_alone_when_identity_already_holds():
    ns = _load()
    model, top, language_model = _vlm(copy_text_config = False)
    restore = ns["_align_vlm_text_config"](model)
    assert "get_text_config" not in vars(top)
    assert model.config.get_text_config() is language_model.config
    restore()


def test_peft_requires_grad_calls_are_held_during_init():
    ns = _load()
    model, _, language_model = _vlm(copy_text_config = True)
    model.peft_config = {"default": object()}
    model.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(False)
    restore = ns["_hold_peft_requires_grad"](model)
    # What TRL's VLM branch does: unfreeze the text model and LM head.
    model.requires_grad_(False)
    language_model.requires_grad_(True)
    model.get_output_embeddings().requires_grad_(True)
    assert not any(p.requires_grad for p in model.parameters())
    restore()
    assert all("requires_grad_" not in vars(m) for m in model.modules())
    language_model.requires_grad_(True)
    assert all(p.requires_grad for p in language_model.parameters())


def test_non_peft_requires_grad_is_untouched():
    ns = _load()
    model, _, language_model = _vlm(copy_text_config = True)
    restore = ns["_hold_peft_requires_grad"](model)
    model.requires_grad_(False)
    language_model.requires_grad_(True)
    assert all(p.requires_grad for p in language_model.parameters())
    restore()


def test_moe_parameter_lora_passes_validation_only_while_held():
    ns = _load()
    peft_config = types.SimpleNamespace(target_parameters = ["mlp.experts.gate_up_proj"])
    model = types.SimpleNamespace(peft_config = {"default": peft_config})
    seen = []

    def validate(peft_model):
        seen.append(peft_config.target_parameters)
        if peft_config.target_parameters:
            raise ValueError("target_parameters")
        return peft_config

    module = types.SimpleNamespace(validate_lora_for_vllm_sync = validate)
    restore = ns["_allow_moe_parameter_lora"](module, model)
    assert module.validate_lora_for_vllm_sync(model) is peft_config
    assert seen == [None]
    assert peft_config.target_parameters == ["mlp.experts.gate_up_proj"]
    restore()
    assert module.validate_lora_for_vllm_sync is validate


def test_unservable_adapter_fails_at_construction(fast_inference_trl):
    mod, _ = fast_inference_trl
    peft = _fast_inference_peft(CausalLM(FastBackbone()))
    seen = []

    def load_lora(directory, load_tensors = False):
        seen.append(
            (os.path.exists(directory), os.path.isdir(os.path.dirname(directory)), load_tensors)
        )
        raise NotImplementedError("cannot serve this LoRA on MoE expert weights")

    peft.load_lora = load_lora
    with pytest.raises(NotImplementedError, match = "MoE expert"):
        mod.AsyncGRPOTrainer(
            peft, None, args = mod.AsyncGRPOConfig("o"), train_dataset = [{"prompt": "hi"}]
        )
    assert seen == [(False, True, True)]
    assert _FakeServer.started == []
    assert mod._calls["init"] == []


def test_a_reused_config_keeps_its_own_server_url(fast_inference_trl):
    mod, _ = fast_inference_trl
    config = mod.AsyncGRPOConfig("o")
    trainer = mod.AsyncGRPOTrainer(
        _fast_inference_peft(CausalLM(FastBackbone())),
        None,
        args = config,
        train_dataset = [{"prompt": "hi"}],
    )
    assert trainer.url_seen == _FakeServer.started[0].url
    assert config.vllm_server_base_url == "http://localhost:8000"
    mod.AsyncGRPOTrainer.lora_sync = False
    try:
        config = mod.AsyncGRPOConfig("o")
        with pytest.raises(ValueError, match = "plain LoRA adapter"):
            mod.AsyncGRPOTrainer(
                _fast_inference_peft(CausalLM(FastBackbone())),
                None,
                args = config,
                train_dataset = [{"prompt": "hi"}],
            )
    finally:
        mod.AsyncGRPOTrainer.lora_sync = True
    assert (config.vllm_server_base_url, config.max_inflight_tasks) == ("http://localhost:8000", -1)
    assert _FakeServer.started[1].stopped == 1


def test_servable_check_reads_only_a_lone_default_adapter():
    ns = _load()
    calls = []
    model = types.SimpleNamespace(
        peft_config = {"policy": object()}, load_lora = lambda *a, **k: calls.append(a)
    )
    ns["_check_async_grpo_adapter_servable"](model)
    model.peft_config = {"default": object(), "other": object()}
    ns["_check_async_grpo_adapter_servable"](model)
    assert calls == []
    model.peft_config = {"default": object()}
    ns["_check_async_grpo_adapter_servable"](model)
    assert len(calls) == 1


def test_image_check_does_not_consume_an_iterable_dataset():
    ns = _load()

    class OneShot:
        def __init__(self):
            self.rows = iter([{"prompt": "a"}, {"prompt": "b"}])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.rows)

    data = OneShot()
    assert ns["_dataset_has_images"](data) is False
    assert next(data) == {"prompt": "a"}
    assert ns["_dataset_has_images"]([{"image": object()}]) is True
    assert ns["_dataset_has_images"]([{"prompt": "hi"}]) is False


def _ppo_compute_loss(
    self,
    model,
    inputs,
    return_outputs = False,
    num_items_in_batch = None,
):
    # The policy-loss lines of TRL 1.13 to 1.15 AsyncGRPOTrainer.compute_loss, verbatim.
    log_probs = model(inputs)
    advantages = inputs["advantages"]
    log_ratio = log_probs - inputs["old_log_probs"]
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    with torch.no_grad():
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        self.clipped = (is_low_clipped | is_high_clipped).tolist()
    return per_token_loss.sum()


def _loss_grad(trainer, ratio, advantages):
    log_probs = torch.zeros(len(ratio), dtype = torch.float64, requires_grad = True)
    inputs = {
        "old_log_probs": -torch.tensor(ratio, dtype = torch.float64).log(),
        "advantages": torch.tensor(advantages, dtype = torch.float64),
    }
    trainer.compute_loss(lambda _: log_probs, inputs).backward()
    return log_probs.grad.tolist()


def test_importance_sampling_mask_gives_the_sao_gradient(fake_trl):
    mod, _ = fake_trl
    mod.AsyncGRPOTrainer.compute_loss = _ppo_compute_loss
    _load(fast_backbone_types = (FastBackbone,))["patch_trl_async_grpo"]()
    ratio = [0.5, 0.7, 0.75, 1.0, 1.3, 5.9, 6.0, 9.0, 9.0, 0.5]
    advantages = [1.0, 1.0, -2.0, 0.5, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    masked = mod.AsyncGRPOTrainer(
        Peft(CausalLM(FastBackbone())), None, importance_sampling_mask = True
    )
    masked.epsilon_low, masked.epsilon_high = 0.3, 5.0
    # SAO Eq. 1-3: d/dlog pi of -f(r) A log pi, f(r) = r inside the open window (0.7, 6.0), else 0.
    expected = [-(r * a if 0.7 < r < 6.0 else 0.0) for r, a in zip(ratio, advantages)]
    assert _loss_grad(masked, ratio, advantages) == pytest.approx(expected)
    assert masked.clipped == [True, True, False, False, False, False, True, True, True, True]
    # Default stays TRL's PPO loss, on TRL's own method: a negative advantage at r = 9 still gets a gradient.
    plain = mod.AsyncGRPOTrainer(Peft(CausalLM(FastBackbone())), None)
    plain.epsilon_low, plain.epsilon_high = 0.3, 5.0
    assert "compute_loss" not in vars(plain)
    assert _loss_grad(plain, ratio, advantages)[8] == pytest.approx(9.0)
    assert plain.clipped[8] is False


def test_importance_sampling_mask_refuses_an_unknown_loss(fast_inference_trl):
    mod, _ = fast_inference_trl

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        return model(inputs).sum()

    mod.AsyncGRPOTrainer.compute_loss = compute_loss
    peft = _fast_inference_peft(CausalLM(FastBackbone()))
    with pytest.raises(ValueError, match = "cannot be masked"):
        mod.AsyncGRPOTrainer(
            peft, None, args = mod.AsyncGRPOConfig("o"), importance_sampling_mask = True
        )
    # Refused before the in-process server starts or TRL's init runs.
    assert _FakeServer.started == [] and mod._calls["init"] == []


def test_importance_sampling_mask_rewrites_the_installed_trl_loss():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer
    except Exception as e:
        pytest.skip(f"TRL without AsyncGRPOTrainer: {e}")
    ns = _load()
    masked = ns["_async_grpo_masked_compute_loss"](AsyncGRPOTrainer.compute_loss)
    code = masked.__code__
    assert masked is ns["_async_grpo_masked_compute_loss"](AsyncGRPOTrainer.compute_loss)
    assert code is not AsyncGRPOTrainer.compute_loss.__code__
    assert "where" in code.co_names and "min" not in code.co_names
    assert masked.__globals__ is AsyncGRPOTrainer.compute_loss.__globals__
