# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import importlib.util
import os
import sys
import textwrap
import types


# trl 1.15.0 experimental/online_dpo/online_dpo_trainer.py OnlineDPOTrainer.__init__, vLLM part only.
_TRL_SOURCE = """
class OnlineDPOTrainer:
    def __init__(self, model, args):
        self.args = args
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode if args.use_vllm else None
        self.temperature = args.temperature
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError("vLLM is not available")

            if self.vllm_mode == "server":
                raise AssertionError("server mode")
            elif self.vllm_mode == "colocate":
                vllm_kwargs = {
                    "model": model.name_or_path,
                    "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
                }
                os.environ["RANK"] = "0"
                ensure_master_addr_port()

                self.llm = LLM(**vllm_kwargs)
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # Set up vLLM generation config
            generation_kwargs = {
                "n": 2,  # 2 generations per prompt for Online DPO
                "temperature": self.temperature,
                "max_tokens": args.max_new_tokens,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = SamplingParams(**generation_kwargs)
        else:
            generation_kwargs = {"max_new_tokens": args.max_new_tokens}
            self.generation_config = GenerationConfig(**generation_kwargs)
"""


class _LLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patched_class(tmp_path):
    from unsloth.models.rl import patch_functions

    path = tmp_path / "online_dpo_trainer_source.py"
    path.write_text(textwrap.dedent(_TRL_SOURCE))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)

    source = patch_functions(
        module.OnlineDPOTrainer, "online_dpo_trainer", "OnlineDPOTrainer", [], []
    )
    namespace = {
        "os": os,
        "LLM": _LLM,
        "SamplingParams": lambda **kwargs: kwargs,
        "vLLMSamplingParams": lambda **kwargs: types.SimpleNamespace(_set_kwargs = kwargs),
        "GenerationConfig": lambda **kwargs: kwargs,
        "is_vllm_available": lambda: True,
        "ensure_master_addr_port": lambda: None,
    }
    exec(source, namespace)
    return namespace["_UnslothOnlineDPOTrainer"]


def _args():
    return types.SimpleNamespace(
        use_vllm = True,
        vllm_mode = "colocate",
        temperature = 0.9,
        max_new_tokens = 16,
        generation_kwargs = None,
        vllm_gpu_memory_utilization = 0.3,
        vllm_enable_sleep_mode = False,
    )


def test_trl_owned_vllm_engine_is_built(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising = False)
    model = types.SimpleNamespace(name_or_path = "tiny-model")
    trainer = _patched_class(tmp_path)(model, _args())
    assert isinstance(trainer.llm, _LLM)
    assert trainer.llm.kwargs["model"] == "tiny-model"
    assert trainer.generation_config["n"] == 2
    assert trainer.generation_config["max_tokens"] == 16
    assert trainer._last_loaded_step == -1


def test_fast_inference_reuses_the_shared_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising = False)
    engine = types.SimpleNamespace(shared_weights = True)
    model = types.SimpleNamespace(name_or_path = "tiny-model", vllm_engine = engine)
    trainer = _patched_class(tmp_path)(model, _args())
    assert trainer.llm is engine
    assert trainer.generation_config["n"] == 2
    assert trainer._last_loaded_step == 0


def test_fast_inference_never_routes_to_a_vllm_server(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising = False)
    engine = types.SimpleNamespace(shared_weights = True)
    model = types.SimpleNamespace(name_or_path = "tiny-model", vllm_engine = engine)
    args = _args()
    args.vllm_mode = "server"
    trainer = _patched_class(tmp_path)(model, args)
    assert trainer.vllm_mode == "colocate"
    assert trainer.llm is engine


def test_vllm_sampling_params_override_trl_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising = False)
    model = types.SimpleNamespace(name_or_path = "tiny-model")
    args = _args()
    args.vllm_sampling_params = types.SimpleNamespace(_set_kwargs = {"temperature": 0.5})
    trainer = _patched_class(tmp_path)(model, args)
    assert trainer.generation_config["temperature"] == 0.5
    assert trainer.generation_config["n"] == 2
