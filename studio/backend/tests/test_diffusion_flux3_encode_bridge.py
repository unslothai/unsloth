# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3's modular text step routes its per-prompt encode through the facade's ``encode_prompt``, so the
cross-render prompt cache and the dead-tail probe apply. CPU only; a fake step stands in for diffusers'."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from core.inference import diffusion_flux3 as f3
from core.inference import diffusion_prompt_cache as prompt_cache
from core.inference.diffusion_te_truncate import _encode_kwargs

LAYERS = (4, 8)


class _TextStep:
    calls: list = []

    def __init__(self):
        self.pad_multiple = 80
        self.inputs = [
            SimpleNamespace(name = "prompt", default = None),
            SimpleNamespace(name = "text_encoder_out_layers", default = LAYERS),
        ]

    @staticmethod
    def _get_qwen3_vl_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt,
        device = None,
        max_sequence_length = 8192,
        pad_multiple = 80,
        hidden_states_layers = (4, 8, 12),
    ):
        _TextStep.calls.append(
            (text_encoder, prompt, max_sequence_length, pad_multiple, tuple(hidden_states_layers))
        )
        seed = sum(map(ord, prompt)) + len(hidden_states_layers)
        return torch.full((1, pad_multiple, 3), float(seed))


class _Modular:
    """Runs the text step as diffusers' block does: the pipeline's components, the step's own defaults."""

    def __init__(self):
        self._blocks = SimpleNamespace(sub_blocks = {"text_encoder": _TextStep()})
        self.text_encoder = self.tokenizer = self.transformer = self.vae = None
        self.scheduler = SimpleNamespace(step = lambda *args, **kwargs: (None,))

    def update_components(self, **updates):
        for name, value in updates.items():
            setattr(self, name, value)

    def __call__(
        self,
        output = None,
        **kwargs,
    ):
        step = self._blocks.sub_blocks["text_encoder"]
        embeds = step._get_qwen3_vl_prompt_embeds(
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            prompt = kwargs["prompt"],
            device = torch.device("cpu"),
            max_sequence_length = kwargs["max_sequence_length"],
            pad_multiple = step.pad_multiple,
            hidden_states_layers = LAYERS,
        )
        return [embeds]


class _Face(f3._Flux3KleinPipelineMixin):
    def __init__(self):
        self.config = SimpleNamespace(blocks_class_name = f3.FLUX3_DISTILLED_BLOCKS)
        self._interrupt = False
        self._modular = None
        self.text_encoder = torch.nn.Linear(2, 2)
        self.tokenizer = object()
        self.transformer = self.vae = self.scheduler = None

    def _set_guidance(self, mp, guidance_scale):
        pass

    def progress_bar(self, total = None):
        return SimpleNamespace(close = lambda: None)


@pytest.fixture
def face(monkeypatch):
    _TextStep.calls = []
    try:
        import diffusers  # noqa: F401 -- modular_pipeline imports it
    except ImportError:
        monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setattr(f3, "flux3_blocks", lambda name: SimpleNamespace(init_pipeline = _Modular))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROMPT_CACHE", raising = False)
    return _Face()


def test_the_text_step_encodes_through_encode_prompt(face, monkeypatch):
    seen = []
    original = face.encode_prompt

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(face, "encode_prompt", spy)
    out = face(prompt = "a fox", num_inference_steps = 1, max_sequence_length = 512)
    assert seen == [
        {
            "prompt": "a fox",
            "max_sequence_length": 512,
            "device": torch.device("cpu"),
            "pad_multiple": 80,
            "hidden_states_layers": LAYERS,
        }
    ]
    assert _TextStep.calls == [(face.text_encoder, "a fox", 512, 80, LAYERS)]
    assert torch.equal(
        out.images[0],
        _TextStep._get_qwen3_vl_prompt_embeds(
            None, None, "a fox", hidden_states_layers = LAYERS, pad_multiple = 80
        ),
    )


def test_a_returning_prompt_skips_the_encoder_once_the_prompt_cache_is_on(face):
    assert prompt_cache.install(face, identity = {"family": "flux.3-klein"})
    first = face(prompt = "a fox", num_inference_steps = 1).images[0]
    again = face(prompt = "a fox", num_inference_steps = 1).images[0]
    other = face(prompt = "a whale", num_inference_steps = 1).images[0]
    assert [c[1] for c in _TextStep.calls] == ["a fox", "a whale"]
    assert torch.equal(first, again) and first.data_ptr() != again.data_ptr()
    assert not torch.equal(first, other)
    assert prompt_cache.cache_for(face).describe()["hits"] == 1


def test_another_encoder_runs_the_steps_own_helper(face):
    mp = face.modular_pipeline()
    face.encode_prompt = lambda **kwargs: pytest.fail("routed a foreign encoder")
    step = mp._blocks.sub_blocks["text_encoder"]
    other = torch.nn.Linear(2, 2)
    step._get_qwen3_vl_prompt_embeds(text_encoder = other, tokenizer = face.tokenizer, prompt = "x")
    assert _TextStep.calls[-1][0] is other


def test_encode_prompt_defaults_come_from_the_step_and_the_probe_can_call_it(face):
    # The dead-tail probe calls encode_prompt with only the prompt and a device: it must read what the step reads.
    kwargs = _encode_kwargs(face, torch.device("cpu"))
    assert kwargs == {"prompt": "a photo of a cat", "device": torch.device("cpu")}
    face.encode_prompt(**kwargs)
    assert _TextStep.calls[-1][2:] == (8192, 80, LAYERS)


def test_an_unknown_helper_keeps_the_step_unbridged(face, monkeypatch):
    def other_helper(
        text_encoder,
        tokenizer,
        prompt,
        device = None,
    ):
        return torch.zeros(1)

    monkeypatch.setattr(_TextStep, "_get_qwen3_vl_prompt_embeds", staticmethod(other_helper))
    mp = face.modular_pipeline()
    assert "_get_qwen3_vl_prompt_embeds" not in vars(mp._blocks.sub_blocks["text_encoder"])
    with pytest.raises(RuntimeError, match = "no known prompt helper"):
        face.encode_prompt("a fox")


def test_the_installed_diffusers_text_step_matches_the_bridge():
    diffusers = pytest.importorskip("diffusers")
    try:
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
    except Exception:  # noqa: BLE001 -- no FLUX.3 in this diffusers
        pytest.skip("diffusers without FLUX.3")
    import inspect

    params = set(inspect.signature(Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds).parameters)
    assert f3._TEXT_HELPER_ARGS <= params
    step = Flux3TextEncoderStep()
    assert isinstance(step.pad_multiple, int)
    assert any(p.name == f3._TEXT_LAYERS_INPUT and p.default for p in step.inputs)
    assert diffusers is not None
