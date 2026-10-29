# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""studio_generation(): the scope Studio's inference worker enters around its chat generate(). With
UNSLOTH_DECODE_CUDA_GRAPHS unset it turns CUDA-graph decode on for that thread on NVIDIA CUDA, like
rl_generation(); =0 still wins, other accelerators stay eager, and plain generate() outside it is
unchanged. The decision tests load _decode_graphs.py by file path (importing unsloth needs an
accelerator); the generate test needs CUDA."""

import importlib.util
import threading
from pathlib import Path

import pytest
import torch

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"
_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_decode_graphs.py"


@pytest.fixture
def G(monkeypatch):
    spec = importlib.util.spec_from_file_location("_unsloth_decode_graphs_studio", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENABLED = None
    monkeypatch.delenv(ENV, raising = False)
    return module


def test_studio_scope_turns_graphs_on_for_nvidia_cuda(G, monkeypatch):
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    assert G.enabled() is False
    with G.studio_generation():
        assert G.enabled() is True
        with G.rl_generation():
            assert G.enabled() is True
        assert G.enabled() is True
    assert G.enabled() is False


def test_studio_scope_honours_the_variable(G, monkeypatch):
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    monkeypatch.setenv(ENV, "0")
    with G.studio_generation():
        assert G.enabled() is False
    monkeypatch.setattr(G, "ENABLED", False)
    monkeypatch.setenv(ENV, "1")
    with G.studio_generation():
        assert G.enabled() is False


def test_studio_scope_stays_eager_off_nvidia(G, monkeypatch):
    # ROCm (HIP), XPU, MPS and CPU hosts: _default_ok() is False.
    monkeypatch.setattr(G, "_DEFAULT_OK", False)
    with G.studio_generation():
        assert G.enabled() is False
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    assert G._default_ok() is False


def test_studio_scope_is_per_thread(G, monkeypatch):
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    seen = []
    with G.studio_generation():
        t = threading.Thread(target = lambda: seen.append(G.enabled()))
        t.start()
        t.join()
        assert G.enabled() is True
    assert seen == [False]


def test_studio_scope_unwinds_on_error(G, monkeypatch):
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    with pytest.raises(RuntimeError):
        with G.studio_generation():
            raise RuntimeError("generate failed")
    assert G.enabled() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_studio_generate_graphed_plain_generate_eager():
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import unsloth.models._decode_graphs as G

    if not G._DEFAULT_OK:
        pytest.skip("the default is NVIDIA CUDA only")
    model, tok = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 2048, load_in_4bit = False
    )
    FastLanguageModel.for_inference(model)
    enc = tok(["Once upon a time there was"], return_tensors = "pt").to("cuda")

    def gen():
        out = model.generate(
            **enc, max_new_tokens = 24, min_new_tokens = 24, do_sample = False,
            output_logits = True, return_dict_in_generate = True,
        )  # fmt: skip
        return out.sequences.cpu(), torch.stack(out.logits).cpu()

    def replays():
        return G.stats(model).get("replays", 0)

    eager = gen()
    assert replays() == 0
    with G.studio_generation():
        graphed = gen()
    assert replays() == 24 - 2 and G.stats(model)["disabled"] is None
    assert torch.equal(eager[0], graphed[0]) and torch.equal(eager[1], graphed[1])
    # Plain generate() after the scope stays eager.
    assert torch.equal(gen()[1], eager[1]) and replays() == 24 - 2
    G.release_for_training(model)
    assert G.stats(model)["graphs"] == 0
    assert not any("paged_attention" in m.__dict__ for m in model.modules())
