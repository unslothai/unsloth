# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dead-tail probe verdict is reused across loads only while nothing it depends on changed."""

from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.inference import diffusion_te_truncate as tt  # noqa: E402
from core.inference import diffusion_te_truncate_cache as cache  # noqa: E402

LAYERS = 8


def _qwen3(seed: int = 0, layers: int = LAYERS):
    cfg = transformers.Qwen3Config(
        vocab_size = 97,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
    )
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _ids(prompt: str, device = None):
    torch.manual_seed(abs(hash(prompt)) % 1000)
    return torch.randint(1, 97, (1, 3 + len(prompt) % 7), device = device)


def _klein(o):
    return torch.stack([o.hidden_states[k] for k in (2, 4, 6)], 1)


def _penultimate(o):
    return o.hidden_states[-2]


class _Pipe:
    def __init__(
        self,
        encoder,
        read = _klein,
    ):
        self.text_encoder = encoder
        self._read = read

    def encode_prompt(
        self,
        prompt,
        device = None,
        num_images_per_prompt = 1,
    ):
        out = self.text_encoder(input_ids = _ids(prompt, device), output_hidden_states = True)
        return self._read(out), torch.ones(3)


PROMPTS = ("a photo of a cat", "two robots playing chess", "x")


def _encode_all(pipe):
    with torch.inference_mode():
        return [pipe.encode_prompt(p)[0].clone() for p in PROMPTS]


@pytest.fixture
def probes(monkeypatch):
    """Counts real probe runs; every test starts from its own empty Studio cache (conftest)."""
    monkeypatch.delenv(tt.TRUNCATE_ENV, raising = False)
    monkeypatch.delenv(cache.ENV, raising = False)
    calls = []
    real = tt.probe_consumption

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tt, "probe_consumption", counted)
    return calls


def test_second_load_reuses_the_verdict_and_stays_byte_identical(probes):
    reference = _Pipe(_qwen3())
    first, second = copy.deepcopy(reference), copy.deepcopy(reference)

    a = tt.truncate_text_encoders(first, family = "flux.2-klein")
    assert len(probes) == 1 and "cached" not in a
    b = tt.truncate_text_encoders(second, family = "flux.2-klein")
    assert len(probes) == 1, "the second identical load must not probe again"
    assert b.get("cached") is True
    assert a["plan"] == b["plan"] and a["truncated"] == b["truncated"]
    live = [i for i, m in enumerate(second.text_encoder.model.layers) if not tt.is_passthrough(m)]
    assert live == list(range(6))
    for x, y, z in zip(_encode_all(reference), _encode_all(first), _encode_all(second)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        assert torch.equal(x.view(torch.uint8), z.view(torch.uint8))


def test_the_verdict_survives_a_new_process(probes):
    """The key holds nothing process-local: a fresh process keyed the same pipeline the same way."""
    pipe = _Pipe(_qwen3())
    attrs = ("text_encoder",)
    fp1 = cache.fingerprint(pipe, family = "f", encoder_attrs = attrs)
    cache._SOURCE_DIGEST.clear()
    fp2 = cache.fingerprint(copy.deepcopy(pipe), family = "f", encoder_attrs = attrs)
    assert fp1 is not None and json.dumps(fp1, sort_keys = True) == json.dumps(fp2, sort_keys = True)


@pytest.mark.parametrize(
    "change",
    ["weights", "config", "reader", "family", "layers", "source", "env"],
)
def test_any_change_the_verdict_depends_on_is_a_miss(probes, monkeypatch, change):
    tt.truncate_text_encoders(_Pipe(_qwen3()), family = "flux.2-klein")
    assert len(probes) == 1
    family = "flux.2-klein"
    encoder = _qwen3()
    read = _klein
    if change == "weights":
        with torch.no_grad():
            next(encoder.parameters()).view(-1)[0] += 1.0
    elif change == "config":
        encoder.config.rms_norm_eps = 1e-5
    elif change == "reader":
        read = _penultimate
    elif change == "family":
        family = "z-image"
    elif change == "layers":
        encoder = _qwen3(layers = LAYERS + 2)
    elif change == "source":
        monkeypatch.setattr(cache, "_SOURCE_DIGEST", ["edited"])
    elif change == "env":
        monkeypatch.setenv("UNSLOTH_DIFFUSION_TE_SOMETHING", "1")
    tt.truncate_text_encoders(_Pipe(encoder, read), family = family)
    assert len(probes) == 2, f"a {change} change reused a stale verdict"


def test_a_changed_reader_gets_its_own_correct_verdict(probes):
    """The stale-verdict hazard itself: a pipeline reading a later layer must not inherit an earlier cut."""
    tt.truncate_text_encoders(_Pipe(_qwen3()), family = "f")
    pipe = _Pipe(_qwen3(), _penultimate)
    reference = copy.deepcopy(pipe)
    report = tt.truncate_text_encoders(pipe, family = "f")
    assert report["plan"]["text_encoder"]["last_consumed"] == LAYERS - 2
    for x, y in zip(_encode_all(reference), _encode_all(pipe)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def test_kill_switch_always_probes_and_writes_nothing(probes, monkeypatch):
    monkeypatch.setenv(cache.ENV, "0")
    tt.truncate_text_encoders(_Pipe(_qwen3()))
    tt.truncate_text_encoders(_Pipe(_qwen3()))
    assert len(probes) == 2
    path = cache._cache_file()
    assert path is None or not path.exists()


def test_a_corrupt_or_tampered_file_probes(probes):
    tt.truncate_text_encoders(_Pipe(_qwen3()))
    path = cache._cache_file()
    data = json.loads(path.read_text(encoding = "utf-8"))
    (entry,) = data.values()
    entry["plan"]["text_encoder"]["last_consumed"] = LAYERS + 5  # out of range
    path.write_text(json.dumps(data), encoding = "utf-8")
    tt.truncate_text_encoders(_Pipe(_qwen3()))
    assert len(probes) == 2
    path.write_text("{not json", encoding = "utf-8")
    tt.truncate_text_encoders(_Pipe(_qwen3()))
    assert len(probes) == 3


def test_explicit_encode_kwargs_and_local_shims_are_never_cached(probes):
    kwargs = {
        "prompt": "a photo of a cat",
        "device": torch.device("cpu"),
        "num_images_per_prompt": 1,
    }
    tt.truncate_text_encoders(_Pipe(_qwen3()), encode_kwargs = kwargs)
    tt.truncate_text_encoders(_Pipe(_qwen3()), encode_kwargs = kwargs)
    assert len(probes) == 2

    def shim(model):
        class _Probe:
            text_encoder = model

            @staticmethod
            def encode_prompt(prompt, device = None):
                return _klein(model(input_ids = _ids(prompt, device), output_hidden_states = True))

        return _Probe()

    tt.truncate_text_encoders(shim(_qwen3()))
    tt.truncate_text_encoders(shim(_qwen3()))
    assert len(probes) == 4


def test_a_declined_probe_is_not_cached(probes, monkeypatch):
    def boom(*_a, **_k):
        probes.append(1)
        raise RuntimeError("no")

    monkeypatch.setattr(tt, "probe_consumption", boom)
    assert tt.truncate_text_encoders(_Pipe(_qwen3()))["skipped"].startswith("probe:")
    path = cache._cache_file()
    assert path is None or not path.exists()


def test_meta_weights_are_unkeyable():
    with torch.device("meta"):
        encoder = transformers.Qwen3ForCausalLM(_qwen3().config)
    assert cache.fingerprint(_Pipe(encoder), encoder_attrs = ("text_encoder",)) is None
