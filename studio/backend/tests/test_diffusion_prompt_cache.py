# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for ``diffusion_prompt_cache.py`` with stub pipes."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_cond_cache as cond_cache
from core.inference import diffusion_prompt_cache as prompt_cache

from .test_diffusion_backend import fake_runtime  # noqa: E402,F401


class _EncodePipe:
    def __init__(self, width = 4):
        self.calls = 0
        self.width = width
        self._execution_device = torch.device("cpu")
        self.text_encoder = torch.nn.Linear(2, 2)

    def encode_prompt(
        self,
        prompt,
        negative_prompt = None,
        device = None,
        num_images_per_prompt = 1,
        max_sequence_length = 256,
        prompt_embeds = None,
        image = None,
        dtype = None,
    ):
        if prompt_embeds is not None:
            return (prompt_embeds, None)
        self.calls += 1
        value = float(sum(map(ord, str(prompt)))) + max_sequence_length
        embeds = torch.full((num_images_per_prompt, self.width), value)
        neg = None if negative_prompt is None else torch.full((1, self.width), -value)
        return (
            embeds,
            torch.ones(num_images_per_prompt, self.width, dtype = torch.bool),
            [neg] if neg is not None else None,
        )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROMPT_CACHE", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", raising = False)


def test_repeat_prompt_skips_the_encoder_and_matches_exactly():
    pipe = _EncodePipe()
    assert prompt_cache.install(pipe, identity = {"repo": "r"})
    first = pipe.encode_prompt("a cat", device = "cpu")
    second = pipe.encode_prompt("a cat", device = "cpu")
    assert pipe.calls == 1
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[2] is None and second[2] is None
    assert pipe._unsloth_prompt_cache.describe()["hits"] == 1


def test_every_argument_keys_the_entry():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe.encode_prompt("a dog")
    pipe.encode_prompt("a cat", max_sequence_length = 128)
    pipe.encode_prompt("a cat", num_images_per_prompt = 2)
    pipe.encode_prompt("a cat", negative_prompt = "blurry")
    pipe.encode_prompt("a cat", negative_prompt = "ugly")
    pipe.encode_prompt("a cat", dtype = torch.float16)
    assert pipe.calls == 7
    neg = pipe.encode_prompt("a cat", negative_prompt = "blurry")
    assert pipe.calls == 7 and neg[2][0] is not None


def test_device_is_placement_not_key():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", device = "cpu")
    pipe.encode_prompt("a cat", device = torch.device("cpu"))
    assert pipe.calls == 1


def test_hit_returns_fresh_tensors_so_in_place_edits_cannot_poison():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat")
    reference = first[0].clone()
    first[0].mul_(0)
    second = pipe.encode_prompt("a cat")
    assert torch.equal(second[0], reference)
    second[0].add_(5)
    third = pipe.encode_prompt("a cat")
    assert torch.equal(third[0], reference)
    assert pipe.calls == 1


def test_non_plain_arguments_bypass():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", image = object())
    pipe.encode_prompt("a cat", image = object())
    assert pipe.calls == 2
    embeds = torch.zeros(1, 4)
    out = pipe.encode_prompt("a cat", prompt_embeds = embeds)
    assert out[0] is embeds
    assert pipe._unsloth_prompt_cache.describe()["bypassed"] == 3


def test_lru_is_bounded_by_bytes(monkeypatch):
    # entry = 1024 B float32 embeds + 256 B bool mask = 1280 B
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", str(3000 / 1024 / 1024))
    pipe = _EncodePipe(width = 256)
    prompt_cache.install(pipe)
    cache = pipe._unsloth_prompt_cache
    pipe.encode_prompt("a")
    pipe.encode_prompt("b")
    assert len(cache) == 2 and cache.bytes == 2560
    pipe.encode_prompt("a")
    pipe.encode_prompt("c")
    assert len(cache) == 2 and cache.bytes <= cache.budget
    assert cache.describe()["evictions"] == 1
    calls = pipe.calls
    pipe.encode_prompt("a")
    assert pipe.calls == calls
    pipe.encode_prompt("b")
    assert pipe.calls == calls + 1


def test_entry_larger_than_budget_is_not_stored(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_MB", str(100 / 1024 / 1024))
    pipe = _EncodePipe(width = 256)
    prompt_cache.install(pipe)
    pipe.encode_prompt("a")
    pipe.encode_prompt("a")
    assert pipe.calls == 2 and len(pipe._unsloth_prompt_cache) == 0


def test_lora_change_misses_and_text_encoder_swap_misses():
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe._unsloth_loras = (("style", "/x/style.safetensors", 0.8),)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 2
    pipe._unsloth_loras = (("style", "/x/style.safetensors", 0.5),)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 3
    pipe._unsloth_loras = ()
    pipe.encode_prompt("a cat")
    assert pipe.calls == 3
    pipe.text_encoder = torch.nn.Linear(2, 2)
    pipe.encode_prompt("a cat")
    assert pipe.calls == 4


def test_lora_owner_is_read_for_workflow_pipes():
    owner = _EncodePipe()
    aux = _EncodePipe()
    prompt_cache.install(aux, lora_owner = owner)
    aux.encode_prompt("a cat")
    owner._unsloth_loras = (("style", "/x", 1.0),)
    aux.encode_prompt("a cat")
    assert aux.calls == 2


def test_workflow_pipes_share_the_owner_budget():
    owner = _EncodePipe()
    aux = _EncodePipe()
    assert prompt_cache.install(owner, identity = {"family": "f"})
    assert prompt_cache.install(
        aux, identity = {"family": "f", "workflow": "Img2Img"}, lora_owner = owner
    )
    assert prompt_cache.cache_for(aux) is prompt_cache.cache_for(owner)
    owner.encode_prompt("a cat")
    aux.encode_prompt("a cat")
    assert owner.calls == 1 and aux.calls == 1
    assert len(prompt_cache.cache_for(owner)) == 2


def test_release_and_disable(monkeypatch):
    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    prompt_cache.release(pipe)
    assert len(pipe._unsloth_prompt_cache) == 0
    pipe.encode_prompt("a cat")
    assert pipe.calls == 2
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE", "0")
    other = _EncodePipe()
    assert not prompt_cache.install(other)
    other.encode_prompt("a")
    other.encode_prompt("a")
    assert other.calls == 2


def test_install_is_idempotent_and_signature_preserved():
    import inspect

    pipe = _EncodePipe()
    prompt_cache.install(pipe)
    wrapped = pipe.encode_prompt
    prompt_cache.install(pipe)
    assert pipe.encode_prompt is wrapped
    assert "max_sequence_length" in inspect.signature(pipe.encode_prompt).parameters


def test_composes_with_the_disk_cache(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COND_CACHE_DIR", str(tmp_path))
    pipe = _EncodePipe()
    assert cond_cache.install(pipe, family = "fam", repo_id = "r", dtype = torch.float32)
    assert prompt_cache.install(pipe)
    pipe.encode_prompt("a cat")
    pipe.encode_prompt("a cat")
    assert pipe.calls == 1
    assert pipe._unsloth_cond_cache_stats["misses"] == 1
    assert pipe._unsloth_cond_cache_stats["hits"] == 0
    fresh = _EncodePipe()
    cond_cache.install(fresh, family = "fam", repo_id = "r", dtype = torch.float32)
    prompt_cache.install(fresh)
    fresh.encode_prompt("a cat")
    assert fresh.calls == 0 and fresh._unsloth_cond_cache_stats["hits"] == 1


def test_h3_shim_caches_text_only_calls(monkeypatch):
    calls = {"n": 0}

    def get_qwen3vl_prompt_embeds(
        text_encoder,
        processor,
        token_ids,
        vision_inputs = None,
        text_encoder_layer = 50,
        device = None,
        dtype = None,
    ):
        calls["n"] += 1
        return torch.full((1, len(token_ids), 3), float(sum(token_ids)))

    module = types.ModuleType("diffusers.modular_pipelines.minimax_h3.encoders")
    module.get_qwen3vl_prompt_embeds = get_qwen3vl_prompt_embeds
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3.encoders", module)

    class _Modular:
        blocks = ()

        def __init__(self):
            self.text_encoder = torch.nn.Linear(2, 2)

    pipe = _Modular()
    assert prompt_cache.install(pipe)
    shim = module.get_qwen3vl_prompt_embeds
    assert shim is not get_qwen3vl_prompt_embeds
    a = shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {},
        text_encoder_layer = 50,
        device = "cpu",
        dtype = torch.float32,
    )
    b = shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {},
        text_encoder_layer = 50,
        device = "cpu",
        dtype = torch.float32,
    )
    assert calls["n"] == 1 and torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    shim(pipe.text_encoder, None, [1, 2, 4], {}, 50, "cpu", torch.float32)
    assert calls["n"] == 2
    shim(
        pipe.text_encoder,
        None,
        [1, 2, 3],
        {"pixel_values": torch.zeros(1)},
        50,
        "cpu",
        torch.float32,
    )
    shim(torch.nn.Linear(2, 2), None, [1, 2, 3], {}, 50, "cpu", torch.float32)
    assert calls["n"] == 4
    other = _Modular()
    prompt_cache.install(other)
    assert module.get_qwen3vl_prompt_embeds is shim


def _h3_stub(monkeypatch):
    calls = {"n": 0}

    def get_qwen3vl_prompt_embeds(
        text_encoder,
        processor,
        token_ids,
        vision_inputs = None,
        text_encoder_layer = 50,
        device = None,
        dtype = None,
    ):
        calls["n"] += 1
        seen = sum(float(v.float().sum()) for v in (vision_inputs or {}).values())
        return torch.full((1, len(token_ids), 3), float(sum(token_ids)) + seen)

    module = types.ModuleType("diffusers.modular_pipelines.minimax_h3.encoders")
    module.get_qwen3vl_prompt_embeds = get_qwen3vl_prompt_embeds
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3.encoders", module)

    class _Modular:
        blocks = ()

        def __init__(self):
            self.text_encoder = torch.nn.Linear(2, 2)

    pipe = _Modular()
    assert prompt_cache.install(pipe)
    return pipe, module.get_qwen3vl_prompt_embeds, calls


def _vision(pixels = None, grid = (1, 4, 4)):
    return {
        "pixel_values": torch.arange(16 * 6, dtype = torch.float32).reshape(16, 6)
        if pixels is None
        else pixels,
        "image_grid_thw": torch.tensor([list(grid)]),
    }


def test_h3_shim_caches_vision_conditioned_calls_by_content(monkeypatch):
    pipe, shim, calls = _h3_stub(monkeypatch)
    te = pipe.text_encoder
    a = shim(te, None, [1, 2, 3], _vision(), 50, "cpu", torch.float32)
    # same content in fresh tensors (a new render re-runs the processor)
    b = shim(te, None, [1, 2, 3], _vision(), 50, "cpu", torch.float32)
    assert calls["n"] == 1 and torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    cache = prompt_cache.cache_for(pipe)
    assert cache.stats["hits"] == 1 and cache.stats["bypassed"] == 0

    changed = _vision()
    changed["pixel_values"][3, 2] += 1.0
    c = shim(te, None, [1, 2, 3], changed, 50, "cpu", torch.float32)
    assert calls["n"] == 2 and not torch.equal(a, c)
    shim(te, None, [1, 2, 3], _vision(grid = (1, 2, 8)), 50, "cpu", torch.float32)
    assert calls["n"] == 3
    half = _vision(pixels = torch.arange(16 * 6, dtype = torch.float32).reshape(16, 6).half())
    shim(te, None, [1, 2, 3], half, 50, "cpu", torch.float32)
    assert calls["n"] == 4
    # pixels only, grid dropped: another key
    shim(te, None, [1, 2, 3], {"pixel_values": _vision()["pixel_values"]}, 50, "cpu", torch.float32)
    assert calls["n"] == 5
    shim(te, None, [1, 2, 3], _vision(), 50, "cpu", torch.float32)
    assert calls["n"] == 5


def test_h3_vision_cache_bypasses_device_tensors_and_the_kill_switch(monkeypatch):
    pipe, shim, calls = _h3_stub(monkeypatch)
    te = pipe.text_encoder
    cache = prompt_cache.cache_for(pipe)
    on_meta = {"pixel_values": torch.zeros(4, 6, device = "meta")}
    with pytest.raises(TypeError):
        prompt_cache._vision_fingerprint(on_meta)
    with pytest.raises(TypeError):
        prompt_cache._vision_fingerprint({"pixel_values": [1, 2]})

    monkeypatch.setenv("UNSLOTH_H3_VISION_PROMPT_CACHE", "0")
    shim(te, None, [1, 2, 3], _vision(), 50, "cpu", torch.float32)
    shim(te, None, [1, 2, 3], _vision(), 50, "cpu", torch.float32)
    assert calls["n"] == 2 and cache.stats["bypassed"] == 2
    # text-only presentations keep caching under the switch
    shim(te, None, [1, 2, 3], {}, 50, "cpu", torch.float32)
    shim(te, None, [1, 2, 3], {}, 50, "cpu", torch.float32)
    assert calls["n"] == 3


def test_h3_vision_fingerprint_covers_bfloat16_bytes():
    x = torch.randn(8, 6).to(torch.bfloat16)
    y = x.clone()
    y.view(torch.int16)[0, 0] ^= 1
    assert prompt_cache._vision_fingerprint({"p": x}) == prompt_cache._vision_fingerprint(
        {"p": x.clone()}
    )
    assert prompt_cache._vision_fingerprint({"p": x}) != prompt_cache._vision_fingerprint({"p": y})
    # non-contiguous views hash their logical content
    t = torch.arange(24.0).reshape(4, 6)
    assert prompt_cache._vision_fingerprint({"p": t.t()}) == prompt_cache._vision_fingerprint(
        {"p": t.t().contiguous()}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_hit_lands_on_the_source_device():
    class _CudaPipe(_EncodePipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
            **kw,
        ):
            out = super().encode_prompt(prompt, **kw)
            return tuple(o.cuda() if torch.is_tensor(o) else o for o in out)

    pipe = _CudaPipe()
    prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat")
    second = pipe.encode_prompt("a cat")
    assert second[0].is_cuda and torch.equal(first[0], second[0]) and pipe.calls == 1


def test_load_installs_the_prompt_cache_and_unload_releases_it(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as diff_mod

    from .test_diffusion_backend import _loaded_backend

    calls = {"install": [], "release": []}
    monkeypatch.setattr(
        diff_mod.prompt_cache, "install", lambda pipe, **k: calls["install"].append(pipe) or True
    )
    monkeypatch.setattr(
        diff_mod.prompt_cache, "release", lambda pipe: calls["release"].append(pipe)
    )
    backend = _loaded_backend(tmp_path)
    pipe = backend._state.pipe
    assert calls["install"] == [pipe]
    backend.unload()
    assert pipe in calls["release"]


def test_budget_respects_a_cgroup_memory_limit(monkeypatch):
    # entries count against memory.max, so the budget is RAM/64 of the 2 GiB limit
    from core.inference import diffusion_memory

    monkeypatch.delenv(prompt_cache._ENV_BUDGET_MB, raising = False)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 2048)
    assert prompt_cache.budget_bytes() == 32 * 1024 * 1024
    monkeypatch.setenv(prompt_cache._ENV_BUDGET_MB, "100")
    assert prompt_cache.budget_bytes() == 100 * 1024 * 1024
    monkeypatch.setenv(prompt_cache._ENV_BUDGET_MB, "lots")
    assert prompt_cache.budget_bytes() == 32 * 1024 * 1024
    monkeypatch.delenv(prompt_cache._ENV_BUDGET_MB)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    host = prompt_cache._host_ram_bytes()  # 0 where sysconf is missing (Windows): no cap
    expected = 256 * 1024 * 1024
    if host:
        expected = min(expected, max(16 * 1024 * 1024, host // 64))
    assert prompt_cache.budget_bytes() == expected


def test_controlnet_pipe_gets_the_prompt_cache(monkeypatch):
    import threading

    from core.inference import diffusion_controlnet as dc
    from core.inference.diffusion import DiffusionBackend

    from . import test_diffusion_controlnet as tcn

    calls = []

    class _EncodingCNPipe(tcn._FakeCNPipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            calls.append(prompt)
            return torch.ones(1, 3, 4)

    fake = tcn._fake_diffusers()
    fake.FluxControlNetPipeline = _EncodingCNPipe
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    tcn._allow_cn_security(monkeypatch)
    backend = DiffusionBackend()
    state = tcn._state()
    backend._state = state
    pipe = backend._controlnet_pipe(
        state, dc.ResolvedControlNet("flux-union-pro", "repo/id", is_local = False), threading.Event()
    )
    assert prompt_cache.cache_for(pipe) is not None
    first = pipe.encode_prompt("a sloth", device = "cpu")
    again = pipe.encode_prompt("a sloth", device = "cpu")
    assert calls == ["a sloth"] and torch.equal(first, again)


def _image(
    color,
    size = (8, 6),
    mode = "RGB",
):
    Image = pytest.importorskip("PIL.Image")
    return Image.new(mode, size, color)


def test_edit_reference_image_is_cached_by_pixel_content():
    pipe = _EncodePipe()
    assert prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat", image = _image((10, 20, 30)))
    again = pipe.encode_prompt("a cat", image = _image((10, 20, 30)))
    as_list = pipe.encode_prompt("a cat", image = [_image((10, 20, 30))])
    assert pipe.calls == 2
    assert torch.equal(first[0], again[0])
    assert torch.equal(first[0], as_list[0])
    stats = pipe._unsloth_prompt_cache.describe()
    assert stats["bypassed"] == 0 and stats["hits"] == 1


@pytest.mark.parametrize(
    "other",
    [
        lambda: _image((10, 20, 31)),
        lambda: _image((10, 20, 30), size = (6, 8)),
        lambda: _image((10, 20, 30, 255), mode = "RGBA"),
    ],
)
def test_a_different_reference_image_misses(other):
    pipe = _EncodePipe()
    assert prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", image = _image((10, 20, 30)))
    pipe.encode_prompt("a cat", image = other())
    assert pipe.calls == 2
    assert pipe._unsloth_prompt_cache.describe()["hits"] == 0


def test_an_edited_image_object_misses_after_in_place_change():
    pipe = _EncodePipe()
    assert prompt_cache.install(pipe)
    image = _image((10, 20, 30))
    pipe.encode_prompt("a cat", image = image)
    image.putpixel((0, 0), (255, 0, 0))
    pipe.encode_prompt("a cat", image = image)
    assert pipe.calls == 2


def test_bypassed_encoder_error_is_not_chained_onto_the_key_error():
    class Boom(_EncodePipe):
        def encode_prompt(
            self,
            prompt,
            image = None,
            device = None,
        ):
            raise RuntimeError("encoder failed")

    pipe = Boom()
    assert prompt_cache.install(pipe)
    with pytest.raises(RuntimeError) as info:
        pipe.encode_prompt("a cat", image = object())
    assert info.value.__context__ is None
    assert pipe._unsloth_prompt_cache.describe()["bypassed"] == 1


def test_numpy_like_object_with_tobytes_still_bypasses():
    class FakeImage:
        mode = "RGB"
        size = (1, 1)

        def tobytes(self):
            return b"\0\0\0"

    pipe = _EncodePipe()
    assert prompt_cache.install(pipe)
    pipe.encode_prompt("a cat", image = FakeImage())
    assert pipe._unsloth_prompt_cache.describe()["bypassed"] == 1


def test_failed_lookup_encoder_error_is_not_chained(monkeypatch):
    class Boom(_EncodePipe):
        def encode_prompt(
            self,
            prompt,
            image = None,
            device = None,
        ):
            raise RuntimeError("encoder failed")

    pipe = Boom()
    assert prompt_cache.install(pipe)

    def broken_get(*_args, **_kwargs):
        raise OSError("lookup failed")

    monkeypatch.setattr(pipe._unsloth_prompt_cache, "get", broken_get)
    with pytest.raises(RuntimeError) as info:
        pipe.encode_prompt("a cat")
    assert info.value.__context__ is None


class WanStubPipeline:
    """Wan's shape: encode_prompt calls _get_t5_prompt_embeds once per text (prompt, then negative)."""

    def __init__(self):
        self.t5_calls = []
        self._execution_device = torch.device("cpu")
        self.text_encoder = torch.nn.Linear(2, 2)

    def _get_t5_prompt_embeds(
        self,
        prompt = None,
        num_videos_per_prompt = 1,
        max_sequence_length = 226,
        device = None,
        dtype = None,
    ):
        self.t5_calls.append(prompt)
        return torch.full((num_videos_per_prompt, 3), float(sum(map(ord, str(prompt)))))

    def encode_prompt(
        self,
        prompt,
        negative_prompt = None,
        do_classifier_free_guidance = True,
        device = None,
        dtype = None,
    ):
        pos = self._get_t5_prompt_embeds(prompt = prompt, device = device, dtype = dtype)
        neg = self._get_t5_prompt_embeds(prompt = negative_prompt, device = device, dtype = dtype)
        return pos, neg


def test_wan_negative_prompt_is_encoded_once_across_new_prompts():
    pipe = WanStubPipeline()
    assert prompt_cache.install(pipe)
    want = WanStubPipeline()
    for text in ("a cat", "a dog", "a boat"):
        got = pipe.encode_prompt(text, negative_prompt = "blurry")
        ref = want.encode_prompt(text, negative_prompt = "blurry")
        assert all(torch.equal(a, b) for a, b in zip(got, ref))
    assert pipe.t5_calls == ["a cat", "blurry", "a dog", "a boat"]


def test_wan_per_text_cache_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_PER_TEXT", "0")
    pipe = WanStubPipeline()
    assert prompt_cache.install(pipe)
    for text in ("a cat", "a dog"):
        pipe.encode_prompt(text, negative_prompt = "blurry")
    assert pipe.t5_calls == ["a cat", "blurry", "a dog", "blurry"]


def test_per_text_cache_only_for_known_pipelines():
    pipe = _EncodePipe()
    pipe._get_t5_prompt_embeds = lambda **kw: None
    pipe._encode_prompt = lambda **kw: None
    assert prompt_cache._per_text_method(pipe) is None
    assert prompt_cache._per_text_method(WanStubPipeline()) == "_get_t5_prompt_embeds"
    assert prompt_cache._per_text_method(ZImageStubPipeline()) == "_encode_prompt"


class ZImageStubPipeline:
    """diffusers ZImagePipeline's shape: encode_prompt encodes the prompt and, under CFG, the negative in ONE call,
    each through _encode_prompt, which templates the caller's prompt list in place."""

    def __init__(self):
        self.calls = []
        self._execution_device = torch.device("cpu")
        self.text_encoder = torch.nn.Linear(2, 2)

    def _encode_prompt(
        self,
        prompt,
        device = None,
        prompt_embeds = None,
        max_sequence_length = 512,
    ):
        if prompt_embeds is not None:
            return prompt_embeds
        if isinstance(prompt, str):
            prompt = [prompt]
        self.calls.append(list(prompt))
        out = [torch.full((len(p) + 1, 2), float(sum(map(ord, p)))) for p in prompt]
        for i, p in enumerate(prompt):
            prompt[i] = f"<user>{p}"
        return out

    def encode_prompt(
        self,
        prompt,
        device = None,
        do_classifier_free_guidance = True,
        negative_prompt = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        max_sequence_length = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        pos = self._encode_prompt(
            prompt = prompt, device = device, max_sequence_length = max_sequence_length
        )
        neg = []
        if do_classifier_free_guidance:
            negative = [negative_prompt or "" for _ in prompt]
            neg = self._encode_prompt(
                prompt = negative, device = device, max_sequence_length = max_sequence_length
            )
        return pos, neg


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _flat(v)]
    return [x] if x is not None else []


def test_cfg_negative_is_encoded_once_across_new_prompts():
    """Under CFG only the new prompt is encoded; the unchanged negative is reused. Results match uncached."""
    pipe = ZImageStubPipeline()
    assert prompt_cache.install(pipe)
    want = ZImageStubPipeline()
    for text in ("a cat", "a dog", "a boat", "a dog"):
        got = pipe.encode_prompt(text, negative_prompt = "blurry")
        ref = want.encode_prompt(text, negative_prompt = "blurry")
        a, b = _flat(got), _flat(ref)
        assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    texts = [c[0] if isinstance(c, list) else c for c in pipe.calls]
    assert texts == ["a cat", "blurry", "a dog", "a boat"]


def test_cfg_negative_per_text_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROMPT_CACHE_PER_TEXT", "0")
    pipe = ZImageStubPipeline()
    assert prompt_cache.install(pipe)
    for text in ("a cat", "a dog"):
        pipe.encode_prompt(text, negative_prompt = "blurry")
    assert [c[0] for c in pipe.calls] == ["a cat", "blurry", "a dog", "blurry"]


SCOPED_ENV = "UNSLOTH_DIFFUSION_TE_SCOPED_LORA_KEY"


class _AdapterPipe(_EncodePipe):
    def __init__(self, listed = None):
        super().__init__()
        self.transformer = torch.nn.Linear(2, 2)
        self.listed = listed or {}

    def get_list_adapters(self):
        return self.listed


class _DenoiserReadingPipe(_AdapterPipe):
    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self.transformer(torch.ones(1, 2)) + float(len(prompt)),)


class _DtypeOnlyPipe(_AdapterPipe):
    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        dtype = (
            self.text_encoder.weight.dtype
            if self.text_encoder is not None
            else self.transformer.dtype
        )
        return (torch.full((1, 2), float(len(prompt)), dtype = dtype),)


def _sweep(pipe, *entries):
    for entry in entries:
        pipe._unsloth_loras = entry
        pipe.encode_prompt("a cat")
    return pipe.calls


def test_a_denoiser_only_strength_sweep_reuses_the_text_embeddings(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _AdapterPipe({"transformer": ["style"]})
    prompt_cache.install(pipe)
    first = pipe.encode_prompt("a cat")
    calls = _sweep(pipe, *((("style", "/x/style.safetensors", w),) for w in (0.25, 0.5, 0.75, 1.0)))
    assert calls == 1
    pipe._unsloth_loras = ()
    assert torch.equal(pipe.encode_prompt("a cat")[0], first[0]) and pipe.calls == 1


@pytest.mark.parametrize(
    "listed",
    [
        {"text_encoder": ["style"]},
        {"transformer": ["style"], "text_encoder": ["style"]},  # one file patching both
        {},  # nothing says where it sits: the conservative key
    ],
)
def test_an_adapter_that_may_reach_the_text_encoder_still_misses(monkeypatch, listed):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _AdapterPipe(listed)
    prompt_cache.install(pipe)
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 2


def test_an_encode_that_runs_the_denoiser_keys_its_adapters(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _DenoiserReadingPipe({"transformer": ["style"]})
    prompt_cache.install(pipe)
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 2


def test_a_denoiser_dtype_read_runs_no_layer(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _DtypeOnlyPipe({"transformer": ["style"]})
    prompt_cache.install(pipe)
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 1


def test_the_scoped_key_switch_keys_every_adapter(monkeypatch):
    monkeypatch.setenv(SCOPED_ENV, "0")
    pipe = _AdapterPipe({"transformer": ["style"]})
    prompt_cache.install(pipe)
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 2
    assert prompt_cache._ENV_SCOPED_LORA_KEY == SCOPED_ENV


def test_adapters_merged_into_quantized_denoiser_weights_are_denoiser_only(monkeypatch):
    from core.inference import diffusion_lora_quant_merge as qm

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _AdapterPipe({})  # merged adapters leave PEFT: only the merge state names them
    prompt_cache.install(pipe)
    setattr(pipe, qm._MERGED_ATTR, {"layers": [], "weights": {"style": 0.5}, "device": "cpu"})
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 1


def test_an_instance_level_encode_override_keeps_the_full_key(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _AdapterPipe({"transformer": ["style"]})
    original = pipe.encode_prompt

    def encode_prompt(prompt, device = None):
        return original(prompt, device = device)

    pipe.encode_prompt = encode_prompt
    prompt_cache.install(pipe)
    assert _sweep(pipe, (("style", "/x", 0.5),), (("style", "/x", 0.7),)) == 2


def test_the_static_check_on_real_pipelines():
    diffusers = pytest.importorskip("diffusers")
    qwen = getattr(diffusers, "QwenImagePipeline", None)
    if qwen is not None:
        assert prompt_cache._encode_skips_denoisers(qwen, "encode_prompt")

    def _run(pipe, prompt):
        return pipe.transformer(prompt)

    class _Helper:
        def encode_prompt(self, prompt):
            return _run(
                self, prompt
            )  # self handed to a function the check does not follow: unprovable

    class _Inherited(_AdapterPipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            return super().encode_prompt(prompt, device = device)

    class _Dynamic(_AdapterPipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            return getattr(self, "transformer")(torch.ones(1, 2))

    class _Indirect(_AdapterPipe):
        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            return self._inner(prompt)

        def _inner(self, prompt):
            return self.unet(prompt)

    for cls in (_Helper, _Inherited, _Dynamic, _Indirect):
        assert not prompt_cache._encode_skips_denoisers(cls, "encode_prompt"), cls


class _HelperPipe(_AdapterPipe):
    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self._embed(prompt),)

    def _embed(self, prompt):
        return torch.full((1, 2), float(len(prompt)))


class _PropertyPipe(_AdapterPipe):
    @property
    def conditioning(self):
        return self.transformer.weight.detach().clone()

    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self.conditioning,)


class _CleanPropertyPipe(_PropertyPipe):
    @property
    def conditioning(self):
        return torch.ones(1, 2)


class _Reads:
    def __get__(
        self,
        obj,
        owner = None,
    ):
        return obj.transformer.weight.detach().clone()


class _DescriptorPipe(_PropertyPipe):
    conditioning = _Reads()


def _strength_sweep(pipe):
    out = []
    for strength in (0.25, 0.75):
        pipe._unsloth_loras = (("style", "/x/style.safetensors", strength),)
        with torch.no_grad():
            pipe.transformer.weight.fill_(strength)
        out.append(pipe.encode_prompt("same")[0].flatten()[0].item())
    return out


def test_an_instance_override_of_a_followed_helper_keeps_the_full_key(monkeypatch):
    import types

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _HelperPipe({"transformer": ["style"]})
    assert prompt_cache._encode_skips_denoisers(_HelperPipe, "encode_prompt")
    pipe._embed = types.MethodType(
        lambda self, prompt: self.transformer.weight.detach().clone(), pipe
    )
    prompt_cache.install(pipe)
    assert _strength_sweep(pipe) == [0.25, 0.75] and pipe.calls == 2


@pytest.mark.parametrize("cls", [_PropertyPipe, _DescriptorPipe])
def test_a_property_or_descriptor_reading_the_denoiser_keeps_the_full_key(monkeypatch, cls):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = cls({"transformer": ["style"]})
    assert not prompt_cache._encode_skips_denoisers(cls, "encode_prompt")
    prompt_cache.install(pipe)
    assert _strength_sweep(pipe) == [0.25, 0.75] and pipe.calls == 2


def test_a_property_running_no_denoiser_still_reuses_the_embeddings(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _CleanPropertyPipe({"transformer": ["style"]})
    prompt_cache.install(pipe)
    assert _strength_sweep(pipe) == [1.0, 1.0] and pipe.calls == 1


class Qwen3VLFakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.seen_use_cache = []

    def forward(
        self,
        x,
        use_cache = True,
    ):
        self.seen_use_cache.append(use_cache)
        return self.proj(x)


class _Qwen21LikePipe(_AdapterPipe):
    def __init__(self, listed = None):
        super().__init__(listed)
        self.text_encoder = Qwen3VLFakeEncoder()

    def _get_qwen_prompt_embeds(self, prompt):
        return self.text_encoder(torch.full((1, 2), float(len(prompt))))

    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self._get_qwen_prompt_embeds(prompt).detach(),)


def test_qwen21_no_kv_wrapper_installed_first_keeps_the_scoped_key(monkeypatch):
    # #17118 installs the no-K/V wrapper on the instance before the prompt cache: still denoiser-free
    from core.inference.diffusion_qwenimage21_te import skip_text_encoder_kv_cache

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_QWEN21_TE_NO_KV", raising = False)
    pipe = _Qwen21LikePipe({"transformer": ["style"]})
    assert skip_text_encoder_kv_cache(pipe)
    assert "_get_qwen_prompt_embeds" in pipe.__dict__
    assert prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)
    prompt_cache.install(pipe)
    calls = _sweep(pipe, *((("style", "/x/style.safetensors", w),) for w in (0.25, 0.5, 0.75)))
    assert calls == 1 and pipe.text_encoder.seen_use_cache == [False]


def test_an_unmarked_wrapper_of_the_same_helper_keeps_the_full_key(monkeypatch):
    import functools

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _Qwen21LikePipe({"transformer": ["style"]})
    original = pipe._get_qwen_prompt_embeds

    @functools.wraps(original)
    def _get_qwen_prompt_embeds(prompt):
        return original(prompt) + pipe.transformer.weight.sum()

    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    assert not prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)


def test_a_wraps_copy_of_the_transparent_marker_keeps_the_full_key(monkeypatch):
    # functools.wraps copies the no-K/V wrapper's marker onto this denoiser-reading wrapper
    import functools

    from core.inference.diffusion_qwenimage21_te import skip_text_encoder_kv_cache

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_QWEN21_TE_NO_KV", raising = False)
    pipe = _Qwen21LikePipe({"transformer": ["style"]})
    assert skip_text_encoder_kv_cache(pipe)
    original = pipe._get_qwen_prompt_embeds

    @functools.wraps(original)
    def _get_qwen_prompt_embeds(prompt):
        return original(prompt) * pipe.transformer.weight.detach().mean()

    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    assert not prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)
    prompt_cache.install(pipe)
    out = []
    for strength in (0.25, 0.75):
        pipe._unsloth_loras = (("style", "/x/style.safetensors", strength),)
        with torch.no_grad():
            pipe.transformer.weight.fill_(strength)
        out.append(float(pipe.encode_prompt("same")[0].flatten()[0]))
    assert out[0] != out[1] and pipe.calls == 2


class _Qwen21SelectablePipe(_Qwen21LikePipe):
    def _get_qwen_prompt_embeds(self, prompt):
        # select_final_hidden_state markers: norm.register_forward_hook outputs.hidden_states[-1]
        return self.text_encoder(torch.full((1, 2), float(len(prompt))))


def test_final_state_wrapper_over_the_no_kv_wrapper_keeps_the_scoped_key(monkeypatch):
    from core.inference.diffusion_qwenimage21_te import (
        select_final_hidden_state,
        skip_text_encoder_kv_cache,
    )

    monkeypatch.delenv(SCOPED_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_QWEN21_TE_NO_KV", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_TE_SELECT_HIDDEN", raising = False)
    pipe = _Qwen21SelectablePipe({"transformer": ["style"]})
    assert skip_text_encoder_kv_cache(pipe)
    assert select_final_hidden_state(pipe)
    assert prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)


def test_a_marked_wrapper_around_a_foreign_callable_keeps_the_full_key(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _Qwen21LikePipe({"transformer": ["style"]})

    def _get_qwen_prompt_embeds(prompt):
        return pipe.transformer(torch.ones(1, 2))

    _get_qwen_prompt_embeds.__wrapped__ = lambda prompt: None
    setattr(_get_qwen_prompt_embeds, prompt_cache.TRANSPARENT_WRAPPER_ATTR, True)
    pipe._get_qwen_prompt_embeds = _get_qwen_prompt_embeds
    assert not prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)


class _CallableHelper:
    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self):
        return self.pipe.transformer.weight.detach().clone()


class _CallableHelperPipe(_AdapterPipe):
    def __init__(self, listed = None):
        super().__init__(listed)
        self.helper = _CallableHelper(self)

    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self.helper(),)


class _WrapsDenoiser(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


class _ModuleHelperPipe(_AdapterPipe):
    def __init__(self, listed = None):
        super().__init__(listed)
        self.helper = _WrapsDenoiser(self.transformer)

    def encode_prompt(
        self,
        prompt,
        device = None,
    ):
        self.calls += 1
        return (self.helper(torch.ones(1, 2)).detach(),)


@pytest.mark.parametrize("cls", [_CallableHelperPipe, _ModuleHelperPipe])
def test_a_callable_instance_helper_reading_the_denoiser_keeps_the_full_key(monkeypatch, cls):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = cls({"transformer": ["style"]})
    assert not prompt_cache._encode_skips_denoisers(cls, "encode_prompt", pipe)
    prompt_cache.install(pipe)
    out = _strength_sweep(pipe)
    assert out[0] != out[1] and pipe.calls == 2


def test_the_text_encoder_module_is_not_an_opaque_helper(monkeypatch):
    monkeypatch.delenv(SCOPED_ENV, raising = False)
    pipe = _Qwen21LikePipe({"transformer": ["style"]})
    assert prompt_cache._encode_skips_denoisers(type(pipe), "encode_prompt", pipe)


def test_this_module_is_the_tree_under_test():
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(prompt_cache.__file__).resolve().is_relative_to(here)


def test_the_real_qwen21_pipeline_keeps_the_scoped_key_after_the_no_kv_wrapper(monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "QwenImage21Pipeline", None)
    if cls is None:
        pytest.skip("this diffusers has no QwenImage21Pipeline")
    from core.inference.diffusion_qwenimage21_te import skip_text_encoder_kv_cache

    monkeypatch.delenv("UNSLOTH_QWEN21_TE_NO_KV", raising = False)
    pipe = cls.__new__(cls)
    pipe.__dict__["text_encoder"] = Qwen3VLFakeEncoder()
    assert prompt_cache._encode_skips_denoisers(cls, "encode_prompt", pipe)
    assert skip_text_encoder_kv_cache(pipe) and "_get_qwen_prompt_embeds" in pipe.__dict__
    assert prompt_cache._encode_skips_denoisers(cls, "encode_prompt", pipe)
