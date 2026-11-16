# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local pre-quant cache (diffusion_prequant_cache): key, round trip, trust, atomicity, budget, host RAM.

Hermetic: a hand-built bf16 module quantized on the CPU (fp8 on torchao <= 0.17, int8 where the install can flatten
it), a base "repo" that is a local directory with one safetensors shard, and a private cache root per test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import prequant_safetensors as PS  # noqa: E402
from core.inference.diffusion_transformer_quant import (  # noqa: E402
    _REQUIRE_BF16_SCHEMES,
    _make_quant_config,
    divisible_for_scheme,
    exclude_tokens_for_scheme,
    make_filter_fn,
)

FAMILY = "tinyfam-cache-test"


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(512, 512)
        self.ff = torch.nn.Linear(512, 1024)
        self.norm = torch.nn.LayerNorm(512)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Root-level plain tensor (z-image's pad tokens): written beside torchao's flat set.
        self.pad_token = torch.nn.Parameter(torch.randn(1, 512))
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(2)])


def _usable_scheme() -> tuple[str, bool]:
    """``(scheme, really_quantized)``: a CPU-quantizable + flattenable scheme, else fp8 over plain bf16 (CPU-only CI);
    the cache never looks inside tensors, so the plain case still covers everything but subclass bit-identity."""
    for scheme in ("fp8", "int8"):
        try:
            m = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(torch.bfloat16)
            from torchao.quantization import quantize_

            quantize_(m, _make_quant_config(scheme))
            helpers = PS._torchao_helpers()
            if helpers is None:
                continue
            helpers[0](m.state_dict())
            return scheme, True
        except Exception:  # noqa: BLE001
            continue
    if PS._torchao_helpers() is None:
        pytest.skip("torchao safetensors helpers unavailable")
    return "fp8", False


_SCHEME: Optional[tuple[str, bool]] = None


@pytest.fixture(scope = "module")
def scheme() -> str:
    global _SCHEME
    if _SCHEME is None:
        _SCHEME = _usable_scheme()
    return _SCHEME[0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(C.ENV_MAX_GB, raising = False)
    monkeypatch.delenv("UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH", raising = False)
    C._HEADER_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    return root


def _fam(base: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(name = FAMILY, transformer_class = "Tiny", base_repo = str(base))


def _write_base(
    tmp_path: Path,
    seed: int = 0,
    name: str = "base",
) -> tuple[Path, Tiny]:
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    model = Tiny().to(torch.bfloat16)
    base = tmp_path / name
    (base / "transformer").mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(base / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    return base, model


def _quantize(model: Tiny, scheme: str) -> Tiny:
    exclude = exclude_tokens_for_scheme(scheme, FAMILY) + ("lora_",)
    divisible = divisible_for_scheme(scheme)
    if _SCHEME is not None and _SCHEME[1] and _SCHEME[0] == scheme:
        from torchao.quantization import quantize_
        quantize_(
            model,
            _make_quant_config(scheme),
            filter_fn = make_filter_fn(
                512,
                exclude_name_tokens = exclude,
                require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
                require_divisible = divisible,
            ),
        )
    model._unsloth_runtime_quant_info = {
        "scheme": scheme,
        "family": FAMILY,
        "min_features": 512,
        "exclude_name_tokens": list(exclude),
        "require_bf16": scheme in _REQUIRE_BF16_SCHEMES,
        "require_divisible": divisible,
        "fast_accum": None,
        "convrot_group": None,
        "convrot_fqns": [],
        "dtype": str(torch.bfloat16),
    }
    return model


def _flat_equal(a: dict, b: dict) -> list:
    fa, _ = C._flatten(a)
    fb, _ = C._flatten(b)
    assert set(fa) == set(fb)
    bad = []
    for k in fa:
        x, y = fa[k], fb[k]
        x = x.tensor() if isinstance(x, C._FlatRef) else x
        y = y.tensor() if isinstance(y, C._FlatRef) else y
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(k)
            continue
        if not torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)):
            bad.append(k)
    return bad


# ------------------------------------------------------------------------------------------------ round trip
def test_round_trip_is_bit_identical_and_validates(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    assert path is not None and Path(path).is_file()
    assert C.lookup(fam, scheme, base_repo = str(base)) == path

    ckpt = PS.load_prequant_safetensors(path, mmap = True)
    assert _flat_equal(model.state_dict(), ckpt["state_dict"]) == []
    meta = ckpt["metadata"]
    assert meta["scheme"] == scheme and meta["component"] == "transformer"
    assert meta["base_model_id"] == str(base) and meta["min_features"] == 512

    from core.inference.diffusion_prequant import _validate_checkpoint

    assert _validate_checkpoint(
        ckpt, scheme, str(base), None, min_features = 512, component = "transformer"
    )
    # A second save of an existing entry is a no-op.
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None


def test_streaming_writer_matches_safetensors_reader(tmp_path):
    tensors = {
        "a": torch.randn(3, 5).to(torch.bfloat16),
        "b": torch.arange(7, dtype = torch.int8),
        "c": torch.randn(2, 2, dtype = torch.float32),
        "empty": torch.empty(0, 4, dtype = torch.float16),
        "flag": torch.tensor([True, False]),
    }
    path = tmp_path / "x.safetensors"
    size = C.write_streaming_safetensors(path, tensors, {"k": "v"})
    assert size == path.stat().st_size
    from safetensors import safe_open

    with safe_open(str(path), framework = "pt") as fh:
        assert fh.metadata() == {"k": "v"}
        for name, value in tensors.items():
            assert torch.equal(fh.get_tensor(name), value), name
    # And through the mmap reader the hosted loader uses (it checks the byte ranges tile the file exactly).
    _meta, mapped = PS._mapped_tensors(str(path))
    for name, value in tensors.items():
        assert torch.equal(mapped[name], value), name


# ------------------------------------------------------------------------------------------------ key
def test_key_changes_with_source_scheme_and_policy(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    assert C.save_runtime_quant(model, fam, base_repo = str(base))
    assert C.lookup(fam, scheme, base_repo = str(base))

    other = "int8" if scheme == "fp8" else "fp8"
    assert C.lookup(fam, other, base_repo = str(base)) is None
    assert C.lookup(fam, scheme, base_repo = str(base), component = "transformer_2") is None

    monkeypatch.setattr(C, "POLICY_VERSION", C.POLICY_VERSION + 1)
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    monkeypatch.setattr(C, "POLICY_VERSION", C.POLICY_VERSION - 1)
    assert C.lookup(fam, scheme, base_repo = str(base))

    # Same architecture (identical header), different weights: the content identity changes the key.
    shard = base / "transformer" / "diffusion_pytorch_model.safetensors"
    header_before = C.read_header_bytes(shard)
    from safetensors.torch import save_file

    torch.manual_seed(123)
    save_file(
        {k: v.contiguous() for k, v in Tiny().to(torch.bfloat16).state_dict().items()}, str(shard)
    )
    os.utime(shard, ns = (time.time_ns() + 10_000_000, time.time_ns() + 10_000_000))
    assert C.read_header_bytes(shard) == header_before
    assert C.lookup(fam, scheme, base_repo = str(base)) is None


def test_hub_blob_name_is_the_content_identity(tmp_path):
    blob = tmp_path / ("a" * 64)
    blob.write_bytes(b"x")
    assert C._content_identity(str(blob), blob.stat()) == "blob:" + "a" * 64
    plain = tmp_path / "model.safetensors"
    plain.write_bytes(b"x")
    assert C._content_identity(str(plain), plain.stat()).startswith("stat:")


def test_missing_source_shards_is_a_miss(env, tmp_path, scheme):
    fam = _fam(tmp_path / "nowhere")
    assert C.lookup(fam, scheme, base_repo = str(tmp_path / "nowhere")) is None


def test_uncached_schemes_never_key(env, tmp_path):
    base, _ = _write_base(tmp_path)
    assert C.lookup(_fam(base), "nvfp4", base_repo = str(base)) is None


# ------------------------------------------------------------------------------------------------ atomicity
def test_interrupted_write_leaves_no_entry(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)

    wrote = []

    def boom(
        path,
        tensors,
        metadata = None,
        cancel_check = None,
        hasher = None,
    ):
        with open(path, "wb") as fh:
            fh.write(b"partial")
        wrote.append(path)
        raise OSError("killed mid-write")

    monkeypatch.setattr(C, "write_streaming_safetensors", boom)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert wrote, "the stub must really write partial bytes, or this passes on a signature error"
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    # The root's lock file and the per-key build lease are the cache's own, not a partial entry.
    leftovers = [
        p
        for p in env.rglob("*")
        if p.is_file()
        and p.name not in (C.LAST_USED_NAME, C._LOCK_NAME)
        and not p.name.endswith(".writing")
    ]
    assert leftovers == []


def test_artifact_without_manifest_or_wrong_size_is_a_miss(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = Path(C.save_runtime_quant(model, fam, base_repo = str(base)))
    manifest = path.parent / C.MANIFEST_NAME
    data = json.loads(manifest.read_text())
    data["bytes"] += 1
    manifest.write_text(json.dumps(data))
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    manifest.unlink()
    assert C.lookup(fam, scheme, base_repo = str(base)) is None


# ------------------------------------------------------------------------------------------------ switches, disk
def test_kill_switch(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    monkeypatch.setenv(C.ENV_MODE, "0")
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert not env.exists() or not any(env.iterdir())
    monkeypatch.delenv(C.ENV_MODE)
    assert C.save_runtime_quant(model, fam, base_repo = str(base))
    monkeypatch.setenv(C.ENV_MODE, "0")
    assert C.lookup(fam, scheme, base_repo = str(base)) is None


def test_disk_short_skips_the_save(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    usage = types.SimpleNamespace(total = 1 << 40, used = 1 << 40, free = 1 << 20)
    monkeypatch.setattr(C.shutil, "disk_usage", lambda _p: usage)
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is None
    assert not any(p.name == C.FILE_NAME for p in env.rglob("*"))


def test_baked_lora_and_native_and_other_dtypes_skip(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    model._unsloth_runtime_quant_info["lora"] = True
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    model._unsloth_runtime_quant_info.pop("lora")
    model._unsloth_runtime_quant_info["dtype"] = str(torch.float16)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    model._unsloth_runtime_quant_info = {"scheme": "int8", "native": True}
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    del model._unsloth_runtime_quant_info
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None


def test_lora_keys_in_the_state_dict_skip(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    model.blocks[0].lora_A = torch.nn.Linear(4, 4)
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is None


def test_runtime_filter_mismatch_skips(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    model._unsloth_runtime_quant_info["min_features"] = 256
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is None


def test_fp8_forced_non_default_accumulate_is_not_cached_under_the_auto_label(env, tmp_path):
    """A precise-accumulate fp8 build must not be stored as the auto (fast) artifact a default load would restore."""
    from core.inference.diffusion_transformer_quant import _resolve_fast_accum

    base, model = _write_base(tmp_path)
    _quantize(model, "fp8")
    model._unsloth_runtime_quant_info["fast_accum"] = not _resolve_fast_accum(None)
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is None
    assert C.lookup(_fam(base), "fp8", base_repo = str(base)) is None
    # The auto mode (and a forced value equal to it) is what the key and the metadata describe.
    model._unsloth_runtime_quant_info["fast_accum"] = _resolve_fast_accum(None)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert path is not None and C.lookup(_fam(base), "fp8", base_repo = str(base)) == path


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_a_save_hands_the_flatten_copies_back_to_the_device(env, tmp_path, scheme):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(4)]).to(torch.bfloat16)
    base = tmp_path / "big"
    (base / "transformer").mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(base / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    model = _quantize(model.cuda(), scheme)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is not None
    assert torch.cuda.memory_reserved() <= before


# ------------------------------------------------------------------------------------------------ eviction
def test_lru_eviction_spares_recent_and_kept_keys(env):
    env.mkdir(parents = True)
    now = time.time()
    dirs = []
    for i, age in enumerate((7200, 3600, 30)):
        d = env / (f"{i:x}" * 32)[:32]
        d.mkdir()
        (d / C.FILE_NAME).write_bytes(b"\0" * 1000)
        os.utime(d / C.FILE_NAME, (now - age, now - age))
        (d / C.LAST_USED_NAME).touch()
        os.utime(d / C.LAST_USED_NAME, (now - age, now - age))
        dirs.append(d)
    removed = C.evict(root = env, max_bytes = 1500)
    # Oldest goes first; the 30 s old key is inside the grace window and stays even over budget.
    assert removed == [dirs[0].name, dirs[1].name]
    assert dirs[2].exists()
    assert (
        C.evict(root = env, max_bytes = 0) == []
    )  # max_bytes=0 -> nothing over budget except grace-held


def test_budget_env(monkeypatch):
    monkeypatch.setenv(C.ENV_MAX_GB, "0")
    assert C.max_cache_bytes() is None
    monkeypatch.setenv(C.ENV_MAX_GB, "2")
    assert C.max_cache_bytes() == 2 << 30
    monkeypatch.setenv(C.ENV_MAX_GB, "junk")
    assert C.max_cache_bytes() == int(C.DEFAULT_MAX_GB * (1 << 30))


# ------------------------------------------------------------------------------------------------ trust
def test_cache_root_is_trusted_without_the_allowlist_and_symlinks_cannot_escape(
    env, tmp_path, scheme
):
    from core.inference.diffusion_prequant import (
        _local_prequant_path_allowed,
        local_prequant_path_ready,
    )

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert C.path_in_cache(path)
    assert _local_prequant_path_allowed(path) and local_prequant_path_ready(path)

    outside = tmp_path / "outside" / C.FILE_NAME
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert not _local_prequant_path_allowed(str(outside))
    link_dir = env / ("f" * 32)
    link_dir.mkdir()
    (link_dir / C.FILE_NAME).symlink_to(outside)
    assert not C.path_in_cache(link_dir / C.FILE_NAME)
    assert not _local_prequant_path_allowed(str(link_dir / C.FILE_NAME))
    # Only the artifact name at key depth counts, not any file under the root.
    stray = env / "stray.safetensors"
    stray.write_bytes(b"x")
    assert not C.path_in_cache(stray)


def test_resolver_falls_back_to_the_cache_only_without_a_hosted_repo(env, tmp_path, scheme):
    from core.inference.diffusion_prequant import resolve_prequant_source, usable_prequant_source

    base, model = _write_base(tmp_path)
    fam = _fam(base)
    assert resolve_prequant_source(fam, scheme, base_repo = str(base)) is None
    _quantize(model, scheme)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    src = resolve_prequant_source(fam, scheme, base_repo = str(base))
    assert src is not None and src.kind == "path" and src.location == path
    usable = usable_prequant_source(fam, scheme, base_repo = str(base))
    assert usable is not None and usable.location == path
    # An explicit override still wins.
    override = resolve_prequant_source(
        fam, scheme, path_override = "/x/y.safetensors", base_repo = str(base)
    )
    assert override.location == "/x/y.safetensors"


# ------------------------------------------------------------------------------------------------ quantize stamp
def test_quantize_transformer_stamps_what_the_cache_needs(env, tmp_path, scheme, monkeypatch):
    import core.inference.diffusion_transformer_quant as Q

    if not _SCHEME[1]:
        pytest.skip("torchao cannot quantize on this host's CPU")
    base, model = _write_base(tmp_path)
    monkeypatch.setattr(Q, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    monkeypatch.setattr(Q, "native_quant_scheme", lambda *a, **k: None)
    pipe = types.SimpleNamespace(transformer = model)
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu")
    assert Q.quantize_transformer(pipe, target, mode = scheme, family = FAMILY) == scheme
    info = model._unsloth_runtime_quant_info
    assert (
        info["scheme"] == scheme
        and info["min_features"] == 512
        and "lora_" in info["exclude_name_tokens"]
    )
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert path and C.lookup(_fam(base), scheme, base_repo = str(base)) == path


# ------------------------------------------------------------------------------------------------ host RAM
def test_streaming_save_does_not_hold_the_state_dict_on_the_host(tmp_path):
    """ru_maxrss of a child writing 384 MB: the streaming writer adds far less than the payload, save_file about it."""
    script = textwrap.dedent(
        f"""
        import resource, sys
        sys.path.insert(0, {str(BACKEND)!r})
        import torch
        from core.inference.diffusion_prequant_cache import write_streaming_safetensors
        tensors = {{f"t{{i}}": torch.ones(1024, 1024, 24, dtype = torch.uint8) for i in range(16)}}
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.argv[1] == "stream":
            write_streaming_safetensors(sys.argv[2], tensors, {{}})
        else:
            from safetensors.torch import save_file
            save_file(tensors, sys.argv[2])
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print((after - before) // 1024)
        """
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    def grow(mode: str) -> int:
        out = subprocess.run(
            [sys.executable, "-c", script, mode, str(tmp_path / f"{mode}.st")],
            capture_output = True,
            text = True,
            env = env,
            timeout = 300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        return int(out.stdout.strip().splitlines()[-1])

    stream_mb = grow("stream")
    assert stream_mb < 64, stream_mb
    assert (tmp_path / "stream.st").stat().st_size > 384 * 2**20


def test_full_pipeline_image_quantize_is_saved_to_the_cache(monkeypatch):
    """A model_kind=pipeline load quantizes in its own loop (not _load_dense_quant_pipeline); it must save too."""
    import inspect

    from core.inference import diffusion as D

    calls = []
    monkeypatch.setattr(
        C, "save_runtime_quant", lambda t, fam, **kw: calls.append((t, fam, kw)) or "/x"
    )
    pipe = types.SimpleNamespace(transformer = object())
    fam = types.SimpleNamespace(name = FAMILY)
    save = D.DiffusionBackend._save_pipeline_quant_to_cache
    assert (
        save(pipe, fam, [("transformer", pipe.transformer)], loras = [], base = "b", source_base = "/src")
        == "/x"
    )
    assert calls and calls[0][0] is pipe.transformer and calls[0][2]["source_base"] == "/src"
    calls.clear()
    # Two denoisers (no lookup slot for the second), an active LoRA, or no family: nothing saved.
    two = [("transformer", object()), ("transformer_2", object())]
    assert save(pipe, fam, two, loras = [], base = "b", source_base = None) is None
    assert (
        save(
            pipe,
            fam,
            [("transformer", pipe.transformer)],
            loras = [("a", 1.0)],
            base = "b",
            source_base = None,
        )
        is None
    )
    assert (
        save(pipe, None, [("transformer", pipe.transformer)], loras = [], base = "b", source_base = None)
        is None
    )
    assert calls == []
    # And the pipeline-kind quantize loop calls it once every denoiser engaged.
    src = inspect.getsource(D.DiffusionBackend.load_pipeline)
    engaged = src.index("if engaged and len(engaged) == len(denoisers):")
    assert "_save_pipeline_quant_to_cache(" in src[engaged : engaged + 600]


def test_source_shards_are_found_in_the_pre_change_hub_root(tmp_path, monkeypatch):
    """After a cache-folder change the key must still come from shards in the import-time Hub root."""
    from huggingface_hub import constants
    from safetensors.torch import save_file

    old_root, live_root = tmp_path / "old_hub", tmp_path / "live_hub"
    live_root.mkdir()
    repo = "org/tiny"
    snap = old_root / "models--org--tiny" / "snapshots" / ("c" * 40) / "transformer"
    snap.mkdir(parents = True)
    (old_root / "models--org--tiny" / "refs").mkdir(parents = True)
    (old_root / "models--org--tiny" / "refs" / "main").write_text("c" * 40)
    save_file({"w": torch.zeros(4)}, str(snap / "diffusion_pytorch_model.safetensors"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(old_root))
    monkeypatch.setattr(C, "_hub_cache_dir", lambda: str(live_root))
    found = C.source_shards(repo)
    assert found and found[0].endswith("diffusion_pytorch_model.safetensors")
    # An explicit cache_dir is still that root only.
    assert C.source_shards(repo, cache_dir = str(live_root)) is None


def _old_entry(root: Path, name: str, nbytes: int, age: float) -> Path:
    d = root / name
    d.mkdir(parents = True)
    (d / C.FILE_NAME).write_bytes(b"\0" * nbytes)
    stamp = time.time() - age
    for f in (C.FILE_NAME, C.LAST_USED_NAME):
        (d / f).touch()
        os.utime(d / f, (stamp, stamp))
    return d


def test_a_full_volume_rotates_stale_entries_out_before_skipping_the_save(
    env, tmp_path, scheme, monkeypatch
):
    """Old entries filling the volume must be evicted to make room, not make every new save fail for good."""
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    stale = _old_entry(env, "a" * 32, 4096, age = 7200)
    disk = {"free": 1 << 20}

    def usage(_p):
        # Free space comes back once the stale entry is gone.
        free = disk["free"] if stale.exists() else 1 << 40
        return types.SimpleNamespace(total = 1 << 41, used = 0, free = free)

    monkeypatch.setattr(C.shutil, "disk_usage", usage)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert path is not None and not stale.exists()


def test_eviction_rechecks_an_entry_another_process_just_resolved(env, monkeypatch):
    """The scan's last-used time is stale once another process resolves the entry: it must survive."""
    env.mkdir(parents = True)
    entry = _old_entry(env, "b" * 32, 1000, age = 7200)
    real_dir_bytes = C._dir_bytes

    def dir_bytes(path):
        # Another process resolves (and so touches) the entry right after this process scanned it.
        if path == entry:
            C._touch(entry)
        return real_dir_bytes(path)

    monkeypatch.setattr(C, "_dir_bytes", dir_bytes)
    assert C.evict(root = env, max_bytes = 0) == []
    assert entry.exists()


def test_lookup_takes_the_eviction_lock_for_its_check_and_touch(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    held = []
    real_entry = C._entry
    monkeypatch.setattr(C, "_entry", lambda key: held.append(C._LOCK_DEPTH[0]) or real_entry(key))
    assert C.lookup(_fam(base), scheme, base_repo = str(base)) == path
    assert held == [1]


def test_the_cache_lock_is_a_real_lock_on_windows(env, monkeypatch):
    """No fcntl (Windows): the lookup / eviction sections must still hold a cross-process lock, via msvcrt."""
    calls = []
    fake = types.SimpleNamespace(
        LK_LOCK = 1,
        LK_UNLCK = 0,
        locking = lambda fd, mode, nbytes: calls.append((mode, nbytes)),
    )
    monkeypatch.setitem(sys.modules, "fcntl", None)  # import fcntl -> ImportError
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    with C._cache_lock(env):
        assert calls == [(fake.LK_LOCK, 1)]
    assert calls == [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)]


def test_a_windows_lock_that_stays_contended_still_runs_the_section(env, monkeypatch):
    def locking(fd, mode, nbytes):
        raise OSError("deadlock avoided")

    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(
        sys.modules, "msvcrt", types.SimpleNamespace(LK_LOCK = 1, LK_UNLCK = 0, locking = locking)
    )
    ran = []
    with C._cache_lock(env):
        ran.append(True)
    assert ran == [True]


def test_an_artifact_over_the_budget_never_evicts_the_existing_cache(
    env, tmp_path, scheme, monkeypatch
):
    """A save the budget will refuse anyway must not first evict valid entries to make disk room for it."""
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    stale = _old_entry(env, "c" * 32, 4096, age = 7200)
    monkeypatch.setenv(
        C.ENV_MAX_GB, str(1 / (1 << 30))
    )  # a 1-byte budget: every artifact is over it
    usage = types.SimpleNamespace(total = 1 << 41, used = 0, free = 1 << 20)
    monkeypatch.setattr(C.shutil, "disk_usage", lambda _p: usage)
    assert C.save_runtime_quant(model, _fam(base), base_repo = str(base)) is None
    assert stale.exists()


def test_bases_sharing_their_weights_but_not_their_identity_never_share_an_entry(
    env, tmp_path, scheme
):
    """Bases sharing one Hub blob fingerprint alike, but the loader checks base_model_id, so each
    base needs its own cache entry."""
    base_a, model = _write_base(tmp_path, name = "model-a")
    shard = "transformer/diffusion_pytorch_model.safetensors"
    blob = tmp_path / "blobs" / ("ab" * 32)
    blob.parent.mkdir()
    os.replace(base_a / shard, blob)
    base_b = tmp_path / "model-b"
    (base_b / "transformer").mkdir(parents = True)
    for b in (base_a, base_b):
        os.symlink(blob, b / shard)
    assert C.source_fingerprint([base_a / shard]) == C.source_fingerprint([base_b / shard])
    _quantize(model, scheme)
    assert C.save_runtime_quant(model, _fam(base_a), base_repo = str(base_a))
    assert C.lookup(_fam(base_b), scheme, base_repo = str(base_b)) is None
    path_b = C.save_runtime_quant(model, _fam(base_b), base_repo = str(base_b))
    assert path_b is not None
    header = json.loads(C.read_header_bytes(path_b)[8:])
    assert json.loads(header["__metadata__"][PS.UNSLOTH_METADATA_KEY])["base_model_id"] == str(
        base_b
    )
    assert C.lookup(_fam(base_a), scheme, base_repo = str(base_a)) != path_b


def _corrupt_in_place(path: str) -> None:
    """Same-size corruption: manifest key and byte count still match, so ``_entry`` accepts it."""
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.write(b"\xff" * min(size, 64))


def test_an_entry_the_loader_rejects_is_replaced_by_the_dense_build(env, tmp_path, scheme):
    from core.inference.diffusion_prequant import PrequantSource, load_prequantized_transformer

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    good = Path(path).read_bytes()
    _corrupt_in_place(path)
    assert C._entry(Path(path).parent.name) is not None  # size-valid: still looks cached

    source = PrequantSource(kind = "path", location = path, filename = None)
    out = load_prequantized_transformer(
        Tiny, str(base), source, device = "cpu", dtype = torch.bfloat16, scheme = scheme
    )
    assert out is None
    # The load's dense fallback then saves: the corrupt entry is replaced, not kept as "already cached".
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) == path
    assert Path(path).read_bytes() == good
    # Replaced once: a further save of the now good entry is skipped again.
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None


def test_a_forced_accumulate_mismatch_is_not_a_bad_entry(env, tmp_path, scheme, monkeypatch):
    """A forced fp8 accumulate the entry does not bake is the request's choice: the entry stays for default loads."""
    from core.inference import diffusion_prequant as P

    monkeypatch.setattr(P, "hosted_fast_accum_conflict", lambda s, f: f is not None)
    rejected = []
    monkeypatch.setattr(C, "note_rejected", lambda path, logger = None: rejected.append(path))
    source = P.PrequantSource(
        kind = "path", location = str(env / ("d" * 32) / C.FILE_NAME), filename = None
    )
    kwargs = dict(device = "cpu", dtype = torch.bfloat16, scheme = "fp8")
    assert P.load_prequantized_transformer(Tiny, "x", source, fast_accum = False, **kwargs) is None
    assert rejected == []
    assert P.load_prequantized_transformer(Tiny, "x", source, **kwargs) is None
    assert len(rejected) == 1


def _flip_payload_byte(path: str, tensor_suffix: str) -> None:
    """Same-size corruption inside one tensor's payload (the header, key and byte count all still match)."""
    raw = C.read_header_bytes(path)
    header = json.loads(raw[8:])
    name = next(k for k in header if k != "__metadata__" and k.endswith(tensor_suffix))
    start = len(raw) + header[name]["data_offsets"][0]
    with open(path, "r+b") as fh:
        fh.seek(start)
        byte = fh.read(1)
        fh.seek(start)
        fh.write(bytes([byte[0] ^ 0x5A]))


@pytest.mark.parametrize(
    "victim", ["to_q._weight_qdata|to_q.weight", "norm.bias", "unsloth_root::pad_token"]
)
def test_same_size_corruption_of_any_tensor_fails_the_payload_digest(
    env, tmp_path, scheme, monkeypatch, victim
):
    """Quantized weights, plain tensors the filter left alone and root-level parameters alike: a flipped byte anywhere
    in the artifact refuses it before its tensors are read."""
    from core.inference import diffusion_prequant as P

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert C.verify_entry(path)
    names = json.loads(C.read_header_bytes(path)[8:])
    suffix = next(v for v in victim.split("|") if any(k.endswith(v) for k in names))
    _flip_payload_byte(path, suffix)
    assert not C.verify_entry(path)

    reads = []
    monkeypatch.setattr(P, "_read_prequant_for", lambda *a, **k: reads.append(a) or {})
    source = P.PrequantSource(kind = "path", location = path, filename = None)
    assert (
        P.load_prequantized_transformer(
            Tiny, str(base), source, device = "cpu", dtype = torch.bfloat16, scheme = scheme
        )
        is None
    )
    assert reads == []


def test_the_save_takes_no_uncancellable_pass_over_the_weights(env, tmp_path, scheme, monkeypatch):
    """The payload digest is taken inside the cancellable writer: no whole-model fingerprint pass before it."""
    from core.inference import diffusion_prequant as P

    def boom(*_a, **_k):
        raise AssertionError("whole-model fingerprint pass outside the writer")

    monkeypatch.setattr(P, "canonical_weight_fingerprint", boom)
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    path = C.save_runtime_quant(model, _fam(base), base_repo = str(base))
    assert path is not None
    manifest = json.loads((Path(path).parent / C.MANIFEST_NAME).read_text())
    import hashlib

    assert manifest["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_rejected_artifact_is_reclaimed_before_measuring_room_for_its_replacement(
    env, tmp_path, scheme, monkeypatch
):
    """A corrupt entry filling a nearly full volume: deleting it is what makes room for its own replacement."""
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    good = Path(path).read_bytes()
    _corrupt_in_place(path)
    C.note_rejected(path)
    kdir = Path(path).parent
    marker = kdir / "old-artifact-still-here"
    marker.touch()

    def usage(_p):
        # Room for the replacement only once the rejected entry's bytes are gone.
        return types.SimpleNamespace(
            total = 1 << 41, used = 0, free = (1 << 20) if marker.exists() else (1 << 40)
        )

    monkeypatch.setattr(C.shutil, "disk_usage", usage)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) == path
    assert Path(path).read_bytes() == good


def test_a_cancelled_load_stops_the_cache_write_between_tensors(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    calls = []

    def cancel_check():
        calls.append(1)
        if len(calls) > 2:
            raise RuntimeError("Diffusion load was cancelled.")

    assert (
        C.save_runtime_quant(model, _fam(base), base_repo = str(base), cancel_check = cancel_check)
        is None
    )
    assert len(calls) == 3
    # the root lock and the per-key build lease are the cache's own files, not a partial entry
    leftovers = [
        p
        for p in env.rglob("*")
        if p.is_file() and p.name != C._LOCK_NAME and not p.name.endswith(".writing")
    ]
    assert leftovers == []


def test_every_load_path_hands_its_cancellation_check_to_the_cache_save():
    import inspect

    from core.inference import diffusion as D
    from core.inference import video as V

    dense = inspect.getsource(D.DiffusionBackend._load_dense_quant_pipeline)
    assert "cancel_check = check_cancelled" in dense
    pipeline = inspect.getsource(D.DiffusionBackend.load_pipeline)
    assert "_save_pipeline_quant_to_cache(" in pipeline and "cancel_check = lambda" in pipeline
    assert "cancel_check = cancel_check" in inspect.getsource(
        D.DiffusionBackend._save_pipeline_quant_to_cache
    )
    video = inspect.getsource(V.VideoBackend.load_pipeline)
    assert (
        "_raise_if_video_load_superseded(_load_token)"
        in video.split("_save_runtime_quant_to_cache(", 1)[1][:400]
    )
    assert "cancel_check = cancel_check" in inspect.getsource(V._save_runtime_quant_to_cache)


def test_a_local_shard_replaced_with_its_size_and_mtime_kept_is_a_new_source(env, tmp_path):
    """A metadata-preserving replacement (same header, size and mtime, different weights) must change the key."""
    from safetensors.torch import save_file

    base, _model = _write_base(tmp_path)
    shard = base / "transformer" / "diffusion_pytorch_model.safetensors"
    before = C.source_fingerprint([shard])
    st = shard.stat()
    torch.manual_seed(321)
    fresh = tmp_path / "fresh.safetensors"
    save_file(
        {k: v.contiguous() for k, v in Tiny().to(torch.bfloat16).state_dict().items()}, str(fresh)
    )
    assert fresh.stat().st_size == st.st_size
    os.replace(fresh, shard)
    os.utime(shard, ns = (st.st_atime_ns, st.st_mtime_ns))
    assert shard.stat().st_mtime_ns == st.st_mtime_ns and shard.stat().st_size == st.st_size
    assert C.source_fingerprint([shard]) != before


def _hub_snapshot(
    tmp_path: Path,
    base_dir: Path,
    commit: str = "c" * 40,
) -> tuple[str, str]:
    """Lay ``base_dir``'s shard out like the Hub cache: snapshots/<commit>/<file> -> blobs/<sha256>."""
    import hashlib

    shard = base_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    data = shard.read_bytes()
    blob_name = hashlib.sha256(data).hexdigest()
    repo_dir = tmp_path / "hub" / "models--org--tiny"
    (repo_dir / "blobs").mkdir(parents = True)
    (repo_dir / "blobs" / blob_name).write_bytes(data)
    snap = repo_dir / "snapshots" / commit / "transformer"
    snap.mkdir(parents = True)
    link = snap / "diffusion_pytorch_model.safetensors"
    os.symlink(repo_dir / "blobs" / blob_name, link)
    return str(link), blob_name


def test_an_online_hit_on_a_hub_snapshot_is_revalidated_against_the_moved_main(
    env, tmp_path, scheme, monkeypatch
):
    import huggingface_hub

    from core.inference import diffusion_prequant as P

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    link, blob = _hub_snapshot(tmp_path, base)
    monkeypatch.setattr(C, "source_shards", lambda *_a, **_k: [link])
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    C._REMOTE_MEMO.clear()
    remote = {"etag": blob, "calls": 0}

    def metadata(url, **_k):
        remote["calls"] += 1
        assert "org/tiny" in url and "transformer/diffusion_pytorch_model.safetensors" in url
        return types.SimpleNamespace(etag = remote["etag"])

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = "org/tiny")
    assert path and C.lookup(fam, scheme, base_repo = "org/tiny") == path  # main unchanged: a hit

    C._REMOTE_MEMO.clear()
    remote["etag"] = "f" * 64  # main moved to new weights
    assert C.lookup(fam, scheme, base_repo = "org/tiny") is None

    # Offline or local-files-only: the local snapshot stands, and the Hub is not asked.
    C._REMOTE_MEMO.clear()
    calls = remote["calls"]
    token = P._LOCAL_FILES_ONLY.set(True)
    try:
        assert C.lookup(fam, scheme, base_repo = "org/tiny") == path
    finally:
        P._LOCAL_FILES_ONLY.reset(token)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    assert C.lookup(fam, scheme, base_repo = "org/tiny") == path
    assert remote["calls"] == calls

    # Unanswerable (network down, gated without a token): the local snapshot stands.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)

    def down(url, **_k):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", down)
    assert C.lookup(fam, scheme, base_repo = "org/tiny") == path


def _share_blob(tmp_path: Path, link: str) -> str:
    """Move ``link``'s repo blob into the huggingface_hub >= 1.32 shared store: ``<repo>/blobs/<etag>`` becomes a
    relative link to ``<cache>/blobs/<xx>/<xet hash>``, so the chain ends on a 64-hex name that is not the etag."""
    import hashlib

    blob = Path(os.path.dirname(link), os.readlink(link))
    hub = tmp_path / "hub"
    xet_hash = hashlib.sha256(b"xet" + blob.read_bytes()).hexdigest()
    try:
        from huggingface_hub.utils._shared_blobs import publish_blob_to_shared_store
    except ImportError:
        publish_blob_to_shared_store = None
    if publish_blob_to_shared_store is not None:
        assert publish_blob_to_shared_store(
            blob_path = str(blob),
            xet_hash = xet_hash,
            cache_dir = str(hub),
            expected_size = blob.stat().st_size,
        )
    else:  # older hub: the same layout by hand
        store = hub / "blobs" / xet_hash[:2] / xet_hash
        store.parent.mkdir(parents = True)
        os.replace(blob, store)
        os.symlink(os.path.relpath(store, blob.parent), blob)
    assert blob.is_symlink() and os.path.basename(os.path.realpath(link)) == xet_hash
    return xet_hash


@pytest.mark.parametrize("relative", [False, True])
def test_a_shared_blob_store_snapshot_is_judged_by_its_repo_blob_etag(
    env, tmp_path, scheme, monkeypatch, relative
):
    import huggingface_hub

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    link, blob = _hub_snapshot(tmp_path, base)
    if relative:  # the hub's own snapshot links are relative
        target = os.readlink(link)
        os.unlink(link)
        os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
    xet_hash = _share_blob(tmp_path, link)
    assert C._hub_coords(link)[2] == blob != xet_hash
    monkeypatch.setattr(C, "source_shards", lambda *_a, **_k: [link])
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    C._REMOTE_MEMO.clear()
    remote = {"etag": blob}
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda url, **_k: types.SimpleNamespace(etag = remote["etag"]),
    )
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = "org/tiny")
    assert path and C.lookup(fam, scheme, base_repo = "org/tiny") == path  # main unchanged: a hit
    C._REMOTE_MEMO.clear()
    remote["etag"] = "f" * 64  # main moved to new weights
    assert C.lookup(fam, scheme, base_repo = "org/tiny") is None


def test_hub_coords_of_legacy_and_copied_snapshots_keep_their_names(tmp_path):
    base = tmp_path / "base" / "transformer"
    base.mkdir(parents = True)
    (base / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    link, blob = _hub_snapshot(tmp_path, tmp_path / "base")
    assert C._hub_coords(link) == (
        "org/tiny",
        "transformer/diffusion_pytorch_model.safetensors",
        blob,
        "c" * 40,
    )
    copy = _copy_snapshot(tmp_path, tmp_path / "base")
    assert C._hub_coords(copy)[2] == "diffusion_pytorch_model.safetensors"


def test_the_freshness_check_of_a_gated_base_carries_the_loads_token(env, monkeypatch):
    import huggingface_hub

    from core.inference.diffusion_prequant import scoped_local_files_only

    seen = []

    def metadata(
        url,
        token = None,
        **_k,
    ):
        seen.append(token)
        return types.SimpleNamespace(etag = "e" * 64)

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    C._REMOTE_MEMO.clear()

    @scoped_local_files_only
    def load(**_kwargs):
        return C._remote_etag("org/gated", "transformer/diffusion_pytorch_model.safetensors")

    assert load(hf_token = "hf_request_token") == "e" * 64
    assert seen == ["hf_request_token"]


def _sharded_hub_snapshot(tmp_path: Path, n: int) -> list[str]:
    import hashlib

    repo_dir = tmp_path / "hub" / "models--org--sharded"
    (repo_dir / "blobs").mkdir(parents = True)
    snap = repo_dir / "snapshots" / ("c" * 40) / "transformer"
    snap.mkdir(parents = True)
    links = []
    for i in range(n):
        data = f"shard {i}".encode()
        blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        link = snap / f"diffusion_pytorch_model-{i + 1:05d}-of-{n:05d}.safetensors"
        os.symlink(blob, link)
        links.append(str(link))
    return links


@pytest.fixture
def blackholed_hub(monkeypatch):
    """Every Hub request times out via a mocked transport; yields the requested URLs."""
    import huggingface_hub

    seen: list = []
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    # huggingface_hub >= 1.33 with telemetry on fetches its agent registry on a cold cache.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    C._REMOTE_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    set_factory = getattr(huggingface_hub, "set_client_factory", None)
    if set_factory is None:  # older huggingface_hub (requests)
        import requests

        def metadata(url, **_k):
            seen.append(str(url))
            raise requests.exceptions.ReadTimeout("read timed out")

        monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
        yield seen
        return
    import httpx

    def handler(request):
        seen.append(str(request.url))
        raise httpx.ReadTimeout("timed out", request = request)

    set_factory(lambda: httpx.Client(transport = httpx.MockTransport(handler)))
    try:
        yield seen
    finally:
        from huggingface_hub.utils._http import default_client_factory
        set_factory(default_client_factory)


def test_a_blackholed_hub_costs_one_probe_per_process_not_one_per_shard_per_lookup(
    env, tmp_path, blackholed_hub
):
    """Unreachable but not flagged offline: 4 shards x 3 lookups used to cost 12 timeouts."""
    shards = _sharded_hub_snapshot(tmp_path, 4)
    for _ in range(3):
        assert C._hub_source_stale(shards) is False
    assert len(blackholed_hub) == 1
    assert "org/sharded" in blackholed_hub[0]


def test_an_http_answer_is_not_taken_for_an_unreachable_hub(env, tmp_path, monkeypatch):
    """A gated / 5xx answer is per repo: it must not silence other probes."""
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    C._REMOTE_MEMO.clear()
    calls = []

    def metadata(url, **_k):
        calls.append(url)
        raise HfHubHTTPError("503 Service Unavailable")

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    shards = _sharded_hub_snapshot(tmp_path, 2)
    assert C._hub_source_stale(shards) is False
    assert len(calls) == 2 and not C._UNREACHABLE


def test_the_hub_is_asked_again_once_the_unreachable_verdict_expires(env, tmp_path, monkeypatch):
    import huggingface_hub

    shards = _sharded_hub_snapshot(tmp_path, 2)
    blob = os.path.basename(os.path.realpath(shards[0]))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    C._REMOTE_MEMO.clear()
    calls = []

    def down(url, **_k):
        calls.append(url)
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", down)
    assert C._hub_source_stale(shards) is False
    assert len(calls) == 1
    C._UNREACHABLE[C._endpoint()] -= C._REMOTE_TTL_SECONDS + 1
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda url, **_k: calls.append(url) or types.SimpleNamespace(etag = "f" * 64),
    )
    assert C._hub_source_stale(shards) is True
    assert len(calls) == 2 and blob != "f" * 64


def test_an_entry_the_loader_rejected_is_a_miss_until_replaced(env, tmp_path, scheme):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    try:
        C.note_rejected(path)
        assert C.lookup(fam, scheme, base_repo = str(base)) is None
        _corrupt_in_place(path)
        assert C.save_runtime_quant(model, fam, base_repo = str(base)) == path  # the replacement
        assert C.lookup(fam, scheme, base_repo = str(base)) == path
    finally:
        C._REJECTED.clear()


def test_entries_written_minutes_ago_do_not_hold_the_cache_over_budget(env):
    """Several saves inside the grace window: the oldest goes once over budget; one resolved seconds ago stays."""
    env.mkdir(parents = True)
    older = _old_entry(env, "1" * 32, 1000, age = 300)
    newer = _old_entry(env, "2" * 32, 1000, age = 200)
    active = _old_entry(env, "3" * 32, 1000, age = 5)
    assert C.evict(root = env, max_bytes = 1500) == [older.name, newer.name]
    assert active.exists()


# ------------------------------------------------------------------------------------------------ review fixes
def test_the_flatten_writes_torchaos_bytes_without_cloning_the_quantized_payload(
    env, tmp_path, scheme
):
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    state = model.state_dict()
    flatten, _ = PS._torchao_helpers()
    # the previous whole-dict flatten: torchao over every non-root key at once, root keys appended under the prefix
    roots = PS._root_level_keys(state)
    whole, whole_meta = flatten({k: v for k, v in state.items() if k not in set(roots)})
    whole, whole_meta = dict(whole), dict(whole_meta)
    for key in roots:
        whole[f"{PS.UNSLOTH_ROOT_PREFIX}{key}"] = state[key]
    if roots:
        whole_meta[PS.UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in roots])
    flat, header = C._flatten(state)
    if any(type(v) is not torch.Tensor for v in state.values()):
        # quantized fields point at the live module's storage instead of a torchao clone
        assert any(isinstance(v, C._FlatRef) for v in flat.values())
    assert sorted(flat) == sorted(whole) and header == {**whole_meta}
    a, b = tmp_path / "ref.safetensors", tmp_path / "new.safetensors"
    C.write_streaming_safetensors(a, dict(whole), dict(whole_meta))
    C.write_streaming_safetensors(b, flat, header)
    assert a.read_bytes() == b.read_bytes()


def _copy_snapshot(
    tmp_path: Path,
    base_dir: Path,
    commit: str = "d" * 40,
) -> str:
    """A Hub cache laid out without symlinks (Windows without Developer Mode): the file copied into the snapshot."""
    shard = base_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    snap = tmp_path / "hub_copy" / "models--org--tiny" / "snapshots" / commit / "transformer"
    snap.mkdir(parents = True)
    target = snap / "diffusion_pytorch_model.safetensors"
    target.write_bytes(shard.read_bytes())
    return str(target)


def test_a_hub_cache_without_symlinks_is_judged_by_its_commit_not_its_filename(
    env, tmp_path, scheme, monkeypatch
):
    import huggingface_hub

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    commit = "d" * 40
    copy = _copy_snapshot(tmp_path, base, commit)
    monkeypatch.setattr(C, "source_shards", lambda *_a, **_k: [copy])
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    C._REMOTE_MEMO.clear()
    etags = {"main": "a" * 64, commit: "a" * 64}

    def metadata(url, **_k):
        rev = "main" if "/resolve/main/" in url else commit
        return types.SimpleNamespace(etag = etags[rev])

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = "org/tiny")
    assert path and C.lookup(fam, scheme, base_repo = "org/tiny") == path  # unchanged weights: a hit
    C._REMOTE_MEMO.clear()
    etags["main"] = "b" * 64  # main moved past the snapshot's commit
    assert C.lookup(fam, scheme, base_repo = "org/tiny") is None


def test_a_builder_rechecks_after_taking_the_key_lease(env, tmp_path, scheme, monkeypatch):
    """Two processes building one key: the second waits on the lease, then skips the first's published entry."""
    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    first = C.save_runtime_quant(model, fam, base_repo = str(base))
    assert first
    entry = Path(first)
    before = entry.stat().st_mtime_ns
    real = C._writer_lock
    took = []

    @C.contextmanager
    def lease(root, key):
        took.append(key)
        with real(root, key) as ok:
            yield ok

    monkeypatch.setattr(C, "_writer_lock", lease)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert took and entry.stat().st_mtime_ns == before


def test_eviction_skips_an_entry_whose_build_is_in_progress(env):
    old = _old_entry(env, "a" * 32, 4 << 20, age = 10_000)
    with C._writer_lock(env, old.name) as ok:
        assert ok and C._writer_active(env, old.name)
        assert C.evict(root = env, max_bytes = 0) == []
        assert old.is_dir()
    assert not C._writer_active(env, old.name)
    assert C.evict(root = env, max_bytes = 0) == [old.name]


def test_auto_never_promotes_a_lower_rung_through_the_local_cache():
    import inspect

    from core.inference import diffusion as D
    from core.inference import video as V

    planned = inspect.getsource(D.DiffusionBackend._pipeline_planned_denoiser_scheme)
    assert "rung != scheme and _is_local_cache_entry(source)" in planned
    retry = inspect.getsource(D.DiffusionBackend._auto_prequant_retry_scheme)
    assert "_is_local_cache_entry(source)" in retry
    video = inspect.getsource(V.VideoBackend.load_pipeline)
    save_at = video.index("_save_runtime_quant_to_cache(")
    assert "if transformer_quant_pinned is not None:" in video[save_at - 400 : save_at]


def test_is_local_cache_entry(env, tmp_path):
    from core.inference import diffusion as D

    inside = env / ("e" * 32) / C.FILE_NAME
    inside.parent.mkdir(parents = True)
    inside.write_bytes(b"x")
    assert D._is_local_cache_entry(types.SimpleNamespace(kind = "path", location = str(inside)))
    outside = tmp_path / "own.safetensors"
    outside.write_bytes(b"x")
    assert not D._is_local_cache_entry(types.SimpleNamespace(kind = "path", location = str(outside)))
    assert not D._is_local_cache_entry(types.SimpleNamespace(kind = "repo", location = "unsloth/x"))


def test_a_commit_pinned_hub_snapshot_without_refs_main_is_found_by_repo_id(
    env, tmp_path, scheme, monkeypatch
):
    """A commit-pinned base (no refs/main) must still be found by the next lookup by repo id."""
    import huggingface_hub

    from core.inference import diffusion_prequant as P

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    commit = "d" * 40
    _hub_snapshot(tmp_path, base, commit = commit)
    hub = tmp_path / "hub"
    repo_dir = hub / "models--org--tiny"
    assert not (repo_dir / "refs" / "main").exists()
    monkeypatch.setattr(C, "_hub_cache_dir", lambda: str(hub))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    fam = _fam(base)
    path = C.save_runtime_quant(
        model, fam, base_repo = "org/tiny", source_base = str(repo_dir / "snapshots" / commit)
    )
    assert path is not None
    token = P._LOCAL_FILES_ONLY.set(True)
    try:
        assert C.lookup(fam, scheme, base_repo = "org/tiny") == path
    finally:
        P._LOCAL_FILES_ONLY.reset(token)


def test_an_entry_the_planner_header_probe_drops_is_replaced_by_the_dense_build(
    env, tmp_path, scheme
):
    """Header corruption dropped by the planner probe (loader hook never fires) must still be replaced."""
    from core.inference.diffusion_prequant import usable_prequant_source

    base, model = _write_base(tmp_path)
    _quantize(model, scheme)
    fam = _fam(base)
    path = C.save_runtime_quant(model, fam, base_repo = str(base))
    good = Path(path).read_bytes()
    _corrupt_in_place(path)
    assert usable_prequant_source(fam, scheme, base_repo = str(base)) is None
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) == path
    assert Path(path).read_bytes() == good
    assert usable_prequant_source(fam, scheme, base_repo = str(base)).location == path


def test_the_video_save_fingerprints_the_repo_that_supplied_the_dense_weights():
    """Without a staged local dir the video key must come from fetch_base's shards (maybe the mirror)."""
    import inspect

    from core.inference import video as V

    video = inspect.getsource(V.VideoBackend.load_pipeline)
    call = video.split("_save_runtime_quant_to_cache(", 1)[1][:400]
    assert "source = _base_local_dir or fetch_base," in call


def _pipeline_read(D, pipe, fam, base, scheme, **kw):
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu")
    return D.DiffusionBackend._pipeline_quant_from_cache(
        pipe,
        fam,
        [("transformer", pipe.transformer)],
        target = target,
        transformer_quant = scheme,
        fast_accum = None,
        loras = kw.pop("loras", []),
        base = str(base),
        fetch_base = str(base),
        source_base = kw.pop("source_base", str(base)),
        hf_token = None,
        **kw,
    )


def _patch_scheme(monkeypatch, D, scheme):
    monkeypatch.setattr(D, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    monkeypatch.setattr(D, "native_quant_scheme", lambda *a, **k: None)


def test_a_returning_pipeline_load_reads_the_cache_bit_identically(
    env, tmp_path, scheme, monkeypatch
):
    """The pipeline-kind quantize loop wrote the cache but never read it: a second load re-quantized from bf16."""
    from core.inference import diffusion as D

    if not _SCHEME[1]:
        pytest.skip("torchao cannot quantize on this host's CPU")
    base, model = _write_base(tmp_path)
    fresh_dense = Tiny().to(torch.bfloat16)
    fresh_dense.load_state_dict(model.state_dict())
    fam = _fam(base)
    _patch_scheme(monkeypatch, D, scheme)
    miss = types.SimpleNamespace(transformer = fresh_dense)
    assert _pipeline_read(D, miss, fam, base, scheme) is None
    assert not D.transformer_is_quantised(fresh_dense)

    _quantize(model, scheme)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is not None
    pipe = types.SimpleNamespace(transformer = fresh_dense)
    assert _pipeline_read(D, pipe, fam, base, scheme) == scheme
    assert pipe.transformer is fresh_dense and D.transformer_is_quantised(fresh_dense)
    assert fresh_dense._unsloth_runtime_quant == scheme
    assert _flat_equal(model.state_dict(), fresh_dense.state_dict()) == []


def test_the_pipeline_read_finds_shards_under_the_source_base(env, tmp_path, scheme):
    """Saved with the dense shards read from a mirror / local snapshot: the lookup must search the same place."""
    mirror, model = _write_base(tmp_path, name = "mirror")
    upstream = tmp_path / "upstream"  # no shards here: only the mirror holds them
    fam = types.SimpleNamespace(name = FAMILY, transformer_class = "Tiny", base_repo = str(upstream))
    _quantize(model, scheme)
    assert C.save_runtime_quant(model, fam, base_repo = str(upstream), source_base = str(mirror))
    assert C.lookup(fam, scheme, base_repo = str(upstream)) is None
    assert C.lookup(fam, scheme, base_repo = str(upstream), source_base = str(mirror))


def test_a_pipeline_read_is_never_served_a_build_of_other_source_shards(env, tmp_path, scheme):
    """A build of the upstream shards must not be served to a load that read a different snapshot / mirror."""
    upstream, model = _write_base(tmp_path, name = "upstream")
    other, _ = _write_base(tmp_path, name = "other", seed = 1)
    fam = types.SimpleNamespace(name = FAMILY, transformer_class = "Tiny", base_repo = str(upstream))
    _quantize(model, scheme)
    assert C.save_runtime_quant(model, fam, base_repo = str(upstream))
    assert C.lookup(fam, scheme, base_repo = str(upstream))
    assert C.lookup(fam, scheme, base_repo = str(upstream), source_base = str(other)) is None


def test_a_pipeline_read_that_does_not_fit_leaves_the_module_dense(
    env, tmp_path, scheme, monkeypatch
):
    from core.inference import diffusion as D

    if not _SCHEME[1]:
        pytest.skip("torchao cannot quantize on this host's CPU")
    base, model = _write_base(tmp_path)
    fam = _fam(base)
    _quantize(model, scheme)
    assert C.save_runtime_quant(model, fam, base_repo = str(base))
    _patch_scheme(monkeypatch, D, scheme)

    class Other(Tiny):
        def __init__(self) -> None:
            super().__init__()
            self.extra = torch.nn.Linear(512, 512)

    other = Other().to(torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = other)
    assert _pipeline_read(D, pipe, fam, base, scheme) is None
    assert not D.transformer_is_quantised(other)
    dense = Tiny().to(torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = dense)
    assert _pipeline_read(D, pipe, fam, base, scheme, loras = [("a", 1.0)]) is None
    assert not D.transformer_is_quantised(dense)


def test_the_pipeline_quant_loop_reads_the_cache_before_quantizing():
    import inspect

    from core.inference import diffusion as D

    src = inspect.getsource(D.DiffusionBackend.load_pipeline)
    read = src.index("self._pipeline_quant_from_cache(")
    assert read < src.index("scheme = quantize_transformer(", read)


def test_a_rotated_int8_pipeline_read_matches_the_fresh_rotated_build(env, tmp_path, monkeypatch):
    """ConvRot: cached build has the offline rotation, the read installs the online half."""
    import core.inference.diffusion_transformer_quant as Q
    from core.inference import diffusion as D
    from core.inference.diffusion_convrot import is_rotated_linear
    from safetensors.torch import save_file

    family = "qwen-image-2.1"
    monkeypatch.setenv(Q.INT8_CONVROT_ENV, "1")

    class Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = torch.nn.Linear(512, 512, bias = False)
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(512, 512, bias = False)])

        def forward(self, x):
            return self.to_out[0](self.to_q(x))

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attn()

        def forward(self, x):
            return x + self.attn(x)

    class Rot(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(2)])

        def forward(self, x):
            for b in self.transformer_blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    built = Rot().to(torch.bfloat16)
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents = True)
    save_file(
        {k: v.contiguous() for k, v in built.state_dict().items()},
        str(base / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    dense = Rot().to(torch.bfloat16)
    dense.load_state_dict(built.state_dict())
    target = types.SimpleNamespace(dtype = torch.bfloat16, device = "cpu")
    monkeypatch.setattr(Q, "native_quant_scheme", lambda *a, **k: None)
    monkeypatch.setattr(Q, "select_transformer_quant_scheme", lambda *a, **k: "int8")
    try:
        engaged = Q.quantize_transformer(
            types.SimpleNamespace(transformer = built), target, mode = "int8", family = family
        )
    except Exception:  # noqa: BLE001
        engaged = None
    if engaged != "int8" or not D.transformer_is_quantised(built):
        pytest.skip("torchao cannot int8-quantize on this host's CPU")
    assert is_rotated_linear(built.transformer_blocks[0].attn.to_q)
    fam = types.SimpleNamespace(name = family, transformer_class = "Rot", base_repo = str(base))
    path = C.save_runtime_quant(built, fam, base_repo = str(base))
    if path is None:
        pytest.skip("this torchao cannot flatten int8 for the cache")

    _patch_scheme(monkeypatch, D, "int8")
    pipe = types.SimpleNamespace(transformer = dense)
    assert _pipeline_read(D, pipe, fam, base, "int8") == "int8"
    assert is_rotated_linear(dense.transformer_blocks[0].attn.to_q)
    assert _flat_equal(built.state_dict(), dense.state_dict()) == []
    x = torch.randn(4, 512, dtype = torch.bfloat16)
    with torch.no_grad():
        try:
            want = built(x)
        except Exception:  # noqa: BLE001 - no CPU int8 GEMM: the bytes check above still stands
            return
        got = dense(x)
    assert torch.equal(want, got)
