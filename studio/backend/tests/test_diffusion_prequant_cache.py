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
    """``(scheme, really_quantized)``: a scheme this torch/torchao can quantize on the CPU AND flatten, else fp8 over
    a plain bf16 state dict (CPU-only CI: torchao refuses fp8 without a capable GPU and int8 below 0.18 does not
    flatten). The cache layer never looks inside the tensors, so the plain case still covers key, container, trust
    and lifecycle; the quantized case adds subclass round-trip bit-identity."""
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
    if _SCHEME is not None and _SCHEME[1]:
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

    def boom(
        path,
        tensors,
        metadata = None,
    ):
        with open(path, "wb") as fh:
            fh.write(b"partial")
        raise OSError("killed mid-write")

    monkeypatch.setattr(C, "write_streaming_safetensors", boom)
    assert C.save_runtime_quant(model, fam, base_repo = str(base)) is None
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    leftovers = [p for p in env.rglob("*") if p.is_file() and p.name != C.LAST_USED_NAME]
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
    """ru_maxrss of a child that writes 384 MB of tensors: the streaming writer adds far less than the payload,
    safetensors' save_file (which builds the whole file in memory) adds about the payload. CPU tensors are
    already on the host, so the comparison is against the writer's own extra buffers."""
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
