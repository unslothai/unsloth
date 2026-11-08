# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image / video / text-encoder pre-quant checkpoints resolve, download and load from .safetensors only."""

from __future__ import annotations

import json

import pytest

import core.inference.diffusion_prequant as pq
import core.inference.diffusion_te_prequant as tp
import core.inference.prequant_safetensors as ps
from core.inference import diffusion_families as df
from core.inference import video_families as vf
from core.inference.safetensors_only import (
    SAFETENSORS_ONLY_MESSAGE,
    is_pickle_weight,
    safetensors_only,
    safetensors_twin,
)

_PICKLES = (".pt", ".pth", ".bin", ".ckpt")


def _families():
    for mod in (df, vf):
        fams = mod._FAMILIES
        yield from (fams.values() if isinstance(fams, dict) else fams)


def _dit_rows():
    for fam in _families():
        schemes = {s for s, _r in getattr(fam, "prequant_repos", ()) or ()}
        schemes |= {s for _b, s, _r in getattr(fam, "prequant_variant_repos", ()) or ()}
        bases = [None] + sorted(
            {b for b, _s, _r in getattr(fam, "prequant_variant_repos", ()) or ()}
        )
        tasks = {None, "transformer_2"}
        tasks |= {e[1] for e in getattr(fam, "prequant_filenames", ()) or () if len(e) == 3}
        for scheme in sorted(schemes):
            for base in bases:
                for task in sorted(tasks, key = lambda t: t or ""):
                    yield fam, scheme, base, task


@pytest.fixture(autouse = True)
def _nvfp4_on(monkeypatch):
    # Cover the nvfp4 rows too.
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")


def test_helpers():
    assert all(is_pickle_weight(f"a{s}") for s in _PICKLES)
    assert not is_pickle_weight("a.safetensors")
    assert safetensors_only(["a.pt", "b.safetensors", None, "c.pth"]) == ["b.safetensors"]
    assert safetensors_twin("X-FP8-2.pt") == "X-FP8-2.safetensors"
    assert safetensors_twin("X.safetensors") == "X.safetensors"
    assert safetensors_twin("README.md") is None


def test_no_family_scheme_task_base_ever_resolves_a_pickle_name():
    seen = 0
    for fam, scheme, base, task in _dit_rows():
        src = pq.resolve_prequant_source(fam, scheme, base_repo = base, task = task)
        if src is None or src.kind != "repo":
            continue
        seen += 1
        names = pq.candidate_filenames_of(src)
        assert names, (fam.name, scheme, base, task)
        assert all(n.endswith(".safetensors") for n in names), (fam.name, scheme, base, task, names)
        assert all(n.endswith(".safetensors") for n in src.declared_filenames), src
    assert seen > 50


def test_no_text_encoder_source_ever_resolves_a_pickle_name():
    seen = 0
    for fam in _families():
        comps = set(tp.TE_PREQUANT_COMPONENTS) | {
            r[1] for r in getattr(fam, "te_prequant_repos", ()) or ()
        }
        for comp in comps:
            for scheme in ("fp8", "int8"):
                src = tp.resolve_te_prequant_source(fam, comp, scheme)
                if src is None or src.kind != "repo":
                    continue
                seen += 1
                names = tp.te_candidate_filenames(src)
                assert names and all(n.endswith(".safetensors") for n in names), (
                    fam.name,
                    comp,
                    names,
                )
    assert seen > 20


def test_family_tables_declare_safetensors_names_only():
    for fam in _families():
        for row in getattr(fam, "prequant_filenames", ()) or ():
            assert str(row[-1]).endswith(".safetensors"), (fam.name, row)


def test_declared_pickle_names_map_to_their_twin_and_the_env_switches_are_inert(monkeypatch):
    names = ["M-INT8-ConvRot.pt", "M-INT8.safetensors", "M-INT8.pt", "transformer_int8.pt"]
    for value in ("0", "1", None):
        if value is None:
            monkeypatch.delenv(pq.SAFETENSORS_TWINS_ENV, raising = False)
        else:
            monkeypatch.setenv(pq.SAFETENSORS_TWINS_ENV, value)
        assert pq.with_declared_safetensors_twins(names, ["M-INT8-ConvRot.pt"]) == [
            "M-INT8-ConvRot.safetensors",
            "M-INT8.safetensors",
        ]


def test_derived_names_are_safetensors_only():
    for scheme in ("fp8", "int8", "nvfp4", "mxfp8"):
        assert pq.derived_prequant_filenames("unsloth/Foo-FP8", scheme) == (
            f"Foo-{scheme.upper()}.safetensors",
        )
        assert pq.derived_prequant_filenames(
            "unsloth/Foo-FP8", scheme, component = "transformer_2"
        ) == (f"Foo-transformer_2-{scheme.upper()}.safetensors",)
    assert tp.te_prequant_repo_filenames("unsloth/LTX-2-FP8", "text_encoder", "fp8") == (
        "LTX-2-text_encoder-FP8.safetensors",
    )


def test_a_cached_pickle_is_never_promoted_or_kept(monkeypatch):
    monkeypatch.setattr(pq, "_hub_name_cached", lambda repo, name, root: f"/c/{name}")
    for value in ("1", "0"):
        monkeypatch.setenv(pq.PREFER_SAFETENSORS_ENV, value)
        out = pq.prefer_cached_pickle_twins(
            "unsloth/Foo-FP8", ["Foo-FP8.safetensors", "Foo-FP8.pt", "transformer_fp8.pt"]
        )
        assert out == ["Foo-FP8.safetensors"]


def test_readability_never_admits_a_pickle():
    for scheme in (None, "fp8", "int8", "nvfp4", "mxfp8"):
        for ext in _PICKLES:
            assert pq.restricted_prequant_load_supported(scheme, f"x{ext}") is False
    assert tp.te_candidate_is_readable("x.pt") is False


def test_a_pickle_is_refused_before_it_is_opened(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "M-FP8.pt"
    path.write_bytes(b"not opened")

    def _boom(*a, **k):
        raise AssertionError("torch.load must not run on the load path")

    monkeypatch.setattr(torch, "load", _boom)
    for scheme in ("fp8", "int8", "nvfp4"):
        with pytest.raises(ValueError, match = "safetensors only"):
            pq._load_prequant_checkpoint(str(path), scheme = scheme, map_location = "cpu")
    assert pq.local_prequant_scheme(str(path)) is None


def test_a_local_pickle_override_is_refused_by_name(tmp_path):
    path = tmp_path / "M-FP8.pt"
    path.write_bytes(b"x")
    fam = next(f for f in _families() if f.name == "flux.1")
    src = pq.resolve_prequant_source(fam, "fp8", path_override = str(path))
    assert src is not None and src.kind == "path"
    with pytest.raises(ValueError, match = "safetensors only"):
        pq._resolve_checkpoint_path(src, None, scheme = "fp8")
    te_src = tp.resolve_te_prequant_source(fam, "text_encoder_2", "fp8", path_override = str(path))
    with pytest.raises(ValueError, match = "safetensors only"):
        tp._resolve_checkpoint_path(te_src, None, cache_dir = str(tmp_path))


def test_the_request_refuses_a_pickle_prequant_path():
    from pydantic import ValidationError

    from models.inference import DiffusionLoadRequest

    with pytest.raises(ValidationError, match = "safetensors"):
        DiffusionLoadRequest(
            model_path = "black-forest-labs/FLUX.1-dev", transformer_prequant_path = "/x/M.pt"
        )
    ok = DiffusionLoadRequest(
        model_path = "black-forest-labs/FLUX.1-dev", transformer_prequant_path = " /x/M.safetensors "
    )
    assert ok.transformer_prequant_path == "/x/M.safetensors"


def test_unreadable_safetensors_never_falls_back_to_a_pickle(monkeypatch):
    src = pq.PrequantSource(
        kind = "repo",
        location = "unsloth/Foo-FP8",
        filename = "Foo-FP8.safetensors",
        fallback_filenames = ("Foo-FP8.pt",),
    )
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme, name = None: False)
    with pytest.raises(ValueError, match = "cannot be read here"):
        pq._resolve_checkpoint_path(src, None, scheme = "fp8")


def test_a_cached_comfy_file_is_still_read_where_torchao_cannot_rebuild(monkeypatch, tmp_path):
    import core.inference.diffusion_comfy_quant as cq

    src = pq.PrequantSource(
        kind = "repo", location = "unsloth/Foo-INT8", filename = "Foo-INT8.safetensors"
    )
    cached = tmp_path / "Foo-INT8.safetensors"
    cached.write_bytes(b"x")
    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, name = None: False
    )
    monkeypatch.setattr(pq, "_cached_in_root", lambda source, root, name = None: str(cached))
    monkeypatch.setattr(pq, "_first_mirrored", lambda *a, **k: None)
    monkeypatch.setattr(pq, "_download_checkpoint_name", lambda source, name, *a, **k: str(cached))
    monkeypatch.setattr(cq, "scan_comfy_quant", lambda path: object())
    # The ComfyUI-format loader has torchao-free backends, so a cached ComfyUI file is not refused.
    assert pq._resolve_checkpoint_path(src, None, scheme = "int8") == str(cached)
    monkeypatch.setattr(cq, "scan_comfy_quant", lambda path: None)
    with pytest.raises(ValueError, match = "cannot be read here"):
        pq._resolve_checkpoint_path(src, None, scheme = "int8")
    with pytest.raises(ValueError, match = "cannot be read here"):
        pq._resolve_checkpoint_path(src, None, scheme = "nvfp4")


def test_the_mirror_serves_the_safetensors_twin_never_the_pickle(tmp_path, monkeypatch):
    root = tmp_path / "mirror" / "unsloth" / "Foo-FP8"
    root.mkdir(parents = True)
    (root / "Foo-FP8.pt").write_bytes(b"x")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(tmp_path / "mirror"))
    assert pq.prequant_mirror_path("unsloth/Foo-FP8", "Foo-FP8.pt") is None
    (root / "Foo-FP8.safetensors").write_bytes(b"x")
    assert pq.prequant_mirror_path("unsloth/Foo-FP8", "Foo-FP8.pt").endswith("Foo-FP8.safetensors")
    monkeypatch.setenv(tp.TE_PREQUANT_MIRROR_ENV, str(tmp_path / "mirror"))
    assert tp.te_prequant_mirror_path("unsloth/Foo-FP8", "Foo-FP8.pt") is None


def _fake_hub(monkeypatch, hosted):
    import core.inference.prequant_header_probe as hp

    def _read(
        repo,
        name,
        token = None,
        **_,
    ):
        return (None, True) if name in hosted else (None, False)

    monkeypatch.setattr(hp, "read_hub_checkpoint_header", _read)
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda *a, **k: None)
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme, name = None: True)
    pq._HEADER_PROBE_MEMO.clear()


@pytest.mark.parametrize("family", ["wan2.2-t2v-a14b", "wan2.2-ti2v-5b"])
@pytest.mark.parametrize("scheme", ["fp8", "int8"])
def test_a_repo_hosting_only_pickles_plans_runtime_quantisation(monkeypatch, family, scheme):
    fam = next(f for f in _families() if f.name == family)
    _fake_hub(monkeypatch, hosted = set())
    assert pq.usable_prequant_source(fam, scheme, hf_token = f"tok-{family}-{scheme}-none") is None
    src = pq.resolve_prequant_source(fam, scheme)
    assert pq.hosted_prequant_refusal(src, scheme, hf_token = f"tok2-{family}-{scheme}") is not None


def test_a_hosted_safetensors_keeps_its_fast_path(monkeypatch):
    fam = next(f for f in _families() if f.name == "flux.1")
    src = pq.resolve_prequant_source(fam, "fp8")
    _fake_hub(monkeypatch, hosted = {src.filename})
    got = pq.usable_prequant_source(fam, "fp8", hf_token = "tok-flux-hosted")
    assert got is not None and got.filename == "FLUX.1-schnell-FP8.safetensors"


def test_native_reader_serves_below_the_torchao_floor(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    import core.inference.prequant_native as pn

    if not ps.safetensors_prequant_supported() or pn._torchao_api() is None:
        pytest.skip(reason = "needs torchao >= 0.16 flatten helpers to write the safetensors fixture")
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import Float8Tensor
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    from safetensors.torch import save_file

    k = 32
    qdata = (torch.randn(8, k) * 50).to(torch.float8_e4m3fn)
    scale = torch.rand(8, 1) / 100 + 1e-4
    weight = Float8Tensor(
        qdata,
        scale,
        block_size = [1, k],
        mm_config = Float8MMConfig(use_fast_accum = True),
        kernel_preference = KernelPreference.AUTO,
        dtype = torch.bfloat16,
    )
    flatten, _ = ps._torchao_helpers()
    flat, meta = flatten({"blk.f.weight": weight, "blk.norm.weight": torch.ones(k)})
    header = dict(meta)
    header[ps.UNSLOTH_FORMAT_KEY] = pq.PREQUANT_FORMAT
    header[ps.UNSLOTH_METADATA_KEY] = json.dumps({"scheme": "fp8"})
    path = tmp_path / "M-FP8.safetensors"
    save_file({n: t.contiguous() for n, t in flat.items()}, str(path), metadata = header)

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    assert ps.safetensors_scheme_supported("fp8") is True
    assert pq.restricted_prequant_load_supported("fp8", "M-FP8.safetensors") is True
    assert pq.restricted_prequant_load_supported("nvfp4", "M-NVFP4.safetensors") is False
    out = ps.load_prequant_safetensors(str(path))
    assert out["reader"] == "native"
    assert torch.equal(
        out["state_dict"]["blk.f.weight"].qdata.view(torch.uint8), qdata.view(torch.uint8)
    )
    monkeypatch.setenv(pn.NATIVE_REBUILD_ENV, "0")
    assert ps.safetensors_scheme_supported("fp8") is False
    with pytest.raises(RuntimeError, match = "torchao >= 0.16"):
        ps.load_prequant_safetensors(str(path))


def test_the_message_names_the_converter():
    assert "convert_prequant_to_safetensors.py" in SAFETENSORS_ONLY_MESSAGE
