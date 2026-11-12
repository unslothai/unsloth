# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The direct SageAttention hub loader: variant names, arch and digest checks, import, and the order
``_load_sage_hub_kernel`` tries it in. CPU only, no network."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import core.inference.diffusion_sage_hub as hub

BUILD = {"torch": (2, 11), "cuda": (13, 0), "cxx11": True}


def test_variants_list_both_naming_schemes_and_older_cuda_minors():
    v = hub.candidate_variants({"torch": (2, 13), "cuda": (13, 2), "cxx11": True}, "x86_64-linux")
    assert v[:2] == ["torch213-cxx11-cu132-x86_64-linux", "torch213-cu132-x86_64-linux"]
    assert "torch213-cxx11-cu130-x86_64-linux" in v
    assert not any("cu12" in x for x in v), "a CUDA 13 torch cannot load a CUDA 12 build"
    assert hub.candidate_variants(None, "x86_64-linux") == []
    assert hub.candidate_variants(BUILD, None) == []


@pytest.mark.parametrize(
    "archs, capability, ok",
    [
        (["8.0", "8.9", "9.0a"], (8, 0), True),  # A100
        (["8.0", "8.9", "9.0a"], (8, 6), True),  # an 8.0 build runs on later 8.x
        (["8.0", "8.9", "9.0a"], (8, 9), True),  # L4
        (["8.0", "8.9", "9.0a"], (9, 0), True),  # H100 (arch-specific 9.0a)
        (["8.0", "8.9", "9.0a"], (10, 0), False),
        (["10.0a", "8.0", "8.9", "9.0a"], (10, 0), True),  # B200
        (["10.0a", "8.0", "8.9", "9.0a"], (12, 0), False),  # RTX PRO 6000 / G4
        (["10.0a", "8.0", "8.9", "9.0a"], (7, 5), False),  # T4
        (None, (7, 5), True),  # unknown: the accuracy probe decides
        (["8.0"], None, True),
    ],
)
def test_arch_compatibility(archs, capability, ok):
    assert hub._arch_ok(archs, capability) is ok


def _write_build(
    root: Path,
    variant: str,
    archs = ("8.0", "8.9", "9.0a"),
    digest = True,
) -> Path:
    """A fake hub build: a package whose __init__ uses a relative import, like the real ones."""
    d = root / "build" / variant
    d.mkdir(parents = True)
    files = {
        "__init__.py": "from .core import sageattn\n",
        "core.py": "def sageattn(q, k, v, tensor_layout='HND', **kw):\n    return ('fake', tensor_layout)\n",
    }
    for name, text in files.items():
        (d / name).write_text(text)
    meta = {"version": 2, "python-depends": [], "backend": {"type": "cuda", "archs": list(archs)}}
    if digest:
        meta["digest"] = {
            "algorithm": "sha256",
            "files": {
                n: base64.b64encode(hashlib.sha256(t.encode()).digest()).decode()
                for n, t in files.items()
            },
        }
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Per (repo_type, revision) build roots; records every download pattern."""
    roots: dict = {}
    downloads: list = []
    variant = hub.candidate_variants(BUILD, "x86_64-linux")[0]

    def _list(repo, repo_type, revision):
        root = roots.get((repo_type, revision))
        return set() if root is None else {p.name for p in (root / "build").iterdir()}

    def _download(repo, repo_type, revision, var, pattern, offline):
        downloads.append((repo_type, revision, var, pattern))
        root = roots.get((repo_type, revision))
        path = None if root is None else root / "build" / var
        return path if path is not None and path.is_dir() else None

    monkeypatch.setattr(hub, "_torch_build", lambda: BUILD)
    monkeypatch.setattr(hub, "_platform_tag", lambda: "x86_64-linux")
    monkeypatch.setattr(hub, "_kernel_repo_type", lambda: "kernel")
    monkeypatch.setattr(hub, "_list_variants", _list)
    monkeypatch.setattr(hub, "_download", _download)
    yield types.SimpleNamespace(roots = roots, downloads = downloads, variant = variant, tmp = tmp_path)
    for name in [m for m in sys.modules if m.startswith(hub._MODULE_PREFIX)]:
        sys.modules.pop(name, None)


def test_loads_the_kernel_repo_build_and_imports_its_relative_modules(fake_hub):
    root = fake_hub.tmp / "kernel_v2"
    _write_build(root, fake_hub.variant)
    fake_hub.roots[("kernel", "v2")] = root
    fn, why = hub.load_sageattn((2, 3), capability = (8, 9))
    assert why == "" and fn(None, None, None, tensor_layout = "NHD") == ("fake", "NHD")
    assert fn.__module__.startswith(hub._MODULE_PREFIX)


def test_falls_back_to_the_model_mirror_when_the_kernel_repo_has_no_build(fake_hub):
    root = fake_hub.tmp / "model_v2"
    _write_build(root, fake_hub.variant)
    fake_hub.roots[(None, "v2")] = root
    fn, why = hub.load_sageattn((2, 3), capability = (8, 0))
    assert callable(fn) and why == ""
    tried = [(t, r) for t, r, _v, p in fake_hub.downloads if p == "metadata.json"]
    assert tried == [(None, "v2")]


def test_an_unsupported_card_never_downloads_the_build(fake_hub):
    root = fake_hub.tmp / "kernel_v2"
    _write_build(root, fake_hub.variant)
    fake_hub.roots[("kernel", "v2")] = root
    fn, why = hub.load_sageattn((2,), capability = (12, 0))
    assert fn is None and "not sm120" in why
    assert all(p == "metadata.json" for _t, _r, _v, p in fake_hub.downloads)


def test_a_tampered_file_is_refused(fake_hub):
    root = fake_hub.tmp / "kernel_v2"
    d = _write_build(root, fake_hub.variant)
    (d / "core.py").write_text("def sageattn(*a, **k):\n    raise SystemExit('tampered')\n")
    fake_hub.roots[("kernel", "v2")] = root
    fn, why = hub.load_sageattn((2,), capability = (8, 9))
    assert fn is None and "core.py does not match its sha256 digest" in why


def test_python_dependencies_are_refused(fake_hub):
    root = fake_hub.tmp / "kernel_v3"
    d = _write_build(root, fake_hub.variant)
    meta = json.loads((d / "metadata.json").read_text())
    meta["python-depends"] = ["einops"]
    (d / "metadata.json").write_text(json.dumps(meta))
    fake_hub.roots[("kernel", "v3")] = root
    fn, why = hub.load_sageattn((3,), capability = (8, 9))
    assert fn is None and "einops" in why


def test_every_failure_is_reported_and_nothing_raises(fake_hub, monkeypatch):
    fn, why = hub.load_sageattn((2, 3), capability = (8, 9))
    assert fn is None
    for where in ("kernel repo v2", "kernel repo v3", "model repo v2", "model repo v3"):
        assert where in why
    monkeypatch.setattr(hub, "_torch_build", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    fn, why = hub.load_sageattn((2,))
    assert fn is None and "boom" in why


def test_offline_reads_the_cached_build_only(fake_hub, monkeypatch):
    root = fake_hub.tmp / "kernel_v2"
    _write_build(root, fake_hub.variant)
    fake_hub.roots[("kernel", "v2")] = root
    monkeypatch.setattr(hub, "_list_variants", lambda *a: None)  # the Hub cannot be asked
    offline = []
    real = hub._download

    def _download(*a):
        offline.append(a[-1])
        return real(*a)

    monkeypatch.setattr(hub, "_download", _download)
    fn, why = hub.load_sageattn((2,), capability = (8, 9))
    assert callable(fn) and why == "" and offline and all(offline)


def test_rocm_or_cpu_torch_declines_without_network(monkeypatch):
    monkeypatch.setattr(hub, "_torch_build", lambda: None)
    monkeypatch.setattr(hub, "_list_variants", lambda *a: pytest.fail("must not reach the Hub"))
    fn, why = hub.load_sageattn((2, 3))
    assert fn is None and "CUDA torch" in why


# The order inside diffusion_attention._load_sage_hub_kernel.
torch = pytest.importorskip("torch")
dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
_HUB = getattr(dispatch.AttentionBackendName, "SAGE_HUB", None)
needs_sage_hub = pytest.mark.skipif(_HUB is None, reason = "this diffusers has no sage_hub backend")

import core.inference.diffusion_attention as att  # noqa: E402


@pytest.fixture
def hub_slot(monkeypatch):
    cfg = dispatch._HUB_KERNELS_REGISTRY[_HUB]
    saved = cfg.kernel_fn
    cfg.kernel_fn = None
    monkeypatch.setattr(att, "_SAGE_HUB_FAILED", [])
    yield cfg
    cfg.kernel_fn = saved


def _sage(*a, **k):
    return "direct"


@needs_sage_hub
def test_the_direct_build_wins_and_kernels_get_kernel_is_never_asked(monkeypatch, hub_slot):
    monkeypatch.setattr(att, "_load_sage_hub_direct", lambda: (_sage, ""))
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        types.SimpleNamespace(get_kernel = lambda *a, **k: pytest.fail("get_kernel")),
    )
    assert att._load_sage_hub_kernel() == (_sage, "")
    assert hub_slot.kernel_fn is _sage


@needs_sage_hub
def test_kernels_get_kernel_is_the_fallback_and_both_reasons_are_kept(monkeypatch, hub_slot):
    monkeypatch.setattr(att, "_load_sage_hub_direct", lambda: (None, "kernel repo v2: no build"))
    asked = []

    def _get_kernel(
        repo,
        version = None,
        **_kw,
    ):
        asked.append(version)
        raise FileNotFoundError(f"no variant v{version}")

    monkeypatch.setitem(sys.modules, "kernels", types.SimpleNamespace(get_kernel = _get_kernel))
    fn, why = att._load_sage_hub_kernel()
    assert fn is None and "direct: kernel repo v2: no build" in why and "no variant v2" in why
    assert asked[:2] == [2, 1]


@needs_sage_hub
def test_no_download_while_kernels_is_not_importable(monkeypatch, hub_slot):
    # diffusers' sage_hub backend refuses to engage without kernels, so fetching the build would be wasted.
    monkeypatch.setattr(att, "_load_sage_hub_direct", lambda: pytest.fail("must not download"))
    monkeypatch.setitem(sys.modules, "kernels", None)
    fn, why = att._load_sage_hub_kernel()
    assert fn is None and "kernels package is not importable" in why
    assert att._SAGE_HUB_FAILED == []
