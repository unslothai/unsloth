# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _select_torchao_spec in install_python_stack.py.

torchao's C++ extensions are built against one exact torch release, so the
installer must pick the torchao version matching the torch installed in the
venv (otherwise the cpp kernels are skipped). This pins that mapping.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"
_EXTRAS_REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements" / "extras.txt"


def _load_module(monkeypatch):
    """(Re-)import install_python_stack and return it (mirrors test_pytorch_mirror)."""
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


@pytest.mark.parametrize(
    "torch_version, expected",
    [
        # torch 2.10 on CUDA <= 12 -> 0.16.0 (cpp built for 2.10.0, CUDA-12).
        ("2.10.0+cu128", "torchao==0.16.0"),
        ("2.10.0+cu126", "torchao==0.16.0"),
        ("2.10.0+rocm6.4", "torchao==0.16.0"),
        ("2.10.0+cpu", "torchao==0.16.0"),
        ("2.10.1", "torchao==0.16.0"),
        ("2.10.0", "torchao==0.16.0"),
        # torch 2.10 on CUDA >= 13: 0.16.0's CUDA-12 cpp fails (libcudart.so.12), use 0.17.0.
        ("2.10.0+cu130", "torchao==0.17.0"),
        ("2.10.0+cu140", "torchao==0.17.0"),
        ("2.10.0rc1", "torchao==0.16.0"),
        ("2.10.0.dev20250804+cu130", "torchao==0.17.0"),
        ("2.10.0.dev20250804+cu128", "torchao==0.16.0"),
        ("2.10rc1", "torchao==0.16.0"),
        ("2.11.0+cu130", "torchao==0.17.0"),
        ("2.11.0", "torchao==0.17.0"),
        ("2.11.1+cu126", "torchao==0.17.0"),
        # 2.12+ -> 0.18.0; 0.17.0's upstream table stops at 2.11.
        ("2.12.0", "torchao==0.18.0"),
        ("2.12.1+cu130", "torchao==0.18.0"),
        ("2.13.0+cu132", "torchao==0.18.0"),
        ("2.14.0+cu130", "torchao==0.18.0"),
        ("2.14.0+xpu", "torchao==0.18.0"),
        ("2.99.0", "torchao==0.18.0"),
        # The CUDA-13 branch belongs to 2.10 alone; it must not leak upward.
        ("2.12.0+cu126", "torchao==0.18.0"),
        ("2.12.0.dev20260801+cu132", "torchao==0.18.0"),
        ("2.9.0+cu128", "torchao==0.14.0"),
        ("2.9.1", "torchao==0.14.0"),
        ("2.8.0", "torchao==0.14.0"),
        ("2.4.0", "torchao==0.14.0"),
        (None, "torchao==0.14.0"),
        ("", "torchao==0.14.0"),
        ("garbage", "torchao==0.14.0"),
        ("2", "torchao==0.14.0"),
        ("3.0.0", "torchao==0.14.0"),
    ],
)
def test_select_torchao_spec(monkeypatch, torch_version, expected):
    mod = _load_module(monkeypatch)
    assert mod._select_torchao_spec(torch_version) == expected


def test_default_spec_matches_table(monkeypatch):
    """The default/floor stays the historical pin so older torch is unchanged."""
    mod = _load_module(monkeypatch)
    assert mod._TORCHAO_DEFAULT_SPEC == "torchao==0.14.0"
    assert mod._select_torchao_spec("2.9.0") == mod._TORCHAO_DEFAULT_SPEC


def test_matching_torchao_pin_does_not_need_force_reinstall(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_installed_distribution_version", lambda _name: "0.17.0")
    assert mod._exact_distribution_spec_is_installed("torchao==0.17.0")
    assert not mod._exact_distribution_spec_is_installed("torchao==0.16.0")


@pytest.mark.parametrize(
    "torch_version, leaf",
    [
        ("2.12.0+cu126", "cu126"),
        ("2.13.0+cu132", "cu132"),
        ("2.14.0+cu130", "cu130"),
        ("2.11.0+rocm7.2", "rocm7.2"),  # rocm DOES publish torchao, unlike torchcodec
        ("2.9.0+rocm6.4", "rocm6.4"),
        ("2.14.0+xpu", "xpu"),
        ("2.12.0+cpu", "cpu"),
        # Untagged torch is PyPI's own build and its counterpart is PyPI's own torchao.
        ("2.14.0", None),
        ("2.11.0", None),
        (None, None),
        ("", None),
        ("garbage", None),
    ],
)
def test_the_torchao_index_follows_the_resident_torch_build(monkeypatch, torch_version, leaf):
    """torchao publishes a wheel per accelerator and PyPI's default is the CUDA-12 one, so
    an unpinned install puts a CUDA-12 cpp beside a CUDA-13 or ROCm torch. That is the
    `libcudart.so.12: cannot open shared object file` the 2.10 CUDA-13 row already dodges by
    picking a build whose cpp gets skipped instead."""
    mod = _load_module(monkeypatch)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    got = mod._torch_accelerator_index_url(torch_version)
    assert got == (f"https://download.pytorch.org/whl/{leaf}" if leaf else None)


# torchao per index leaf, only leaves that do NOT cover every requested release.
_TORCHAO_INDEX_GAPS = {
    "cu118": ({m: f"0.{m}.0" for m in range(3, 12)}, range(5, 8)),
    "cu129": (
        {12: "0.12.0", 13: "0.13.0", 14: "0.14.1", 15: "0.15.0", 16: "0.16.0", 17: "0.17.0"},
        range(8, 14),
    ),
    "rocm7.0": ({16: "0.16.0"}, range(9, 11)),
}


def test_the_index_pin_starves_only_where_the_retry_covers_it(monkeypatch):
    """A pin that could not be served would fail an install, because this step is fatal.

    Four cells cannot be served, and none of them is predictable from a rule: cu118 stops at
    torchao 0.11.0, rocm7.0 carries 0.16.0 alone, and cu129 has 0.14.1 where the 2.9 row asks
    for 0.14.0 exactly -- a hole in the MIDDLE of its range, which no floor could describe.
    All four resolve from the default index, which is where they came from before this step
    pinned anything, so the retry makes them identical to today rather than broken. Recording
    them here means a fifth cannot appear unnoticed.
    """
    mod = _load_module(monkeypatch)
    starved = set()
    for leaf, (published, torch_minors) in _TORCHAO_INDEX_GAPS.items():
        for minor in torch_minors:
            version = f"2.{minor}.0+{leaf}"
            assert mod._torch_accelerator_index_url(version).endswith("/" + leaf)
            wanted = mod._select_torchao_spec(version).split("==", 1)[1]
            if wanted not in published.values():
                starved.add((leaf, minor, wanted))
    assert starved == {
        # cu118 tops out at torchao 0.11.0, so every torch it serves wants more than it has.
        ("cu118", 5, "0.14.0"),
        ("cu118", 6, "0.14.0"),
        ("cu118", 7, "0.14.0"),
        # cu129 publishes 0.14.1, not 0.14.0, and stops at 0.17.0.
        ("cu129", 8, "0.14.0"),
        ("cu129", 9, "0.14.0"),
        ("cu129", 12, "0.18.0"),
        ("cu129", 13, "0.18.0"),
        # rocm7.0 publishes 0.16.0 alone, which is what its torch 2.10 row already wants.
        ("rocm7.0", 9, "0.14.0"),
    }, sorted(starved)


def _torchao_installer_source():
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    body = source.split("def _install_torchao_for_torch(", 1)[1]
    return body.split("\ndef ", 1)[0]


def test_the_torchao_step_pins_the_index_and_retries_without_it():
    """cu129 serves torch to 2.13 but stops at torchao 0.17.0, and a leaf added upstream
    after this ships can lag a release, so the pin must not be able to fail an install.
    Unlike torchcodec the retry stays FATAL if it also fails: torchao is not optional."""
    body = _torchao_installer_source()
    assert "index = None if default_index else _torch_accelerator_index_url(torch_version)" in body
    assert '"--index-url", index, spec' in body
    assert "retrying from the default index" in body
    # The unpinned attempt is pip_install, not pip_install_try: still fatal on failure.
    retry = body.split("retrying from the default index", 1)[1]
    assert 'pip_install("Installing dependency overrides", *args, spec)' in retry
    assert "--index-url" not in retry
    # And the printed line redacts, since a mirror URL can carry credentials.
    assert "_strip_index_url_credentials(index)" in body


def test_the_fallback_is_never_conditioned_on_the_accelerator(monkeypatch):
    """A wrong-accelerator torchao costs its kernels, not its import, so the fallback stays
    unconditional -- guarding it on the CUDA major regressed CUDA-13/ROCm/XPU hosts from a
    slow torchao to none. torchao/__init__.py has wrapped the cpp load since 0.12.0."""
    body = _torchao_installer_source()
    fallback = body.split("retrying from the default index", 1)[1]
    assert 'pip_install("Installing dependency overrides", *args, spec)' in fallback
    # No branch between the failed pin and the retry. Comments carry the word; compare code.
    between = body.split("if pip_install_try(", 1)[1].split("retrying from the default index", 1)[0]
    code = [l for l in between.split("\n") if not l.strip().startswith("#")]
    assert not any(l.strip().startswith(("if ", "elif ")) for l in code), between
    mod = _load_module(monkeypatch)
    assert not hasattr(mod, "_default_index_torchao_can_load")


@pytest.mark.parametrize(
    "installed, spec, want_tag, expected",
    [
        ("0.18.0", "torchao==0.18.0", "<none>", False),
        ("0.18.0+cu130", "torchao==0.18.0", "<none>", True),
        ("0.17.0", "torchao==0.18.0", "<none>", True),
        (None, "torchao==0.18.0", "<none>", True),
        # Pinned: 0.18.0+cu126 satisfies ==0.18.0, so pip would keep the wrong build.
        ("0.18.0+cu130", "torchao==0.18.0", "cu130", False),
        ("0.18.0+cu126", "torchao==0.18.0", "cu130", True),
        ("0.18.0", "torchao==0.18.0", "cu130", True),
        ("0.18.0+rocm7.2", "torchao==0.18.0", "rocm7.2", False),
        ("0.17.0+cu130", "torchao==0.18.0", "cu130", True),
        # An opaque mirror proves nothing, so an untagged wheel is replaced.
        ("0.18.0", "torchao==0.18.0", None, True),
        ("0.18.0+cu130", "torchao==0.18.0", None, True),
    ],
)
def test_pin_needs_reinstall(monkeypatch, installed, spec, want_tag, expected):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_installed_distribution_version", lambda _name: installed)
    tag = "" if want_tag == "<none>" else want_tag
    assert mod._pin_needs_reinstall(spec, tag) is expected


def test_the_wanted_tag_follows_the_index_that_will_be_pinned(monkeypatch):
    """The provenance tag has to come from the leaf the pin resolves to, not from the
    resident torch. With UNSLOTH_TORCH_INDEX_FAMILY=cu130 over a +cu128 venv the pin goes to
    cu130 while the old comparison asked for cu128, so an 0.18.0+cu128 wheel looked correct,
    pip found the requirement satisfied and the cu130 build was never fetched."""
    mod = _load_module(monkeypatch)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu130")
    assert mod._torch_accelerator_index_url("2.13.0+cu128").endswith("/cu130")
    assert mod._torch_index_tag("2.13.0+cu128") == "cu130"

    monkeypatch.setattr(mod, "_installed_distribution_version", lambda _name: "0.18.0+cu128")
    assert mod._pin_needs_reinstall("torchao==0.18.0", mod._torch_index_tag("2.13.0+cu128"))
    monkeypatch.setattr(mod, "_installed_distribution_version", lambda _name: "0.18.0+cu130")
    assert not mod._pin_needs_reinstall("torchao==0.18.0", mod._torch_index_tag("2.13.0+cu128"))

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/whl/cu130")
    assert mod._torch_index_tag("2.13.0+cu128") is None

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")
    assert mod._torch_index_tag("2.13.0+cu128") == "cu128"


def test_every_torchao_call_site_asks_for_the_pinned_tag():
    """Passing the torch version rather than the pinned tag would reintroduce the drift the
    helper exists to remove, so no call site may spell it any other way."""
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    assert source.count("_pin_needs_reinstall(") == 3  # the def plus both call sites
    assert '_torch_index_tag(torch_version) if index else ""' in source
    assert '_torch_index_tag(_label_after) if _ao_index else ""' in source


def test_no_torchao_install_can_resolve_a_dependency():
    """Both call sites pass --no-deps, for the post-repair one: it runs right after step 13
    fixed the torch build. No torchao release declares a runtime torch dependency today, so
    this is hardening that must stay if one ever gains a pin."""
    body = _torchao_installer_source()
    assert 'args = ["--no-deps", "--no-cache-dir"]' in body
    # --force-reinstall must not be able to widen the install back out.
    for call in ("pip_install(", "pip_install_try("):
        for fragment in body.split(call)[1:]:
            assert "*args" in fragment.split(")")[0], fragment[:120]
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    resync = source.split("def _resync_torch_coupled_packages", 1)[1]
    ao = resync.split("_ao_index", 1)[1][:1200]
    assert "--no-deps" in ao


def test_torchao_is_re_selected_after_the_linux_torch_repair():
    """Step 4 chooses torchao from the torch present BEFORE step 13's repairs, which move
    torch across families and releases. The explicit XPU pin is the sharp case: its spec is
    torch>=2.6,<2.11.0, so it necessarily lands below the 2.11 floor torchao 0.18.0 needs,
    leaving 0.18.0 beside torch 2.10. Only the Windows flavor repair reaches
    _resync_torch_coupled_packages, so on Linux nothing re-selected it."""
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    step = source.split('_progress(_torch_step_label("final"))', 1)[1]
    step = step.split("# 13w.", 1)[0]
    assert '_torch_before_repair = str(_probe_installed_torch_version() or "")' in step
    assert "_install_torchao_for_torch(_torch_after_repair)" in step
    # Guarded on an actual move, so an install where nothing shifted pays no second resolve.
    assert "if _torch_after_repair and _torch_after_repair != _torch_before_repair:" in step
    assert '"torch>=2.6,<2.11.0",' in source


def test_windows_first_hop_uses_einx_wheel_without_shared_test_tree():
    requirements = _EXTRAS_REQUIREMENTS.read_text(encoding = "utf-8")
    assert 'einx<0.4.3; sys_platform == "win32"' in requirements
    # einx dropped 3.9 in 0.4.0, so the non-Windows side is split by interpreter.
    assert 'einx==0.4.3; sys_platform != "win32" and python_version >= "3.10"' in requirements
    assert 'einx==0.3.0; sys_platform != "win32" and python_version < "3.10"' in requirements


@pytest.mark.parametrize(
    ("rocm_windows_torch_installed", "installed_torch_is_windows_rocm"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("torchao_install_ok", [True, False])
def test_installs_pypi_torchao_on_windows_rocm(
    monkeypatch,
    tmp_path,
    rocm_windows_torch_installed,
    installed_torch_is_windows_rocm,
    torchao_install_ok,
):
    """Windows ROCm gets the torch-matched torchao from PyPI (download.pytorch.org's rocm leaves
    serve Linux only); the export worker loads it through unsloth/_torchao_nodist.py. A failed
    torchao install does not fail the install: only export needs it."""
    mod = _load_module(monkeypatch)
    pip_calls: list[list[str]] = []
    progress_labels: list[str] = []

    def _record_pip_install(*args, **kwargs):
        pip_calls.append([str(arg) for arg in args])
        return 0

    unstructured_plugin = tmp_path / "unstructured"
    github_plugin = tmp_path / "github"
    unstructured_plugin.mkdir()
    github_plugin.mkdir()

    subprocess_result = MagicMock()
    subprocess_result.returncode = 0
    subprocess_result.stdout = ""

    monkeypatch.setenv("SKIP_STUDIO_BASE", "1")
    monkeypatch.setattr(mod, "IS_WINDOWS", True)
    monkeypatch.setattr(mod, "IS_MACOS", False)
    monkeypatch.setattr(mod, "IS_MAC_ARM", False)
    monkeypatch.setattr(mod, "NO_TORCH", False)
    monkeypatch.setattr(mod, "_rocm_windows_torch_installed", rocm_windows_torch_installed)
    monkeypatch.setattr(
        mod, "_installed_torch_is_windows_rocm", lambda: installed_torch_is_windows_rocm
    )
    # require_present refuses when a managed distribution is absent (SKIP_STUDIO_BASE
    # guarantees that here); stub it so the test does not depend on the env.
    monkeypatch.setattr(mod, "_repair_damaged_core_payload", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_bootstrap_uv", lambda: False)
    monkeypatch.setattr(mod, "_repair_bad_anyio", lambda: None)
    monkeypatch.setattr(mod, "_repair_bad_accelerate", lambda: None)
    monkeypatch.setattr(mod, "_ensure_rocm_torch", lambda: None)
    monkeypatch.setattr(mod, "_ensure_cuda_torch", lambda: None)
    # A Windows ROCm box has no NVIDIA GPU; _expected_torch_flavor_tag reads this flag.
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: False)
    # The installed torch is ambient; pin it so the verdict does not depend on the host.
    monkeypatch.setattr(mod, "_RECORDED_TORCH_TAG", "")
    monkeypatch.setattr(
        mod, "_probe_torch_runtime", lambda *args, **kwargs: (True, True, "2.9.1+cpu", "", "")
    )
    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: None)

    def _fatal_pip_install(*args, **kwargs):
        # pip_install exits the installer on failure.
        _record_pip_install(*args, **kwargs)
        if not torchao_install_ok and any(str(a).startswith("torchao") for a in args):
            raise SystemExit(1)
        return 0

    monkeypatch.setattr(mod, "pip_install", _fatal_pip_install)
    monkeypatch.setattr(
        mod,
        "pip_install_try",
        lambda *a, **k: (
            _record_pip_install(*a, **k),
            torchao_install_ok or not any(str(x).startswith("torchao") for x in a),
        )[1],
    )
    monkeypatch.setattr(mod, "_progress", lambda label: progress_labels.append(label))
    monkeypatch.setattr(mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin)
    monkeypatch.setattr(mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: subprocess_result)

    # Checked first so a regression names its cause instead of `assert 1 == 0` below.
    assert mod._expected_torch_flavor_tag() == "", (
        "no CUDA expectation may exist on a Windows ROCm host: a non-empty tag means "
        "the Windows flavor invariant will demand a cu* build, not find one, and fail "
        "the install long after the torchao branch this test is about"
    )

    assert mod.install_python_stack() == 0

    torchao_calls = [c for c in pip_calls if any(a.startswith("torchao") for a in c)]
    assert torchao_calls
    assert all("--index-url" not in c for c in torchao_calls)
    assert "dependency overrides (Windows ROCm)" in progress_labels


# Windows ROCm torchao export: real torchao via unsloth/_torchao_nodist.py, the stub as fallback.

import types

import core._torchao_stub as _stub

_BACKEND = Path(__file__).resolve().parents[1]
_EXPORT_HELPERS = ("_torchao_export_supported", "_torchao_runtime_unavailable", "_is_torchao_alias")


def _export_helpers():
    """Exec the gate helpers alone, avoiding export.py's heavy import chain."""
    import ast

    src = (_BACKEND / "core" / "export" / "export.py").read_text(encoding = "utf-8")
    ns: dict = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in _EXPORT_HELPERS:
            exec(ast.get_source_segment(src, node), ns)
    return ns


def _fake_normalize_torchao(save_method):
    if not isinstance(save_method, str):
        return None
    key = save_method.lower().strip().replace("-", "_").replace(" ", "_")
    ok = key in {"torchao_fp8", "torchao_int8", "portable_fp8", "portable_int8"}
    return ("fp8", "torchao-fp8") if ok else None


def _install_fake_unsloth_save(monkeypatch, *, has_method):
    unsloth = types.ModuleType("unsloth")
    save = types.ModuleType("unsloth.save")
    if has_method:
        save._normalize_torchao_method = _fake_normalize_torchao
    unsloth.save = save
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", save)


def _set_torchao(monkeypatch, state):
    if state == "absent":
        monkeypatch.delitem(sys.modules, "torchao", raising = False)
    elif state == "stub":
        monkeypatch.setitem(sys.modules, "torchao", _stub._make_mod_stub("torchao"))
    else:
        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))


@pytest.mark.parametrize(
    ("win_rocm", "torchao", "has_method", "expected"),
    [
        (True, "stub", True, False),
        (True, "absent", True, False),
        (True, "real", True, True),
        (False, "real", True, True),
        (False, "absent", True, True),
        (False, "real", False, False),
    ],
)
def test_torchao_export_gate(monkeypatch, win_rocm, torchao, has_method, expected):
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: win_rocm)
    _set_torchao(monkeypatch, torchao)
    _install_fake_unsloth_save(monkeypatch, has_method = has_method)
    ns = _export_helpers()
    ns["sys"] = sys
    assert ns["_torchao_export_supported"]() is expected


def test_is_torchao_alias_recognizes_all_forms(monkeypatch):
    _install_fake_unsloth_save(monkeypatch, has_method = True)
    fn = _export_helpers()["_is_torchao_alias"]
    for alias in ("torchao_fp8", "portable_int8", "portable-fp8", "Portable FP8"):
        assert fn(alias) is True
    for alias in ("fp8", "nvfp4", "w8a8", "", None):
        assert fn(alias) is False


def _load_export_module_no_torch(monkeypatch):
    """Import core.export.export with torch/unsloth blocked, so the guard runs without a GPU."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in {"torch", "unsloth"} and top not in sys.modules:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    for m in [k for k in list(sys.modules) if k.split(".")[0] in {"torch", "unsloth"}]:
        monkeypatch.delitem(sys.modules, m, raising = False)
    monkeypatch.delitem(sys.modules, "core.export.export", raising = False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)
    return importlib.import_module("core.export.export")


def _bare_backend(mod):
    be = mod.ExportBackend.__new__(mod.ExportBackend)
    be.current_model = object()
    be.current_tokenizer = object()
    be._audio_type = None
    be.is_peft = True
    return be


@pytest.mark.parametrize("alias", ["torchao_fp8", "portable_fp8"])
def test_torchao_export_rejected_early_when_stubbed(monkeypatch, alias):
    mod = _load_export_module_no_torch(monkeypatch)
    monkeypatch.setattr(mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    _set_torchao(monkeypatch, "stub")
    _install_fake_unsloth_save(monkeypatch, has_method = True)

    ok, message, out = _bare_backend(mod).export_merged_model("/tmp/x", compressed_method = alias)
    assert ok is False and out is None
    assert "Windows ROCm" in message and "torchao" in message.lower()


def test_torchao_export_not_rejected_with_real_torchao_on_windows_rocm(monkeypatch):
    mod = _load_export_module_no_torch(monkeypatch)
    monkeypatch.setattr(mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    _set_torchao(monkeypatch, "real")
    _install_fake_unsloth_save(monkeypatch, has_method = True)

    ok, message, _ = _bare_backend(mod).export_merged_model(
        "/tmp/x", compressed_method = "torchao_int8"
    )
    assert ok is False and "Windows ROCm" not in message


@pytest.mark.parametrize(
    ("platform", "is_rocm", "loadable", "expected"),
    [
        ("win32", True, False, False),
        ("win32", True, True, True),
        ("win32", False, False, True),
        ("linux", True, False, True),
    ],
)
def test_export_capability_torchao_flag(monkeypatch, platform, is_rocm, loadable, expected):
    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(hw, "IS_ROCM", is_rocm)
    monkeypatch.setattr(_stub, "torchao_export_loadable", lambda: loadable)
    assert hw.export_capability()["torchao_export_supported"] is expected


def test_torchao_export_loadable_ignores_the_stub(monkeypatch):
    # The main process has already stubbed torchao; only an installed torchao counts.
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    _set_torchao(monkeypatch, "stub")
    real = importlib.machinery.PathFinder.find_spec
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        classmethod(lambda cls, name, *a, **k: None if name == "torchao" else real(name, *a, **k)),
    )
    assert _stub.torchao_export_loadable() is False


@pytest.mark.parametrize(
    ("installed", "expect"),
    [("0.14.0", False), ("0.15.0", True), ("0.17.0", True), ("0.18.0+rocm", True)],
)
def test_torchao_export_loadable_needs_transformers_minimum(
    monkeypatch, tmp_path, installed, expect
):
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    (tmp_path / "unsloth").mkdir()
    (tmp_path / "unsloth" / "_torchao_nodist.py").write_text("")
    specs = {
        "torchao": importlib.machinery.ModuleSpec("torchao", None),
        "unsloth": importlib.machinery.ModuleSpec("unsloth", None, is_package = True),
    }
    specs["unsloth"].submodule_search_locations = [str(tmp_path / "unsloth")]
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        classmethod(lambda cls, name, *a, **k: specs.get(name)),
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: specs.get(name))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    assert _stub.torchao_export_loadable() is expect


@pytest.mark.parametrize(("fix_result", "expect_real"), [(True, True), (False, False)])
def test_real_or_stub(monkeypatch, fix_result, expect_real):
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    monkeypatch.setattr(_stub, "torchao_export_loadable", lambda: True)
    monkeypatch.delitem(sys.modules, "torchao", raising = False)
    calls = []

    def fix():
        calls.append(1)
        if fix_result:
            sys.modules["torchao"] = types.ModuleType("torchao")
        return fix_result

    monkeypatch.setattr(
        _stub,
        "_load_torchao_nodist",
        lambda: types.SimpleNamespace(fix_torchao_without_torch_distributed = fix),
    )
    stubbed = []
    monkeypatch.setattr(_stub, "install_torchao_windows_rocm_stub", lambda: stubbed.append(1))
    try:
        assert _stub.install_torchao_windows_rocm_real_or_stub() is expect_real
        assert calls == [1]
        assert bool(stubbed) is (not expect_real)
    finally:
        sys.modules.pop("torchao", None)


@pytest.mark.parametrize("consumer_loaded", [False, True])
def test_real_or_stub_replaces_a_stub_inherited_from_run_py(monkeypatch, consumer_loaded):
    """spawn re-runs run.py as __mp_main__, which stubs torchao before the export worker starts."""
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    monkeypatch.setattr(_stub, "torchao_export_loadable", lambda: True)
    # Simulated "already imported" consumer: json is always loaded; none of the real ones is required.
    monkeypatch.setattr(_stub, "_STUB_CONSUMERS", ("json",) if consumer_loaded else ())
    saved = {n: m for n, m in sys.modules.items() if n == "torchao" or n.startswith("torchao.")}
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    calls = []

    def fix():
        calls.append("torchao" in sys.modules)
        sys.modules["torchao"] = types.ModuleType("torchao")
        return True

    monkeypatch.setattr(
        _stub,
        "_load_torchao_nodist",
        lambda: types.SimpleNamespace(fix_torchao_without_torch_distributed = fix),
    )
    try:
        _stub.install_torchao_windows_rocm_stub()
        assert _stub.is_stubbed("torchao")
        result = _stub.install_torchao_windows_rocm_real_or_stub()
        if consumer_loaded:
            assert result is False and calls == [] and _stub.is_stubbed("torchao")
        else:
            assert result is True and calls == [False] and not _stub.is_stubbed("torchao")
            assert not any(n.startswith("torchao.") for n in sys.modules)
    finally:
        for name in [n for n in sys.modules if n == "torchao" or n.startswith("torchao.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_real_or_stub_keeps_the_stub_when_torchao_is_not_loadable(monkeypatch):
    """torch <= 2.9 pairs with torchao 0.14, which transformers 5 rejects: never load it."""
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: True)
    monkeypatch.setattr(_stub, "torchao_export_loadable", lambda: False)
    monkeypatch.delitem(sys.modules, "torchao", raising = False)
    monkeypatch.setattr(
        _stub, "_load_torchao_nodist", lambda: pytest.fail("must not load the shim")
    )
    stubbed = []
    monkeypatch.setattr(_stub, "install_torchao_windows_rocm_stub", lambda: stubbed.append(1))
    assert _stub.install_torchao_windows_rocm_real_or_stub() is False
    assert stubbed == [1]


def test_shim_found_through_an_editable_finder(monkeypatch, tmp_path):
    """setuptools editable installs can expose unsloth only through a meta-path finder."""
    (tmp_path / "unsloth").mkdir()
    (tmp_path / "unsloth" / "_torchao_nodist.py").write_text(
        "def fix_torchao_without_torch_distributed():\n    return 'ok'\n"
    )

    class EditableFinder:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name != "unsloth":
                return None
            spec = importlib.machinery.ModuleSpec("unsloth", None, is_package = True)
            spec.submodule_search_locations = [str(tmp_path / "unsloth")]
            return spec

    monkeypatch.delitem(sys.modules, "unsloth", raising = False)
    real = importlib.machinery.PathFinder.find_spec
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        classmethod(lambda cls, name, *a, **k: None if name == "unsloth" else real(name, *a, **k)),
    )
    monkeypatch.setattr(sys, "meta_path", [EditableFinder(), *sys.meta_path])
    module = _stub._load_torchao_nodist()
    assert module is not None and module.fix_torchao_without_torch_distributed() == "ok"


def test_real_or_stub_noop_off_windows_rocm(monkeypatch):
    monkeypatch.setattr(_stub, "_is_windows_rocm", lambda: False)
    monkeypatch.setattr(
        _stub, "_load_torchao_nodist", lambda: pytest.fail("must not load the shim")
    )
    assert _stub.install_torchao_windows_rocm_real_or_stub() is False


def test_load_torchao_nodist_reads_unsloths_file():
    module = _stub._load_torchao_nodist()
    if importlib.util.find_spec("unsloth") is None:
        assert module is None
    else:
        assert callable(module.fix_torchao_without_torch_distributed)
