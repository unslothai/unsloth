# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _select_torchao_spec in install_python_stack.py.

torchao's C++ extensions are built against one exact torch release, so the
installer must pick the torchao version matching the torch installed in the
venv (otherwise the cpp kernels are skipped). This pins that mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# install_python_stack.py lives at repo_root/studio/install_python_stack.py
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
        # torch 2.10 on CUDA <= 12 -> 0.16.0 (its cpp is built for torch 2.10.0 and
        # loads against the CUDA-12 PyPI wheel). Independent of patch level.
        ("2.10.0+cu128", "torchao==0.16.0"),
        ("2.10.0+cu126", "torchao==0.16.0"),
        ("2.10.0+rocm6.4", "torchao==0.16.0"),
        ("2.10.0+cpu", "torchao==0.16.0"),
        ("2.10.1", "torchao==0.16.0"),
        ("2.10.0", "torchao==0.16.0"),
        # torch 2.10 on CUDA >= 13 (Blackwell / cu130): 0.16.0's CUDA-12 cpp can't
        # load against a CUDA-13 torch (libcudart.so.12 error), so use 0.17.0.
        ("2.10.0+cu130", "torchao==0.17.0"),
        ("2.10.0+cu140", "torchao==0.17.0"),
        # Pre-release / dev / rc builds: the minor is cleaned of non-digits; the
        # CUDA tag still decides 0.16.0 vs 0.17.0.
        ("2.10.0rc1", "torchao==0.16.0"),
        ("2.10.0.dev20250804+cu130", "torchao==0.17.0"),
        ("2.10.0.dev20250804+cu128", "torchao==0.16.0"),
        ("2.10rc1", "torchao==0.16.0"),
        # 2.11 -> 0.17.0, whose cpp is built for it.
        ("2.11.0+cu130", "torchao==0.17.0"),
        ("2.11.0", "torchao==0.17.0"),
        ("2.11.1+cu126", "torchao==0.17.0"),
        # 2.12+ -> 0.18.0, whose release CI is pinned to 2.13. 0.17.0's upstream table stops
        # at 2.11, so leaving this range there ran it outside its declared window.
        ("2.12.0", "torchao==0.18.0"),
        ("2.12.1+cu130", "torchao==0.18.0"),
        ("2.13.0+cu132", "torchao==0.18.0"),
        ("2.14.0+cu130", "torchao==0.18.0"),
        ("2.14.0+xpu", "torchao==0.18.0"),
        ("2.99.0", "torchao==0.18.0"),
        # The CUDA-13 branch belongs to 2.10 alone; it must not leak upward.
        ("2.12.0+cu126", "torchao==0.18.0"),
        ("2.12.0.dev20260801+cu132", "torchao==0.18.0"),
        # torch <=2.9 keeps today's pin (already a correct match for 2.9.0).
        ("2.9.0+cu128", "torchao==0.14.0"),
        ("2.9.1", "torchao==0.14.0"),
        ("2.8.0", "torchao==0.14.0"),
        ("2.4.0", "torchao==0.14.0"),
        # Unparseable / missing / non-2.x major -> conservative default.
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


# torchao per leaf, from the live listings. Only leaves that do NOT cover every release this
# selector can ask for; everything absent serves its whole range.
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
    assert "index = _torch_accelerator_index_url(torch_version)" in body
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
        # No index pinned: the wheel comes from the default index, which stamps no tag.
        ("0.18.0", "torchao==0.18.0", "<none>", False),
        # A tagged wheel on the unpinned path came from elsewhere, so it is replaced once,
        # then settles: what lands is bare.
        ("0.18.0+cu130", "torchao==0.18.0", "<none>", True),
        ("0.17.0", "torchao==0.18.0", "<none>", True),
        (None, "torchao==0.18.0", "<none>", True),
        # Pinned: the release can be right while the BUILD is wrong. 0.18.0+cu126 satisfies
        # ==0.18.0, so pip fetches nothing and the wrong build stays.
        ("0.18.0+cu130", "torchao==0.18.0", "cu130", False),
        ("0.18.0+cu126", "torchao==0.18.0", "cu130", True),
        ("0.18.0", "torchao==0.18.0", "cu130", True),
        ("0.18.0+rocm7.2", "torchao==0.18.0", "rocm7.2", False),
        ("0.17.0+cu130", "torchao==0.18.0", "cu130", True),
        # An opaque mirror proves nothing, so it is replaced: otherwise an untagged wheel
        # satisfies the pin and the mirror is never contacted.
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

    # An explicit URL is opaque, so nothing can prove where a wheel came from.
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/whl/cu130")
    assert mod._torch_index_tag("2.13.0+cu128") is None

    # And with no override at all the resident tag is still what is asked for.
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
    # The XPU repair really does land below the 0.18.0 floor.
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
        # Both signals agree: the ordinary Windows ROCm host, and the case a
        # two-mixed-only parametrization never covered.
        (True, True),
    ],
)
def test_skips_torchao_on_windows_rocm(
    monkeypatch, tmp_path, rocm_windows_torch_installed, installed_torch_is_windows_rocm
):
    """The overrides step must skip torchao on Windows ROCm: no working build exists
    there (it imports an absent c10d backend and crashes transformers.quantizers),
    so the installer skips it and relies on the runtime stub instead."""
    mod = _load_module(monkeypatch)
    installed_specs: list[str] = []
    progress_labels: list[str] = []

    def _record_pip_install(*args, **kwargs):
        installed_specs.extend(str(arg) for arg in args)
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
    # #10053 added a require_present gate to install_python_stack: after the core phase
    # it refuses when a managed distribution is not installed at all, which SKIP_STUDIO_BASE
    # guarantees here. Unstubbed, this test asks whether unsloth happens to be installed in
    # whatever environment runs it -- it passes on a developer machine that has it and fails
    # in CI, which is not what the test is about. Stubbed like every other installer side
    # effect below.
    monkeypatch.setattr(mod, "_repair_damaged_core_payload", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_bootstrap_uv", lambda: False)
    monkeypatch.setattr(mod, "_repair_bad_anyio", lambda: None)
    monkeypatch.setattr(mod, "_ensure_rocm_torch", lambda: None)
    monkeypatch.setattr(mod, "_ensure_cuda_torch", lambda: None)
    # A Windows ROCm box has no usable NVIDIA GPU. Claiming one here described a
    # machine that cannot exist, and _expected_torch_flavor_tag reads exactly this
    # flag to decide whether a CUDA expectation exists at all: with it True, the
    # Windows flavor invariant demanded a cu* build, found the runner's CPU torch,
    # and failed the whole install long after the torchao branch under test.
    monkeypatch.setattr(mod, "_has_usable_nvidia_gpu", lambda: False)
    # The installed torch is ambient, so leaving it unpatched made the verdict depend
    # on the developer's machine: a CUDA workstation passed and a CPU-only CI runner
    # failed, on identical code.
    monkeypatch.setattr(mod, "_RECORDED_TORCH_TAG", "")
    monkeypatch.setattr(
        mod, "_probe_torch_runtime", lambda *args, **kwargs: (True, True, "2.9.1+cpu", "", "")
    )
    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "pip_install", _record_pip_install)
    monkeypatch.setattr(mod, "_progress", lambda label: progress_labels.append(label))
    monkeypatch.setattr(mod, "LOCAL_DD_UNSTRUCTURED_PLUGIN", unstructured_plugin)
    monkeypatch.setattr(mod, "LOCAL_DD_GITHUB_PLUGIN", github_plugin)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: subprocess_result)

    # Checked BEFORE the install so a regression names its cause here rather than as
    # an opaque `assert 1 == 0` on the line below, which is how this surfaced: the run
    # returned 1 on a CPU-only runner and 0 on a CUDA workstation, on identical code.
    assert mod._expected_torch_flavor_tag() == "", (
        "no CUDA expectation may exist on a Windows ROCm host: a non-empty tag means "
        "the Windows flavor invariant will demand a cu* build, not find one, and fail "
        "the install long after the torchao branch this test is about"
    )

    assert mod.install_python_stack() == 0

    assert not any(spec.startswith("torchao") for spec in installed_specs)
    assert "dependency overrides (skipped, Windows ROCm)" in progress_labels
