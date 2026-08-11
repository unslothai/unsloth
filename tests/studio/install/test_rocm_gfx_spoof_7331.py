# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #7331 -- Studio segfaults at startup on Strix Halo.

Reporter: Ryzen AI MAX+ 395 with Radeon 8060S (Strix Halo, physically gfx1151),
Linux Mint, ROCm 6.3.42134, single GPU. Their rocminfo reported **gfx1100**, not
gfx1151, because HSA_OVERRIDE_GFX_VERSION=11.0.0 -- the widely circulated Strix
Halo workaround -- makes ROCr hand the spoofed ISA to every consumer, rocminfo
included. The installer believed it, routed torch to
download.pytorch.org/whl/rocm6.3 (torch>=2.4,<2.11.0 -> 2.9.1+rocm6.3), and the
first real allocation on the GPU ran gfx1100 kernels on gfx1151 silicon and died
with SIGSEGV. The reporter later self-corrected the hardware to "gfx1151 / Strix
Halo" and pointed at pytorch/pytorch#173367.

Unsloth already infers gfx1151 correctly from the product name in /proc/cpuinfo,
and already knows to route gfx1151 to repo.amd.com/rocm/whl/gfx1151/ with
torch>=2.11.0. The spoof defeated it: the ISA probe answered, so the correct
inference was discarded and the Strix reroute intersected {gfx1151, gfx1150,
gfx1152} against ["gfx1100"] and got nothing.

The mocks are shaped as the reporter's host, and deliberately not more:

  * ``_infer_linux_amd_gfx_arch -> "gfx1151"``   the cpuinfo product-name path,
    which is right and was being thrown away.
  * ``_detect_amd_gfx_codes -> ["gfx1100"]``     one device, the spoofed ISA.
    ONE entry matters: the correction only fires for a single-GPU probe, so the
    mixed Strix-APU-plus-dGPU host that the current precedence exists to protect
    (#7305) can never reach it.
  * ``_has_rocm_gpu -> True``                    rocminfo enumerates fine; the
    host is not the runtime-less #7301 case.
  * ``_detect_rocm_version -> (6, 3)``           the reporter's ROCm, and below
    the (7, 13) AMD per-arch floor, so the reroute gate is open.
  * ``HSA_OVERRIDE_GFX_VERSION = "11.0.0"``      the whole premise. With it unset
    every test here must fall back to today's behaviour.

There is no AMD hardware and no ROCm CI in this repo, so torch, rocminfo and pip
are all mocked; nothing below was validated on real silicon.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_7331", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_INSTALL_SH = PACKAGE_ROOT / "install.sh"

# torch probe stdout: "<hip marker>|<version>". Empty marker = CPU/CUDA torch.
_CPU_TORCH = b"|2.10.0+cpu\n"
_ROCM63_TORCH = b"6.3.42134|2.9.1+rocm6.3\n"  # what the reporter ended up with


def _run_install(
    *,
    torch_probe_stdout = _CPU_TORCH,
    gfx_devices = ("gfx1100",),
    inferred = "gfx1151",
    rocm_version = (6, 3),
    env = None,
    reprobe_devices = None,
    kfd_targets = (),
):
    """Drive _ensure_rocm_torch() over the reporter's host and return the pip calls.

    ``reprobe_devices`` is what rocminfo reports once HSA_OVERRIDE_GFX_VERSION is
    stripped from its environment; None means the re-probe is not available (it
    returns the same spoofed answer), which is the case the static
    HSA_OVERRIDE_GFX_VERSION -> gfx fallback has to carry on its own.

    ``kfd_targets`` is what the kernel reports. It defaults to empty -- no KFD
    sysfs, the common case on a CI box -- so each test says which evidence source
    it is exercising instead of inheriting whatever the test host happens to have.
    """
    probe = MagicMock()
    probe.returncode = 0
    probe.stdout = torch_probe_stdout

    def _fake_detect(dedup = True, ignore_hsa_override = False):
        if ignore_hsa_override and reprobe_devices is not None:
            codes = list(reprobe_devices)
        else:
            codes = list(gfx_devices)
        return list(dict.fromkeys(codes)) if dedup else codes

    _env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"} if env is None else env
    # The real _infer_linux_amd_gfx_arch() returns UNSLOTH_ROCM_GFX_ARCH before it
    # ever looks at /proc/cpuinfo; a stub that ignored it would model a host that
    # cannot exist and would make the escape-hatch test assert nothing.
    if _env.get("UNSLOTH_ROCM_GFX_ARCH"):
        inferred = _env["UNSLOTH_ROCM_GFX_ARCH"]

    with patch.object(stack_mod, "IS_WINDOWS", False), patch.object(
        stack_mod, "pip_install_try", return_value = True
    ) as pip_try, patch.object(stack_mod, "pip_install") as pip, patch.object(
        stack_mod, "_has_usable_nvidia_gpu", return_value = False
    ), patch.object(
        stack_mod, "_has_rocm_gpu", return_value = True
    ), patch.object(
        stack_mod, "_infer_linux_amd_gfx_arch", return_value = inferred
    ), patch.object(
        stack_mod, "_detect_amd_gfx_codes", side_effect = _fake_detect
    ), patch.object(
        stack_mod, "_detect_rocm_version", return_value = rocm_version
    ), patch.object(
        stack_mod, "_kfd_gfx_targets", return_value = list(kfd_targets)
    ), patch.dict(
        os.environ, _env, clear = False
    ):
        # patch.dict cannot REMOVE a key the outer environment happens to set, and
        # every one of these silently decides the outcome: the UNSLOTH_* pair
        # redirects the index, HSA_* is the premise, and a stray visible-device
        # mask re-indexes gfx_devices (a developer box exporting
        # CUDA_VISIBLE_DEVICES=4 picked a different device than CI would).
        for _stale in (
            "UNSLOTH_ROCM_GFX_ARCH",
            "UNSLOTH_AMD_ROCM_MIRROR",
            "HSA_OVERRIDE_GFX_VERSION",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            if _stale not in _env:
                os.environ.pop(_stale, None)
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = probe):
                stack_mod._ensure_rocm_torch()
    return str(pip.call_args_list) + str(pip_try.call_args_list)


# ── The reported host ────────────────────────────────────────────────────────


class TestSpoofedStrixHaloRouting:
    """gfx1151 spoofed to gfx1100 by HSA_OVERRIDE_GFX_VERSION must still get
    AMD's per-gfx1151 wheels, both on a fresh install and on a repair."""

    def test_kernel_settles_it(self):
        """The strongest path: amdkfd wrote gfx_target_version 110501 long before
        ROCr read the env var, so the kernel naming gfx1151 ends the argument
        without re-probing anything."""
        calls = _run_install(kfd_targets = ["gfx1151"], reprobe_devices = None)
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls

    def test_fresh_install_routes_to_the_gfx1151_index(self):
        """The reported path, with no KFD sysfs to read, so the corroboration is
        the re-probe. Before the fix this produced
        download.pytorch.org/whl/rocm6.3 with torch>=2.4,<2.11.0, i.e. the
        2.9.1+rocm6.3 that segfaulted."""
        calls = _run_install(reprobe_devices = ["gfx1151"])
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls
        assert "download.pytorch.org" not in calls, calls
        assert "rocm6.3" not in calls, calls

    def test_repair_over_the_broken_rocm63_torch(self):
        """`studio update` on the host as the reporter left it: torch is already
        2.9.1+rocm6.3, so has_hip_torch is True and only the Strix reroute can
        fire. It must, or the repair is a no-op and the segfault survives it."""
        calls = _run_install(
            torch_probe_stdout = _ROCM63_TORCH, reprobe_devices = ["gfx1151"]
        )
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls

    def test_static_fallback_when_the_reprobe_cannot_disprove_the_spoof(self):
        """Some ROCr builds keep reporting the override even with it stripped
        from the child environment (the override is baked into a config file, or
        the userland predates gfx1151 entirely). Then the re-probe returns the
        same gfx1100 and the decision falls back to the static reading of
        HSA_OVERRIDE_GFX_VERSION=11.0.0 -> gfx1100, which is exactly the arch the
        probe reported, on a host whose product name says gfx1151."""
        calls = _run_install(reprobe_devices = None)
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls

    def test_gfx1150_strix_point_spoofed_the_same_way(self):
        """11.0.0 is the circulated workaround for every RDNA 3.5 part, not just
        Strix Halo, so Strix Point (gfx1150) takes the same correction."""
        calls = _run_install(inferred = "gfx1150", reprobe_devices = ["gfx1150"])
        assert "repo.amd.com/rocm/whl/gfx1150/" in calls, calls


# ── Everything the correction must NOT touch ─────────────────────────────────


class TestPrecedenceStillHolds:
    """The runtime-visible arch keeps winning everywhere except the one narrow
    spoof shape. These are the cases the #7305 precedence exists for."""

    def test_no_override_set_keeps_todays_behaviour(self):
        """Without HSA_OVERRIDE_GFX_VERSION there is no evidence of a spoof, so a
        probe that disagrees with the product name is taken at face value -- the
        gfx1100 dGPU on a Strix box really is the runtime target. This is
        test_rocm_support.py::test_inference_yields_to_runtime_visible_gpu's
        premise and it is deliberately unchanged."""
        calls = _run_install(env = {}, rocm_version = (7, 1))
        assert "gfx1151" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_kernel_showing_a_second_gpu_vetoes_the_correction(self):
        """The mixed host seen through the kernel: the spoofed probe collapsed a
        Strix APU and a gfx1100 dGPU into one apparent gfx1100, so the
        single-device premise held. KFD sysfs sees both, which withdraws it. The
        correction must decline rather than take the first entry."""
        calls = _run_install(
            kfd_targets = ["gfx1151", "gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_mixed_host_with_two_devices_is_never_corrected(self):
        """The mixed Strix APU + discrete AMD GPU host, with the override set
        globally so the APU spoofs to gfx1100 as well. The probe reports two
        DEVICES, so the correction declines outright and the existing
        visible-mask selection decides, exactly as it does today. This is the
        case that makes the single-device requirement load-bearing rather than
        decorative."""
        calls = _run_install(
            gfx_devices = ("gfx1100", "gfx1100"),
            reprobe_devices = ["gfx1151", "gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_probe_unrelated_to_the_inferred_arch_is_not_a_spoof(self):
        """HSA_OVERRIDE_GFX_VERSION=11.0.0 cannot make a part report gfx1030, so
        a gfx1030 probe on a box whose name says gfx1151 is a real second GPU (or
        a broken inference), not the spoof. Left alone."""
        calls = _run_install(gfx_devices = ("gfx1030",), rocm_version = (7, 1))
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_override_matching_the_physical_arch_is_not_a_spoof(self):
        """HSA_OVERRIDE_GFX_VERSION=11.5.1 on a real gfx1151 names the arch the
        hardware already is. Nothing is being masked, so there is nothing to
        correct and the ordinary gfx1151 reroute handles it."""
        calls = _run_install(
            gfx_devices = ("gfx1151",),
            env = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
        )
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls

    def test_explicit_gfx_arch_override_still_wins(self):
        """UNSLOTH_ROCM_GFX_ARCH is the documented escape hatch and outranks every
        inference, including this one: a gfx906 host that also carries a stale
        HSA_OVERRIDE_GFX_VERSION in its profile still gets the gfx906 legacy
        routing, never Strix wheels."""
        calls = _run_install(
            gfx_devices = ("gfx1100",),
            kfd_targets = ["gfx1151"],
            env = {
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                "UNSLOTH_ROCM_GFX_ARCH": "gfx906",
            },
            rocm_version = (7, 1),
        )
        assert "gfx1151" not in calls, calls
        assert "rocm6.3" in calls, calls  # the gfx906 legacy index


# ── The HSA_OVERRIDE_GFX_VERSION -> gfx reading ──────────────────────────────


class TestHsaOverrideArch:
    """ROCr builds the target name from the version triple as
    gfx<major><minor><hex(stepping)>, which is why 9.0.10 is gfx90a and not
    gfx9010."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("11.0.0", "gfx1100"),
            ("11.5.1", "gfx1151"),
            ("10.3.0", "gfx1030"),
            ("9.0.10", "gfx90a"),
            ("9.4.2", "gfx942"),
            ("  11.0.0  ", "gfx1100"),
            ("11.0", None),  # not a triple
            ("", None),
            ("garbage", None),
            ("11.0.x", None),
        ],
    )
    def test_reading(self, value, expected):
        assert stack_mod._hsa_override_gfx_arch(value) == expected


class TestKfdGfxTargets:
    """amdkfd encodes gfx_target_version as major*10000 + minor*100 + stepping,
    the stepping again in hex. gfx1151 is 110501; the 110511 that circulates
    online is wrong and would decode to a nonexistent part, so it is pinned here.
    Reads a fabricated sysfs tree because CI has no AMD GPU."""

    def _nodes(self, tmp_path, monkeypatch, nodes):
        root = tmp_path / "nodes"
        for i, body in enumerate(nodes):
            node = root / f"{i}"
            node.mkdir(parents = True)
            (node / "properties").write_text(body, encoding = "utf-8")
        monkeypatch.setattr(stack_mod.sys, "platform", "linux")
        real_listdir, real_open = os.listdir, open

        def _listdir(path):
            return real_listdir(str(root) if path.endswith("topology/nodes") else path)

        def _open(path, *a, **kw):
            path = str(path)
            if "topology/nodes" in path:
                path = str(root / path.split("topology/nodes/", 1)[1])
            return real_open(path, *a, **kw)

        monkeypatch.setattr(stack_mod.os, "listdir", _listdir)
        monkeypatch.setattr("builtins.open", _open)

    def test_decodes_the_reporter_gpu(self, tmp_path, monkeypatch):
        self._nodes(
            tmp_path,
            monkeypatch,
            [
                "cpu_cores_count 16\nvendor_id 0\ngfx_target_version 0\n",  # CPU node
                "simd_count 640\nvendor_id 4098\ngfx_target_version 110501\n",
            ],
        )
        assert stack_mod._kfd_gfx_targets() == ["gfx1151"]

    def test_hex_stepping_and_non_amd_nodes(self, tmp_path, monkeypatch):
        self._nodes(
            tmp_path,
            monkeypatch,
            [
                "vendor_id 4098\ngfx_target_version 90010\n",  # gfx90a
                "vendor_id 4318\ngfx_target_version 110000\n",  # NVIDIA KFD node
                "vendor_id 4098\ngfx_target_version 110000\n",  # gfx1100
            ],
        )
        assert stack_mod._kfd_gfx_targets() == ["gfx90a", "gfx1100"]


# ── install.sh parity ────────────────────────────────────────────────────────


def _sh_func(name: str) -> str:
    """Extract one top-level POSIX function definition out of install.sh."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    start = source.find(f"\n{name}() {{\n")
    assert start != -1, f"{name}() not found in install.sh"
    end = source.find("\n}\n", start)
    assert end != -1, f"{name}() has no closing brace"
    return source[start + 1 : end + 3]


def _run_sh(script: str, env = None) -> str:
    """Run a /bin/sh snippet with the install.sh helpers sourced."""
    import subprocess as _sp

    preamble = (
        _sh_func("_hsa_override_gfx_arch")
        + _sh_func("_hsa_spoofed_physical_gfx")
        # The kernel source is stubbed: CI has no /sys/class/kfd, and the point
        # here is the DECISION, which the Python side unit-tests node parsing for.
        + "\n_kfd_gfx_targets() { printf '%s\\n' \"${FAKE_KFD:-}\" | awk 'NF'; }\n"
    )
    _env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    _env.update(env or {})
    proc = _sp.run(
        ["/bin/sh", "-c", preamble + script],
        capture_output = True,
        text = True,
        timeout = 60,
        env = _env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


class TestInstallShParity:
    """install.sh routes the curl | sh install that #7331 was reported against,
    so it must reach the same verdict as install_python_stack.py. A host that a
    `studio update` fixes and a fresh install re-breaks is the worst outcome, so
    these EXECUTE the shell helpers rather than grepping for them."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("11.0.0", "gfx1100"),
            ("11.5.1", "gfx1151"),
            ("10.3.0", "gfx1030"),
            ("9.0.10", "gfx90a"),
            ("9.4.2", "gfx942"),
            ("  11.0.0  ", "gfx1100"),
            ("11.0", ""),
            ("", ""),
            ("garbage", ""),
            ("11.0.x", ""),
        ],
    )
    def test_override_reading_matches_python(self, value, expected):
        got = _run_sh(f'_hsa_override_gfx_arch "{value}"')
        assert got == expected
        assert got == (stack_mod._hsa_override_gfx_arch(value) or "")

    @pytest.mark.parametrize(
        "inferred,probed,kfd,override,expected,why",
        [
            ("gfx1151", "gfx1100", "gfx1151", "11.0.0", "gfx1151", "kernel settles it"),
            ("gfx1151", "gfx1100", "", "11.0.0", "gfx1151", "static fallback"),
            ("gfx1151", "gfx1100", "gfx1151\ngfx1100", "11.0.0", "", "mixed host via kernel"),
            ("gfx1151", "gfx1100", "gfx1151", "", "", "no override set"),
            ("gfx1151", "gfx1100\ngfx1100", "gfx1151", "11.0.0", "", "two devices probed"),
            ("gfx1151", "gfx1030", "", "11.0.0", "", "probe unrelated to the name"),
            ("gfx1151", "gfx1151", "gfx1151", "11.5.1", "", "override is a no-op remap"),
            ("gfx1030", "gfx1100", "gfx1030", "11.0.0", "", "not a spoofable arch"),
        ],
    )
    def test_decision_matches_python(self, inferred, probed, kfd, override, expected, why):
        """The same eight shapes on both paths. `why` names the rule each pins."""
        env = {"FAKE_KFD": kfd}
        if override:
            env["HSA_OVERRIDE_GFX_VERSION"] = override
        got = _run_sh(
            f'_hsa_spoofed_physical_gfx "{inferred}" "{probed}" 2>/dev/null', env = env
        )
        assert got == expected, why

        with patch.dict(os.environ, env, clear = False), patch.object(
            stack_mod, "_kfd_gfx_targets", return_value = [c for c in kfd.split("\n") if c]
        ), patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = []):
            if not override:
                os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
            py = stack_mod._hsa_spoofed_physical_gfx(inferred, probed.split("\n"))
        assert (py or "") == expected, f"{why}: shell said {got!r}, python said {py!r}"

    def test_reroute_consults_the_correction(self):
        """The helper existing is not enough; the Strix reroute has to call it."""
        source = _INSTALL_SH.read_text(encoding = "utf-8")
        # rfind: an earlier case on the same variable sets UNSLOTH_TORCH_BACKEND.
        idx = source.rfind('case "$_torch_index_leaf" in')
        assert idx != -1
        assert "_hsa_spoofed_physical_gfx" in source[idx : idx + 6000], (
            "install.sh's Strix reroute must correct the spoofed probe before it "
            "matches gfx1150/1151/1152"
        )

    def test_kfd_reader_uses_the_kernel_encoding(self):
        """gfx_target_version is major*10000 + minor*100 + stepping, and the
        stepping is hex (110501 -> gfx1151). A copy that decoded it decimally
        would silently classify every Strix host as something else."""
        body = _sh_func("_kfd_gfx_targets")
        assert "10000" in body and "vendor_id" in body
        assert "%x" in body, "the stepping must be rendered in hex"
