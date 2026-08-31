# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #7331 -- Unsloth segfaults at startup on Strix Halo.

Reporter: Ryzen AI MAX+ 395 with Radeon 8060S (Strix Halo, physically gfx1151),
Linux Mint, ROCm 6.3.42134, single GPU. Their rocminfo reported **gfx1100** because
HSA_OVERRIDE_GFX_VERSION=11.0.0 -- the circulated Strix Halo workaround -- makes ROCr
hand the spoofed ISA to every consumer. The installer believed it, routed torch to
download.pytorch.org/whl/rocm6.3 (torch>=2.4,<2.11.0 -> 2.9.1+rocm6.3), and the first
allocation ran gfx1100 kernels on gfx1151 silicon and died with SIGSEGV. The reporter
later self-corrected the hardware to "gfx1151 / Strix Halo" and pointed at
pytorch/pytorch#173367.

Unsloth already infers gfx1151 from the product name in /proc/cpuinfo and already
routes gfx1151 to repo.amd.com/rocm/whl/gfx1151/ with torch>=2.11.0. The spoof
defeated it: the ISA probe answered, so the correct inference was discarded and the
Strix reroute intersected {gfx1151, gfx1150, gfx1152} against ["gfx1100"].

The mocks are shaped as the reporter's host, and deliberately not more:

  * ``_infer_linux_amd_gfx_arch -> "gfx1151"``   the cpuinfo product-name path,
    which is right and was being thrown away.
  * ``_detect_amd_gfx_codes -> ["gfx1100"]``     one device, the spoofed ISA. ONE
    arch matters: the correction only fires for a single-arch probe, so the mixed
    Strix-APU-plus-dGPU host the current precedence protects (#7305) is filtered
    out before the evidence ladder is consulted.
  * ``_has_rocm_gpu -> True``                    rocminfo enumerates fine; not the
    runtime-less #7301 case.
  * ``_detect_rocm_version -> (6, 3)``           the reporter's ROCm, below the
    (7, 13) AMD per-arch floor, so the reroute gate is open.
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

# torch probe stdout: "<version>|<hip>|<cuda>". An empty HIP field = CPU/CUDA torch.
_CPU_TORCH = "2.10.0+cpu||\n"
_ROCM63_TORCH = "2.9.1+rocm6.3|6.3.42134|\n"  # what the reporter ended up with


@pytest.fixture(autouse = True)
def _reset_torch_runtime_probe():
    """The torch classification is memoized for the life of an install run, so one
    test's mocked probe must not leak into the next."""
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


def _run_install(
    *,
    torch_probe_stdout = _CPU_TORCH,
    gfx_devices = ("gfx1100",),
    inferred = "gfx1151",
    rocm_version = (6, 3),
    env = None,
    reprobe_devices = None,
    kfd_targets = (),
    env_probe = None,
):
    """Drive _ensure_rocm_torch() over the reporter's host and return the pip calls.

    ``env_probe``, when a dict is passed, records os.environ["HSA_OVERRIDE_GFX_VERSION"]
    at the moment torch is installed. That instant is the one that matters: patch.dict
    restores the environment on exit, so an assertion made after the call cannot see a
    pop, and the env the install ends with is what later processes inherit.

    ``reprobe_devices`` is what rocminfo reports once HSA_OVERRIDE_GFX_VERSION is
    stripped; None means the re-probe answers with the same arch and so cannot
    disprove the spoof, the shape with no honest reading that must be declined.

    ``kfd_targets`` is what the kernel reports, defaulting to empty (no KFD sysfs, the
    common CI case) so each test names the evidence source it exercises.
    """
    probe = MagicMock()
    probe.returncode = 0
    probe.stdout = torch_probe_stdout

    def _fake_detect(
        dedup = True,
        ignore_hsa_override = False,
        ignore_visible_masks = False,
    ):
        if ignore_hsa_override and reprobe_devices is not None:
            codes = list(reprobe_devices)
        else:
            codes = list(gfx_devices)
        return list(dict.fromkeys(codes)) if dedup else codes

    _env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"} if env is None else env
    # The real _infer_linux_amd_gfx_arch() returns UNSLOTH_ROCM_GFX_ARCH before it ever
    # looks at /proc/cpuinfo; a stub ignoring it would make the escape-hatch test assert
    # nothing.
    if _env.get("UNSLOTH_ROCM_GFX_ARCH"):
        inferred = _env["UNSLOTH_ROCM_GFX_ARCH"]

    def _record_env(*a, **kw):
        if env_probe is not None and "HSA_OVERRIDE_GFX_VERSION" not in env_probe:
            env_probe["HSA_OVERRIDE_GFX_VERSION"] = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        return True

    with (
        patch.object(stack_mod, "IS_WINDOWS", False),
        patch.object(stack_mod, "pip_install_try", side_effect = _record_env) as pip_try,
        patch.object(stack_mod, "pip_install") as pip,
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack_mod, "_has_rocm_gpu", return_value = True),
        patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = inferred),
        patch.object(stack_mod, "_detect_amd_gfx_codes", side_effect = _fake_detect),
        patch.object(stack_mod, "_detect_rocm_version", return_value = rocm_version),
        # create = True: without it, a tree predating the fix raises AttributeError,
        # which fails for the wrong reason. With it, the old code runs untouched and
        # fails on the assertion instead.
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(kfd_targets), create = True),
        patch.dict(os.environ, _env, clear = False),
    ):
        # patch.dict cannot REMOVE a key the outer environment sets, and each of these
        # silently decides the outcome: the UNSLOTH_* pair redirects the index, HSA_* is
        # the premise, and a stray visible-device mask re-indexes gfx_devices.
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
    """gfx1151 spoofed to gfx1100 by HSA_OVERRIDE_GFX_VERSION must still get AMD's
    per-gfx1151 wheels, on a fresh install and on a repair."""

    def test_kernel_settles_it(self):
        """The strongest path: amdkfd wrote gfx_target_version 110501 long before ROCr
        read the env var, so the kernel naming gfx1151 ends the argument."""
        calls = _run_install(kfd_targets = ["gfx1151"], reprobe_devices = None)
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls

    def test_fresh_install_routes_to_the_gfx1151_index(self):
        """The reported path with no KFD sysfs, so the re-probe corroborates. Before the
        fix this produced download.pytorch.org/whl/rocm6.3 with torch>=2.4,<2.11.0, the
        2.9.1+rocm6.3 that segfaulted."""
        calls = _run_install(reprobe_devices = ["gfx1151"])
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls
        assert "download.pytorch.org" not in calls, calls
        assert "rocm6.3" not in calls, calls

    def test_repair_over_the_broken_rocm63_torch(self):
        """`studio update` on the host as the reporter left it: torch is already
        2.9.1+rocm6.3, so has_hip_torch is True and only the Strix reroute can fire. It
        must, or the repair is a no-op and the segfault survives."""
        calls = _run_install(torch_probe_stdout = _ROCM63_TORCH, reprobe_devices = ["gfx1151"])
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert "torch>=2.11.0,<2.12.0" in calls, calls

    def test_uncorroborated_spoof_is_left_alone(self):
        """The shape with no honest answer: the kernel is silent and the re-probe still
        says gfx1100 with the override stripped, which is exactly what a real gfx1100
        dGPU looks like. Rerouting a working machine to the wrong wheels is worse than
        #7331 itself, so this host keeps today's routing rather than a coin flip."""
        calls = _run_install(reprobe_devices = None, rocm_version = (7, 1))
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_gfx1150_strix_point_spoofed_the_same_way(self):
        """11.0.0 is the circulated workaround for every RDNA 3.5 part, so Strix Point
        (gfx1150) takes the same correction."""
        calls = _run_install(inferred = "gfx1150", reprobe_devices = ["gfx1150"])
        assert "repo.amd.com/rocm/whl/gfx1150/" in calls, calls


# ── Everything the correction must NOT touch ─────────────────────────────────


class TestPrecedenceStillHolds:
    """The runtime-visible arch keeps winning outside the one narrow spoof shape.
    These are the cases the #7305 precedence exists for."""

    def test_no_override_set_keeps_todays_behaviour(self):
        """Without the override there is no evidence of a spoof, so a probe that
        disagrees with the product name is taken at face value. This is
        test_rocm_support.py::test_inference_yields_to_runtime_visible_gpu's premise,
        deliberately unchanged."""
        calls = _run_install(env = {}, rocm_version = (7, 1))
        assert "gfx1151" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_kernel_showing_a_second_gpu_vetoes_the_correction(self):
        """The mixed host seen through the kernel: the spoofed probe collapsed a Strix
        APU and a gfx1100 dGPU into one apparent gfx1100, so the single-device premise
        held. KFD sysfs sees both and withdraws it, so the correction must decline."""
        calls = _run_install(
            kfd_targets = ["gfx1151", "gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_mixed_host_with_two_devices_is_never_corrected(self):
        """The mixed Strix APU + discrete AMD GPU host, override set globally so the APU
        spoofs to gfx1100 too. The probe reports two DEVICES, so the correction declines
        and today's visible-mask selection decides. This is what makes the single-device
        requirement load-bearing rather than decorative."""
        calls = _run_install(
            gfx_devices = ("gfx1100", "gfx1100"),
            reprobe_devices = ["gfx1151", "gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_probe_unrelated_to_the_inferred_arch_is_not_a_spoof(self):
        """11.0.0 cannot make a part report gfx1030, so a gfx1030 probe on a box whose
        name says gfx1151 is a real second GPU, not the spoof. Left alone."""
        calls = _run_install(gfx_devices = ("gfx1030",), rocm_version = (7, 1))
        assert "repo.amd.com" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_override_matching_the_physical_arch_is_not_a_spoof(self):
        """11.5.1 on a real gfx1151 names the arch the hardware already is: nothing is
        masked, so the ordinary gfx1151 reroute handles it."""
        calls = _run_install(
            gfx_devices = ("gfx1151",),
            env = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
        )
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls

    def test_real_gfx1100_dgpu_in_a_ryzen_ai_max_chassis(self):
        """The nastiest shape. The product name comes from the CPU model in
        /proc/cpuinfo, so a Ryzen AI Max machine infers gfx1151 whatever card is in the
        slot; drop a real RX 7900 XTX in it (APU off in the BIOS, so one device) and a
        user carrying the override for an unrelated reason presents EXACTLY the
        reporter's fingerprint: inferred gfx1151, probed gfx1100, one device, override
        naming gfx1100.

        Only the kernel tells them apart, and it must be believed: gfx_target_version
        110000 is a real gfx1100 and the card keeps its own wheels."""
        calls = _run_install(
            kfd_targets = ["gfx1100"],
            reprobe_devices = ["gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "gfx1151" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_real_gfx1100_dgpu_with_no_kfd_sysfs(self):
        """The same card in a container with no /sys/class/kfd, so only the re-probe can
        arbitrate: it answers gfx1100 with the override stripped, which is evidence FOR
        the probe. This is what the removed "override names the probe, so assume a spoof"
        fallback used to get wrong."""
        calls = _run_install(kfd_targets = (), reprobe_devices = ["gfx1100"], rocm_version = (7, 1))
        assert "repo.amd.com" not in calls, calls
        assert "gfx1151" not in calls, calls
        assert "rocm7.1" in calls, calls

    def test_deliberate_rdna2_override_is_not_a_strix_spoof(self):
        """10.3.0 on an RX 6800 is the documented and CORRECT setting for that card. The
        chassis is still a Ryzen AI Max, so the name still infers gfx1151 and the probe
        still disagrees, but that does not make gfx1030 a spoof."""
        calls = _run_install(
            gfx_devices = ("gfx1030",),
            reprobe_devices = ["gfx1030"],
            env = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"},
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls
        assert "gfx1151" not in calls, calls

    def test_override_naming_a_third_arch_is_never_a_spoof(self):
        """ROCr can only rename an agent to the target the variable names, so 10.3.0
        alongside a gfx1100 reading means gfx1100 came from the silicon. No spoof to
        undo, even if the kernel is silent and the re-probe agrees."""
        calls = _run_install(
            env = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"},
            reprobe_devices = ["gfx1151"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls

    def test_masked_mixed_host_reprobes_the_whole_machine(self):
        """ROCR_VISIBLE_DEVICES pinned to the dGPU on a Strix + 7900 XTX box collapses
        the probe to one gfx1100, so the single-arch premise holds and the kernel is
        unavailable in the container. The re-probe must clear the mask as well as the
        override, or the second GPU that vetoes the correction stays hidden."""
        calls = _run_install(
            env = {
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                "ROCR_VISIBLE_DEVICES": "1",
            },
            reprobe_devices = ["gfx1151", "gfx1100"],
            rocm_version = (7, 1),
        )
        assert "repo.amd.com" not in calls, calls

    def test_explicit_gfx_arch_override_still_wins(self):
        """UNSLOTH_ROCM_GFX_ARCH is the documented escape hatch and outranks every
        inference: a gfx906 host also carrying a stale override still gets the gfx906
        legacy routing, never Strix wheels."""
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
    gfx<major><minor><hex(stepping)>, which is why 9.0.10 is gfx90a."""

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
    """amdkfd encodes gfx_target_version as major*10000 + minor*100 + stepping, the
    stepping in hex. gfx1151 is 110501; the 110511 circulating online decodes to a
    nonexistent part, so it is pinned here. Reads a fabricated sysfs tree."""

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
        # The kernel source is stubbed: CI has no /sys/class/kfd and the point here is
        # the DECISION; node parsing is unit-tested on the Python side.
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
    """install.sh routes the curl | sh install #7331 was reported against, so it must
    reach the same verdict as install_python_stack.py: a host a `studio update` fixes
    and a fresh install re-breaks is the worst outcome. These EXECUTE the shell helpers
    rather than grepping for them."""

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
            ("gfx1151", "gfx1100", "", "11.0.0", "", "uncorroborated, no rocminfo"),
            ("gfx1151", "gfx1100", "gfx1100", "11.0.0", "", "real 7900 XTX per the kernel"),
            ("gfx1151", "gfx1100", "gfx1151\ngfx1100", "11.0.0", "", "mixed host via kernel"),
            ("gfx1151", "gfx1100", "gfx1151", "", "", "no override set"),
            ("gfx1151", "gfx1100\ngfx1100", "gfx1151", "11.0.0", "gfx1151", "one arch, repeated"),
            ("gfx1151", "gfx1100\ngfx1030", "gfx1151", "11.0.0", "", "two arches probed"),
            ("gfx1151", "gfx1030", "", "11.0.0", "", "probe unrelated to the name"),
            ("gfx1151", "gfx1030", "gfx1151", "11.0.0", "", "override names no such probe"),
            ("gfx1151", "gfx1151", "gfx1151", "11.5.1", "", "override is a no-op remap"),
            ("gfx1030", "gfx1100", "gfx1030", "11.0.0", "", "not a spoofable arch"),
        ],
    )
    def test_decision_matches_python(self, inferred, probed, kfd, override, expected, why):
        """The same named shapes on both paths. `why` names the rule each pins."""
        env = {"FAKE_KFD": kfd}
        if override:
            env["HSA_OVERRIDE_GFX_VERSION"] = override
        got = _run_sh(f'_hsa_spoofed_physical_gfx "{inferred}" "{probed}" 2>/dev/null', env = env)
        assert got == expected, why

        with (
            patch.dict(os.environ, env, clear = False),
            patch.object(
                stack_mod, "_kfd_gfx_targets", return_value = [c for c in kfd.split("\n") if c]
            ),
            patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = []),
        ):
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

    @pytest.mark.parametrize(
        "nodes,expected",
        [
            (
                [
                    "cpu_cores_count 16\nvendor_id 0\ngfx_target_version 0\n",
                    "simd_count 640\nvendor_id 4098\ngfx_target_version 110501\n",
                ],
                ["gfx1151"],
            ),
            (
                [
                    "vendor_id 4098\ngfx_target_version 90010\n",
                    "vendor_id 4318\ngfx_target_version 110000\n",
                    "vendor_id 4098\ngfx_target_version 110000\n",
                ],
                ["gfx90a", "gfx1100"],
            ),
            (["vendor_id 4098\n"], []),  # no gtv line
            (["vendor_id 4098\ngfx_target_version notanumber\n"], []),  # malformed
            ([""], []),  # empty properties
            ([], []),  # no nodes at all
        ],
    )
    def test_kfd_reader_matches_python(self, tmp_path, nodes, expected):
        """The shell KFD parser EXECUTED against a fabricated topology tree.
        gfx_target_version is major*10000 + minor*100 + stepping in hex (110501 ->
        gfx1151); decoding it decimally, or letting a CPU / NVIDIA node through, would
        misroute every Strix host. Only the sysfs root differs from the shipped one."""
        root = tmp_path / "nodes"
        root.mkdir()
        for i, body in enumerate(nodes):
            (root / str(i)).mkdir()
            (root / str(i) / "properties").write_text(body, encoding = "utf-8")
        body = _sh_func("_kfd_gfx_targets").replace("/sys/class/kfd/kfd/topology/nodes", str(root))
        assert str(root) in body, "the sysfs root substitution must have applied"
        got = _run_sh(body + "\n_kfd_gfx_targets\n")
        assert [c for c in got.split("\n") if c] == expected

    def test_reprobe_falls_through_to_amd_smi_like_python_does(self, tmp_path):
        """rocminfo FAILS on the host this feature exists for: strip the override on a
        ROCm stack older than the physical arch and hsa_init errors with no agents
        listed. amd-smi reads the driver and still answers, and _detect_amd_gfx_codes
        falls through to it, so install.sh must too or a `studio update` fixes a host a
        fresh `curl | sh` install leaves on the segfaulting wheels."""
        _bin = tmp_path / "bin"
        _bin.mkdir()
        (_bin / "rocminfo").write_text(
            "#!/bin/sh\n"
            'if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then exit 1; fi\n'
            'echo "  Name:                    gfx1100"\n'
            'echo "    Name:                    amdgcn-amd-amdhsa--gfx1100"\n',
            encoding = "utf-8",
        )
        (_bin / "amd-smi").write_text(
            '#!/bin/sh\necho "        TARGET_GRAPHICS_VERSION: gfx1151"\n', encoding = "utf-8"
        )
        for _f in ("rocminfo", "amd-smi"):
            (_bin / _f).chmod(0o755)
        env = {
            "FAKE_KFD": "",  # no /sys/class/kfd, so the re-probe is the only witness
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "PATH": str(_bin) + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
        }
        got = _run_sh('_hsa_spoofed_physical_gfx "gfx1151" "gfx1100\ngfx1100" 2>/dev/null', env = env)
        assert got == "gfx1151", got

        # The Python side, same host, so the parity claim is asserted and not assumed.
        with (
            patch.dict(os.environ, env, clear = False),
            patch.object(stack_mod, "_kfd_gfx_targets", return_value = []),
            patch.object(stack_mod, "_amd_smi_allowed", return_value = True),
            patch.object(stack_mod, "_safe_print"),
        ):
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            py = stack_mod._hsa_spoofed_physical_gfx("gfx1151", ["gfx1100", "gfx1100"])
        assert py == got, (py, got)

    def test_reprobe_clears_the_visible_masks_too(self):
        """install.sh's re-probe runs `unset HSA_OVERRIDE_GFX_VERSION
        ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES`, so the Python one must drop all three
        too: a mask left in place hides the second GPU that vetoes the correction."""
        _unset = [
            ln for ln in _sh_func("_hsa_spoofed_physical_gfx").splitlines() if "(unset " in ln
        ]
        # One per re-probe tool (rocminfo, then the two amd-smi fallbacks); any that
        # dropped fewer would re-probe a masked or still-spoofed machine.
        assert _unset, _unset
        for _line in _unset:
            for _var in (
                "HSA_OVERRIDE_GFX_VERSION",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
            ):
                assert _var in _line, (_var, _line)

        seen = {}

        def _fake_run(cmd, **kw):
            seen["env"] = kw.get("env")
            out = MagicMock()
            out.returncode = 0
            out.stdout = "Agent 1\n  Name: gfx1151\n"
            return out

        env = {
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "ROCR_VISIBLE_DEVICES": "1",
            "HIP_VISIBLE_DEVICES": "1",
        }
        with (
            patch.dict(os.environ, env, clear = False),
            patch.object(
                stack_mod.shutil,
                "which",
                side_effect = lambda c: "/usr/bin/rocminfo" if c == "rocminfo" else None,
            ),
            patch.object(stack_mod.subprocess, "run", side_effect = _fake_run),
        ):
            stack_mod._detect_amd_gfx_codes(
                dedup = False, ignore_hsa_override = True, ignore_visible_masks = True
            )
        assert seen["env"] is not None, "the probe must not inherit the environment"
        for _var in env:
            assert _var not in seen["env"], _var


# ── Randomized shell/Python parity ───────────────────────────────────────────

# A fake rocminfo shared by both implementations, so each executes its OWN re-probe
# (env stripping included) rather than a mock. It models the behaviour under test: ROCr
# renames every agent to HSA_OVERRIDE_GFX_VERSION's target when set, and reports the
# physical arch when not; ROCR_VISIBLE_DEVICES selects agents. The output repeats the
# token per Name / ISA line, which is what makes the shell's raw `grep -oE` produce
# several lines per single GPU.
_FAKE_ROCMINFO = r"""#!/bin/sh
_phys="${FAKE_PHYSICAL:-}"
_spoof=$(printf '%s' "${HSA_OVERRIDE_GFX_VERSION:-}" | awk '
    { if ($0 !~ /^[0-9]+\.[0-9]+\.[0-9]+$/) exit
      split($0, p, "."); maj = p[1] + 0; min = p[2] + 0; st = p[3] + 0
      if (maj <= 0 || min > 9 || st > 15) exit
      printf "gfx%d%d%x", maj, min, st }')
_sel="${ROCR_VISIBLE_DEVICES:-}"
echo "ROCk module is loaded"
_i=0
for _a in $_phys; do
    if [ -n "$_sel" ]; then
        case ",$_sel," in *",$_i,"*) : ;; *) _i=$((_i + 1)); continue ;; esac
    fi
    [ -n "$_spoof" ] && _a="$_spoof"
    echo "*******"
    echo "Agent $((_i + 2))"
    echo "*******"
    echo "  Name:                    $_a"
    echo "  Device Type:             GPU"
    echo "    Name:                    amdgcn-amd-amdhsa--$_a"
    _i=$((_i + 1))
done
echo "*** Done ***"
"""


@pytest.fixture(scope = "module")
def fake_rocminfo(tmp_path_factory):
    _bin = tmp_path_factory.mktemp("fakebin")
    _rocminfo = _bin / "rocminfo"
    _rocminfo.write_text(_FAKE_ROCMINFO, encoding = "utf-8")
    _rocminfo.chmod(0o755)
    return str(_bin)


def _shapes(seed: int, count: int):
    """A randomized host matrix: override x physical arches x probed arches x KFD
    contents x mask. The override values include what users actually produce (empty,
    whitespace, non-numeric, wrong component count, absurd magnitudes), which reach the
    same helper as 11.0.0."""
    import random

    rng = random.Random(seed)
    overrides = [
        "",
        "11.0.0",
        "11.5.1",
        "10.3.0",
        "9.0.10",
        "9.4.2",
        "garbage",
        "11.0",
        "11.0.0.0",
        "  11.0.0  ",
        "11.0.x",
        "-1.0.0",
        "999999.0.0",
        "11.0.16",
        "0.0.0",
        "11.10.0",
        " ",
        "11..0",
    ]
    arches = ["gfx1151", "gfx1150", "gfx1152", "gfx1100", "gfx1030", "gfx906", "gfx942"]
    inferreds = arches + [""]
    for _ in range(count):
        physical = [rng.choice(arches) for _ in range(rng.choice([0, 1, 1, 1, 2, 2, 3]))]
        # The probe list as install.sh builds it: raw tokens, repeated per agent.
        probed = []
        for _a in physical:
            probed.extend([_a] * rng.choice([1, 2, 2, 3]))
        yield {
            "inferred": rng.choice(inferreds),
            "probed": probed,
            "kfd": [rng.choice(arches) for _ in range(rng.choice([0, 0, 1, 1, 2]))],
            "override": rng.choice(overrides),
            "physical": physical,
            "mask": rng.choice(["", "", "", "0", "1", "-1"]),
        }


class TestRandomizedParity:
    """install.sh and install_python_stack.py must reach the SAME verdict on every host
    shape, or a `studio update` fixes a machine a fresh `curl | sh` install re-breaks.
    The named shapes above pin the rules; this pins the whole space, including the
    re-probe, which they cannot reach on a box with no rocminfo."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_verdicts_agree(self, seed, fake_rocminfo):
        divergences = []
        for shape in _shapes(seed, 100):
            env = {
                "FAKE_KFD": "\n".join(shape["kfd"]),
                "FAKE_PHYSICAL": " ".join(shape["physical"]),
                "PATH": fake_rocminfo + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
            }
            if shape["override"]:
                env["HSA_OVERRIDE_GFX_VERSION"] = shape["override"]
            if shape["mask"]:
                env["ROCR_VISIBLE_DEVICES"] = shape["mask"]

            sh = _run_sh(
                '_hsa_spoofed_physical_gfx "$SHAPE_INFERRED" "$SHAPE_PROBED" 2>/dev/null',
                env = dict(
                    env,
                    SHAPE_INFERRED = shape["inferred"],
                    SHAPE_PROBED = "\n".join(shape["probed"]),
                ),
            )

            with (
                patch.dict(os.environ, env, clear = False),
                patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(shape["kfd"])),
                patch.object(stack_mod, "_amd_smi_allowed", return_value = False),
                patch.object(stack_mod, "_safe_print"),
            ):
                for _stale in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES"):
                    if _stale not in env:
                        os.environ.pop(_stale, None)
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
                py = stack_mod._hsa_spoofed_physical_gfx(
                    shape["inferred"] or None, list(shape["probed"])
                )
            if (py or "") != sh:
                divergences.append((shape, sh, py))
        assert not divergences, divergences[:5]

    @pytest.mark.parametrize("seed", [0])
    def test_never_corrects_without_corroboration(self, seed, fake_rocminfo):
        """The safety invariant over the same random matrix: a verdict is only returned
        when a source the override cannot reach (the kernel, or rocminfo re-probed
        without it) named that exact arch and nothing else."""
        for shape in _shapes(seed, 100):
            env = {
                "FAKE_KFD": "\n".join(shape["kfd"]),
                "FAKE_PHYSICAL": " ".join(shape["physical"]),
                "PATH": fake_rocminfo + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
            }
            if shape["override"]:
                env["HSA_OVERRIDE_GFX_VERSION"] = shape["override"]
            if shape["mask"]:
                env["ROCR_VISIBLE_DEVICES"] = shape["mask"]
            with (
                patch.dict(os.environ, env, clear = False),
                patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(shape["kfd"])),
                patch.object(stack_mod, "_amd_smi_allowed", return_value = False),
                patch.object(stack_mod, "_safe_print"),
            ):
                for _stale in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES"):
                    if _stale not in env:
                        os.environ.pop(_stale, None)
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
                py = stack_mod._hsa_spoofed_physical_gfx(
                    shape["inferred"] or None, list(shape["probed"])
                )
            if py is None:
                continue
            # Corroborated by the kernel or by the unspoofed physical truth the
            # re-probe reads back. Never by the variable alone.
            assert shape["kfd"] == [py] or list(dict.fromkeys(shape["physical"])) == [py], (
                shape,
                py,
            )
            assert py == shape["inferred"], (shape, py)


class TestCallSiteParity:
    """The parity a helper-level comparison cannot see: each side builds its OWN probe
    input from the same rocminfo the way its call site does. install.sh feeds
    `rocminfo | grep -oE 'gfx...'` STRAIGHT in, so one GPU arrives as two or three
    repeated tokens, while install_python_stack.py splits on agent headers and gets one
    entry per device. A pre-shaped list hides that difference, and it is exactly what
    decides whether the correction fires on the host #7331 was reported from."""

    @pytest.mark.parametrize(
        "override", ["", "11.0.0", "11.5.1", "10.3.0", "garbage", "999999.0.0"]
    )
    def test_verdicts_agree_end_to_end(self, override, fake_rocminfo):
        import itertools
        import subprocess as _sp

        inferreds = ["gfx1151", "gfx1150", "gfx1100", "gfx1030", ""]
        physicals = [
            [],
            ["gfx1151"],
            ["gfx1100"],
            ["gfx1030"],
            ["gfx1151", "gfx1100"],
            ["gfx1100", "gfx1100"],
            ["gfx1151", "gfx1151"],
        ]
        kfds = [[], ["gfx1151"], ["gfx1100"], ["gfx1151", "gfx1100"]]
        masks = ["", "0", "1"]

        _path = fake_rocminfo + ":" + os.environ.get("PATH", "/usr/bin:/bin")
        divergences, corrections, shapes = [], 0, 0
        for inferred, physical, kfd, mask in itertools.product(inferreds, physicals, kfds, masks):
            env = {
                "FAKE_KFD": "\n".join(kfd),
                "FAKE_PHYSICAL": " ".join(physical),
                "PATH": _path,
            }
            if override:
                env["HSA_OVERRIDE_GFX_VERSION"] = override
            if mask:
                env["ROCR_VISIBLE_DEVICES"] = mask

            # install.sh's call site, verbatim: raw grep output, duplicates and all.
            sh_probe = _sp.run(
                [
                    "/bin/sh",
                    "-c",
                    "rocminfo 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true",
                ],
                capture_output = True,
                text = True,
                env = env,
                timeout = 60,
            ).stdout.strip()
            sh = (
                _run_sh(
                    '_hsa_spoofed_physical_gfx "$SI" "$SP" 2>/dev/null',
                    env = dict(env, SI = inferred, SP = sh_probe),
                )
                if sh_probe
                else ""
            )

            # install_python_stack.py's call site: one entry per agent.
            with (
                patch.dict(os.environ, env, clear = False),
                patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(kfd)),
                patch.object(stack_mod, "_amd_smi_allowed", return_value = False),
                patch.object(stack_mod, "_safe_print"),
            ):
                for _stale in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES"):
                    if _stale not in env:
                        os.environ.pop(_stale, None)
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
                py_probe = stack_mod._detect_amd_gfx_codes(dedup = False)
                py = (
                    stack_mod._hsa_spoofed_physical_gfx(inferred or None, list(py_probe))
                    if py_probe
                    else None
                )

            shapes += 1
            corrections += bool(py)
            if (py or "") != sh:
                divergences.append(
                    {
                        "override": override,
                        "inferred": inferred,
                        "physical": physical,
                        "kfd": kfd,
                        "mask": mask,
                        "sh_probe": sh_probe.split("\n"),
                        "py_probe": py_probe,
                        "shell_said": sh,
                        "python_said": py,
                    }
                )
        assert not divergences, divergences[:5]
        if override == "11.0.0":
            # Guards against the sweep passing because neither side ever fires: 11.0.0
            # on a gfx1151 chassis is the reported case and MUST correct somewhere.
            assert corrections, f"no shape corrected across {shapes} shapes"


# ── Clearing the confirmed spoof from the launched runtime ───────────────────


class TestConfirmedSpoofIsClearedBeforeLaunch:
    """Routing the wheels is only half of #7331. ROCr rebuilds the agent from
    HSA_OVERRIDE_GFX_VERSION in every later process (libhsakmt's topology.c writes
    props->EngineId straight from the variable) and AMD's per-gfx index ships code
    objects for one arch only, so a runtime still reporting gfx1100 gets a gfx1151 wheel
    with no code for the device it sees and fails at the first allocation."""

    def test_reroute_clears_the_variable_it_disproved(self):
        probe = {}
        calls = _run_install(kfd_targets = ["gfx1151"], env_probe = probe)
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert probe["HSA_OVERRIDE_GFX_VERSION"] is None, probe

    def test_declined_spoof_keeps_the_users_override(self):
        """A real gfx1100 dGPU in a Ryzen AI Max chassis keeps generic wheels, where the
        override is the user's own business. Nothing was disproved, so nothing is taken
        away."""
        probe = {}
        calls = _run_install(
            kfd_targets = ["gfx1100"],
            reprobe_devices = ["gfx1100"],
            rocm_version = (7, 1),
            env_probe = probe,
        )
        assert "repo.amd.com" not in calls, calls
        assert probe["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0", probe

    def test_unspoofed_strix_reroute_leaves_the_environment_alone(self):
        """A gfx1151 host reporting itself honestly reroutes for the ordinary reason.
        No confirmed spoof, so the clear must not fire."""
        probe = {}
        calls = _run_install(
            gfx_devices = ("gfx1151",),
            env = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
            env_probe = probe,
        )
        assert "repo.amd.com/rocm/whl/gfx1151/" in calls, calls
        assert probe["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1", probe

    def test_install_sh_clears_it_on_the_same_branch(self):
        """The shell path is what matters for the reported flow: install.sh execs Unsloth
        from this very shell and install_python_stack.py runs as a grandchild, so a pop
        there cannot reach the launch."""
        source = _INSTALL_SH.read_text(encoding = "utf-8")
        start = source.find('if [ -n "$_strix_gfx" ] && _rocm_leaf_below')
        assert start != -1, "install.sh's Strix reroute branch moved"
        block = source[start : source.find("\n        fi\n", start)]
        assert "unset HSA_OVERRIDE_GFX_VERSION" in block, (
            "the Strix reroute must clear a corroborated spoof before this shell "
            "execs Unsloth, or the new wheels meet a runtime still claiming gfx1100"
        )
        assert '[ -n "$_spoof_physical" ]' in block, (
            "the clear must be guarded by the corroborated-spoof verdict, never "
            "applied to a host whose override was not disproved"
        )
        # Exactly one clear of the caller's own environment; the only other unset scopes
        # three variables inside the re-probe's subshell. Every other branch keeps the
        # override, which on generic wheels is what makes the GPU usable at all.
        _lasting = [
            ln.strip()
            for ln in source.splitlines()
            if ln.strip().startswith("unset HSA_OVERRIDE_GFX_VERSION")
        ]
        assert _lasting == ["unset HSA_OVERRIDE_GFX_VERSION"], _lasting


def _spoof_clear_guard() -> str:
    """The real `if` condition guarding the HSA_OVERRIDE_GFX_VERSION unset, out of install.sh."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    marker = "\n                unset HSA_OVERRIDE_GFX_VERSION\n"
    at = source.find(marker)
    assert at != -1, "the spoof-clearing unset is no longer in install.sh"
    line_start = source.rfind("\n            if ", 0, at)
    assert line_start != -1, "no guarding if for the spoof-clearing unset"
    return source[line_start + 1 : at].strip()


def test_no_torch_keeps_the_override_because_no_per_gfx_wheels_are_installed():
    """--no-torch must not clear the spoof, since it installs nothing to replace it.

    Clearing is only sound because native per-gfx wheels are going in on that branch.
    `--no-torch` (and the Intel Mac auto-detection, which sets the same SKIP_TORCH)
    reaches the reroute and installs no torch, so clearing there strands the host with
    the generic wheels it already had AND no override, its only source of usable kernels.

    Executes the guard as written rather than matching its text, so a rewrite that keeps
    the words and loses the behaviour still fails.
    """
    guard = _spoof_clear_guard()
    assert "SKIP_TORCH" in guard, guard

    for skip_torch, expected in (("false", "<cleared>"), ("true", "11.0.0")):
        out = _run_sh(
            f"{guard}\n"
            "    unset HSA_OVERRIDE_GFX_VERSION\n"
            "fi\n"
            'printf "%s\\n" "${HSA_OVERRIDE_GFX_VERSION:-<cleared>}"\n',
            env = {
                "SKIP_TORCH": skip_torch,
                "_spoof_physical": "gfx1151",
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            },
        )
        assert out.strip() == expected, f"SKIP_TORCH={skip_torch}: {out!r}"
