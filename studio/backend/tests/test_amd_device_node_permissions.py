# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An AMD device node this account cannot open must be named, not read as "no GPU".

Every AMD probe in this tree tests that ``/dev/kfd`` and ``/dev/dri/renderD*`` EXIST.
On a stock Linux distribution they are ``root:render`` mode 0660, so an account
outside that group passes all of them and then cannot open the device. HIP counts
zero devices and the Vulkan loader enumerates none, which is byte-for-byte what
owning no GPU looks like -- so Studio ran on CPU and told the user to install a
driver stack that was already loaded (#10466: a fresh Strix Halo install, fixed by
adding the account to the render and video groups).

Both messages under test used to describe a different host: the capability line
blamed a PyTorch mismatch and offered Repair installation, and the empty-probe
explanation blamed the Vulkan probe. Neither repair changes group membership.

No AMD hardware here, and the node paths are patched rather than created: a test
that had to mknod would need root, which is the one account this bug cannot reach.
"""

from __future__ import annotations

import builtins
import getpass
import grp
import inspect
import io
import json
import os
import pwd
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils.hardware import amd


_GPU_MASK_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


@pytest.fixture(autouse = True)
def _no_inherited_gpu_mask(monkeypatch):
    """No per-GPU selector unless a test sets one.

    An open render node stops being evidence of a usable path under a mask, so any of these
    inherited from the runner would decide a case the test never mentioned. This box
    exports CUDA_VISIBLE_DEVICES, and it silently answered for a control that was supposed
    to be testing an empty HIP mask. GPU_DEVICE_ORDINAL is here for the same reason and not
    because any test sets it: a ROCm or OpenCL environment exports it, the production rule
    reads it, and a test naming no mask would then be answered by the runner's own.
    """
    for _var in _GPU_MASK_VARS:
        monkeypatch.delenv(_var, raising = False)


@pytest.fixture(autouse = True)
def _the_account_this_process_runs_as(monkeypatch):
    """The repair commands name the account os.access answered for, so fix what that is.

    A stubbed passwd answer rather than the runner's own, which differs per machine. The
    environment fallback, for a uid with no passwd entry, has its own test.
    """
    # A real struct_passwd, not a SimpleNamespace: getpass.getuser() falls through to
    # pwd.getpwuid(os.getuid())[0] when none of LOGNAME/USER/LNAME/USERNAME is set, and
    # pytest calls it while building tmp_path. A non-subscriptable stub raises TypeError
    # there, which is not among the exceptions pytest catches, so on a runner with no
    # username in the environment every tmp_path test would die in fixture setup rather
    # than run.
    _record = pwd.struct_passwd(("ada", "x", os.getuid(), os.getgid(), "", "/home/ada", "/bin/sh"))
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _record)


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")


def _nodes(
    monkeypatch,
    *,
    present: list[str],
    openable: set[str],
    amd_owned: bool = True,
    vendor_readable: bool = True,
):
    """A host whose ``present`` nodes exist and whose ``openable`` subset can be opened.

    ``amd_owned`` is the vendor of the hardware behind those nodes, stubbed here and
    exercised for real in the two tests below it. ``vendor_readable`` is whether sysfs will
    say so: a container can map the node and hide the entry that names its vendor, and the
    two are different answers.
    """
    # Only the device-node enumeration. The module's other caller of the same helper walks
    # the Vulkan icd.d directories, and answering that one with a list of render nodes made
    # every loader question read as "no drivers at all" -- which is a verdict, not an
    # absence, so it would have passed silently.
    _real_glob = amd.glob.glob
    monkeypatch.setattr(
        amd.glob,
        "glob",
        lambda pattern: (
            [p for p in present if p.startswith("/dev/dri/renderD")]
            if pattern.startswith("/dev/dri/")
            else _real_glob(pattern)
        ),
    )
    monkeypatch.setattr(amd.os.path, "exists", lambda p: p in present)
    monkeypatch.setattr(amd.os, "access", lambda p, mode: p in openable)
    monkeypatch.setattr(amd, "_render_node_is_amd", lambda p: vendor_readable and amd_owned)
    # The vendor, not the verdict: _amd_render_node_exists reads it directly, so that an
    # unreadable entry can be told apart from one that named another vendor.
    monkeypatch.setattr(
        amd,
        "_render_node_vendor",
        lambda p: None if not vendor_readable else ("0x1002" if amd_owned else "0x10de"),
    )
    monkeypatch.setattr(amd, "_kfd_topology_has_an_amd_gpu", lambda: amd_owned)
    # These paths are patched rather than created, so stat cannot name their groups; say
    # so explicitly instead of leaving it to whether the runner happens to have a node at
    # the same path. The derivation itself is exercised in its own tests below.
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], [], []))


def test_a_node_this_user_cannot_open_is_reported(monkeypatch, linux):
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
    )
    assert amd.amd_nodes_closed_to_this_user() == ["/dev/kfd", "/dev/dri/renderD128"]


def test_a_host_whose_nodes_open_reports_nothing(monkeypatch, linux):
    """The control. Without it every assertion here also passes on a host with no AMD
    hardware at all, where the list is empty for a reason that is not this bug."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/kfd", "/dev/dri/renderD128"},
    )
    assert amd.amd_nodes_closed_to_this_user() == []
    assert amd.amd_node_permission_hint() is None


def test_a_host_with_no_amd_nodes_reports_nothing(monkeypatch, linux):
    _nodes(monkeypatch, present = [], openable = set())
    assert amd.amd_nodes_closed_to_this_user() == []


def test_an_nvidia_hosts_closed_render_nodes_are_not_reported(monkeypatch, linux):
    """The false positive this nearly shipped with. Render nodes are root:render for
    EVERY vendor, so the box this was written on -- 8 NVIDIA cards, an account outside
    the render group -- listed all eight and would have told a CUDA user to join the
    AMD groups. CUDA opens /dev/nvidia* and does not care."""
    _nodes(
        monkeypatch,
        present = ["/dev/dri/renderD128"],
        openable = set(),
        amd_owned = False,
    )
    assert amd.amd_nodes_closed_to_this_user() == []
    assert amd.amd_node_permission_hint() is None


def test_the_vendor_is_read_from_sysfs(monkeypatch):
    """The reader itself, since the test above stubs it. sysfs is world-readable, so
    ownership is answerable without the access being tested for."""
    real_open = builtins.open

    def _fake(path, *a, **k):
        if str(path) == "/sys/class/drm/renderD128/device/vendor":
            return io.StringIO("0x1002\n")
        if str(path) == "/sys/class/drm/renderD129/device/vendor":
            return io.StringIO("0x10de\n")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", _fake)
    assert amd._render_node_is_amd("/dev/dri/renderD128") is True
    assert amd._render_node_is_amd("/dev/dri/renderD129") is False
    assert amd._render_node_is_amd("/dev/dri/renderD130") is False  # unreadable


def test_the_probe_is_linux_only(monkeypatch):
    """macOS and Windows have no render nodes, and ``os.access`` on Windows answers
    for a permission model this message does not describe."""
    monkeypatch.setattr(amd.platform, "system", lambda: "Windows")
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    assert amd.amd_nodes_closed_to_this_user() == []


def test_the_hint_names_the_nodes_the_groups_and_the_account(monkeypatch, linux):
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "/dev/kfd" in hint
    assert "usermod -a -G render,video ada" in hint


def _no_passwd_entry(monkeypatch):
    """A uid the passwd database does not know, which is where the environment is read."""

    def _missing(_uid):
        raise KeyError(_uid)

    monkeypatch.setattr(pwd, "getpwuid", _missing)


def test_a_uid_with_no_passwd_entry_is_not_given_a_usermod(monkeypatch, linux):
    """`docker run --user 1234` leaves the uid with no passwd entry while USER commonly
    still says root. usermod against that name succeeds, changes an identity nothing is
    running as, and leaves the nodes exactly as shut -- so the repair here is the
    container's group wiring, not an account.

    Fails before the fix, which fell back to USER and then to a literal $USER."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "root")
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], [], []))
    _no_passwd_entry(monkeypatch)
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G" not in hint and "root" not in hint
    assert "--group-add render" in hint


def test_an_account_the_system_knows_still_gets_the_command(monkeypatch, linux):
    """The control: a uid with a passwd entry is an account usermod can name, and that is
    the repair on every ordinary host. Without it the fix could be "never prescribe
    usermod", which removes what #10466 asked for."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "root")
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], [], []))
    monkeypatch.setattr(amd, "_repair_account", lambda: "ada")
    hint = amd.amd_node_permission_hint()
    assert "sudo usermod -a -G render ada" in hint
    assert "--group-add" not in hint


def test_an_unnamed_gid_under_that_uid_drops_the_groupadd_half_too(monkeypatch, linux):
    """The unnamed-GID repair is a groupadd AND a usermod, and the second half needs the
    same account the first branch does. With no passwd entry the container flag is the
    whole repair, so printing the pair would be two commands that cannot both work."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "root")
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    _no_passwd_entry(monkeypatch)
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G" not in hint and "groupadd -g" not in hint
    assert "--group-add 993" in hint


def test_a_node_that_cannot_be_stat_ed_is_skipped(monkeypatch, linux):
    """A probe that raises must not break a load; the caller is a diagnostic."""

    def _boom(_p):
        raise OSError("stale handle")

    monkeypatch.setattr(amd.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(amd.os.path, "exists", _boom)
    assert amd.amd_nodes_closed_to_this_user() == []


def test_read_only_access_is_not_enough(monkeypatch, linux):
    """HIP and the Vulkan loader both open the node read-write; a node that only
    reads is still unusable, and answering "fine" here would restore the silence."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd.os, "access", lambda p, mode: mode == os.R_OK)
    assert amd.amd_nodes_closed_to_this_user() == ["/dev/kfd"]


def test_the_capability_message_names_the_permission_not_a_torch_mismatch(monkeypatch, linux):
    """Fails before the fix: the old message blamed PyTorch and offered Repair
    installation, which cannot add an account to a group."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+rocm7.0"),
    )
    assert "usermod -a -G render,video ada" in message
    assert "Repair installation" not in message


def test_the_empty_probe_explanation_names_the_permission(monkeypatch, linux):
    """The load-time line. It ran ahead of the Vulkan branch on purpose: a closed
    render node is the reason UNDERNEATH "the Vulkan probe reported no device", and
    that phrasing sends the user after a driver that is already working."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "usermod -a -G render,video ada" in reason
    assert "the Vulkan probe reported no device" not in reason


def test_the_empty_probe_explanation_is_unchanged_when_the_nodes_open(monkeypatch, linux):
    """Control: a Vulkan host with openable nodes must keep its own reason."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    assert LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server") == (
        "the Vulkan probe reported no device"
    )


def test_the_capability_message_is_unchanged_when_the_nodes_open(monkeypatch, linux):
    """The control for the one above: on any other unusable-GPU host the existing
    PyTorch wording has to survive, or this fix trades one wrong answer for another."""
    from utils.hardware import hardware

    # A render node too, and open: "the nodes open" has to mean every node this host
    # needs, or the control describes a container missing /dev/dri and the message it
    # gets back is about that instead.
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/kfd", "/dev/dri/renderD128"},
    )
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+rocm7.0"),
    )
    assert "Repair installation" in message
    assert "usermod" not in message


def test_the_hint_is_rocm_specific_when_only_kfd_is_closed(monkeypatch, linux):
    """Vulkan never opens /dev/kfd, so a closed one does not stop every backend.
    Claiming it did sent a Vulkan user with an unrelated failure after ROCm groups."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    hint = amd.amd_node_permission_hint()
    assert "ROCm cannot use" in hint
    assert "no GPU backend" not in hint


def test_the_hint_covers_every_backend_when_a_render_node_is_closed(monkeypatch, linux):
    """The pair to the test above: HIP and the Vulkan loader both open this one."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    assert "no GPU backend can use" in amd.amd_node_permission_hint()


def test_a_vulkan_only_caller_is_not_answered_with_a_closed_kfd_node(monkeypatch, linux):
    """``needs_kfd = False`` is a Vulkan binary saying a closed KFD node is not its
    problem. Without it the render-node-open case still returned a hint."""
    # With an open render node, so the only thing wrong on this host is the closed KFD
    # node the caller has just said it does not need. Without one the answer is a real
    # Vulkan blocker rather than the None this asserts.
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/dri/renderD128"},
    )
    assert amd.amd_node_permission_hint(needs_kfd = False) is None
    assert amd.amd_node_permission_hint() is not None


def test_a_vulkan_caller_is_still_answered_about_a_closed_render_node(monkeypatch, linux):
    """The control for the one above, so the narrowing cannot silence the real case."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    assert amd.amd_node_permission_hint(needs_kfd = False) is not None


def test_the_vulkan_probe_keeps_its_own_reason_when_only_kfd_is_closed(monkeypatch, linux):
    """End to end through the caller: a Vulkan binary on a host whose render node
    opens must not be told about ROCm's node."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/dri/renderD128"},
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    assert LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server") == (
        "the Vulkan probe reported no device"
    )


def test_a_rocm_binary_is_still_told_about_the_closed_kfd_node(monkeypatch, linux):
    """The control: the same host, a non-Vulkan binary, and the hint must survive."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/dri/renderD128"},
    )
    monkeypatch.setenv("USER", "ada")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"hip"})),
    )
    assert "usermod -a -G render,video ada" in (
        LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    )


def test_an_nvidia_mismatch_keeps_the_pytorch_message(monkeypatch, linux):
    """A hybrid host whose NVIDIA card raised the verdict while an AMD node happens
    to be closed. Joining the render group repairs nothing there, and reinstalling
    the GPU build might, so the existing diagnosis has to survive."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"nvidia"}))
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+cu130"),
    )
    assert "Repair installation" in message
    assert "usermod" not in message


def test_a_hybrid_host_whose_amd_card_raised_it_still_gets_the_permission_hint(monkeypatch, linux):
    """The pair to the test above, differing only in the recorded vendor."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(
        hardware,
        "CHAT_ONLY_MISMATCH_VENDORS",
        frozenset({"amd", "nvidia"}),
    )
    # Stubbed rather than read off the box. The workspace this was written on happens
    # to carry a ROCm torch, so the assertion below was passing for the host's reason
    # instead of the test's; on a hybrid host the hint needs the install to target AMD,
    # and a venv that asked for ROCm and got a CPU wheel is exactly this verdict.
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: False)
    # And the third of the trio, for the same reason: it reads whatever torch happens to
    # be installed beside the test, so leaving it live made this pass or fail on the
    # runner's wheel rather than on the host the test describes.
    monkeypatch.setattr(hardware, "_torch_reports_another_vendors_runtime", lambda: False)
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cpu_build", None),
    )
    assert "usermod -a -G render,video ada" in message


def test_a_cuda_only_build_is_not_sent_after_the_amd_render_group(monkeypatch, linux):
    """A hybrid host whose CUDA build enumerated nothing for its own reasons.

    Every AMD node is closed, so the hint is available and would be returned by any
    build that could use the card. This one cannot, so the mask diagnosis below has
    to survive.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"cuda"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "usermod" not in reason
    assert "CUDA_VISIBLE_DEVICES" in reason


def test_a_build_whose_backend_cannot_be_read_still_gets_the_hint(monkeypatch, linux):
    """The control for the test above, and the reason it names CUDA rather than
    "not ROCm": an install this probe cannot read must not lose the diagnosis."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset()),
    )
    assert "usermod -a -G render,video ada" in (
        LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    )


def test_a_cpu_wheel_beside_a_closed_node_is_told_to_do_both(monkeypatch, linux):
    """Opening the node leaves a CPU-only wheel with no GPU path, so the reinstall
    step has to survive the permission hint rather than being replaced by it."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cpu_build", None),
    )
    assert "CPU-only build" in message
    assert "Repair installation" in message
    assert "usermod -a -G render,video ada" in message


def test_a_gpu_wheel_beside_a_closed_node_is_told_only_the_permission(monkeypatch, linux):
    """The pair: a ROCm wheel that cannot open a device is fully explained by the
    node, so the reinstall advice would send the user after the wrong repair."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+rocm7.0"),
    )
    assert "usermod -a -G render,video ada" in message
    assert "Repair installation" not in message


def _kernel_stack_hint_runs(
    closed_nodes: str,
    *,
    route: bool = True,
    nvidia: bool = False,
) -> bool:
    """Whether install.sh's missing-kernel-stack branch fires for this closed set.

    The guard is lifted out of install.sh by text rather than restated here: a test
    that restated it would pass whatever the installer went on to say. Only the
    condition is taken, and its two probes are stubbed true so the answer depends on
    nothing but the closed-node reasoning.
    """
    lines = _install_sh_lines()
    # Anchored on the part of the condition this change does NOT touch, then walked
    # back over the continuations to the "if". Anchoring on the new closed-node text
    # instead would make the control vacuous: reverting the guard would stop the
    # extraction finding anything, and "the text changed" would read as "the
    # behaviour changed".
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    guard = "\n".join(line.strip() for line in lines[start : end + 1])
    script = "\n".join(
        [
            "_has_amd_rocm_gpu() { return 1; }",  # ROCm cannot see the card
            "_amd_gpu_present_via_pci() { return 0; }",  # but the PCI bus can
            # Carried by the guard now that it sits after the case rather than inside the
            # */cpu arm, which supplied them. Set here so the answer still depends only on
            # the closed-node reasoning.
            "SKIP_TORCH=false",
            "OS=linux",
            # The run-scope predicate the guard now asks in place of a bare SKIP_TORCH
            # test. Lifted, not stubbed, so this arm goes through the installer's own rule.
            *_run_scope_defs(lines, nvidia = nvidia),
            # The route gate. True by default for the same reason the two probes are
            # stubbed: this harness asks about the closed-node reasoning, and the route
            # has its own tests below.
            f"_amd_node_diag_route={'true' if route else 'false'}",
            guard,
            "    echo FIRED",
            "fi",
        ]
    )
    # The set arrives as an exported variable rather than a generated assignment: a
    # repr() inside shell single quotes turns the newline separating two nodes into a
    # literal backslash-n, which reads as one unmatched line and looks exactly like the
    # suppression failing.
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        check = True,
        env = {**os.environ, "_closed_amd_nodes": closed_nodes},
    )
    return "FIRED" in out.stdout


def test_a_closed_kfd_node_suppresses_the_kernel_stack_hint():
    """/dev/kfd existing is the evidence the stack is already loaded, so telling the
    user to install one cannot help; the group advice printed after the case is the
    repair."""
    assert not _kernel_stack_hint_runs("/dev/kfd")
    assert not _kernel_stack_hint_runs("/dev/kfd\n/dev/dri/renderD128")


def test_a_missing_kfd_node_keeps_the_kernel_stack_hint():
    """The case the suppression must not swallow: no /dev/kfd at all, and a render
    node this account cannot open. No amount of group membership creates /dev/kfd,
    so both diagnoses apply and both have to print."""
    assert _kernel_stack_hint_runs("/dev/dri/renderD128")


def test_the_hint_still_runs_on_a_host_with_nothing_closed():
    """The negative control: the branch's original behaviour is untouched."""
    assert _kernel_stack_hint_runs("")


def test_a_hybrid_host_running_cuda_torch_keeps_the_pytorch_message(monkeypatch, linux):
    """A supported AMD card qualifies for the mismatch whatever wheel is installed, so
    a hybrid host records both vendors even when the verdict is about the NVIDIA card.
    No group membership makes a CUDA wheel use the AMD card, so the reinstall advice is
    the right one there."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(
        hardware,
        "CHAT_ONLY_MISMATCH_VENDORS",
        frozenset({"amd", "nvidia"}),
    )
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: False)
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+cu130"),
    )
    assert "Repair installation" in message
    assert "usermod" not in message


def test_an_amd_only_host_needs_no_runtime_evidence(monkeypatch, linux):
    """The control, and the #10466 host: AMD is the only vendor that qualified, so the
    verdict can only be about it and the wheel's own tag adds nothing."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: False)
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+rocm7.0"),
    )
    assert "usermod -a -G render,video ada" in message


def test_a_cuda_plus_vulkan_build_is_treated_as_cuda(monkeypatch, linux):
    """_is_vulkan_backend defers such a build to CUDA, so it is a CUDA install for
    every other purpose and must be one here too. Requiring CUDA to be the ONLY shipped
    library left this layout uncovered."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"cuda", "vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "usermod" not in reason
    assert "CUDA_VISIBLE_DEVICES" in reason


def test_a_cpu_only_llama_build_is_not_sent_after_the_groups(monkeypatch, linux):
    """A build with no GPU library cannot offload to any card, so its empty probe is
    not a permission problem and the groups cannot change it."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"cpu", "base"})),
    )
    assert "usermod" not in LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_a_mask_is_reported_alongside_the_permission_hint(monkeypatch, linux):
    """Two independent blockers need two fixes. The groups do not clear a visibility
    mask, so returning the hint alone hid the half the user also has to undo."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"hip"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "usermod -a -G render,video ada" in reason
    assert "HIP_VISIBLE_DEVICES='-1'" in reason


def test_no_mask_leaves_the_hint_alone(monkeypatch, linux):
    """The control: the sentence must not grow a trailing clause on a host with no mask
    set, which is every host the #10466 wording was written for."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"hip"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.endswith("sudo usermod -a -G render,video ada")


def test_the_hint_says_so_when_kfd_does_not_exist_at_all(monkeypatch, linux):
    """A closed render node is real and the groups open it, but no membership creates
    /dev/kfd, so a ROCm caller is not repaired by them alone and both sentences print.

    The kernel-stack wording this used to assert was wrong, and asserting it is what
    kept it: the sentence is reachable only once the KFD topology names an AMD GPU, and
    that topology is the amdkfd driver's own sysfs, so the stack is already loaded on
    every host that can reach it. install.sh's kernel-stack branch is gated the other
    way round, on the topology being ABSENT."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render,video ada" in hint
    assert "/dev/kfd" in hint
    assert "kernel stack" not in hint


def test_a_closed_but_present_kfd_node_says_nothing_about_the_kernel_stack(monkeypatch, linux):
    """The control: /dev/kfd exists, so the stack is loaded and the groups are the whole
    repair. Telling this user to install it would be the #10466 mistake."""
    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    assert "kernel stack" not in amd.amd_node_permission_hint()


def test_a_vulkan_caller_is_not_told_about_a_kernel_stack_it_does_not_need(monkeypatch, linux):
    """Vulkan never opens /dev/kfd, so its absence is not that caller's problem."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    assert "kernel stack" not in amd.amd_node_permission_hint(needs_kfd = False)


def test_the_repair_names_the_groups_the_closed_nodes_belong_to(monkeypatch, linux):
    """render and video are the usual pair, not a universal truth. A container is passed
    the host's numeric gids by --group-add and has no matching group NAMES inside it, a
    minimal distribution can ship no render group, and a node left root:root by a udev
    rule is not fixed by joining either. The command has to name the groups that own the
    files that were refused, or it is a repair that cannot work.

    Fails before the fix, which hard-coded render,video whatever the nodes said."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
    )
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda paths: (["kfd", "gpu"], [], [], [], [], [], [])
    )
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G kfd,gpu ada" in hint
    assert "render,video" not in hint


def test_a_single_owning_group_is_not_pluralised(monkeypatch, linux):
    """A host where both nodes belong to one group gets one group named, and the sentence
    has to agree with the command rather than saying "groups" over a single name."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render ada" in hint
    assert "render group and then log out" in hint


def test_unreadable_nodes_fall_back_to_the_documented_pair(monkeypatch, linux):
    """The control: the derivation is best effort, so a host whose nodes cannot be stat'd
    must still get advice rather than an empty -G argument, and that advice is the pair
    the AMD documentation names."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    assert "usermod -a -G render,video ada" in amd.amd_node_permission_hint()


def _stat_nodes(monkeypatch, modes: dict, names: dict):
    """Stub os.stat and grp for the node set ``modes`` maps to (gid, mode).

    Paths outside the map raise, which covers the vanished-node arm. The stub takes
    ``follow_symlinks`` because pytest itself stats files while this patch is in force,
    and a two-argument lambda takes the whole session down with it rather than failing
    the test that installed it.
    """

    def _stat(path, *, follow_symlinks = True):
        if str(path) not in modes:
            raise OSError("gone")
        _entry = modes[str(path)]
        gid, mode = _entry[0], _entry[1]
        # Real device nodes are root-owned, and POSIX consults the owner class first, so a
        # fake without st_uid would take the owner branch on whatever uid the runner has.
        uid = _entry[2] if len(_entry) > 2 else 0
        # ... and root IS the uid a CI runner often has, which would send every test below
        # down that same owner branch and assert nothing about the group classification it
        # names. No caller here is testing owner precedence -- the tests that do build real
        # files under tmp_path -- so the synthetic owner is simply moved off the runner.
        if uid == amd.os.getuid():
            uid += 1
        return type("st", (), {"st_gid": gid, "st_mode": mode, "st_uid": uid})()

    def _getgrgid(gid):
        if gid not in names:
            raise KeyError(gid)
        return type("gr", (), {"gr_name": names[gid]})()

    monkeypatch.setattr(amd.os, "stat", _stat)
    monkeypatch.setattr(grp, "getgrgid", _getgrgid)


def test_the_group_derivation_reads_the_node(monkeypatch):
    """The helper itself, since every test above stubs it. Two nodes owned by one group
    name it once, and order is first-seen so the command reads like the node list."""
    _stat_nodes(
        monkeypatch,
        {
            "/dev/kfd": (44, 0o660),
            "/dev/dri/renderD128": (44, 0o660),
            "/dev/dri/renderD129": (39, 0o660),
        },
        {44: "video", 39: "render"},
    )
    assert amd._groups_that_own(["/dev/dri/renderD129", "/dev/kfd", "/dev/dri/renderD128"]) == (
        ["render", "video"],
        [],
        [],
        [],
        [],
        [],
        [],
    )


def test_a_gid_with_no_group_entry_is_reported_rather_than_prescribed(monkeypatch):
    """The container case docker/run.sh documents: --group-add passes the host's numeric
    gids, and inside the container no group entry matches them.

    Fails before the fix, which returned the bare number for usermod to consume. shadow
    4.13 answers ``group '993' does not exist`` and exits 6 on that command, verified on
    this host, so the number belongs in a sentence rather than in the -G argument."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660)}, {})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [993], [], [], [], [], [])


def test_a_node_whose_own_group_cannot_open_it_is_not_a_membership_problem(monkeypatch):
    """A udev rule leaving a node root:render 0600 denies the group as well, so joining
    render opens nothing. Read the mode before naming the group, or the repair is a
    command that runs, succeeds, and changes nothing.

    Fails before the fix, which read st_gid alone and would have prescribed render."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (44, 0o600)}, {44: "render"})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [], ["/dev/kfd"], [], [], [], [])


def test_group_read_without_write_is_not_enough(monkeypatch):
    """Its boundary: HIP and the Vulkan loader both open the node read-write, which is
    the bar the probe itself applies, so 0640 is still not a joinable group."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (44, 0o640)}, {44: "render"})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [], ["/dev/kfd"], [], [], [], [])


def test_a_node_that_cannot_be_stat_contributes_nothing(monkeypatch):
    """And the failure mode that must not raise: diagnostics run on the path where things
    are already wrong, so a node that vanished between the probe and the message drops
    out rather than taking the whole hint down."""
    _stat_nodes(monkeypatch, {"/dev/dri/renderD128": (44, 0o660)}, {44: "video"})
    assert amd._groups_that_own(["/dev/kfd", "/dev/dri/renderD128"]) == (
        ["video"],
        [],
        [],
        [],
        [],
        [],
        [],
    )


# The sentence the kernel-stack branch prints. The harnesses that lift that branch anchor
# on it rather than on either condition, because neither condition is stable enough: the
# mapping one is what an earlier change edited, and a revert that stopped the extraction
# finding anything would read "the text changed" as "the behaviour changed". The predicate
# _amd_gpu_present_via_pci is named twice in this installer, so it is not an anchor either.
_PCI_SENTENCE = "An AMD GPU is on the PCI bus but ROCm cannot see it"


def _install_sh_lines() -> "list[str]":
    """install.sh, split into lines. Every harness below lifts what it needs out of this."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    return install_sh.read_text(encoding = "utf-8").splitlines()


def _install_sh_if_above(lines: "list[str]", i: int) -> int:
    """Walk back from a line inside a condition to the `if` opening its block.

    A multi-line condition puts the `if` and its last test on different lines, so requiring
    both on one line stopped finding anything the moment a gate was added -- and a harness
    that finds nothing raises here rather than silently testing a shorter script.
    """
    while not lines[i].lstrip().startswith("if "):
        i -= 1
    return i


def _install_sh_anchor(lines: "list[str]", text: str) -> int:
    """The index of the line containing ``text``."""
    return next(i for i, line in enumerate(lines) if text in line)


def _install_sh_if(lines: "list[str]", tail: str) -> int:
    """The index of the `if` opening the block whose condition ENDS with ``tail``."""
    return _install_sh_if_above(
        lines, next(j for j, line in enumerate(lines) if line.rstrip().endswith(tail))
    )


def _shell_fn(lines: "list[str]", name: str) -> str:
    """One function definition lifted out of install.sh, by brace depth.

    The harnesses below used to slice from a definition down to the block under test,
    which worked only while the two were adjacent. A shell function has to be defined
    before the line that calls it, so the run-scope predicates now sit above the diagnoses
    they gate and each is lifted by name instead. A miss raises rather than returning an
    empty string: an undefined function would leave the block under test failing for a
    reason that has nothing to do with the case.
    """
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated {name}() in install.sh")


def _run_scope_defs(lines: "list[str]", *, nvidia: bool = False) -> "list[str]":
    """The run-scope predicates the install.sh harnesses below share.

    Lifted rather than restated so every arm goes through the installer's own rule.
    _is_pip_rocm_family_leaf comes with _torch_opens_amd_nodes, which classifies a ROCm
    index through it, and _shell_quote with anything that pastes a command: without it
    every interpolated name comes back EMPTY and the arm reads as a command naming nobody.
    _has_usable_nvidia_gpu is the exception, stubbed because the real one runs nvidia-smi
    and would answer from the runner's own hardware -- false by default, so each arm reads
    as the AMD-only host it was written for.
    """
    return [
        _shell_fn(lines, "_torch_index_url_leaf"),
        _shell_fn(lines, "_is_pip_rocm_family_leaf"),
        _shell_fn(lines, "_torch_opens_amd_nodes"),
        f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
        _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
        _shell_fn(lines, "_run_may_open_kfd"),
        _shell_fn(lines, "_shell_quote"),
    ]


def _install_sh_run(script: str, *, env: "dict | None" = None) -> str:
    """Run a lifted script under bash and return its stdout, failing with its own stderr."""
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _install_sh_env(
    closed_nodes: str,
    env_user: str,
    backend: "str | None",
    torch_index: str = "https://download.pytorch.org/whl/rocm6.4",
) -> dict:
    """The environment install.sh reads: the closed set, the account, the request, and the
    torch index, which the run-scope predicates read to tell a CPU wheel from a ROCm one.
    Defaulted to a ROCm index so every existing case keeps the run it was written for."""
    env = {
        **os.environ,
        "_closed_amd_nodes": closed_nodes,
        "USER": env_user,
        "TORCH_INDEX_URL": torch_index,
    }
    env.pop("UNSLOTH_LLAMA_CPP_BACKEND", None)
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
    return env


def _install_sh_hint(
    closed_nodes: str,
    *,
    render_present: bool = True,
    amd_present: bool = True,
    self_uid: str = "4242",
    self_gids: str = "65534",
    render_open: bool = False,
    repairs: "str | None" = None,
    skip_torch: bool = False,
    backend: "str | None" = None,
    torch_index: str = "https://download.pytorch.org/whl/rocm6.4",
    env_user: str = "ada",
    id_user: "str | None" = "ada",
    nvidia: bool = False,
) -> str:
    """The installer's closed-node message, run for a given closed set.

    Lifted from install.sh rather than restated, and the whole block rather than a
    condition, because the thing under test is the sentence it prints. substep is stubbed
    to plain echo; the node list comes in through the environment, since embedding it in
    the script would put a literal backslash-n inside shell quotes and turn two nodes into
    one unmatched line.
    """
    lines = _install_sh_lines()
    start = _install_sh_if(lines, '[ -n "$_closed_amd_nodes" ]; then')
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[start : end + 1])

    helper = _shell_fn(lines, "_amd_node_repairs")

    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            'C_WARN=""',
            # Stubbed rather than lifted: the real one reads /sys and /dev, so leaving it
            # live would make every arm depend on the runner's own hardware.
            f"_amd_render_node_present() {{ return {0 if render_present else 1}; }}",
            # A real device node is root-owned; a tmp_path node standing in for one belongs
            # to the runner, and the installer stops at the owner class when those match.
            # Stubbed so the arms below choose which case they are testing.
            # Both spellings: the owner-class test asks `id -u`, the repair command asks
            # `id -un`, and a stub answering one for the other names a uid as an account.
            # id_user None is a uid with no passwd entry, where the real `id -un` FAILS:
            # the ordinary shape of `docker run --user 1234`, and the case the container
            # repair exists for.
            (
                # printf with the value single-quoted, not echo: a name carrying a
                # backslash is de-escaped by the stub itself otherwise, and the arm testing
                # how such a name is QUOTED then never sees one.
                f'id() {{ case "$1" in -un) printf %s\\\\n {shlex.quote(id_user or "")} ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
                if id_user is not None
                else f'id() {{ case "$1" in -un) return 1 ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
            ),
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if amd_present else 1}; }}",
            # Stubbed for the same reason as _amd_render_node_present: the real one reads
            # /dev and /sys, so a live one would answer from the runner's own hardware.
            f"_an_amd_render_node_is_open() {{ return {0 if render_open else 1}; }}",
            # The route the diagnoses are gated on; the gate has its own tests below.
            "_amd_node_diag_route=true",
            "OS=linux",
            # The run-scope predicate the block now asks. Lifted rather than stubbed, so
            # the default arms below go through the same rule the installer applies.
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            # Stubbed like _amd_render_node_present: the real one runs nvidia-smi, so a
            # live one would answer from the runner's own hardware. False by default, so
            # every arm below reads as the AMD-only host it was written for.
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            # The block also asks which nodes THIS run opens, to name the right --device
            # pair, so the predicate has to exist before the span that calls it.
            _shell_fn(lines, "_torch_index_url_leaf"),
            # _torch_opens_amd_nodes classifies a ROCm index through this,
            # so lifting one without the other measures a missing function.
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_kfd"),
            # The block quotes every name it interpolates into a pasted command through
            # this. Lifted rather than stubbed: without it the substitutions come back
            # EMPTY and the arms below read as commands that name nobody.
            _shell_fn(lines, "_shell_quote"),
            # The real derivation by default. An override stands in only where the case
            # cannot be built on disk -- a node whose GID has no entry in the group
            # database -- and _amd_node_repairs has its own tests either way.
            helper if repairs is None else f"_amd_node_repairs() {{ printf '%s\\n' '{repairs}'; }}",
            block,
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        env = _install_sh_env(closed_nodes, env_user, backend, torch_index),
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def _a_node_a_membership_would_open(tmp_path, *, mode: int = 0o660):
    """A fixture node whose owning group is one the rule calls JOINABLE, and its name.

    A tmp_path file gets the runner's primary group, and on a root CI runner that group is
    root -- which both halves deliberately classify as privileged and refuse to prescribe.
    The arms below then assert the privileged sentence instead of the derivation they are
    named for, so the group is CHOSEN rather than inherited. root may chgrp to any group;
    an ordinary account may only use one it belongs to, and skips when every one of those
    is privileged, since there is no node it could build that would test anything.
    """
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(mode)
    _candidates = (
        [_g.gr_gid for _g in grp.getgrall()]
        if os.geteuid() == 0
        else [os.getgid(), *os.getgroups()]
    )
    for _gid in _candidates:
        try:
            _name = grp.getgrgid(_gid).gr_name
        except KeyError:
            continue
        if _gid == 0 or _name in amd._PRIVILEGED_GROUPS:
            continue
        try:
            os.chown(node, -1, _gid)
        except OSError:
            continue
        return node, _name
    pytest.skip("every group this account can use is one the rule refuses to prescribe")


def test_the_group_fixture_never_hands_back_a_group_the_rule_refuses(tmp_path, monkeypatch):
    """The guard for it, since a fixture that quietly picks a privileged group does not
    fail -- it makes the arms below assert the wrong sentence, which is how this was
    found. Standing in for the root runner by making the group this account would
    otherwise inherit privileged, so the search has to move off it."""
    monkeypatch.setattr(
        amd,
        "_PRIVILEGED_GROUPS",
        frozenset(amd._PRIVILEGED_GROUPS | {grp.getgrgid(os.getgid()).gr_name}),
    )
    _node, _group = _a_node_a_membership_would_open(tmp_path)
    assert _group not in amd._PRIVILEGED_GROUPS


def test_the_installer_names_the_group_the_node_actually_has(tmp_path):
    """The shell half of the same item, and the only arm of it that reads a real file:
    the message must name the group that owns the node it just refused. Fails before the
    fix, which printed render,video for every host.

    The expected group is read with the same stat the installer uses rather than assumed,
    since a test runner's primary group is not knowable in advance -- but asserting it is
    NOT render,video is what makes that comparison mean something."""
    # 0660, the mode a real render node has: the installer now reads the mode as well as
    # the group, and a default 0644 is a node no membership opens.
    node, _group = _a_node_a_membership_would_open(tmp_path)
    owner = subprocess.run(
        ["stat", "-c", "%G", str(node)],
        capture_output = True,
        text = True,
    ).stdout.strip()
    out = _install_sh_hint(str(node))
    assert f"usermod -a -G {owner} ada" in out
    if owner not in ("render", "video"):
        assert "render,video" not in out


def test_the_installer_falls_back_when_the_node_is_gone(tmp_path):
    """The control: a path that cannot be stat'd still has to produce advice, and a
    fallback that produced an empty -G argument would be worse than the hard-coded pair
    it replaced."""
    out = _install_sh_hint(str(tmp_path / "renderD128"))
    assert "usermod -a -G render,video ada" in out


def _reason_with_mask(monkeypatch, var: str, value: str, backends: set) -> str:
    """The empty-probe reason on a closed-node host carrying one visibility mask."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setenv("USER", "ada")
    for other in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(other, raising = False)
    monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset(backends)),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda _b: backends == {"vulkan"}),
    )
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_a_selector_that_still_exposes_a_device_is_not_a_second_blocker(monkeypatch, linux):
    """HIP_VISIBLE_DEVICES=0 names a device rather than hiding one, so it is not why the
    probe came back empty and clearing it changes nothing. Reported as a second blocker
    it sends the user after a fix that cannot help, on top of the one that can.

    Fails before the fix, which listed every variable that was merely SET."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "0", {"hip"})
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_a_mask_that_hides_everything_is_still_reported(monkeypatch, linux):
    """The control that keeps the test above honest: -1 names no device, so that host
    really does need both fixes and must still be told both.

    Not an EMPTY value, which is the one thing clr's parser is never entered on: the
    guard is on the first byte, so an empty HIP mask is not a filter and is not the
    variable clr reads either."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "-1", {"hip"})
    assert "usermod -a -G render,video ada" in reason
    assert "HIP_VISIBLE_DEVICES='-1'" in reason


def test_a_negative_first_entry_hides_everything(monkeypatch, linux):
    """CUDA and HIP parse the list left to right and stop at the first entry that names
    no device, so -1 leading the list leaves nothing enumerated."""
    reason = _reason_with_mask(monkeypatch, "CUDA_VISIBLE_DEVICES", "-1", {"hip"})
    assert "CUDA_VISIBLE_DEVICES='-1'" in reason
    assert "visibility mask is also in force" in reason


def test_a_leading_valid_entry_survives_a_later_invalid_one(monkeypatch, linux):
    """And its control: 0,-1 stops at the -1 but has already exposed GPU 0, so the mask
    is not the blocker. Without this the fix could be "any minus sign anywhere hides
    everything", which passes the test above and is wrong."""
    reason = _reason_with_mask(monkeypatch, "CUDA_VISIBLE_DEVICES", "0,-1", {"hip"})
    assert "visibility mask" not in reason


def test_a_vulkan_build_is_not_told_about_a_mask_it_never_reads(monkeypatch, linux):
    """A Vulkan-only install reads none of these four, so an inherited HIP or CUDA mask
    is not a blocker for it at any value -- the render node it cannot open is."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "-1", {"vulkan"})
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_the_same_hiding_mask_still_counts_for_a_hip_build(monkeypatch, linux):
    """The control for the arm above, one backend apart: the identical environment must
    still report the mask when the install is one that actually reads it."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "-1", {"hip"})
    assert "visibility mask is also in force" in reason


def test_an_unnamed_gid_is_not_handed_to_usermod(monkeypatch, linux):
    """A GID with no group entry is never the -G argument. usermod -a -G takes names only:
    shadow 4.13 answers ``group '993' does not exist`` and exits 6, run live on this host to
    check rather than read out of the man page.

    Fails before the fix, which put the bare number in the -G argument. The assertion is on
    the NUMBER rather than on the command: the repair now names usermod on purpose, after a
    groupadd that gives the GID a name, and asserting the command itself is absent would
    forbid the half that makes the account a member."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G 993" not in hint
    assert "--group-add 993" in hint


def test_a_joinable_group_beside_an_unnamed_gid_is_still_prescribed(monkeypatch, linux):
    """The control that keeps the suppression narrow: a host with one node in a real
    group and another in an unnamed one can still fix half of it by joining, so the
    command has to survive and name only the group that works."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
    )
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda paths: (["render"], [993], [], [], [], [], [])
    )
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render ada" in hint
    assert "GID 993" in hint


def test_a_node_no_membership_opens_is_not_answered_with_usermod(monkeypatch, linux):
    """A udev rule leaving the node root:render 0600 denies its own group, so joining
    render runs, succeeds, and opens nothing. The repair there is the rule.

    Fails before the fix, which named the owning group whatever the mode said."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda paths: ([], [], ["/dev/kfd"], [], [], [], [])
    )
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G" not in hint
    assert "udev rule" in hint


def test_a_host_whose_nodes_could_not_be_read_still_gets_the_documented_pair(monkeypatch, linux):
    """And the control for BOTH suppressions: all three lists empty means the nodes could
    not be stat'd at all, which is a detection miss rather than evidence that joining
    cannot work. Some advice beats none there, and it is the pair AMD documents."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    assert "usermod -a -G render,video ada" in amd.amd_node_permission_hint()


def _unusable_message(
    monkeypatch,
    detail,
    *,
    rocm_expected = False,
    hip_runtime = False,
):
    """The capability message on an AMD-only host with a closed node, for a given wheel."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    # Pinned rather than inherited: both probes answer for the venv this suite runs in,
    # so leaving them live would make the arms depend on the test host's torch.
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: rocm_expected)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: hip_runtime)
    monkeypatch.setenv("USER", "ada")
    return hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", detail),
    )


def test_a_cuda_wheel_on_an_amd_only_host_keeps_the_reinstall_advice(monkeypatch, linux):
    """An AMD-only host running a CUDA-tagged wheel raises the same verdict, and opening
    the node does not make that wheel use the card -- the reinstall is still the repair,
    and the closed node is still real. So this host needs both, exactly like the CPU-wheel
    arm above.

    Fails before the fix, where "AMD is the only vendor" alone made the hint REPLACE the
    reinstall advice, leaving the user with a group command and no way to use the GPU."""
    message = _unusable_message(monkeypatch, "2.11.0+cu128")
    assert "usermod -a -G render,video ada" in message
    assert "Repair installation" in message


def test_a_rocm_wheel_on_the_same_host_still_replaces_it(monkeypatch, linux):
    """Its control, one label apart: a ROCm wheel that cannot initialise the device IS
    fully explained by the closed node, so the reinstall advice would send the user after
    the wrong repair and must still be suppressed."""
    message = _unusable_message(monkeypatch, "2.11.0+rocm7.0")
    assert "usermod -a -G render,video ada" in message
    assert "Repair installation" not in message


def _installer_index_summary(
    index_url: str,
    closed_nodes: str,
    *,
    nvidia: bool = False,
) -> str:
    """install.sh's index summary and the two diagnoses that follow it, run for one index.

    The whole span is lifted rather than the guard alone, because the thing under test is
    WHERE the diagnosis sits relative to the case: a copy of the condition would answer
    the same whichever arm it had been left in.
    """
    lines = _install_sh_lines()
    start = max(i for i, line in enumerate(lines) if line == 'case "$TORCH_INDEX_URL" in')
    anchor = next(i for i in range(start, len(lines)) if "needs a recent kernel" in lines[i])
    # Through the closed-node block as well, so one run shows which of the two
    # diagnoses this index gets.
    last = next(i for i in range(anchor, len(lines)) if "membership opens it" in lines[i])
    end = next(i for i in range(last, len(lines)) if lines[i] == "fi")
    script = "\n".join(
        [
            _shell_fn(lines, "_amd_node_repairs"),
            'substep() { echo "$1"; }',
            'C_WARN=""',
            "_amd_gpu_radeon=false",
            '_strip_index_url_credentials() { printf "%s\\n" "$1"; }',
            "_has_amd_rocm_gpu() { return 1; }",  # ROCm cannot see the card
            "_amd_gpu_present_via_pci() { return 0; }",  # but the PCI bus can
            "SKIP_TORCH=false",
            "OS=linux",
            "_amd_render_node_present() { return 0; }",
            # The route gate classifies the index by its canonical leaf, so the
            # classifiers are lifted rather than stubbed: stubbing them would make the
            # per-URL cases below assert about the stub instead of about the rule. They
            # are defined above the case in install.sh, so the span lifted below calls
            # them without carrying them; a shell function has to exist before the call.
            *_run_scope_defs(lines, nvidia = nvidia),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            *lines[start : end + 1],
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        env = {
            **os.environ,
            "TORCH_INDEX_URL": index_url,
            "_closed_amd_nodes": closed_nodes,
        },
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_kernel_stack_diagnosis_reaches_a_rerouted_gfx_index():
    """The runtime-less host this was written for: no /dev/kfd, so the per-arch reroute
    rewrote its cpu index to a gfx one and it took the other arm of the case. Left inside
    the */cpu arm the diagnosis never printed for exactly the host that needed it.

    Fails before the fix, which produced only the wheels line for this index."""
    out = _installer_index_summary("https://repo.radeon.com/rocm/manylinux/gfx1151", "")
    assert "ROCm cannot see it" in out


def test_a_cpu_index_on_the_same_host_still_gets_it():
    """The control: the arm the diagnosis used to live in must keep it, or the hoist has
    moved the message rather than widened it."""
    out = _installer_index_summary("https://download.pytorch.org/whl/cpu", "")
    assert "ROCm cannot see it" in out


def test_a_closed_kfd_node_still_suppresses_it_after_the_case():
    """And the suppression the hoist must carry with it: /dev/kfd existing is the evidence
    the kernel stack is already loaded, so on a gfx index too the group advice is the
    repair and "install the ROCm kernel stack" is not."""
    out = _installer_index_summary(
        "https://repo.radeon.com/rocm/manylinux/gfx1151",
        "/dev/kfd",
    )
    assert "ROCm cannot see it" not in out
    assert "cannot open its device nodes" in out


def _reason_with_masks(
    monkeypatch,
    env: dict,
    backends: set,
    gpu_count: "int | None" = None,
) -> str:
    """The empty-probe reason on a closed-node host carrying several visibility masks.

    ``gpu_count`` is what KFD enumerates, so a selector can be judged against something.
    None is the default because it is what an unreadable topology answers, which is the
    state every test written before that check ran in.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: gpu_count)
    monkeypatch.setenv("USER", "ada")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset(backends)),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda _b: backends == {"vulkan"}),
    )
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_an_empty_ordinal_variable_is_not_a_filter(monkeypatch, linux):
    """_gpu_device_ordinal_active reads a whitespace GPU_DEVICE_ORDINAL as no filter at
    all, so an empty one hides nothing and clearing it changes nothing.

    Fails before the fix, which applied the CUDA/HIP first-token rule to all four names
    alike and told this user to clear a variable that was already inert."""
    reason = _reason_with_masks(monkeypatch, {"GPU_DEVICE_ORDINAL": ""}, {"hip"})
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_an_ordinal_that_hides_everything_is_still_reported(monkeypatch, linux):
    """Its control: a value that IS a filter and whose first entry names no device leaves
    nothing enumerated, so that host really does need both fixes."""
    reason = _reason_with_masks(monkeypatch, {"GPU_DEVICE_ORDINAL": "-1"}, {"hip"})
    assert "visibility mask is also in force" in reason
    assert "GPU_DEVICE_ORDINAL='-1'" in reason


def test_an_empty_cuda_mask_behind_a_valid_hip_one_is_not_consulted(monkeypatch, linux):
    """clr reads HIP_VISIBLE_DEVICES when it is non-empty and CUDA_VISIBLE_DEVICES
    otherwise, so an empty CUDA mask underneath a valid HIP one is never looked at.
    Naming it sends the user after a variable that hides nothing.

    Fails before the fix, which judged each of the four on its own value."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": ""},
        {"hip"},
    )
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_the_same_empty_cuda_mask_blocks_once_hip_is_unset(monkeypatch, linux):
    """The control, one variable apart: with no HIP mask above it the same empty CUDA
    value is the one clr reads, and it exposes nothing."""
    reason = _reason_with_masks(monkeypatch, {"CUDA_VISIBLE_DEVICES": ""}, {"hip"})
    assert "CUDA_VISIBLE_DEVICES is empty" in reason
    assert "visibility mask is also in force" in reason


def test_an_empty_rocr_mask_blinds_the_runtime_under_a_valid_hip_one(monkeypatch, linux):
    """ROCr is a LOWER layer than clr and composes with it rather than deferring: it
    filters the agent list hsa_iterate_agents returns, and the HIP ordinals then index
    what it left. An empty ROCr mask leaves nothing to index, whatever HIP says.

    This is what keeps the fix from being "only the winner of the precedence chain
    counts", which passes the two tests above and is wrong here."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""},
        {"hip"},
    )
    assert "ROCR_VISIBLE_DEVICES is empty" in reason
    assert "visibility mask is also in force" in reason


def test_a_container_given_kfd_but_no_render_node_is_told_so(monkeypatch, linux):
    """/dev/kfd mapped without /dev/dri passes every probe in this file and still cannot
    initialise ROCm, because ROCr opens a render node to reach amdgpu -- docker/run.sh
    passes both devices for that reason. No group creates the missing one, so it is an
    independent blocker and has to be said alongside the permission repair.

    Fails before the fix, which named only a missing /dev/kfd."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "No AMD render node" in hint
    assert "--device /dev/kfd --device /dev/dri" in hint


def test_a_host_that_has_a_render_node_is_not_told_to_map_one(monkeypatch, linux):
    """Its control, and the one that matters: an ordinary AMD host has a render node and
    only cannot open it, so claiming the device mapping is wrong would send the user
    after a second repair that does not exist."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
    )
    monkeypatch.setenv("USER", "ada")
    assert "No AMD render node" not in amd.amd_node_permission_hint()


def test_the_installer_says_the_same_thing_about_a_missing_render_node(tmp_path):
    """The shell half. Stubbed on both sides so the arms differ only in the answer, since
    the real helper reads the runner's own /sys and /dev."""
    node = tmp_path / "kfd"
    node.write_bytes(b"")
    node.chmod(0o660)
    assert "No AMD render node" in _install_sh_hint(str(node), render_present = False)
    assert "No AMD render node" not in _install_sh_hint(str(node), render_present = True)


def _diag_route(
    index_url: str,
    *,
    skip_torch: bool = False,
    backend: "str | None" = None,
    nvidia: bool = False,
) -> bool:
    """Whether install.sh routes the two node diagnoses for this wheel index.

    Lifted from install.sh rather than restated, since the thing under test is which
    patterns the case actually lists.
    """
    lines = _install_sh_lines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_amd_node_diag_leaf="))
    esac_at = next(i for i in range(start, len(lines)) if lines[i] == "esac")
    # The --no-torch override and the explicit-backend case below it are part of the same
    # decision, so the span runs to the end of both rather than stopping at the first esac.
    _skip_torch_end = next(i for i in range(esac_at, len(lines)) if lines[i] == "fi")
    end = next(i for i in range(_skip_torch_end, len(lines)) if lines[i] == "esac")
    script = "\n".join(
        [
            f"TORCH_INDEX_URL={index_url!r}",
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            # Set either way, so the arms below do not inherit whatever the runner exports.
            f"export UNSLOTH_LLAMA_CPP_BACKEND={backend or ''!r}",
            # The classifiers, not stubs: which leaves count as a ROCm route is exactly what
            # these tests are about, so a stub would have them assert about the stub.
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            # Stubbed like _amd_render_node_present: the real one runs nvidia-smi, so a
            # live one would answer from the runner's own hardware. False by default, so
            # every arm below reads as the AMD-only host it was written for.
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            *lines[start : end + 1],
            'echo "$_amd_node_diag_route"',
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
    return out.stdout.strip() == "true"


@pytest.mark.parametrize(
    "index_url",
    [
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/rocm7.0",
        "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151",
    ],
)
def test_the_node_diagnoses_run_on_the_routes_the_case_reports(index_url):
    """The two arms this installer prints a wheel line for are the two the diagnoses
    belong to, and #10466's host reaches the second by reroute rather than the first."""
    assert _diag_route(index_url) is True


@pytest.mark.parametrize(
    "index_url",
    ["https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/xpu"],
)
def test_a_cuda_route_is_not_told_to_install_the_rocm_kernel_stack(index_url):
    """Moving the diagnoses out of the */cpu arm let them reach an index the case above
    has no arm for at all. _has_amd_rocm_gpu returns false on ANY host with a usable
    NVIDIA GPU, so on a CUDA route its condition is satisfied by every hybrid box with an
    AMD card on the bus, and someone correctly installing CUDA wheels was told to install
    the ROCm kernel stack for a card this install does not use."""
    assert _diag_route(index_url) is False


def test_the_route_gate_actually_suppresses_the_kernel_stack_hint():
    """And that the variable is consulted rather than merely computed."""
    assert not _kernel_stack_hint_runs("", route = False)


def test_a_wheel_tagged_for_another_vendor_keeps_the_reinstall_advice(monkeypatch, linux):
    """A venv that recorded ROCm intent and then had a CUDA build installed over it still
    answers yes to _expected_rocm_flavor_was_chosen, and reading that as "the wheel targets
    AMD" replaced the repair this host needs -- reinstalling ROCm torch -- with a sentence
    about group membership."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: False)
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0+cu128"),
    )
    # Appended rather than replacing, which is the distinction the fix restores: the node
    # is real and still worth saying, and the wheel is still the repair.
    assert "matching PyTorch build fixes it" in message
    assert "cannot open" in message


def test_a_label_that_names_no_vendor_still_lets_the_intent_speak(monkeypatch, linux):
    """The control: +cpu names no accelerator, so it settles nothing about which vendor
    this install targets and the recorded intent is still the best evidence there is.
    Without this the fix reads as "any non-ROCm label wins", which silences the node hint
    on the CPU-torch host #10466 was reported from."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd", "nvidia"}))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_torch_reports_a_hip_runtime", lambda: False)
    # And the third of the trio, for the same reason: it reads whatever torch happens to
    # be installed beside the test, so leaving it live made this pass or fail on the
    # runner's wheel rather than on the host the test describes.
    monkeypatch.setattr(hardware, "_torch_reports_another_vendors_runtime", lambda: False)
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cpu_build", "2.11.0+cpu"),
    )
    assert "cannot open" in message


def test_a_container_with_an_open_kfd_and_no_render_node_is_still_told(monkeypatch, linux):
    """--device /dev/kfd without --device /dev/dri: the one node it has opens, so the
    closed list is empty and this returned None while ROCr had no render node to open.
    A missing node is not a permission problem, so it cannot be gated on one."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    hint = amd.amd_node_permission_hint()
    assert "AMD render node" in hint
    assert "--device /dev/dri" in hint
    assert "usermod" not in hint


def test_a_host_with_no_amd_card_is_not_told_to_map_a_render_node(monkeypatch, linux):
    """Its control, and the trap a bare "the glob is empty" test falls into: every
    vendor's render nodes live under /dev/dri/renderD*, so the AMD-presence signal has to
    come from somewhere that survives having no render node at all. The KFD topology names
    the vendor and is world-readable, which is why it is the one asked."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"}, amd_owned = False)
    assert amd.amd_node_permission_hint() is None


def test_an_ordinal_naming_a_device_that_is_not_there_hides_everything(monkeypatch, linux):
    """HIP reads the list left to right and stops at the first index no device answers to,
    so HIP_VISIBLE_DEVICES=3 on a one-GPU host exposes nothing -- which is exactly the
    empty probe being explained, and was read as a valid selector."""
    reason = _reason_with_masks(monkeypatch, {"HIP_VISIBLE_DEVICES": "3"}, {"hip"}, gpu_count = 1)
    assert "visibility mask is also in force" in reason
    assert "HIP_VISIBLE_DEVICES='3'" in reason


def test_an_ordinal_that_does_name_a_device_is_still_not_a_blocker(monkeypatch, linux):
    """The control: the same host and the same variable pointing at a GPU it has. Without
    it the fix could be "any ordinal blocks", which sends every host with a legitimate
    selector after a change that would take its GPU away."""
    reason = _reason_with_masks(monkeypatch, {"HIP_VISIBLE_DEVICES": "0"}, {"hip"}, gpu_count = 1)
    assert "visibility mask is also in force" not in reason


def test_an_unreadable_device_count_leaves_the_selector_alone(monkeypatch, linux):
    """The other control: an unreadable KFD topology is a detection miss, and reading it
    as "no devices" would call every selector on the host a blocker."""
    reason = _reason_with_masks(monkeypatch, {"HIP_VISIBLE_DEVICES": "3"}, {"hip"}, gpu_count = None)
    assert "visibility mask is also in force" not in reason


def test_a_node_carrying_an_acl_is_not_answered_with_usermod(monkeypatch, tmp_path):
    """acl(5): once an access ACL is present, the group-class bits in st_mode are the ACL
    MASK rather than the owning group's grant, so a node whose mask reads rw can still
    deny its group. Prescribing membership from the mode there is a promise the stat
    cannot support."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: True)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, _priv, _already = amd._groups_that_own([str(node)])
    assert acl == [str(node)]
    assert joinable == [] and unnamed == [] and no_group == []


def test_the_same_node_without_an_acl_is_still_prescribed_for(monkeypatch, tmp_path):
    """The control: the ordinary node, whose mode bits ARE the group's grant. Without it
    the fix could decline to prescribe anywhere, which removes the repair #10466 needs."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    # The fixture can only chgrp to a group this account holds, and a node whose owning
    # group the account already has is filed under `already` rather than `joinable`. This
    # arm is about the derivation, so stand the account outside that group.
    _not_my_group = os.getgid() + 1
    monkeypatch.setattr(amd.os, "getgid", lambda: _not_my_group)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, _priv, _already = amd._groups_that_own([str(node)])
    assert acl == []
    assert joinable or unnamed


def test_the_installer_reports_an_acl_rather_than_prescribing_membership(tmp_path):
    """The shell twin of the same rule: ls marks such a node with a trailing "+", which
    is the marker available without getfacl."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    try:
        _set = subprocess.run(["setfacl", "-m", "u:nobody:rw", str(node)], capture_output = True)
    except OSError:
        pytest.skip("setfacl is not installed")
    if _set.returncode != 0:
        pytest.skip("this filesystem does not support ACLs")
    out = _install_sh_hint(str(node))
    assert "carries a POSIX ACL" in out
    assert "usermod -a -G" not in out


def test_the_installer_says_the_missing_render_node_with_nothing_closed():
    """The installer's half of the container case: nothing closed, no render node, and an
    AMD GPU in the KFD topology."""
    out = _install_sh_hint("", render_present = False, amd_present = True)
    assert "no AMD render node" in out
    assert "--device /dev/dri" in out


def test_the_installer_stays_quiet_on_a_host_with_no_amd_gpu():
    """Its control, and the same vendor trap: without the KFD topology test this fires on
    every host whose /dev/dri holds another vendor's nodes, or none at all."""
    out = _install_sh_hint("", render_present = False, amd_present = False)
    assert out.strip() == ""


def test_an_ordinary_node_reports_no_acl(tmp_path):
    """The probe itself, on a file with none: it has to answer False for the common node
    or every host stops being prescribed for. The positive direction needs setfacl, which
    the shell test above skips on when it is absent."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    assert amd._has_an_access_acl(str(node)) is False


def test_a_path_that_cannot_be_read_reports_no_acl(tmp_path):
    """And the failure mode that must not raise: this runs where things are already
    wrong, so an unreadable path answers False rather than taking the hint down."""
    assert amd._has_an_access_acl(str(tmp_path / "gone")) is False


def test_the_acl_probe_matches_the_name_type_listxattr_returns(monkeypatch, tmp_path):
    """os.listxattr returns ``str`` names for a ``str`` path, so the bytes literal this
    first shipped with could never match one and the whole ACL branch was dead.

    Stubbed rather than written with setfacl, which is not installed here: the shell twin
    skips when it is absent, and a probe whose positive direction is only ever exercised
    by a skipping test is not exercised at all. That is how this survived a round."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    monkeypatch.setattr(
        amd.os, "listxattr", lambda path: ["security.selinux", "system.posix_acl_access"]
    )
    assert amd._has_an_access_acl(str(node)) is True


def test_the_acl_probe_also_reads_bytes_names(monkeypatch, tmp_path):
    """A bytes path yields bytes names, and the caller chooses the path type, so both are
    accepted. The control for the test above: without it, swapping one literal for the
    other passes just as well and nothing says which type is actually returned."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    monkeypatch.setattr(amd.os, "listxattr", lambda path: [b"system.posix_acl_access"])
    assert amd._has_an_access_acl(str(node)) is True


def test_another_xattr_is_not_read_as_an_acl(monkeypatch, tmp_path):
    """The negative control. An ACL is claimed from one exact name, so a node carrying
    only other attributes stays prescribed for."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    monkeypatch.setattr(amd.os, "listxattr", lambda path: ["security.selinux", "user.note"])
    assert amd._has_an_access_acl(str(node)) is False


def test_every_unnamed_gid_reaches_the_docker_repair(monkeypatch, linux):
    """docker's --group-add takes ONE value, so a host whose nodes sit in two unnamed
    groups needs the flag twice. Naming only the first leaves the second node shut and
    the user with a command that half works.

    Fails before the fix, which interpolated unnamed[0] alone."""
    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993, 994], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "--group-add 993 --group-add 994" in hint
    assert "GIDs 993, 994" in hint


def test_a_lone_unnamed_gid_is_still_named_in_the_singular(monkeypatch, linux):
    """The control on the wording: the one-GID host is the common one and must not start
    reading as though it had several."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "GID 993, which has" in hint
    assert "--group-add 993." in hint


def test_a_rocr_selector_naming_a_uuid_is_reported_as_unresolved(monkeypatch, linux):
    """ROCr accepts a UUID as well as an ordinal, and one naming no device on this host
    stops the list exactly as a bad ordinal does. Nothing here can match a UUID against
    the KFD ordinal count, so it is reported as unresolved rather than dismissed -- which
    is what the ordinal check did to it, leaving the user with no mention of the one
    variable that may be hiding their card.

    Fails before the fix, which returned False for any non-digit entry and said nothing."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58"},
        {"hip"},
        gpu_count = 1,
    )
    assert "cannot resolve" in reason
    assert "ROCR_VISIBLE_DEVICES" in reason
    # Reported, not judged: claiming it blocks would invent a fault this cannot see.
    assert "which the groups do not clear" not in reason


def test_a_rocr_ordinal_that_names_a_device_is_still_left_alone(monkeypatch, linux):
    """The control that keeps it narrow: an ordinal the count can resolve is judged as
    before, so the new sentence cannot appear on every host that sets the variable."""
    reason = _reason_with_masks(monkeypatch, {"ROCR_VISIBLE_DEVICES": "0"}, {"hip"}, gpu_count = 2)
    assert "cannot resolve" not in reason
    assert "visibility mask" not in reason


def test_a_uuid_in_the_hip_layer_is_unresolved_rather_than_a_blocker(monkeypatch, linux):
    """And the other boundary. An earlier revision asserted the opposite here, on the
    claim that only ROCr accepts a UUID; rocdevice.cpp refutes it, matching a "GPU-" token
    against each agent's own HSA_AMD_AGENT_INFO_UUID before it falls back to an ordinal.
    So the token may name a device and may name nothing, exactly as under ROCr, and
    nothing here can tell which: calling it a blocker invents a fault on a host whose
    selector is fine, and saying nothing hides the one variable that may be the cause.

    The bound is deliberately known here, since a UUID is not an index into it and must
    not be judged against it either way."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58"},
        {"hip"},
        gpu_count = 1,
    )
    assert "names a device this cannot resolve" in reason
    assert "visibility mask is also in force" not in reason


def _vulkan_reason_with_open_sibling(monkeypatch, openable: set) -> str:
    """The empty-probe reason for a Vulkan build with renderD128 closed."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(
        monkeypatch,
        present = ["/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = openable,
    )
    monkeypatch.setenv("USER", "ada")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda _b: frozenset({"vulkan"}))
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b: True))
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_an_open_sibling_node_keeps_the_vulkan_reason(monkeypatch, linux):
    """A closed node explains an empty probe only when it is the node the runtime would
    have used. With renderD129 open the Vulkan loader had one to enumerate and still
    reported nothing, so the closed renderD128 is a second finding and returning it alone
    sends the user after a repair that leaves the probe just as empty.

    Fails before the fix, which returned the node hint unconditionally."""
    reason = _vulkan_reason_with_open_sibling(monkeypatch, {"/dev/dri/renderD129"})
    assert "the Vulkan probe reported no device" in reason
    # Still said, because it is still true and still worth repairing.
    assert "/dev/dri/renderD128" in reason


def test_no_open_sibling_still_gives_the_node_hint_alone(monkeypatch, linux):
    """The control, and the #10466 host itself: with every AMD node closed there is no
    sibling the loader could have used, so the closed node IS the reason and must not be
    demoted to a footnote behind a Vulkan sentence that explains nothing."""
    reason = _vulkan_reason_with_open_sibling(monkeypatch, set())
    assert "the Vulkan probe reported no device" not in reason
    assert "/dev/dri/renderD128" in reason


def test_a_hip_ordinal_is_judged_against_what_rocr_left(monkeypatch, linux):
    """ROCr filters the physical list first and renumbers the survivors; the HIP layer then
    indexes those. With ROCR_VISIBLE_DEVICES=0 on a two-GPU host one device survives, so
    HIP ordinal 1 names nothing and hides everything -- judged against the physical count of
    2 it reads as a valid selector and the user is told only about the groups.

    Fails before the fix, which used the KFD count for every layer."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"},
        {"hip"},
        gpu_count = 2,
    )
    assert "HIP_VISIBLE_DEVICES='1'" in reason
    assert "which the groups do not clear" in reason


def test_the_same_ordinal_inside_what_rocr_left_is_not_a_blocker(monkeypatch, linux):
    """The control: ROCr leaving both devices makes HIP ordinal 1 a real device again, so
    the composed reading must not call every stacked pair a blocker."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "1"},
        {"hip"},
        gpu_count = 2,
    )
    assert "which the groups do not clear" not in reason


def test_an_unresolvable_rocr_entry_leaves_the_hip_ordinal_alone(monkeypatch, linux):
    """And the boundary: a UUID in the ROCr layer means the survivors cannot be counted, so
    the HIP ordinal is judged against nothing rather than against a number this invented."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58", "HIP_VISIBLE_DEVICES": "1"},
        {"hip"},
        gpu_count = 2,
    )
    assert "which the groups do not clear" not in reason


def _hip_reason_with_nodes(monkeypatch, present: list, openable: set) -> str:
    """The empty-probe reason for a HIP build over a given node layout."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = present, openable = openable)
    monkeypatch.setenv("USER", "ada")
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda _b: frozenset({"hip"}))
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b: False))
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_a_closed_sibling_beside_an_open_rocm_path_is_not_the_reason(monkeypatch, linux):
    """ROCm needs /dev/kfd and a render node. With both open on a multi-AMD host, a closed
    SECOND render node is not why the probe came back empty, and returning the group repair
    as the sole diagnosis leaves the user fixing something that changes nothing.

    Fails before the fix, which asked the sibling question of Vulkan builds only."""
    reason = _hip_reason_with_nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    assert "Separately, and not why the probe is empty" in reason
    assert "/dev/dri/renderD128" in reason


def test_a_closed_kfd_is_still_the_reason_for_a_hip_build(monkeypatch, linux):
    """The control that keeps it narrow: /dev/kfd has no sibling, so a closed one blocks
    ROCm outright however many render nodes are open, and that host must still be told the
    closed node IS the reason."""
    reason = _hip_reason_with_nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/dri/renderD129"},
    )
    assert "Separately, and not why the probe is empty" not in reason
    assert "/dev/kfd" in reason


def test_a_missing_render_node_blocks_the_runtime(monkeypatch, linux):
    """--device /dev/kfd without --device /dev/dri. The one node mapped opens, so nothing
    is CLOSED, and this answered False -- which made hardware.py suppress the very hint
    that names the repair, and llama_cpp.py file it as "not why the probe is empty" when
    the absent render node is exactly why."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    assert amd.amd_closed_nodes_block_the_runtime() is True


def test_a_complete_open_mapping_still_does_not_block(monkeypatch, linux):
    """The control. Without it the fix could be "always blocks", which suppresses nothing
    and labels every empty probe a permission problem."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/kfd", "/dev/dri/renderD128"},
    )
    assert amd.amd_closed_nodes_block_the_runtime() is False


def test_a_container_given_only_the_render_node_is_told_about_kfd(monkeypatch, linux):
    """The mirror image of the case above, and the common asymmetric mapping: --device
    /dev/dri alone. The render node is present and open, so `closed` is empty and
    `_render_missing` is false, and the hint returned None before reaching its own
    missing-KFD sentence -- leaving the caller on generic reinstall advice for a host
    where HIP has no /dev/kfd to open."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
    hint = amd.amd_node_permission_hint()
    assert "/dev/kfd" in hint
    assert "usermod" not in hint


def test_the_same_mapping_says_nothing_to_a_vulkan_caller(monkeypatch, linux):
    """The control, and the reason needs_kfd exists: Vulkan never opens /dev/kfd, so a
    Vulkan failure with some other cause must not be sent after the ROCm kernel stack."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
    assert amd.amd_node_permission_hint(needs_kfd = False) is None


def test_a_node_this_account_owns_is_not_answered_with_a_group(monkeypatch, tmp_path):
    """POSIX resolves the owner class exclusively once the uid matches, so a node this
    account owns whose owner bits deny cannot be opened by joining its group however the
    group bits read. Prescribing usermod there sends the user after a command that
    succeeds and changes nothing."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o060)  # group rw, owner nothing: os.access() says shut, POSIX says owner
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        [str(node)]
    )
    assert owned == [str(node)]
    assert joinable == [] and unnamed == [] and no_group == [] and privileged == []


def test_the_same_node_owned_by_someone_else_is_still_a_group(monkeypatch, tmp_path):
    """The control: identical mode, a different owner. The owner class no longer applies,
    the group bits are the grant, and membership IS the repair. Without this the rule
    could be "never prescribe a group", which removes what #10466 asked for."""
    node, _group = _a_node_a_membership_would_open(tmp_path, mode = 0o060)
    # The fixture can only chgrp to a group this account holds, and a node whose owning
    # group the account already has is filed under `already` rather than `joinable`. This
    # arm is about the derivation, so stand the account outside that group.
    _not_my_group = os.getgid() + 1
    monkeypatch.setattr(amd.os, "getgid", lambda: _not_my_group)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        [str(node)]
    )
    assert owned == []
    assert joinable or unnamed


def test_a_root_owned_node_is_not_answered_with_usermod_root(monkeypatch, linux):
    """root:root 0660 opens for anyone in the root group, so the group derivation would
    accept the name and print `sudo usermod -a -G root`. That membership grants a
    great deal besides the GPU, so it is a udev misconfiguration to report rather than a
    repair to prescribe."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (0, 0o660, 0)}, {0: "root"})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert privileged == ["root"]
    assert joinable == []


def test_an_ordinary_owning_group_is_still_prescribed(monkeypatch, linux):
    """The control: render is not privileged, so the same shape still yields the command.
    Without it the rule could be "never name a group"."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (39, 0o660, 0)}, {39: "render"})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert joinable == ["render"]
    assert privileged == []


def test_the_hint_for_a_privileged_owner_says_it_is_not_the_repair(monkeypatch, linux):
    """The sentence a user actually reads, since the buckets above only decide it."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], ["root"], []))
    hint = amd.amd_node_permission_hint()
    assert "usermod" not in hint
    assert "root" in hint and "udev" in hint


def test_the_installer_does_not_dangle_the_group_sentence(tmp_path):
    """When every refused node has an unnamed GID, an ACL, or a mode no group can open,
    _amd_node_repairs names no group on purpose. The installer printed "Add yourself to
    the" above that branch regardless, so the message read as an instruction cut off
    mid-sentence and then contradicted."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o600)  # owner-only: no membership opens it, so no group is named
    out = _install_sh_hint(str(node))
    assert "Add yourself to the" not in out
    assert "no" in out and "membership opens it" in out


def test_the_installer_still_offers_the_group_when_there_is_one(tmp_path):
    """The control: a node whose group grants read and write still gets the sentence and
    the command, in one piece."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node))
    assert "Add yourself to the" in out
    assert "usermod -a -G" in out


def test_the_installer_stops_at_the_owner_class_too(tmp_path):
    """The shell half of the owner-precedence item: with the caller as the owner, the
    installer must not print a usermod line for a node no membership opens."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o060)
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()))
    assert "usermod" not in out
    assert "owned by this account" in out


def _kernel_stack_hint_text(*, topology: bool, nvidia: bool = False) -> str:
    """What install.sh actually PRINTS in the missing-/dev/kfd branch.

    `_kernel_stack_hint_runs` above lifts only the guard, so it answers whether the
    branch fires and nothing about which repair it names -- which is exactly where the
    branch was wrong. This lifts the guard AND its body, through the closing `fi`, so a
    revert changes the text this returns.
    """
    lines = _install_sh_lines()
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    close = next(i for i in range(end + 1, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[start : close + 1])
    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            "_has_amd_rocm_gpu() { return 1; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if topology else 1}; }}",
            "SKIP_TORCH=false",
            "OS=linux",
            "C_WARN=",
            "_amd_node_diag_route=true",
            *_run_scope_defs(lines, nvidia = nvidia),
            block,
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        check = True,
        env = {**os.environ, "_closed_amd_nodes": ""},
    )
    return out.stdout


def test_the_installer_does_not_prescribe_a_reinstall_when_the_topology_is_there():
    """A container created with --device /dev/dri and no --device /dev/kfd sees the
    host's /sys and not its /dev, so the KFD topology names an AMD GPU while the node
    is absent. The driver is therefore already loaded, and "install the ROCm kernel
    stack" is a repair that leaves HIP exactly as unavailable as before."""
    out = _kernel_stack_hint_text(topology = True)
    assert "Install the ROCm kernel stack" not in out
    assert "--device /dev/kfd" in out
    assert "the node itself" in out


def test_the_installer_keeps_the_kernel_stack_advice_without_a_topology():
    """The control, and the case the branch was written for: no KFD topology at all, so
    the driver really is missing and the reinstall is the repair. Without this the fix
    could be "never mention the kernel stack", which removes a correct diagnosis."""
    out = _kernel_stack_hint_text(topology = False)
    assert "Install the ROCm kernel stack" in out
    assert "--device /dev/kfd" not in out


def test_a_container_missing_kfd_is_told_to_map_it_rather_than_reinstall(monkeypatch, linux):
    """The runtime half of the same item. `_amd_nodes_the_runtime_lacks` reports a
    missing node only once the KFD topology names an AMD GPU, and that topology is the
    amdkfd driver's own sysfs -- so on every host this sentence can reach, the kernel
    stack is already loaded and the advice to install it is unreachable-by-construction
    wrong."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
    hint = amd.amd_node_permission_hint()
    assert "--device /dev/kfd" in hint
    assert "kernel stack" not in hint
    assert "the kernel driver is loaded" in hint


def _install_sh_missing_kfd(
    *,
    topology: bool,
    amd_smi_sees_it: "bool | None" = None,
    rocm_visible: "bool | None" = None,
    skip_torch: bool = False,
    backend: "str | None" = None,
    nvidia: bool = False,
) -> str:
    """What the installer says when /dev/kfd is absent, for a given pair of probes.

    Lifts the two branches together, through the closing `fi`, because which of them runs
    is the thing under test. The `[ ! -e /dev/kfd ]` test is left live rather than stubbed
    -- a test operator cannot be stubbed, and rewriting it would be editing the code under
    test -- so the case needs a host without the node.
    """
    if os.path.exists(amd._KFD_NODE):
        pytest.skip("this arm needs a host with no /dev/kfd, and cannot remove a device node")
    lines = _install_sh_lines()
    # Anchored on the kernel-stack SENTENCE, then walked back to the `if` above it, since
    # neither branch's condition is stable enough to anchor on: the mapping one is what an
    # earlier change edited, and a revert that stopped the extraction finding anything would
    # read "the text changed" as "the behaviour changed". The predicate _amd_gpu_present_via_pci
    # is named twice in this installer, so it is not an anchor either.
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    close = next(i for i in range(end + 1, len(lines)) if lines[i] == "fi")
    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            'C_WARN=""',
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            "OS=linux",
            "_amd_node_diag_route=true",
            *_run_scope_defs(lines, nvidia = nvidia),
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if topology else 1}; }}",
            # Stubbed when the arm is about something else and only needs a verdict; run
            # for real over stubbed command lookups when the arm IS about which probe the
            # branch consults, since a stub of the probe under test would answer for it.
            # Running it for real also exercises the "ignore-nvidia" argument the branch
            # passes, which is what keeps the diagnosis off the NVIDIA short-circuit.
            *(
                [f"_has_amd_rocm_gpu() {{ return {0 if amd_smi_sees_it else 1}; }}"]
                if rocm_visible is None
                else [
                    "_ensure_rocm_probe_env() { :; }",
                    'command() { case "$2" in rocminfo) return 0 ;; *) return 1 ;; esac; }',
                    "rocminfo() { echo '  Name: gfx1151'; }"
                    if rocm_visible
                    else "rocminfo() { return 1; }",
                    _shell_fn(lines, "_has_amd_rocm_gpu"),
                ]
            ),
            "_amd_gpu_present_via_pci() { return 0; }",
            "\n".join(lines[start : close + 1]),
        ]
    )
    env = {**os.environ, "_closed_amd_nodes": ""}
    env.pop("UNSLOTH_LLAMA_CPP_BACKEND", None)
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
    out = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, check = True, env = env
    )
    return out.stdout


def test_amd_smi_does_not_suppress_the_missing_kfd_mapping_advice():
    """amd-smi reads the driver over sysfs and libdrm, so it lists the card in a container
    given only --device /dev/dri, where HIP has no /dev/kfd to open -- llama_cpp.py's
    _rocm_hip_is_reachable documents exactly that disagreement. Behind _has_amd_rocm_gpu the
    mapping advice was therefore suppressed on the container shape it was written for, and
    nothing else spoke: the render node is open, so no node is closed and none is missing."""
    out = _install_sh_missing_kfd(topology = True, amd_smi_sees_it = True)
    assert "--device /dev/kfd" in out
    assert "Install the ROCm kernel stack" not in out


def test_the_kernel_stack_advice_still_needs_rocm_to_be_blind():
    """The control, and the reason the two are separate branches rather than one branch
    with an inner test: they need different evidence. With no KFD topology the driver
    really is missing, and that diagnosis is still gated on ROCm seeing nothing, so a host
    whose amd-smi answers is not told to install what it already has."""
    assert "Install the ROCm kernel stack" in _install_sh_missing_kfd(
        topology = False, amd_smi_sees_it = False
    )
    assert "Install the ROCm kernel stack" not in _install_sh_missing_kfd(
        topology = False, amd_smi_sees_it = True
    )


def test_a_docker_owned_node_is_not_answered_with_usermod(monkeypatch, linux):
    """Membership in docker is root by another route -- a container started with the host
    filesystem mounted -- so prescribing it to open a GPU node is a privilege escalation
    dressed as a device repair, exactly as for wheel."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (999, 0o660, 0)}, {999: "docker"})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert privileged == ["docker"]
    assert joinable == []


def test_the_installer_denies_the_same_groups_the_runtime_does():
    """The two lists are maintained by hand in two languages, so drift is the failure mode.
    Read install.sh's alternation and compare it to the constant rather than restating
    either: a group added to one half alone fails here."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    text = install_sh.read_text(encoding = "utf-8")
    match = re.search(r"\$2 ~ /\^\(([a-z|]+)\)\$/", text)
    assert match, "install.sh no longer carries the privileged-group alternation"
    assert set(match.group(1).split("|")) == set(amd._PRIVILEGED_GROUPS)


def test_the_installer_repeats_group_add_for_every_unnamed_gid():
    """--group-add takes a SINGLE value, so a comma-joined pair is one group name that does
    not exist, and naming only the first leaves the second node shut. docker/run.sh repeats
    the flag and the Python half already emits it repeated; the installer said "the numeric
    GID", singular, for a value it had just printed as "993,994"."""
    out = _install_sh_hint("/dev/kfd\n/dev/dri/renderD128", repairs = "gid:993\ngid:994")
    assert "--group-add 993 --group-add 994" in out
    assert "GIDs 993,994" in out


def test_one_unnamed_gid_still_reads_as_one():
    """The control: the singular wording and a single flag, so the fix is not "always say
    GIDs"."""
    out = _install_sh_hint("/dev/kfd", repairs = "gid:993")
    assert "--group-add 993" in out
    assert "--group-add 993 --group-add" not in out
    assert "GID 993" in out


def test_an_empty_rocr_token_keeps_the_prefix_it_already_counted(monkeypatch, linux):
    """ROCr's RvdFilter builds its list from tokens that are "Legal and NOT Terminating",
    so an entry it cannot evaluate ends the list and the devices BEFORE it still survive.
    ROCR_VISIBLE_DEVICES='0,' therefore leaves one device, and HIP ordinal 1 hides it --
    read as an unresolvable list instead, the HIP mask went unmentioned."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "0,", "HIP_VISIBLE_DEVICES": "1"},
        {"hip"},
        gpu_count = 2,
    )
    assert "HIP_VISIBLE_DEVICES='1'" in reason
    assert "which the groups do not clear" in reason


def test_a_repeated_rocr_ordinal_surfaces_one_device(monkeypatch, linux):
    """The same rule from the other side: an enumeration index is Terminating when it "maps
    to a device that has been previously selected", so '0,0' surfaces one device rather than
    two. Counting every token read it as two survivors and called a HIP ordinal that hides
    the only device a valid selector."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "0,0", "HIP_VISIBLE_DEVICES": "1"},
        {"hip"},
        gpu_count = 2,
    )
    assert "HIP_VISIBLE_DEVICES='1'" in reason
    assert "which the groups do not clear" in reason


def test_a_live_cuda_runtime_outranks_a_stale_rocm_intent(monkeypatch, linux):
    """A conda or locally built CUDA wheel carries no +cu tag, so the label names no vendor
    and a venv that once recorded a ROCm flavor made this read the wheel as AMD-targeted --
    replacing the reinstall advice with group membership on a host whose CUDA build cannot
    use the AMD card however open its nodes are. torch.version.cuda is the build's own
    answer and is stubbed here rather than the predicate that reads it."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setenv("USER", "ada")

    _torch = types.SimpleNamespace(
        version = types.SimpleNamespace(cuda = "12.8", hip = None), __version__ = "2.11.0"
    )
    monkeypatch.setitem(sys.modules, "torch", _torch)
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0"),
    )
    assert "Repair installation" in message


def test_a_live_hip_runtime_still_gets_the_node_hint_alone(monkeypatch, linux):
    """The control, on the same untagged label: torch.version.hip set means the wheel IS a
    ROCm build, so the closed node explains it fully and the reinstall advice would send
    the user after the wrong repair. Without this the rule could be "an untagged label is
    never AMD"."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setenv("USER", "ada")

    _torch = types.SimpleNamespace(
        version = types.SimpleNamespace(cuda = "12.8", hip = "6.4.0"), __version__ = "2.11.0"
    )
    monkeypatch.setitem(sys.modules, "torch", _torch)
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0"),
    )
    assert "Repair installation" not in message
    assert "usermod -a -G render,video ada" in message


def test_a_closed_node_beside_an_open_sibling_does_not_speak_for_the_card(monkeypatch, linux):
    """One shut render node on a multi-AMD host leaves the other GPU fully reachable: HIP has
    /dev/kfd plus an open render node, and the Vulkan loader has the same. Claiming "no GPU
    backend can use the AMD card" there contradicts _explain_empty_gpu_probe, which appends
    this sentence right after saying the closed node is NOT why the probe came back empty."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    hint = amd.amd_node_permission_hint()
    assert "/dev/dri/renderD128" in hint
    assert "no GPU backend can use the AMD card" not in hint
    assert "the card behind them" in hint and "another AMD render node" in hint


def test_the_same_node_with_no_open_sibling_still_speaks_for_the_card(monkeypatch, linux):
    """The control, and the case the wording was written for: the only render node is shut,
    so nothing enumerates and the claim about the card is exactly right. Without this the fix
    could be "never claim the card", which is the #10466 message gone."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/kfd"},
    )
    hint = amd.amd_node_permission_hint()
    assert "no GPU backend can use the AMD card" in hint


def test_a_kfd_only_closed_set_still_claims_only_rocm(monkeypatch, linux):
    """The second control: the render node is open here too, but /dev/kfd has no sibling, so
    the narrowing must not weaken this arm. Vulkan works; ROCm does not."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/dri/renderD128"},
    )
    hint = amd.amd_node_permission_hint()
    assert "ROCm cannot use the AMD card" in hint


def test_an_unnamed_gid_hint_also_adds_the_account(monkeypatch, linux):
    """groupadd gives the numeric owner a NAME. It does not put this account in the group, so
    a user who follows the sentence to the letter still cannot open the node. Both halves, and
    one per GID: with two unnamed GIDs the singular instruction repaired at most one node."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993, 994], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "993, 994" in hint
    assert "each of them" in hint
    # A pair per GID: one groupadd names one numeric owner, so the second node stays shut
    # for anyone who runs only the first command.
    assert "sudo groupadd -g 993 amdgpu993 && sudo usermod -a -G amdgpu993 ada" in hint
    assert "sudo groupadd -g 994 amdgpu994 && sudo usermod -a -G amdgpu994 ada" in hint
    assert "--group-add 993 --group-add 994" in hint


def test_a_single_unnamed_gid_reads_singular(monkeypatch, linux):
    """The control: the plural wording must not be the only wording, or one GID reads as two
    and the sentence stops matching what it printed."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "GID 993" in hint and "GIDs" not in hint
    assert "create a group for it" in hint
    assert "sudo groupadd -g 993" in hint


def test_the_installer_also_adds_the_account_for_unnamed_gids(tmp_path):
    """The installer twin of the rule above: it printed the container flags per GID after the
    earlier fix, but still said only "create a group" for the bare host."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993\ngid:994")
    assert "sudo groupadd -g 993 amdgpu993" in out
    assert "sudo usermod -a -G amdgpu993 ada" in out
    assert "sudo groupadd -g 994 amdgpu994" in out
    assert "sudo usermod -a -G amdgpu994 ada" in out
    assert "--group-add 993 --group-add 994" in out
    assert "create a group for each" in out


def test_the_installer_unnamed_gid_repair_says_to_start_a_new_session(tmp_path):
    """The installer twin of the session-refresh rule: same usermod, same reason to open a
    new session before retrying, and the same silence. Its own sentence rather than the
    named-group one, which this fixture does not reach at all."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993")
    assert "sudo groupadd -g 993 amdgpu993" in out
    assert "log out and back in" in out


def _install_sh_classify(stat_line: str, *, self_uid: str = "4242") -> str:
    """One classified line from the real _amd_node_repairs, for a synthetic stat record.

    The awk program is lifted whole, and only ``stat`` and ``ls`` are stubbed, because the
    cases that matter cannot be built as files: a node whose GID has no group entry needs a
    GID this host does not name, and a root-owned one needs root. The record is exactly what
    `stat -c '%a|%G|%g|%n|%u'` prints, so the input under test is the shipped format.
    """
    lines = _install_sh_lines()
    script = "\n".join(
        [
            f"stat() {{ printf '%s\\n' {shlex.quote(stat_line)}; }}",
            "ls() { printf '%s\\n' '-rw-rw---- 1 root root 0 Jan 1 00:00 node'; }",
            f"id() {{ echo {self_uid}; }}",
            _shell_fn(lines, "_amd_node_repairs"),
            "_amd_node_repairs /dev/kfd",
        ]
    )
    return _install_sh_run(script).strip()


def _install_sh_diag_route(leaf: str) -> str:
    """Whether install.sh routes a given torch index leaf into the AMD node diagnosis.

    The case statement is lifted whole rather than re-expressed, so the globs under test
    are the shipped ones; only the leaf it reads is supplied, through a stubbed extractor,
    because the URL-to-leaf step has its own tests.
    """
    lines = _install_sh_lines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_amd_node_diag_leaf="))
    end = next(i for i in range(start, len(lines)) if lines[i] == "esac")
    script = "\n".join(
        [
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            f"_torch_index_url_leaf() {{ printf '%s' {shlex.quote(leaf)}; }}",
            "TORCH_INDEX_URL=stub",
            "\n".join(lines[start : end + 1]),
            'printf "%s" "$_amd_node_diag_route"',
        ]
    )
    return _install_sh_run(script).strip()


def _install_sh_kfd_scope(
    closed_nodes: str,
    *,
    skip_torch: bool,
    backend: "str | None",
    torch_index: str = "https://download.pytorch.org/whl/rocm6.4",
    nvidia: bool = False,
) -> str:
    """The closed-node message with the KFD scoping in front of it.

    A separate lift from _install_sh_hint because the filter sits ABOVE the block that
    harness extracts -- deliberately, so the diagnosis itself stays one self-contained
    block -- and the thing under test here is which nodes reach it.
    """
    lines = _install_sh_lines()
    _filter_start = next(
        i for i, line in enumerate(lines) if line == "if ! _run_may_open_kfd; then"
    )
    _filter_end = next(i for i in range(_filter_start, len(lines)) if lines[i] == "fi")
    block_start = _install_sh_if(lines, '[ -n "$_closed_amd_nodes" ]; then')
    end = next(i for i in range(block_start, len(lines)) if lines[i] == "fi")
    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            'C_WARN=""',
            "_amd_render_node_present() { return 0; }",
            'id() { case "$1" in -un) echo ada ;; *) echo 4242 ;; esac; }',
            "_amd_node_diag_route=true",
            "OS=linux",
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            "_amd_node_repairs() { printf '%s\\n' 'join:render'; }",
            *_run_scope_defs(lines, nvidia = nvidia),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            "\n".join(lines[_filter_start : _filter_end + 1]),
            "\n".join(lines[block_start : end + 1]),
        ]
    )
    env = _install_sh_env(closed_nodes, "ada", backend, torch_index)
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_vulkan_only_no_torch_run_is_not_sent_after_kfd():
    """/dev/kfd is opened by ROCm and by nothing else, so a run installing neither ROCm torch
    nor a ROCm GGUF bundle has no use for it. This is the shell twin of the needs_kfd argument
    the runtime half already takes."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "vulkan")
    assert out.strip() == ""


def test_no_torch_alone_still_reports_a_closed_kfd():
    """The control, and the reason SKIP_TORCH cannot decide this on its own: --no-torch still
    installs a GGUF bundle, the ROCm bundle opens /dev/kfd exactly as torch would, and which
    bundle it will be is chosen later, in setup.sh. Suppressing here would hide the #10466
    diagnosis from the GGUF users it was written for."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = None)
    assert "cannot open its device nodes" in out
    assert "/dev/kfd" in out


def test_a_vulkan_only_run_still_reports_a_closed_render_node():
    """The second control: Vulkan opens the render node, so the scoping must take /dev/kfd and
    nothing else. A filter that dropped the whole diagnosis would silence the node that blocks
    every backend."""
    out = _install_sh_kfd_scope("/dev/kfd\n/dev/dri/renderD128", skip_torch = True, backend = "vulkan")
    assert "/dev/dri/renderD128" in out
    assert "/dev/kfd" not in out


def test_a_torch_install_is_unaffected_by_the_backend_request():
    """And the control for every ordinary run: ROCm torch opens /dev/kfd whatever the GGUF
    bundle is, so an explicit Vulkan llama.cpp request must not scope the torch diagnosis."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = False, backend = "vulkan")
    assert "/dev/kfd" in out


def test_a_no_torch_rocm_bundle_is_still_told_its_kfd_node_is_missing():
    """The mirror of the closed-node scoping, on the branch that reports an ABSENT node.
    It was gated on SKIP_TORCH=false, so a --no-torch run whose GGUF bundle is ROCm -- which
    opens /dev/kfd exactly as torch would -- finished silently in a container mapping
    /dev/dri and not /dev/kfd, with nothing closed, nothing missing said, and no account of
    why the backend cannot initialise."""
    out = _install_sh_missing_kfd(
        topology = True, amd_smi_sees_it = True, skip_torch = True, backend = None
    )
    assert "/dev/kfd is not present" in out


def test_a_no_torch_vulkan_run_is_not_told_about_a_missing_kfd_node():
    """The control that keeps the rule the same one: Vulkan opens no /dev/kfd, so the node
    being absent explains nothing about it and the run must stay silent."""
    out = _install_sh_missing_kfd(
        topology = True, amd_smi_sees_it = True, skip_torch = True, backend = "vulkan"
    )
    assert out.strip() == ""


def test_a_torch_install_still_gets_the_missing_kfd_advice():
    """The other control: the ordinary install is unchanged by the gate swap."""
    out = _install_sh_missing_kfd(topology = True, amd_smi_sees_it = True)
    assert "/dev/kfd is not present" in out


def test_a_no_torch_cpu_run_is_not_sent_after_a_closed_render_node():
    """An explicit CPU bundle beside --no-torch opens no GPU node at all, so the group and
    udev repair below described a card nothing in the run was going to touch. Only /dev/kfd
    was filtered, which is right for Vulkan and one node short for CPU."""
    out = _install_sh_kfd_scope("/dev/dri/renderD128", skip_torch = True, backend = "cpu")
    assert out.strip() == ""


def test_a_no_torch_cpu_run_is_silent_about_the_kfd_node_too():
    """The same run, the other node. Both were already suppressed for /dev/kfd; this pins
    that the wider rule did not lose the narrower one."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "cpu")
    assert out.strip() == ""


def test_a_torch_install_asking_for_cpu_llama_still_reports_its_nodes():
    """The control, and the reason the predicate reads BOTH: a CPU llama.cpp bundle beside a
    ROCm torch install still has torch opening the nodes, so the backend request alone
    cannot silence the diagnosis."""
    out = _install_sh_kfd_scope("/dev/dri/renderD128", skip_torch = False, backend = "cpu")
    assert "cannot open its device nodes" in out


def test_an_illegal_rocr_selector_is_reported_as_a_blocker(monkeypatch, linux):
    """ROCr calls a token Illegal when it "can\'t be evaluated into an instance of Device
    UUID or Enumeration Index" (ROCR-Runtime, core/inc/amd_filter_device.h), and an Illegal
    token ends the list -- so an illegal FIRST token leaves zero survivors. _post_rocr_device_count
    already counts it that way, while this predicate called every non-index merely unresolved:
    the message then offered the mask as something to check if the group change fails, when it
    is an independent blocker no membership clears."""
    reason = _reason_with_masks(
        monkeypatch, {"ROCR_VISIBLE_DEVICES": "garbage"}, {"hip"}, gpu_count = 2
    )
    assert "ROCR_VISIBLE_DEVICES" in reason
    assert "cannot resolve" not in reason


def test_a_uuid_selector_is_still_only_unresolved(monkeypatch, linux):
    """The control that keeps the two apart: a UUID is a form ROCr accepts, so it may well
    name a device, and this host has no way to match it against an ordinal count. Calling it
    a blocker would invent a fault. Without this the rule reads as "any non-index blocks"."""
    reason = _reason_with_masks(
        monkeypatch,
        {"ROCR_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58"},
        {"hip"},
        gpu_count = 2,
    )
    assert "cannot resolve" in reason


def test_an_untagged_cuda_wheel_that_will_not_import_is_still_another_vendors(monkeypatch, linux):
    """_torch_reports_a_hip_runtime answers an import failure from torch/version.py on disk;
    its mirror returned False there instead, so the untagged-CUDA clearing was inert on the
    one path it exists for. A stale recorded ROCm flavor then spoke for a CUDA wheel, and the
    closed node replaced the reinstall guidance with group membership that cannot make that
    wheel use the card."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", ImportError("libcudart.so.13"))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_installed_torch_label_on_disk", lambda: "2.11.0")
    monkeypatch.setattr(
        hardware,
        "_installed_torch_markers_on_disk",
        lambda: {"cuda": "13.0", "hip": None, "xpu": None},
    )
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0"),
    )
    assert "Repair installation" in message


def test_an_unimportable_rocm_wheel_still_gets_the_node_hint_alone(monkeypatch, linux):
    """The control, and the reason the hip reading leads: a ROCm build records hip and may
    record cuda besides, so ordering the two the other way round would send every AMD host
    whose torch fails to import after a reinstall it does not need."""
    from utils.hardware import hardware

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", ImportError("libamdhip64.so"))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_installed_torch_label_on_disk", lambda: "2.11.0")
    monkeypatch.setattr(
        hardware,
        "_installed_torch_markers_on_disk",
        lambda: {"cuda": "13.0", "hip": "6.4.0", "xpu": None},
    )
    monkeypatch.setenv("USER", "ada")
    message = hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = ("torch_cuda_unavailable", "2.11.0"),
    )
    assert "Repair installation" not in message
    assert "usermod -a -G render,video ada" in message


def test_a_no_torch_cuda_run_is_not_sent_after_the_amd_nodes():
    """REQUESTABLE_BACKENDS is auto/cpu/cuda/rocm/vulkan, and a CUDA bundle opens
    /dev/nvidia* and neither AMD node. The predicates listed vulkan and cpu, so a
    --no-torch run asking for the CUDA bundle on a host with an AMD card on the bus was
    still handed group and udev repairs for a card nothing in the run would touch."""
    assert _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "cuda").strip() == ""
    assert (
        _install_sh_kfd_scope("/dev/dri/renderD128", skip_torch = True, backend = "cuda").strip() == ""
    )


def test_a_no_torch_cuda_run_is_not_told_about_a_missing_kfd_node_either():
    """The same request on the branch that reports an ABSENT node."""
    out = _install_sh_missing_kfd(
        topology = True, amd_smi_sees_it = True, skip_torch = True, backend = "cuda"
    )
    assert out.strip() == ""


def test_a_torch_install_asking_for_cuda_llama_still_reports_its_nodes():
    """The control: a CUDA llama.cpp bundle beside a ROCm torch install still has torch
    opening the nodes, so the backend request alone cannot silence the diagnosis."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = False, backend = "cuda")
    assert "cannot open its device nodes" in out


def test_the_acl_sentence_names_every_path_it_lists(monkeypatch, linux):
    """The sentence lists every ACL-carrying node and then ran getfacl on the first one
    alone, so a user following it read one node's grant and was left with the second
    blocker undiagnosed. ROCm needs both nodes and their ACLs need not agree."""
    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(
        amd,
        "_groups_that_own",
        lambda paths: ([], [], [], ["/dev/kfd", "/dev/dri/renderD128"], [], [], []),
    )
    hint = amd.amd_node_permission_hint()
    assert "getfacl /dev/kfd /dev/dri/renderD128" in hint


def test_the_acl_sentence_is_unchanged_for_a_single_node(monkeypatch, linux):
    """The control: one path in, one path out, so the fix cannot have introduced a stray
    separator into the common case."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda paths: ([], [], [], ["/dev/kfd"], [], [], [])
    )
    hint = amd.amd_node_permission_hint()
    assert "getfacl /dev/kfd before" in hint


@pytest.mark.parametrize(
    "index_url",
    [
        "https://download.pytorch.org/whl/gfx-mirror",
        "https://example.invalid/wheels/rocm7.2-private/",
    ],
)
def test_a_custom_pin_is_not_read_as_a_rocm_route(index_url):
    """The gate globbed the raw URL for */rocm* and */gfx*, which matches exactly the
    custom pins _is_pip_rocm_family_leaf exists to reject -- a mirror named for an arch, or
    a private ROCm build -- so an install that is not on a published ROCm route was told to
    repair the AMD kernel stack. Classifying the leaf reuses that rejection."""
    assert _diag_route(index_url) is False


def test_the_real_gfx_route_is_still_read_as_one():
    """The control that keeps the rejection narrow: repo.radeon.com's arch leaf is a real
    ROCm route and must stay one, which the parametrized case above covers by URL and this
    one states as the rule."""
    assert _diag_route("https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151") is True


def test_a_vulkan_caller_is_not_told_to_map_the_kfd_node(monkeypatch, linux):
    """The mapping advice named both nodes whatever the caller was. Vulkan never opens
    /dev/kfd -- which is why _amd_nodes_the_runtime_lacks already excludes it under
    needs_kfd = False -- so this handed a container another host device for nothing."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    hint = amd.amd_node_permission_hint(needs_kfd = False)
    assert "--device /dev/dri." in hint
    assert "/dev/kfd" not in hint


def test_a_rocm_caller_is_still_told_to_map_both(monkeypatch, linux):
    """The control: HIP opens both, so the pair is right there and the fix must not
    narrow the advice for the caller it was written for."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    hint = amd.amd_node_permission_hint()
    assert "--device /dev/kfd --device /dev/dri." in hint


def test_a_no_torch_vulkan_installer_names_only_the_render_node(tmp_path):
    """The installer twin, on the run that reaches it: --no-torch with an explicit Vulkan
    bundle opens no /dev/kfd, so the device pair it prints must not name one."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    out = _install_sh_hint(str(node), render_present = False, skip_torch = True, backend = "vulkan")
    assert "Docker that is --device /dev/dri." in out
    assert "--device /dev/kfd" not in out


def test_an_ordinary_installer_run_still_names_both(tmp_path):
    """The control: a torch install opens /dev/kfd, so the pair stays."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    out = _install_sh_hint(str(node), render_present = False)
    assert "--device /dev/kfd --device /dev/dri." in out


def test_a_render_node_whose_vendor_cannot_be_read_is_not_called_absent(monkeypatch, linux):
    """A container can map /dev/dri and still mask the sysfs entry that names the vendor.
    Reading that unknown as "no AMD render node exists" told the user to recreate the
    container with --device /dev/dri, which that shape has already done."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/kfd", "/dev/dri/renderD128"},
        vendor_readable = False,
    )
    assert amd._amd_render_node_exists() is True
    assert amd.amd_node_permission_hint() is None


def test_a_host_with_no_render_node_at_all_still_says_so(monkeypatch, linux):
    """The control, and the reason the fix is "unknown reads as present" rather than
    "always present": a host whose glob finds nothing has nothing unreadable either, and
    the sentence #10466 needs must still be printed."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    assert amd._amd_render_node_exists() is False
    assert "No AMD render node" in amd.amd_node_permission_hint()


def test_another_vendors_open_render_node_is_seen(monkeypatch, linux):
    """The evidence the Vulkan caller needs: a node that was read, named another vendor,
    and opens. An unreadable one is not evidence either way and must not count."""
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        amd.glob, "glob", lambda pattern: ["/dev/dri/renderD128", "/dev/dri/renderD129"]
    )
    monkeypatch.setattr(
        amd,
        "_render_node_vendor",
        lambda p: "0x1002" if p.endswith("128") else "0x10de",
    )
    monkeypatch.setattr(amd.os, "access", lambda p, mode: p.endswith("129"))
    assert amd.a_non_amd_render_node_is_open() is True
    monkeypatch.setattr(amd, "_render_node_vendor", lambda p: None)
    assert amd.a_non_amd_render_node_is_open() is False


def test_a_vulkan_probe_keeps_its_finding_when_another_vendor_is_open(monkeypatch, linux):
    """A Vulkan-only build enumerates any vendor, so an open Intel or NVIDIA render node is
    a complete path for it: the closed AMD node cannot then be the whole reason the probe
    came back empty, and returning it alone dropped the finding that was."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")
    assert "Separately" in reason and "usermod" in reason


def test_the_same_host_with_no_other_vendor_still_returns_the_hint_alone(monkeypatch, linux):
    """The control: with no open node of any vendor the closed AMD one IS the reason, and
    the hint must still replace the bare probe sentence."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_a_rocm_build_is_unaffected_by_another_vendors_node(monkeypatch, linux):
    """The second control: HIP needs /dev/kfd and an AMD render node, which no other
    vendor's node substitutes for, so the question is not even asked for it."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = {"/dev/dri/renderD128"},
    )
    monkeypatch.setenv("USER", "ada")
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"hip"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "Separately" not in reason
    assert "usermod -a -G render,video ada" in reason


def test_the_repair_names_the_account_the_access_tests_answered_for(monkeypatch, linux):
    """USER is inherited, so a container that changes its numeric user without resetting it
    names somebody else and the command modifies the wrong account, leaving the running one
    still unable to open the node."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setenv("USER", "root")
    hint = amd.amd_node_permission_hint()
    assert hint.rstrip().endswith("render,video ada") or "render,video ada" in hint
    assert "render,video root" not in hint


def test_the_installer_names_the_account_id_reports(tmp_path):
    """The shell twin: `id -un` is the account the mode tests above answered for."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), env_user = "root", id_user = "ada")
    # The group is whatever owns a tmp_path file on the runner, so the account is what is
    # asserted -- naming a group here would be asserting about the runner.
    assert re.search(r"usermod -a -G \S+ ada", out)
    assert " root" not in out


def test_a_backend_value_with_internal_whitespace_is_not_a_backend():
    """The bundle selector normalizes with `awk '{$1=$1}'` -- trim and collapse, never
    delete -- and then REJECTS a value it does not recognise, falling back to automatic
    selection, which may install ROCm and open /dev/kfd. Deleting internal whitespace here
    made "vul kan" match instead, so the run went quiet about the very node that selection
    needs. Asserted through the KFD scope rather than the diagnosis as a whole, because the
    wider predicate matches cpu|cuda only and lets every Vulkan spelling through either
    way -- which is what made an earlier version of this test pass on both forms."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "vul kan")
    assert "cannot open its device nodes" in out
    assert "/dev/kfd" in out


def test_the_same_value_spelled_properly_is_still_a_backend():
    """The control, and the whitespace the selector DOES forgive: a padded, upper-case value
    names Vulkan, so the KFD scoping must still take it. Without this the rule could be
    "never recognise anything", which silently un-scopes every Vulkan install."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "  VULKAN  ")
    assert out.strip() == ""


def test_the_unnamed_gid_repair_is_a_command_a_shell_will_run(monkeypatch, linux):
    """`<name>` is not a placeholder in a shell, it is a redirection: `groupadd -g 993
    <name>` parses as a read from ./name followed by a `>` with no target, which bash, dash
    and sh all reject with a syntax error before groupadd runs. Every character of this
    sentence is meant to be pasted, so it must contain no shell metacharacter it does not
    mean. The name is derived from the GID, which by definition here has no group entry."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993, 994], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "<" not in hint and ">" not in hint
    assert "sudo groupadd -g 993 amdgpu993 && sudo usermod -a -G amdgpu993 ada" in hint
    assert "sudo groupadd -g 994 amdgpu994 && sudo usermod -a -G amdgpu994 ada" in hint


def test_the_installer_unnamed_gid_repair_is_runnable_too(tmp_path):
    """The shell twin, checked the same way and then actually parsed: `bash -n` on the two
    emitted lines is the assertion that a placeholder would fail. Without the parse this
    would only be testing that a string changed."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993")
    _cmds = [
        _line.strip()
        for _line in out.splitlines()
        if _line.strip().startswith("sudo group") or _line.strip().startswith("sudo usermod")
    ]
    assert _cmds, out
    for _cmd in _cmds:
        assert "<" not in _cmd and ">" not in _cmd
        _parsed = subprocess.run(["bash", "-n", "-c", _cmd], capture_output = True, text = True)
        assert _parsed.returncode == 0, f"{_cmd!r}: {_parsed.stderr}"


def test_a_render_node_the_installer_cannot_read_the_vendor_of_is_not_absent(tmp_path):
    """The shell twin of the Python rule, which round sixteen fixed on one side only. A
    container mapping /dev/dri while denying its sysfs attributes has a render node; calling
    that absence told the user to recreate the container with the device it already has.

    Only the two path ROOTS are substituted, so the logic under test is the shipped one.
    """
    lines = _install_sh_lines()
    fn = _shell_fn(lines, "_amd_render_node_present")
    fn = fn.replace("/dev/dri/renderD*", f"{tmp_path}/dev/dri/renderD*")
    fn = fn.replace("/sys/class/drm/", f"{tmp_path}/sys/class/drm/")
    (tmp_path / "dev/dri").mkdir(parents = True)
    (tmp_path / "dev/dri/renderD128").write_bytes(b"")
    # The sysfs directory exists and the vendor file does not, which is what a container
    # denying the attribute looks like from here.
    (tmp_path / "sys/class/drm/renderD128/device").mkdir(parents = True)
    out = subprocess.run(
        [
            "bash",
            "-c",
            fn + "\nif _amd_render_node_present; then echo PRESENT; else echo ABSENT; fi",
        ],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "PRESENT"


def test_a_render_node_the_installer_reads_as_another_vendor_is_still_absent(tmp_path):
    """The control, and the reason the rule is "unknown", not "any node": a readable vendor
    that is not AMD is a real answer, and treating it as presence would claim an AMD card on
    an NVIDIA-only host."""
    lines = _install_sh_lines()
    fn = _shell_fn(lines, "_amd_render_node_present")
    fn = fn.replace("/dev/dri/renderD*", f"{tmp_path}/dev/dri/renderD*")
    fn = fn.replace("/sys/class/drm/", f"{tmp_path}/sys/class/drm/")
    (tmp_path / "dev/dri").mkdir(parents = True)
    (tmp_path / "dev/dri/renderD128").write_bytes(b"")
    (tmp_path / "sys/class/drm/renderD128/device").mkdir(parents = True)
    (tmp_path / "sys/class/drm/renderD128/device/vendor").write_text("0x10de\n")
    out = subprocess.run(
        [
            "bash",
            "-c",
            fn + "\nif _amd_render_node_present; then echo PRESENT; else echo ABSENT; fi",
        ],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ABSENT"


def test_an_open_sibling_is_not_a_way_in_when_a_mask_selects_the_closed_one(monkeypatch, linux):
    """HIP_VISIBLE_DEVICES=0 narrows the runtime to one GPU, and nothing here maps a render
    node back to the index it was selected by, so an OPEN sibling may well belong to the GPU
    the mask excludes. Reading it as an alternative suppressed the repair for the node the
    run will actually use and sent the user to reinstall a runtime instead."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert amd.amd_closed_nodes_block_the_runtime() is True


def test_the_same_host_with_no_mask_still_credits_the_open_sibling(monkeypatch, linux):
    """The control, and the behaviour this must not undo: with no selector the runtime is
    free to use the open node, so the closed one is a second finding rather than the cause.
    Without this the rule could be "a closed node always blocks", which is what the open
    sibling test was added to stop."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    # _no_inherited_gpu_mask has already cleared all three.
    assert amd.amd_closed_nodes_block_the_runtime() is False


def test_an_empty_mask_is_not_a_mask(monkeypatch, linux):
    """The second control: an exported but empty variable narrows nothing, and reading it as
    a selector would keep the repair on every host that merely has the name exported."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "  ")
    assert amd.amd_closed_nodes_block_the_runtime() is False


def test_a_cpu_torch_index_with_a_vulkan_bundle_is_not_sent_after_kfd():
    """SKIP_TORCH is not the only run that opens no AMD node. An explicitly CPU index
    installs a wheel with no ROCm runtime in it, so with a non-ROCm bundle requested as well
    nothing in the install opens /dev/kfd -- and the early return declared otherwise purely
    because torch was being installed at all."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = "vulkan",
        torch_index = "https://download.pytorch.org/whl/cpu",
    )
    assert out.strip() == ""


def test_a_cpu_torch_index_alone_still_reports_a_closed_kfd():
    """The control, and the reason the backend still has to settle it: "cpu" is deliberately
    kept in the diagnosis route one layer up, because the GGUF bundle is chosen later and may
    be the ROCm one, which opens /dev/kfd exactly as ROCm torch would."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = None,
        torch_index = "https://download.pytorch.org/whl/cpu",
    )
    assert "cannot open its device nodes" in out
    assert "/dev/kfd" in out


def test_a_rocm_torch_index_is_unaffected_by_a_vulkan_bundle():
    """The second control: ROCm torch opens /dev/kfd whatever the bundle is, so the index has
    to be read rather than assumed. Without this the rule could be "any explicit non-ROCm
    backend suppresses", which silences the #10466 diagnosis for every ROCm torch install."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = "vulkan",
        torch_index = "https://download.pytorch.org/whl/rocm6.4",
    )
    assert "cannot open its device nodes" in out
    assert "/dev/kfd" in out


def test_an_unnamed_gid_zero_is_the_root_group_not_a_group_to_create(monkeypatch, linux):
    """A minimal container can own the node root:root and carry no group database entry for
    gid 0. The name lookup raises there, and filing that as an ordinary unnamed GID produced
    `sudo groupadd -g 0 amdgpu0` plus a usermod into the ROOT group -- a grant far beyond the
    GPU, and exactly what the privileged branch below the lookup exists to refuse. gid 0 is
    the root group whether or not the database can name it."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (0, 0o660, 0)}, {})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert privileged == ["root"]
    assert unnamed == [] and joinable == []


def test_an_unnamed_ordinary_gid_is_still_a_group_to_create(monkeypatch, linux):
    """The control, and the reason the rule is keyed on gid 0 rather than on the lookup
    failing: an unnamed NON-privileged GID is the documented container case, and it must
    still get its groupadd pair."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert unnamed == [993]
    assert privileged == [] and joinable == []


def test_the_installer_also_refuses_an_unnamed_gid_zero():
    """The shell twin, run through the real classifier rather than a stubbed repair line.
    `stat -c %G` prints UNKNOWN for a gid the group database cannot name, and the unnamed
    test ran first, so a root-owned node in a minimal container came back as `gid:0` and the
    installer printed groupadd -g 0 plus a usermod into the root group."""
    assert _install_sh_classify("660|UNKNOWN|0|/dev/kfd|0") == "privileged:root"


def test_the_installer_still_names_an_ordinary_unnamed_gid():
    """The control: the same shape one GID up is the documented container case and must
    still come back as a GID to create a group for."""
    assert _install_sh_classify("660|UNKNOWN|993|/dev/kfd|0") == "gid:993"


def test_the_installer_keeps_the_name_of_a_named_root_group():
    """The second control: gid 0 WITH an entry keeps the name it has, so the reordering did
    not turn every privileged node into the literal word root."""
    assert _install_sh_classify("660|wheel|0|/dev/kfd|0") == "privileged:wheel"


def test_a_vulkan_probe_ignores_a_hip_mask_when_a_sibling_is_open(monkeypatch, linux):
    """HIP's selectors are HIP's. Vulkan reads none of them -- which is what needs_kfd is
    for -- so a Vulkan caller on a masked host is still free to use the open sibling, and
    returning the permission hint as the sole cause would send an empty Vulkan probe after
    a group change that cannot fix it."""
    _nodes(
        monkeypatch,
        present = ["/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/dri/renderD129"},
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert amd.amd_closed_nodes_block_the_runtime(needs_kfd = False) is False


def test_a_hip_caller_on_the_same_masked_host_still_blocks(monkeypatch, linux):
    """The control: the same host, the same mask, asked by the caller the mask applies to.
    Without this the rule could be "ignore selectors", which undoes the fix that added
    them."""
    # /dev/kfd present AND open, or this returns True on the missing-node branch and says
    # nothing about the mask at all -- which is how the first version of this passed.
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert amd.amd_closed_nodes_block_the_runtime(needs_kfd = True) is True


def test_gpu_device_ordinal_is_a_selector_too(monkeypatch, linux):
    """ROCm's fourth visibility variable, which this repository already models in
    test_amd_smi_inventory_matches_hip.py and in llama_cpp.py's own selector check. Omitting
    it left one of the four narrowing the runtime while the open sibling was still credited
    as a way in."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = {"/dev/kfd", "/dev/dri/renderD129"},
    )
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    assert amd.amd_closed_nodes_block_the_runtime() is True


def test_a_suffixed_rocm_rel_leaf_is_somebody_elses_mirror():
    """repo.radeon.com publishes rocm-rel-6.4, rocm-rel-6.5.0, rocm-rel-7.2.1 -- digits and
    dots after the prefix, nothing else. A pin that merely starts with it is a mirror, and
    it may serve wheels that open no AMD node, so it gets the same anchoring the sibling
    rocm[0-9]* arm already applies to rocm7.2-private."""
    assert _install_sh_diag_route("rocm-rel-7.0-private") == "false"
    assert _install_sh_diag_route("rocm-rel-7.0.beta") == "false"


def test_every_real_rocm_rel_leaf_still_takes_the_amd_route():
    """The control, and the reason this is anchored on the character class rather than on a
    fixed version shape: all six rocm-rel leaves this repository names must keep routing, two
    and three components alike."""
    for _leaf in (
        "rocm-rel-6.1",
        "rocm-rel-6.4",
        "rocm-rel-6.5.0",
        "rocm-rel-7.0",
        "rocm-rel-7.2.1",
        "rocm-rel-7.3.1",
    ):
        assert _install_sh_diag_route(_leaf) == "true", _leaf


def test_a_suffixed_gfx_leaf_is_not_narrowed_the_same_way():
    """The second control, and the reason the gfx family was deliberately left alone: AMD's
    own indexes are gfx110X-all, gfx120X-all and gfx103X-all, so a suffix there is the
    naming convention. Anchoring gfx the way rocm-rel is anchored would drop the routes this
    repository ships."""
    for _leaf in ("gfx110X-all", "gfx120X-all", "gfx103X-all", "gfx1151"):
        assert _install_sh_diag_route(_leaf) == "true", _leaf


def test_the_passwd_stub_answers_a_positional_read(monkeypatch):
    """getpass.getuser() reads pwd.getpwuid(os.getuid())[0] once none of LOGNAME, USER,
    LNAME or USERNAME is set, and pytest calls it while building tmp_path. A stub that only
    carries pw_name raises TypeError there, which pytest does not catch, so the whole suite
    would die in fixture setup on a runner with no username in its environment rather than
    run. Reproduced by clearing all four, which is the only thing that makes it reachable."""
    for _var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(_var, raising = False)
    assert getpass.getuser() == "ada"


def test_an_empty_hip_mask_defers_to_the_cuda_one_below_it(monkeypatch, linux):
    """The precedence test in clr is on the first BYTE of the value, not on whether the
    variable exists: its flag defaults to the empty string, so an empty HIP mask reads
    exactly like an unset one and the CUDA value below it is what selects devices. That
    value names a device here, so nothing is hidden and no mask is a second blocker.

    Fails before the fix, which took an empty HIP mask as the winner of the chain and
    reported it as hiding every device."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "0"},
        {"hip"},
    )
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_the_cuda_mask_under_an_empty_hip_one_still_blocks_when_it_hides(monkeypatch, linux):
    """The control, one value apart: deferring to CUDA is only right if CUDA is then
    judged on its own merits, so the same shape with a CUDA value that names no device
    must still be reported. Without this the fix could be "an empty HIP mask silences the
    whole chain"."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "-1"},
        {"hip"},
    )
    assert "CUDA_VISIBLE_DEVICES='-1'" in reason
    assert "visibility mask is also in force" in reason


def test_the_group_derivation_survives_a_root_test_runner(monkeypatch, linux):
    """CI commonly runs as root, and the fakes in this file give their nodes the root
    owner a real device node has. POSIX resolves the owner class exclusively once the uid
    matches, so a harness that let the two coincide would answer every group test with
    "this account owns it" and assert nothing about the classification under test.

    Fails before the fix with os.getuid patched to 0, which is what a root runner is."""
    monkeypatch.setattr(amd.os, "getuid", lambda: 0)
    _stat_nodes(monkeypatch, {"/dev/kfd": (44, 0o660, 0)}, {44: "render"})
    joinable, unnamed, no_group, acl, owned, privileged, _already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert joinable == ["render"]
    assert owned == []


def test_the_unnamed_gid_repair_says_to_start_a_new_session(monkeypatch, linux):
    """usermod changes /etc/group, not the groups the running login already holds, so the
    named-group branch has always ended "then log out and back in". The unnamed-GID branch
    prescribes the same usermod and said nothing, so an immediate retry fails and the
    command reads as the one that did not work."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "groupadd -g 993 amdgpu993" in hint
    assert "log out and back in" in hint


_KFD_GPU_NODE = "vendor_id 4098\nsimd_count 8\n"
_KFD_CPU_NODE = "vendor_id 0\nsimd_count 0\n"


def _kfd_topology(monkeypatch, entries: dict):
    """Stub /sys/class/kfd: entry name -> its properties text, or None for unreadable."""
    monkeypatch.setattr(amd.os, "listdir", lambda _path: list(entries))

    def _open(path, *_a, **_k):
        _text = entries.get(os.path.basename(os.path.dirname(str(path))))
        if _text is None:
            raise OSError("unreadable")
        return io.StringIO(_text)

    monkeypatch.setattr(amd, "open", _open, raising = False)


def test_the_gpu_count_reads_the_topology(monkeypatch):
    """The control for the arm below, and for the helper itself: the CPU node every KFD
    topology carries is excluded and the GPU node is counted."""
    _kfd_topology(monkeypatch, {"0": _KFD_CPU_NODE, "1": _KFD_GPU_NODE})
    assert amd.amd_kfd_gpu_node_count() == 1


def test_an_unreadable_topology_entry_makes_the_whole_count_unknown(monkeypatch):
    """One entry temporarily unreadable on a two-GPU host used to answer 1 rather than
    unknown, and an understated bound is what calls a valid selector a blocker: with a
    count of 1, HIP_VISIBLE_DEVICES=1 reads as hiding every device and the user is told to
    clear a mask that hides nothing. Unknown bounds nothing, which is the documented
    contract and what _amd_render_node_exists already does with an unreadable vendor."""
    _kfd_topology(monkeypatch, {"0": _KFD_GPU_NODE, "1": None, "2": _KFD_GPU_NODE})
    assert amd.amd_kfd_gpu_node_count() is None


def test_an_unusable_hip_selector_is_a_blocker(monkeypatch, linux):
    """clr's list terminates at the first token it cannot use, and a token does not have
    to look numeric to be unusable: rocdevice.cpp takes `index = atoi(str_id)` and rejects
    it unless `str_id` is that index written back out. So HIP_VISIBLE_DEVICES=garbage
    leaves zero agents exactly as -1 does, and joining the group leaves the probe as empty
    as it was.

    Fails before the fix, which asked only whether the first token was a digit and let
    everything else through as "not a filter"."""
    reason = _reason_with_masks(monkeypatch, {"HIP_VISIBLE_DEVICES": "garbage"}, {"hip"})
    assert "HIP_VISIBLE_DEVICES='garbage'" in reason
    assert "visibility mask is also in force" in reason


def test_the_installer_scopes_the_claim_when_a_sibling_node_is_open(tmp_path):
    """The Python half already says an open sibling means the closed nodes stop the card
    rather than the host, and the installer claimed every backend was blocked regardless --
    on a host where a Vulkan run is working through the open node as the user reads it.

    Fails before the fix, which printed the unconditional claim."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), render_open = True)
    assert "another AMD" in out and "render node on this host is open" in out
    # The repair is unchanged: these nodes are still shut and membership still opens them.
    assert "usermod -a -G" in out


def test_the_installer_still_claims_every_backend_when_none_is_open(tmp_path):
    """The control, and the #10466 host: with no AMD node open anywhere the closed set does
    block every backend, so the fix must not soften the claim into always saying maybe."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), render_open = False)
    assert "Every backend needs them, ROCm and Vulkan alike." in out
    assert "render node on this host is open" not in out


def test_a_no_torch_vulkan_run_still_diagnoses_its_render_nodes():
    """--no-torch installs no wheel, so the index resolved above this gate describes
    nothing that will run; on a hybrid host it is the CUDA one, and reading it as the route
    silenced every node diagnosis for a run whose Vulkan bundle opens the very render node
    they are about.

    Fails before the fix, which read the unused index."""
    assert (
        _diag_route("https://download.pytorch.org/whl/cu128", skip_torch = True, backend = "vulkan")
        is True
    )


def test_a_no_torch_cpu_backend_run_still_does_not():
    """The control: --no-torch with a CPU llama.cpp bundle opens no AMD node at all, so the
    diagnoses stay off. Without it the fix could be "--no-torch always diagnoses"."""
    assert (
        _diag_route("https://download.pytorch.org/whl/cu128", skip_torch = True, backend = "cpu")
        is False
    )


def test_a_cuda_wheel_install_is_still_off_the_route():
    """The other control, and the boundary the case was written for: a run installing CUDA
    wheels stays off the route as long as its bundle opens no AMD node either, so the fix is
    scoped to the run that installs nothing rather than reopening the arm above. An explicit
    rocm or vulkan request is the one case that does reopen it, and it has its own arm
    below the override."""
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = "cuda") is False
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = "cpu") is False


def test_an_icd_override_stops_another_vendors_node_from_excusing_the_amd_one(
    monkeypatch, linux, tmp_path
):
    """VK_DRIVER_FILES REPLACES the loader's driver search rather than adding to it, and
    the probe child inherits it, so a list naming AMD alone means the loader never opened
    the other vendor's driver. Crediting its node then suppressed the closed-node hint on
    a run that had no other path -- a wrong suppression, which hides a real repair.

    Fails before the fix, which asked only whether the node was open."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.setenv("VK_DRIVER_FILES", _icd_manifest(tmp_path, "radeon_icd.json"))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_the_deprecated_spelling_of_that_override_counts_too(monkeypatch, linux, tmp_path):
    """VK_ICD_FILENAMES is the deprecated name for the same replacing list, and the loader
    still honours it when VK_DRIVER_FILES is unset."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.delenv("VK_DRIVER_FILES", raising = False)
    monkeypatch.setenv("VK_ICD_FILENAMES", _icd_manifest(tmp_path, "radeon_icd.json"))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "the Vulkan probe reported no device" not in reason


def test_the_additive_icd_variable_leaves_the_other_vendor_credited(monkeypatch, linux):
    """The control, and the reason the two are not treated alike: VK_ADD_DRIVER_FILES ADDS
    to the standard search, so every driver the loader would have found is still found and
    the other vendor's open node is still a complete path. Without this the fix could be
    "any VK_ variable suppresses", which turns the finding off for a host that set the one
    variable that changes nothing about which vendors load."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    for _var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        monkeypatch.delenv(_var, raising = False)
    monkeypatch.setenv("VK_ADD_DRIVER_FILES", "/opt/extra/icd.json")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")
    assert "Separately" in reason


def test_an_empty_icd_override_is_not_an_override(monkeypatch, linux):
    """The second control: the loader treats an empty value as unset, so an exported but
    blank VK_DRIVER_FILES must not suppress the other vendor either."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising = False)
    monkeypatch.setenv("VK_DRIVER_FILES", "")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")


def test_an_icd_list_naming_another_vendor_does_not_suppress_the_vulkan_finding(monkeypatch, linux):
    """The other edge of the same rule. A list pinned to an Intel or NVIDIA ICD leaves the
    loader unable to use the AMD card at all, so its closed node cannot be why the probe
    was empty -- and returning the group repair alone sends the user after a change that
    cannot help. Only an AMD-only list is evidence that the other vendor is unreachable.

    Fails before the fix, which read any non-empty list as AMD's."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising = False)
    monkeypatch.setenv("VK_DRIVER_FILES", "/etc/vulkan/icd.d/intel_icd.x86_64.json")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")
    assert "Separately" in reason and "usermod" in reason


def test_a_list_carrying_both_vendors_does_not_suppress_either(monkeypatch, linux):
    """A list is AMD-only or it is not; one non-AMD entry leaves that vendor loadable and
    its open node a complete path, whatever else the list names."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising = False)
    monkeypatch.setenv(
        "VK_DRIVER_FILES",
        os.pathsep.join(
            [
                "/etc/vulkan/icd.d/radeon_icd.x86_64.json",
                "/etc/vulkan/icd.d/nvidia_icd.json",
            ]
        ),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")


def test_an_unclassifiable_icd_entry_keeps_the_unpinned_behaviour(monkeypatch, linux):
    """A directory, or a name this does not recognise, is not evidence of anything. The
    guard fires on positive evidence alone, so an unreadable list leaves the finding
    exactly as it is without one -- rather than suppressing on a guess."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising = False)
    monkeypatch.setenv("VK_DRIVER_FILES", "/opt/vendor/icd.d")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert reason.startswith("the Vulkan probe reported no device")


def test_every_amd_icd_spelling_the_installer_knows_counts_here_too():
    """The two lists are the same convention twice, and a driver named in one but not the
    other would make the installer and this diagnosis disagree about the same host."""
    import install_llama_prebuilt
    assert amd._AMD_VULKAN_ICD_NEEDLES == install_llama_prebuilt._AMD_VULKAN_ICD_NEEDLES


def test_a_blocking_hip_mask_cancels_the_verdict_before_any_node_advice(monkeypatch):
    """A mask that hides every accelerator is the configuration working, so the whole
    chat-only classification is cancelled and no message -- node repair included -- is
    built. This is the first of the two gates that keep a blocking selector away from the
    node hint, and it sits well above the code that returns it."""
    from utils.hardware import hardware

    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda **_kw: {"devices": [{"vendor": "amd"}], "unknown": False},
    )
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)
    assert hardware._masks_hide_every_accelerator(block_inventory = True) is False
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    assert hardware._masks_hide_every_accelerator(block_inventory = True) is True
    assert hardware.classify_torch_build(block_inventory = True) is None


def test_a_blocking_mask_drops_amd_from_the_vendors_the_node_hint_needs(monkeypatch):
    """The second gate, for the hybrid host the first one does not cover: an NVIDIA card
    still raises the verdict, but the masked AMD card is dropped from the inventory that
    establishes it, so "amd" never reaches CHAT_ONLY_MISMATCH_VENDORS and the node hint is
    not even computed. An empty active CUDA mask is the same story, since HIP reads that
    variable too."""
    from utils.hardware import hardware

    devices = [{"vendor": "amd"}, {"vendor": "nvidia"}]
    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda **_kw: {"devices": devices, "unknown": False},
    )
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)
    kept = hardware._devices_that_can_establish_a_mismatch(devices)
    assert {device["vendor"] for device in kept} == {"amd", "nvidia"}
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    kept = hardware._devices_that_can_establish_a_mismatch(devices)
    assert {device["vendor"] for device in kept} == {"nvidia"}
    # HIP reads CUDA_VISIBLE_DEVICES too, so an emptied one hides both cards rather than
    # just the NVIDIA half -- nothing establishes the mismatch at all and the verdict is
    # cancelled a step earlier. Either way "amd" is not among the vendors, which is the
    # invariant the node hint is gated on.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    kept = hardware._devices_that_can_establish_a_mismatch(devices)
    assert {device["vendor"] for device in kept} == set()


def _icd_manifest(
    tmp_path,
    name,
    *,
    library = "libvulkan_radeon.so",
    present = True,
):
    """A Vulkan ICD manifest on disk, and the path to it.

    Written rather than named, because the rule under test is that a manifest has to point
    at a library that is actually there: a path string alone proves nothing.
    """
    lib = tmp_path / library
    if present:
        lib.write_bytes(b"")
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": str(lib), "api_version": "1.3.0"},
            }
        ),
        encoding = "utf-8",
    )
    return str(path)


def _vulkan_reason_under_icd_list(
    monkeypatch,
    value,
    *,
    var = "VK_DRIVER_FILES",
    search_dirs = None,
    filters = None,
):
    """The empty-probe reason for a Vulkan build with the AMD node shut and another
    vendor's node open, under a given loader configuration.

    ``value`` is a forced driver list, or None for a host that has none and is answered by
    the loader's own search; ``search_dirs`` stands in for that search, so an arm cannot be
    decided by whatever drivers the runner happens to have installed."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    for _var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES"):
        monkeypatch.delenv(_var, raising = False)
    if filters is not None or search_dirs is not None:
        # Cleared only for the arms that state their own loader configuration, since an arm
        # that names a filter is testing that filter and must keep it.
        for _var in ("VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE"):
            monkeypatch.delenv(_var, raising = False)
    for _var, _value in (filters or {}).items():
        monkeypatch.setenv(_var, _value)
    if search_dirs is not None:
        monkeypatch.setattr(amd, "_vulkan_icd_search_dirs", lambda: list(search_dirs))
    if value is not None:
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_an_amd_manifest_whose_library_is_gone_is_not_a_driver(monkeypatch, linux, tmp_path):
    """A registration with no library behind it loads nothing, so a list holding only that
    leaves the loader with no driver at all -- and then the closed AMD node is not why the
    probe was empty either. The filename says AMD; the manifest says nothing is there.

    Fails before the fix, which read the name alone."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.json", present = False)
    reason = _vulkan_reason_under_icd_list(monkeypatch, manifest)
    assert reason.startswith("the Vulkan probe reported no device")


def test_an_amd_manifest_that_is_not_there_at_all_is_not_a_driver(monkeypatch, linux):
    """The same for the manifest itself, which is the shape a stale VK_DRIVER_FILES has
    after a driver is uninstalled."""
    reason = _vulkan_reason_under_icd_list(monkeypatch, "/nonexistent/icd.d/radeon_icd.x86_64.json")
    assert reason.startswith("the Vulkan probe reported no device")


def test_an_amd_driver_the_loader_filters_out_is_not_a_driver(monkeypatch, linux, tmp_path):
    """VK_LOADER_DRIVERS_DISABLE applies to a forced list too, so naming AMD and then
    disabling it leaves the loader with nothing. The manifest here is entirely valid; only
    the filter makes it unloadable, which is what separates this from the two arms above."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.setenv("VK_LOADER_DRIVERS_DISABLE", "radeon*")
    reason = _vulkan_reason_under_icd_list(monkeypatch, manifest)
    assert reason.startswith("the Vulkan probe reported no device")


def test_a_valid_amd_manifest_still_suppresses(monkeypatch, linux, tmp_path):
    """The control for all three: a manifest that is present, parses, points at a library
    that exists and survives the loader's filters IS an AMD-only driver list, and the other
    vendor's node is then unreachable. Without it the fix could be "never suppress"."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.delenv("VK_LOADER_DRIVERS_DISABLE", raising = False)
    reason = _vulkan_reason_under_icd_list(monkeypatch, manifest)
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_the_loader_filter_rule_matches_the_installers(tmp_path):
    """Both halves implement the loader's four globs, and a host where they disagree gets
    one answer from the Vulkan route and another from this diagnosis."""
    import install_llama_prebuilt

    cases = [
        ("radeon_icd.x86_64.json", "radeon*"),
        ("radeon_icd.x86_64.json", "*radeon*"),
        ("radeon_icd.x86_64.json", "*json"),
        ("radeon_icd.x86_64.json", "radeon_icd.x86_64.json"),
        ("radeon_icd.x86_64.json", "nvidia*"),
        ("nvidia_icd.json", "radeon*"),
    ]
    for name, pattern in cases:
        for var in ("VK_LOADER_DRIVERS_DISABLE", "VK_LOADER_DRIVERS_SELECT"):
            os.environ.pop("VK_LOADER_DRIVERS_DISABLE", None)
            os.environ.pop("VK_LOADER_DRIVERS_SELECT", None)
            os.environ[var] = pattern
            try:
                assert amd._vulkan_loader_allows(name) == (
                    install_llama_prebuilt._vulkan_loader_allows(name)
                ), (name, pattern, var)
            finally:
                os.environ.pop(var, None)


def _blocks_under_selector(
    monkeypatch,
    value,
    *,
    count,
    var = "HIP_VISIBLE_DEVICES",
    also = None,
):
    """Whether a closed render node blocks a HIP runtime, with one sibling open.

    ``also`` sets a second selector, since which of two the runtime reads is itself a
    question here."""
    _nodes(
        monkeypatch,
        present = ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/kfd"],
        openable = {"/dev/dri/renderD129", "/dev/kfd"},
    )
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: count)
    for _name in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(_name, raising = False)
    if value is not None:
        monkeypatch.setenv(var, value)
    for _name, _value in (also or {}).items():
        monkeypatch.setenv(_name, _value)
    return amd.amd_closed_nodes_block_the_runtime()


def test_a_selector_naming_every_gpu_leaves_the_open_sibling_as_evidence(monkeypatch, linux):
    """HIP_VISIBLE_DEVICES=0,1 on a two-GPU host selects the whole host, so the open
    sibling is still a complete ROCm path and the closed node is a second finding rather
    than the cause. Reading any selector as a narrowing handed that host the group repair
    in place of the driver diagnosis it needs.

    Fails before the fix, which asked only whether a selector was set."""
    assert _blocks_under_selector(monkeypatch, "0,1", count = 2) is False


def test_a_selector_naming_one_of_them_still_discards_the_sibling(monkeypatch, linux):
    """The control, and the reason the rule exists: nothing here maps a render node back to
    the index a selector chose it by, so under a real narrowing the open node may be the
    excluded GPU's and stops being evidence."""
    assert _blocks_under_selector(monkeypatch, "0", count = 2) is True


def test_a_selector_this_cannot_map_still_discards_it(monkeypatch, linux):
    """A UUID names a device by identity, and an unreadable GPU count answers nothing.
    Both leave the selector unmapped, and unmapped goes on meaning narrowed."""
    assert _blocks_under_selector(monkeypatch, "GPU-abcdef0123456789", count = 2) is True
    assert _blocks_under_selector(monkeypatch, "0,1", count = None) is True


def test_no_selector_at_all_still_keeps_the_sibling(monkeypatch, linux):
    """The second control: with nothing set the sibling was always evidence, and the fix
    must not have made every host look narrowed."""
    assert _blocks_under_selector(monkeypatch, None, count = 2) is False


def test_a_hybrid_host_is_not_told_to_repair_the_card_its_bundle_will_not_use(tmp_path):
    """An unset or `auto` backend is resolved by _linux_published_attempts, which takes the
    CUDA bundle under `if host.has_usable_nvidia:` and reaches ROCm only in the `elif
    host.has_rocm` below it. So on a hybrid box with a usable NVIDIA GPU neither a CPU
    torch nor the automatic bundle opens an AMD node, and the AMD-evidence gates cannot
    tell -- the card is there and its nodes are shut, they are just unused.

    Fails before the fix, which read an unresolved `auto` as "may open"."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(
        str(node),
        torch_index = "https://download.pytorch.org/whl/cpu",
        nvidia = True,
    )
    assert out.strip() == ""


def test_the_same_host_with_an_explicit_rocm_request_is_still_told(tmp_path):
    """The control: an explicit rocm request IS a decision, and the resolver honours it, so
    the nodes that bundle opens are the user's problem to repair whatever else is on the
    bus."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(
        str(node),
        torch_index = "https://download.pytorch.org/whl/cpu",
        backend = "rocm",
        nvidia = True,
    )
    assert "cannot open its device nodes" in out


def test_an_amd_only_host_on_the_same_route_is_still_told(tmp_path):
    """The second control, and the #10466 host: with no NVIDIA GPU the automatic bundle may
    well be the ROCm one, so nothing about `auto` is a reason to go quiet."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), torch_index = "https://download.pytorch.org/whl/cpu")
    assert "cannot open its device nodes" in out


def test_a_rocm_torch_index_ignores_the_nvidia_card_entirely(tmp_path):
    """The third control: a run installing ROCm wheels opens AMD nodes whatever bundle is
    chosen later, so the bundle question is never reached."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), nvidia = True)
    assert "cannot open its device nodes" in out


def test_the_installer_does_not_usermod_a_uid_with_no_passwd_entry(tmp_path):
    """The shell half of the same item. `id -un` fails outright for a uid the passwd
    database does not know, and the old fallback then named the inherited $USER -- which in
    a container commonly still says root, so the command would succeed against an identity
    nothing is running as and leave the node shut.

    Fails before the fix, which fell back to ${USER}."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), id_user = None, env_user = "root")
    assert "usermod -a -G" not in out and "root" not in out
    assert "--group-add" in out


def test_the_installer_still_names_an_account_the_system_knows(tmp_path):
    """The control: with a passwd entry the command is the repair, exactly as before."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node))
    assert "sudo usermod -a -G" in out
    assert "--group-add" not in out


def test_the_installers_unnamed_gid_repair_drops_its_groupadd_half_too(tmp_path):
    """The unnamed-GID branch prints a groupadd AND a usermod, and the second needs the
    same account. With no passwd entry the container flag is the whole repair."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), id_user = None, env_user = "root", repairs = "gid:993")
    assert "usermod -a -G" not in out and "groupadd -g" not in out
    assert "--group-add 993" in out


def test_a_repeated_rocr_token_ends_the_list_and_narrows(monkeypatch, linux):
    """ROCr's filter terminates on a token naming a device it has already selected, so
    ROCR_VISIBLE_DEVICES=0,0,1 surfaces ONE GPU on a two-GPU host. Counting distinct
    ordinals read that as selecting the whole host and kept the open sibling as evidence
    for a runtime that can no longer reach it.

    Fails before the fix, which accumulated a set with no repeat rule."""
    assert _blocks_under_selector(monkeypatch, "0,0,1", count = 2, var = "ROCR_VISIBLE_DEVICES") is True


def test_the_same_repeat_under_hip_does_not_narrow(monkeypatch, linux):
    """The control, and the reason the rule is per layer rather than global: clr's parser
    stops only on a token that is not its own index written back out, so it accepts the
    repeat and both GPUs survive. A repeat rule applied everywhere would hand this host the
    group repair instead of the driver diagnosis."""
    assert _blocks_under_selector(monkeypatch, "0,0,1", count = 2, var = "HIP_VISIBLE_DEVICES") is False


def test_a_cuda_selector_under_a_hip_one_is_shadowed(monkeypatch, linux):
    """The HIP layer reads HIP_VISIBLE_DEVICES when it is non-empty and
    CUDA_VISIBLE_DEVICES only otherwise, so a CUDA value set beneath a HIP one selects
    nothing and narrows nothing. Reading all four side by side let the shadowed value
    discard an open sibling the runtime can still reach.

    Fails before the fix, which asked every name independently."""
    assert (
        _blocks_under_selector(monkeypatch, "0,1", count = 2, also = {"CUDA_VISIBLE_DEVICES": "0"})
        is False
    )


def test_a_cuda_selector_on_its_own_still_narrows(monkeypatch, linux):
    """The control: with no HIP value the CUDA one IS the HIP layer's selector, which is
    the precedence _explain_empty_gpu_probe's _hip_layer_var already applies. A rule that
    simply stopped reading CUDA would lose every host masked that way."""
    assert (
        _blocks_under_selector(monkeypatch, None, count = 2, also = {"CUDA_VISIBLE_DEVICES": "0"})
        is True
    )


def test_a_filter_that_leaves_only_amd_is_the_same_as_naming_it(monkeypatch, linux, tmp_path):
    """VK_LOADER_DRIVERS_SELECT is applied to whatever the loader would load, forced list
    or search, so a host with no list at all can still be pinned to AMD alone. Reading only
    the two force-list variables left that host crediting the other vendor's open node to a
    binary whose loader never opens that vendor's driver.

    Fails before the fix, which asked what a forced list was named rather than what the
    loader would load."""
    amd_icd = _icd_manifest(tmp_path, "radeon_icd.x86_64.json", library = "libamd.so")
    other = _icd_manifest(tmp_path, "nvidia_icd.json", library = "libnv.so")
    assert amd_icd and other
    reason = _vulkan_reason_under_icd_list(
        monkeypatch,
        None,
        search_dirs = [str(tmp_path)],
        filters = {"VK_LOADER_DRIVERS_SELECT": "radeon*"},
    )
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_the_same_two_drivers_unfiltered_still_credit_the_other_vendor(
    monkeypatch, linux, tmp_path
):
    """The control, on the same two manifests: with no filter the loader loads both, the
    other vendor's open node IS a path for this binary, and the closed AMD node cannot be
    why the probe came back empty. Without it the rule could be "a search always means
    AMD only", which suppresses the finding on every host."""
    _icd_manifest(tmp_path, "radeon_icd.x86_64.json", library = "libamd.so")
    _icd_manifest(tmp_path, "nvidia_icd.json", library = "libnv.so")
    reason = _vulkan_reason_under_icd_list(monkeypatch, None, search_dirs = [str(tmp_path)])
    assert reason.startswith("the Vulkan probe reported no device")


def test_a_search_that_enumerates_nothing_answers_nothing(monkeypatch, linux, tmp_path):
    """Positive evidence only. An empty search is not "AMD alone", it is a loader this
    cannot read, and a loader with no driver at all explains the empty probe by itself --
    so the closed AMD node is not the cause either and must not be suppressed."""
    reason = _vulkan_reason_under_icd_list(
        monkeypatch, None, search_dirs = [str(tmp_path / "nothing-here")]
    )
    assert reason.startswith("the Vulkan probe reported no device")


def test_the_search_dirs_follow_the_xdg_variables(monkeypatch):
    """The loader falls back to its defaults only when a variable is unset, so reading the
    defaults regardless both misses a custom layout's only manifest and counts stale ones
    the loader would never read. install_llama_prebuilt._vulkan_icd_search_dirs is the same
    list, and the test below holds them together."""
    monkeypatch.setenv("XDG_DATA_DIRS", "/opt/one:/opt/two")
    monkeypatch.setenv("XDG_CONFIG_DIRS", "/opt/conf")
    dirs = amd._vulkan_icd_search_dirs()
    assert "/opt/one/vulkan/icd.d" in dirs
    assert "/opt/two/vulkan/icd.d" in dirs
    assert "/opt/conf/vulkan/icd.d" in dirs
    assert "/usr/share/vulkan/icd.d" not in dirs
    assert "/etc/xdg/vulkan/icd.d" not in dirs
    assert dirs.count("/etc/vulkan/icd.d") == 1


def test_the_search_dirs_match_the_installers(monkeypatch):
    """The installer resolves the same question for the same loader, so the two lists are
    one list; a copy that drifts sends the two halves of this diagnosis to different
    drivers."""
    import install_llama_prebuilt

    monkeypatch.setenv("XDG_DATA_DIRS", "/opt/one:/opt/two")
    monkeypatch.setenv("XDG_CONFIG_DIRS", "/opt/conf")
    assert amd._vulkan_icd_search_dirs() == [
        str(directory) for directory in install_llama_prebuilt._vulkan_icd_search_dirs()
    ]


def test_an_unrecognised_backend_is_the_automatic_route(tmp_path):
    """setup.sh warns "Ignoring UNSLOTH_LLAMA_CPP_BACKEND=..." for anything outside its
    list and the installer normalises it to auto, so a typo installs the automatically
    chosen bundle -- which on a hybrid box is CUDA, opening no AMD node. Listing the two
    spellings of "no decision" instead of the values that ARE decisions let a rejected
    value pose as one.

    Fails before the fix, which matched "" and auto alone."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "vul kan", nvidia = True)
    assert out.strip() == ""


def test_an_explicit_rocm_request_survives_a_usable_nvidia_gpu(tmp_path):
    """The control, and the whole point of listing decisions: an explicit rocm request is
    honoured by the resolver, so its bundle opens /dev/kfd on a hybrid box exactly as it
    would on an AMD-only one. A rule that read every hybrid host as CUDA would silence the
    #10466 diagnosis for the users who asked for ROCm."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "rocm", nvidia = True)
    assert "cannot open its device nodes" in out
    assert "/dev/kfd" in out


def test_the_same_rejected_value_on_an_amd_only_host_still_reports(tmp_path):
    """The second control: the automatic route is only silent where it resolves away from
    AMD. With no NVIDIA card it resolves to ROCm, and the closed node is the diagnosis."""
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = "vul kan", nvidia = False)
    assert "/dev/kfd" in out


def _nvidia_probe_calls(backend = None):
    """How many times the two scope predicates run the NVIDIA probe for one install."""
    lines = _install_sh_lines()
    script = "\n".join(
        [
            "SKIP_TORCH=true",
            "TORCH_INDEX_URL=''",
            f"export UNSLOTH_LLAMA_CPP_BACKEND={backend or ''!r}",
            "_probe_calls=0",
            "_has_usable_nvidia_gpu() { _probe_calls=$((_probe_calls + 1)); return 1; }",
            _shell_fn(lines, "_torch_index_url_leaf"),
            # _torch_opens_amd_nodes classifies a ROCm index through this,
            # so lifting one without the other measures a missing function.
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_kfd"),
            # The block quotes every name it interpolates into a pasted command through
            # this. Lifted rather than stubbed: without it the substitutions come back
            # EMPTY and the arms below read as commands that name nobody.
            _shell_fn(lines, "_shell_quote"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            "for _i in 1 2 3 4; do _run_may_open_kfd; _run_may_open_a_gpu_node; done",
            'echo "$_probe_calls"',
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
    return int(out.stdout.strip())


def test_the_nvidia_probe_runs_once_per_install():
    """_has_usable_nvidia_gpu shells out to a bounded `nvidia-smi -L` on every call and is
    not memoized, while the two scope predicates are consulted at every diagnosis. Nothing
    it reads changes within a run, so an NVIDIA-less host was paying a subprocess per gate.

    Fails before the fix, which probed on every call."""
    assert _nvidia_probe_calls() == 1


def test_an_explicit_backend_never_probes_at_all():
    """The control: a decision the resolver honours settles the question without asking
    about the other vendor's hardware, so the memo is not merely cheaper, it is unreached."""
    assert _nvidia_probe_calls("rocm") == 0


def test_an_explicit_gpu_bundle_keeps_the_node_diagnoses(tmp_path):
    """The route is derived from the torch index alone, so a CUDA-pinned index asked for
    the ROCm bundle read as a CUDA route and silenced all three diagnoses -- for a run
    whose bundle opens the very nodes they are about. The SKIP_TORCH override below the
    case could not catch it, since it only runs when no wheel is installed at all.

    Fails before the fix, which had no arm for the backend request here."""
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = "rocm") is True
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = "vulkan") is True


def test_a_cpu_bundle_on_the_same_index_still_stays_quiet(tmp_path):
    """The control, twice over: the new arm only ever turns the route ON, so a cpu request
    and no request at all are both left to the case above and to the two scope predicates,
    which is where a bundle that opens no AMD node belongs."""
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = "cpu") is False
    assert _diag_route("https://download.pytorch.org/whl/cu128") is False


def test_a_hybrid_rocm_route_is_not_told_to_install_the_kernel_stack(tmp_path):
    """_has_amd_rocm_gpu opens with `if _has_usable_nvidia_gpu; then return 1`, which is
    right where it is choosing a torch index and wrong here: this branch has already
    established the run opens AMD nodes, so the veto made rocminfo's answer unreachable and
    a healthy hybrid ROCm host was told to install the ROCm kernel stack it already has.

    Fails before the fix, which read the wrapped probe."""
    out = _install_sh_missing_kfd(topology = False, nvidia = True, backend = "rocm", rocm_visible = True)
    assert "ROCm cannot see it" not in out


def test_the_same_host_without_rocm_still_gets_the_kernel_stack_hint(tmp_path):
    """The control: an AMD card on the bus that ROCm genuinely cannot see is exactly what
    the sentence is for, and dropping the NVIDIA veto must not have dropped the finding."""
    out = _install_sh_missing_kfd(topology = False, nvidia = True, backend = "rocm", rocm_visible = False)
    assert "ROCm cannot see it" in out


def test_an_added_driver_outside_the_search_is_still_a_driver(monkeypatch, linux, tmp_path):
    """VK_ADD_DRIVER_FILES is read FIRST and then the search, and it may name a manifest no
    search directory holds. Leaving it out made a host whose only other-vendor driver came
    in that way look like one the loader can only answer with AMD, and the other vendor's
    open node then stopped excusing the closed AMD one.

    Fails before the fix, which read the search alone."""
    search = tmp_path / "icd.d"
    search.mkdir()
    _icd_manifest(search, "radeon_icd.x86_64.json", library = "libamd.so")
    elsewhere = tmp_path / "vendor"
    elsewhere.mkdir()
    added = _icd_manifest(elsewhere, "nvidia_icd.json", library = "libnv.so")
    reason = _vulkan_reason_under_icd_list(
        monkeypatch,
        None,
        search_dirs = [str(search)],
        filters = {"VK_ADD_DRIVER_FILES": added},
    )
    assert reason.startswith("the Vulkan probe reported no device")


def test_a_forced_list_still_ignores_the_added_one(monkeypatch, linux, tmp_path):
    """The control the loader's own rule demands: VK_ADD_DRIVER_FILES is ignored entirely
    when VK_DRIVER_FILES or VK_ICD_FILENAMES is set, so an added other-vendor driver beside
    a forced AMD list is not a path this binary has."""
    search = tmp_path / "icd.d"
    search.mkdir()
    forced = _icd_manifest(search, "radeon_icd.x86_64.json", library = "libamd.so")
    elsewhere = tmp_path / "vendor"
    elsewhere.mkdir()
    added = _icd_manifest(elsewhere, "nvidia_icd.json", library = "libnv.so")
    reason = _vulkan_reason_under_icd_list(
        monkeypatch,
        forced,
        search_dirs = [str(search)],
        filters = {"VK_ADD_DRIVER_FILES": added},
    )
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_an_added_amd_driver_does_not_credit_another_vendor(monkeypatch, linux, tmp_path):
    """The second control: reading the additive list must not have made every host that has
    one look mixed-vendor. An AMD manifest added to an AMD-only search is still AMD alone."""
    search = tmp_path / "icd.d"
    search.mkdir()
    _icd_manifest(search, "radeon_icd.x86_64.json", library = "libamd.so")
    elsewhere = tmp_path / "vendor"
    elsewhere.mkdir()
    added = _icd_manifest(elsewhere, "amdvlk64.json", library = "libamdvlk.so")
    reason = _vulkan_reason_under_icd_list(
        monkeypatch,
        None,
        search_dirs = [str(search)],
        filters = {"VK_ADD_DRIVER_FILES": added},
    )
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_a_shut_node_whose_vendor_is_hidden_is_still_reported(monkeypatch, linux):
    """A container can map the render node and hide the sysfs entry naming its vendor.
    Requiring a confirmed AMD vendor dropped the node, so nothing was CLOSED -- while
    _amd_render_node_exists reads the same unknown as PRESENT and withdraws the
    missing-node sentence too, leaving #10466's own shape with no diagnosis at all.

    Fails before the fix, which asked for a vendor the container had hidden."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
        vendor_readable = False,
    )
    assert amd.amd_nodes_closed_to_this_user() == ["/dev/kfd", "/dev/dri/renderD128"]
    hint = amd.amd_node_permission_hint()
    assert "/dev/dri/renderD128" in hint


def test_the_same_hidden_vendor_says_nothing_without_an_amd_topology(monkeypatch, linux):
    """The control, and what keeps the render-group advice off every NVIDIA host: KFD is the
    independent evidence. Its topology reports vendor 0x10DE there, so an unreadable render
    node is not credited to AMD and neither diagnosis fires."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
        vendor_readable = False,
        amd_owned = False,
    )
    assert amd.amd_nodes_closed_to_this_user() == []


def test_a_readable_non_amd_node_is_still_dropped(monkeypatch, linux):
    """The second control: a vendor that CAN be read and is not AMD is positive evidence
    the node belongs to somebody else, and stays out of the list even on a host whose KFD
    topology does report an AMD GPU."""
    _nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128"],
        openable = set(),
        amd_owned = True,
    )
    monkeypatch.setattr(amd, "_render_node_vendor", lambda path: "0x10de")
    assert amd.amd_nodes_closed_to_this_user() == ["/dev/kfd"]


def test_a_cuda_wheel_beside_a_vulkan_bundle_is_not_sent_after_kfd():
    """Only a ROCm wheel opens /dev/kfd, and reading "any index that is not cpu" as one that
    does was harmless only while _amd_node_diag_route dropped every non-ROCm index. The
    explicit-backend arm keeps the route for a vulkan request, so a CUDA wheel beside a
    Vulkan bundle reached the KFD scope and told a healthy hybrid host to repair permissions
    on a node neither of them opens.

    Fails before the fix, which excluded the cpu leaf alone."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = "vulkan",
        torch_index = "https://download.pytorch.org/whl/cu128",
        nvidia = True,
    )
    assert out.strip() == ""


def test_the_same_pair_still_reports_a_closed_render_node():
    """The control that keeps the round-twenty-four fix: the Vulkan bundle DOES open a
    render node, so scoping /dev/kfd must not have taken the diagnosis the explicit-backend
    arm exists to reach."""
    out = _install_sh_kfd_scope(
        "/dev/kfd\n/dev/dri/renderD128",
        skip_torch = False,
        backend = "vulkan",
        torch_index = "https://download.pytorch.org/whl/cu128",
        nvidia = True,
    )
    assert "/dev/dri/renderD128" in out
    assert "/dev/kfd" not in out


def test_a_cuda_wheel_asking_for_the_rocm_bundle_still_gets_kfd():
    """The second control: the bundle is the other half of the question. A CUDA wheel with
    an explicit rocm request installs a bundle that opens /dev/kfd, so the node stays in the
    list even though the wheel never touches it."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = "rocm",
        torch_index = "https://download.pytorch.org/whl/cu128",
        nvidia = True,
    )
    assert "/dev/kfd" in out


def test_a_radeon_repo_wheel_is_still_a_kfd_consumer():
    """The third control, on the route this installer reroutes #10466's own host to:
    repo.radeon.com's leaf is rocm-rel-X.Y, which is not a pip family, so a classifier that
    only knew the pip spellings would have silenced the diagnosis for exactly the host it
    was written for."""
    out = _install_sh_kfd_scope(
        "/dev/kfd",
        skip_torch = False,
        backend = "vulkan",
        torch_index = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151",
    )
    assert "/dev/kfd" in out


def test_a_terminated_selector_still_exposes_the_prefix_it_accepted(monkeypatch, linux):
    """clr BREAKS out of its parse on a token it cannot map, having already pushed every
    device it accepted, so HIP_VISIBLE_DEVICES=0,1,-1 on a two-GPU host leaves both GPUs
    visible. Discarding the prefix read that host as narrowed and handed it the group
    repair in place of the driver diagnosis it needs.

    Fails before the fix, which returned False on the unmappable token."""
    assert _blocks_under_selector(monkeypatch, "0,1,-1", count = 2) is False


def test_a_non_numeric_token_ends_the_list_the_same_way(monkeypatch, linux):
    """The other way the parse terminates: atoi returns 0 for a word, and the token is
    then not 0 written back out, so the same break runs. Kept separate because the two
    reach it down different arms of the same test."""
    assert _blocks_under_selector(monkeypatch, "0,1,later", count = 2) is False


def test_a_prefix_that_stops_short_still_narrows(monkeypatch, linux):
    """The control, and the reason the prefix is judged rather than trusted: 0,-1 accepts
    ONE of two GPUs before terminating, so that host really is narrowed and the open
    sibling really is out of reach. Without it the fix could be "a terminated list never
    narrows", which suppresses the mask sentence on every masked host."""
    assert _blocks_under_selector(monkeypatch, "0,-1", count = 2) is True


def test_a_repeat_still_ends_the_rocr_list(monkeypatch, linux):
    """And the ROCr half is unchanged: RvdFilter terminates on an already-selected token,
    so ROCR_VISIBLE_DEVICES=0,0,1 surfaces one device however long the string is."""
    assert _blocks_under_selector(monkeypatch, "0,0,1", count = 2, var = "ROCR_VISIBLE_DEVICES") is True


def test_a_32_bit_amd_registration_is_not_a_driver_this_binary_can_load(
    monkeypatch, linux, tmp_path
):
    """A distribution registers the 32-bit ICD beside the 64-bit one, and a 64-bit
    llama-server cannot load it. A host left with only that has no driver at all, so the
    other vendor's open node is not shut out by an AMD-only loader -- there is no loader.

    Fails before the fix, which read the name as AMD and suppressed."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.i686.json")
    reason = _vulkan_reason_under_icd_list(monkeypatch, manifest)
    assert reason.startswith("the Vulkan probe reported no device")


def test_the_32_bit_manifest_beside_the_64_bit_one_is_still_amd_only(monkeypatch, linux, tmp_path):
    """The control, and the layout every multilib host actually has: the pair is one AMD
    driver registered twice, so the loader still loads AMD and only AMD. Without it the
    fix could be "any i686 name means no AMD driver", which is the normal case."""
    _thirty_two = _icd_manifest(tmp_path, "radeon_icd.i686.json")
    _sixty_four = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join([_thirty_two, _sixty_four]))
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_a_32_bit_manifest_from_another_vendor_does_not_credit_that_vendor(
    monkeypatch, linux, tmp_path
):
    """The reason the rule is asked of every vendor rather than of AMD alone. Treating a
    32-bit manifest as simply "not AMD" would make nvidia_icd.i686.json evidence that this
    loader can reach another vendor's card, when a 64-bit binary can load neither it nor
    the card behind it. Both are excluded, so what remains is the AMD driver."""
    _theirs = _icd_manifest(tmp_path, "nvidia_icd.i686.json", library = "libGLX_nvidia32.so")
    _ours = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join([_theirs, _ours]))
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_a_64_bit_manifest_from_another_vendor_still_counts(monkeypatch, linux, tmp_path):
    """The control for that one: the same foreign driver in its loadable build IS a path
    this binary has, so the loader is no longer AMD-only and the open node explains the
    empty probe. Without it the exclusion could be "ignore other vendors"."""
    _theirs = _icd_manifest(tmp_path, "nvidia_icd.json", library = "libGLX_nvidia.so")
    _ours = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join([_theirs, _ours]))
    assert reason.startswith("the Vulkan probe reported no device")


def test_the_32_bit_rule_matches_the_installers(tmp_path):
    """The installer decides whether an AMD driver is installed from the same filenames,
    and a host where the two disagree gets one answer from the Vulkan route and another
    from this diagnosis. Only the needles are shared: the installer asks it of AMD names
    alone, this asks it of every vendor, which is a difference in question not in rule."""
    import install_llama_prebuilt

    assert set(amd._VULKAN_ICD_32_BIT_NEEDLES) == set(
        install_llama_prebuilt._AMD_VULKAN_ICD_32_BIT_NEEDLES
    )
    for name in ("radeon_icd.i686.json", "radeon_icd.i386.json", "radeon_icd32.json"):
        assert amd._is_a_32_bit_icd_name(name), name
    for name in ("radeon_icd.x86_64.json", "radeon_icd.aarch64.json"):
        assert not amd._is_a_32_bit_icd_name(name), name


def _vulkan_node_hint_under_icd_list(
    monkeypatch,
    value,
    *,
    search_dirs = None,
):
    """The empty-probe reason for a Vulkan build with the AMD node shut and NO other
    vendor's node open, so the answer is the node repair and the question is what is
    appended to it."""
    from core.inference.llama_cpp import LlamaCppBackend

    _nodes(monkeypatch, present = ["/dev/kfd", "/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: False)
    for _var in (
        "VK_DRIVER_FILES",
        "VK_ICD_FILENAMES",
        "VK_ADD_DRIVER_FILES",
        "VK_LOADER_DRIVERS_SELECT",
        "VK_LOADER_DRIVERS_DISABLE",
    ):
        monkeypatch.delenv(_var, raising = False)
    if search_dirs is not None:
        monkeypatch.setattr(amd, "_vulkan_icd_search_dirs", lambda: list(search_dirs))
    if value is not None:
        monkeypatch.setenv("VK_DRIVER_FILES", value)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"vulkan"})),
    )
    return LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")


def test_a_loader_with_no_loadable_driver_is_said_beside_the_node_repair(
    monkeypatch, linux, tmp_path
):
    """Two things are wrong at once and the node repair clears only one of them: with
    every registered manifest unloadable the probe stays empty however the node is owned,
    so a user who runs the usermod and nothing else is left exactly where they were."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.json", present = False)
    reason = _vulkan_node_hint_under_icd_list(monkeypatch, manifest)
    assert "usermod" in reason
    assert "no driver it can load" in reason


def test_a_loader_that_could_not_be_enumerated_says_nothing_about_drivers(
    monkeypatch, linux, tmp_path
):
    """The control, and the fail-closed direction that matters here: an enumeration that
    found NO manifests means this could not read the loader's configuration, not that the
    loader has nothing. Emitting the sentence there sends a working host after a driver
    reinstall it does not need, on top of a repair it does."""
    reason = _vulkan_node_hint_under_icd_list(
        monkeypatch, None, search_dirs = [str(tmp_path / "empty")]
    )
    assert "usermod" in reason
    assert "no driver it can load" not in reason


def test_a_loadable_driver_says_nothing_about_drivers_either(monkeypatch, linux, tmp_path):
    """The second control: manifests found AND loadable is the ordinary host, where the
    closed node is the whole story. Without it the sentence could be unconditional."""
    manifest = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    reason = _vulkan_node_hint_under_icd_list(monkeypatch, manifest)
    assert "usermod" in reason
    assert "no driver it can load" not in reason


def test_a_group_this_account_already_holds_is_not_prescribed(monkeypatch, linux):
    """os.access already said the node is shut, so a group this account is ALREADY in is
    not what is denying it: usermod exits 0 and leaves the node exactly as closed. The
    denial is outside the file mode -- a container device cgroup, or an LSM.

    Fails before the fix, which read the group bits and prescribed the membership."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (39, 0o660, 0)}, {39: "render"})
    monkeypatch.setattr(amd.os, "getgroups", lambda: [39])
    _joinable, _unnamed, _no_group, _acl, _owned, _priv, already = amd._groups_that_own(
        ["/dev/kfd"]
    )
    assert already == ["render"]
    assert _joinable == []


def test_a_group_this_account_is_outside_is_still_prescribed(monkeypatch, linux):
    """The control: the same node, the same mode, a group this account does not hold. The
    membership IS the repair there, and it is what #10466 asked for. Without it the rule
    could be "never prescribe a group"."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (39, 0o660, 0)}, {39: "render"})
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])
    monkeypatch.setattr(amd.os, "getgid", lambda: 1)
    joinable, _unnamed, _no_group, _acl, _owned, _priv, already = amd._groups_that_own(["/dev/kfd"])
    assert joinable == ["render"]
    assert already == []


def test_an_unnamed_gid_this_account_already_holds_is_not_prescribed_either(monkeypatch, linux):
    """The same rule one branch over. A numeric owner with no group-database entry is the
    container shape, and the repair there is groupadd plus --group-add -- which is the same
    empty promise once this account already carries the gid. Asked BEFORE the naming
    branch for exactly that reason."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {})
    monkeypatch.setattr(amd.os, "getgroups", lambda: [993])
    _joinable, unnamed, _no_group, _acl, _owned, _priv, already = amd._groups_that_own(["/dev/kfd"])
    assert already == ["993"]
    assert unnamed == []


def test_an_unnamed_gid_this_account_lacks_is_still_reported(monkeypatch, linux):
    """Its control, and the reason the unnamed branch exists: a numeric owner this account
    is outside still gets the groupadd pair."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {})
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])
    monkeypatch.setattr(amd.os, "getgid", lambda: 1)
    _joinable, unnamed, _no_group, _acl, _owned, _priv, already = amd._groups_that_own(["/dev/kfd"])
    assert unnamed == [993]
    assert already == []


def test_the_hint_for_a_group_already_held_names_the_cgroup_instead(monkeypatch, linux):
    """The sentence a user actually reads, since the buckets above only decide it: no
    usermod, and a statement of what is left to look at."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], [], ["render"]))
    hint = amd.amd_node_permission_hint()
    # The command, not the word: the sentence itself says usermod would change nothing.
    assert "usermod -a -G" not in hint
    assert "already in the render group" in hint
    assert "cgroup" in hint


def test_the_installer_does_not_prescribe_a_group_this_account_holds(tmp_path):
    """The shell half of the same rule, read from `id -G`. Without it the two halves
    disagree on the same host: the installer prints the usermod the Python side just
    declined to."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), self_gids = str(os.stat(node).st_gid))
    assert "%s usermod -a -G" % ("s" + "udo") not in out
    assert "already in the" in out
    assert "cgroup" in out


def test_the_installer_still_prescribes_a_group_this_account_lacks(tmp_path):
    """Its control: the same node, an account outside its group, and the command comes
    back. Without it the shell rule could be "never prescribe"."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), self_gids = str(os.stat(node).st_gid + 1))
    assert "already in the" not in out
    assert "sudo usermod -a -G" in out


def test_the_installer_gid_match_is_not_a_substring_match(tmp_path):
    """The trap in reading `id -G` as text: the list is space separated, so an unpadded
    match makes gid 100 look held by an account that is only in 1001. The node here is
    owned by a group the account does NOT have, and its gid is a prefix of one it does."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    _gid = os.stat(node).st_gid
    out = _install_sh_hint(str(node), self_gids = f"{_gid}7 {_gid}9")
    assert "already in the" not in out
    assert "sudo usermod -a -G" in out


def _bare_soname_manifest(
    tmp_path,
    name,
    soname = "libvulkan_radeon.so",
):
    """An ICD manifest naming its library by soname alone, and the path to it.

    The form NVIDIA registers under, and the one _icd_manifest cannot produce: that helper
    writes an absolute path so it can put the library on disk, which is exactly the case
    this is not.
    """
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": soname, "api_version": "1.3.0"},
            }
        ),
        encoding = "utf-8",
    )
    return str(path)


def test_a_bare_soname_nothing_can_resolve_is_not_a_driver(monkeypatch, linux, tmp_path):
    """A manifest may name its library by soname and leave the loader to find it, so the
    package can be removed and leave the registration behind. Trusting the name counted a
    driver that is not there, and withheld the reinstall half of the repair.

    Fails before the fix, which returned True for every bare name."""
    manifest = _bare_soname_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset({"libc.so.6"}))
    assert amd._icd_manifest_is_usable(manifest) is False


def test_a_bare_soname_on_the_search_path_is_a_driver(monkeypatch, linux, tmp_path):
    """The control: the same manifest with the library where ld.so would find it. Without
    it the fix could be "a bare name is never a driver", which is the ordinary case for
    every vendor that registers one."""
    _lib = tmp_path / "lib"
    _lib.mkdir()
    (_lib / "libvulkan_radeon.so").write_bytes(b"")
    manifest = _bare_soname_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(_lib)])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset())
    assert amd._icd_manifest_is_usable(manifest) is True


def test_a_bare_soname_only_the_loader_cache_knows_is_a_driver(monkeypatch, linux, tmp_path):
    """The second control, and the reason the cache is consulted at all: a versioned
    soname such as libGLX_nvidia.so.0 lives wherever ld.so.conf put it, which need not be
    a directory this enumerates. Present in the cache is present."""
    manifest = _bare_soname_manifest(tmp_path, "nvidia_icd.json", "libGLX_nvidia.so.0")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset({"libGLX_nvidia.so.0"}))
    assert amd._icd_manifest_is_usable(manifest) is True


def test_a_loader_cache_that_cannot_be_read_leaves_the_registration_alone(
    monkeypatch, linux, tmp_path
):
    """The fail-closed control, and the direction that matters. musl ships no ldconfig -p
    and a minimal container may ship no ldconfig at all, so "not found" there is ignorance
    rather than absence. Calling a live driver stale would promote the AMD node as the sole
    cause on a host whose other vendor really does have a path."""
    manifest = _bare_soname_manifest(tmp_path, "nvidia_icd.json", "libGLX_nvidia.so.0")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: None)
    assert amd._icd_manifest_is_usable(manifest) is True


def test_the_loader_cache_reader_says_none_rather_than_empty_when_ldconfig_is_gone(
    monkeypatch, linux
):
    """The distinction the arm above rests on, at its source: no ldconfig has to answer
    None, because an empty set would read as "no library is installed" and call every bare
    registration on the host stale."""
    monkeypatch.setattr(amd, "_ld_cache_read", False)
    monkeypatch.setattr(amd, "_ld_cache_sonames_cached", None)
    monkeypatch.setattr(amd.shutil, "which", lambda _name: None)
    monkeypatch.setattr(amd.os.path, "exists", lambda _p: False)
    assert amd._ld_cache_sonames() is None


def test_a_stale_bare_registration_reaches_the_driver_sentence(monkeypatch, linux, tmp_path):
    """What the classification is for: with the only registration unresolvable the loader
    has no driver at all, so the node repair alone would leave the probe empty."""
    manifest = _bare_soname_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset({"libc.so.6"}))
    reason = _vulkan_node_hint_under_icd_list(monkeypatch, manifest)
    assert "usermod" in reason
    assert "no driver it can load" in reason


def test_the_mask_fixture_isolates_every_selector_the_rule_reads():
    """The fixture decides what "no mask" means for this whole suite, so a selector the
    production rule reads and the fixture does not clear is answered by the runner's own
    environment: a ROCm or OpenCL host exports GPU_DEVICE_ORDINAL, and the no-mask controls
    would then quietly take the narrowing branch they exist to rule out.

    Read out of the rule's own source rather than restated, so adding a fifth selector
    fails here instead of drifting."""
    _source = inspect.getsource(amd._a_per_gpu_mask_narrows_the_runtime)
    _read = set(re.findall(r'"([A-Z_]+(?:VISIBLE_DEVICES|DEVICE_ORDINAL))"', _source))
    assert _read, "the rule named no selector, so this test proves nothing"
    assert _read <= set(_GPU_MASK_VARS), _read - set(_GPU_MASK_VARS)


def test_the_installer_chains_the_groupadd_pair(tmp_path):
    """groupadd and usermod are a pair, and the name is generated from the GID, which says
    nothing about whether that NAME is free. Printed as two separate lines, a host that
    already has an amdgpu993 group at another GID fails the groupadd and then SUCCEEDS the
    usermod against the wrong group, leaving the node shut having reported success.

    Fails before the fix, which printed them unchained."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993")
    _lines = out.splitlines()
    _at = next(i for i, l in enumerate(_lines) if "groupadd" in l)
    # && plus a continuation, so the pair pastes as one command across the two lines.
    assert _lines[_at].rstrip().endswith("&& \\"), _lines[_at]
    assert "993 amdgpu993" in _lines[_at]
    assert "usermod -a -G amdgpu993" in _lines[_at + 1]


def test_the_python_half_chains_the_groupadd_pair_too(monkeypatch, linux):
    """Its twin, which already chained: asserted so that the two halves cannot drift apart
    the way they just did."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "groupadd -g 993 amdgpu993 && sudo usermod -a -G amdgpu993" in hint


def test_the_named_group_repair_is_still_one_command(tmp_path):
    """The control: a node whose owning group HAS a name needs no groupadd, so the repair
    is a single usermod and must not have grown a chain."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "join:render")
    assert "groupadd" not in out
    _line = next(l for l in out.splitlines() if "usermod" in l)
    assert "&&" not in _line


def _install_sh_closed_nodes(nodes, *, vendors, topology: bool) -> "list[str]":
    """The installer's closed-node enumeration, run over a named set of files.

    The real one reads /dev and /sys, which no test can reach, so the two seams it now has
    are stubbed and the node files themselves are real: the mode tests are the rule under
    test and must run against actual permissions.
    """
    lines = _install_sh_lines()
    _vendor_cases = " ".join(
        f"{shlex.quote(str(_path))}) printf %s {shlex.quote(_vendor)} ;;"
        for _path, _vendor in vendors.items()
    )
    script = "\n".join(
        [
            "_amd_candidate_nodes() { printf '%s\\n' "
            + " ".join(shlex.quote(str(_n)) for _n in nodes)
            + "; }",
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if topology else 1}; }}",
            # A vendor sysfs will not name exits non-zero, which is the case under test.
            '_amd_render_node_vendor() { case "$1" in ' + _vendor_cases + " *) return 1 ;; esac; }",
            _shell_fn(lines, "_amd_nodes_closed_to_this_user"),
            "_amd_nodes_closed_to_this_user",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr
    return [line for line in out.stdout.splitlines() if line.strip()]


def _a_closed_node_file(tmp_path, name):
    """A file standing in for a device node this account cannot open."""
    node = tmp_path / name
    node.write_bytes(b"")
    node.chmod(0o000)
    return node


def test_the_installer_keeps_a_closed_node_whose_vendor_is_hidden(tmp_path):
    """A container can map /dev/dri and hide the sysfs attribute naming its vendor, which
    is the shape #10466 is about. Dropping the node left the installer printing no
    render-node repair at all, while _amd_render_node_present reads the same unknown as
    PRESENT and withdraws the missing-node sentence -- so that host got no diagnosis.

    Fails before the fix, which required a readable vendor. The Python half has answered
    this since round twenty-five; this is the installer catching up."""
    if os.geteuid() == 0:
        pytest.skip("root can open a mode 000 node, so nothing here is closed")
    node = _a_closed_node_file(tmp_path, "renderD128")
    assert _install_sh_closed_nodes([node], vendors = {}, topology = True) == [str(node)]


def test_the_installer_drops_a_hidden_vendor_when_no_amd_gpu_is_in_the_topology(tmp_path):
    """The control, and the reason the vendor guard exists at all: render nodes are
    root:render for EVERY vendor, so an NVIDIA-only box has the same closed list and none
    of the problem. Without it the fallback would hand that host AMD group advice."""
    if os.geteuid() == 0:
        pytest.skip("root can open a mode 000 node, so nothing here is closed")
    node = _a_closed_node_file(tmp_path, "renderD128")
    assert _install_sh_closed_nodes([node], vendors = {}, topology = False) == []


def test_the_installer_still_drops_a_node_that_names_another_vendor(tmp_path):
    """The second control: a vendor that IS readable and is not AMD stays excluded however
    the topology reads, since positive evidence beats the fallback. A mixed box has an
    NVIDIA render node beside the AMD one and must not be told to chgrp it."""
    if os.geteuid() == 0:
        pytest.skip("root can open a mode 000 node, so nothing here is closed")
    node = _a_closed_node_file(tmp_path, "renderD129")
    assert _install_sh_closed_nodes([node], vendors = {node: "0x10de"}, topology = True) == []


def test_the_installer_keeps_a_node_that_names_amd(tmp_path):
    """The third control, the ordinary host: a readable AMD vendor is kept, which is what
    the whole enumeration is for."""
    if os.geteuid() == 0:
        pytest.skip("root can open a mode 000 node, so nothing here is closed")
    node = _a_closed_node_file(tmp_path, "renderD128")
    assert _install_sh_closed_nodes([node], vendors = {node: "0x1002"}, topology = False) == [str(node)]


def _icd_manifest_with(
    tmp_path,
    name,
    *,
    library = None,
    arch = None,
    elf = None,
):
    """An ICD manifest with a declared architecture, an ELF library, or neither."""
    icd = {"api_version": "1.3.0"}
    if library is not None:
        lib = tmp_path / library
        if elf is not None:
            # e_ident: magic, then EI_CLASS 1 for 32-bit and 2 for 64-bit.
            lib.write_bytes(b"\x7fELF" + bytes([1 if elf == 32 else 2]) + b"\x00" * 11)
        else:
            lib.write_bytes(b"")
        icd["library_path"] = str(lib)
    else:
        icd["library_path"] = "libvulkan_radeon.so"
    if arch is not None:
        icd["library_arch"] = arch
    path = tmp_path / name
    path.write_text(json.dumps({"file_format_version": "1.0.1", "ICD": icd}), encoding = "utf-8")
    return str(path)


def test_a_manifest_that_declares_32_bit_is_not_loadable(monkeypatch, linux, tmp_path):
    """library_arch is the loader's own field and the loader reads it for exactly this
    purpose: to skip a driver whose bitness cannot match the process. A neutrally named
    32-bit registration passed the filename test and credited a driver this binary cannot
    open.

    Fails before the fix, which asked the filename alone."""
    manifest = _icd_manifest_with(tmp_path, "radeon_icd.json", library = "a.so", arch = "32")
    assert amd._an_icd_is_32_bit(manifest) is True


def test_a_declared_64_bit_manifest_wins_over_its_own_filename(monkeypatch, linux, tmp_path):
    """The control that proves the field is read rather than the name: a manifest named
    i686 that declares 64 is loadable, because the declaration is the loader's answer and
    the name is only ever a guess at it."""
    manifest = _icd_manifest_with(tmp_path, "radeon_icd.i686.json", library = "b.so", arch = "64")
    assert amd._an_icd_is_32_bit(manifest) is False


def test_the_elf_class_answers_when_the_manifest_declares_nothing(monkeypatch, linux, tmp_path):
    """library_arch is optional and Debian strips it back out of Mesa's manifests to keep
    one file across architectures, so its absence is ordinary. The object itself still
    says: EI_CLASS is byte 4 of every ELF."""
    manifest = _icd_manifest_with(tmp_path, "radeon_icd.json", library = "c.so", elf = 32)
    assert amd._an_icd_is_32_bit(manifest) is True


def test_a_64_bit_elf_with_a_neutral_name_stays_loadable(monkeypatch, linux, tmp_path):
    """Its control, and the ordinary case for every stripped manifest on the host."""
    manifest = _icd_manifest_with(tmp_path, "radeon_icd.json", library = "d.so", elf = 64)
    assert amd._an_icd_is_32_bit(manifest) is False


def test_the_filename_still_answers_when_nothing_else_can(monkeypatch, linux, tmp_path):
    """The last resort, unchanged: no declaration and no library to read leaves the name,
    which is also all the installer ever has."""
    manifest = _icd_manifest_with(tmp_path, "radeon_icd.i686.json")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [])
    assert amd._an_icd_is_32_bit(manifest) is True


def test_a_declared_32_bit_manifest_reaches_the_empty_probe_reason(monkeypatch, linux, tmp_path):
    """What the classification decides: with the only other registration 32-bit, that
    vendor's open render node is not a path this binary has, so the closed AMD node stays
    the answer rather than being demoted."""
    _theirs = _icd_manifest_with(tmp_path, "nvidia_icd.json", library = "libGLX_nvidia.so", arch = "32")
    _ours = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join([_theirs, _ours]))
    assert "the Vulkan probe reported no device" not in reason
    assert "usermod" in reason


def test_a_name_the_shell_would_mangle_is_quoted(monkeypatch, linux):
    """These are commands to paste. NSS names are not identifiers -- winbind hands back
    DOMAIN\\user -- so an unquoted one is de-escaped by the shell and usermod then names an
    account that does not exist, leaving the node shut.

    Fails before the fix, which interpolated the name raw."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_repair_account", lambda: "DOMAIN\\ada")
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render 'DOMAIN\\ada'" in hint


def test_a_group_name_that_carries_a_space_is_quoted_too(monkeypatch, linux):
    """The other half of the same command, and the one that would silently split into two
    arguments rather than failing outright."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_repair_account", lambda: "ada")
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda paths: (["gpu users"], [], [], [], [], [], [])
    )
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G 'gpu users' ada" in hint


def test_an_ordinary_name_is_left_alone(monkeypatch, linux):
    """The control, and why shlex.quote rather than unconditional quoting: the command a
    user actually sees on a normal host must not grow quotes it does not need."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_repair_account", lambda: "ada")
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render ada" in hint
    assert "'" not in hint


def test_the_installer_quotes_the_account_the_same_way(tmp_path):
    """The shell twin: the installer prints the same command from the same kind of name,
    so a host whose account carries a backslash must not get an unquoted one there either."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), id_user = "DOMAIN\\ada", env_user = "DOMAIN\\ada")
    assert "'DOMAIN\\ada'" in out


def test_the_two_quoting_rules_are_the_same_rule(tmp_path):
    """Both halves print the same command, so a value one quotes and the other does not is
    a host where the two disagree about what the user should paste. Run against the real
    shell function rather than a restatement of it."""
    lines = _install_sh_lines()
    helper = _shell_fn(lines, "_shell_quote")
    for value in (
        "ada",
        "render",
        "DOMAIN\\ada",
        "gpu users",
        "ada;reboot",
        "a'b",
        "user@host",
        "",
    ):
        out = subprocess.run(
            ["bash", "-c", helper + '\n_shell_quote "$1"', "_", value],
            capture_output = True,
            text = True,
        )
        assert out.returncode == 0, out.stderr
        # What actually matters, asked of the shell rather than of the spelling: each
        # quoting has to be ONE word that comes back as the name it started as.
        for _quoted in (out.stdout, shlex.quote(value)):
            _back = subprocess.run(
                ["bash", "-c", "printf %s " + _quoted], capture_output = True, text = True
            )
            assert _back.returncode == 0, _back.stderr
            assert _back.stdout == value, (value, _quoted, _back.stdout)
        # And the printed text is identical too, so the two halves show the same command.
        # An embedded quote is the one place the spellings differ ('"'"' against \\''), and
        # both are correct, so that case is carried by the round trip above alone.
        if "'" not in value:
            assert out.stdout == shlex.quote(value), (value, out.stdout)


# The two helpers every tests/sh ROCm harness lifts alongside _has_amd_rocm_gpu.
_ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT = frozenset(
    {"_ensure_rocm_probe_env", "_has_usable_nvidia_gpu"}
)


def test_the_rocm_probe_calls_nothing_the_shell_harnesses_do_not_lift():
    """tests/sh lifts probes out of install.sh one function at a time, by name, with
    `sed -n '/^_name()/,/^}/p'`. So _has_amd_rocm_gpu may only call helpers those
    harnesses already lift: a call to anything else is an undefined function there, the
    ROCm branch falls through to the CPU wheel index, and four harnesses fail on a
    torch-index assertion that has nothing to do with what changed.

    That is not hypothetical. Splitting the probe into a wrapper over a private
    _amd_rocm_gpu_visible did exactly this, and the NVIDIA veto now lives inside the
    function so there is nothing to forget. Fails if the split comes back."""
    lines = _install_sh_lines()
    defined = {line.split("(")[0] for line in lines if re.match(r"^_?[A-Za-z0-9_]+\(\) \{", line)}
    body = _shell_fn(lines, "_has_amd_rocm_gpu").splitlines()[1:]
    called = {
        name
        for name in defined
        if name != "_has_amd_rocm_gpu"
        and any(re.search(rf"(^|[\s;&|(]){re.escape(name)}($|[\s;&|)])", l) for l in body)
    }
    assert called <= _ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT, called


def test_that_check_sees_a_helper_the_harnesses_would_not_have():
    """The control. Without it the test above could be passing because the scan matches
    nothing at all, which is what a name-based scan usually does when it is wrong."""
    lines = [
        "_ensure_rocm_probe_env() {",
        "    :",
        "}",
        "_amd_rocm_gpu_visible() {",
        "    return 1",
        "}",
        "_has_amd_rocm_gpu() {",
        "    _ensure_rocm_probe_env",
        "    _amd_rocm_gpu_visible",
        "}",
    ]
    defined = {line.split("(")[0] for line in lines if re.match(r"^_?[A-Za-z0-9_]+\(\) \{", line)}
    body = _shell_fn(lines, "_has_amd_rocm_gpu").splitlines()[1:]
    called = {
        name
        for name in defined
        if name != "_has_amd_rocm_gpu"
        and any(re.search(rf"(^|[\s;&|(]){re.escape(name)}($|[\s;&|)])", l) for l in body)
    }
    assert called == {"_ensure_rocm_probe_env", "_amd_rocm_gpu_visible"}


def _multilib_soname(tmp_path, *, bitnesses):
    """Two loader search directories carrying one soname, and the manifest naming it.

    Named the way a Debian multilib host names them, because the ORDER is the subject:
    sorted(glob) puts i386-linux-gnu ahead of x86_64-linux-gnu, so a search that stops at
    the first hit reads the wrong copy on the commonest multilib layout there is.
    """
    dirs = []
    for _arch, _bits in (("i386-linux-gnu", 32), ("x86_64-linux-gnu", 64)):
        _dir = tmp_path / _arch
        _dir.mkdir()
        dirs.append(str(_dir))
        if _bits in bitnesses:
            (_dir / "libvk.so").write_bytes(
                b"\x7fELF" + bytes([1 if _bits == 32 else 2]) + b"\x00" * 11
            )
    path = tmp_path / "nvidia_icd.json"
    path.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": "libvk.so", "api_version": "1.3.0"},
            }
        ),
        encoding = "utf-8",
    )
    return str(path), dirs


def test_a_bare_soname_resolves_to_the_copy_this_process_could_load(monkeypatch, tmp_path):
    """Both bitnesses of one soname, which is what a multilib driver install looks like.
    ld.so picks the copy matching the process; the reconstructed search order does not, and
    i386 sorts first, so the first hit was the 32-bit one. _an_icd_is_32_bit then read that
    object and discarded a manifest whose driver the loader loads -- which either withholds
    the whole no-driver repair or reports the loader as AMD-only when it is not.

    Fails before the fix, which returned the first match."""
    manifest, dirs = _multilib_soname(tmp_path, bitnesses = {32, 64})
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: dirs)
    assert amd._icd_library_path(manifest).startswith(dirs[1])
    assert amd._an_icd_is_32_bit(manifest) is False


def test_a_soname_only_the_wrong_bitness_answers_is_still_32_bit(monkeypatch, tmp_path):
    """The control. Without it the rule could be "never 32-bit", which puts back every
    unloadable 32-bit registration the bitness filter exists to drop."""
    manifest, dirs = _multilib_soname(tmp_path, bitnesses = {32})
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: dirs)
    assert amd._icd_library_path(manifest).startswith(dirs[0])
    assert amd._an_icd_is_32_bit(manifest) is True


def _manifest_missing(tmp_path, name, *, drop):
    """A manifest with one loader-required field removed, and its library on disk."""
    lib = tmp_path / f"{name}.so"
    lib.write_bytes(b"\x7fELF\x02" + b"\x00" * 11)
    icd = {"library_path": str(lib), "api_version": "1.3.0"}
    body = {"file_format_version": "1.0.0", "ICD": icd}
    if drop in icd:
        del icd[drop]
    else:
        del body[drop]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(body), encoding = "utf-8")
    return str(path)


@pytest.mark.parametrize("field", ["file_format_version", "api_version"])
def test_a_manifest_the_loader_skips_is_not_a_driver(tmp_path, field):
    """loader_parse_icd_manifest returns VK_ERROR_INCOMPATIBLE_DRIVER on a missing
    file_format_version and on a missing api_version, exactly as it does on a missing
    library_path. Reading library_path alone counted a registration the loader refuses,
    which suppresses the no-driver repair or calls the loader AMD-only when it is neither.

    Fails before the fix for both fields."""
    assert amd._icd_manifest_is_usable(_manifest_missing(tmp_path, field, drop = field)) is False


def test_a_version_the_loader_does_not_recognise_is_still_a_driver(tmp_path):
    """The control, and the line between the two. An unknown file_format_version major is
    the one thing here the loader does NOT skip for: it logs "may cause errors" and carries
    on, so refusing it would drop a driver that loads. Only absence decides."""
    lib = tmp_path / "libvk.so"
    lib.write_bytes(b"\x7fELF\x02" + b"\x00" * 11)
    path = tmp_path / "future_icd.json"
    path.write_text(
        json.dumps(
            {
                "file_format_version": "9.0.0",
                "ICD": {"library_path": str(lib), "api_version": "1.3.0"},
            }
        ),
        encoding = "utf-8",
    )
    assert amd._icd_manifest_is_usable(str(path)) is True
