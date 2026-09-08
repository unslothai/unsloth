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

import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils.hardware import amd


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")


def _nodes(
    monkeypatch,
    *,
    present: list[str],
    openable: set[str],
    amd_owned: bool = True,
):
    """A host whose ``present`` nodes exist and whose ``openable`` subset can be opened.

    ``amd_owned`` is the vendor of the hardware behind those nodes, stubbed here and
    exercised for real in the two tests below it.
    """
    monkeypatch.setattr(
        amd.glob, "glob", lambda pattern: [p for p in present if p.startswith("/dev/dri/renderD")]
    )
    monkeypatch.setattr(amd.os.path, "exists", lambda p: p in present)
    monkeypatch.setattr(amd.os, "access", lambda p, mode: p in openable)
    monkeypatch.setattr(amd, "_render_node_is_amd", lambda p: amd_owned)
    monkeypatch.setattr(amd, "_kfd_topology_has_an_amd_gpu", lambda: amd_owned)
    # These paths are patched rather than created, so stat cannot name their groups; say
    # so explicitly instead of leaving it to whether the runner happens to have a node at
    # the same path. The derivation itself is exercised in its own tests below.
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], []))


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
    import builtins

    real_open = builtins.open

    def _fake(path, *a, **k):
        if str(path) == "/sys/class/drm/renderD128/device/vendor":
            import io
            return io.StringIO("0x1002\n")
        if str(path) == "/sys/class/drm/renderD129/device/vendor":
            import io
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


def test_the_hint_survives_an_environment_with_no_user(monkeypatch, linux):
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.delenv("USER", raising = False)
    monkeypatch.delenv("LOGNAME", raising = False)
    assert "$USER" in amd.amd_node_permission_hint()


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


def _kernel_stack_hint_runs(closed_nodes: str, *, route: bool = True) -> bool:
    """Whether install.sh's missing-kernel-stack branch fires for this closed set.

    The guard is lifted out of install.sh by text rather than restated here: a test
    that restated it would pass whatever the installer went on to say. Only the
    condition is taken, and its two probes are stubbed true so the answer depends on
    nothing but the closed-node reasoning.
    """
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    # Anchored on the part of the condition this change does NOT touch, then walked
    # back over the continuations to the "if". Anchoring on the new closed-node text
    # instead would make the control vacuous: reverting the guard would stop the
    # extraction finding anything, and "the text changed" would read as "the
    # behaviour changed".
    end = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("! _has_amd_rocm_gpu && _amd_gpu_present_via_pci; then")
    )
    start = end
    while not lines[start].lstrip().startswith("if "):
        start -= 1
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
            _shell_fn(lines, "_run_may_open_kfd"),
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
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda _b: frozenset({"hip"})),
    )
    reason = LlamaCppBackend._explain_empty_gpu_probe("/nonexistent/llama-server")
    assert "usermod -a -G render,video ada" in reason
    assert "HIP_VISIBLE_DEVICES is empty" in reason


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
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["kfd", "gpu"], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G kfd,gpu ada" in hint
    assert "render,video" not in hint


def test_a_single_owning_group_is_not_pluralised(monkeypatch, linux):
    """A host where both nodes belong to one group gets one group named, and the sentence
    has to agree with the command rather than saying "groups" over a single name."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render ada" in hint
    assert "render group and then log out" in hint


def test_unreadable_nodes_fall_back_to_the_documented_pair(monkeypatch, linux):
    """The control: the derivation is best effort, so a host whose nodes cannot be stat'd
    must still get advice rather than an empty -G argument, and that advice is the pair
    the AMD documentation names."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    assert "usermod -a -G render,video ada" in amd.amd_node_permission_hint()


def _stat_nodes(monkeypatch, modes: dict, names: dict):
    """Stub os.stat and grp for the node set ``modes`` maps to (gid, mode).

    Paths outside the map raise, which covers the vanished-node arm. The stub takes
    ``follow_symlinks`` because pytest itself stats files while this patch is in force,
    and a two-argument lambda takes the whole session down with it rather than failing
    the test that installed it.
    """
    import grp as _grp

    def _stat(path, *, follow_symlinks = True):
        if str(path) not in modes:
            raise OSError("gone")
        _entry = modes[str(path)]
        gid, mode = _entry[0], _entry[1]
        # Real device nodes are root-owned, and POSIX consults the owner class first, so a
        # fake without st_uid would take the owner branch on whatever uid the runner has.
        uid = _entry[2] if len(_entry) > 2 else 0
        return type("st", (), {"st_gid": gid, "st_mode": mode, "st_uid": uid})()

    def _getgrgid(gid):
        if gid not in names:
            raise KeyError(gid)
        return type("gr", (), {"gr_name": names[gid]})()

    monkeypatch.setattr(amd.os, "stat", _stat)
    monkeypatch.setattr(_grp, "getgrgid", _getgrgid)


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
    )


def test_a_gid_with_no_group_entry_is_reported_rather_than_prescribed(monkeypatch):
    """The container case docker/run.sh documents: --group-add passes the host's numeric
    gids, and inside the container no group entry matches them.

    Fails before the fix, which returned the bare number for usermod to consume. shadow
    4.13 answers ``group '993' does not exist`` and exits 6 on that command, verified on
    this host, so the number belongs in a sentence rather than in the -G argument."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660)}, {})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [993], [], [], [], [])


def test_a_node_whose_own_group_cannot_open_it_is_not_a_membership_problem(monkeypatch):
    """A udev rule leaving a node root:render 0600 denies the group as well, so joining
    render opens nothing. Read the mode before naming the group, or the repair is a
    command that runs, succeeds, and changes nothing.

    Fails before the fix, which read st_gid alone and would have prescribed render."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (44, 0o600)}, {44: "render"})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [], ["/dev/kfd"], [], [], [])


def test_group_read_without_write_is_not_enough(monkeypatch):
    """Its boundary: HIP and the Vulkan loader both open the node read-write, which is
    the bar the probe itself applies, so 0640 is still not a joinable group."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (44, 0o640)}, {44: "render"})
    assert amd._groups_that_own(["/dev/kfd"]) == ([], [], ["/dev/kfd"], [], [], [])


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
    )


def _install_sh_if(lines: "list[str]", tail: str) -> int:
    """The index of the `if` opening the block whose condition ENDS with ``tail``.

    Anchored on the condition and walked back, as _install_sh_missing_kfd already does:
    a multi-line condition puts the `if` and its last test on different lines, so requiring
    both on one line stopped finding anything the moment a gate was added -- and a harness
    that finds nothing raises here rather than silently testing a shorter script.
    """
    i = next(j for j, line in enumerate(lines) if line.rstrip().endswith(tail))
    while not lines[i].lstrip().startswith("if "):
        i -= 1
    return i


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


def _install_sh_hint(
    closed_nodes: str,
    *,
    render_present: bool = True,
    amd_present: bool = True,
    self_uid: str = "4242",
    repairs: "str | None" = None,
) -> str:
    """The installer's closed-node message, run for a given closed set.

    Lifted from install.sh rather than restated, and the whole block rather than a
    condition, because the thing under test is the sentence it prints. substep is stubbed
    to plain echo; the node list comes in through the environment, since embedding it in
    the script would put a literal backslash-n inside shell quotes and turn two nodes into
    one unmatched line.
    """
    import subprocess

    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    text = install_sh.read_text(encoding = "utf-8")
    lines = text.splitlines()
    start = _install_sh_if(lines, '[ -n "$_closed_amd_nodes" ]; then')
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[start : end + 1])

    fn_start = next(i for i, line in enumerate(lines) if line.startswith("_amd_node_repairs() {"))
    depth = 0
    for fn_end in range(fn_start, len(lines)):
        depth += lines[fn_end].count("{") - lines[fn_end].count("}")
        if depth == 0:
            break
    helper = "\n".join(lines[fn_start : fn_end + 1])

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
            f"id() {{ echo {self_uid}; }}",
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if amd_present else 1}; }}",
            # The route the diagnoses are gated on; the gate has its own tests below.
            "_amd_node_diag_route=true",
            "OS=linux",
            # The run-scope predicate the block now asks. Lifted rather than stubbed, so
            # the default arms below go through the same rule the installer applies.
            "SKIP_TORCH=false",
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
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
        env = {**os.environ, "_closed_amd_nodes": closed_nodes, "USER": "ada"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_installer_names_the_group_the_node_actually_has(tmp_path):
    """The shell half of the same item, and the only arm of it that reads a real file:
    the message must name the group that owns the node it just refused. Fails before the
    fix, which printed render,video for every host.

    The expected group is read with the same stat the installer uses rather than assumed,
    since a test runner's primary group is not knowable in advance -- but asserting it is
    NOT render,video is what makes that comparison mean something."""
    import subprocess

    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    # 0660, the mode a real render node has: the installer now reads the mode as well as
    # the group, and a default 0644 is a node no membership opens.
    node.chmod(0o660)
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
    """The control that keeps the test above honest: an empty value exposes no device at
    all, so that host really does need both fixes and must still be told both."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "", {"hip"})
    assert "usermod -a -G render,video ada" in reason
    assert "HIP_VISIBLE_DEVICES is empty" in reason


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
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "", {"vulkan"})
    assert "usermod -a -G render,video ada" in reason
    assert "visibility mask" not in reason


def test_the_same_empty_mask_still_counts_for_a_hip_build(monkeypatch, linux):
    """The control for the arm above, one backend apart: the identical environment must
    still report the mask when the install is one that actually reads it."""
    reason = _reason_with_mask(monkeypatch, "HIP_VISIBLE_DEVICES", "", {"hip"})
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
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], []))
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
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: (["render"], [993], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G render ada" in hint
    assert "GID 993" in hint


def test_a_node_no_membership_opens_is_not_answered_with_usermod(monkeypatch, linux):
    """A udev rule leaving the node root:render 0600 denies its own group, so joining
    render runs, succeeds, and opens nothing. The repair there is the rule.

    Fails before the fix, which named the owning group whatever the mode said."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], ["/dev/kfd"], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G" not in hint
    assert "udev rule" in hint


def test_a_host_whose_nodes_could_not_be_read_still_gets_the_documented_pair(monkeypatch, linux):
    """And the control for BOTH suppressions: all three lists empty means the nodes could
    not be stat'd at all, which is a detection miss rather than evidence that joining
    cannot work. Some advice beats none there, and it is the pair AMD documents."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], []))
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


def _installer_index_summary(index_url: str, closed_nodes: str) -> str:
    """install.sh's index summary and the two diagnoses that follow it, run for one index.

    The whole span is lifted rather than the guard alone, because the thing under test is
    WHERE the diagnosis sits relative to the case: a copy of the condition would answer
    the same whichever arm it had been left in.
    """
    import subprocess

    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = max(i for i, line in enumerate(lines) if line == 'case "$TORCH_INDEX_URL" in')
    anchor = next(i for i in range(start, len(lines)) if "needs a recent kernel" in lines[i])
    # Through the closed-node block as well, so one run shows which of the two
    # diagnoses this index gets.
    last = next(i for i in range(anchor, len(lines)) if "membership opens it" in lines[i])
    end = next(i for i in range(last, len(lines)) if lines[i] == "fi")
    fn_start = next(i for i, line in enumerate(lines) if line.startswith("_amd_node_repairs() {"))
    depth = 0
    for fn_end in range(fn_start, len(lines)):
        depth += lines[fn_end].count("{") - lines[fn_end].count("}")
        if depth == 0:
            break
    script = "\n".join(
        [
            *lines[fn_start : fn_end + 1],
            'substep() { echo "$1"; }',
            'C_WARN=""',
            "_amd_gpu_radeon=false",
            '_strip_index_url_credentials() { printf "%s\\n" "$1"; }',
            "_has_amd_rocm_gpu() { return 1; }",  # ROCm cannot see the card
            "_amd_gpu_present_via_pci() { return 0; }",  # but the PCI bus can
            "SKIP_TORCH=false",
            "OS=linux",
            "_amd_render_node_present() { return 0; }",
            # The route gate classifies the index by its canonical leaf, so both
            # classifiers are lifted rather than stubbed: stubbing them would make the
            # per-URL cases below assert about the stub instead of about the rule.
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            # Defined above the case in install.sh, so the span lifted below calls them
            # without carrying them; a shell function has to exist before the call.
            _shell_fn(lines, "_run_may_open_kfd"),
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
    """clr reads HIP_VISIBLE_DEVICES when it is set and CUDA_VISIBLE_DEVICES only
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


def _diag_route(index_url: str) -> bool:
    """Whether install.sh routes the two node diagnoses for this wheel index.

    Lifted from install.sh rather than restated, since the thing under test is which
    patterns the case actually lists.
    """
    import subprocess

    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("_amd_node_diag_leaf=")
    )
    end = next(i for i in range(start, len(lines)) if lines[i] == "esac")
    script = "\n".join(
        [
            f"TORCH_INDEX_URL={index_url!r}",
            # The classifiers, not stubs: which leaves count as a ROCm route is exactly what
            # these tests are about, so a stub would have them assert about the stub.
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
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
    joinable, unnamed, no_group, acl, owned, _priv = amd._groups_that_own([str(node)])
    assert acl == [str(node)]
    assert joinable == [] and unnamed == [] and no_group == []


def test_the_same_node_without_an_acl_is_still_prescribed_for(monkeypatch, tmp_path):
    """The control: the ordinary node, whose mode bits ARE the group's grant. Without it
    the fix could decline to prescribe anywhere, which removes the repair #10466 needs."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, _priv = amd._groups_that_own([str(node)])
    assert acl == []
    assert joinable or unnamed


def test_the_installer_reports_an_acl_rather_than_prescribing_membership(tmp_path):
    """The shell twin of the same rule: ls marks such a node with a trailing "+", which
    is the marker available without getfacl."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    import subprocess

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
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993, 994], [], [], [], []))
    monkeypatch.setenv("USER", "ada")
    hint = amd.amd_node_permission_hint()
    assert "--group-add 993 --group-add 994" in hint
    assert "GIDs 993, 994" in hint


def test_a_lone_unnamed_gid_is_still_named_in_the_singular(monkeypatch, linux):
    """The control on the wording: the one-GID host is the common one and must not start
    reading as though it had several."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], []))
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


def test_a_uuid_in_the_hip_layer_is_not_reported_as_unresolved(monkeypatch, linux):
    """And the other boundary: only ROCr accepts a UUID. Reporting one for HIP would send
    the user after the wrong variable, and HIP's own handling of a non-ordinal entry is a
    separate question this does not answer."""
    reason = _reason_with_masks(
        monkeypatch,
        {"HIP_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58"},
        {"hip"},
        gpu_count = 1,
    )
    assert "cannot resolve" not in reason


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
    joinable, unnamed, no_group, acl, owned, privileged = amd._groups_that_own([str(node)])
    assert owned == [str(node)]
    assert joinable == [] and unnamed == [] and no_group == [] and privileged == []


def test_the_same_node_owned_by_someone_else_is_still_a_group(monkeypatch, tmp_path):
    """The control: identical mode, a different owner. The owner class no longer applies,
    the group bits are the grant, and membership IS the repair. Without this the rule
    could be "never prescribe a group", which removes what #10466 asked for."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o060)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, privileged = amd._groups_that_own([str(node)])
    assert owned == []
    assert joinable or unnamed


def test_a_root_owned_node_is_not_answered_with_usermod_root(monkeypatch, linux):
    """root:root 0660 opens for anyone in the root group, so the group derivation would
    accept the name and print `sudo usermod -a -G root`. That membership grants a
    great deal besides the GPU, so it is a udev misconfiguration to report rather than a
    repair to prescribe."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (0, 0o660, 0)}, {0: "root"})
    joinable, unnamed, no_group, acl, owned, privileged = amd._groups_that_own(["/dev/kfd"])
    assert privileged == ["root"]
    assert joinable == []


def test_an_ordinary_owning_group_is_still_prescribed(monkeypatch, linux):
    """The control: render is not privileged, so the same shape still yields the command.
    Without it the rule could be "never name a group"."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (39, 0o660, 0)}, {39: "render"})
    joinable, unnamed, no_group, acl, owned, privileged = amd._groups_that_own(["/dev/kfd"])
    assert joinable == ["render"]
    assert privileged == []


def test_the_hint_for_a_privileged_owner_says_it_is_not_the_repair(monkeypatch, linux):
    """The sentence a user actually reads, since the buckets above only decide it."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], [], [], ["root"]))
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
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
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


def _kernel_stack_hint_text(*, topology: bool) -> str:
    """What install.sh actually PRINTS in the missing-/dev/kfd branch.

    `_kernel_stack_hint_runs` above lifts only the guard, so it answers whether the
    branch fires and nothing about which repair it names -- which is exactly where the
    branch was wrong. This lifts the guard AND its body, through the closing `fi`, so a
    revert changes the text this returns.
    """
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    end = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("! _has_amd_rocm_gpu && _amd_gpu_present_via_pci; then")
    )
    start = end
    while not lines[start].lstrip().startswith("if "):
        start -= 1
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
            _shell_fn(lines, "_run_may_open_kfd"),
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
    amd_smi_sees_it: bool,
    skip_torch: bool = False,
    backend: "str | None" = None,
) -> str:
    """What the installer says when /dev/kfd is absent, for a given pair of probes.

    Lifts the two branches together, through the closing `fi`, because which of them runs
    is the thing under test. The `[ ! -e /dev/kfd ]` test is left live rather than stubbed
    -- a test operator cannot be stubbed, and rewriting it would be editing the code under
    test -- so the case needs a host without the node.
    """
    if os.path.exists(amd._KFD_NODE):
        pytest.skip("this arm needs a host with no /dev/kfd, and cannot remove a device node")
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    # Anchored on the kernel-stack condition, which this change does not touch, then walked
    # back to the `if` above it. Anchoring on the new mapping condition would make the
    # control vacuous: a revert would stop the extraction finding anything, and "the text
    # changed" would read as "the behaviour changed".
    end = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip().endswith("! _has_amd_rocm_gpu && _amd_gpu_present_via_pci; then")
    )
    start = end
    while not lines[start].lstrip().startswith("if "):
        start -= 1
    close = next(i for i in range(end + 1, len(lines)) if lines[i] == "fi")
    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            'C_WARN=""',
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            "OS=linux",
            "_amd_node_diag_route=true",
            _shell_fn(lines, "_run_may_open_kfd"),
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if topology else 1}; }}",
            f"_has_amd_rocm_gpu() {{ return {0 if amd_smi_sees_it else 1}; }}",
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
    joinable, unnamed, no_group, acl, owned, privileged = amd._groups_that_own(["/dev/kfd"])
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
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993, 994], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "993, 994" in hint
    assert "each of them" in hint
    assert "sudo groupadd -g 993" in hint and "sudo usermod -a -G <name>" in hint
    assert "--group-add 993 --group-add 994" in hint


def test_a_single_unnamed_gid_reads_singular(monkeypatch, linux):
    """The control: the plural wording must not be the only wording, or one GID reads as two
    and the sentence stops matching what it printed."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [993], [], [], [], []))
    hint = amd.amd_node_permission_hint()
    assert "GID 993" in hint and "GIDs" not in hint
    assert "create a group for it" in hint
    assert "sudo groupadd -g 993" in hint


def test_the_installer_also_adds_the_account_for_unnamed_gids(tmp_path):
    """The installer twin of the rule above: it printed the container flags per GID after the
    earlier fix, but still said only "create a group" for the bare host."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993\ngid:994")
    assert "sudo groupadd -g 993" in out
    assert "sudo usermod -a -G <name> ada" in out
    assert "--group-add 993 --group-add 994" in out
    assert "create a group for each" in out


def _install_sh_kfd_scope(closed_nodes: str, *, skip_torch: bool, backend: "str | None") -> str:
    """The closed-node message with the KFD scoping in front of it.

    A separate lift from _install_sh_hint because the filter sits ABOVE the block that
    harness extracts -- deliberately, so the diagnosis itself stays one self-contained
    block -- and the thing under test here is which nodes reach it.
    """
    import subprocess

    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
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
            "id() { echo 4242; }",
            "_amd_node_diag_route=true",
            "OS=linux",
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            "_amd_node_repairs() { printf '%s\\n' 'join:render'; }",
            _shell_fn(lines, "_run_may_open_kfd"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            "\n".join(lines[_filter_start : _filter_end + 1]),
            "\n".join(lines[block_start : end + 1]),
        ]
    )
    env = {**os.environ, "_closed_amd_nodes": closed_nodes, "USER": "ada"}
    env.pop("UNSLOTH_LLAMA_CPP_BACKEND", None)
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
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
        _install_sh_kfd_scope("/dev/dri/renderD128", skip_torch = True, backend = "cuda").strip()
        == ""
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
        lambda paths: ([], [], [], ["/dev/kfd", "/dev/dri/renderD128"], [], []),
    )
    hint = amd.amd_node_permission_hint()
    assert "getfacl /dev/kfd /dev/dri/renderD128" in hint


def test_the_acl_sentence_is_unchanged_for_a_single_node(monkeypatch, linux):
    """The control: one path in, one path out, so the fix cannot have introduced a stray
    separator into the common case."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: ([], [], [], ["/dev/kfd"], [], []))
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
