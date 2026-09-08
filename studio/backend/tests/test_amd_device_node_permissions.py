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
import subprocess
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

    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
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
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
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


def _kernel_stack_hint_runs(closed_nodes: str) -> bool:
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
