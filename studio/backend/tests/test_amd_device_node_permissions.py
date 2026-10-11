# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unopenable AMD device node must be named, not read as no GPU; the fix is group membership."""

from __future__ import annotations

import builtins
import getpass
import inspect
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path


# grp/pwd are POSIX-only; a module-level import breaks collection on Windows.
try:
    import grp
    import pwd
except ModuleNotFoundError:
    grp = None
    pwd = None


def _running_as_root() -> bool:
    """Root check that is safe on Windows: skipif decorators run at import, where os.geteuid is absent."""
    return getattr(os, "geteuid", lambda: -1)() == 0


import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import amd

# No /dev or POSIX groups on Windows; macOS still runs the faked-Linux cases.
pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason = (
        "POSIX-only subject: no /dev device nodes, no grp/pwd, and the probe under test "
        "returns [] on Windows by construction"
    ),
)
from utils.hardware import hardware


_GPU_MASK_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


@pytest.fixture(autouse = True)
def _no_inherited_gpu_mask(monkeypatch):
    """Clear every GPU mask variable per test, including GPU_DEVICE_ORDINAL, which ROCm or OpenCL export."""
    for _var in _GPU_MASK_VARS:
        monkeypatch.delenv(_var, raising = False)


@pytest.fixture(autouse = True)
def _the_account_this_process_runs_as(monkeypatch):
    """Stubs passwd so the account is not the runner's own, which differs per machine."""
    # Real struct_passwd: getpass.getuser() subscripts it if no LOGNAME/USER/LNAME/USERNAME is set.
    _record = pwd.struct_passwd(("ada", "x", os.getuid(), os.getgid(), "", "/home/ada", "/bin/sh"))
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _record)


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")


_NO_SUCH_SERVER = "/nonexistent/llama-server"

_AMD_NODES = ["/dev/kfd", "/dev/dri/renderD128"]

_BUCKETS = (
    "joinable",
    "unnamed",
    "no_group",
    "acl",
    "owned",
    "privileged",
    "already",
    "external",
)


def _buckets(**named) -> tuple:
    """The tuple ``_groups_that_own`` returns, naming only the buckets a case populates."""
    assert not set(named) - set(_BUCKETS), f"unknown bucket: {set(named) - set(_BUCKETS)}"
    return tuple(list(named.get(_name, [])) for _name in _BUCKETS)


# Captured before _nodes() stubs it, so derivation tests can put the real one back.
_REAL_GROUPS_THAT_OWN = amd._groups_that_own


def _the_real_group_derivation(monkeypatch):
    """Undo _nodes()'s stub, for a case whose point is what the nodes themselves say."""
    monkeypatch.setattr(amd, "_groups_that_own", _REAL_GROUPS_THAT_OWN)


def _owning(monkeypatch, **named):
    """Stubbed outside the derivation's own test, since patched node paths have no real groups to stat."""
    _answer = _buckets(**named)
    monkeypatch.setattr(amd, "_groups_that_own", lambda paths: _answer)


def _ggml(monkeypatch, backends):
    """The ggml libraries this llama.cpp install ships, which decides whose node it opens."""
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda _b: frozenset(backends))
    )


def _empty_probe() -> str:
    """The reason ``_explain_empty_gpu_probe`` gives for a GPU probe that found nothing."""
    return LlamaCppBackend._explain_empty_gpu_probe(_NO_SUCH_SERVER)


def _capability_message(verdict: str, detail: "str | None" = None) -> str:
    """The line Studio shows for a GPU it can see and cannot use, for a given verdict."""
    return hardware._gpu_present_but_unusable_message(
        "video generation",
        verdict = (verdict, detail),
    )


def _mismatch_vendors(monkeypatch, vendors):
    """Which vendors qualified for the capability mismatch, which is whose card raised it."""
    monkeypatch.setattr(hardware, "CHAT_ONLY_MISMATCH_VENDORS", frozenset(vendors))


def _nodes(
    monkeypatch,
    *,
    present: list[str],
    openable: set[str],
    amd_owned: bool = True,
    vendor_readable: bool = True,
    topology: "bool | None | str" = "as-owned",
    gpu_count: "int | None | str" = "as-present",
):
    """gpu_count is stubbed: a live KFD read reports the runner's GPUs, flipping selector verdicts."""
    # Only /dev/dri: the Vulkan icd.d walk shares glob and must see the real directories.
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
    monkeypatch.setattr(
        amd,
        "_render_node_vendor",
        lambda p: None if not vendor_readable else ("0x1002" if amd_owned else "0x10de"),
    )
    _topology = amd_owned if topology == "as-owned" else topology
    monkeypatch.setattr(amd, "_kfd_topology_has_an_amd_gpu", lambda: _topology is True)
    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: _topology)
    if gpu_count == "as-present":
        _renders = [p for p in present if p.startswith("/dev/dri/renderD")]
        gpu_count = len(_renders) if (_renders and amd_owned) else None
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: gpu_count)
    _owning(monkeypatch)


# Hand-wrapped tables, one case per row.
# fmt: off
_KFD_SHUT = dict(present = ["/dev/kfd"], openable = set())
_RENDER_SHUT = dict(present = ["/dev/dri/renderD128"], openable = set())
_BOTH_SHUT = dict(present = _AMD_NODES, openable = set())
_BOTH_OPEN = dict(present = _AMD_NODES, openable = set(_AMD_NODES))
_KFD_OPEN = dict(present = ["/dev/kfd"], openable = {"/dev/kfd"})
_RENDER_OPEN = dict(present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
_ONLY_KFD_SHUT = dict(present = _AMD_NODES, openable = {"/dev/dri/renderD128"})
_NO_NODES = dict(present = [], openable = set())
_NVIDIA_RENDER_SHUT = dict(present = ["/dev/dri/renderD128"], openable = set(), amd_owned = False)
_NVIDIA_KFD_OPEN = dict(present = ["/dev/kfd"], openable = {"/dev/kfd"}, amd_owned = False)


@pytest.mark.parametrize("case", [
    pytest.param(("Linux", _BOTH_SHUT, _AMD_NODES), id = "a_node_this_user_cannot_open"),
    pytest.param(("Linux", _BOTH_OPEN, []), id = "a_host_whose_nodes_open"),
    pytest.param(("Linux", _NO_NODES, []), id = "a_host_with_no_amd_nodes"),
    # Render nodes are root:render for every vendor; NVIDIA hosts must not be flagged.
    pytest.param(("Linux", _NVIDIA_RENDER_SHUT, []), id = "an_nvidia_hosts_closed_render_node"),
    pytest.param(("Windows", _KFD_SHUT, []), id = "the_probe_is_linux_only"),
])
def test_which_nodes_are_reported_closed(monkeypatch, case):
    """The closed list itself. What these hosts are then TOLD is the two families below."""
    system, layout, closed = case
    monkeypatch.setattr(amd.platform, "system", lambda: system)
    _nodes(monkeypatch, **layout)
    assert amd.amd_nodes_closed_to_this_user() == closed


def test_the_vendor_is_read_from_sysfs(monkeypatch):
    """The reader the cases above stub. sysfs is world-readable, so ownership is answerable
    without the access being tested for."""
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
    assert amd._render_node_is_amd("/dev/dri/renderD130") is False


def test_a_node_that_cannot_be_stat_ed_is_skipped(monkeypatch, linux):
    """A probe that raises must not break a load; the caller is a diagnostic."""

    def _boom(_p):
        raise OSError("stale handle")

    monkeypatch.setattr(amd.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(amd.os.path, "exists", _boom)
    assert amd.amd_nodes_closed_to_this_user() == []


def test_read_only_access_is_not_enough(monkeypatch, linux):
    """HIP and the Vulkan loader both open the node read-write, so a node that only reads is
    still unusable and answering "fine" here would restore the silence."""
    _nodes(monkeypatch, **_KFD_SHUT)
    monkeypatch.setattr(amd.os, "access", lambda p, mode: mode == os.R_OK)
    assert amd.amd_nodes_closed_to_this_user() == ["/dev/kfd"]


_USERMOD = "usermod -a -G render,video ada"


def _asserts(text: str, says, does_not_say):
    """Every ``says`` is in ``text`` and every ``does_not_say`` is not. Both directions run:
    a family that asserted only the positive half would pass on a message that also said the
    thing it is meant to have stopped saying."""
    for _wanted in says:
        assert _wanted in text
    for _unwanted in does_not_say:
        assert _unwanted not in text


@pytest.mark.parametrize("case", [
    pytest.param((_KFD_SHUT, {}, True, ("/dev/kfd", _USERMOD), ()),
                 id = "the_hint_names_the_nodes_the_groups_and_the_account"),
    pytest.param((_KFD_SHUT, {}, True, ("ROCm cannot use",), ("no GPU backend",)),
                 id = "the_hint_is_rocm_specific_when_only_kfd_is_closed"),
    pytest.param((_RENDER_SHUT, {}, True, ("no GPU backend can use",), ()),
                 id = "the_hint_covers_every_backend_when_a_render_node_is_closed"),
    # No membership creates /dev/kfd, so a ROCm caller needs both sentences.
    pytest.param((_RENDER_SHUT, {}, True, (_USERMOD, "/dev/kfd"), ("kernel stack",)),
                 id = "the_hint_says_so_when_kfd_does_not_exist_at_all"),
    pytest.param((_BOTH_SHUT, {}, True, (), ("kernel stack",)),
                 id = "a_closed_but_present_kfd_node_says_nothing_about_the_kernel_stack"),
    pytest.param((_RENDER_SHUT, {}, False, (), ("kernel stack",)),
                 id = "a_vulkan_caller_is_not_told_about_a_kernel_stack_it_does_not_need"),
    # Groups are derived, not hard-coded render,video (containers, minimal distros).
    pytest.param((_BOTH_SHUT, dict(joinable = ["kfd", "gpu"]), True, ("usermod -a -G kfd,gpu ada",),
                  ("render,video",)),
                 id = "the_repair_names_the_groups_the_closed_nodes_belong_to"),
    pytest.param((_RENDER_SHUT, dict(joinable = ["render"]), True,
                  ("usermod -a -G render ada", "render group and then log out"), ()),
                 id = "a_single_owning_group_is_not_pluralised"),
    pytest.param((_KFD_SHUT, {}, True, (_USERMOD,), ()),
                 id = "unreadable_nodes_fall_back_to_the_documented_pair"),
    # usermod -G takes names only; numeric GIDs go to --group-add.
    pytest.param((_KFD_SHUT, dict(unnamed = [993]), True, ("--group-add 993",),
                  ("usermod -a -G 993",)),
                 id = "an_unnamed_gid_is_not_handed_to_usermod"),
    pytest.param((_BOTH_SHUT, dict(joinable = ["render"], unnamed = [993]), True,
                  ("usermod -a -G render ada", "GID 993"), ()),
                 id = "a_joinable_group_beside_an_unnamed_gid_is_still_prescribed"),
    # A 0600 udev rule denies its own group, so joining fixes nothing.
    pytest.param((_KFD_SHUT, dict(no_group = ["/dev/kfd"]), True, ("udev rule",),
                  ("usermod -a -G",)),
                 id = "a_node_no_membership_opens_is_not_answered_with_usermod"),
    pytest.param((_KFD_SHUT, {}, True, (_USERMOD,), ()),
                 id = "a_host_whose_nodes_could_not_be_read_still_gets_the_documented_pair"),
    # ROCr also needs a render node to reach amdgpu.
    pytest.param((_KFD_SHUT, {}, True,
                  ("No AMD render node", "--device /dev/kfd --device /dev/dri"), ()),
                 id = "a_container_given_kfd_but_no_render_node_is_told_so"),
    pytest.param((_BOTH_SHUT, {}, True, (), ("No AMD render node",)),
                 id = "a_host_that_has_a_render_node_is_not_told_to_map_one"),
    pytest.param((_KFD_SHUT, dict(external = ["/dev/kfd"]), True,
                  ("granted read and write by the permission bits that apply to this account",
                   "container device cgroup or an LSM"),
                  ("chmod", "is owned by this account")),
                 id = "the_external_denial_sentence_does_not_prescribe_a_mode_change"),
    # docker --group-add takes one value, so the flag repeats.
    pytest.param((_BOTH_SHUT, dict(unnamed = [993, 994]), True,
                  ("--group-add 993 --group-add 994", "GIDs 993, 994"), ()),
                 id = "every_unnamed_gid_reaches_the_docker_repair"),
    pytest.param((_KFD_SHUT, dict(unnamed = [993]), True,
                  ("GID 993, which has", "--group-add 993."), ()),
                 id = "a_lone_unnamed_gid_is_still_named_in_the_singular"),
    # Never prescribe joining root.
    pytest.param((_KFD_SHUT, dict(privileged = ["root"]), True, ("root", "udev"), ("usermod",)),
                 id = "the_hint_for_a_privileged_owner_says_it_is_not_the_repair"),
    pytest.param((_KFD_OPEN, {}, True, ("AMD render node", "--device /dev/dri"), ("usermod",)),
                 id = "a_container_with_an_open_kfd_and_no_render_node_is_still_told"),
    pytest.param((_RENDER_OPEN, {}, True, ("/dev/kfd",), ("usermod",)),
                 id = "a_container_given_only_the_render_node_is_told_about_kfd"),
])
def test_what_the_hint_says_about_a_host(monkeypatch, linux, case):
    """The sentence a user reads, per host and per owning-group derivation. ``owning`` stubs
    _groups_that_own, which _nodes defaults to the empty answer; the derivation itself is
    exercised against real modes further down."""
    layout, owning, needs_kfd, says, does_not_say = case
    _nodes(monkeypatch, **layout)
    _owning(monkeypatch, **owning)
    hint = amd.amd_node_permission_hint(needs_kfd = needs_kfd)
    assert hint is not None
    _asserts(hint, says, does_not_say)


@pytest.mark.parametrize("case", [
    pytest.param((_BOTH_OPEN, True, False), id = "a_host_whose_nodes_open_is_told_nothing"),
    pytest.param((_NVIDIA_RENDER_SHUT, True, False), id = "an_nvidia_host_is_told_nothing"),
    pytest.param((_ONLY_KFD_SHUT, False, False),
                 id = "a_vulkan_only_caller_is_not_answered_with_a_closed_kfd_node"),
    pytest.param((_ONLY_KFD_SHUT, True, True), id = "a_rocm_caller_on_that_host_is_answered"),
    pytest.param((_RENDER_SHUT, False, True),
                 id = "a_vulkan_caller_is_still_answered_about_a_closed_render_node"),
    pytest.param((_NVIDIA_KFD_OPEN, True, False),
                 id = "a_host_with_no_amd_card_is_not_told_to_map_a_render_node"),
    pytest.param((_RENDER_OPEN, False, False),
                 id = "the_same_mapping_says_nothing_to_a_vulkan_caller"),
])
def test_whether_the_hint_answers_at_all(monkeypatch, linux, case):
    """Which hosts get a hint at all, before asking what it says."""
    layout, needs_kfd, answered = case
    _nodes(monkeypatch, **layout)
    assert (amd.amd_node_permission_hint(needs_kfd = needs_kfd) is not None) is answered


def _no_passwd_entry(monkeypatch):
    """A uid the passwd database does not know, which is where the environment is read."""

    def _missing(_uid):
        raise KeyError(_uid)

    monkeypatch.setattr(pwd, "getpwuid", _missing)


@pytest.mark.parametrize("case", [
    # docker run --user <uid> has no passwd entry; usermod on USER fixes nothing.
    pytest.param((dict(joinable = ["render"]), None, ("--group-add render",),
                  ("usermod -a -G", "root")),
                 id = "a_uid_with_no_passwd_entry_is_not_given_a_usermod"),
    pytest.param((dict(joinable = ["render"]), "ada", ("sudo usermod -a -G render ada",),
                  ("--group-add",)),
                 id = "an_account_the_system_knows_still_gets_the_command"),
    pytest.param((dict(unnamed = [993]), None, ("--group-add 993",),
                  ("usermod -a -G", "groupadd -g")),
                 id = "an_unnamed_gid_under_that_uid_drops_the_groupadd_half_too"),
])
def test_which_account_the_repair_names(monkeypatch, linux, case):
    """Whether the repair can name an account at all. USER is root in every arm, so a command
    that names it is one that read the environment instead of the passwd entry."""
    owning, account, says, does_not_say = case
    _nodes(monkeypatch, **_KFD_SHUT)
    monkeypatch.setenv("USER", "root")
    _owning(monkeypatch, **owning)
    if account is None:
        _no_passwd_entry(monkeypatch)
    else:
        monkeypatch.setattr(amd, "_repair_account", lambda: account)
    hint = amd.amd_node_permission_hint()
    _asserts(hint, says, does_not_say)


def _pins(
    rocm_intent = None,
    hip_runtime = None,
    other_vendor = None,
) -> dict:
    """Runtime probes are pinned: live ones would read the runner's torch wheel, not the described host."""
    _named = {
        "_expected_rocm_flavor_was_chosen": rocm_intent,
        "_torch_reports_a_hip_runtime": hip_runtime,
        "_torch_reports_another_vendors_runtime": other_vendor,
    }
    return {_probe: _answer for _probe, _answer in _named.items() if _answer is not None}


@pytest.mark.parametrize("case", [
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  (_USERMOD,), ("Repair installation",)),
                 id = "the_capability_message_names_the_permission_not_a_torch_mismatch"),
    pytest.param((_BOTH_OPEN, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  ("Repair installation",), ("usermod",)),
                 id = "the_capability_message_is_unchanged_when_the_nodes_open"),
    pytest.param((_KFD_SHUT, {"nvidia"}, "torch_cuda_unavailable", "2.11.0+cu130", _pins(),
                  ("Repair installation",), ("usermod",)),
                 id = "an_nvidia_mismatch_keeps_the_pytorch_message"),
    pytest.param((_KFD_SHUT, {"amd", "nvidia"}, "torch_cpu_build", None,
                  _pins(rocm_intent = True, hip_runtime = False, other_vendor = False), (_USERMOD,),
                  ()),
                 id = "a_hybrid_host_whose_amd_card_raised_it_still_gets_the_hint"),
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cpu_build", None, _pins(),
                  ("CPU-only build", "Repair installation", _USERMOD), ()),
                 id = "a_cpu_wheel_beside_a_closed_node_is_told_to_do_both"),
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  (_USERMOD,), ("Repair installation",)),
                 id = "a_gpu_wheel_beside_a_closed_node_is_told_only_the_permission"),
    pytest.param((_KFD_SHUT, {"amd", "nvidia"}, "torch_cuda_unavailable", "2.11.0+cu130",
                  _pins(rocm_intent = False, hip_runtime = False), ("Repair installation",),
                  ("usermod",)),
                 id = "a_hybrid_host_running_cuda_torch_keeps_the_pytorch_message"),
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0",
                  _pins(rocm_intent = False, hip_runtime = False), (_USERMOD,), ()),
                 id = "an_amd_only_host_needs_no_runtime_evidence"),
    # CUDA wheel on an AMD host: the hint is appended and reinstall advice kept.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+cu128",
                  _pins(rocm_intent = False, hip_runtime = False),
                  (_USERMOD, "Repair installation"), ()),
                 id = "a_cuda_wheel_on_an_amd_only_host_keeps_the_reinstall_advice"),
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0",
                  _pins(rocm_intent = False, hip_runtime = False), (_USERMOD,),
                  ("Repair installation",)),
                 id = "a_rocm_wheel_on_the_same_host_still_replaces_it"),
    pytest.param((_BOTH_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+cu128",
                  _pins(rocm_intent = True, hip_runtime = False),
                  ("matching PyTorch build fixes it", "cannot open"), ()),
                 id = "a_wheel_tagged_for_another_vendor_keeps_the_reinstall_advice"),
    pytest.param((_BOTH_SHUT, {"amd", "nvidia"}, "torch_cpu_build", "2.11.0+cpu",
                  _pins(rocm_intent = True, hip_runtime = False, other_vendor = False),
                  ("cannot open",), ()),
                 id = "a_label_that_names_no_vendor_still_lets_the_intent_speak"),
])
def test_the_capability_message_for_a_host(monkeypatch, linux, case):
    """The line Studio shows for a GPU it can see and cannot use."""
    layout, vendors, verdict, detail, pins, says, does_not_say = case
    _nodes(monkeypatch, **layout)
    _mismatch_vendors(monkeypatch, vendors)
    for _probe, _answer in pins.items():
        monkeypatch.setattr(hardware, _probe, lambda _a = _answer: _a)
    message = _capability_message(verdict, detail)
    _asserts(message, says, does_not_say)


@pytest.mark.parametrize("case", [
    pytest.param((_RENDER_SHUT, {"vulkan"}, {}, (_USERMOD,),
                  ("the Vulkan probe reported no device",)),
                 id = "the_empty_probe_explanation_names_the_permission"),
    pytest.param((_ONLY_KFD_SHUT, {"hip"}, {}, (_USERMOD,), ()),
                 id = "a_rocm_binary_is_still_told_about_the_closed_kfd_node"),
    pytest.param((_BOTH_SHUT, {"cuda"}, {"CUDA_VISIBLE_DEVICES": ""}, ("CUDA_VISIBLE_DEVICES",),
                  ("usermod",)),
                 id = "a_cuda_only_build_is_not_sent_after_the_amd_render_group"),
    pytest.param((_BOTH_SHUT, set(), {}, (_USERMOD,), ()),
                 id = "a_build_whose_backend_cannot_be_read_still_gets_the_hint"),
    pytest.param((_BOTH_SHUT, {"cuda", "vulkan"}, {"CUDA_VISIBLE_DEVICES": ""},
                  ("CUDA_VISIBLE_DEVICES",), ("usermod",)),
                 id = "a_cuda_plus_vulkan_build_is_treated_as_cuda"),
    pytest.param((_BOTH_SHUT, {"cpu", "base"}, {}, (), ("usermod",)),
                 id = "a_cpu_only_llama_build_is_not_sent_after_the_groups"),
    pytest.param((_BOTH_SHUT, {"hip"}, {"HIP_VISIBLE_DEVICES": "-1"},
                  (_USERMOD, "HIP_VISIBLE_DEVICES='-1'"), ()),
                 id = "a_mask_is_reported_alongside_the_permission_hint"),
])
def test_the_empty_probe_reason_for_an_install(monkeypatch, linux, case):
    """Which reason the load-time explanation gives, per install and node layout."""
    layout, backends, env, says, does_not_say = case
    _nodes(monkeypatch, **layout)
    for _var, _value in env.items():
        monkeypatch.setenv(_var, _value)
    _ggml(monkeypatch, backends)
    reason = _empty_probe()
    _asserts(reason, says, does_not_say)


@pytest.mark.parametrize("layout", [
    pytest.param(_RENDER_OPEN, id = "the_explanation_is_unchanged_when_the_nodes_open"),
    pytest.param(_ONLY_KFD_SHUT, id = "the_vulkan_reason_survives_a_closed_kfd_node"),
])
def test_a_vulkan_host_whose_own_node_opens_keeps_its_own_reason(monkeypatch, linux, layout):
    """The control for the family above: a Vulkan host that can open what Vulkan uses keeps
    its own reason, whole and unqualified."""
    _nodes(monkeypatch, **layout)
    _ggml(monkeypatch, {"vulkan"})
    assert _empty_probe() == "the Vulkan probe reported no device"


def test_no_mask_leaves_the_hint_alone(monkeypatch, linux):
    """The control: the sentence must not grow a trailing clause on a host with no mask set,
    which is every host the #10466 wording was written for."""
    _nodes(monkeypatch, **_BOTH_SHUT)
    for var in _GPU_MASK_VARS:
        monkeypatch.delenv(var, raising = False)
    _ggml(monkeypatch, {"hip"})
    reason = _empty_probe()
    assert reason.endswith("sudo usermod -a -G render,video ada")


def _kernel_stack_hint_runs(
    tmp_path,
    closed_nodes: str,
    *,
    route: bool = True,
    nvidia: bool = False,
) -> bool:
    """Lifted from install.sh by text, since a restated copy would pass whatever the installer prints."""
    lines = _install_sh_lines()
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    guard = "\n".join(line.strip() for line in lines[start : end + 1])
    script = "\n".join(
        [
            "_has_amd_rocm_gpu() { return 1; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            "SKIP_TORCH=false",
            "OS=linux",
            *_run_scope_defs(lines, nvidia = nvidia),
            f"_amd_node_diag_route={'true' if route else 'false'}",
            # Stub substep and topology: undefined ones fall through or exit 127.
            'substep() { echo "$1"; }',
            'C_WARN=""',
            # Stubbed false so the third arm answers, even on a real gfx1151.
            "_kfd_topology_has_an_amd_gpu() { return 1; }",
            guard,
            "    echo FIRED",
            "fi",
        ]
    )
    script = _kfd_node_the_case_owns(script, tmp_path, present = False)
    # Exported, not repr()'d into single quotes, which would make the newline a literal backslash-n.
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        check = True,
        env = {**os.environ, "_closed_amd_nodes": closed_nodes},
    )
    return "FIRED" in out.stdout


@pytest.mark.parametrize("case", [
    pytest.param(("/dev/kfd", True, False), id = "a_closed_kfd_node_suppresses_it"),
    pytest.param(("/dev/kfd\n/dev/dri/renderD128", True, False),
                 id = "a_closed_pair_suppresses_it"),
    pytest.param(("/dev/dri/renderD128", True, True), id = "a_missing_kfd_node_keeps_it"),
    pytest.param(("", True, True), id = "nothing_closed_keeps_it"),
    pytest.param(("", False, False), id = "the_route_gate_actually_suppresses_it"),
])
def test_when_the_installers_kernel_stack_hint_fires(tmp_path, case):
    """Which hosts install.sh tells to install a ROCm kernel stack."""
    closed_nodes, route, fires = case
    assert _kernel_stack_hint_runs(tmp_path, closed_nodes, route = route) is fires


def _stat_nodes(monkeypatch, modes: dict, names: dict):
    """The stat stub takes follow_symlinks, as pytest stats files while the patch is live."""

    def _stat(path, *, follow_symlinks = True):
        if str(path) not in modes:
            raise OSError("gone")
        _entry = modes[str(path)]
        gid, mode = _entry[0], _entry[1]
        # Default uid 0: real nodes are root-owned and the owner class is checked first.
        uid = _entry[2] if len(_entry) > 2 else 0
        if uid == amd.os.getuid():
            uid += 1
        return type("st", (), {"st_gid": gid, "st_mode": mode, "st_uid": uid})()

    def _getgrgid(gid):
        if gid not in names:
            raise KeyError(gid)
        return type("gr", (), {"gr_name": names[gid]})()

    monkeypatch.setattr(amd.os, "stat", _stat)
    monkeypatch.setattr(grp, "getgrgid", _getgrgid)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda _path: False)
    monkeypatch.setattr(amd.os, "getgid", lambda: 1)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])


_THREE_NODES = {
    "/dev/kfd": (44, 0o660),
    "/dev/dri/renderD128": (44, 0o660),
    "/dev/dri/renderD129": (39, 0o660),
}


@pytest.mark.parametrize("case", [
    pytest.param((_THREE_NODES, {44: "video", 39: "render"},
                  ["/dev/dri/renderD129", "/dev/kfd", "/dev/dri/renderD128"],
                  dict(joinable = ["render", "video"])),
                 id = "the_group_derivation_reads_the_node"),
    pytest.param(({"/dev/kfd": (993, 0o660)}, {}, ["/dev/kfd"], dict(unnamed = [993])),
                 id = "a_gid_with_no_group_entry_is_reported_rather_than_prescribed"),
    pytest.param(({"/dev/kfd": (44, 0o600)}, {44: "render"}, ["/dev/kfd"],
                  dict(no_group = ["/dev/kfd"])),
                 id = "a_node_whose_own_group_cannot_open_it_is_not_a_membership_problem"),
    pytest.param(({"/dev/kfd": (44, 0o640)}, {44: "render"}, ["/dev/kfd"],
                  dict(no_group = ["/dev/kfd"])),
                 id = "group_read_without_write_is_not_enough"),
    pytest.param(({"/dev/dri/renderD128": (44, 0o660)}, {44: "video"},
                  ["/dev/kfd", "/dev/dri/renderD128"], dict(joinable = ["video"])),
                 id = "a_node_that_cannot_be_stat_contributes_nothing"),
    pytest.param(({"/dev/kfd": (0, 0o660, 0)}, {0: "root"}, ["/dev/kfd"],
                  dict(privileged = ["root"])),
                 id = "a_root_owned_node_is_not_answered_with_usermod_root"),
    pytest.param(({"/dev/kfd": (39, 0o660, 0)}, {39: "render"}, ["/dev/kfd"],
                  dict(joinable = ["render"])),
                 id = "an_ordinary_owning_group_is_still_prescribed"),
])
def test_the_group_derivation_over_a_node_set(monkeypatch, case):
    """The helper itself, since every hint case above stubs it. Every bucket a case does not
    name has to come back empty, or the hint offers a repair the node does not support."""
    modes, names, paths, buckets = case
    _stat_nodes(monkeypatch, modes, names)
    assert amd._groups_that_own(paths) == _buckets(**buckets)


# Anchor on the printed sentence: the conditions change and the PCI predicate appears twice.
_PCI_SENTENCE = "An AMD GPU is on the PCI bus but ROCm cannot see it"


def _install_sh_lines() -> "list[str]":
    """install.sh, split into lines. Every harness below lifts what it needs out of this."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    return install_sh.read_text(encoding = "utf-8").splitlines()


# `[ -e /dev/kfd ]` cannot be stubbed, so only that path is redirected inside existence tests;
# the condition and printed sentences stay verbatim.
_KFD_EXISTENCE_TEST = re.compile(r"(\[ +!? ?-e +)/dev/kfd\b")


def _kfd_node_the_case_owns(script: str, tmp_path, *, present: bool) -> str:
    """Point install.sh's `-e /dev/kfd` tests at a node this case creates or withholds."""
    node = tmp_path / "kfd"
    if present:
        node.write_bytes(b"")
    elif node.exists():
        node.unlink()
    redirected, n = _KFD_EXISTENCE_TEST.subn(rf"\g<1>{node}", script)
    assert n, "install.sh no longer tests `-e /dev/kfd` where this harness expects it"
    return redirected


def _install_sh_if_above(lines: "list[str]", i: int) -> int:
    """Finds the if opening a multi-line condition, so a miss raises rather than testing a shorter
    script."""
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
    """Lifts one shell function by brace depth; a miss raises, since an undefined one fails the block."""
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated {name}() in install.sh")


def _run_scope_defs(lines: "list[str]", *, nvidia: bool = False) -> "list[str]":
    """Lifted, so arms run the installer's own predicates; only the NVIDIA probe is stubbed."""
    return [
        _shell_fn(lines, "_requested_llama_backend"),
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


def _the_real_stat_derivation_runs_here() -> None:
    """Skipped off Linux, since BSD stat rejects -c and every node would wrongly read as join-a-group."""
    if platform.system() != "Linux":
        pytest.skip(
            "install.sh's stat|awk node classifier needs GNU stat -c, and the installer "
            "gates every diagnosis that uses it on OS != macos"
        )


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
    """Lifts the whole block, since the printed sentence is under test; the node list comes in via env."""
    if repairs is None:
        _the_real_stat_derivation_runs_here()
    lines = _install_sh_lines()
    start = _install_sh_if(lines, '[ -n "$_closed_amd_nodes" ]; then')
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[start : end + 1])

    helper = _shell_fn(lines, "_amd_node_repairs")

    script = "\n".join(
        [
            'substep() { echo "$1"; }',
            'C_WARN=""',
            # Stubbed: the real probes read /sys, /dev or run nvidia-smi on the runner.
            f"_amd_render_node_present() {{ return {0 if render_present else 1}; }}",
            # Stub both id -u and id -un; id_user None mimics docker --user with no passwd entry.
            (
                # printf with a single-quoted value: echo would de-escape backslashes.
                f'id() {{ case "$1" in -un) printf %s\\\\n {shlex.quote(id_user or "")} ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
                if id_user is not None
                else f'id() {{ case "$1" in -un) return 1 ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
            ),
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if amd_present else 1}; }}",
            f"_an_amd_render_node_is_open() {{ return {0 if render_open else 1}; }}",
            "_amd_node_diag_route=true",
            "OS=linux",
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_requested_llama_backend"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_kfd"),
            # Lifted: without _shell_quote the substitutions come back empty.
            _shell_fn(lines, "_shell_quote"),
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
    """Chooses a joinable group, since a root runner's inherited primary group is one the rule refuses."""
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
    """Guards the fixture: a privileged group pick would not fail, just assert the wrong sentence."""
    monkeypatch.setattr(
        amd,
        "_PRIVILEGED_GROUPS",
        frozenset(amd._PRIVILEGED_GROUPS | {grp.getgrgid(os.getgid()).gr_name}),
    )
    _node, _group = _a_node_a_membership_would_open(tmp_path)
    assert _group not in amd._PRIVILEGED_GROUPS


def test_the_installer_names_the_group_the_node_actually_has(tmp_path):
    """The installer must name the node's actual owning group, never a fixed render,video pair."""
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
    """The control: a path that cannot be stat'd still has to produce advice, and an empty -G
    argument would be worse than the hard-coded pair it replaced."""
    out = _install_sh_hint(str(tmp_path / "renderD128"))
    assert "usermod -a -G render,video ada" in out


def _reason_with_mask(monkeypatch, var: str, value: str, backends: set) -> str:
    """The empty-probe reason on a closed-node host carrying one visibility mask."""
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
    for other in _GPU_MASK_VARS:
        monkeypatch.delenv(other, raising = False)
    monkeypatch.setenv(var, value)
    _ggml(monkeypatch, backends)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda _b: backends == {"vulkan"}),
    )
    return _empty_probe()


@pytest.mark.parametrize("case", [
    pytest.param(("HIP_VISIBLE_DEVICES", "0", {"hip"}, (_USERMOD,), ("visibility mask",)),
                 id = "a_selector_that_still_exposes_a_device_is_not_a_second_blocker"),
    # -1 names no device; an empty HIP mask is not a filter to clr.
    pytest.param(("HIP_VISIBLE_DEVICES", "-1", {"hip"}, (_USERMOD, "HIP_VISIBLE_DEVICES='-1'"), ()),
                 id = "a_mask_that_hides_everything_is_still_reported"),
    # CUDA and HIP parse the list left to right and stop at the first entry that names no
    # device, so -1 leading the list leaves nothing enumerated.
    pytest.param(("CUDA_VISIBLE_DEVICES", "-1", {"hip"},
                  ("CUDA_VISIBLE_DEVICES='-1'", "visibility mask is also in force"), ()),
                 id = "a_negative_first_entry_hides_everything"),
    pytest.param(("CUDA_VISIBLE_DEVICES", "0,-1", {"hip"}, (), ("visibility mask",)),
                 id = "a_leading_valid_entry_survives_a_later_invalid_one"),
    pytest.param(("HIP_VISIBLE_DEVICES", "-1", {"vulkan"}, (_USERMOD,), ("visibility mask",)),
                 id = "a_vulkan_build_is_not_told_about_a_mask_it_never_reads"),
    pytest.param(("HIP_VISIBLE_DEVICES", "-1", {"hip"}, ("visibility mask is also in force",), ()),
                 id = "the_same_hiding_mask_still_counts_for_a_hip_build"),
])
def test_which_lone_masks_are_reported(monkeypatch, linux, case):
    """Whether one inherited visibility mask is named as a second blocker beside the groups.
    Every arm runs on the same host, with every AMD node closed."""
    var, value, backends, says, does_not_say = case
    reason = _reason_with_mask(monkeypatch, var, value, backends)
    _asserts(reason, says, does_not_say)


def _installer_index_summary(
    index_url: str,
    closed_nodes: str,
    tmp_path,
    *,
    nvidia: bool = False,
) -> str:
    """Lifts the whole span; /dev/kfd and the KFD topology are stubbed so the host cannot decide the arm."""
    lines = _install_sh_lines()
    start = max(i for i, line in enumerate(lines) if line == 'case "$TORCH_INDEX_URL" in')
    anchor = next(i for i in range(start, len(lines)) if "needs a recent kernel" in lines[i])
    last = next(i for i in range(anchor, len(lines)) if "membership opens it" in lines[i])
    end = next(i for i in range(last, len(lines)) if lines[i] == "fi")
    script = "\n".join(
        [
            _shell_fn(lines, "_amd_node_repairs"),
            'substep() { echo "$1"; }',
            'C_WARN=""',
            "_amd_gpu_radeon=false",
            '_strip_index_url_credentials() { printf "%s\\n" "$1"; }',
            "_has_amd_rocm_gpu() { return 1; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            "SKIP_TORCH=false",
            "OS=linux",
            "_amd_render_node_present() { return 0; }",
            "_kfd_topology_has_an_amd_gpu() { return 1; }",
            *_run_scope_defs(lines, nvidia = nvidia),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            *lines[start : end + 1],
        ]
    )
    script = _kfd_node_the_case_owns(script, tmp_path, present = False)
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


_GFX_INDEX = "https://repo.radeon.com/rocm/manylinux/gfx1151"


@pytest.mark.parametrize("case", [
    pytest.param((_GFX_INDEX, "", ("ROCm cannot see it",), ()),
                 id = "the_kernel_stack_diagnosis_reaches_a_rerouted_gfx_index"),
    pytest.param(("https://download.pytorch.org/whl/cpu", "", ("ROCm cannot see it",), ()),
                 id = "a_cpu_index_on_the_same_host_still_gets_it"),
    pytest.param((_GFX_INDEX, "/dev/kfd", ("cannot open its device nodes",),
                  ("ROCm cannot see it",)),
                 id = "a_closed_kfd_node_still_suppresses_it_after_the_case"),
])
def test_which_index_summary_carries_the_kernel_stack_diagnosis(tmp_path, case):
    """Which wheel index the diagnosis prints for, now that it sits after the case."""
    index_url, closed_nodes, says, does_not_say = case
    out = _installer_index_summary(index_url, closed_nodes, tmp_path)
    _asserts(out, says, does_not_say)


def _reason_with_masks(
    monkeypatch,
    env: dict,
    backends: set,
    gpu_count: "int | None" = None,
) -> str:
    """gpu_count is what KFD enumerates so selectors can be judged; None means an unreadable topology."""
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: gpu_count)
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "GGML_VK_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising = False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    _ggml(monkeypatch, backends)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_is_vulkan_backend",
        staticmethod(lambda _b: backends == {"vulkan"}),
    )
    return _empty_probe()


_UUID = "GPU-4b2c9f1e0a7d3b58"
_IN_FORCE = "visibility mask is also in force"


@pytest.mark.parametrize("case", [
    pytest.param(({"GPU_DEVICE_ORDINAL": ""}, None, (_USERMOD,), ("visibility mask",)),
                 id = "an_empty_ordinal_variable_is_not_a_filter"),
    pytest.param(({"GPU_DEVICE_ORDINAL": "-1"}, None, (_IN_FORCE, "GPU_DEVICE_ORDINAL='-1'"), ()),
                 id = "an_ordinal_that_hides_everything_is_still_reported"),
    # clr reads HIP_VISIBLE_DEVICES if non-empty, else CUDA_VISIBLE_DEVICES.
    pytest.param(({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": ""}, None, (_USERMOD,),
                  ("visibility mask",)),
                 id = "an_empty_cuda_mask_behind_a_valid_hip_one_is_not_consulted"),
    pytest.param(({"CUDA_VISIBLE_DEVICES": ""}, None, ("CUDA_VISIBLE_DEVICES is empty", _IN_FORCE),
                  ()),
                 id = "the_same_empty_cuda_mask_blocks_once_hip_is_unset"),
    # ROCr filters below clr and composes: an empty ROCr mask hides everything.
    pytest.param(({"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""}, None,
                  ("ROCR_VISIBLE_DEVICES is empty", _IN_FORCE), ()),
                 id = "an_empty_rocr_mask_blinds_the_runtime_under_a_valid_hip_one"),
    # HIP stops at the first index with no device.
    pytest.param(({"HIP_VISIBLE_DEVICES": "3"}, 1, (_IN_FORCE, "HIP_VISIBLE_DEVICES='3'"), ()),
                 id = "an_ordinal_naming_a_device_that_is_not_there_hides_everything"),
    pytest.param(({"HIP_VISIBLE_DEVICES": "0"}, 1, (), (_IN_FORCE,)),
                 id = "an_ordinal_that_does_name_a_device_is_still_not_a_blocker"),
    pytest.param(({"HIP_VISIBLE_DEVICES": "3"}, None, (), (_IN_FORCE,)),
                 id = "an_unreadable_device_count_leaves_the_selector_alone"),
    pytest.param(({"ROCR_VISIBLE_DEVICES": _UUID}, 1, ("cannot resolve", "ROCR_VISIBLE_DEVICES"),
                  ("which the groups do not clear",)),
                 id = "a_rocr_selector_naming_a_uuid_is_reported_as_unresolved"),
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0"}, 2, (), ("cannot resolve", "visibility mask")),
                 id = "a_rocr_ordinal_that_names_a_device_is_still_left_alone"),
    # HIP also accepts GPU- UUIDs (rocdevice.cpp), so this cannot resolve them.
    pytest.param(({"HIP_VISIBLE_DEVICES": _UUID}, 1, ("names a device this cannot resolve",),
                  (_IN_FORCE,)),
                 id = "a_uuid_in_the_hip_layer_is_unresolved_rather_than_a_blocker"),
    # ROCr renumbers survivors; HIP ordinals index those, not the physical count.
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"}, 2,
                  ("HIP_VISIBLE_DEVICES='1'", "which the groups do not clear"), ()),
                 id = "a_hip_ordinal_is_judged_against_what_rocr_left"),
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "1"}, 2, (),
                  ("which the groups do not clear",)),
                 id = "the_same_ordinal_inside_what_rocr_left_is_not_a_blocker"),
    pytest.param(({"ROCR_VISIBLE_DEVICES": _UUID, "HIP_VISIBLE_DEVICES": "1"}, 2, (),
                  ("which the groups do not clear",)),
                 id = "an_unresolvable_rocr_entry_leaves_the_hip_ordinal_alone"),
])
def test_which_stacked_masks_are_reported(monkeypatch, linux, case):
    """How the four selectors compose, on a HIP build with every AMD node closed. The groups
    never clear a mask, so anything named here is a second repair the user also needs."""
    env, gpu_count, says, does_not_say = case
    reason = _reason_with_masks(monkeypatch, env, {"hip"}, gpu_count)
    _asserts(reason, says, does_not_say)


def test_the_installer_says_the_same_thing_about_a_missing_render_node(tmp_path):
    """The shell half of the container case. Stubbed on both sides so the arms differ only in
    the answer, since the real helper reads the runner's own /sys and /dev."""
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
    """Whether install.sh routes the two node diagnoses for this wheel index. Lifted rather
    than restated, since the thing under test is which patterns the case actually lists."""
    lines = _install_sh_lines()
    start = next(i for i, line in enumerate(lines) if line.startswith("_amd_node_diag_leaf="))
    esac_at = next(i for i in range(start, len(lines)) if lines[i] == "esac")
    _skip_torch_end = next(i for i in range(esac_at, len(lines)) if lines[i] == "fi")
    end = next(i for i in range(_skip_torch_end, len(lines)) if lines[i] == "esac")
    script = "\n".join(
        [
            f"TORCH_INDEX_URL={index_url!r}",
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            f"export UNSLOTH_LLAMA_CPP_BACKEND={backend or ''!r}",
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_requested_llama_backend"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            *lines[start : end + 1],
            'echo "$_amd_node_diag_route"',
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
    return out.stdout.strip() == "true"


@pytest.mark.parametrize("index_url, routed", [
    ("https://download.pytorch.org/whl/cpu", True),
    ("https://download.pytorch.org/whl/rocm7.0", True),
    ("https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151", True),
    # _has_amd_rocm_gpu is false with any usable NVIDIA GPU: CUDA routes must not get ROCm advice.
    ("https://download.pytorch.org/whl/cu128", False),
    ("https://download.pytorch.org/whl/xpu", False),
])
def test_the_node_diagnoses_run_on_the_routes_the_case_reports(index_url, routed):
    """Which wheel routes reach the two node diagnoses at all."""
    assert _diag_route(index_url) is routed


def test_the_installer_makes_the_same_owner_versus_external_distinction(tmp_path):
    """Read off install.sh, since the two halves must agree: the awk owner branch printed
    owner: for every node this account owns, so a node whose OWNER digit already grants rw got
    the mode advice there too. Lifted rather than restated, so a revert fails here."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")

    node.chmod(0o600)
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()), repairs = None)
    assert "granted read and write by the" in out
    assert "is owned by this account and its owner bits" not in out
    # The installer wraps one sentence over several substep lines, so match a fragment
    # that cannot straddle the break.
    assert "device cgroup or an LSM" in out
    assert "fix the mode" not in out

    node.chmod(0o060)
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()), repairs = None)
    assert "fix the mode" in out
    assert "already grant read and write" not in out


@pytest.mark.parametrize("mode, bucket", [
    pytest.param(0o600, "external", id = "owner_bits_that_already_grant_it_are_external"),
    # POSIX resolves the owner class exclusively once the uid matches.
    pytest.param(0o060, "owned", id = "a_node_this_account_owns_is_not_answered_with_a_group"),
])
def test_how_a_node_this_account_owns_is_classified(monkeypatch, tmp_path, mode, bucket):
    """Which bucket an owned node lands in, by owner bits. Every other bucket must stay empty:
    a group named here is a repair that cannot work."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(mode)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda _path: False)
    assert amd._groups_that_own([str(node)]) == _buckets(**{bucket: [str(node)]})


# fmt: off
@pytest.mark.parametrize("mode, in_the_group, bucket", [
    # The other class is also exclusive: rw other bits mean the mode is not what denies it.
    pytest.param(0o666, False, "external",
                 id = "other_bits_that_already_grant_it_are_external"),
    pytest.param(0o660, False, "joinable",
                 id = "other_bits_that_deny_still_leave_a_group_worth_joining"),
    pytest.param(0o666, True, "already",
                 id = "a_member_is_still_told_the_group_it_already_holds"),
])
# fmt: on
def test_how_a_node_this_account_neither_owns_nor_shares_a_group_with_is_classified(
    monkeypatch, tmp_path, mode, in_the_group, bucket
):
    """Which bucket the third permission class lands in. The two halves of this rule were
    already written for the owner class and for a group this account holds; this is the same
    question for the one class that was left reading its neighbour's bits."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(mode)
    _gid = node.stat().st_gid
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda _path: False)
    # Read before patching: amd.os is os, so os.getuid() in the lambda would recurse.
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    _held = {_gid} if in_the_group else {_gid + 10_000}
    monkeypatch.setattr(amd.os, "getgid", lambda: sorted(_held)[0])
    monkeypatch.setattr(amd.os, "getgroups", lambda: sorted(_held))

    joinable, unnamed, no_group, acl, owned, privileged, already, external = (
        amd._groups_that_own([str(node)])
    )
    _by_name = dict(
        joinable = joinable, unnamed = unnamed, no_group = no_group, acl = acl,
        owned = owned, privileged = privileged, already = already, external = external,
    )
    assert _by_name[bucket], f"{bucket} is empty: {_by_name}"
    for _name, _got in _by_name.items():
        if _name != bucket:
            assert not _got, f"{_name} must stay empty, got {_got}"


def test_the_installer_makes_the_same_other_class_distinction(tmp_path):
    """The shell half of the rule above, read off install.sh so the two cannot drift: the awk
    classifier reached its group digit for a node in the other class too, so mode 0666 printed
    a group to join. A revert fails here as well as in the Python case."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    _not_us = str(os.getuid() + 1)

    node.chmod(0o666)
    out = _install_sh_hint(str(node), self_uid = _not_us, repairs = None)
    assert "device cgroup or an LSM" in out
    assert "usermod" not in out

    node.chmod(0o660)
    out = _install_sh_hint(str(node), self_uid = _not_us, repairs = None)
    assert "device cgroup or an LSM" not in out


def test_a_node_carrying_an_acl_is_not_answered_with_usermod(monkeypatch, tmp_path):
    """acl(5): once an access ACL is present, the group-class bits in st_mode are the ACL MASK
    rather than the owning group's grant, so a node whose mask reads rw can still deny its
    group. Prescribing membership from the mode there is a promise the stat cannot support."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: True)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, _priv, _already, _ext = amd._groups_that_own(
        [str(node)]
    )
    assert acl == [str(node)]
    assert joinable == [] and unnamed == [] and no_group == []


@pytest.mark.parametrize("mode", [
    pytest.param(0o660, id = "the_same_node_without_an_acl_is_still_prescribed_for"),
    pytest.param(0o060, id = "the_same_node_owned_by_someone_else_is_still_a_group"),
])
def test_an_ordinary_node_owned_elsewhere_is_still_prescribed_for(monkeypatch, tmp_path, mode):
    """The control for both suppressions above: the ordinary node, whose mode bits ARE the
    group's grant and whose owner is somebody else, so the owner class does not apply. Without
    it the rule could decline to prescribe anywhere, which removes the repair #10466 needs."""
    node, _group = _a_node_a_membership_would_open(tmp_path, mode = mode)
    _not_my_group = os.getgid() + 1
    monkeypatch.setattr(amd.os, "getgid", lambda: _not_my_group)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _not_the_owner = os.getuid() + 1
    monkeypatch.setattr(amd.os, "getuid", lambda: _not_the_owner)
    joinable, unnamed, no_group, acl, owned, privileged, _already, _ext = amd._groups_that_own(
        [str(node)]
    )
    assert acl == [] and owned == []
    assert joinable or unnamed


def test_the_installer_reports_an_acl_rather_than_prescribing_membership(tmp_path):
    """The shell twin of the same rule: ls marks such a node with a trailing "+", which is the
    marker available without getfacl."""
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
    """The installer's half of the container case: nothing closed, no render node, and an AMD
    GPU in the KFD topology."""
    out = _install_sh_hint("", render_present = False, amd_present = True)
    assert "no AMD render node" in out
    assert "--device /dev/dri" in out


def test_the_installer_stays_quiet_on_a_host_with_no_amd_gpu():
    """Its control, and the same vendor trap: without the KFD topology test this fires on
    every host whose /dev/dri holds another vendor's nodes, or none at all."""
    out = _install_sh_hint("", render_present = False, amd_present = False)
    assert out.strip() == ""


@pytest.mark.parametrize("case", [
    pytest.param((True, None, False), id = "an_ordinary_node_reports_no_acl"),
    pytest.param((False, None, False), id = "a_path_that_cannot_be_read_reports_no_acl"),
    # os.listxattr returns str names for a str path.
    pytest.param((True, ["security.selinux", "system.posix_acl_access"], True),
                 id = "the_acl_probe_matches_the_name_type_listxattr_returns"),
    pytest.param((True, [b"system.posix_acl_access"], True),
                 id = "the_acl_probe_also_reads_bytes_names"),
    pytest.param((True, ["security.selinux", "user.note"], False),
                 id = "another_xattr_is_not_read_as_an_acl"),
])
def test_what_the_acl_probe_answers(monkeypatch, tmp_path, case):
    """The probe the bucket rule above consults."""
    create, xattrs, carries_one = case
    node = tmp_path / "renderD128"
    if create:
        node.write_bytes(b"")
    if xattrs is not None:
        # raising=False: os.listxattr does not exist on macOS/Windows.
        monkeypatch.setattr(amd.os, "listxattr", lambda path: xattrs, raising = False)
    assert amd._has_an_access_acl(str(node)) is carries_one


def _vulkan_reason_with_open_sibling(monkeypatch, openable: set) -> str:
    """The empty-probe reason for a Vulkan build with renderD128 closed."""
    _nodes(
        monkeypatch,
        present = ["/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = openable,
    )
    for var in _GPU_MASK_VARS:
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda _b: frozenset({"vulkan"}))
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b: True))
    return _empty_probe()


_VULKAN_REASON = "the Vulkan probe reported no device"


@pytest.mark.parametrize("case", [
    pytest.param(({"/dev/dri/renderD129"}, (_VULKAN_REASON, "/dev/dri/renderD128"), ()),
                 id = "an_open_sibling_node_keeps_the_vulkan_reason"),
    pytest.param((set(), ("/dev/dri/renderD128",), (_VULKAN_REASON,)),
                 id = "no_open_sibling_still_gives_the_node_hint_alone"),
])
def test_whether_a_vulkan_sibling_node_answers_the_empty_probe(monkeypatch, linux, case):
    """Whether a closed render node is the reason, given what the loader could still open."""
    openable, says, does_not_say = case
    reason = _vulkan_reason_with_open_sibling(monkeypatch, openable)
    _asserts(reason, says, does_not_say)


def _hip_reason_with_nodes(monkeypatch, present: list, openable: set) -> str:
    """The empty-probe reason for a HIP build over a given node layout."""
    _nodes(monkeypatch, present = present, openable = openable)
    for var in _GPU_MASK_VARS:
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda _b: frozenset({"hip"}))
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b: False))
    return _empty_probe()


_SEPARATELY = "Separately, and not why the probe is empty"


@pytest.mark.parametrize("case", [
    pytest.param(({"/dev/kfd", "/dev/dri/renderD129"}, (_SEPARATELY, "/dev/dri/renderD128"), ()),
                 id = "a_closed_sibling_beside_an_open_rocm_path_is_not_the_reason"),
    pytest.param(({"/dev/dri/renderD129"}, ("/dev/kfd",), (_SEPARATELY,)),
                 id = "a_closed_kfd_is_still_the_reason_for_a_hip_build"),
])
def test_whether_a_hip_sibling_node_answers_the_empty_probe(monkeypatch, linux, case):
    """The same sibling question asked of a ROCm build, over one three-node host."""
    openable, says, does_not_say = case
    reason = _hip_reason_with_nodes(
        monkeypatch,
        present = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"],
        openable = openable,
    )
    _asserts(reason, says, does_not_say)


@pytest.mark.parametrize("layout, blocks", [
    pytest.param(_KFD_OPEN, True, id = "a_missing_render_node_blocks_the_runtime"),
    pytest.param(_BOTH_OPEN, False, id = "a_complete_open_mapping_still_does_not_block"),
])
def test_whether_the_node_layout_blocks_the_runtime(monkeypatch, linux, layout, blocks):
    """The predicate both callers gate their hints on."""
    _nodes(monkeypatch, **layout)
    assert amd.amd_closed_nodes_block_the_runtime() is blocks


def test_the_installer_does_not_dangle_the_group_sentence(tmp_path):
    """Without a named group, the installer must not print the dangling Add yourself to the sentence."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o600)
    out = _install_sh_hint(str(node))
    assert "Add yourself to the" not in out
    assert "no" in out and "membership opens it" in out


# fmt: on
def _says(
    out: str,
    contains,
    absent = (),
):
    """Assert what a lifted message says; ``contains = None`` means it says nothing at all."""
    if contains is None:
        assert out.strip() == "", out
        return
    for _text in contains:
        assert _text in out, _text
    for _text in absent:
        assert _text not in out, _text


def _cases(*rows):
    """Parametrize rows written id first, so one case of a family reads as one line."""
    return [pytest.param(*row[1:], id = row[0]) for row in rows]


def test_the_installer_stops_at_the_owner_class_too(tmp_path):
    """The shell half of the owner-precedence item: with the caller as the owner, the
    installer must not print a usermod line for a node no membership opens."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o060)
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()))
    assert "usermod" not in out
    assert "owned by this account" in out


_SEEN = dict(topology = True, amd_smi_sees_it = True)
_NO_TORCH = {**_SEEN, "skip_torch": True, "backend": None}
_BLIND = dict(topology = False, amd_smi_sees_it = False)
_SEEING = dict(topology = False, amd_smi_sees_it = True)
_STACK = "Install the ROCm kernel stack"


def _kernel_stack_hint_text(
    tmp_path,
    *,
    topology: bool,
    kfd_present: bool = False,
    topology_readable: bool = True,
    nvidia: bool = False,
) -> str:
    """Lifts the missing-/dev/kfd guard and body, so the printed repair text is what the test checks."""
    return _install_sh_missing_kfd(
        tmp_path,
        topology = topology,
        kfd_present = kfd_present,
        topology_readable = topology_readable,
        amd_smi_sees_it = False,
        nvidia = nvidia,
    )


# fmt: off
@pytest.mark.parametrize("topology, contains, absent", _cases(
    ("a_topology_means_the_driver_is_loaded", True, ["--device /dev/kfd", "the node itself"],
     [_STACK]),
    ("no_topology_keeps_the_kernel_stack_advice", False, [_STACK], ["--device /dev/kfd"]),
))
# fmt: on
def test_the_installers_missing_kfd_repair_follows_the_topology(tmp_path, topology, contains, absent):
    """A KFD topology with no node means map the device; no topology means the kernel stack is missing."""
    _says(_kernel_stack_hint_text(tmp_path, topology = topology), contains, absent)


def test_a_container_missing_kfd_is_told_to_map_it_rather_than_reinstall(monkeypatch, linux):
    """A missing node is reported only once KFD names a GPU, so the kernel stack is already loaded there."""
    _nodes(monkeypatch, present = ["/dev/dri/renderD128"], openable = {"/dev/dri/renderD128"})
    hint = amd.amd_node_permission_hint()
    assert "--device /dev/kfd" in hint
    assert "kernel stack" not in hint
    assert "the kernel driver is loaded" in hint


def _install_sh_missing_kfd(
    tmp_path,
    *,
    topology: bool,
    kfd_present: bool = False,
    topology_readable: bool = True,
    confirmed_drm: bool = False,
    amd_smi_sees_it: "bool | None" = None,
    rocm_visible: "bool | None" = None,
    skip_torch: bool = False,
    backend: "str | None" = None,
    nvidia: bool = False,
) -> str:
    """Lifts both branches to the closing fi; -e /dev/kfd runs for real on a path this case owns."""
    lines = _install_sh_lines()
    # Anchored on the kernel-stack sentence; the conditions are unstable anchors.
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
            # Real helper lifted, inputs stubbed. State 0 = AMD, 1 = none, 2 = unreadable.
            f"_kfd_topology_amd_state() {{ return {0 if topology else (1 if topology_readable else 2)}; }}",
            f"_a_confirmed_amd_render_node_exists() {{ return {0 if confirmed_drm else 1}; }}",
            _shell_fn(lines, "_amd_silicon_behind_a_missing_kfd"),
            _shell_fn(lines, "_kfd_node_is_amds"),
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
    script = _kfd_node_the_case_owns(script, tmp_path, present = kfd_present)
    env = {**os.environ, "_closed_amd_nodes": ""}
    env.pop("UNSLOTH_LLAMA_CPP_BACKEND", None)
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
    out = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, check = True, env = env
    )
    return out.stdout


_ABSENT_KFD = "/dev/kfd is not present"


# fmt: off
@pytest.mark.parametrize("kwargs, contains, absent", _cases(
    ("amd_smi_does_not_suppress_the_mapping_advice", _SEEN, ["--device /dev/kfd"], [_STACK]),
    ("no_topology_and_a_blind_rocm_keeps_the_advice", _BLIND, [_STACK], []),
    ("no_topology_but_a_seeing_amd_smi_withholds_it", _SEEING, [], [_STACK]),
    ("a_torch_install_still_gets_the_missing_kfd_advice", _SEEN, [_ABSENT_KFD], []),
    ("a_no_torch_rocm_bundle_is_told_its_kfd_is_missing", _NO_TORCH, [_ABSENT_KFD], []),
    ("a_no_torch_vulkan_run_is_not_told_about_it", {**_NO_TORCH, "backend": "vulkan"}, None, []),
    ("a_no_torch_cuda_run_is_not_told_about_it_either", {**_NO_TORCH, "backend": "cuda"}, None, []),
    # --device /dev/dri without /dev/kfd and masked sysfs: fall back to the AMD render node.
    ("an_unreadable_topology_with_a_confirmed_amd_render_node",
     {"topology": False, "amd_smi_sees_it": True, "topology_readable": False,
      "confirmed_drm": True}, ["--device /dev/kfd"], [_STACK]),
    ("an_unreadable_topology_with_no_confirmed_node_stays_silent",
     {"topology": False, "amd_smi_sees_it": True, "topology_readable": False,
      "confirmed_drm": False}, None, ["--device /dev/kfd"]),
    ("a_readable_topology_naming_no_amd_is_not_rescued_by_drm",
     {"topology": False, "amd_smi_sees_it": True, "topology_readable": True,
      "confirmed_drm": True}, None, ["--device /dev/kfd"]),
))
# fmt: on
def test_what_the_installer_says_when_the_kfd_node_is_absent(tmp_path, kwargs, contains, absent):
    """A --no-torch ROCm bundle opens /dev/kfd, so the missing-node report keys on that, not SKIP_TORCH."""
    _says(_install_sh_missing_kfd(tmp_path, **kwargs), contains, absent)


# fmt: off
@pytest.mark.parametrize("assignment, needs", _cases(
    ("the_closed_node_read", '_closed_amd_nodes="$(_amd_nodes_closed_to_this_user',
     ("stat", "awk")),
    ("the_index_leaf_read", "_amd_node_diag_leaf=$(_torch_index_url_leaf", ("tr",)),
    ("the_repair_classifier", "_closed_amd_repairs=$(_amd_node_repairs", ("awk",)),
))
# fmt: on
def test_a_diagnostic_cannot_take_the_install_down_with_it(assignment, needs):
    """Under set -e a missing stat, awk or tr must yield no advice, never an aborted install."""
    line = next(_l for _l in _install_sh_lines() if _l.lstrip().startswith(assignment))
    assert "|| true" in line, f"{assignment} is unguarded under set -e: {line.strip()}"


# fmt: off
@pytest.mark.parametrize("helper, tools", _cases(
    ("the_closed_node_read_survives", "_amd_nodes_closed_to_this_user", ("stat", "awk", "tr")),
    ("the_repair_classifier_survives", "_amd_node_repairs", ("stat", "awk", "tr")),
    ("the_index_leaf_read_survives", "_torch_index_url_leaf", ("stat", "awk", "tr")),
))
# fmt: on
def test_the_guard_actually_holds_when_those_tools_are_missing(helper, tools):
    """Runs the script with every named tool stubbed to exit 127, so the || true guard must really hold."""
    lines = _install_sh_lines()
    _lifted, _seen, _queue = [], set(), [helper]
    while _queue:
        _name = _queue.pop(0)
        if _name in _seen:
            continue
        _seen.add(_name)
        try:
            _body = _shell_fn(lines, _name)
        except StopIteration:
            continue
        _lifted.append(_body)
        _queue += sorted(set(re.findall(r"\b(_[a-z0-9_]+)\b", _body)) - _seen)
    script = "\n".join(
        [
            *[f"{_t}() {{ return 127; }}" for _t in tools],
            *_lifted,
            "set -e",
            f'_answer="$({helper} /dev/kfd || true)"',
            'echo "REACHED THE END"',
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr
    assert "REACHED THE END" in out.stdout


def test_the_installer_denies_the_same_groups_the_runtime_does():
    """The two lists are maintained by hand in two languages, so drift is the failure mode.
    Read install.sh's alternation and compare it to the constant rather than restating
    either: a group added to one half alone fails here."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    text = install_sh.read_text(encoding = "utf-8")
    match = re.search(r"gname ~ /\^\(([a-z|]+)\)\$/", text)
    assert match, "install.sh no longer carries the privileged-group alternation"
    assert set(match.group(1).split("|")) == set(amd._PRIVILEGED_GROUPS)


_TWO_GIDS = "gid:993\ngid:994"
_BOTH_FLAGS = "--group-add 993 --group-add 994"
_RENDER = "/dev/dri/renderD128"


# fmt: off
@pytest.mark.parametrize("closed_nodes, repairs, contains, absent", _cases(
    ("the_container_flag_is_repeated_for_every_gid", f"/dev/kfd\n{_RENDER}", _TWO_GIDS,
     [_BOTH_FLAGS, "GIDs 993,994"], []),
    ("one_gid_still_reads_as_one", "/dev/kfd", "gid:993", ["--group-add 993", "GID 993"],
     ["--group-add 993 --group-add"]),
    ("the_bare_host_repair_also_adds_the_account", _RENDER, _TWO_GIDS,
     ["sudo groupadd -g 993 amdgpu993", "sudo usermod -a -G amdgpu993 ada",
     "sudo groupadd -g 994 amdgpu994", "sudo usermod -a -G amdgpu994 ada", _BOTH_FLAGS,
     "create a group for each"], []),
    ("and_says_to_start_a_new_session", _RENDER, "gid:993",
     ["sudo groupadd -g 993 amdgpu993", "log out and back in"], []),
))
# fmt: on
def test_the_installers_unnamed_gid_repair(closed_nodes, repairs, contains, absent):
    """--group-add takes one value, so a comma-joined pair names no group; each GID needs its own pair."""
    _says(_install_sh_hint(closed_nodes, repairs = repairs), contains, absent)


_HIP_1_SURVIVES = ["HIP_VISIBLE_DEVICES='1'", "which the groups do not clear"]


# fmt: off
@pytest.mark.parametrize("env, contains, absent", _cases(
    ("an_empty_token_keeps_the_prefix_it_already_counted",
     {"ROCR_VISIBLE_DEVICES": "0,", "HIP_VISIBLE_DEVICES": "1"}, _HIP_1_SURVIVES, []),
    ("a_repeated_ordinal_surfaces_one_device",
     {"ROCR_VISIBLE_DEVICES": "0,0", "HIP_VISIBLE_DEVICES": "1"}, _HIP_1_SURVIVES, []),
    ("an_illegal_selector_is_reported_as_a_blocker", {"ROCR_VISIBLE_DEVICES": "garbage"},
     ["ROCR_VISIBLE_DEVICES"], ["cannot resolve"]),
    ("a_uuid_selector_is_still_only_unresolved", {"ROCR_VISIBLE_DEVICES": "GPU-4b2c9f1e0a7d3b58"},
     ["cannot resolve"], []),
))
# fmt: on
def test_how_a_rocr_selector_reads_on_a_two_gpu_host(monkeypatch, linux, env, contains, absent):
    """ROCr stops at an illegal or repeated token, so an illegal first token leaves zero survivors."""
    _says(_reason_with_masks(monkeypatch, env, {"hip"}, gpu_count = 2), contains, absent)


_REPAIR = "Repair installation"
_JOIN_THE_GROUPS = "usermod -a -G render,video ada"


# fmt: off
@pytest.mark.parametrize("hip, contains, absent", _cases(
    ("a_live_cuda_runtime_outranks_a_stale_rocm_intent", None, [_REPAIR], []),
    ("a_live_hip_runtime_still_gets_the_node_hint_alone", "6.4.0", [_JOIN_THE_GROUPS], [_REPAIR]),
))
# fmt: on
def test_an_untagged_label_is_settled_by_the_live_runtime(
    monkeypatch, linux, hip, contains, absent
):
    """An untagged wheel names no vendor, so torch.version.hip and cuda decide, not the label."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _mismatch_vendors(monkeypatch, {"amd"})
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    _torch = types.SimpleNamespace(
        version = types.SimpleNamespace(cuda = "12.8", hip = hip), __version__ = "2.11.0"
    )
    monkeypatch.setitem(sys.modules, "torch", _torch)
    _says(_capability_message("torch_cuda_unavailable", "2.11.0"), contains, absent)


_NO_BACKEND = "no GPU backend can use the AMD card"
_THREE_NODE_PATHS = ["/dev/kfd", "/dev/dri/renderD128", "/dev/dri/renderD129"]


# fmt: off
@pytest.mark.parametrize("present, openable, contains, absent", _cases(
    ("an_open_sibling_stops_it_speaking_for_the_card", _THREE_NODE_PATHS,
     {"/dev/kfd", "/dev/dri/renderD129"},
     ["/dev/dri/renderD128", "the card behind them", "another AMD render node"], [_NO_BACKEND]),
    ("no_open_sibling_still_speaks_for_the_card", _AMD_NODES, {"/dev/kfd"}, [_NO_BACKEND], []),
    ("a_kfd_only_closed_set_still_claims_only_rocm", _AMD_NODES, {"/dev/dri/renderD128"},
     ["ROCm cannot use the AMD card"], []),
))
# fmt: on
def test_how_wide_a_claim_the_closed_set_supports(
    monkeypatch, linux, present, openable, contains, absent
):
    """An open sibling node keeps the other GPU reachable, so the claim must narrow, not vanish."""
    _nodes(monkeypatch, present = present, openable = openable)
    _says(amd.amd_node_permission_hint(), contains, absent)


_GID_993_PAIR = "sudo groupadd -g 993 amdgpu993 && sudo usermod -a -G amdgpu993 ada"
_GID_994_PAIR = "sudo groupadd -g 994 amdgpu994 && sudo usermod -a -G amdgpu994 ada"


# fmt: off
@pytest.mark.parametrize("gids, contains, absent", _cases(
    ("the_hint_also_adds_the_account_once_per_gid", [993, 994],
     ["993, 994", "each of them", _GID_993_PAIR, _GID_994_PAIR, _BOTH_FLAGS], []),
    ("a_single_gid_reads_singular", [993],
     ["GID 993", "create a group for it", "sudo groupadd -g 993"], ["GIDs"]),
    ("and_says_to_start_a_new_session", [993], ["groupadd -g 993 amdgpu993", "log out and back in"],
     []),
    ("and_is_a_command_a_shell_will_run", [993, 994], [_GID_993_PAIR, _GID_994_PAIR], ["<", ">"]),
))
# fmt: on
def test_the_unnamed_gid_hint(monkeypatch, linux, gids, contains, absent):
    """Each GID needs groupadd and usermod, since groupadd does not add the account; <name> is a
    redirect."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda path: False)
    _owning(monkeypatch, unnamed = gids)
    _says(amd.amd_node_permission_hint(), contains, absent)


def _install_sh_stat_format() -> str:
    """The stat format install.sh reads its node records in, lifted rather than restated."""
    line = next(_l for _l in _install_sh_lines() if "stat -c '" in _l and '"$_anr_node"' in _l)
    return line.split("'")[1]


def _install_sh_classify(
    *,
    mode: str = "660",
    group: str = "UNKNOWN",
    gid: str = "0",
    path: str = "/dev/kfd",
    uid: str = "0",
    self_uid: str = "4242",
) -> str:
    """Builds the stat record in install.sh's own field order, so the name field cannot shift the rest."""
    _by_spec = {"%a": mode, "%G": group, "%g": gid, "%n": path, "%u": uid}
    stat_line = "|".join(_by_spec[_spec] for _spec in _install_sh_stat_format().split("|"))
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
    """Lifts the case statement whole, so the shipped globs are tested; only the leaf is stubbed."""
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
    """Lifts the KFD filter and the closed-node message it gates; the filter sits above the other lift."""
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


_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.4"
_CLOSED = "cannot open its device nodes"
_KFD_REPORTED = [_CLOSED, "/dev/kfd"]


# fmt: off
@pytest.mark.parametrize("closed, skip_torch, backend, index, contains, absent", _cases(
    ("a_vulkan_only_no_torch_run_is_not_sent_after_kfd", "/dev/kfd", True, "vulkan", _ROCM_INDEX,
     None, []),
    ("no_torch_alone_still_reports_a_closed_kfd", "/dev/kfd", True, None, _ROCM_INDEX,
     _KFD_REPORTED, []),
    ("a_vulkan_only_run_still_reports_a_closed_render_node", f"/dev/kfd\n{_RENDER}", True, "vulkan",
     _ROCM_INDEX, [_RENDER], ["/dev/kfd"]),
    ("a_torch_install_is_unaffected_by_the_backend_request", "/dev/kfd", False, "vulkan",
     _ROCM_INDEX, ["/dev/kfd"], []),
    ("a_no_torch_cpu_run_is_not_sent_after_a_closed_render_node", _RENDER, True, "cpu", _ROCM_INDEX,
     None, []),
    ("a_no_torch_cpu_run_is_silent_about_the_kfd_node_too", "/dev/kfd", True, "cpu", _ROCM_INDEX,
     None, []),
    ("a_torch_install_asking_for_cpu_llama_still_reports_its_nodes", _RENDER, False, "cpu",
     _ROCM_INDEX, [_CLOSED], []),
    ("a_no_torch_cuda_run_is_not_sent_after_the_kfd_node", "/dev/kfd", True, "cuda", _ROCM_INDEX,
     None, []),
    ("a_no_torch_cuda_run_is_not_sent_after_the_render_node_either", _RENDER, True, "cuda",
     _ROCM_INDEX, None, []),
    ("a_torch_install_asking_for_cuda_llama_still_reports_its_nodes", "/dev/kfd", False, "cuda",
     _ROCM_INDEX, [_CLOSED], []),
    ("a_backend_value_with_internal_whitespace_is_not_a_backend", "/dev/kfd", True, "vul kan",
     _ROCM_INDEX, _KFD_REPORTED, []),
    ("the_same_value_spelled_properly_is_still_a_backend", "/dev/kfd", True, "  VULKAN  ",
     _ROCM_INDEX, None, []),
    ("a_cpu_torch_index_with_a_vulkan_bundle_is_not_sent_after_kfd", "/dev/kfd", False, "vulkan",
     _CPU_INDEX, None, []),
    ("a_cpu_torch_index_alone_still_reports_a_closed_kfd", "/dev/kfd", False, None, _CPU_INDEX,
     _KFD_REPORTED, []),
    ("a_rocm_torch_index_is_unaffected_by_a_vulkan_bundle", "/dev/kfd", False, "vulkan",
     _ROCM_INDEX, _KFD_REPORTED, []),
))
# fmt: on
def test_which_closed_nodes_reach_the_installers_diagnosis(
    closed, skip_torch, backend, index, contains, absent
):
    """Only ROCm opens /dev/kfd, so SKIP_TORCH alone cannot decide it; a ROCm GGUF bundle still does."""
    out = _install_sh_kfd_scope(closed, skip_torch = skip_torch, backend = backend, torch_index = index)
    _says(out, contains, absent)


# fmt: off
@pytest.mark.parametrize("error, hip, contains, absent", _cases(
    ("an_untagged_cuda_wheel_that_will_not_import_is_still_another_vendors", "libcudart.so.13",
     None, [_REPAIR], []),
    ("an_unimportable_rocm_wheel_still_gets_the_node_hint_alone", "libamdhip64.so", "6.4.0",
     [_JOIN_THE_GROUPS], [_REPAIR]),
))
# fmt: on
def test_an_unimportable_torch_is_read_from_its_markers_on_disk(
    monkeypatch, linux, error, hip, contains, absent
):
    """An unimportable torch is read from on-disk markers, and hip is checked before cuda."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _mismatch_vendors(monkeypatch, {"amd"})
    monkeypatch.setattr(hardware, "TORCH_IMPORT_ERROR", ImportError(error))
    monkeypatch.setattr(hardware, "_expected_rocm_flavor_was_chosen", lambda: True)
    monkeypatch.setattr(hardware, "_installed_torch_label_on_disk", lambda: "2.11.0")
    monkeypatch.setattr(
        hardware,
        "_installed_torch_markers_on_disk",
        lambda: {"cuda": "13.0", "hip": hip, "xpu": None},
    )
    _says(_capability_message("torch_cuda_unavailable", "2.11.0"), contains, absent)


# fmt: off
@pytest.mark.parametrize("present, acl, expected", _cases(
    ("the_sentence_names_every_path_it_lists", _AMD_NODES, _AMD_NODES,
     "getfacl /dev/kfd /dev/dri/renderD128"),
    ("one_path_in_one_path_out", ["/dev/kfd"], ["/dev/kfd"], "getfacl /dev/kfd before"),
))
# fmt: on
def test_the_acl_sentence_runs_getfacl_on_every_node_it_lists(
    monkeypatch, linux, present, acl, expected
):
    """The ACL sentence must run getfacl on every node it lists, not only the first."""
    _nodes(monkeypatch, present = present, openable = set())
    _owning(monkeypatch, acl = acl)
    assert expected in amd.amd_node_permission_hint()


_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


# fmt: off
@pytest.mark.parametrize("index_url, kwargs, routes", _cases(
    ("a_mirror_named_for_an_arch_is_not_a_rocm_route",
     "https://download.pytorch.org/whl/gfx-mirror", {}, False),
    ("a_private_rocm_build_is_not_one_either", "https://example.invalid/wheels/rocm7.2-private/",
     {}, False),
    ("the_real_gfx_route_is_still_read_as_one",
     "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151", {}, True),
    ("a_no_torch_vulkan_run_still_diagnoses_its_render_nodes", _CUDA_INDEX,
     dict(skip_torch = True, backend = "vulkan"), True),
    ("a_no_torch_cpu_backend_run_still_does_not", _CUDA_INDEX,
     dict(skip_torch = True, backend = "cpu"), False),
    ("a_cuda_wheel_install_asking_for_cuda_is_off_the_route", _CUDA_INDEX, dict(backend = "cuda"),
     False),
    ("a_cuda_wheel_install_asking_for_cpu_is_too", _CUDA_INDEX, dict(backend = "cpu"), False),
))
# fmt: on
def test_which_runs_take_the_amd_node_diagnosis_route(index_url, kwargs, routes):
    """Classifies the index leaf, since a raw */rocm* glob also matches custom mirror pins."""
    assert _diag_route(index_url, **kwargs) is routes


# fmt: off
@pytest.mark.parametrize("needs_kfd, contains, absent", _cases(
    ("a_vulkan_caller_is_not_told_to_map_the_kfd_node", False, ["--device /dev/dri."],
     ["/dev/kfd"]),
    ("a_rocm_caller_is_still_told_to_map_both", True, ["--device /dev/kfd --device /dev/dri."], []),
))
# fmt: on
def test_the_mapping_advice_names_the_nodes_the_caller_opens(
    monkeypatch, linux, needs_kfd, contains, absent
):
    """Names only the nodes the caller opens, since Vulkan never opens /dev/kfd while HIP opens both."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = {"/dev/kfd"})
    _says(amd.amd_node_permission_hint(needs_kfd = needs_kfd), contains, absent)


# fmt: off
@pytest.mark.parametrize("kwargs, contains, absent", _cases(
    ("a_no_torch_vulkan_installer_names_only_the_render_node",
     dict(skip_torch = True, backend = "vulkan"), ["Docker that is --device /dev/dri."],
     ["--device /dev/kfd"]),
    ("an_ordinary_installer_run_still_names_both", {}, ["--device /dev/kfd --device /dev/dri."],
     []),
))
# fmt: on
def test_the_installers_device_pair_follows_the_run(tmp_path, kwargs, contains, absent):
    """The installer twin: --no-torch with an explicit Vulkan bundle opens no /dev/kfd, so the
    device pair it prints must not name one, while a torch install opens it and the pair stays."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o660)
    _says(_install_sh_hint(str(node), render_present = False, **kwargs), contains, absent)


# fmt: off
@pytest.mark.parametrize("present, openable, vendor_readable, exists, hint_says", _cases(
    ("a_node_whose_vendor_cannot_be_read_is_not_called_absent", _AMD_NODES, set(_AMD_NODES), False,
     True, None),
    ("a_host_with_no_render_node_at_all_still_says_so", ["/dev/kfd"], {"/dev/kfd"}, True, False,
     "No AMD render node"),
))
# fmt: on
def test_an_unreadable_render_node_vendor_reads_as_present(
    monkeypatch, linux, present, openable, vendor_readable, exists, hint_says
):
    """An unknown render-node vendor counts as present; only a glob that finds no node reports it
    missing."""
    _nodes(monkeypatch, present = present, openable = openable, vendor_readable = vendor_readable)
    assert amd._amd_render_node_exists() is exists
    if hint_says is None:
        assert amd.amd_node_permission_hint() is None
    else:
        assert hint_says in amd.amd_node_permission_hint()


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


_PROBE_SENTENCE = "the Vulkan probe reported no device"


def _finding(
    reason: str,
    *,
    credited,
    contains = (),
    absent = (),
):
    """True keeps the Vulkan probe sentence, False replaces it, None means the question is not asked."""
    if credited is True:
        assert reason.startswith(_PROBE_SENTENCE), reason
    elif credited is False:
        assert _PROBE_SENTENCE not in reason, reason
    _says(reason, contains, absent)


# fmt: off
@pytest.mark.parametrize("openable, other_open, backends, credited, contains, absent", _cases(
    ("a_vulkan_probe_keeps_its_finding_when_another_vendor_is_open", set(), True, {"vulkan"}, True,
     ["Separately", "usermod"], []),
    ("no_other_vendor_still_gives_the_hint_alone", set(), False, {"vulkan"}, False, ["usermod"],
     []),
    ("a_rocm_build_is_unaffected_by_another_vendors_node", {_RENDER}, True, {"hip"}, None,
     [_JOIN_THE_GROUPS], ["Separately"]),
))
# fmt: on
def test_whether_another_vendors_open_node_is_a_path_for_this_build(
    monkeypatch, linux, openable, other_open, backends, credited, contains, absent
):
    """Another vendor's open node is a Vulkan path, but never a HIP one, which needs an AMD node."""
    _nodes(monkeypatch, present = _AMD_NODES, openable = openable)
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: other_open)
    _ggml(monkeypatch, backends)
    _finding(_empty_probe(), credited = credited, contains = contains, absent = absent)


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
    assert re.search(r"usermod -a -G \S+ ada", out)
    assert " root" not in out


def test_the_installer_unnamed_gid_repair_is_runnable_too(tmp_path):
    """The shell twin, checked the same way and then actually parsed: `bash -n` on the two
    emitted lines is the assertion that a placeholder would fail. Without the parse this
    would only be testing that a string changed."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993")
    # Rejoin `&& \` continuations: bash 3.2 rejects a trailing backslash at EOF.
    _cmds: "list[str]" = []
    _pending = ""
    for _line in out.splitlines():
        _stripped = _line.strip()
        if not _pending and not _stripped.startswith(("sudo group", "sudo usermod")):
            continue
        _pending = f"{_pending} {_stripped}" if _pending else _stripped
        if _pending.endswith("\\"):
            _pending = _pending[:-1].rstrip()
            continue
        _cmds.append(_pending)
        _pending = ""
    assert not _pending, f"repair ends mid-continuation: {_pending!r}"
    assert _cmds, out
    for _cmd in _cmds:
        assert "<" not in _cmd and ">" not in _cmd
        _parsed = subprocess.run(["bash", "-n", "-c", _cmd], capture_output = True, text = True)
        assert _parsed.returncode == 0, f"{_cmd!r}: {_parsed.stderr}"


# fmt: off
@pytest.mark.parametrize("vendor, verdict", _cases(
    ("a_vendor_the_installer_cannot_read_is_not_absent", None, "PRESENT"),
    ("a_vendor_that_names_another_is_still_absent", "0x10de\n", "ABSENT"),
))
# fmt: on
def test_what_the_installer_makes_of_a_render_node_vendor(tmp_path, vendor, verdict):
    """An unreadable render-node vendor counts as present, but a readable non-AMD one does not."""
    lines = _install_sh_lines()
    fn = _shell_fn(lines, "_amd_render_node_present")
    fn = fn.replace("/dev/dri/renderD*", f"{tmp_path}/dev/dri/renderD*")
    fn = fn.replace("/sys/class/drm/", f"{tmp_path}/sys/class/drm/")
    (tmp_path / "dev/dri").mkdir(parents = True)
    (tmp_path / "dev/dri/renderD128").write_bytes(b"")
    (tmp_path / "sys/class/drm/renderD128/device").mkdir(parents = True)
    if vendor is not None:
        (tmp_path / "sys/class/drm/renderD128/device/vendor").write_text(vendor)
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
    assert out.stdout.strip() == verdict


_HIP = "HIP_VISIBLE_DEVICES"
_SHUT = dict(present = _THREE_NODE_PATHS, openable = {"/dev/kfd", "/dev/dri/renderD129"})
_SHUT_NO_KFD = dict(present = [_RENDER, "/dev/dri/renderD129"], openable = {"/dev/dri/renderD129"})


# fmt: off
@pytest.mark.parametrize("host, env, needs_kfd, blocks", _cases(
    ("a_mask_selecting_the_closed_one_leaves_no_way_in", _SHUT, {_HIP: "0"}, True, True),
    ("no_mask_still_credits_the_open_sibling", _SHUT, {}, True, False),
    ("an_empty_mask_is_not_a_mask", _SHUT, {_HIP: "  "}, True, False),
    ("gpu_device_ordinal_is_a_selector_too", _SHUT, {"GPU_DEVICE_ORDINAL": "1"}, True, True),
    ("a_vulkan_probe_ignores_a_hip_mask", _SHUT_NO_KFD, {_HIP: "0"}, False, False),
    ("a_hip_caller_on_the_same_masked_host_still_blocks", _SHUT, {_HIP: "0"}, True, True),
))
# fmt: on
def test_whether_an_open_sibling_is_a_way_in(monkeypatch, linux, host, env, needs_kfd, blocks):
    """A HIP selector can exclude the GPU behind an open sibling node; an empty variable narrows nothing."""
    _nodes(monkeypatch, **host)
    for _name, _value in env.items():
        monkeypatch.setenv(_name, _value)
    assert amd.amd_closed_nodes_block_the_runtime(needs_kfd = needs_kfd) is blocks


# fmt: off
@pytest.mark.parametrize("gid, names, runner_uid, expected", _cases(
    ("a_docker_owned_node_is_not_answered_with_usermod", 999, {999: "docker"}, None,
     _buckets(privileged = ["docker"])),
    ("an_unnamed_gid_zero_is_the_root_group_not_a_group_to_create", 0, {}, None,
     _buckets(privileged = ["root"])),
    ("an_unnamed_ordinary_gid_is_still_a_group_to_create", 993, {}, None,
     _buckets(unnamed = [993])),
    ("the_derivation_survives_a_root_test_runner", 44, {44: "render"}, 0,
     _buckets(joinable = ["render"])),
))
# fmt: on
def test_which_bucket_the_owning_group_lands_in(
    monkeypatch, linux, gid, names, runner_uid, expected
):
    """gid 0 is root even when unnamed, so it is refused by value, not by the name lookup failing."""
    if runner_uid is not None:
        monkeypatch.setattr(amd.os, "getuid", lambda: runner_uid)
    _stat_nodes(monkeypatch, {"/dev/kfd": (gid, 0o660, 0)}, names)
    assert amd._groups_that_own(["/dev/kfd"]) == expected


# fmt: off
@pytest.mark.parametrize("name, expected", _cases(
    # usermod -G splits on commas, so render,sudo would add sudo.
    ("a_name_carrying_the_usermod_separator", "render,sudo", _buckets(unnamed = [993])),
    # The shell twin splits stat output on '|'.
    ("a_name_carrying_the_field_separator", "render|x", _buckets(unnamed = [993])),
    ("a_name_carrying_a_shell_metacharacter", "render;id", _buckets(unnamed = [993])),
    ("a_name_that_is_only_whitespace", "  ", _buckets(unnamed = [993])),
    ("an_ordinary_group_name", "render", _buckets(joinable = ["render"])),
    ("a_samba_machine_account", "host$", _buckets(joinable = ["host$"])),
    ("a_dotted_distribution_group", "gpu.users-1", _buckets(joinable = ["gpu.users-1"])),
))
# fmt: on
def test_a_group_name_usermod_would_read_as_structure_is_not_prescribed(
    monkeypatch, name, expected
):
    """A name with a comma is not prescribed, since usermod reads it as a list; report the GID instead."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {993: name})
    assert amd._groups_that_own(["/dev/kfd"]) == expected


def test_the_printed_command_never_names_a_group_it_did_not_mean(monkeypatch, linux):
    """End to end through the message, since the bucket above is only half the story: what
    matters is what the user pastes. The GID repair is what this host gets instead."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {993: "render,sudo"})
    _the_real_group_derivation(monkeypatch)
    hint = amd.amd_node_permission_hint()
    assert "render,sudo" not in hint
    assert "-G render" not in hint
    assert "--group-add 993" in hint


# fmt: off
@pytest.mark.parametrize("record, classified", _cases(
    ("the_installer_also_refuses_an_unnamed_gid_zero", {"group": "UNKNOWN", "gid": "0"},
     "privileged:root"),
    ("and_still_names_an_ordinary_unnamed_gid", {"group": "UNKNOWN", "gid": "993"}, "gid:993"),
    ("and_keeps_the_name_of_a_named_root_group", {"group": "wheel", "gid": "0"},
     "privileged:wheel"),
    # usermod splits on commas after the shell, so quoting cannot help; report by GID.
    ("a_name_carrying_the_usermod_separator_is_not_prescribed",
     {"group": "render,sudo", "gid": "993"}, "gid:993"),
    ("a_name_carrying_the_field_separator_is_not_prescribed", {"group": "render|x", "gid": "993"},
     "gid:993"),
    ("an_ordinary_group_name_is_still_prescribed", {"group": "render", "gid": "993"},
     "join:render"),
))
# fmt: on
def test_how_the_installer_classifies_a_nodes_owner(record, classified):
    """gid 0 must be classed privileged even when stat prints UNKNOWN, not as an unnamed GID to create."""
    assert _install_sh_classify(**record) == classified


# fmt: off
@pytest.mark.parametrize("leaf, routes", [
    ("rocm-rel-7.0-private", False), ("rocm-rel-7.0.beta", False), ("rocm-rel-6.1", True),
    ("rocm-rel-6.4", True), ("rocm-rel-6.5.0", True), ("rocm-rel-7.0", True),
    ("rocm-rel-7.2.1", True), ("rocm-rel-7.3.1", True), ("gfx110X-all", True),
    ("gfx120X-all", True), ("gfx103X-all", True), ("gfx1151", True),
])
# fmt: on
def test_which_index_leaves_the_installer_reads_as_a_rocm_route(leaf, routes):
    """rocm-rel leaves take digits and dots only, anchored like rocm[0-9]*; gfx suffixes are the norm."""
    assert _install_sh_diag_route(leaf) == ("true" if routes else "false"), leaf


def test_the_passwd_stub_answers_a_positional_read(monkeypatch):
    """The passwd stub must answer positionally: getpass.getuser() subscripts it when no USER env is set."""
    for _var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(_var, raising = False)
    assert getpass.getuser() == "ada"


_ALSO_IN_FORCE = "visibility mask is also in force"
_CUDA = "CUDA_VISIBLE_DEVICES"


# fmt: off
@pytest.mark.parametrize("env, contains, absent", _cases(
    ("an_unusable_hip_selector_is_a_blocker", {_HIP: "garbage"},
     ["HIP_VISIBLE_DEVICES='garbage'", _ALSO_IN_FORCE], []),
    ("an_empty_hip_mask_defers_to_the_cuda_one_below_it", {_HIP: "", _CUDA: "0"},
     [_JOIN_THE_GROUPS], ["visibility mask"]),
    ("the_cuda_mask_under_it_still_blocks_when_it_hides", {_HIP: "", _CUDA: "-1"},
     ["CUDA_VISIBLE_DEVICES='-1'", _ALSO_IN_FORCE], []),
))
# fmt: on
def test_how_the_hip_selector_chain_is_read(monkeypatch, linux, env, contains, absent):
    """A non-numeric HIP token leaves zero agents, since clr rejects any token that is not its own index."""
    _says(_reason_with_masks(monkeypatch, env, {"hip"}), contains, absent)


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


# fmt: off
@pytest.mark.parametrize("entries, count", _cases(
    ("the_cpu_node_every_topology_carries_is_excluded", {"0": _KFD_CPU_NODE, "1": _KFD_GPU_NODE},
     1),
    ("an_unreadable_entry_makes_the_whole_count_unknown",
     {"0": _KFD_GPU_NODE, "1": None, "2": _KFD_GPU_NODE}, None),
))
# fmt: on
def test_the_gpu_count_reads_the_topology(monkeypatch, entries, count):
    """An unreadable topology entry must give unknown, not a smaller count that flags valid selectors."""
    _kfd_topology(monkeypatch, entries)
    assert amd.amd_kfd_gpu_node_count() == count


# fmt: off
@pytest.mark.parametrize("render_open, contains, absent", _cases(
    ("the_claim_is_scoped_when_a_sibling_node_is_open", True,
     ["another AMD", "render node on this host is open", "usermod -a -G"], []),
    ("and_covers_every_backend_when_none_is", False,
     ["Every backend needs them, ROCm and Vulkan alike."], ["render node on this host is open"]),
))
# fmt: on
def test_how_wide_a_claim_the_installer_makes(tmp_path, render_open, contains, absent):
    """The blanket claim that every backend is blocked holds only when no AMD node is open."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    _says(_install_sh_hint(str(node), render_open = render_open), contains, absent)


_AMD_MANIFEST = "@amd-manifest"
_AMD_MANIFEST_WITHOUT_ITS_LIBRARY = "@amd-manifest-without-its-library"


def _icd_list_value(tmp_path, value):
    """Resolve a parametrized driver-list value, writing a manifest for the two sentinels."""
    if value == _AMD_MANIFEST:
        return _icd_manifest(tmp_path, "radeon_icd.json")
    if value == _AMD_MANIFEST_WITHOUT_ITS_LIBRARY:
        return _icd_manifest(tmp_path, "radeon_icd.json", present = False)
    return value


# fmt: off
@pytest.mark.parametrize("var, value, credited, contains", _cases(
    ("a_replacing_list_naming_amd_alone_stops_the_other_vendor_excusing_it", "VK_DRIVER_FILES",
     _AMD_MANIFEST, False, ["usermod"]),
    ("the_deprecated_spelling_of_that_override_counts_too", "VK_ICD_FILENAMES", _AMD_MANIFEST,
     False, []),
    ("the_additive_variable_leaves_the_other_vendor_credited", "VK_ADD_DRIVER_FILES",
     "/opt/extra/icd.json", True, ["Separately"]),
    ("an_empty_override_is_not_an_override", "VK_DRIVER_FILES", "", True, []),
))
# fmt: on
def test_which_loader_variable_replaces_the_driver_search(
    monkeypatch, linux, tmp_path, var, value, credited, contains
):
    """VK_DRIVER_FILES and VK_ICD_FILENAMES replace the search, VK_ADD_DRIVER_FILES adds; blank is unset."""
    reason = _vulkan_reason_under_icd_list(monkeypatch, _icd_list_value(tmp_path, value), var = var)
    _finding(reason, credited = credited, contains = contains)


_BOTH_VENDORS = os.pathsep.join(
    ["/etc/vulkan/icd.d/radeon_icd.x86_64.json", "/etc/vulkan/icd.d/nvidia_icd.json"]
)


# fmt: off
@pytest.mark.parametrize("value, filters, credited, contains", _cases(
    ("a_list_naming_another_vendor_does_not_suppress_the_vulkan_finding",
     "/etc/vulkan/icd.d/intel_icd.x86_64.json", None, True, ["Separately", "usermod"]),
    ("a_list_carrying_both_vendors_does_not_suppress_either", _BOTH_VENDORS, None, True, []),
    ("an_unclassifiable_entry_keeps_the_unpinned_behaviour", "/opt/vendor/icd.d", None, True, []),
    ("an_amd_manifest_whose_library_is_gone_is_not_a_driver", _AMD_MANIFEST_WITHOUT_ITS_LIBRARY,
     None, True, []),
    ("an_amd_manifest_that_is_not_there_at_all_is_not_a_driver",
     "/nonexistent/icd.d/radeon_icd.x86_64.json", None, True, []),
    ("an_amd_driver_the_loader_filters_out_is_not_a_driver", _AMD_MANIFEST,
     {"VK_LOADER_DRIVERS_DISABLE": "radeon*"}, True, []),
    ("a_valid_amd_manifest_still_suppresses", _AMD_MANIFEST, {}, False, ["usermod"]),
))
# fmt: on
def test_which_driver_lists_are_evidence_that_amd_is_all_the_loader_has(
    monkeypatch, linux, tmp_path, value, filters, credited, contains
):
    """Only a list that is AMD-only is evidence; a manifest whose library is missing loads nothing."""
    reason = _vulkan_reason_under_icd_list(
        monkeypatch, _icd_list_value(tmp_path, value), filters = filters
    )
    _finding(reason, credited = credited, contains = contains)


def test_every_amd_icd_spelling_the_installer_knows_counts_here_too():
    """The two lists are the same convention twice, and a driver named in one but not the
    other would make the installer and this diagnosis disagree about the same host."""
    import install_llama_prebuilt
    assert amd._AMD_VULKAN_ICD_NEEDLES == install_llama_prebuilt._AMD_VULKAN_ICD_NEEDLES


def test_a_blocking_hip_mask_cancels_the_verdict_before_any_node_advice(monkeypatch):
    """A mask hiding every accelerator cancels the chat-only verdict before any node advice is built."""
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
    """On a hybrid host a mask drops the masked AMD card from the vendors, so the node hint is skipped."""
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
    # HIP reads CUDA_VISIBLE_DEVICES too, so emptying it hides both cards.
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
    """Writes an ICD manifest, and its library only if present, since a bare path string proves nothing."""
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
    """search_dirs replaces the loader's search, so the runner's installed ICDs cannot decide the arm."""
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: True)
    for _var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES"):
        monkeypatch.delenv(_var, raising = False)
    if filters is not None or search_dirs is not None:
        for _var in ("VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE"):
            monkeypatch.delenv(_var, raising = False)
    for _var, _value in (filters or {}).items():
        monkeypatch.setenv(_var, _value)
    if search_dirs is not None:
        monkeypatch.setattr(amd, "_vulkan_icd_search_dirs", lambda: list(search_dirs))
    if value is not None:
        monkeypatch.setenv(var, value)
    _ggml(monkeypatch, {"vulkan"})
    return _empty_probe()


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
    """The also argument sets a second selector, because which of two the runtime reads is in question."""
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


# fmt: off
@pytest.mark.parametrize("value, count, var, blocks", _cases(
    ("a_selector_naming_every_gpu_leaves_the_open_sibling_as_evidence", "0,1", 2, _HIP, False),
    ("a_selector_naming_one_of_them_still_discards_the_sibling", "0", 2, _HIP, True),
    ("a_uuid_this_cannot_map_still_discards_it", "GPU-abcdef0123456789", 2, _HIP, True),
    ("an_unreadable_gpu_count_does_too", "0,1", None, _HIP, True),
    ("no_selector_at_all_still_keeps_the_sibling", None, 2, _HIP, False),
    ("a_repeated_rocr_token_ends_the_list_and_narrows", "0,0,1", 2, "ROCR_VISIBLE_DEVICES", True),
    ("the_same_repeat_under_hip_does_not_narrow", "0,0,1", 2, _HIP, False),
))
# fmt: on
def test_whether_a_selector_narrows_the_host(monkeypatch, linux, value, count, var, blocks):
    """A UUID or unreadable GPU count stays narrowing; only a selector naming every GPU does not."""
    assert _blocks_under_selector(monkeypatch, value, count = count, var = var) is blocks


# fmt: off
@pytest.mark.parametrize("kwargs, contains, absent", _cases(
    ("a_hybrid_host_is_not_told_to_repair_the_card_its_bundle_will_not_use",
     dict(torch_index = _CPU_INDEX, nvidia = True), None, []),
    ("the_same_host_with_an_explicit_rocm_request_is_still_told",
     dict(torch_index = _CPU_INDEX, backend = "rocm", nvidia = True), [_CLOSED], []),
    ("an_amd_only_host_on_the_same_route_is_still_told", dict(torch_index = _CPU_INDEX), [_CLOSED],
     []),
    ("a_rocm_torch_index_ignores_the_nvidia_card_entirely", dict(nvidia = True), [_CLOSED], []),
))
# fmt: on
def test_whether_the_installer_repairs_a_card_the_run_will_not_use(
    tmp_path, kwargs, contains, absent
):
    """Auto on a hybrid NVIDIA host opens no AMD node, but an explicit rocm request does."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    _says(_install_sh_hint(str(node), **kwargs), contains, absent)


# fmt: off
@pytest.mark.parametrize("kwargs, contains, absent", _cases(
    ("a_uid_with_no_passwd_entry_is_not_handed_a_usermod", dict(id_user = None, env_user = "root"),
     ["--group-add"], ["usermod -a -G", "root"]),
    ("an_account_the_system_knows_still_gets_the_command", {}, ["sudo usermod -a -G"],
     ["--group-add"]),
    ("and_the_group_sentence_comes_with_it_in_one_piece", {},
     ["Add yourself to the", "usermod -a -G"], []),
    ("the_unnamed_gid_repair_drops_its_groupadd_half_too",
     dict(id_user = None, env_user = "root", repairs = "gid:993"), ["--group-add 993"],
     ["usermod -a -G", "groupadd -g"]),
))
# fmt: on
def test_the_installer_only_names_an_account_the_system_knows(tmp_path, kwargs, contains, absent):
    """Names the account only when passwd knows it, since $USER may say root while nothing runs as it."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    _says(_install_sh_hint(str(node), **kwargs), contains, absent)


def _assert_amd_only_loader(reason: str, amd_only: bool) -> None:
    """AMD-only means another vendor's open node is no path, so the node repair stands."""
    if amd_only:
        assert "the Vulkan probe reported no device" not in reason
        assert "usermod" in reason
    else:
        assert reason.startswith("the Vulkan probe reported no device")


_VK_OVERRIDE_VARS = (
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_ADD_DRIVER_FILES",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
)


def _assert_scope(
    out: str,
    *,
    silent = False,
    present = (),
    absent = (),
):
    """What the KFD-scoped closed-node message did and did not say."""
    if silent:
        assert out.strip() == ""
    for _text in present:
        assert _text in out, _text
    for _text in absent:
        assert _text not in out, _text


_UNDER_CUDA = {"also": {"CUDA_VISIBLE_DEVICES": "0"}}


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("0,1", _UNDER_CUDA, False), id = "a_cuda_selector_under_a_hip_one"),
    pytest.param((None, _UNDER_CUDA, True), id = "a_cuda_selector_on_its_own"),
    pytest.param(("0,1,-1", {}, False), id = "a_list_terminated_by_an_unmappable_token"),
    pytest.param(("0,1,later", {}, False), id = "a_list_terminated_by_a_word"),
    pytest.param(("0,-1", {}, True), id = "a_prefix_that_stops_short"),
    pytest.param(("0,0,1", {"var": "ROCR_VISIBLE_DEVICES"}, True),
                 id = "a_repeat_in_the_rocr_list"),
])
# fmt: on
def test_which_selector_lists_narrow_a_two_gpu_host(monkeypatch, linux, case):
    """clr stops at the first unmappable token, so only the prefix before it counts; 0,-1 narrows to one."""
    value, extra, blocks = case
    assert _blocks_under_selector(monkeypatch, value, count = 2, **extra) is blocks


def _two_vendor_icd_dir(tmp_path) -> "list[str]":
    """One search directory holding a loadable AMD manifest and a loadable foreign one."""
    _amd = _icd_manifest(tmp_path, "radeon_icd.x86_64.json", library = "libamd.so")
    _other = _icd_manifest(tmp_path, "nvidia_icd.json", library = "libnv.so")
    assert _amd and _other
    return [str(tmp_path)]


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({"VK_LOADER_DRIVERS_SELECT": "radeon*"}, True), id = "filtered_to_amd"),
    pytest.param((None, False), id = "the_same_two_drivers_unfiltered"),
])
# fmt: on
def test_a_loader_search_is_read_through_its_filters(monkeypatch, linux, tmp_path, case):
    """VK_LOADER_DRIVERS_SELECT also filters the search; a host with no forced list can be AMD-only."""
    filters, amd_only = case
    reason = _vulkan_reason_under_icd_list(
        monkeypatch, None, search_dirs = _two_vendor_icd_dir(tmp_path), filters = filters
    )
    _assert_amd_only_loader(reason, amd_only)


def test_a_search_that_enumerates_nothing_answers_nothing(monkeypatch, linux, tmp_path):
    """Positive evidence only. An empty search is not "AMD alone", it is a loader this cannot
    read -- and a loader with no driver explains the empty probe by itself, so the closed AMD
    node is not the cause either and must not be suppressed."""
    reason = _vulkan_reason_under_icd_list(
        monkeypatch, None, search_dirs = [str(tmp_path / "nothing-here")]
    )
    assert reason.startswith("the Vulkan probe reported no device")


def test_the_search_dirs_follow_the_xdg_variables(monkeypatch):
    """The loader falls back to its defaults only when a variable is unset, so reading the
    defaults regardless both misses a custom layout's only manifest and counts stale ones the
    loader would never read. The test below holds this list and the installer's together."""
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
    """One loader, one question, so the two lists are one list: a copy that drifts sends the
    two halves of this diagnosis to different drivers."""
    import install_llama_prebuilt

    monkeypatch.setenv("XDG_DATA_DIRS", "/opt/one:/opt/two")
    monkeypatch.setenv("XDG_CONFIG_DIRS", "/opt/conf")
    assert amd._vulkan_icd_search_dirs() == [
        str(directory) for directory in install_llama_prebuilt._vulkan_icd_search_dirs()
    ]


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("vul kan", True, True, ()), id = "a_rejected_value_on_a_hybrid_box"),
    pytest.param(("rocm", True, False, ("cannot open its device nodes", "/dev/kfd")),
                 id = "an_explicit_rocm_request_on_the_same_box"),
    pytest.param(("vul kan", False, False, ("/dev/kfd",)),
                 id = "the_rejected_value_on_an_amd_only_box"),
])
# fmt: on
def test_where_the_automatic_route_reports_a_closed_kfd(case):
    """A rejected backend value normalises to auto, so only real choices may count as a decision."""
    backend, nvidia, silent, present = case
    out = _install_sh_kfd_scope("/dev/kfd", skip_torch = True, backend = backend, nvidia = nvidia)
    _assert_scope(out, silent = silent, present = present)


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
            *(
                _shell_fn(lines, _name)
                for _name in (
                    "_torch_index_url_leaf",
                    "_is_pip_rocm_family_leaf",
                    "_requested_llama_backend",
                    "_torch_opens_amd_nodes",
                    "_auto_bundle_opens_amd_nodes",
                    "_run_may_open_kfd",
                    "_shell_quote",
                    "_run_may_open_a_gpu_node",
                )
            ),
            "for _i in 1 2 3 4; do _run_may_open_kfd; _run_may_open_a_gpu_node; done",
            'echo "$_probe_calls"',
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
    return int(out.stdout.strip())


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((None, 1), id = "the_automatic_route_probes_once"),
    pytest.param(("rocm", 0), id = "an_explicit_backend_never_probes_at_all"),
])
# fmt: on
def test_how_often_the_nvidia_probe_runs(case):
    """The NVIDIA probe runs nvidia-smi on every call; nothing it reads changes in a run, so memoise it."""
    backend, calls = case
    assert _nvidia_probe_calls(backend) == calls


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("rocm", True), id = "an_explicit_rocm_bundle"),
    pytest.param(("vulkan", True), id = "an_explicit_vulkan_bundle"),
    pytest.param(("cpu", False), id = "a_cpu_bundle"),
    pytest.param((None, False), id = "no_request_at_all"),
])
# fmt: on
def test_whether_an_explicit_bundle_routes_the_node_diagnoses(case):
    """A CUDA-pinned index must not hide a ROCm bundle request, which opens the nodes the diagnoses name."""
    backend, routed = case
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = backend) is routed


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((True, False), id = "a_hybrid_host_rocm_can_see"),
    pytest.param((False, True), id = "a_hybrid_host_rocm_cannot_see"),
])
# fmt: on
def test_the_kernel_stack_hint_on_a_hybrid_rocm_host(tmp_path, case):
    """No NVIDIA veto here: the run already opens AMD nodes, so the rocminfo probe must still answer."""
    rocm_visible, says_rocm_cannot_see_it = case
    out = _install_sh_missing_kfd(
        tmp_path, topology = False, nvidia = True, backend = "rocm", rocm_visible = rocm_visible
    )
    assert ("ROCm cannot see it" in out) is says_rocm_cannot_see_it


def _search_plus_added_driver(tmp_path, name, library):
    """An AMD-only search directory, and one more manifest registered from outside it."""
    search = tmp_path / "icd.d"
    search.mkdir()
    amd_manifest = _icd_manifest(search, "radeon_icd.x86_64.json", library = "libamd.so")
    elsewhere = tmp_path / "vendor"
    elsewhere.mkdir()
    return str(search), amd_manifest, _icd_manifest(elsewhere, name, library = library)


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("nvidia_icd.json", "libnv.so", False, False), id = "a_foreign_driver_added"),
    pytest.param(("nvidia_icd.json", "libnv.so", True, True),
                 id = "the_same_one_under_a_forced_list"),
    pytest.param(("amdvlk64.json", "libamdvlk.so", False, True), id = "an_amd_driver_added"),
])
# fmt: on
def test_how_an_added_driver_list_is_read(monkeypatch, linux, tmp_path, case):
    """VK_ADD_DRIVER_FILES is read too, but ignored once a forced list is set; an AMD add stays AMD-only."""
    name, library, forced, amd_only = case
    search, amd_manifest, added = _search_plus_added_driver(tmp_path, name, library)
    reason = _vulkan_reason_under_icd_list(
        monkeypatch,
        amd_manifest if forced else None,
        search_dirs = [search],
        filters = {"VK_ADD_DRIVER_FILES": added},
    )
    _assert_amd_only_loader(reason, amd_only)


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({"vendor_readable": False}, None, _AMD_NODES, "/dev/dri/renderD128"),
                 id = "a_hidden_vendor_over_an_amd_topology"),
    pytest.param(({"vendor_readable": False, "amd_owned": False}, None, [], None),
                 id = "a_hidden_vendor_over_a_foreign_topology"),
    pytest.param(({}, "0x10de", ["/dev/kfd"], None), id = "a_readable_foreign_vendor"),
    pytest.param(({"topology": None}, None, _AMD_NODES, None),
                 id = "a_hidden_topology_drm_confirms"),
    pytest.param(({"amd_owned": False, "topology": False}, None, [], None),
                 id = "a_readable_topology_naming_no_amd_gpu"),
    pytest.param(({"vendor_readable": False, "topology": None}, None, [], None),
                 id = "a_hidden_topology_drm_cannot_confirm"),
])
# fmt: on
def test_which_shut_nodes_are_credited_to_amd(monkeypatch, linux, case):
    """Unknown vendor still counts as AMD given KFD evidence; the DRM fallback needs a confirmed one."""
    node_kwargs, vendor, closed, hint = case
    _nodes(monkeypatch, present = _AMD_NODES, openable = set(), **node_kwargs)
    if vendor is not None:
        monkeypatch.setattr(amd, "_render_node_vendor", lambda path: vendor)
    assert amd.amd_nodes_closed_to_this_user() == closed
    if hint is not None:
        assert hint in amd.amd_node_permission_hint()


_CUDA_WHEEL = "https://download.pytorch.org/whl/cu128"
_RADEON_WHEEL = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("/dev/kfd", "vulkan", _CUDA_WHEEL, True, True, (), ()),
                 id = "a_cuda_wheel_beside_a_vulkan_bundle"),
    pytest.param(("/dev/kfd\n/dev/dri/renderD128", "vulkan", _CUDA_WHEEL, True, False,
                  ("/dev/dri/renderD128",), ("/dev/kfd",)),
                 id = "the_same_pair_with_a_closed_render_node"),
    pytest.param(("/dev/kfd", "rocm", _CUDA_WHEEL, True, False, ("/dev/kfd",), ()),
                 id = "a_cuda_wheel_asking_for_the_rocm_bundle"),
    pytest.param(("/dev/kfd", "vulkan", _RADEON_WHEEL, False, False, ("/dev/kfd",), ()),
                 id = "a_radeon_repo_wheel_under_a_vulkan_bundle"),
])
# fmt: on
def test_which_nodes_reach_the_kfd_scope(case):
    """Only a ROCm wheel opens /dev/kfd, so a CUDA wheel must not pull it into the KFD scope."""
    nodes, backend, wheel, nvidia, silent, present, absent = case
    out = _install_sh_kfd_scope(
        nodes, skip_torch = False, backend = backend, torch_index = wheel, nvidia = nvidia
    )
    _assert_scope(out, silent = silent, present = present, absent = absent)


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(([("radeon_icd.i686.json", None)], False), id = "a_32_bit_amd_registration"),
    pytest.param(([("radeon_icd.i686.json", None), ("radeon_icd.x86_64.json", None)], True),
                 id = "the_multilib_pair"),
    pytest.param(([("nvidia_icd.i686.json", "libGLX_nvidia32.so"),
                   ("radeon_icd.x86_64.json", None)], True),
                 id = "a_32_bit_foreign_manifest_beside_an_amd_one"),
    pytest.param(([("nvidia_icd.json", "libGLX_nvidia.so"),
                   ("radeon_icd.x86_64.json", None)], False),
                 id = "a_64_bit_foreign_manifest_beside_an_amd_one"),
])
# fmt: on
def test_which_registrations_a_64_bit_binary_can_load(monkeypatch, linux, tmp_path, case):
    """A 32-bit manifest is no evidence of a rival vendor, since a 64-bit binary cannot load it."""
    manifests, amd_only = case
    paths = [
        _icd_manifest(tmp_path, _name, **({"library": _library} if _library else {}))
        for _name, _library in manifests
    ]
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join(paths))
    _assert_amd_only_loader(reason, amd_only)


def test_the_32_bit_rule_matches_the_installers(tmp_path):
    """Shares the 32-bit AMD ICD needles with the installer, so the two routes cannot disagree."""
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
    env = None,
    sibling_open = False,
):
    """sibling_open demotes the node repair to a second finding when another vendor's node is open."""
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: sibling_open)
    for _var in _VK_OVERRIDE_VARS:
        monkeypatch.delenv(_var, raising = False)
    if search_dirs is not None:
        monkeypatch.setattr(amd, "_vulkan_icd_search_dirs", lambda: list(search_dirs))
    if value is not None:
        monkeypatch.setenv("VK_DRIVER_FILES", value)
    for _name, _value in (env or {}).items():
        monkeypatch.setenv(_name, _value)
    _ggml(monkeypatch, {"vulkan"})
    return _empty_probe()


def test_the_no_driver_diagnosis_stays_primary_when_a_sibling_node_is_open(
    monkeypatch, linux, tmp_path
):
    """With a sibling node open the no-driver diagnosis stays primary, since it alone is sufficient."""
    _icd_manifest(tmp_path, "radeon_icd.json", present = False)
    reason = (
        _vulkan_node_hint_under_icd_list(
            monkeypatch, None, search_dirs = [str(tmp_path)], sibling_open = True
        )
        or ""
    )
    assert "no driver it can load" in reason
    assert "reinstall the Vulkan driver" in reason
    _demoted = reason.index("Separately, and not why the probe is empty")
    assert reason.index("no driver it can load") < _demoted
    assert "loader also has no driver" not in reason


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((lambda tmp: {"value": _icd_manifest(tmp, "radeon_icd.json", present = False)},
                  True),
                 id = "a_registration_with_no_library_behind_it"),
    pytest.param((lambda tmp: {"value": None, "search_dirs": [str(tmp / "empty")]}, False),
                 id = "a_loader_that_could_not_be_enumerated"),
    pytest.param((lambda tmp: {"value": _icd_manifest(tmp, "radeon_icd.x86_64.json")}, False),
                 id = "a_loadable_driver"),
])
# fmt: on
def test_when_the_no_driver_sentence_joins_the_node_repair(monkeypatch, linux, tmp_path, case):
    """Finding no manifests means the config is unreadable, not empty, so no driver sentence is printed."""
    configuration, says_no_driver = case
    _kwargs = configuration(tmp_path)
    reason = _vulkan_node_hint_under_icd_list(monkeypatch, _kwargs.pop("value"), **_kwargs)
    assert "usermod" in reason
    assert ("no driver it can load" in reason) is says_no_driver


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((39, {39: "render"}, True, {"already": ["render"], "joinable": []}),
                 id = "a_named_group_this_account_holds"),
    pytest.param((39, {39: "render"}, False, {"joinable": ["render"], "already": []}),
                 id = "a_named_group_this_account_is_outside"),
    pytest.param((993, {}, True, {"already": ["993"], "unnamed": []}),
                 id = "an_unnamed_gid_this_account_holds"),
    pytest.param((993, {}, False, {"unnamed": [993], "already": []}),
                 id = "an_unnamed_gid_this_account_lacks"),
])
# fmt: on
def test_which_bucket_an_owning_group_lands_in(monkeypatch, linux, case):
    """A group already held is not the denial, since os.access already failed; a cgroup or LSM may be."""
    gid, names, held, expected = case
    _stat_nodes(monkeypatch, {"/dev/kfd": (gid, 0o660, 0)}, names)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [gid] if held else [])
    monkeypatch.setattr(amd.os, "getgid", lambda: gid if held else 1)
    _got = dict(zip(_BUCKETS, amd._groups_that_own(["/dev/kfd"])))
    for _bucket, _value in expected.items():
        assert _got[_bucket] == _value, _bucket


def test_the_hint_for_a_group_already_held_names_the_cgroup_instead(monkeypatch, linux):
    """The sentence a user actually reads, since the buckets above only decide it: no usermod,
    and a statement of what is left to look at."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _owning(monkeypatch, already = ["render"])
    hint = amd.amd_node_permission_hint()
    assert "usermod -a -G" not in hint
    assert "already in the render group" in hint
    assert "cgroup" in hint


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((lambda gid: str(gid), True), id = "the_owning_gid_itself"),
    pytest.param((lambda gid: str(gid + 1), False), id = "a_neighbouring_gid"),
    pytest.param((lambda gid: f"{gid}7 {gid}9", False), id = "two_gids_it_is_a_prefix_of"),
])
# fmt: on
def test_whether_the_installer_prescribes_the_owning_group(tmp_path, case):
    """Matches id -G gids with padding, since 100 would otherwise match an account that only holds 1001."""
    gids_for, already = case
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), self_gids = gids_for(os.stat(node).st_gid))
    assert ("already in the" in out) is already
    assert ("sudo usermod -a -G" in out) is (not already)
    if already:
        assert "cgroup" in out


def _bare_soname_manifest(
    tmp_path,
    name,
    soname = "libvulkan_radeon.so",
):
    """Writes a manifest that names its library by bare soname, which _icd_manifest cannot produce."""
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


_RADEON_SONAME = "libvulkan_radeon.so"
_NVIDIA_SONAME = "libGLX_nvidia.so.0"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("radeon_icd.json", _RADEON_SONAME, False, frozenset({"libc.so.6"}), False),
                 id = "a_soname_nothing_can_resolve"),
    pytest.param(("radeon_icd.json", _RADEON_SONAME, True, frozenset(), True),
                 id = "a_soname_on_the_search_path"),
    pytest.param(("nvidia_icd.json", _NVIDIA_SONAME, False, frozenset({_NVIDIA_SONAME}), True),
                 id = "a_soname_only_the_loader_cache_knows"),
    pytest.param(("nvidia_icd.json", _NVIDIA_SONAME, False, None, True),
                 id = "a_soname_under_a_cache_that_cannot_be_read"),
])
# fmt: on
def test_when_a_bare_soname_registration_is_a_driver(monkeypatch, linux, tmp_path, case):
    """A bare soname counts only if ld.so finds it; a missing ldconfig cache means unknown, not stale."""
    name, soname, on_disk, cache, usable = case
    lib = tmp_path / "lib"
    lib.mkdir()
    if on_disk:
        (lib / soname).write_bytes(b"")
    manifest = _bare_soname_manifest(tmp_path, name, soname)
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(lib)])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: cache)
    assert amd._icd_manifest_is_usable(manifest) is usable


def test_the_loader_cache_reader_says_none_rather_than_empty_when_ldconfig_is_gone(
    monkeypatch, linux
):
    """The distinction the arm above rests on, at its source: no ldconfig has to answer None,
    because an empty set would read as "no library is installed" and call every bare
    registration on the host stale."""
    monkeypatch.setattr(amd, "_ld_cache_read", False)
    monkeypatch.setattr(amd, "_ld_cache_sonames_cached", None)
    monkeypatch.setattr(amd.shutil, "which", lambda _name: None)
    monkeypatch.setattr(amd.os.path, "exists", lambda _p: False)
    assert amd._ld_cache_sonames() is None


def test_a_stale_bare_registration_reaches_the_driver_sentence(monkeypatch, linux, tmp_path):
    """What the classification is for: with the only registration unresolvable the loader has
    no driver at all, so the node repair alone would leave the probe empty."""
    manifest = _bare_soname_manifest(tmp_path, "radeon_icd.json")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    monkeypatch.setattr(amd, "_ld_cache_sonames", lambda: frozenset({"libc.so.6"}))
    reason = _vulkan_node_hint_under_icd_list(monkeypatch, manifest)
    assert "usermod" in reason
    assert "no driver it can load" in reason


def test_the_mask_fixture_isolates_every_selector_the_rule_reads():
    """Clears each selector the rule reads, taken from its source, so runner exports cannot narrow."""
    _source = inspect.getsource(amd._a_per_gpu_mask_narrows_the_runtime)
    _read = set(re.findall(r'"([A-Z_]+(?:VISIBLE_DEVICES|DEVICE_ORDINAL))"', _source))
    assert _read, "the rule named no selector, so this test proves nothing"
    assert _read <= set(_GPU_MASK_VARS), _read - set(_GPU_MASK_VARS)


def test_the_installer_chains_the_groupadd_pair(tmp_path):
    """Chains groupadd and usermod with &&, so a name clash cannot let usermod act on the wrong group."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "gid:993")
    _lines = out.splitlines()
    _at = next(i for i, l in enumerate(_lines) if "groupadd" in l)
    # && plus a continuation, so the pair pastes as one command across the two lines.
    assert _lines[_at].rstrip().endswith("&& \\"), _lines[_at]
    assert "993 amdgpu993" in _lines[_at]
    assert "usermod -a -G amdgpu993" in _lines[_at + 1]


def test_the_python_half_chains_the_groupadd_pair_too(monkeypatch, linux):
    """Its twin, which already chained: asserted so that the two halves cannot drift apart the
    way they just did."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _owning(monkeypatch, unnamed = [993])
    hint = amd.amd_node_permission_hint()
    assert "groupadd -g 993 amdgpu993 && sudo usermod -a -G amdgpu993" in hint


def test_the_named_group_repair_is_still_one_command(tmp_path):
    """The control: a node whose owning group HAS a name needs no groupadd, so the repair is a
    single usermod and must not have grown a chain."""
    out = _install_sh_hint("/dev/dri/renderD128", repairs = "join:render")
    assert "groupadd" not in out
    _line = next(l for l in out.splitlines() if "usermod" in l)
    assert "&&" not in _line


def _a_closed_node_file(tmp_path, name):
    """A file standing in for a device node this account cannot open."""
    node = tmp_path / name
    node.write_bytes(b"")
    node.chmod(0o000)
    return node


def _install_sh_closed_nodes(nodes, *, vendors, topology: bool) -> "list[str]":
    """Runs the closed-node enumeration over real files, stubbing only the /dev and /sys seams it reads."""
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
            '_amd_render_node_vendor() { case "$1" in ' + _vendor_cases + " *) return 1 ;; esac; }",
            _shell_fn(lines, "_amd_nodes_closed_to_this_user"),
            "_amd_nodes_closed_to_this_user",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr
    return [line for line in out.stdout.splitlines() if line.strip()]


# fmt: off
@pytest.mark.skipif(_running_as_root(), reason = "root can open a mode 000 node, nothing is shut")
@pytest.mark.parametrize("case", [
    pytest.param((None, True, True), id = "a_hidden_vendor_over_an_amd_topology"),
    pytest.param((None, False, False), id = "a_hidden_vendor_over_no_amd_topology"),
    pytest.param(("0x10de", True, False), id = "a_readable_foreign_vendor"),
    pytest.param(("0x1002", False, True), id = "a_readable_amd_vendor"),
])
# fmt: on
def test_which_nodes_the_installer_enumerates_as_closed(tmp_path, case):
    """The installer half of the closed-node walk, over a real mode 000 file.

    A container can map /dev/dri and hide the sysfs attribute naming its vendor, the shape
    #10466 is about; dropping the node left the installer printing no render-node repair at
    all, while _amd_render_node_present reads the same unknown as PRESENT and withdraws the
    missing-node sentence, so that host got no diagnosis. The vendor guard exists because
    render nodes are root:render for EVERY vendor, so an NVIDIA-only box has the same closed
    list and none of the problem. A readable non-AMD vendor stays excluded however the
    topology reads, since positive evidence beats the fallback and a mixed box must not be
    told to chgrp its NVIDIA node; a readable AMD one is kept, which is what the enumeration
    is for.

    Fails before the fix, which required a readable vendor. The Python half has answered this
    since round twenty-five; this is the installer catching up."""
    vendor, topology, kept = case
    node = _a_closed_node_file(tmp_path, "renderD128")
    closed = _install_sh_closed_nodes(
        [node], vendors = {node: vendor} if vendor else {}, topology = topology
    )
    assert closed == ([str(node)] if kept else [])


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


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("radeon_icd.json", {"library": "a.so", "arch": "32"}, True),
                 id = "a_declared_32"),
    pytest.param(("radeon_icd.i686.json", {"library": "b.so", "arch": "64"}, False),
                 id = "a_declared_64_under_an_i686_name"),
    pytest.param(("radeon_icd.json", {"library": "c.so", "elf": 32}, True), id = "a_32_bit_elf"),
    pytest.param(("radeon_icd.json", {"library": "d.so", "elf": 64}, False), id = "a_64_bit_elf"),
    pytest.param(("radeon_icd.i686.json", {}, True), id = "the_filename_alone"),
])
# fmt: on
def test_what_decides_an_icd_manifests_bitness(monkeypatch, linux, tmp_path, case):
    """Bitness comes from library_arch, then ELF EI_CLASS byte 4, and the filename only as a last resort."""
    name, manifest_kwargs, is_32 = case
    manifest = _icd_manifest_with(tmp_path, name, **manifest_kwargs)
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [])
    assert amd._an_icd_is_32_bit(manifest) is is_32


def test_a_declared_32_bit_manifest_reaches_the_empty_probe_reason(monkeypatch, linux, tmp_path):
    """What the classification decides: with the only other registration 32-bit, that vendor's
    open render node is not a path this binary has, so the closed AMD node stays the answer
    rather than being demoted."""
    _theirs = _icd_manifest_with(tmp_path, "nvidia_icd.json", library = "libGLX_nvidia.so", arch = "32")
    _ours = _icd_manifest(tmp_path, "radeon_icd.x86_64.json")
    _assert_amd_only_loader(
        _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join([_theirs, _ours])), True
    )


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("DOMAIN\\ada", "render", "usermod -a -G render 'DOMAIN\\ada'", False),
                 id = "an_account_name_the_shell_would_mangle"),
    pytest.param(("ada", "gpu users", "usermod -a -G 'gpu users' ada", False),
                 id = "a_group_name_that_carries_a_space"),
    pytest.param(("ada", "render", "usermod -a -G render ada", True), id = "ordinary_names"),
])
# fmt: on
def test_the_pasted_command_quotes_the_names_that_need_it(monkeypatch, linux, case):
    """These are commands to paste. NSS names are not identifiers -- winbind hands back
    DOMAIN\\user -- so an unquoted one is de-escaped by the shell and usermod then names an
    account that does not exist, leaving the node shut; a group name carrying a space is the
    other half of the same command, and the one that would silently split into two arguments
    rather than failing outright. shlex.quote rather than unconditional quoting, so the
    command a user sees on a normal host does not grow quotes it does not need.

    Fails before the fix, which interpolated the name raw."""
    account, group, command, unquoted = case
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    monkeypatch.setattr(amd, "_repair_account", lambda: account)
    _owning(monkeypatch, joinable = [group])
    hint = amd.amd_node_permission_hint()
    assert command in hint
    if unquoted:
        assert "'" not in hint


def test_the_installer_quotes_the_account_the_same_way(tmp_path):
    """The shell twin: the installer prints the same command from the same kind of name, so a
    host whose account carries a backslash must not get an unquoted one there either."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    out = _install_sh_hint(str(node), id_user = "DOMAIN\\ada", env_user = "DOMAIN\\ada")
    assert "'DOMAIN\\ada'" in out


def test_the_two_quoting_rules_are_the_same_rule(tmp_path):
    """Both halves print the same command, so a value one quotes and the other does not is a
    host where the two disagree about what the user should paste. Run against the real shell
    function rather than a restatement of it."""
    lines = _install_sh_lines()
    helper = _shell_fn(lines, "_shell_quote")
    _values = ("ada", "render", "DOMAIN\\ada", "gpu users", "ada;reboot", "a'b", "user@host", "")
    for value in _values:
        out = subprocess.run(
            ["bash", "-c", helper + '\n_shell_quote "$1"', "_", value],
            capture_output = True,
            text = True,
        )
        assert out.returncode == 0, out.stderr
        for _quoted in (out.stdout, shlex.quote(value)):
            _back = subprocess.run(
                ["bash", "-c", "printf %s " + _quoted], capture_output = True, text = True
            )
            assert _back.returncode == 0, _back.stderr
            assert _back.stdout == value, (value, _quoted, _back.stdout)
        if "'" not in value:
            assert out.stdout == shlex.quote(value), (value, out.stdout)


# Each name here must be lifted by every tests/sh harness that lifts the probe.
_ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT = frozenset(
    {
        "_ensure_rocm_probe_env",
        "_has_usable_nvidia_gpu",
        # Unlifted, the undefined command reads as 'not requested' and the test passes vacuously.
        "_rocm_torch_explicitly_requested",
    }
)


def _functions_called_by(lines: "list[str]", name: str) -> set:
    """Which other functions defined in ``lines`` the named one calls."""
    defined = {line.split("(")[0] for line in lines if re.match(r"^_?[A-Za-z0-9_]+\(\) \{", line)}
    body = _shell_fn(lines, name).splitlines()[1:]
    return {
        _other
        for _other in defined
        if _other != name
        and any(re.search(rf"(^|[\s;&|(]){re.escape(_other)}($|[\s;&|)])", l) for l in body)
    }


def test_the_rocm_probe_calls_nothing_the_shell_harnesses_do_not_lift():
    """tests/sh lifts probes by name, so _has_amd_rocm_gpu may call only helpers those harnesses lift."""
    called = _functions_called_by(_install_sh_lines(), "_has_amd_rocm_gpu")
    assert called <= _ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT, called


def test_the_harnesses_really_lift_every_name_the_allowlist_claims():
    """Each tests/sh harness that lifts _has_amd_rocm_gpu must also lift every allowlisted callee."""
    sh_dir = Path(__file__).resolve().parents[3] / "tests" / "sh"
    harnesses = [
        path
        for path in sorted(sh_dir.glob("*.sh"))
        if "_has_amd_rocm_gpu" in path.read_text(encoding = "utf-8")
    ]
    # Guard the guard: a glob that matched nothing would pass this vacuously.
    assert len(harnesses) >= 5, [p.name for p in harnesses]
    for path in harnesses:
        source = path.read_text(encoding = "utf-8")
        for callee in sorted(_ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT):
            assert callee in source, f"{path.name} lifts _has_amd_rocm_gpu but not {callee}"


def test_that_check_sees_a_helper_the_harnesses_would_not_have():
    """The control. Without it the test above could be passing because the scan matches
    nothing at all, which is what a name-based scan usually does when it is wrong."""
    lines = (
        "_ensure_rocm_probe_env() {\n    :\n}\n"
        "_amd_rocm_gpu_visible() {\n    return 1\n}\n"
        "_has_amd_rocm_gpu() {\n    _ensure_rocm_probe_env\n    _amd_rocm_gpu_visible\n}"
    ).splitlines()
    assert _functions_called_by(lines, "_has_amd_rocm_gpu") == {
        "_ensure_rocm_probe_env",
        "_amd_rocm_gpu_visible",
    }


def _multilib_soname(tmp_path, *, bitnesses):
    """Builds a 32-bit and 64-bit soname copy in Debian multilib order, since that order is under test."""
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


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({32, 64}, 1, False), id = "both_bitnesses_installed"),
    pytest.param(({32}, 0, True), id = "only_the_wrong_bitness_installed"),
])
# fmt: on
def test_which_copy_of_a_multilib_soname_answers(monkeypatch, tmp_path, case):
    """A multilib soname must resolve to the copy matching the process, not the first directory searched."""
    bitnesses, chosen, is_32 = case
    manifest, dirs = _multilib_soname(tmp_path, bitnesses = bitnesses)
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: dirs)
    assert amd._icd_library_path(manifest).startswith(dirs[chosen])
    assert amd._an_icd_is_32_bit(manifest) is is_32


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
    """A manifest missing file_format_version or api_version is refused by the loader, so it is no
    driver."""
    assert amd._icd_manifest_is_usable(_manifest_missing(tmp_path, field, drop = field)) is False


def test_a_version_the_loader_does_not_recognise_is_still_a_driver(tmp_path):
    """The control, and the line between the two. An unknown file_format_version major is the
    one thing here the loader does NOT skip for: it logs "may cause errors" and carries on, so
    refusing it would drop a driver that loads. Only absence decides."""
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


def _ldconfig_answering(monkeypatch, *, returncode: int, stdout: str) -> None:
    """A host whose only ldconfig answers exactly this, with the cache read state reset."""
    monkeypatch.setattr(amd, "_ld_cache_read", False)
    monkeypatch.setattr(amd, "_ld_cache_sonames_cached", None)
    monkeypatch.setattr(amd.shutil, "which", lambda _name: "/sbin/ldconfig")
    monkeypatch.setattr(amd.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(
        amd.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], returncode, stdout, ""),
    )


_EMPTY_CACHE = "0 libs found in cache `/etc/ld.so.cache'\n"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((0, _EMPTY_CACHE, frozenset()), id = "a_readable_but_empty_cache"),
    pytest.param((1, "", None), id = "an_ldconfig_that_fails"),
])
# fmt: on
def test_what_an_ldconfig_answer_means(monkeypatch, linux, case):
    """A non-zero ldconfig exit means the cache is absent, which must stay None, not an empty set."""
    returncode, stdout, sonames = case
    _ldconfig_answering(monkeypatch, returncode = returncode, stdout = stdout)
    if sonames is None:
        assert amd._ld_cache_sonames() is None
    else:
        assert amd._ld_cache_sonames() == sonames


def test_a_bare_soname_is_stale_when_the_readable_cache_does_not_carry_it(
    monkeypatch, linux, tmp_path
):
    """A soname missing from a readable ld cache is a stale registration, not an unknown one."""
    manifest = _bare_soname_manifest(tmp_path, "radeon_icd.json", "libvulkan_radeon.so")
    monkeypatch.setattr(amd, "_dynamic_loader_search_dirs", lambda: [str(tmp_path / "lib")])
    _ldconfig_answering(monkeypatch, returncode = 0, stdout = _EMPTY_CACHE)
    assert amd._icd_manifest_is_usable(manifest) is False


def _kernel_stack_hint_block() -> str:
    """The whole diagnosis chain, from its `if` through the closing `fi`."""
    lines = _install_sh_lines()
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    close = next(i for i in range(end + 1, len(lines)) if lines[i] == "fi")
    return "\n".join(lines[start : close + 1])


def test_the_kernel_stack_advice_is_gated_on_the_node_being_absent():
    """Kernel-stack advice is gated on /dev/kfd being absent, since its presence proves the stack loaded."""
    block = _kernel_stack_hint_block()
    present = block.index("[ -e /dev/kfd ] && _kfd_node_is_amds")
    absent = block.index("[ ! -e /dev/kfd ] || ! _kfd_node_is_amds")
    assert present < block.index("kernel stack is already loaded") < absent
    assert absent < block.index("Install the ROCm kernel stack")


def test_the_installer_names_the_userspace_when_the_node_is_already_there(tmp_path):
    """Runs the -e /dev/kfd branch for real with a path this case owns, so it executes on every host."""
    out = _kernel_stack_hint_text(tmp_path, topology = True, kfd_present = True)
    assert "Install the ROCm kernel stack" not in out
    assert "kernel stack is already loaded" in out
    assert "rocminfo" in out
    # An existing /dev/kfd does not prove the AMD driver created it.
    out = _kernel_stack_hint_text(tmp_path, topology = False, kfd_present = True)
    assert "kernel stack is already loaded" not in out
    assert "Install the ROCm kernel stack" in out
    out = _kernel_stack_hint_text(
        tmp_path, topology = False, topology_readable = False, kfd_present = True
    )
    assert "kernel stack is already loaded" in out


def _backend_env(**env: str) -> dict:
    """The runner's environment with both backend requests cleared, plus this case's."""
    _base = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    return {**_base, **env}


def _resolved_backend(**env: str) -> str:
    """What install.sh resolves the backend request to, for one environment."""
    script = "\n".join(
        [
            _shell_fn(_install_sh_lines(), "_requested_llama_backend"),
            "_requested_llama_backend",
        ]
    )
    return _install_sh_run(script, env = _backend_env(**env)).strip()


def _gpu_node_scope(**env: str) -> str:
    """On an NVIDIA --no-torch host only an explicit backend request can make the check fire."""
    lines = _install_sh_lines()
    script = "\n".join(
        [
            "SKIP_TORCH=true",
            "TORCH_INDEX_URL=''",
            "_has_usable_nvidia_gpu() { return 0; }",
            _shell_fn(lines, "_requested_llama_backend"),
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            "_run_may_open_a_gpu_node && echo yes || echo no",
        ]
    )
    return _install_sh_run(script, env = _backend_env(**env)).strip()


_FORCE_VULKAN = "UNSLOTH_FORCE_VULKAN"
_BACKEND = "UNSLOTH_LLAMA_CPP_BACKEND"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({_FORCE_VULKAN: "1"}, "vulkan", "yes"), id = "the_legacy_flag_alone"),
    pytest.param(({_BACKEND: "cuda", _FORCE_VULKAN: "1"}, "cuda", "no"),
                 id = "cuda_over_the_legacy_flag"),
    pytest.param(({_BACKEND: "auto", _FORCE_VULKAN: "1"}, "auto", "no"),
                 id = "auto_over_the_legacy_flag"),
    pytest.param(({_FORCE_VULKAN: "0"}, "", "no"), id = "the_legacy_flag_set_to_zero"),
])
# fmt: on
def test_how_the_legacy_vulkan_flag_is_resolved(case):
    """The legacy UNSLOTH_FORCE_VULKAN flag yields to any recognised backend value, including auto."""
    env, resolved, scope = case
    assert _resolved_backend(**env) == resolved
    assert _gpu_node_scope(**env) == scope


_VK_MASK = "GGML_VK_VISIBLE_DEVICES"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({_VK_MASK: ""}, {"vulkan"},
                  ("visibility mask is also in force", f"{_VK_MASK} is empty"), ()),
                 id = "an_empty_mask_on_a_vulkan_build"),
    pytest.param(({_VK_MASK: "3"}, {"vulkan"},
                  ("names a device this cannot resolve", f"{_VK_MASK}='3'"), ()),
                 id = "an_unboundable_mask_on_a_vulkan_build"),
    pytest.param(({_VK_MASK: ""}, {"hip"}, (), (_VK_MASK,)), id = "the_same_mask_on_a_hip_build"),
    pytest.param(({}, {"vulkan"}, (), (_VK_MASK,)), id = "no_such_mask_at_all"),
    # ggml extracts as size_t, so -1 wraps to 2**64-1 and always throws: a blocker.
    pytest.param(({_VK_MASK: "-1"}, {"vulkan"},
                  ("visibility mask is also in force", f"{_VK_MASK}='-1'"),
                  ("names a device this cannot resolve",)),
                 id = "a_negative_ordinal_always_throws_so_it_blocks"),
    pytest.param(({_VK_MASK: "0,-1"}, {"vulkan"},
                  ("visibility mask is also in force",),
                  ("names a device this cannot resolve",)),
                 id = "a_negative_ordinal_behind_a_valid_one_still_blocks"),
    # ggml stops at the first token that does not extract.
    pytest.param(({_VK_MASK: "abc,-1"}, {"vulkan"},
                  ("visibility mask is also in force",), ()),
                 id = "a_negative_ordinal_after_a_dead_token_is_never_read"),
    # -0 wraps to 0, which is in range.
    pytest.param(({_VK_MASK: "-0"}, {"vulkan"},
                  ("names a device this cannot resolve", f"{_VK_MASK}='-0'"), ()),
                 id = "a_negative_zero_is_in_range_and_stays_unresolved"),
])
# fmt: on
def test_when_the_vulkan_selector_is_named_beside_the_node(monkeypatch, linux, case):
    """A Vulkan build reads GGML_VK_VISIBLE_DEVICES, not the HIP selectors; a negative ordinal blocks."""
    env, backends, present, absent = case
    reason = _reason_with_masks(monkeypatch, env, backends)
    for _text in present:
        assert _text in reason, _text
    for _text in absent:
        assert _text not in reason, _text


def test_the_two_topology_readers_agree_on_this_host():
    """The shell and Python KFD readers must agree, asserted rather than fixed so it holds on any host."""
    lines = _install_sh_lines()
    script = "\n".join(
        [
            _shell_fn(lines, "_kfd_topology_amd_state"),
            "_st=0",
            "_kfd_topology_amd_state || _st=$?",
            'printf "%s" "$_st"',
        ]
    )
    shell_state = int(_install_sh_run(script).strip())
    python_state = amd._kfd_topology_amd_state()
    assert shell_state == {True: 0, False: 1, None: 2}[python_state]


def test_the_two_confirmed_render_node_readers_agree_on_this_host():
    """Its companion, and the other half of the fallback: the shell one has to be as strict as
    the Python one, or an NVIDIA-only host claims an AMD node in the installer and not in the
    backend."""
    lines = _install_sh_lines()
    script = "\n".join(
        [
            _shell_fn(lines, "_amd_render_node_vendor"),
            _shell_fn(lines, "_a_confirmed_amd_render_node_exists"),
            "_a_confirmed_amd_render_node_exists && printf yes || printf no",
        ]
    )
    assert _install_sh_run(script).strip() == (
        "yes" if amd._a_confirmed_amd_render_node_exists() else "no"
    )


def test_the_installer_kfd_arm_consults_the_same_fallback():
    """The installer's KFD arm must use the same DRM fallback, as the two nodes may differ in group."""
    lines = _install_sh_lines()
    body = _shell_fn(lines, "_amd_nodes_closed_to_this_user")
    _kfd = body.index("= /dev/kfd ]")
    _elif = body.index("elif _node_vendor=")
    _arm = body[_kfd:_elif]
    assert "_kfd_topology_amd_state" in _arm
    assert "_a_confirmed_amd_render_node_exists" in _arm
    assert "-eq 1 ]; then" in _arm and "continue" in _arm


def _loader_blame(
    monkeypatch,
    manifests: dict,
    searched = (),
    **env: str,
) -> "str | None":
    """Each manifest's value says whether it still resolves to a library, since test paths never exist."""
    for var in _VK_OVERRIDE_VARS:
        monkeypatch.delenv(var, raising = False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    usable = dict(manifests)
    usable.update({path: True for path in searched})
    monkeypatch.setattr(amd, "_vulkan_icd_manifest_paths", lambda: list(manifests))
    monkeypatch.setattr(amd, "_searched_vulkan_icd_manifest_paths", lambda: list(searched))
    monkeypatch.setattr(amd, "_icd_manifest_is_usable", lambda path: bool(usable.get(path)))
    monkeypatch.setattr(amd, "_an_icd_is_32_bit", lambda _path: False)
    return amd.the_vulkan_loader_override_to_blame()


_SELECT = "VK_LOADER_DRIVERS_SELECT"
_DISABLE = "VK_LOADER_DRIVERS_DISABLE"
_FORCED = "VK_DRIVER_FILES"
_ICD_RADEON = "/etc/vulkan/icd.d/radeon_icd.x86_64.json"
_ICD_SEARCHED = "/etc/vulkan/icd.d/radeon.json"
_ICD_GONE = "/gone/radeon.json"


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({_ICD_RADEON: True}, (), {_DISABLE: "*"}, _DISABLE), id = "a_disable_glob"),
    pytest.param(({_ICD_RADEON: True}, (), {_SELECT: "nvidia*"}, _SELECT), id = "a_select_list"),
    pytest.param(({_ICD_GONE: False}, (_ICD_SEARCHED,), {_FORCED: _ICD_GONE}, _FORCED),
                 id = "a_forced_list_pointing_at_nothing"),
    pytest.param(({_ICD_RADEON: True}, (), {}, None), id = "no_override_at_all"),
    pytest.param(({_ICD_RADEON: True}, (), {_SELECT: "radeon*", _DISABLE: "radeon*"}, _DISABLE),
                 id = "select_and_disable_naming_the_same_driver"),
    pytest.param(({_ICD_RADEON: True}, (), {_SELECT: "nvidia*", _DISABLE: "intel*"}, _SELECT),
                 id = "select_excluding_it_and_disable_missing_it"),
    pytest.param(({_ICD_RADEON: True}, (), {_SELECT: "nvidia*", _DISABLE: "radeon*"},
                  f"{_SELECT} and {_DISABLE} together"),
                 id = "select_and_disable_each_excluding_it"),
    pytest.param(({__file__: False}, (), {_FORCED: __file__}, None),
                 id = "a_forced_list_whose_manifest_is_permitted"),
    pytest.param(({__file__: False}, (_ICD_SEARCHED,), {_FORCED: __file__}, _FORCED),
                 id = "the_same_list_hiding_a_usable_search"),
    pytest.param(({_ICD_GONE: False}, (_ICD_SEARCHED,), {_FORCED: _ICD_GONE, _DISABLE: "radeon*"},
                  f"{_FORCED} and {_DISABLE} together"),
                 id = "a_stale_forced_path_under_a_filter_that_also_blocks_it"),
    pytest.param(({_ICD_GONE: False}, (_ICD_SEARCHED,), {_FORCED: _ICD_GONE}, _FORCED),
                 id = "the_forced_list_of_that_pair_on_its_own"),
    pytest.param(({__file__: True}, (), {_FORCED: __file__, _DISABLE: "*"}, _DISABLE),
                 id = "the_filter_of_that_pair_on_its_own"),
    pytest.param(({_ICD_RADEON: False}, (), {_DISABLE: "*"}, None),
                 id = "a_filter_over_a_manifest_whose_library_is_gone"),
    pytest.param(({_ICD_RADEON: True}, (), {_DISABLE: "*"}, _DISABLE),
                 id = "the_same_filter_over_a_loadable_manifest"),
    pytest.param(({}, (), {_DISABLE: "*"}, None), id = "a_loader_with_no_manifests_at_all"),
])
# fmt: on
def test_which_override_a_driverless_loader_blames(monkeypatch, case):
    """Blames the override whose clearing restores a loadable driver, or both where neither alone does."""
    manifests, searched, env, blamed = case
    _answer = _loader_blame(monkeypatch, manifests, searched = searched, **env)
    if blamed is None:
        assert _answer is None
    else:
        assert _answer == blamed


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((True, {_DISABLE: "*"}, ("no driver it can load", _DISABLE),
                  ("reinstall the Vulkan driver",)),
                 id = "a_filter_over_a_manifest_that_resolves"),
    pytest.param((False, None, ("reinstall the Vulkan driver",), ()),
                 id = "no_override_and_a_library_that_is_gone"),
])
# fmt: on
def test_which_repair_the_no_driver_sentence_prescribes(monkeypatch, linux, tmp_path, case):
    """Says reinstall only when the library is missing; a filter over a good manifest needs clearing."""
    library_present, env, present, absent = case
    _icd_manifest(tmp_path, "radeon_icd.json", present = library_present)
    reason = _vulkan_node_hint_under_icd_list(
        monkeypatch, None, search_dirs = [str(tmp_path)], env = env
    )
    for _text in present:
        assert _text in reason, _text
    for _text in absent:
        assert _text not in reason, _text


def _repair_user_under(id_stub: str) -> str:
    """Runs install.sh's own _amd_repair_user assignment under a stubbed id, so reverting it fails."""
    lines = _install_sh_lines()
    i = _install_sh_anchor(lines, "_amd_repair_user=$(id -un")
    script = "\n".join([id_stub, lines[i].strip(), 'printf "%s" "$_amd_repair_user"'])
    out = subprocess.run(["sh", "-c", script], capture_output = True, text = True, check = True)
    return out.stdout


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(("id() { echo 12345; return 1; }", ""), id = "a_uid_with_no_passwd_entry"),
    pytest.param(("id() { echo ada; return 0; }", "ada"), id = "a_resolvable_account"),
])
# fmt: on
def test_which_account_the_installer_names(case):
    """GNU id prints the uid and exits 1 for an unknown one, so an empty fallback never runs."""
    id_stub, named = case
    assert _repair_user_under(id_stub) == named


def _topology_state(monkeypatch, tmp_path, entries: "dict[str, str | None]"):
    """Redirects os.listdir and open on the module, since a global open patch recurses through pathlib."""
    real = tmp_path / "nodes"
    for name, body in entries.items():
        node = real / name
        node.mkdir(parents = True)
        properties = node / "properties"
        properties.write_text(body or "", encoding = "utf-8")
        if body is None:
            properties.chmod(0o000)
    prefix = "/sys/class/kfd/kfd/topology/nodes"

    def _redirect(path):
        return str(path).replace(prefix, str(real))

    real_listdir, real_open = os.listdir, open
    monkeypatch.setattr(amd.os, "listdir", lambda p: real_listdir(_redirect(p)))
    monkeypatch.setattr(
        amd, "open", lambda p, *a, **k: real_open(_redirect(p), *a, **k), raising = False
    )
    return amd._kfd_topology_amd_state()


# fmt: off
@pytest.mark.skipif(_running_as_root(), reason = "root opens a 0000 file, so nothing is unreadable")
@pytest.mark.parametrize("case", [
    pytest.param(({"0": "cpu_cores_count 16\nsimd_count 0\nvendor_id 0\n", "1": None}, None),
                 id = "a_gpu_node_that_will_not_open"),
    pytest.param(({"0": "cpu_cores_count 16\nvendor_id 0\n",
                   "1": "simd_count 128\nvendor_id 4318\n"}, False),
                 id = "every_node_read_and_none_amd"),
    pytest.param(({"0": "cpu_cores_count 16\nvendor_id 0\n", "1": None,
                   "2": "simd_count 256\nvendor_id 4098\n"}, True),
                 id = "an_amd_node_beside_an_unreadable_sibling"),
])
# fmt: on
def test_what_an_unreadable_topology_node_answers(monkeypatch, tmp_path, case):
    """The CPU node opens and the GPU node does not: one entry short of "names none". False
    there drops /dev/kfd from the closed list, so a host whose KFD is owned by video and whose
    render node is owned by render is told to join render alone and left with KFD shut -- the
    node the ROCm caller actually needs. install.sh states the same rule for the same
    decision: a topology that could not be READ is not one that named another vendor. The
    controls keep the fix from becoming "never answer False", which would let an NVIDIA-only
    host whose KFD nodes all read as 4318 claim an AMD card, and confirm that one AMD node
    answers True however many siblings failed."""
    entries, state = case
    assert _topology_state(monkeypatch, tmp_path, entries) is state


@pytest.mark.parametrize(
    "select,disable,allowed",
    [
        ("", "", True),
        ("radeon*", "", True),
        ("nvidia*", "", False),
        ("", "radeon*", False),
        # Khronos: disable applies before select, so no driver remains.
        ("radeon*", "radeon*", False),
        ("radeon*", "nvidia*", True),
        ("radeon*,nvidia*", "nvidia*", True),
    ],
)
def test_the_loader_filters_are_an_allowlist_then_a_denylist(monkeypatch, select, disable, allowed):
    """VK_LOADER_DRIVERS_DISABLE is checked before VK_LOADER_DRIVERS_SELECT, so a denylist wins."""
    monkeypatch.setenv("VK_LOADER_DRIVERS_SELECT", select)
    monkeypatch.setenv("VK_LOADER_DRIVERS_DISABLE", disable)
    assert amd._vulkan_loader_allows("/usr/share/vulkan/icd.d/radeon_icd.x86_64.json") is allowed


def _lacking(monkeypatch, *, topology, confirmed_drm, kfd_present, render_present):
    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: topology)
    monkeypatch.setattr(amd, "_kfd_topology_has_an_amd_gpu", lambda: topology is True)
    monkeypatch.setattr(amd, "_a_confirmed_amd_render_node_exists", lambda: confirmed_drm)
    monkeypatch.setattr(amd, "_amd_render_node_exists", lambda: render_present)
    monkeypatch.setattr(
        amd.os.path, "exists", lambda p: kfd_present if p == amd._KFD_NODE else os.path.exists(p)
    )
    return amd._amd_nodes_the_runtime_lacks(needs_kfd = True)


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((None, True, [amd._KFD_NODE]), id = "a_masked_topology_drm_confirms"),
    pytest.param((None, False, []), id = "a_masked_topology_drm_cannot_confirm"),
    pytest.param((False, True, []), id = "a_topology_that_names_no_amd_gpu"),
])
# fmt: on
def test_when_a_missing_kfd_is_reported_as_missing(monkeypatch, case):
    """Confirmed DRM names an AMD render node even when sysfs hides KFD, so /dev/kfd is reported missing."""
    topology, confirmed_drm, lacks = case
    assert (
        _lacking(
            monkeypatch,
            topology = topology,
            confirmed_drm = confirmed_drm,
            kfd_present = False,
            render_present = True,
        )
        == lacks
    )


def test_the_wording_does_not_claim_a_loaded_driver_it_cannot_prove(monkeypatch):
    """The masked-topology route has not read the amdkfd driver's own sysfs, so it must not
    say the kernel driver is loaded and a reinstall is pointless. The confirmed-topology route
    still does, which is the control."""
    monkeypatch.setattr(amd, "_amd_nodes_the_runtime_lacks", lambda **_k: [amd._KFD_NODE])
    monkeypatch.setattr(amd, "amd_nodes_closed_to_this_user", lambda **_k: [])

    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: None)
    masked = amd.amd_node_permission_hint(needs_kfd = True) or ""
    assert "the kernel driver is loaded" not in masked
    assert "/dev/kfd" in masked and "--device /dev/kfd" in masked

    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: True)
    named = amd.amd_node_permission_hint(needs_kfd = True) or ""
    assert "the kernel driver is loaded" in named


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(([amd._KFD_NODE], ("/dev/kfd",)), id = "with_a_second_finding_after_it"),
    pytest.param(([], ()), id = "when_the_command_is_the_only_finding"),
])
# fmt: on
def test_the_message_ends_with_the_command_a_user_pastes(monkeypatch, case):
    """The command ends the message, so a user copying it to end of line does not pick up the next word."""
    lacks, also_present = case
    monkeypatch.setattr(amd, "amd_nodes_closed_to_this_user", lambda **_k: ["/dev/dri/renderD128"])
    monkeypatch.setattr(amd, "_amd_nodes_the_runtime_lacks", lambda **_k: lacks)
    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: True)
    monkeypatch.setattr(amd, "_repair_account", lambda: "ada")
    monkeypatch.setattr(
        amd, "_groups_that_own", lambda _paths: (["render"], [], [], [], [], [], [], [])
    )
    hint = amd.amd_node_permission_hint(needs_kfd = True) or ""

    assert "sudo usermod -a -G render ada" in hint
    assert hint.rstrip().endswith("sudo usermod -a -G render ada")
    for _text in also_present:
        assert _text in hint
