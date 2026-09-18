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
import platform
import pwd
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import amd
from utils.hardware import hardware


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
    # ...and the environment fallback agrees with it, so a case that is not about which
    # account is named does not have to say. The two tests that ARE about it -- a uid with
    # no passwd entry, and a container whose USER disagrees -- set their own.


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(amd.platform, "system", lambda: "Linux")


# The llama-server that is never there: every empty-probe case below asks about an install
# whose binary does not exist, because the reason under test is decided before it is run.
_NO_SUCH_SERVER = "/nonexistent/llama-server"

# The two device nodes a ROCm host has: the KFD, which only HIP opens, and one render
# node, which HIP and the Vulkan loader both open. Named because most cases below are
# about one of the two being shut while the other is not.
_AMD_NODES = ["/dev/kfd", "/dev/dri/renderD128"]

# The order _groups_that_own returns its buckets in. Named here so a case can say which
# bucket it is about instead of counting commas in an eight-element tuple.
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


# The real derivation, captured before any case stubs it. _nodes() stubs _groups_that_own
# for every case that is not about it, so a case that IS about it has to put the real one
# back rather than describe the host twice.
_REAL_GROUPS_THAT_OWN = amd._groups_that_own


def _the_real_group_derivation(monkeypatch):
    """Undo _nodes()'s stub, for a case whose point is what the nodes themselves say."""
    monkeypatch.setattr(amd, "_groups_that_own", _REAL_GROUPS_THAT_OWN)


def _owning(monkeypatch, **named):
    """Stub the group derivation with the buckets this case is about.

    Every arm that is not testing the derivation itself stubs it, because these node paths
    are patched rather than created and stat cannot name their groups. Saying so explicitly
    keeps the answer off whatever the runner happens to have at the same path.
    """
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
    """A host whose ``present`` nodes exist and whose ``openable`` subset can be opened.

    ``amd_owned`` is the vendor of the hardware behind those nodes, stubbed here and
    exercised for real in the two tests below it. ``vendor_readable`` is whether sysfs will
    say so: a container can map the node and hide the entry that names its vendor, and the
    two are different answers.

    ``gpu_count`` is how many AMD GPU agents KFD enumerates, and it defaults to the AMD
    render nodes this host was given, since that is what the described host would report.
    It is stubbed here rather than left live for the reason this whole helper exists: it
    reads /sys/class/kfd, so a case that merely sets a visibility mask was asking the
    RUNNER how many GPUs it has. On a machine with no AMD card that read is None, which
    means "no bound", so every selector looks like a narrowing; on a real gfx1151 it is 1,
    so a selector naming device 0 looks like the whole host. Those are opposite verdicts,
    and that is precisely how this file passed here and failed on the hardware it is
    written for. Pass None explicitly for the unreadable-topology case.
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
    # None is a topology that could not be READ, which the closed-node walk answers
    # differently from one that read and named another vendor. Defaults to the hardware so
    # every arm written before the distinction existed is unaffected.
    _topology = amd_owned if topology == "as-owned" else topology
    monkeypatch.setattr(amd, "_kfd_topology_has_an_amd_gpu", lambda: _topology is True)
    monkeypatch.setattr(amd, "_kfd_topology_amd_state", lambda: _topology)
    if gpu_count == "as-present":
        _renders = [p for p in present if p.startswith("/dev/dri/renderD")]
        # Zero is not a bound, and the production reader says so too: a topology naming no
        # GPU bounds nothing, so it answers None rather than 0.
        gpu_count = len(_renders) if (_renders and amd_owned) else None
    monkeypatch.setattr(amd, "amd_kfd_gpu_node_count", lambda: gpu_count)
    # These paths are patched rather than created, so stat cannot name their groups; say
    # so explicitly instead of leaving it to whether the runner happens to have a node at
    # the same path. The derivation itself is exercised in its own tests below.
    _owning(monkeypatch)


# The parameter tables below are hand-wrapped, one case to a row. Left to the formatter
# each row becomes one argument per line, which is most of the length the review objected
# to, so this span is fenced off. Nothing but the wrapping depends on the fence.
# fmt: off
# The node layouts the families below are written over, named once so a case can say which
# host it is about. SHUT is present and refused, OPEN is present and openable.
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
    # The control. Without it every assertion here also passes on a host with no AMD
    # hardware, where the list is empty for a reason that is not this bug.
    pytest.param(("Linux", _BOTH_OPEN, []), id = "a_host_whose_nodes_open"),
    pytest.param(("Linux", _NO_NODES, []), id = "a_host_with_no_amd_nodes"),
    # The false positive this nearly shipped with: render nodes are root:render for EVERY
    # vendor, so this box (8 NVIDIA cards, an account outside the render group) listed
    # all eight and sent a CUDA user after the AMD groups. CUDA opens /dev/nvidia*.
    pytest.param(("Linux", _NVIDIA_RENDER_SHUT, []), id = "an_nvidia_hosts_closed_render_node"),
    # macOS and Windows have no render nodes, and os.access on Windows answers for a
    # permission model this message does not describe.
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
    assert amd._render_node_is_amd("/dev/dri/renderD130") is False  # unreadable


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
    # Vulkan never opens /dev/kfd, so a closed one does not stop every backend. Claiming
    # it did sent a Vulkan user with an unrelated failure after ROCm groups.
    pytest.param((_KFD_SHUT, {}, True, ("ROCm cannot use",), ("no GPU backend",)),
                 id = "the_hint_is_rocm_specific_when_only_kfd_is_closed"),
    # Its pair: HIP and the Vulkan loader both open the render node.
    pytest.param((_RENDER_SHUT, {}, True, ("no GPU backend can use",), ()),
                 id = "the_hint_covers_every_backend_when_a_render_node_is_closed"),
    # No membership creates /dev/kfd, so a ROCm caller needs both sentences. The
    # kernel-stack wording this used to assert was wrong, and asserting it is what kept
    # it: that sentence needs the KFD topology, which is amdkfd's own sysfs, so the stack
    # is loaded wherever it is readable. install.sh's branch is gated on it being ABSENT.
    pytest.param((_RENDER_SHUT, {}, True, (_USERMOD, "/dev/kfd"), ("kernel stack",)),
                 id = "the_hint_says_so_when_kfd_does_not_exist_at_all"),
    # The control: /dev/kfd exists, so the stack is loaded and the groups are the whole
    # repair. Telling this user to install one would be the #10466 mistake.
    pytest.param((_BOTH_SHUT, {}, True, (), ("kernel stack",)),
                 id = "a_closed_but_present_kfd_node_says_nothing_about_the_kernel_stack"),
    # And Vulkan never opens /dev/kfd, so its absence is not that caller's problem.
    pytest.param((_RENDER_SHUT, {}, False, (), ("kernel stack",)),
                 id = "a_vulkan_caller_is_not_told_about_a_kernel_stack_it_does_not_need"),
    # render,video is the usual pair, not a universal truth: a container gets numeric gids
    # with no matching NAMES, a minimal distribution can ship no render group, and a
    # root:root node is not fixed by joining. Fails before the fix, which hard-coded it.
    pytest.param((_BOTH_SHUT, dict(joinable = ["kfd", "gpu"]), True, ("usermod -a -G kfd,gpu ada",),
                  ("render,video",)),
                 id = "the_repair_names_the_groups_the_closed_nodes_belong_to"),
    # One group named, so the sentence has to agree with the command rather than saying
    # "groups" over a single name.
    pytest.param((_RENDER_SHUT, dict(joinable = ["render"]), True,
                  ("usermod -a -G render ada", "render group and then log out"), ()),
                 id = "a_single_owning_group_is_not_pluralised"),
    # The control: the derivation is best effort, so a host whose nodes cannot be stat'd
    # still gets advice rather than an empty -G argument, and that is the documented pair.
    pytest.param((_KFD_SHUT, {}, True, (_USERMOD,), ()),
                 id = "unreadable_nodes_fall_back_to_the_documented_pair"),
    # usermod -a -G takes names only: shadow 4.13 answers ``group '993' does not exist``
    # and exits 6, run live here rather than read out of the man page. Fails before the
    # fix, which put the bare number in -G. The assertion is on the NUMBER, not on the
    # command, because the repair does name usermod after a groupadd that names the GID.
    pytest.param((_KFD_SHUT, dict(unnamed = [993]), True, ("--group-add 993",),
                  ("usermod -a -G 993",)),
                 id = "an_unnamed_gid_is_not_handed_to_usermod"),
    # The control that keeps that suppression narrow: a host with one node in a real
    # group and another in an unnamed one can still fix half of it by joining.
    pytest.param((_BOTH_SHUT, dict(joinable = ["render"], unnamed = [993]), True,
                  ("usermod -a -G render ada", "GID 993"), ()),
                 id = "a_joinable_group_beside_an_unnamed_gid_is_still_prescribed"),
    # A udev rule leaving the node root:render 0600 denies its own group, so joining
    # render runs, succeeds, and opens nothing: the repair there is the rule. Fails before
    # the fix, which named the owning group whatever the mode said.
    pytest.param((_KFD_SHUT, dict(no_group = ["/dev/kfd"]), True, ("udev rule",),
                  ("usermod -a -G",)),
                 id = "a_node_no_membership_opens_is_not_answered_with_usermod"),
    # The control for BOTH suppressions: every list empty means the nodes could not be
    # stat'd at all, a detection miss rather than evidence that joining cannot work.
    pytest.param((_KFD_SHUT, {}, True, (_USERMOD,), ()),
                 id = "a_host_whose_nodes_could_not_be_read_still_gets_the_documented_pair"),
    # /dev/kfd without /dev/dri passes every probe here and still cannot initialise ROCm,
    # because ROCr opens a render node to reach amdgpu, which is why docker/run.sh passes
    # both. No group creates the missing one. Fails before the fix, which named only kfd.
    pytest.param((_KFD_SHUT, {}, True,
                  ("No AMD render node", "--device /dev/kfd --device /dev/dri"), ()),
                 id = "a_container_given_kfd_but_no_render_node_is_told_so"),
    # Its control: an ordinary AMD host has a render node and only cannot open it, so
    # claiming the device mapping is wrong invents a second repair.
    pytest.param((_BOTH_SHUT, {}, True, (), ("No AMD render node",)),
                 id = "a_host_that_has_a_render_node_is_not_told_to_map_one"),
    # The sentence for the external-denial bucket, since the bucket cases below only
    # decide it: naming a mode to fix on a node whose mode already grants rw cannot work.
    pytest.param((_KFD_SHUT, dict(external = ["/dev/kfd"]), True,
                  ("owner bits already grant read and write", "container device cgroup or an LSM"),
                  ("chmod",)),
                 id = "the_external_denial_sentence_does_not_prescribe_a_mode_change"),
    # docker's --group-add takes ONE value, so two unnamed groups need the flag twice.
    # Fails before the fix, which interpolated unnamed[0] alone.
    pytest.param((_BOTH_SHUT, dict(unnamed = [993, 994]), True,
                  ("--group-add 993 --group-add 994", "GIDs 993, 994"), ()),
                 id = "every_unnamed_gid_reaches_the_docker_repair"),
    # The control on the wording: the one-GID host is the common one.
    pytest.param((_KFD_SHUT, dict(unnamed = [993]), True,
                  ("GID 993, which has", "--group-add 993."), ()),
                 id = "a_lone_unnamed_gid_is_still_named_in_the_singular"),
    # root:root 0660 opens for anyone in the root group, so the derivation would print
    # `sudo usermod -a -G root`. That grants far more than the GPU: a udev
    # misconfiguration to report rather than a repair.
    pytest.param((_KFD_SHUT, dict(privileged = ["root"]), True, ("root", "udev"), ("usermod",)),
                 id = "the_hint_for_a_privileged_owner_says_it_is_not_the_repair"),
    # The same mapping with that node OPEN: the closed list is empty, and this returned
    # None while ROCr still had no render node. A missing node is not a permission
    # problem, so the hint cannot be gated on one.
    pytest.param((_KFD_OPEN, {}, True, ("AMD render node", "--device /dev/dri"), ("usermod",)),
                 id = "a_container_with_an_open_kfd_and_no_render_node_is_still_told"),
    # The mirror image, --device /dev/dri alone: `closed` is empty and `_render_missing`
    # false, so the hint returned None before its own missing-KFD sentence and left the
    # caller on reinstall advice.
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
    # The control for the closed list above: a host whose nodes open is told nothing, so
    # the assertions here cannot be passing for the absence of hardware.
    pytest.param((_BOTH_OPEN, True, False), id = "a_host_whose_nodes_open_is_told_nothing"),
    # And an NVIDIA host's closed render node is not this bug, whatever its mode says.
    pytest.param((_NVIDIA_RENDER_SHUT, True, False), id = "an_nvidia_host_is_told_nothing"),
    # ``needs_kfd = False`` is a Vulkan binary saying a closed KFD node is not its
    # problem. The render node is open, so the only thing wrong is the node that caller
    # disclaimed; without an open one the answer would be a real Vulkan blocker.
    pytest.param((_ONLY_KFD_SHUT, False, False),
                 id = "a_vulkan_only_caller_is_not_answered_with_a_closed_kfd_node"),
    # The same host asked by a caller that DOES need the KFD node still gets an answer,
    # or the narrowing above has simply silenced the hint.
    pytest.param((_ONLY_KFD_SHUT, True, True), id = "a_rocm_caller_on_that_host_is_answered"),
    # The other half of that control, so the narrowing cannot silence the real case.
    pytest.param((_RENDER_SHUT, False, True),
                 id = "a_vulkan_caller_is_still_answered_about_a_closed_render_node"),
    # The trap a bare "the glob is empty" test falls into: every vendor's render nodes
    # live under /dev/dri/renderD*, so the AMD signal must survive having none. The KFD
    # topology names the vendor and is world-readable.
    pytest.param((_NVIDIA_KFD_OPEN, True, False),
                 id = "a_host_with_no_amd_card_is_not_told_to_map_a_render_node"),
    # And the reason needs_kfd exists: a Vulkan failure with some other cause must not be
    # sent after the ROCm kernel stack.
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
    # `docker run --user 1234` leaves the uid with no passwd entry while USER commonly
    # still says root. usermod against that name succeeds, changes an identity nothing is
    # running as, and leaves the nodes shut: the repair is the container's group wiring.
    # Fails before the fix, which fell back to USER and then to a literal $USER.
    pytest.param((dict(joinable = ["render"]), None, ("--group-add render",),
                  ("usermod -a -G", "root")),
                 id = "a_uid_with_no_passwd_entry_is_not_given_a_usermod"),
    # The control: a uid with a passwd entry is an account usermod can name, and that is
    # the repair on every ordinary host. Without it the fix could be "never prescribe
    # usermod", which removes what #10466 asked for.
    pytest.param((dict(joinable = ["render"]), "ada", ("sudo usermod -a -G render ada",),
                  ("--group-add",)),
                 id = "an_account_the_system_knows_still_gets_the_command"),
    # The unnamed-GID repair is a groupadd AND a usermod, and the second half needs the
    # same account, so printing the pair here would be two commands that cannot both work.
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
    """Which of the three runtime probes a capability case pins, and to what.

    Pinned rather than inherited: all three read whatever torch is installed beside the test,
    and this workspace carries a ROCm one, so a live probe makes an arm pass or fail on the
    runner's wheel rather than on the host the case describes.
    """
    _named = {
        "_expected_rocm_flavor_was_chosen": rocm_intent,
        "_torch_reports_a_hip_runtime": hip_runtime,
        "_torch_reports_another_vendors_runtime": other_vendor,
    }
    return {_probe: _answer for _probe, _answer in _named.items() if _answer is not None}


@pytest.mark.parametrize("case", [
    # Fails before the fix: the old message blamed PyTorch and offered Repair
    # installation, which cannot add an account to a group.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  (_USERMOD,), ("Repair installation",)),
                 id = "the_capability_message_names_the_permission_not_a_torch_mismatch"),
    # The control: on any other unusable-GPU host the PyTorch wording has to survive, or
    # this fix trades one wrong answer for another. A render node too, and open: "the
    # nodes open" must mean every node this host needs, or the control is a container
    # missing /dev/dri getting a message about that instead.
    pytest.param((_BOTH_OPEN, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  ("Repair installation",), ("usermod",)),
                 id = "the_capability_message_is_unchanged_when_the_nodes_open"),
    # A hybrid host whose NVIDIA card raised the verdict while an AMD node happens to be
    # closed. Joining the render group repairs nothing there and reinstalling might.
    pytest.param((_KFD_SHUT, {"nvidia"}, "torch_cuda_unavailable", "2.11.0+cu130", _pins(),
                  ("Repair installation",), ("usermod",)),
                 id = "an_nvidia_mismatch_keeps_the_pytorch_message"),
    # Its pair, one recorded vendor apart: on a hybrid host the hint needs the install to
    # target AMD, and a venv that asked for ROCm and got a CPU wheel is this verdict.
    pytest.param((_KFD_SHUT, {"amd", "nvidia"}, "torch_cpu_build", None,
                  _pins(rocm_intent = True, hip_runtime = False, other_vendor = False), (_USERMOD,),
                  ()),
                 id = "a_hybrid_host_whose_amd_card_raised_it_still_gets_the_hint"),
    # Opening the node leaves a CPU-only wheel with no GPU path, so the reinstall step has
    # to survive the permission hint rather than be replaced by it.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cpu_build", None, _pins(),
                  ("CPU-only build", "Repair installation", _USERMOD), ()),
                 id = "a_cpu_wheel_beside_a_closed_node_is_told_to_do_both"),
    # The pair: a ROCm wheel that cannot open a device is fully explained by the node.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0", _pins(),
                  (_USERMOD,), ("Repair installation",)),
                 id = "a_gpu_wheel_beside_a_closed_node_is_told_only_the_permission"),
    # A supported AMD card qualifies for the mismatch whatever wheel is installed, so a
    # hybrid host records both vendors even when the verdict is about the NVIDIA card. No
    # membership makes a CUDA wheel use the AMD card.
    pytest.param((_KFD_SHUT, {"amd", "nvidia"}, "torch_cuda_unavailable", "2.11.0+cu130",
                  _pins(rocm_intent = False, hip_runtime = False), ("Repair installation",),
                  ("usermod",)),
                 id = "a_hybrid_host_running_cuda_torch_keeps_the_pytorch_message"),
    # The control, and the #10466 host: AMD is the only vendor that qualified, so the
    # verdict can only be about it and the wheel's own tag adds nothing.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0",
                  _pins(rocm_intent = False, hip_runtime = False), (_USERMOD,), ()),
                 id = "an_amd_only_host_needs_no_runtime_evidence"),
    # An AMD-only host running a CUDA-tagged wheel raises the same verdict, and opening
    # the node does not make that wheel use the card, so it needs both. Fails before the
    # fix, where "AMD is the only vendor" made the hint REPLACE the reinstall advice and
    # left the user with a group command and no way to use the GPU.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+cu128",
                  _pins(rocm_intent = False, hip_runtime = False),
                  (_USERMOD, "Repair installation"), ()),
                 id = "a_cuda_wheel_on_an_amd_only_host_keeps_the_reinstall_advice"),
    # Its control, one label apart: a ROCm wheel that cannot initialise the device IS
    # fully explained by the closed node, so the reinstall advice stays suppressed.
    pytest.param((_KFD_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+rocm7.0",
                  _pins(rocm_intent = False, hip_runtime = False), (_USERMOD,),
                  ("Repair installation",)),
                 id = "a_rocm_wheel_on_the_same_host_still_replaces_it"),
    # A venv that recorded ROCm intent and then had a CUDA build installed over it still
    # answers yes to _expected_rocm_flavor_was_chosen, and reading that as "the wheel
    # targets AMD" replaced the repair this host needs with a sentence about groups. The
    # hint is APPENDED instead: the node is real, and the wheel is still the repair.
    pytest.param((_BOTH_SHUT, {"amd"}, "torch_cuda_unavailable", "2.11.0+cu128",
                  _pins(rocm_intent = True, hip_runtime = False),
                  ("matching PyTorch build fixes it", "cannot open"), ()),
                 id = "a_wheel_tagged_for_another_vendor_keeps_the_reinstall_advice"),
    # The control: +cpu names no accelerator, so it settles nothing about which vendor
    # the install targets and the recorded intent is the best evidence there is. Without
    # it the fix reads as "any non-ROCm label wins", which silences the hint on the
    # CPU-torch host #10466 was reported from.
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
    # The load-time line. It runs ahead of the Vulkan branch on purpose: a closed render
    # node is the reason UNDERNEATH "the Vulkan probe reported no device", and that
    # phrasing sends the user after a driver that is already working.
    pytest.param((_RENDER_SHUT, {"vulkan"}, {}, (_USERMOD,),
                  ("the Vulkan probe reported no device",)),
                 id = "the_empty_probe_explanation_names_the_permission"),
    # The control for that narrowing: same host, a non-Vulkan binary, hint must survive.
    pytest.param((_ONLY_KFD_SHUT, {"hip"}, {}, (_USERMOD,), ()),
                 id = "a_rocm_binary_is_still_told_about_the_closed_kfd_node"),
    # A hybrid host whose CUDA build enumerated nothing for its own reasons. Every AMD
    # node is closed, so the hint is available and any build that could use the card gets
    # it. This one cannot, so the mask diagnosis has to survive.
    pytest.param((_BOTH_SHUT, {"cuda"}, {"CUDA_VISIBLE_DEVICES": ""}, ("CUDA_VISIBLE_DEVICES",),
                  ("usermod",)),
                 id = "a_cuda_only_build_is_not_sent_after_the_amd_render_group"),
    # Its control, and why the rule names CUDA rather than "not ROCm": an install this
    # probe cannot read must not lose the diagnosis.
    pytest.param((_BOTH_SHUT, set(), {}, (_USERMOD,), ()),
                 id = "a_build_whose_backend_cannot_be_read_still_gets_the_hint"),
    # _is_vulkan_backend defers such a build to CUDA, so it is a CUDA install here too.
    # Requiring CUDA to be the ONLY shipped library left this layout uncovered.
    pytest.param((_BOTH_SHUT, {"cuda", "vulkan"}, {"CUDA_VISIBLE_DEVICES": ""},
                  ("CUDA_VISIBLE_DEVICES",), ("usermod",)),
                 id = "a_cuda_plus_vulkan_build_is_treated_as_cuda"),
    # A build with no GPU library cannot offload to any card, so its empty probe is not a
    # permission problem and the groups cannot change it.
    pytest.param((_BOTH_SHUT, {"cpu", "base"}, {}, (), ("usermod",)),
                 id = "a_cpu_only_llama_build_is_not_sent_after_the_groups"),
    # Two independent blockers need two fixes. The groups do not clear a visibility mask,
    # so returning the hint alone hid the half the user also has to undo.
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
    # End to end through the caller: a Vulkan binary on a host whose render node opens
    # must not be told about ROCm's node.
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
    """Whether install.sh's missing-kernel-stack branch fires for this closed set.

    The guard is lifted by text, not restated: a restatement would pass whatever the
    installer went on to say. Only the condition is taken, and its two probes are stubbed
    true so the answer depends on nothing but the closed-node reasoning.
    """
    lines = _install_sh_lines()
    # Anchored on the part of the condition this change does NOT touch, then walked back to
    # the "if". Anchoring on the new closed-node text would make the control vacuous: a
    # revert would stop the extraction finding anything, and "the text changed" would read
    # as "the behaviour changed".
    end = _install_sh_anchor(lines, _PCI_SENTENCE)
    start = _install_sh_if_above(lines, end)
    guard = "\n".join(line.strip() for line in lines[start : end + 1])
    script = "\n".join(
        [
            "_has_amd_rocm_gpu() { return 1; }",  # ROCm cannot see the card
            "_amd_gpu_present_via_pci() { return 0; }",  # but the PCI bus can
            # Carried by the guard now that it sits after the case rather than inside the
            # */cpu arm, which supplied them.
            "SKIP_TORCH=false",
            "OS=linux",
            # The run-scope predicate the guard asks in place of a bare SKIP_TORCH test.
            # Lifted, not stubbed, so this arm goes through the installer's own rule.
            *_run_scope_defs(lines, nvidia = nvidia),
            # The route gate, true by default: this harness asks about the closed-node
            # reasoning, and the route has its own tests below.
            f"_amd_node_diag_route={'true' if route else 'false'}",
            # The guard prints through substep on the way to the arm under test, and the
            # chain asks the topology before it. Neither was defined, so on a host with no
            # /dev/kfd the script fell through to `echo FIRED` and looked like it worked,
            # and on a host WITH one it exited 127 from an undefined substep.
            'substep() { echo "$1"; }',
            'C_WARN=""',
            # False, because the arm under test is the third of three and the first
            # requires a topology. Stubbed rather than read: on a real gfx1151 the
            # topology names an AMD GPU and the first arm would answer instead.
            "_kfd_topology_has_an_amd_gpu() { return 1; }",
            guard,
            "    echo FIRED",
            "fi",
        ]
    )
    # The arm under test requires the node to be ABSENT. Owned by the case, so a runner
    # that has one does not answer for it.
    script = _kfd_node_the_case_owns(script, tmp_path, present = False)
    # The set arrives as an exported variable rather than a generated assignment: a repr()
    # inside shell single quotes turns the newline separating two nodes into a literal
    # backslash-n, which reads as one unmatched line and looks like the suppression failing.
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        check = True,
        env = {**os.environ, "_closed_amd_nodes": closed_nodes},
    )
    return "FIRED" in out.stdout


@pytest.mark.parametrize("case", [
    # /dev/kfd existing is the evidence the stack is already loaded, so telling the user
    # to install one cannot help; the group advice after the case is the repair.
    pytest.param(("/dev/kfd", True, False), id = "a_closed_kfd_node_suppresses_it"),
    pytest.param(("/dev/kfd\n/dev/dri/renderD128", True, False),
                 id = "a_closed_pair_suppresses_it"),
    # The case the suppression must not swallow: no /dev/kfd at all, and a render node
    # this account cannot open. No membership creates /dev/kfd, so both have to print.
    pytest.param(("/dev/dri/renderD128", True, True), id = "a_missing_kfd_node_keeps_it"),
    # The negative control: the branch's original behaviour is untouched.
    pytest.param(("", True, True), id = "nothing_closed_keeps_it"),
    # And that the route variable is consulted rather than merely computed.
    pytest.param(("", False, False), id = "the_route_gate_actually_suppresses_it"),
])
def test_when_the_installers_kernel_stack_hint_fires(tmp_path, case):
    """Which hosts install.sh tells to install a ROCm kernel stack."""
    closed_nodes, route, fires = case
    assert _kernel_stack_hint_runs(tmp_path, closed_nodes, route = route) is fires


def _stat_nodes(monkeypatch, modes: dict, names: dict):
    """Stub os.stat and grp for the node set ``modes`` maps to (gid, mode).

    Paths outside the map raise, which covers the vanished-node arm. The stub takes
    ``follow_symlinks`` because pytest itself stats files while this patch is in force, and a
    two-argument lambda takes the whole session down rather than failing the test.
    """

    def _stat(path, *, follow_symlinks = True):
        if str(path) not in modes:
            raise OSError("gone")
        _entry = modes[str(path)]
        gid, mode = _entry[0], _entry[1]
        # Real device nodes are root-owned and POSIX consults the owner class first, so a
        # fake without st_uid would take the owner branch on whatever uid the runner has --
        # often root on CI, which would assert nothing about the group classification these
        # cases name. The tests that DO cover owner precedence build real files in tmp_path.
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
    # Synthetic nodes have no ACL and the account starts outside their groups. Membership
    # tests override these defaults explicitly after building the nodes.
    monkeypatch.setattr(amd, "_has_an_access_acl", lambda _path: False)
    monkeypatch.setattr(amd.os, "getgid", lambda: 1)
    monkeypatch.setattr(amd.os, "getgroups", lambda: [])


_THREE_NODES = {
    "/dev/kfd": (44, 0o660),
    "/dev/dri/renderD128": (44, 0o660),
    "/dev/dri/renderD129": (39, 0o660),
}


@pytest.mark.parametrize("case", [
    # Two nodes owned by one group name it once, and order is first-seen so the command
    # reads like the node list.
    pytest.param((_THREE_NODES, {44: "video", 39: "render"},
                  ["/dev/dri/renderD129", "/dev/kfd", "/dev/dri/renderD128"],
                  dict(joinable = ["render", "video"])),
                 id = "the_group_derivation_reads_the_node"),
    # The container case docker/run.sh documents: --group-add passes the host's numeric
    # gids and no group entry inside matches them. Fails before the fix, which returned
    # the bare number for usermod to consume.
    pytest.param(({"/dev/kfd": (993, 0o660)}, {}, ["/dev/kfd"], dict(unnamed = [993])),
                 id = "a_gid_with_no_group_entry_is_reported_rather_than_prescribed"),
    # A udev rule leaving a node root:render 0600 denies the group as well, so joining
    # render opens nothing. Read the mode before naming the group. Fails before the fix,
    # which read st_gid alone and would have prescribed render.
    pytest.param(({"/dev/kfd": (44, 0o600)}, {44: "render"}, ["/dev/kfd"],
                  dict(no_group = ["/dev/kfd"])),
                 id = "a_node_whose_own_group_cannot_open_it_is_not_a_membership_problem"),
    # Its boundary: the probe's own bar is read-write, so 0640 is not a joinable group.
    pytest.param(({"/dev/kfd": (44, 0o640)}, {44: "render"}, ["/dev/kfd"],
                  dict(no_group = ["/dev/kfd"])),
                 id = "group_read_without_write_is_not_enough"),
    # And the failure mode that must not raise: diagnostics run where things are already
    # wrong, so a node that vanished between the probe and the message drops out.
    pytest.param(({"/dev/dri/renderD128": (44, 0o660)}, {44: "video"},
                  ["/dev/kfd", "/dev/dri/renderD128"], dict(joinable = ["video"])),
                 id = "a_node_that_cannot_be_stat_contributes_nothing"),
    # root:root 0660 opens for anyone in the root group, so the derivation would accept
    # the name and hand it to usermod. That grants a great deal besides the GPU.
    pytest.param(({"/dev/kfd": (0, 0o660, 0)}, {0: "root"}, ["/dev/kfd"],
                  dict(privileged = ["root"])),
                 id = "a_root_owned_node_is_not_answered_with_usermod_root"),
    # The control: render is not privileged, so the same shape still yields the command.
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


# The sentence the kernel-stack branch prints. The harnesses that lift that branch anchor on
# it rather than on either condition: the mapping one is what an earlier change edited, and a
# revert that stopped the extraction finding anything would read "the text changed" as "the
# behaviour changed". _amd_gpu_present_via_pci is named twice, so it is not an anchor either.
_PCI_SENTENCE = "An AMD GPU is on the PCI bus but ROCm cannot see it"


def _install_sh_lines() -> "list[str]":
    """install.sh, split into lines. Every harness below lifts what it needs out of this."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    return install_sh.read_text(encoding = "utf-8").splitlines()


# The one thing a lifted guard reads that no stub can reach: `[ -e /dev/kfd ]`. A test
# operator is not a command, so it cannot be shadowed by a function, and on a host that
# really has an AMD GPU the node really is there -- so the branch taken was decided by the
# runner rather than by the case. That is why this file passed on a box with no AMD card
# and failed on the gfx1151 the feature exists for.
#
# Only the PATH is redirected, and only inside the existence tests. The condition, its
# ordering and its `&&` chain are still lifted verbatim, exactly as _has_amd_rocm_gpu and
# _amd_gpu_present_via_pci are stubbed rather than restated. The node name inside the
# printed sentences is untouched, because the arms below assert on that text.
_KFD_EXISTENCE_TEST = re.compile(r"(\[ +!? ?-e +)/dev/kfd\b")


def _kfd_node_the_case_owns(script: str, tmp_path, *, present: bool) -> str:
    """Point install.sh's `-e /dev/kfd` tests at a node this case creates or withholds."""
    node = tmp_path / "kfd"
    if present:
        node.write_bytes(b"")
    elif node.exists():
        node.unlink()
    redirected, n = _KFD_EXISTENCE_TEST.subn(rf"\g<1>{node}", script)
    # A guard that stopped containing the test would silently go back to reading the host,
    # and every arm here would agree for the wrong reason.
    assert n, "install.sh no longer tests `-e /dev/kfd` where this harness expects it"
    return redirected


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

    Slicing from a definition down to the block under test worked only while the two were
    adjacent, and a shell function must be defined before the line that calls it, so the
    run-scope predicates now sit above the diagnoses they gate and each is lifted by name. A
    miss raises rather than returning "": an undefined function would fail the block under
    test for a reason unrelated to the case.
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
    _is_pip_rocm_family_leaf comes with _torch_opens_amd_nodes, which classifies a ROCm index
    through it; _shell_quote with anything that pastes a command, since without it every
    interpolated name comes back EMPTY and the arm reads as a command naming nobody; and
    _requested_llama_backend because all three predicates read the request through it, which
    is what makes the legacy UNSLOTH_FORCE_VULKAN reach them as in a real run.
    _has_usable_nvidia_gpu is the exception, stubbed because the real one runs nvidia-smi and
    would answer from the runner's hardware -- false by default, so each arm is AMD-only.
    """
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
    """Skip a case that lifts install.sh's REAL stat|awk classifier onto this host.

    install.sh reads each node with GNU ``stat -c '%a|%g|%u|%n|%G'``. BSD stat on macOS
    rejects -c outright, so the record arrives empty, the owner branch cannot match a uid it
    never parsed, and every node falls through to "join a group" -- an artifact of running
    the harness there, not a finding about the installer, which gates every diagnosis that
    reaches this classifier on `[ "$OS" != "macos" ]` and so never runs it on a Mac at all.

    Keyed on the OS, and never on hardware: every Linux host runs all of these whatever GPU
    it has, which is the property the rest of this file was fixed to keep. The synthetic-record
    harness beside this one stubs stat and therefore still runs everywhere.
    """
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
    """The installer's closed-node message, run for a given closed set.

    Lifted from install.sh rather than restated, and the whole block rather than a condition,
    because the thing under test is the sentence it prints. substep is stubbed to plain echo;
    the node list comes in through the environment, since embedding it in the script would
    put a literal backslash-n inside shell quotes and turn two nodes into one unmatched line.
    """
    # repairs None means the REAL derivation runs, reading this host's own stat(1). A
    # stubbed one is a synthetic record and stays portable, so only the real arm is gated.
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
            # Stubbed rather than lifted, along with _an_amd_render_node_is_open and
            # _has_usable_nvidia_gpu below: the real ones read /sys and /dev or run
            # nvidia-smi, so a live one would answer from the runner's own hardware. All
            # three are false by default, so an arm reads as the AMD-only host it describes.
            f"_amd_render_node_present() {{ return {0 if render_present else 1}; }}",
            # A real device node is root-owned; a tmp_path node belongs to the runner, and
            # the installer stops at the owner class when those match, so the arms choose
            # which case they test. Both spellings: the owner-class test asks `id -u` and the
            # repair asks `id -un`, and a stub answering one for the other names a uid as an
            # account. id_user None is a uid with no passwd entry, where the real `id -un`
            # FAILS: the shape of `docker run --user 1234`, and why the container repair
            # exists.
            (
                # printf with the value single-quoted, not echo: a name carrying a backslash
                # is de-escaped by the stub itself otherwise, and the arm testing how such a
                # name is QUOTED then never sees one.
                f'id() {{ case "$1" in -un) printf %s\\\\n {shlex.quote(id_user or "")} ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
                if id_user is not None
                else f'id() {{ case "$1" in -un) return 1 ;; '
                f'-G) echo "{self_gids}" ;; *) echo {self_uid} ;; esac; }}'
            ),
            f"_kfd_topology_has_an_amd_gpu() {{ return {0 if amd_present else 1}; }}",
            f"_an_amd_render_node_is_open() {{ return {0 if render_open else 1}; }}",
            # The route the diagnoses are gated on; the gate has its own tests below.
            "_amd_node_diag_route=true",
            "OS=linux",
            # The run-scope predicate the block asks. Lifted rather than stubbed, so the
            # default arms go through the same rule the installer applies.
            f"SKIP_TORCH={'true' if skip_torch else 'false'}",
            f"_has_usable_nvidia_gpu() {{ return {0 if nvidia else 1}; }}",
            _shell_fn(lines, "_auto_bundle_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_a_gpu_node"),
            # The block also asks which nodes THIS run opens, to name the right --device
            # pair, so the predicate has to exist before the span that calls it.
            _shell_fn(lines, "_torch_index_url_leaf"),
            # _torch_opens_amd_nodes classifies a ROCm index through _is_pip_rocm_family_leaf
            # and every scope predicate reads the request through _requested_llama_backend,
            # so a harness that omits either measures a missing function rather than a rule.
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_requested_llama_backend"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            _shell_fn(lines, "_run_may_open_kfd"),
            # The block quotes every name it pastes into a command through this. Lifted, not
            # stubbed: without it the substitutions come back EMPTY and the arms below read
            # as commands that name nobody.
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
    root -- which both halves classify as privileged and refuse to prescribe, so the arms
    below would assert the privileged sentence instead of the derivation. The group is
    therefore CHOSEN rather than inherited: root may chgrp to any group, an ordinary account
    only to one it belongs to, and it skips when all of those are privileged.
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
    """The guard for it, since a fixture that quietly picks a privileged group does not fail,
    it makes the arms below assert the wrong sentence -- which is how this was found. Stands
    in for the root runner by making the group this account would otherwise inherit
    privileged, so the search has to move off it."""
    monkeypatch.setattr(
        amd,
        "_PRIVILEGED_GROUPS",
        frozenset(amd._PRIVILEGED_GROUPS | {grp.getgrgid(os.getgid()).gr_name}),
    )
    _node, _group = _a_node_a_membership_would_open(tmp_path)
    assert _group not in amd._PRIVILEGED_GROUPS


def test_the_installer_names_the_group_the_node_actually_has(tmp_path):
    """The shell half of the same item, and its only arm that reads a real file: the message
    must name the group that owns the node it refused. Fails before the fix, which printed
    render,video for every host. The group is read with the same stat the installer uses,
    since a runner's primary group is not knowable in advance -- but asserting it is NOT
    render,video is what makes that comparison mean something. 0660 is the mode a real render
    node has, and a default 0644 is a node no membership opens."""
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
    # HIP_VISIBLE_DEVICES=0 names a device rather than hiding one, so it is not why the
    # probe came back empty and reporting it sends the user after a fix that cannot help,
    # on top of the one that can. Fails before the fix, which listed every variable that
    # was merely SET.
    pytest.param(("HIP_VISIBLE_DEVICES", "0", {"hip"}, (_USERMOD,), ("visibility mask",)),
                 id = "a_selector_that_still_exposes_a_device_is_not_a_second_blocker"),
    # The control: -1 names no device, so that host really does need both fixes. Not an
    # EMPTY value, which is the one thing clr's parser is never entered on: the guard is
    # on the first byte, so an empty HIP mask is not a filter and is not the variable clr
    # reads either.
    pytest.param(("HIP_VISIBLE_DEVICES", "-1", {"hip"}, (_USERMOD, "HIP_VISIBLE_DEVICES='-1'"), ()),
                 id = "a_mask_that_hides_everything_is_still_reported"),
    # CUDA and HIP parse the list left to right and stop at the first entry that names no
    # device, so -1 leading the list leaves nothing enumerated.
    pytest.param(("CUDA_VISIBLE_DEVICES", "-1", {"hip"},
                  ("CUDA_VISIBLE_DEVICES='-1'", "visibility mask is also in force"), ()),
                 id = "a_negative_first_entry_hides_everything"),
    # And its control: 0,-1 stops at the -1 but has already exposed GPU 0. Without this
    # the fix could be "any minus sign anywhere hides everything", which passes the case
    # above and is wrong.
    pytest.param(("CUDA_VISIBLE_DEVICES", "0,-1", {"hip"}, (), ("visibility mask",)),
                 id = "a_leading_valid_entry_survives_a_later_invalid_one"),
    # A Vulkan-only install reads none of these four, so an inherited HIP or CUDA mask is
    # not a blocker for it at any value -- the render node it cannot open is.
    pytest.param(("HIP_VISIBLE_DEVICES", "-1", {"vulkan"}, (_USERMOD,), ("visibility mask",)),
                 id = "a_vulkan_build_is_not_told_about_a_mask_it_never_reads"),
    # Its control, one backend apart: the identical environment must still report the
    # mask when the install is one that actually reads it.
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
    """install.sh's index summary and the two diagnoses that follow it, run for one index.

    The whole span is lifted rather than the guard alone, because the thing under test is
    WHERE the diagnosis sits relative to the case: a copy of the condition would answer the
    same whichever arm it had been left in.

    The three arms of that case are told apart by ``-e /dev/kfd`` and the KFD topology, and
    both of those are the HOST's until this harness takes them: the case here is a machine
    with no runtime, so the node is withheld and the topology stubbed empty. Left to the
    host these cases passed on a box with no AMD GPU and failed on the hardware the feature
    is for, which is the wrong way round for every one of them.
    """
    lines = _install_sh_lines()
    start = max(i for i, line in enumerate(lines) if line == 'case "$TORCH_INDEX_URL" in')
    anchor = next(i for i in range(start, len(lines)) if "needs a recent kernel" in lines[i])
    # Through the closed-node block as well, so one run shows which diagnosis this index gets.
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
            "_kfd_topology_has_an_amd_gpu() { return 1; }",
            # The route gate classifies the index by its canonical leaf, so the classifiers
            # are lifted rather than stubbed, or the per-URL cases below would assert about
            # the stub. They are defined above the case in install.sh, so the span lifted
            # below calls them without carrying them.
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
    # The runtime-less host this was written for: no /dev/kfd, so the per-arch reroute
    # rewrote its cpu index to a gfx one and it took the other arm of the case. Left
    # inside the */cpu arm the diagnosis never printed for the host that needed it. Fails
    # before the fix, which produced only the wheels line for this index.
    pytest.param((_GFX_INDEX, "", ("ROCm cannot see it",), ()),
                 id = "the_kernel_stack_diagnosis_reaches_a_rerouted_gfx_index"),
    # The control: the arm the diagnosis used to live in must keep it, or the hoist has
    # moved the message rather than widened it.
    pytest.param(("https://download.pytorch.org/whl/cpu", "", ("ROCm cannot see it",), ()),
                 id = "a_cpu_index_on_the_same_host_still_gets_it"),
    # And the suppression the hoist must carry with it: on a gfx index too, an existing
    # /dev/kfd means the group advice is the repair and the kernel stack is not.
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
    """The empty-probe reason on a closed-node host carrying several visibility masks.

    ``gpu_count`` is what KFD enumerates, so a selector can be judged against something. None
    is the default because it is what an unreadable topology answers, which is the state
    every test written before that check ran in.
    """
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
    # _gpu_device_ordinal_active reads a whitespace GPU_DEVICE_ORDINAL as no filter, so
    # an empty one hides nothing. Fails before the fix, which applied the CUDA/HIP
    # first-token rule to all four names alike and named a variable already inert.
    pytest.param(({"GPU_DEVICE_ORDINAL": ""}, None, (_USERMOD,), ("visibility mask",)),
                 id = "an_empty_ordinal_variable_is_not_a_filter"),
    # Its control: a value that IS a filter and whose first entry names no device leaves
    # nothing enumerated, so that host needs both fixes.
    pytest.param(({"GPU_DEVICE_ORDINAL": "-1"}, None, (_IN_FORCE, "GPU_DEVICE_ORDINAL='-1'"), ()),
                 id = "an_ordinal_that_hides_everything_is_still_reported"),
    # clr reads HIP_VISIBLE_DEVICES when it is non-empty and CUDA_VISIBLE_DEVICES
    # otherwise, so an empty CUDA mask underneath a valid HIP one is never looked at.
    # Fails before the fix, which judged each of the four on its own value.
    pytest.param(({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": ""}, None, (_USERMOD,),
                  ("visibility mask",)),
                 id = "an_empty_cuda_mask_behind_a_valid_hip_one_is_not_consulted"),
    # The control, one variable apart: with no HIP mask above it the same empty CUDA
    # value is the one clr reads, and it exposes nothing.
    pytest.param(({"CUDA_VISIBLE_DEVICES": ""}, None, ("CUDA_VISIBLE_DEVICES is empty", _IN_FORCE),
                  ()),
                 id = "the_same_empty_cuda_mask_blocks_once_hip_is_unset"),
    # ROCr is a LOWER layer than clr and composes rather than defers: it filters the
    # agent list hsa_iterate_agents returns and the HIP ordinals index what it left, so
    # an empty ROCr mask leaves nothing to index whatever HIP says. This is what keeps
    # the fix from being "only the winner of the precedence chain counts".
    pytest.param(({"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""}, None,
                  ("ROCR_VISIBLE_DEVICES is empty", _IN_FORCE), ()),
                 id = "an_empty_rocr_mask_blinds_the_runtime_under_a_valid_hip_one"),
    # HIP stops at the first index no device answers to, so HIP_VISIBLE_DEVICES=3 on a
    # one-GPU host exposes nothing -- which is the empty probe being explained, and was
    # read as a valid selector.
    pytest.param(({"HIP_VISIBLE_DEVICES": "3"}, 1, (_IN_FORCE, "HIP_VISIBLE_DEVICES='3'"), ()),
                 id = "an_ordinal_naming_a_device_that_is_not_there_hides_everything"),
    # The control: the same host and variable pointing at a GPU it has. Without it the
    # fix could be "any ordinal blocks", which would take the GPU off every host with a
    # legitimate selector.
    pytest.param(({"HIP_VISIBLE_DEVICES": "0"}, 1, (), (_IN_FORCE,)),
                 id = "an_ordinal_that_does_name_a_device_is_still_not_a_blocker"),
    # The other control: an unreadable KFD topology is a detection miss, and reading it
    # as "no devices" would call every selector on the host a blocker.
    pytest.param(({"HIP_VISIBLE_DEVICES": "3"}, None, (), (_IN_FORCE,)),
                 id = "an_unreadable_device_count_leaves_the_selector_alone"),
    # ROCr accepts a UUID as well as an ordinal, and one naming no device stops the list
    # exactly as a bad ordinal does. Nothing here can match a UUID against the KFD count,
    # so it is reported as unresolved rather than dismissed, which is what the ordinal
    # check did. Reported, not judged: claiming it blocks would invent a fault this
    # cannot see. Fails before the fix, which said nothing for any non-digit entry.
    pytest.param(({"ROCR_VISIBLE_DEVICES": _UUID}, 1, ("cannot resolve", "ROCR_VISIBLE_DEVICES"),
                  ("which the groups do not clear",)),
                 id = "a_rocr_selector_naming_a_uuid_is_reported_as_unresolved"),
    # The control that keeps it narrow: an ordinal the count can resolve is judged as
    # before, so the new sentence cannot appear on every host that sets the variable.
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0"}, 2, (), ("cannot resolve", "visibility mask")),
                 id = "a_rocr_ordinal_that_names_a_device_is_still_left_alone"),
    # The other boundary. An earlier revision asserted the opposite here, on the claim
    # that only ROCr accepts a UUID; rocdevice.cpp refutes it, matching a "GPU-" token
    # against each agent's own HSA_AMD_AGENT_INFO_UUID before falling back to an ordinal.
    # So the token may name a device or nothing, as under ROCr, and nothing here can tell
    # which. The count is deliberately known here: a UUID is not an index into it.
    pytest.param(({"HIP_VISIBLE_DEVICES": _UUID}, 1, ("names a device this cannot resolve",),
                  (_IN_FORCE,)),
                 id = "a_uuid_in_the_hip_layer_is_unresolved_rather_than_a_blocker"),
    # ROCr filters the physical list first and renumbers the survivors; HIP indexes
    # those. With ROCR_VISIBLE_DEVICES=0 on a two-GPU host one survives, so HIP ordinal 1
    # names nothing, while against the physical count of 2 it reads as a valid selector.
    # Fails before the fix, which used the KFD count for every layer.
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"}, 2,
                  ("HIP_VISIBLE_DEVICES='1'", "which the groups do not clear"), ()),
                 id = "a_hip_ordinal_is_judged_against_what_rocr_left"),
    # The control: ROCr leaving both devices makes HIP ordinal 1 a real device again, so
    # the composed reading must not call every stacked pair a blocker.
    pytest.param(({"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "1"}, 2, (),
                  ("which the groups do not clear",)),
                 id = "the_same_ordinal_inside_what_rocr_left_is_not_a_blocker"),
    # And the boundary: a UUID in the ROCr layer means the survivors cannot be counted, so
    # the HIP ordinal is judged against nothing rather than against an invented count.
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
            # these tests are about. _requested_llama_backend comes with them because every
            # scope predicate reads the backend request through it.
            _shell_fn(lines, "_torch_index_url_leaf"),
            _shell_fn(lines, "_is_pip_rocm_family_leaf"),
            _shell_fn(lines, "_requested_llama_backend"),
            _shell_fn(lines, "_torch_opens_amd_nodes"),
            # Stubbed: the real one runs nvidia-smi and would answer from the runner's own
            # hardware. False by default, so each arm reads as an AMD-only host.
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
    # The two arms this installer prints a wheel line for are the two the diagnoses belong
    # to, and #10466's host reaches the second by reroute rather than the first.
    ("https://download.pytorch.org/whl/cpu", True),
    ("https://download.pytorch.org/whl/rocm7.0", True),
    ("https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/gfx1151", True),
    # Moving the diagnoses out of the */cpu arm let them reach an index the case has no
    # arm for at all. _has_amd_rocm_gpu returns false on ANY host with a usable NVIDIA
    # GPU, so on a CUDA route its condition is satisfied by every hybrid box with an AMD
    # card on the bus, and someone correctly installing CUDA wheels was told to install
    # the ROCm kernel stack for a card this install does not use.
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

    node.chmod(0o600)  # owner rw: the mode is not what is shutting it
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()), repairs = None)
    assert "owner bits" in out and "already grant read and write" in out
    # The installer wraps one sentence over several substep lines, so match a fragment
    # that cannot straddle the break.
    assert "device cgroup or an LSM" in out
    assert "fix the mode" not in out

    node.chmod(0o060)  # owner has nothing: the mode IS the repair, and still is
    out = _install_sh_hint(str(node), self_uid = str(os.getuid()), repairs = None)
    assert "fix the mode" in out
    assert "already grant read and write" not in out


@pytest.mark.parametrize("mode, bucket", [
    # Codex 4041533207. os.access said the node is shut, so on a node this account owns
    # whose OWNER bits already read rw, the mode is not what denies it: an LSM or a
    # container device policy is, and a mode change or a udev rule repairs nothing. It was
    # filed as an owner-mode problem and prescribed exactly that.
    pytest.param(0o600, "external", id = "owner_bits_that_already_grant_it_are_external"),
    # The control, and the owner-precedence rule itself: POSIX resolves the owner class
    # exclusively once the uid matches, so a node this account owns whose owner bits deny
    # cannot be opened by joining its group however the group bits read. The mode is the
    # repair, and usermod there is a command that succeeds and changes nothing.
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
    # Codex 4049299005. Neither the owner nor in the owning group puts this account in the
    # OTHER class, which POSIX resolves exclusively just as it does the owner one: bits that
    # already grant rw mean the mode is not what denies a node os.access() called shut, so
    # neither a chmod nor a usermod repairs it. It was classified from the group bits and
    # answered with a group to join.
    pytest.param(0o666, False, "external",
                 id = "other_bits_that_already_grant_it_are_external"),
    # The control, or the branch would be "never name a group", which removes a correct
    # repair: with the other bits denying, joining the group really would open the node.
    pytest.param(0o660, False, "joinable",
                 id = "other_bits_that_deny_still_leave_a_group_worth_joining"),
    # And membership outranks the other class, because a member is in the GROUP class and
    # the other bits are never consulted there. `already` names the group rather than
    # filing the node under a bucket that names nothing.
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
    # Not the owner, so the owner class cannot claim the node first. Read BEFORE the patch
    # lands: amd.os is the os module itself, so a lambda calling os.getuid() would call the
    # replacement and recurse until the stack ends.
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

    node.chmod(0o666)  # other rw: no membership and no mode change opens this
    out = _install_sh_hint(str(node), self_uid = _not_us, repairs = None)
    assert "device cgroup or an LSM" in out
    assert "usermod" not in out

    node.chmod(0o660)  # other denies: the group really is the repair
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
    # The fixture can only chgrp to a group this account holds, and a node whose owning group
    # the account already has is filed under `already` rather than `joinable`. These arms are
    # about the derivation, so stand the account outside that group.
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
    # The probe on a file with none: it has to answer False for the common node or every
    # host stops being prescribed for.
    pytest.param((True, None, False), id = "an_ordinary_node_reports_no_acl"),
    # And the failure mode that must not raise: this runs where things are already wrong,
    # so an unreadable path answers False rather than taking the hint down.
    pytest.param((False, None, False), id = "a_path_that_cannot_be_read_reports_no_acl"),
    # os.listxattr returns ``str`` names for a ``str`` path, so the bytes literal this
    # first shipped with could never match one and the whole ACL branch was dead. Stubbed
    # rather than written with setfacl, which is not installed here: the shell twin above
    # skips when it is absent, and a probe whose positive direction is only ever exercised
    # by a skipping test is not exercised at all. That is how this survived a round.
    pytest.param((True, ["security.selinux", "system.posix_acl_access"], True),
                 id = "the_acl_probe_matches_the_name_type_listxattr_returns"),
    # A bytes path yields bytes names, and the caller chooses the path type, so both are
    # accepted. The control for the case above: without it, swapping one literal for the
    # other passes just as well and nothing says which type is actually returned.
    pytest.param((True, [b"system.posix_acl_access"], True),
                 id = "the_acl_probe_also_reads_bytes_names"),
    # The negative control. An ACL is claimed from one exact name, so a node carrying only
    # other attributes stays prescribed for.
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
        # raising=False: os.listxattr does not exist on macOS or Windows, so without it
        # monkeypatch fails on the PATCH rather than the probe, and the case never runs.
        # _has_an_access_acl already answers False through AttributeError there, which is
        # what these cases assert about a host that cannot report an ACL.
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
    # A closed node explains an empty probe only when it is the node the runtime would
    # have used. With renderD129 open the loader had one to enumerate and still reported
    # nothing, so the closed renderD128 is a second finding; returning it alone sends the
    # user after a repair that leaves the probe just as empty. It is still said, because
    # it is still true. Fails before the fix, which returned the hint unconditionally.
    pytest.param(({"/dev/dri/renderD129"}, (_VULKAN_REASON, "/dev/dri/renderD128"), ()),
                 id = "an_open_sibling_node_keeps_the_vulkan_reason"),
    # The control, and the #10466 host itself: with every AMD node closed there is no
    # sibling the loader could have used, so the closed node IS the reason and must not be
    # demoted to a footnote behind a Vulkan sentence that explains nothing.
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
    # ROCm needs /dev/kfd and a render node. With both open on a multi-AMD host, a closed
    # SECOND render node is not why the probe came back empty, and returning the group
    # repair as the sole diagnosis leaves the user fixing something that changes nothing.
    # Fails before the fix, which asked the sibling question of Vulkan builds only.
    pytest.param(({"/dev/kfd", "/dev/dri/renderD129"}, (_SEPARATELY, "/dev/dri/renderD128"), ()),
                 id = "a_closed_sibling_beside_an_open_rocm_path_is_not_the_reason"),
    # The control that keeps it narrow: /dev/kfd has no sibling, so a closed one blocks
    # ROCm outright however many render nodes are open.
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
    # --device /dev/kfd without --device /dev/dri. The one node mapped opens, so nothing
    # is CLOSED, and this answered False -- which made hardware.py suppress the very hint
    # that names the repair, and llama_cpp.py file it as "not why the probe is empty" when
    # the absent render node is exactly why.
    pytest.param(_KFD_OPEN, True, id = "a_missing_render_node_blocks_the_runtime"),
    # The control. Without it the fix could be "always blocks", which suppresses nothing
    # and labels every empty probe a permission problem.
    pytest.param(_BOTH_OPEN, False, id = "a_complete_open_mapping_still_does_not_block"),
])
def test_whether_the_node_layout_blocks_the_runtime(monkeypatch, linux, layout, blocks):
    """The predicate both callers gate their hints on."""
    _nodes(monkeypatch, **layout)
    assert amd.amd_closed_nodes_block_the_runtime() is blocks


def test_the_installer_does_not_dangle_the_group_sentence(tmp_path):
    """When every refused node has an unnamed GID, an ACL, or a mode no group can open,
    _amd_node_repairs names no group on purpose. The installer printed "Add yourself to the"
    above that branch regardless, so the message read as an instruction cut off mid-sentence
    and then contradicted."""
    node = tmp_path / "renderD128"
    node.write_bytes(b"")
    node.chmod(0o600)  # owner-only: no membership opens it, so no group is named
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
    nvidia: bool = False,
) -> str:
    """What install.sh actually PRINTS in the missing-/dev/kfd branch, with ROCm blind.

    `_kernel_stack_hint_runs` above lifts only the guard, so it answers whether the branch
    fires and nothing about which repair it names, which is exactly where the branch was
    wrong. _install_sh_missing_kfd below lifts the guard AND its body, through the closing
    `fi`, so a revert changes the text this returns; this name is the ROCm-sees-nothing
    corner of it, which is the shape the kernel-stack advice is written for.
    """
    return _install_sh_missing_kfd(
        tmp_path,
        topology = topology,
        kfd_present = kfd_present,
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
    """A container created with --device /dev/dri and no --device /dev/kfd sees the host's
    /sys and not its /dev, so the KFD topology names an AMD GPU while the node is absent: the
    driver is loaded, and "install the ROCm kernel stack" leaves HIP as unavailable as before.
    No topology is the case the branch was written for, the driver really is missing, and
    without that arm the fix could be "never mention the kernel stack", which removes a
    correct diagnosis."""
    _says(_kernel_stack_hint_text(tmp_path, topology = topology), contains, absent)


def test_a_container_missing_kfd_is_told_to_map_it_rather_than_reinstall(monkeypatch, linux):
    """The runtime half of the same item. `_amd_nodes_the_runtime_lacks` reports a missing
    node only once the KFD topology names an AMD GPU, and that topology is the amdkfd
    driver's own sysfs, so on every host this sentence can reach the kernel stack is already
    loaded and the advice to install it is unreachable-by-construction wrong."""
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
    amd_smi_sees_it: "bool | None" = None,
    rocm_visible: "bool | None" = None,
    skip_torch: bool = False,
    backend: "str | None" = None,
    nvidia: bool = False,
) -> str:
    """What the installer says when /dev/kfd is absent, for a given pair of probes.

    Lifts the two branches together, through the closing `fi`, because which of them runs
    is the thing under test. A test operator cannot be stubbed, so `[ -e /dev/kfd ]` is
    kept as an `-e` test on a path this case owns: the operator still runs, and what it
    runs against is `kfd_present` rather than whatever the machine happens to have.

    This used to skip on a host that has /dev/kfd, which meant the whole family ran on dev
    boxes with no AMD GPU and vanished on the hardware it is about -- and its mirror at
    `test_the_installer_names_the_userspace_when_the_node_is_already_there` skipped
    everywhere else, so no single host ever ran both arms.
    """
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
))
# fmt: on
def test_what_the_installer_says_when_the_kfd_node_is_absent(tmp_path, kwargs, contains, absent):
    """amd-smi reads the driver over sysfs and libdrm, so it lists the card in a container
    given only --device /dev/dri, where HIP has no /dev/kfd to open; llama_cpp.py's
    _rocm_hip_is_reachable documents that disagreement. Behind _has_amd_rocm_gpu the mapping
    advice was suppressed on the container shape it was written for, and nothing else spoke:
    the render node is open, so no node is closed and none is missing. The two are separate
    branches rather than one with an inner test because they need different evidence -- with
    no KFD topology the driver really is missing, and that diagnosis is still gated on ROCm
    seeing nothing, so a host whose amd-smi answers is not told to install what it has.

    The missing-node report was gated on SKIP_TORCH=false, so a --no-torch run whose GGUF
    bundle is ROCm -- which opens /dev/kfd exactly as torch would -- finished silently in a
    container mapping /dev/dri and not /dev/kfd: nothing closed, nothing missing said, no
    account of why the backend cannot initialise. A Vulkan or CUDA bundle opens no /dev/kfd,
    so its absence explains nothing about them and the run stays silent, and the ordinary
    torch install is unchanged by the gate swap."""
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
    """install.sh runs under `set -e` from line 5, so an unguarded command substitution
    aborts the whole installer when its helper's last command fails. These three shell out
    to stat, awk and tr, none of which is guaranteed on a minimal container, and two of
    them run on every Linux install whether or not anything is closed.

    The answer on such a host has to be "no advice", never "no install": the diagnostic
    exists to explain a GPU that is not working, and an install that dies instead is
    strictly worse than the silence it replaced. Every consumer already reads empty as
    nothing to report.

    Verified live rather than by reading: with tr stubbed to exit 127 the unguarded form
    aborts at the leaf read before installing anything, and the guarded form continues.
    Asserted on the source rather than by running the installer, because reaching these
    lines for real means running an install."""
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
    """The half that makes the test above mean something: the `|| true` is only worth
    asserting if the assignment really does survive. Every named tool is stubbed to exit
    127, which is what a minimal container looks like, and the script must reach its last
    line with status 0."""
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
    """--group-add takes a SINGLE value, so a comma-joined pair is one group name that does
    not exist, and naming only the first leaves the second node shut. docker/run.sh repeats
    the flag and the Python half already emits it repeated; the installer said "the numeric
    GID", singular, for a value it had just printed as "993,994". The singular wording still
    has to survive for one GID, so the fix is not "always say GIDs". groupadd gives the
    numeric owner a NAME and does not put this account in the group, so the bare-host repair
    needs the usermod too -- the installer printed the container flags per GID after the
    earlier fix but still said only "create a group" -- and that usermod needs the same
    session refresh the named-group branch ends with, in its own sentence, since this fixture
    never reaches the named-group branch."""
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
    """ROCr's RvdFilter builds its list from tokens that are "Legal and NOT Terminating", so
    an entry it cannot evaluate ends the list and the devices BEFORE it still survive:
    ROCR_VISIBLE_DEVICES='0,' leaves one device and HIP ordinal 1 hides it, which read as an
    unresolvable list left the HIP mask unmentioned. An enumeration index is Terminating when
    it "maps to a device that has been previously selected", so '0,0' surfaces one device
    rather than two -- counting every token read that as two survivors and called a HIP
    ordinal that hides the only device a valid selector. A token is Illegal when it "can't be
    evaluated into an instance of Device UUID or Enumeration Index" (ROCR-Runtime,
    core/inc/amd_filter_device.h), and an Illegal token ends the list, so an illegal FIRST
    token leaves zero survivors: _post_rocr_device_count already counts it that way, while
    this predicate called every non-index merely unresolved and the message then offered the
    mask as something to check if the group change fails, when it is an independent blocker no
    membership clears. A UUID is a form ROCr accepts, so it may name a device and this host
    cannot match it against an ordinal count; calling that a blocker would invent a fault, and
    without the arm the rule reads as "any non-index blocks"."""
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
    """A conda or locally built CUDA wheel carries no +cu tag, so the label names no vendor
    and a venv that once recorded a ROCm flavor made this read the wheel as AMD-targeted --
    replacing the reinstall advice with group membership on a host whose CUDA build cannot
    use the AMD card however open its nodes are. torch.version.cuda is the build's own
    answer and is stubbed here rather than the predicate that reads it. torch.version.hip
    set means the wheel IS a ROCm build, so the closed node explains it fully and the
    reinstall advice would send the user after the wrong repair -- without that arm the
    rule could be "an untagged label is never AMD"."""
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
    """One shut render node on a multi-AMD host leaves the other GPU fully reachable: HIP has
    /dev/kfd plus an open render node, and the Vulkan loader has the same. Claiming "no GPU
    backend can use the AMD card" there contradicts _explain_empty_gpu_probe, which appends
    this sentence right after saying the closed node is NOT why the probe came back empty.
    With the only render node shut nothing enumerates and the claim is exactly right, so the
    fix cannot be "never claim the card", which is the #10466 message gone; and /dev/kfd with
    an open render node beside it has no sibling of its own, so the narrowing must not weaken
    that arm either -- Vulkan works there, ROCm does not."""
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
    """groupadd gives the numeric owner a NAME. It does not put this account in the group, so
    a user who follows the sentence to the letter still cannot open the node: both halves are
    needed, one pair per GID, since one groupadd names one numeric owner and the second node
    stays shut for anyone who runs only the first command. The plural wording must not be the
    only wording either, or one GID reads as two and stops matching what it printed. usermod
    changes /etc/group, not the groups the running login holds, so the named-group branch has
    always ended "then log out and back in"; the unnamed-GID branch prescribes the same
    usermod and said nothing, so an immediate retry fails and the command reads as the one
    that did not work. `<name>` is not a placeholder in a shell, it is a redirection:
    `groupadd -g 993 <name>` parses as a read from ./name followed by a `>` with no target,
    which bash, dash and sh all reject with a syntax error before groupadd runs. Every
    character here is meant to be pasted, so it must carry no shell metacharacter it does not
    mean. The name is derived from the GID, which by definition here has no group entry."""
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
    """One classified line from the real _amd_node_repairs, for a synthetic stat record.

    The awk program is lifted whole, and only ``stat`` and ``ls`` are stubbed, because the
    cases that matter cannot be built as files: a node whose GID has no group entry needs a
    GID this host does not name, and a root-owned one needs root.

    The record is assembled in install.sh's OWN field order, read off the shipped format
    string, so a case names the field it means instead of counting pipes. A hand-written
    record in a fixed order would keep passing after the format was reordered while
    measuring the wrong fields, which is exactly what the order exists to defend against:
    the group NAME is the one field NSS controls, so it is last, and a name carrying the
    separator can no longer shift the mode, the GID or the uid that the classifier branches
    on.
    """
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
    """/dev/kfd is opened by ROCm and nothing else, the shell twin of the runtime's needs_kfd.
    SKIP_TORCH cannot decide it alone: --no-torch still installs a GGUF bundle, the ROCm one
    opens /dev/kfd exactly as torch would, and which it will be is chosen later in setup.sh,
    so suppressing there hides the #10466 diagnosis from the GGUF users it was written for.
    Vulkan opens the render node, so the scoping takes /dev/kfd and nothing else; dropping the
    whole diagnosis would silence the node that blocks every backend. ROCm torch opens
    /dev/kfd whatever the bundle, so a request cannot scope the torch diagnosis.

    A CPU bundle beside --no-torch opens no GPU node at all, so the group and udev repair
    described a card nothing in the run would touch: only /dev/kfd was filtered, right for
    Vulkan and one node short for CPU, and the narrower rule must survive the wider one.
    REQUESTABLE_BACKENDS is auto/cpu/cuda/rocm/vulkan and a CUDA bundle opens /dev/nvidia* and
    neither AMD node, so that run was handed the same repairs. Both report again once torch is
    installing, which is why the predicate reads BOTH halves.

    An explicitly CPU torch index installs a wheel with no ROCm runtime in it, so with a
    non-ROCm bundle too nothing opens /dev/kfd; the early return declared otherwise purely
    because torch was being installed at all. "cpu" is deliberately kept in the diagnosis
    route one layer up, since the GGUF bundle is chosen later and may be the ROCm one, and a
    ROCm index opens /dev/kfd whatever the bundle, so the rule is not "any explicit non-ROCm
    backend suppresses".

    The bundle selector normalizes with `awk '{$1=$1}'` -- trim and collapse, never delete --
    then REJECTS what it does not recognise and falls back to automatic selection, which may
    install ROCm. Deleting internal whitespace made "vul kan" match, so the run went quiet
    about the very node selection needs, while a padded upper-case value does name Vulkan and
    must still be scoped. Asserted through the KFD scope rather than the whole diagnosis,
    because the wider predicate matches cpu|cuda only and lets every Vulkan spelling through
    either way, which made an earlier version of this test pass on both forms."""
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
    """_torch_reports_a_hip_runtime answers an import failure from torch/version.py on disk;
    its mirror returned False there instead, so the untagged-CUDA clearing was inert on the
    one path it exists for. A stale recorded ROCm flavor then spoke for a CUDA wheel, and the
    closed node replaced the reinstall guidance with group membership that cannot make that
    wheel use the card. The hip reading leads because a ROCm build records hip and may record
    cuda besides, so ordering the two the other way round would send every AMD host whose
    torch fails to import after a reinstall it does not need."""
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
    """The sentence lists every ACL-carrying node and then ran getfacl on the first one
    alone, so a user following it read one node's grant and was left with the second blocker
    undiagnosed. ROCm needs both nodes and their ACLs need not agree. The single-node arm is
    there so the fix cannot have introduced a stray separator into the common case."""
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
    """The gate globbed the raw URL for */rocm* and */gfx*, which matches exactly the custom
    pins _is_pip_rocm_family_leaf exists to reject -- a mirror named for an arch, or a private
    ROCm build -- so an install not on a published ROCm route was told to repair the AMD
    kernel stack. Classifying the leaf reuses that rejection, and it stays narrow:
    repo.radeon.com's arch leaf is a real ROCm route.

    --no-torch installs no wheel, so the index resolved above this gate describes nothing that
    will run; on a hybrid host it is the CUDA one, and reading it as the route silenced every
    node diagnosis for a run whose Vulkan bundle opens the very render node they are about. A
    CPU llama.cpp bundle opens no AMD node at all, so those runs stay off and the fix is not
    "--no-torch always diagnoses". A run installing CUDA wheels stays off the route as long as
    its bundle opens no AMD node either, so the fix is scoped to the run that installs nothing
    rather than reopening the arm above; an explicit rocm or vulkan request is the one case
    that does reopen it, and it has its own arm below the override. Fails before the fix,
    which read the unused index."""
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
    """The mapping advice named both nodes whatever the caller was. Vulkan never opens
    /dev/kfd -- which is why _amd_nodes_the_runtime_lacks already excludes it under
    needs_kfd = False -- so this handed a container another host device for nothing. HIP
    opens both, so the fix must not narrow the advice for the caller it was written for."""
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
    """A container can map /dev/dri and still mask the sysfs entry that names the vendor.
    Reading that unknown as "no AMD render node exists" told the user to recreate the
    container with --device /dev/dri, which that shape has already done. The fix is "unknown
    reads as present" rather than "always present": a host whose glob finds nothing has
    nothing unreadable either, and the sentence #10466 needs must still be printed."""
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
    """Check an empty-probe reason.

    ``credited`` is whether another vendor's open node is still a complete path, which keeps
    the Vulkan probe sentence in front of the reason; False means the node hint replaced it,
    and None means this build never asks the question.
    """
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
    """A Vulkan-only build enumerates any vendor, so an open Intel or NVIDIA render node is
    a complete path for it: the closed AMD node cannot then be the whole reason the probe
    came back empty, and returning it alone dropped the finding that was. With no open node
    of any vendor the closed AMD one IS the reason, and the hint must still replace the bare
    probe sentence. HIP needs /dev/kfd and an AMD render node, which no other vendor's node
    substitutes for, so the question is not even asked for it."""
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
    # The group is whatever owns a tmp_path file on the runner, so the account is what is
    # asserted -- naming a group here would be asserting about the runner.
    assert re.search(r"usermod -a -G \S+ ada", out)
    assert " root" not in out


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


# fmt: off
@pytest.mark.parametrize("vendor, verdict", _cases(
    ("a_vendor_the_installer_cannot_read_is_not_absent", None, "PRESENT"),
    ("a_vendor_that_names_another_is_still_absent", "0x10de\n", "ABSENT"),
))
# fmt: on
def test_what_the_installer_makes_of_a_render_node_vendor(tmp_path, vendor, verdict):
    """The shell twin of the Python rule, which round sixteen fixed on one side only. A
    container mapping /dev/dri while denying its sysfs attributes has a render node; calling
    that absence told the user to recreate the container with the device it already has. The
    rule is "unknown", not "any node": a readable vendor that is not AMD is a real answer, and
    treating it as presence would claim an AMD card on an NVIDIA-only host. Only the two path
    ROOTS are substituted, so the logic under test is the shipped one, and a sysfs directory
    with no vendor file in it is what a container denying the attribute looks like from here.
    """
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
    """HIP_VISIBLE_DEVICES=0 narrows the runtime to one GPU, and nothing here maps a render node
    back to the index it was selected by, so an OPEN sibling may belong to the GPU the mask
    excludes. Reading it as an alternative suppressed the repair for the node the run will
    actually use and sent the user to reinstall a runtime instead. With no selector the runtime
    is free to use the open node, so the closed one is a second finding rather than the cause --
    the rule cannot be "a closed node always blocks", which the open sibling arm was added to
    stop -- and an exported but empty variable narrows nothing, so reading it as a selector
    would keep the repair on every host that merely exports the name.

    GPU_DEVICE_ORDINAL is ROCm's fourth visibility variable, already modelled in
    test_amd_smi_inventory_matches_hip.py and llama_cpp.py's own selector check; omitting it
    left one of the four narrowing the runtime while the sibling was still credited. HIP's
    selectors are HIP's, though: Vulkan reads none of them -- which is what needs_kfd is for --
    so a Vulkan caller on a masked host may still use the open sibling, and returning the
    permission hint as the sole cause would send an empty Vulkan probe after a group change that
    cannot fix it. The HIP caller on that host must still block, or the rule becomes "ignore
    selectors"; its host keeps /dev/kfd present AND open, or the answer comes back True on the
    missing-node branch and says nothing about the mask, which is how that arm first passed."""
    _nodes(monkeypatch, **host)
    for _name, _value in env.items():
        monkeypatch.setenv(_name, _value)
    # _no_inherited_gpu_mask has already cleared every mask variable a case does not set.
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
    """Membership in docker is root by another route -- a container started with the host
    filesystem mounted -- so prescribing it to open a GPU node is a privilege escalation dressed
    as a device repair, exactly as for wheel. A minimal container can own the node root:root and
    carry no group database entry for gid 0; the name lookup raises there, and filing that as an
    ordinary unnamed GID produced `sudo groupadd -g 0 amdgpu0` plus a usermod into the ROOT
    group, a grant far beyond the GPU and exactly what the privileged branch below the lookup
    exists to refuse. gid 0 is the root group whether or not the database can name it, and the
    rule is keyed on gid 0 rather than on the lookup failing because an unnamed NON-privileged
    GID is the documented container case and must still get its groupadd pair.

    CI commonly runs as root, and the fakes in this file give their nodes the root owner a real
    device node has. POSIX resolves the owner class exclusively once the uid matches, so a
    harness that let the two coincide would answer every group test with "this account owns it"
    and assert nothing about the classification under test. The last case fails before that fix
    with os.getuid patched to 0, which is what a root runner is."""
    if runner_uid is not None:
        monkeypatch.setattr(amd.os, "getuid", lambda: runner_uid)
    _stat_nodes(monkeypatch, {"/dev/kfd": (gid, 0o660, 0)}, names)
    assert amd._groups_that_own(["/dev/kfd"]) == expected


# fmt: off
@pytest.mark.parametrize("name, expected", _cases(
    # usermod -G takes a COMMA-SEPARATED list, so this is two groups to it and the
    # account lands in sudo. _PRIVILEGED_GROUPS compares whole names and never matches.
    ("a_name_carrying_the_usermod_separator", "render,sudo", _buckets(unnamed = [993])),
    # The shell twin parses `stat -c` output with awk -F'|', where such a name shifts
    # every field after it. Refused in both halves for one rule rather than two.
    ("a_name_carrying_the_field_separator", "render|x", _buckets(unnamed = [993])),
    ("a_name_carrying_a_shell_metacharacter", "render;id", _buckets(unnamed = [993])),
    ("a_name_that_is_only_whitespace", "  ", _buckets(unnamed = [993])),
    # The controls, without which this collapses into "never prescribe a group" and
    # deletes what #10466 asked for. A Samba machine account carries the trailing $,
    # and a distribution group may carry a dot or a dash.
    ("an_ordinary_group_name", "render", _buckets(joinable = ["render"])),
    ("a_samba_machine_account", "host$", _buckets(joinable = ["host$"])),
    ("a_dotted_distribution_group", "gpu.users-1", _buckets(joinable = ["gpu.users-1"])),
))
# fmt: on
def test_a_group_name_usermod_would_read_as_structure_is_not_prescribed(
    monkeypatch, name, expected
):
    """Shell quoting is the wrong layer for this, which is why it was not caught by it.

    `_shell_word` hands the shell ONE argument, correctly; the splitting that grants sudo
    happens inside usermod afterwards, on its own comma syntax. So a name outside the
    portable groupadd(8) charset is treated as no name at all and the node is reported by
    its GID, which is what `--group-add` takes anyway, so nothing is lost.

    Reverting the check puts 'render,sudo' back in the joinable bucket and the printed
    command back to `usermod -a -G render,sudo`."""
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {993: name})
    assert amd._groups_that_own(["/dev/kfd"]) == expected


def test_the_printed_command_never_names_a_group_it_did_not_mean(monkeypatch, linux):
    """End to end through the message, since the bucket above is only half the story: what
    matters is what the user pastes. The GID repair is what this host gets instead."""
    _nodes(monkeypatch, present = ["/dev/kfd"], openable = set())
    _stat_nodes(monkeypatch, {"/dev/kfd": (993, 0o660, 0)}, {993: "render,sudo"})
    _the_real_group_derivation(monkeypatch)
    hint = amd.amd_node_permission_hint()
    # The name is never handed to usermod, in any form. Not a bare "sudo not in hint": the
    # GID repair legitimately says `sudo groupadd`, and an assertion that cannot tell the
    # two apart would have to be weakened later rather than tightened.
    assert "render,sudo" not in hint
    assert "-G render" not in hint
    # What this host gets instead: the GID, which is what --group-add takes anyway.
    assert "--group-add 993" in hint


# fmt: off
@pytest.mark.parametrize("record, classified", _cases(
    ("the_installer_also_refuses_an_unnamed_gid_zero", {"group": "UNKNOWN", "gid": "0"},
     "privileged:root"),
    ("and_still_names_an_ordinary_unnamed_gid", {"group": "UNKNOWN", "gid": "993"}, "gid:993"),
    ("and_keeps_the_name_of_a_named_root_group", {"group": "wheel", "gid": "0"},
     "privileged:wheel"),
    # usermod -G takes a comma-separated list, so a group genuinely named "render,sudo"
    # is TWO groups to it and the alternation above, which compares whole names, walks
    # straight past it. Quoting is the wrong layer: the split happens inside usermod,
    # after the shell has handed it one argument. Reported by GID instead.
    ("a_name_carrying_the_usermod_separator_is_not_prescribed",
     {"group": "render,sudo", "gid": "993"}, "gid:993"),
    # And the field-shift the format order defends against: with the name anywhere but
    # last, a pipe inside it moved the GID and uid the classifier branches on.
    ("a_name_carrying_the_field_separator_is_not_prescribed", {"group": "render|x", "gid": "993"},
     "gid:993"),
    # The control for both, so this cannot collapse into "never prescribe a group".
    ("an_ordinary_group_name_is_still_prescribed", {"group": "render", "gid": "993"},
     "join:render"),
))
# fmt: on
def test_how_the_installer_classifies_a_nodes_owner(record, classified):
    """The shell twin, run through the real classifier rather than a stubbed repair line.
    `stat -c %G` prints UNKNOWN for a gid the group database cannot name, and the unnamed
    test ran first, so a root-owned node in a minimal container came back as `gid:0` and the
    installer printed groupadd -g 0 plus a usermod into the root group. The same shape one
    GID up is the documented container case and must still come back as a GID to create a
    group for, and gid 0 WITH an entry keeps the name it has, so the reordering did not turn
    every privileged node into the literal word root."""
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
    """repo.radeon.com publishes rocm-rel-6.4, rocm-rel-6.5.0, rocm-rel-7.2.1 -- digits and
    dots after the prefix, nothing else. A pin that merely starts with it is a mirror, and it
    may serve wheels that open no AMD node, so it gets the same anchoring the sibling
    rocm[0-9]* arm already applies to rocm7.2-private. The anchor is on the character class
    rather than a fixed version shape, since all six rocm-rel leaves this repository names
    must keep routing, two and three components alike. The gfx family was deliberately left
    alone: AMD's own indexes are gfx110X-all, gfx120X-all and gfx103X-all, so a suffix there
    is the naming convention, and anchoring gfx the same way would drop routes we ship."""
    assert _install_sh_diag_route(leaf) == ("true" if routes else "false"), leaf


def test_the_passwd_stub_answers_a_positional_read(monkeypatch):
    """getpass.getuser() reads pwd.getpwuid(os.getuid())[0] once none of LOGNAME, USER,
    LNAME or USERNAME is set, and pytest calls it while building tmp_path. A stub that only
    carries pw_name raises TypeError there, which pytest does not catch, so the whole suite
    would die in fixture setup on a runner with no username in its environment rather than
    run. Reproduced by clearing all four, which is the only thing that makes it reachable."""
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
    """clr's list terminates at the first token it cannot use, and a token does not have to
    look numeric to be unusable: rocdevice.cpp takes `index = atoi(str_id)` and rejects it
    unless `str_id` is that index written back out, so HIP_VISIBLE_DEVICES=garbage leaves zero
    agents exactly as -1 does and joining the group leaves the probe as empty as it was. That
    arm fails before the fix, which asked only whether the first token was a digit and let
    everything else through as "not a filter".

    The precedence test in clr is on the first BYTE of the value, not on whether the variable
    exists: its flag defaults to the empty string, so an empty HIP mask reads exactly like an
    unset one and the CUDA value below it selects devices. That value names a device in the
    second case, so nothing is hidden and no mask is a second blocker; before the fix an empty
    HIP mask won the chain and was reported as hiding every device. Deferring to CUDA is only
    right if CUDA is then judged on its own merits, so a CUDA value that names no device must
    still be reported, or the fix becomes "an empty HIP mask silences the whole chain"."""
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
    """The control for the helper itself: the CPU node every KFD topology carries is excluded
    and the GPU node is counted. One entry temporarily unreadable on a two-GPU host used to
    answer 1 rather than unknown, and an understated bound is what calls a valid selector a
    blocker: with a count of 1, HIP_VISIBLE_DEVICES=1 reads as hiding every device and the user
    is told to clear a mask that hides nothing. Unknown bounds nothing, which is the documented
    contract and what _amd_render_node_exists already does with an unreadable vendor."""
    _kfd_topology(monkeypatch, entries)
    assert amd.amd_kfd_gpu_node_count() == count


# fmt: off
@pytest.mark.parametrize("render_open, contains, absent", _cases(
    # The repair is unchanged: these nodes are still shut and membership opens them.
    ("the_claim_is_scoped_when_a_sibling_node_is_open", True,
     ["another AMD", "render node on this host is open", "usermod -a -G"], []),
    ("and_covers_every_backend_when_none_is", False,
     ["Every backend needs them, ROCm and Vulkan alike."], ["render node on this host is open"]),
))
# fmt: on
def test_how_wide_a_claim_the_installer_makes(tmp_path, render_open, contains, absent):
    """The Python half already says an open sibling means the closed nodes stop the card
    rather than the host, and the installer claimed every backend was blocked regardless --
    on a host where a Vulkan run is working through the open node as the user reads it. With
    no AMD node open anywhere the closed set does block every backend, so the fix must not
    soften the claim into always saying maybe.

    Fails before the fix, which printed the unconditional claim."""
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
    """VK_DRIVER_FILES REPLACES the loader's driver search rather than adding to it, and the
    probe child inherits it, so a list naming AMD alone means the loader never opened the
    other vendor's driver; crediting its node suppressed the closed-node hint on a run that
    had no other path, which hides a real repair. VK_ICD_FILENAMES is the deprecated name for
    the same replacing list, honoured when VK_DRIVER_FILES is unset. VK_ADD_DRIVER_FILES ADDS
    to the standard search, so every driver the loader would have found is still found and the
    other vendor's open node is still a complete path: without that arm the fix could be "any
    VK_ variable suppresses", turning the finding off for a host that set the one variable
    that changes nothing about which vendors load. The loader treats an empty value as unset,
    so a blank VK_DRIVER_FILES must not suppress either. Fails before the fix, which asked
    only whether the node was open."""
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
    """A list pinned to an Intel or NVIDIA ICD leaves the loader unable to use the AMD card at
    all, so its closed node cannot be why the probe was empty, and the group repair alone
    sends the user after a change that cannot help. Only an AMD-only list is evidence the
    other vendor is unreachable, and a list is AMD-only or it is not: one non-AMD entry leaves
    that vendor loadable and its open node a complete path. A directory, or a name this does
    not recognise, is evidence of nothing, and the guard fires on positive evidence alone, so
    an unreadable list leaves the finding as it is rather than suppressing on a guess.

    A registration with no library behind it loads nothing, so a list holding only that leaves
    the loader with no driver at all -- the filename says AMD; the manifest says nothing is
    there -- as does a manifest missing outright, the shape a stale VK_DRIVER_FILES has after
    a driver is uninstalled. VK_LOADER_DRIVERS_DISABLE applies to a forced list too, so naming
    AMD and then disabling it leaves the loader with nothing; that manifest is entirely valid
    and only the filter makes it unloadable, which separates it from the two arms above. The
    last case is the control for all of them: present, parses, points at a library that exists
    and survives the filters IS an AMD-only list, and the other vendor's node is then
    unreachable. Without it the fix could be "never suppress". Fails before the fix, which
    read any non-empty list as AMD's, and the name alone as a driver."""
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
    """A mask that hides every accelerator is the configuration working, so the whole
    chat-only classification is cancelled and no message -- node repair included -- is
    built. This is the first of the two gates that keep a blocking selector away from the
    node hint, and it sits well above the code that returns it."""
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
    """The second gate, for the hybrid host the first one does not cover: an NVIDIA card still
    raises the verdict, but the masked AMD card is dropped from the inventory that establishes
    it, so "amd" never reaches CHAT_ONLY_MISMATCH_VENDORS and the node hint is not even
    computed. An empty active CUDA mask is the same story, since HIP reads that variable too."""
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
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
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
    """HIP_VISIBLE_DEVICES=0,1 on a two-GPU host selects the whole host, so the open sibling
    is still a complete ROCm path and the closed node is a second finding rather than the
    cause; reading any selector as a narrowing handed that host the group repair in place of
    the driver diagnosis, and with nothing set the sibling was always evidence. The rule
    exists because nothing here maps a render node back to the index a selector chose it by,
    so under a real narrowing the open node may be the excluded GPU's. A UUID names a device
    by identity and an unreadable GPU count answers nothing: both leave the selector unmapped,
    and unmapped goes on meaning narrowed.

    ROCr's filter terminates on a token naming a device it has already selected, so
    ROCR_VISIBLE_DEVICES=0,0,1 surfaces ONE GPU on a two-GPU host; counting distinct ordinals
    read that as the whole host and kept the sibling as evidence for a runtime that can no
    longer reach it. The rule is per layer rather than global, because clr's parser stops only
    on a token that is not its own index written back out, so it accepts the repeat and both
    GPUs survive -- a global repeat rule would hand that host the group repair instead of the
    driver diagnosis. Fails before the fix, which asked only whether a selector was set, and
    accumulated a set with no repeat rule."""
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
    """An unset or `auto` backend is resolved by _linux_published_attempts, which takes the
    CUDA bundle under `if host.has_usable_nvidia:` and reaches ROCm only in the `elif
    host.has_rocm` below it. So on a hybrid box with a usable NVIDIA GPU neither a CPU torch
    nor the automatic bundle opens an AMD node, and the AMD-evidence gates cannot tell -- the
    card is there and its nodes are shut, they are just unused. An explicit rocm request IS a
    decision and the resolver honours it, so the nodes that bundle opens are the user's
    problem to repair whatever else is on the bus; with no NVIDIA GPU the automatic bundle may
    well be the ROCm one, which is the #10466 host, so nothing about `auto` is a reason to go
    quiet; and a run installing ROCm wheels opens AMD nodes whatever bundle is chosen later,
    so the bundle question is never reached. Fails before the fix, which read an unresolved
    `auto` as "may open"."""
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
    """The shell half of the same item. `id -un` fails outright for a uid the passwd database
    does not know, and the old fallback then named the inherited $USER -- which in a container
    commonly still says root, so the command would succeed against an identity nothing is
    running as and leave the node shut. With a passwd entry the command is the repair, exactly
    as before. The unnamed-GID branch prints a groupadd AND a usermod, and the second needs
    the same account, so with no passwd entry the container flag is the whole repair. The
    last two cases are the control for the ordinary host: a node whose group grants read and
    write gets the sentence and the command, in one piece. Fails before the fix, which fell
    back to ${USER}."""
    node, _group = _a_node_a_membership_would_open(tmp_path)
    _says(_install_sh_hint(str(node), **kwargs), contains, absent)


def _assert_amd_only_loader(reason: str, amd_only: bool) -> None:
    """The two verdicts a loader configuration reaches for a host with the AMD node shut.

    AMD-only means the other vendor's open node is not a path this binary has, so the node
    repair stands; anything else leaves the empty probe explained without it.
    """
    if amd_only:
        assert "the Vulkan probe reported no device" not in reason
        assert "usermod" in reason
    else:
        assert reason.startswith("the Vulkan probe reported no device")


# Every loader override an arm below may have to clear before stating its own.
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
    """Whether a closed render node blocks the runtime, for each shape of selector list.

    The HIP layer reads HIP_VISIBLE_DEVICES when non-empty and CUDA_VISIBLE_DEVICES only
    otherwise, so a CUDA value under a HIP one narrows nothing while the same value alone IS
    the selector, the precedence _hip_layer_var already applies; reading all four side by side
    let the shadowed one discard an open sibling the runtime can still reach. clr BREAKS out
    of its parse on a token it cannot map, having already pushed what it accepted, so 0,1,-1
    leaves both GPUs visible and a word ends the list identically (atoi returns 0, which is
    not the token written back) -- read as narrowed, those hosts got the group repair in place
    of the driver diagnosis they need. The prefix is judged rather than trusted because 0,-1
    accepts ONE of two and really is narrowed. ROCr's RvdFilter ends the same way on an
    already-selected token.

    Fails before the fix, which asked every name independently and returned False on the
    unmappable token."""
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
    """VK_LOADER_DRIVERS_SELECT applies to whatever the loader would load, forced list or
    search, so a host with no list at all can still be pinned to AMD alone; reading the two
    force-list variables only left it crediting the other vendor's open node to a binary whose
    loader never opens that vendor's driver. Unfiltered the loader loads both manifests, so
    that node IS a path and the closed AMD one cannot be why the probe was empty -- otherwise
    the rule becomes "a search always means AMD only", which suppresses the finding on every
    host.

    Fails before the fix, which asked what a forced list was named rather than what the
    loader would load."""
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
    """setup.sh warns "Ignoring UNSLOTH_LLAMA_CPP_BACKEND=..." for anything outside its list
    and the installer normalises it to auto, so a typo installs the automatically chosen
    bundle -- on a hybrid box CUDA, which opens no AMD node. Listing the two spellings of "no
    decision" instead of the values that ARE decisions let a rejected value pose as one. An
    explicit rocm request is honoured by the resolver, so its bundle opens /dev/kfd on a
    hybrid box exactly as on an AMD-only one, and reading every hybrid host as CUDA would
    silence #10466 for the users who asked for ROCm. The automatic route is silent only where
    it resolves away from AMD: with no NVIDIA card it resolves to ROCm and reports.

    Fails before the fix, which matched "" and auto alone."""
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
            # Every one of these is lifted rather than stubbed, for the reasons
            # _install_sh_hint gives: a harness that omits one measures a missing function
            # rather than a rule, and _shell_quote left out makes every interpolated name
            # come back EMPTY.
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
    """_has_usable_nvidia_gpu shells out to a bounded `nvidia-smi -L` on every call and is not
    memoized, while the two scope predicates are consulted at every diagnosis; nothing it
    reads changes within a run, so an NVIDIA-less host was paying a subprocess per gate. A
    decision the resolver honours settles the question without asking about the other vendor's
    hardware at all, so the memo is not merely cheaper there, it is unreached.

    Fails before the fix, which probed on every call."""
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
    """The route was derived from the torch index alone, so a CUDA-pinned index asked for the
    ROCm bundle read as a CUDA route and silenced all three diagnoses -- for a run whose bundle
    opens the very nodes they are about. The SKIP_TORCH override below the case could not
    catch it, since it only runs when no wheel is installed at all. The new arm only ever
    turns the route ON, so a cpu request and no request at all stay with the two scope
    predicates, which is where a bundle that opens no AMD node belongs.

    Fails before the fix, which had no arm for the backend request here."""
    backend, routed = case
    assert _diag_route("https://download.pytorch.org/whl/cu128", backend = backend) is routed


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param((True, False), id = "a_hybrid_host_rocm_can_see"),
    pytest.param((False, True), id = "a_hybrid_host_rocm_cannot_see"),
])
# fmt: on
def test_the_kernel_stack_hint_on_a_hybrid_rocm_host(tmp_path, case):
    """_has_amd_rocm_gpu opens with `if _has_usable_nvidia_gpu; then return 1`, right where it
    is choosing a torch index and wrong here: this branch has already established the run
    opens AMD nodes, so the veto made rocminfo's answer unreachable and a healthy hybrid ROCm
    host was told to install the ROCm kernel stack it already has. An AMD card on the bus that
    ROCm genuinely cannot see is what the sentence is for, so dropping the veto must not have
    dropped the finding.

    Fails before the fix, which read the wrapped probe."""
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
    """VK_ADD_DRIVER_FILES is read FIRST and then the search, and it may name a manifest no
    search directory holds; leaving it out made a host whose only foreign driver arrived that
    way look AMD-only, and that vendor's open node then stopped excusing the closed AMD one.
    The loader ignores the additive list entirely when VK_DRIVER_FILES or VK_ICD_FILENAMES is
    set, so a foreign driver added beside a forced AMD list is not a path this binary has; and
    reading the list must not make every host that has one look mixed-vendor, so an AMD
    manifest added to an AMD-only search is still AMD alone.

    Fails before the fix, which read the search alone."""
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
    """Which of the two shut nodes reaches the closed list, for each shape of evidence.

    A container can map the render node and hide the sysfs entry naming its vendor. Requiring
    a confirmed AMD vendor dropped the node, so nothing was CLOSED -- while
    _amd_render_node_exists reads the same unknown as PRESENT and withdraws the missing-node
    sentence too, leaving #10466's own shape with no diagnosis at all. KFD is the independent
    evidence that keeps this off every NVIDIA host, since its topology reports 0x10DE there,
    and a vendor that CAN be read and is not AMD is positive evidence the node is somebody
    else's even where the topology does report an AMD GPU.

    The same distinction one node over: a container can map /dev/kfd and hide /sys/class/kfd,
    and the guard read that as "not an AMD GPU". DRM says otherwise and independently, so
    /dev/kfd was dropped from a list the render node stayed in -- and where the two carry
    different owning groups the hint then named a membership that leaves KFD shut and ROCm
    with nothing to open. A topology that READS and names no AMD GPU is positive evidence
    rather than absence, so the fallback must not fire there; and it is vendor-CONFIRMED,
    never assumed, so an unreadable render vendor cannot stand in for a topology this could
    not read either.

    Fails before the fix, which asked for a vendor the container had hidden."""
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
    """Which closed nodes survive the KFD filter, for a wheel index and a bundle request.

    Only a ROCm wheel opens /dev/kfd, and reading "any index that is not cpu" as one that does
    was harmless only while _amd_node_diag_route dropped every non-ROCm index; the
    explicit-backend arm keeps the route for a vulkan request, so a CUDA wheel beside a Vulkan
    bundle reached the KFD scope and told a healthy hybrid host to repair a node neither opens.
    That bundle DOES open a render node, so scoping /dev/kfd must not take the diagnosis the
    explicit-backend arm exists to reach (the round-twenty-four fix), and the bundle is the
    other half of the question, so an explicit rocm request keeps /dev/kfd even though the CUDA
    wheel never touches it. On the route this installer reroutes #10466's own host to,
    repo.radeon.com's leaf is rocm-rel-X.Y, not a pip family: a classifier knowing only the pip
    spellings silences the diagnosis for the host it was written for.

    Fails before the fix, which excluded the cpu leaf alone."""
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
    """A distribution registers the 32-bit ICD beside the 64-bit one, and a 64-bit
    llama-server cannot load it: a host left with only that has no driver at all, so the other
    vendor's open node is not shut out by an AMD-only loader -- there is no loader. The pair is
    the layout every multilib host has, one AMD driver registered twice, so the loader still
    loads AMD and only AMD; otherwise the rule becomes "any i686 name means no AMD driver",
    which is the normal case. Asked of every vendor rather than of AMD alone, because treating
    a 32-bit manifest as simply "not AMD" would make nvidia_icd.i686.json evidence that this
    loader reaches another vendor's card, when a 64-bit binary can load neither it nor the
    card behind it. The same foreign driver in its loadable build IS such a path, so the
    exclusion cannot be "ignore other vendors".

    Fails before the fix, which read the name as AMD and suppressed."""
    manifests, amd_only = case
    paths = [
        _icd_manifest(tmp_path, _name, **({"library": _library} if _library else {}))
        for _name, _library in manifests
    ]
    reason = _vulkan_reason_under_icd_list(monkeypatch, os.pathsep.join(paths))
    _assert_amd_only_loader(reason, amd_only)


def test_the_32_bit_rule_matches_the_installers(tmp_path):
    """The installer decides whether an AMD driver is installed from the same filenames, and
    a host where the two disagree gets one answer from the Vulkan route and another from this
    diagnosis. Only the needles are shared: the installer asks it of AMD names alone, this of
    every vendor, which is a difference in question not in rule."""
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
    """The empty-probe reason for a Vulkan build with the AMD node shut, so the answer is the
    node repair and the question is what is appended to it.

    ``sibling_open`` is another vendor's node being open, which demotes the node repair to a
    second finding; the default is no such sibling.
    """
    _nodes(monkeypatch, present = _AMD_NODES, openable = set())
    monkeypatch.setattr(amd, "a_non_amd_render_node_is_open", lambda: sibling_open)
    for _var in _VK_OVERRIDE_VARS:
        monkeypatch.delenv(_var, raising = False)
    if search_dirs is not None:
        monkeypatch.setattr(amd, "_vulkan_icd_search_dirs", lambda: list(search_dirs))
    if value is not None:
        monkeypatch.setenv("VK_DRIVER_FILES", value)
    # After the clearing above, since that is what makes an arm about ONE override able to
    # set it. `value` reaches the loader through VK_DRIVER_FILES, so an arm that is about a
    # different override passes None and names its search dirs instead.
    for _name, _value in (env or {}).items():
        monkeypatch.setenv(_name, _value)
    _ggml(monkeypatch, {"vulkan"})
    return _empty_probe()


def test_the_no_driver_diagnosis_stays_primary_when_a_sibling_node_is_open(
    monkeypatch, linux, tmp_path
):
    """Codex 4040623258. With another vendor's node open, the closed AMD node is a SECOND
    finding: the runtime had a complete path and enumerated nothing anyway. The loader having
    no loadable driver is not second, it is sufficient on its own, and folding it into
    node_hint demoted it along with the permission text to "not why the probe is empty" -- the
    one sentence that always holds, filed under the one that does not."""
    _icd_manifest(tmp_path, "radeon_icd.json", present = False)
    reason = (
        _vulkan_node_hint_under_icd_list(
            monkeypatch, None, search_dirs = [str(tmp_path)], sibling_open = True
        )
        or ""
    )
    assert "no driver it can load" in reason
    assert "reinstall the Vulkan driver" in reason
    # The demotion applies to the permission finding only, so the loader sentence must come
    # BEFORE it rather than inside it.
    _demoted = reason.index("Separately, and not why the probe is empty")
    assert reason.index("no driver it can load") < _demoted
    # And "also" is dropped, since no node repair precedes it here.
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
    """Two things are wrong at once and the node repair clears only one of them: with every
    registered manifest unloadable the probe stays empty however the node is owned, so a user
    who runs the usermod and nothing else is left exactly where they were. Fail-closed in the
    other direction, which is the one that matters here: an enumeration that found NO
    manifests means this could not read the loader's configuration, not that the loader has
    nothing, and emitting the sentence there sends a working host after a driver reinstall it
    does not need on top of a repair it does. Manifests found AND loadable is the ordinary
    host, where the closed node is the whole story and the sentence must stay off."""
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
    """os.access already said the node is shut, so a group this account is ALREADY in is not
    what is denying it: usermod exits 0 and leaves the node exactly as closed, and the denial
    is outside the file mode -- a container device cgroup, or an LSM. A group it does not hold
    IS the repair, and is what #10466 asked for, so the rule cannot be "never prescribe a
    group". The same one branch over: a numeric owner with no group-database entry is the
    container shape, where the repair is groupadd plus --group-add, itself the same empty
    promise once the account carries the gid -- which is why the already-held question is
    asked BEFORE the naming branch.

    Fails before the fix, which read the group bits and prescribed the membership."""
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
    # The command, not the word: the sentence itself says usermod would change nothing.
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
    """The shell half of the rule above, read from `id -G`; without it the two halves disagree
    on the same host and the installer prints the usermod the Python side just declined to.
    The control is an account outside the group, where the command comes back, so the shell
    rule cannot be "never prescribe". The trap in reading `id -G` as text is that the list is
    space separated, so an unpadded match makes gid 100 look held by an account that is only
    in 1001: the last case is owned by a group the account does NOT have, whose gid is a
    prefix of two it does."""
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
    """An ICD manifest naming its library by soname alone, and the path to it.

    The form NVIDIA registers under, and the one _icd_manifest cannot produce: that helper
    writes an absolute path so it can put the library on disk, which is exactly the case this
    is not.
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
    """A manifest may name its library by soname and leave the loader to find it, so the
    package can be removed and leave the registration behind; trusting the name counted a
    driver that is not there and withheld the reinstall half of the repair. The control is the
    same manifest with the library where ld.so would find it, since a bare name IS the
    ordinary case for every vendor that registers one. The cache is consulted because a
    versioned soname such as libGLX_nvidia.so.0 lives wherever ld.so.conf put it, which need
    not be a directory this enumerates. And it fails closed: musl ships no ldconfig -p and a
    minimal container may ship none at all, so "not found" there is ignorance rather than
    absence, and calling a live driver stale would promote the AMD node as the sole cause on a
    host whose other vendor really does have a path.

    Fails before the fix, which returned True for every bare name."""
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
    """The fixture decides what "no mask" means for this whole suite, so a selector the
    production rule reads and the fixture does not clear is answered by the runner's own
    environment: a ROCm or OpenCL host exports GPU_DEVICE_ORDINAL, and the no-mask controls
    would then quietly take the narrowing branch they exist to rule out. Read out of the
    rule's own source rather than restated, so adding a fifth selector fails here instead of
    drifting."""
    _source = inspect.getsource(amd._a_per_gpu_mask_narrows_the_runtime)
    _read = set(re.findall(r'"([A-Z_]+(?:VISIBLE_DEVICES|DEVICE_ORDINAL))"', _source))
    assert _read, "the rule named no selector, so this test proves nothing"
    assert _read <= set(_GPU_MASK_VARS), _read - set(_GPU_MASK_VARS)


def test_the_installer_chains_the_groupadd_pair(tmp_path):
    """groupadd and usermod are a pair, and the name is generated from the GID, which says
    nothing about whether that NAME is free. Printed as two separate lines, a host that
    already has an amdgpu993 group at another GID fails the groupadd and then SUCCEEDS the
    usermod against the wrong group, leaving the node shut having reported success. Fails
    before the fix, which printed them unchained."""
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


# fmt: off
@pytest.mark.skipif(os.geteuid() == 0, reason = "root can open a mode 000 node, nothing is shut")
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
    """library_arch is the loader's own field, read for exactly this purpose: to skip a driver
    whose bitness cannot match the process. A neutrally named 32-bit registration passed the
    filename test and credited a driver this binary cannot open, and a manifest named i686
    that declares 64 is loadable, because the declaration is the loader's answer and the name
    is only ever a guess at it. The field is optional -- Debian strips it back out of Mesa's
    manifests to keep one file across architectures -- so its absence is ordinary and the
    object itself still says: EI_CLASS is byte 4 of every ELF. The filename is the last
    resort, unchanged, and is also all the installer ever has.

    Fails before the fix, which asked the filename alone."""
    name, manifest_kwargs, is_32 = case
    manifest = _icd_manifest_with(tmp_path, name, **manifest_kwargs)
    # Empty for every arm, so the filename fallback is reached only where there is no
    # declaration and no library on disk to read.
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
    """tests/sh lifts probes out of install.sh one function at a time, by name, with
    `sed -n '/^_name()/,/^}/p'`. So _has_amd_rocm_gpu may only call helpers those harnesses
    already lift: a call to anything else is an undefined function there, the ROCm branch
    falls through to the CPU wheel index, and four harnesses fail on a torch-index assertion
    that has nothing to do with what changed. That is not hypothetical -- splitting the probe
    into a wrapper over a private _amd_rocm_gpu_visible did exactly this, and the NVIDIA veto
    now lives inside the function so there is nothing to forget. Fails if the split comes
    back."""
    called = _functions_called_by(_install_sh_lines(), "_has_amd_rocm_gpu")
    assert called <= _ROCM_PROBE_CALLEES_THE_SH_HARNESSES_LIFT, called


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


# fmt: off
@pytest.mark.parametrize("case", [
    pytest.param(({32, 64}, 1, False), id = "both_bitnesses_installed"),
    pytest.param(({32}, 0, True), id = "only_the_wrong_bitness_installed"),
])
# fmt: on
def test_which_copy_of_a_multilib_soname_answers(monkeypatch, tmp_path, case):
    """Both bitnesses of one soname is what a multilib driver install looks like. ld.so picks
    the copy matching the process; the reconstructed search order does not, and i386 sorts
    first, so the first hit was the 32-bit one -- _an_icd_is_32_bit then read that object and
    discarded a manifest whose driver the loader loads, which either withholds the whole
    no-driver repair or reports the loader as AMD-only when it is not. The control keeps the
    rule from becoming "never 32-bit", which would put back every unloadable 32-bit
    registration the bitness filter exists to drop.

    Fails before the fix, which returned the first match."""
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
    """loader_parse_icd_manifest returns VK_ERROR_INCOMPATIBLE_DRIVER on a missing
    file_format_version and on a missing api_version, exactly as it does on a missing
    library_path. Reading library_path alone counted a registration the loader refuses, which
    suppresses the no-driver repair or calls the loader AMD-only when it is neither. Fails
    before the fix for both fields."""
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
    """glibc separates the two states by EXIT STATUS, not by row count: a cache that is
    present and empty prints "0 libs found in cache" and exits 0, while an absent cache file
    exits 1 with nothing on stdout. Storing only a non-empty set collapsed them, so a fresh
    container whose cache has not been built read as "cannot enumerate". The non-zero exit is
    the absent-cache case and must stay None, so a live driver is never called stale."""
    returncode, stdout, sonames = case
    _ldconfig_answering(monkeypatch, returncode = returncode, stdout = stdout)
    if sonames is None:
        assert amd._ld_cache_sonames() is None
    else:
        assert amd._ld_cache_sonames() == sonames


def test_a_bare_soname_is_stale_when_the_readable_cache_does_not_carry_it(
    monkeypatch, linux, tmp_path
):
    """What the distinction is for. The soname is on no directory ld.so searches and in a
    cache that could be read, so it does not resolve -- and the manifest naming it is a
    registration with no driver behind it. Read as unknown, it answered usable, which is the
    arm that withholds the reinstall half of the repair."""
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
    """The chain's guard is that /dev/kfd is not CLOSED, which is equally true when the node is
    absent and when it is open, so an openable /dev/kfd reached a sentence saying there was
    none -- and /dev/kfd IS the amdkfd char device, so its presence proves the stack is loaded
    and a reinstall repairs nothing. Read off install.sh because a host with the node cannot
    be fabricated here: the test operator is live by design, and the arm below skips without
    one. A revert removes the `[ -e /dev/kfd ]` arm and this raises rather than passing
    quietly."""
    block = _kernel_stack_hint_block()
    present = block.index("[ -e /dev/kfd ]; then")
    absent = block.index("[ ! -e /dev/kfd ]; then")
    assert present < block.index("kernel stack is already loaded") < absent
    assert absent < block.index("Install the ROCm kernel stack")


def test_the_installer_names_the_userspace_when_the_node_is_already_there(tmp_path):
    """The executed half, for a host that has the node. It used to skip unless the RUNNER
    had /dev/kfd, which is the one machine shape a dev box never is; the arm it pairs with
    skipped on exactly the complement, so the pair was never both run anywhere. The `-e`
    operator is still executed -- only the path it is given belongs to this case."""
    out = _kernel_stack_hint_text(tmp_path, topology = False, kfd_present = True)
    assert "Install the ROCm kernel stack" not in out
    assert "kernel stack is already loaded" in out
    assert "rocminfo" in out


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
    """Whether _run_may_open_a_gpu_node fires, on an NVIDIA host with --no-torch.

    That host is the one the automatic route answers NO for, so anything that says yes here
    says it because the run named a backend.
    """
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
    """effective_backend_request resolves UNSLOTH_FORCE_VULKAN through
    environment_backend_override, so a run that sets only the legacy flag installs the Vulkan
    bundle; reading the new variable alone left that run on the automatic route here, where an
    NVIDIA card answers no and every render-node diagnosis is suppressed for an install that
    opens exactly those nodes. A recognised public value is authoritative, so the legacy
    boolean cannot pull a cuda install back to Vulkan -- and "auto" is such a value, a request
    to DETECT, which is why the override returns it rather than falling through; otherwise the
    fix reads as "the legacy flag always wins" and takes a host that asked for detection off
    the automatic route. Only the four truthy spellings count, exactly as the Python side
    lists them, so a 0 leaves the automatic route alone."""
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
    # A negative ordinal is the one out-of-range value decidable WITHOUT the raw device
    # count: ggml extracts with `size_t tmp; while (ss >> tmp)`, unsigned extraction runs
    # strtoull, and "-1" wraps to 2**64-1, which is >= any possible num_available_devices.
    # So it always throws "Invalid Vulkan device index" and no group membership repairs it.
    # Reported as a blocker, never as a mask merely worth checking after the group repair.
    pytest.param(({_VK_MASK: "-1"}, {"vulkan"},
                  ("visibility mask is also in force", f"{_VK_MASK}='-1'"),
                  ("names a device this cannot resolve",)),
                 id = "a_negative_ordinal_always_throws_so_it_blocks"),
    # Extraction WALKS the list, so a negative one throws wherever it sits, as long as
    # every token ahead of it still extracts.
    pytest.param(({_VK_MASK: "0,-1"}, {"vulkan"},
                  ("visibility mask is also in force",),
                  ("names a device this cannot resolve",)),
                 id = "a_negative_ordinal_behind_a_valid_one_still_blocks"),
    # ... but only as far as the first token that does NOT extract: ggml stops reading
    # there, so the negative one is never reached and cannot be what throws.
    pytest.param(({_VK_MASK: "abc,-1"}, {"vulkan"},
                  ("visibility mask is also in force",), ()),
                 id = "a_negative_ordinal_after_a_dead_token_is_never_read"),
    # "-0" wraps to 0, which is in range on any host with a device, so it is not a throw
    # and must not be promoted to a blocker: that would invent the fault this guards.
    pytest.param(({_VK_MASK: "-0"}, {"vulkan"},
                  ("names a device this cannot resolve", f"{_VK_MASK}='-0'"), ()),
                 id = "a_negative_zero_is_in_range_and_stays_unresolved"),
])
# fmt: on
def test_when_the_vulkan_selector_is_named_beside_the_node(monkeypatch, linux, case):
    """A Vulkan build reads none of the four HIP/CUDA selectors, so the loop above skips them
    all and the node repair was returned as the complete explanation. ggml reads
    GGML_VK_VISIBLE_DEVICES itself and _run_vulkan_probe passes it through, and an empty value
    extracts no ordinal at all, so the probe stays empty however the node is owned. Its
    ordinals index the RAW vkEnumeratePhysicalDevices list, before CPU devices are dropped and
    ICDs deduplicated, so this process does not have the bound and a merely large positive
    ordinal throws rather than hiding: reported to check rather than called a blocker, since
    naming it one would invent a fault. A NEGATIVE one is the exception, decidable without the
    bound because it wraps past every possible one. A build reads its own selectors and no
    others, so naming this
    one to a HIP install sends the user after a variable its runtime never reads; and unset is
    not empty, so the rule cannot be "always mention it"."""
    env, backends, present, absent = case
    reason = _reason_with_masks(monkeypatch, env, backends)
    for _text in present:
        assert _text in reason, _text
    for _text in absent:
        assert _text not in reason, _text


def test_the_two_topology_readers_agree_on_this_host():
    """The shell and Python tri-states are the same rule written twice, and the closed-node
    walk on each side now branches on the third value, so a divergence would give one half the
    DRM fallback and not the other. Asserted as AGREEMENT rather than as a fixed value, so the
    arm is meaningful on a runner with a real KFD topology as well as on one without."""
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
    """Read off install.sh because a /dev/kfd cannot be fabricated here: the arm is gated on
    the literal path and on live -e/-r/-w tests, which are the rule under test. Without the
    fallback the installer names only the render node's group, and where the two nodes carry
    different owning groups that membership leaves KFD shut -- and the PCI branch further down
    then reads the node as openable. A revert removes these names and this fails rather than
    passing quietly."""
    lines = _install_sh_lines()
    body = _shell_fn(lines, "_amd_nodes_closed_to_this_user")
    _kfd = body.index("= /dev/kfd ]")
    _elif = body.index("elif _node_vendor=")
    _arm = body[_kfd:_elif]
    assert "_kfd_topology_amd_state" in _arm
    assert "_a_confirmed_amd_render_node_exists" in _arm
    # The readable-but-not-AMD state still drops the node, which is what keeps an
    # NVIDIA-only host silent; only the unreadable one reaches DRM.
    assert "-eq 1 ]; then" in _arm and "continue" in _arm


def _loader_blame(
    monkeypatch,
    manifests: dict,
    searched = (),
    **env: str,
) -> "str | None":
    """Which override amd.py blames for a loader that can load none of its manifests.

    The dict value says whether that manifest still resolves to a library, which is the
    difference between "clear the variable" and "reinstall the driver": these paths do not
    exist on the test host, so the real _icd_manifest_is_usable would call every one of them
    broken. ``searched`` is what the ordinary search would find with a forced list cleared,
    which is the only way a forced list can be the repair.
    """
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
    """Which variable clearing would actually restore a driver, and None where none would.

    DISABLE applies to every driver the loader knows, so a list matching them all leaves it
    with none and no reinstall changes an environment variable; the sentence prescribed
    exactly that repair. SELECT is read first because _vulkan_loader_allows reads it first: a
    set select list answers alone, so one naming a driver this host lacks excludes the ones it
    has. A forced list REPLACES the search, so a stale path leaves the loader with manifests
    that do not resolve while the real drivers sit unread, and reinstalling puts a driver
    where nothing is looking. With no override the manifests are what they are and installing
    a driver IS the repair, so the rule cannot be "always blame the environment".

    Disable WINS over select, so clearing select after both name the same driver changes
    nothing -- advice that cannot work, and what this answered before. Select is named where
    select is what excludes the manifest; both are named where neither removal alone repairs,
    and the same across kinds, since clearing a filter can leave a manifest that is not there
    while clearing the list exposes a search the filter then empties. The one-sided controls
    sit beside those, so this is not "always say both". A forced list is only the repair when
    clearing it leaves a driver: a present, permitted manifest that has merely lost its
    library is the reinstall case, while the same list hiding a search that WOULD have found
    one is the repair after all.

    Codex 4040623253: the counterfactual tested filename allowance, not loadability, so a
    filter over a manifest that no longer resolves was named as the whole repair -- clear the
    variable, and the loader keeps the same unloadable manifest and no driver. Its control
    keeps that from becoming "never blame a filter". Finding no manifests at all says only
    that this cannot read the loader's configuration, which
    the_vulkan_loader_has_no_usable_driver already answers False for, so there is nothing to
    name."""
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
    """The message a user actually reads, since the helper above only decides it. The manifest
    in the first case resolves perfectly well and the filter is the whole reason the loader
    has nothing, so "reinstall the Vulkan driver" is a repair that cannot work; the control is
    the case the sentence WAS written for, without which the fix could be "never say
    reinstall"."""
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
    """What install.sh captures as the account to name, given this `id`.

    The assignment is lifted from the file rather than restated, so reverting it fails this.
    """
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
    """`docker run --user 1234`, which is the shape the paragraph above the assignment is
    about. GNU id PRINTS the uid and THEN exits 1 for a uid it cannot resolve (coreutils id.c,
    print_user falls back to uidtostr), so a `|| printf ''` fallback never runs and the
    captured value was the number. The callers read empty as "no account to name" and print
    the container repair; a numeric one reached `usermod -a -G render 1234`, which usermod
    rejects with "user '1234' does not exist". Verified against real GNU coreutils in a
    container before this was written. The control keeps the fix from becoming "never name an
    account", which removes the usermod prescription on every ordinary host."""
    id_stub, named = case
    assert _repair_user_under(id_stub) == named


def _topology_state(monkeypatch, tmp_path, entries: "dict[str, str | None]"):
    """_kfd_topology_amd_state over a fabricated node tree.

    A value of None is a properties file that will not open, which is what a masked sysfs
    and an LSM both produce; the others are the file's contents. os.listdir and open are
    redirected on the MODULE rather than on builtins: a global open patch recurses, since
    pathlib opens files to answer the patch.
    """
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
@pytest.mark.skipif(os.geteuid() == 0, reason = "root opens a 0000 file, so nothing is unreadable")
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
        # The case Khronos settles: disable is considered BEFORE select, and drivers have no
        # VK_LOADER_LAYERS_ALLOW counterpart to name one back, so the real loader ends up
        # with no driver here. Read as "select answers alone" this counted Radeon usable and
        # reported only the device-node repair for a host groups cannot fix.
        ("radeon*", "radeon*", False),
        ("radeon*", "nvidia*", True),
        ("radeon*,nvidia*", "nvidia*", True),
    ],
)
def test_the_loader_filters_are_an_allowlist_then_a_denylist(monkeypatch, select, disable, allowed):
    """Vulkan-Loader, LoaderInterfaceArchitecture.md: "The values from the disable
    environment variable will be considered before the enable or select environment
    variable", and VK_LOADER_DRIVERS_DISABLE is "also checked before other driver
    environment variables (such as VK_LOADER_DRIVERS_SELECT)"."""
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
    """`docker run --device /dev/dri` without `--device /dev/kfd`, which commonly masks
    /sys/class/kfd too. The topology proves nothing there, but DRM names an AMD render node
    outright, and a HIP caller cannot run without /dev/kfd -- so suppressing the diagnosis left
    that host with the generic "no GPU" reading of its own missing device mapping. The DRM
    evidence must be the CONFIRMED kind, since an NVIDIA-only host has render nodes under the
    same glob and reading an unreadable vendor as AMD would hand it a ROCm device-mapping
    repair for a card it does not have. And READ and denied is not unknown, so a host whose
    KFD nodes all report another vendor gets nothing whatever DRM says."""
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
    """The membership sentence ends in a command a user copies, and nothing may follow it.

    Rendered in place it ran straight into the next sentence: "... -a -G render ada ROCm needs
    /dev/kfd ...", where the command a reader selects to end-of-line picks up the word after
    it. Ending the whole message with it is what keeps the command copyable as written; a full
    stop would be selected along with the account name instead. The sentence that used to
    follow it is still present, just no longer glued on, and with nothing else to say the hint
    is still the repair, so nothing is lost by holding it back to the end."""
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
