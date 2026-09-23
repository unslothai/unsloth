# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""On RDNA1 (gfx101x) Triton's AMD buffer ops silently do nothing.

Every Triton kernel launches with hipSuccess and leaves its outputs untouched, because the
buffer resource descriptor Triton builds is laid out for gfx10.3+ and gfx10.1 reads it
differently. Found on an RX 5700 XT (gfx1010): a ten-line ``x * 2`` kernel returned 0 of
1024 correct values, and every value with ``AMDGCN_USE_BUFFER_OPS=0``. unsloth sets that
variable at import when such a GPU is visible. These are decision-table checks, no GPU.
"""

import pytest

torch = pytest.importorskip("torch")

from unsloth import device_type


@pytest.mark.parametrize(
    "arch, lacks",
    [
        ("gfx1010", True),  # RDNA1, RX 5700 XT: where it was measured
        ("gfx1010:xnack-", True),  # the suffixed form torch reports
        ("gfx1011", True),  # RDNA1, Radeon Pro V520
        ("gfx1012", True),  # RDNA1, RX 5500 XT
        ("gfx1013", True),  # RDNA1, Cyan Skillfish
        ("gfx1030", False),  # RDNA2 reads the newer descriptor, buffer ops are fine
        ("gfx1034", False),  # RDNA2, RX 6500 XT: measured correct with buffer ops on
        ("gfx1034:sramecc-:xnack-", False),
        ("gfx1100", False),  # RDNA3
        ("gfx1201", False),  # RDNA4
        ("gfx90a", False),  # CDNA
        ("gfx942", False),  # CDNA3
        ("", False),  # unreadable: fail open, never guess
        (None, False),
    ],
)
def test_only_gfx101x_loses_buffer_ops(arch, lacks):
    """The prefix has to be exactly ``gfx101``. ``gfx10`` would also switch RDNA2 to global
    loads, which is slower for no reason; ``gfx1`` would reach RDNA3 and RDNA4."""
    assert device_type.arch_lacks_buffer_ops(arch) is lacks


def test_an_unreadable_arch_does_not_narrow_anything():
    for unreadable in ("", None, "unknown", "   "):
        assert device_type.arch_lacks_buffer_ops(unreadable) is False


def test_bf16_gate_and_buffer_ops_gate_disagree_on_rdna2():
    """Both gates exist because the two problems have different scopes: all of gfx10 lacks
    bf16, only gfx10.1 mishandles the descriptor. A refactor that merges them breaks RDNA2."""
    assert device_type.arch_lacks_bf16("gfx1034") is True
    assert device_type.arch_lacks_buffer_ops("gfx1034") is False
    assert device_type.arch_lacks_bf16("gfx1010") is True
    assert device_type.arch_lacks_buffer_ops("gfx1010") is True


def test_workaround_sets_knob_and_a_separate_cache_dir():
    """The knob alone is not enough: Inductor's cache does not key on AMDGCN_USE_BUFFER_OPS,
    so kernels compiled with buffer ops on are reused with them off. Measured on an RX 5700
    XT: the same run trained cleanly with empty caches and went -inf / nan once a
    buffer-ops-on process had populated them."""
    env = {}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is True
    assert env["AMDGCN_USE_BUFFER_OPS"] == "0"
    assert env["TRITON_CACHE_DIR"].replace("\\", "/") == "/th/.triton/cache-no-buffer-ops"
    # Inductor caches the Triton kernels it generates on its own, keyed the same blind way.
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("_no_buffer_ops")
    assert "torchinductor_" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_workaround_honours_a_user_who_already_chose():
    env = {"AMDGCN_USE_BUFFER_OPS": "1"}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is False
    assert env == {
        "AMDGCN_USE_BUFFER_OPS": "1"
    }, "an explicit choice is not overridden and no cache dir is forced"
    env = {"TRITON_CACHE_DIR": "/mine", "TORCHINDUCTOR_CACHE_DIR": "/mine-inductor"}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is True
    assert env["TRITON_CACHE_DIR"] == "/mine"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/mine-inductor"


def test_workaround_follows_triton_home():
    env = {"TRITON_HOME": "/custom/triton"}
    device_type.apply_gfx101x_triton_workaround(env)
    assert (
        env["TRITON_CACHE_DIR"].replace("\\", "/") == "/custom/triton/.triton/cache-no-buffer-ops"
    )


@pytest.mark.parametrize("preset", ["0", "false", "off", ""])
def test_a_user_who_already_turned_buffer_ops_off_still_gets_clean_caches(preset):
    """AMDGCN_USE_BUFFER_OPS=0 is the manual workaround from #11614. Without separate caches
    that user keeps loading the kernels an earlier buffer-ops-on run left behind."""
    env = {"AMDGCN_USE_BUFFER_OPS": preset}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is True
    assert env["AMDGCN_USE_BUFFER_OPS"] == preset
    assert env["TRITON_CACHE_DIR"].replace("\\", "/") == "/th/.triton/cache-no-buffer-ops"
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("_no_buffer_ops")


@pytest.mark.parametrize("preset", ["1", "true", "ON", "yes", "y"])
def test_every_spelling_triton_reads_as_on_is_left_alone(preset):
    env = {"AMDGCN_USE_BUFFER_OPS": preset}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is False
    assert env == {"AMDGCN_USE_BUFFER_OPS": preset}


def test_inductor_default_already_in_environ_is_still_redirected():
    """`import torch._dynamo` writes Inductor's default cache dir into os.environ before
    unsloth gets here, so "already set" usually means "torch set the shared default"."""
    default = device_type._default_inductor_cache_dir()
    env = {"TORCHINDUCTOR_CACHE_DIR": default}
    device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == default + "_no_buffer_ops"


def test_import_time_env_is_redirected_after_torch_dynamo_import(monkeypatch):
    import torch._dynamo  # noqa: F401  populates TORCHINDUCTOR_CACHE_DIR as a side effect
    import os

    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS", raising = False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising = False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", device_type._default_inductor_cache_dir())
    assert device_type.apply_gfx101x_triton_workaround() is True
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].endswith("_no_buffer_ops")


def test_missing_passwd_entry_does_not_break_import(monkeypatch):
    """getpass.getuser raises in a container whose uid has no passwd entry."""
    import getpass
    import sys

    def boom():
        raise KeyError("getpwuid(): uid not found: 12345")

    monkeypatch.setattr(getpass, "getuser", boom)
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.cache_dir_utils", None)
    env = {}
    assert device_type.apply_gfx101x_triton_workaround(env, triton_home = "/th") is True
    assert "torchinductor_" in env["TORCHINDUCTOR_CACHE_DIR"]


def test_reapplying_after_patch_torch_compile_restores_the_inductor_dir(monkeypatch):
    """unsloth_zoo's patch_torch_compile pops TORCHINDUCTOR_CACHE_DIR after _gpu_init ran;
    models/_utils.py re-applies the workaround, which must put the separate dir back and
    leave everything else as the first call set it."""
    import os

    for name in ("AMDGCN_USE_BUFFER_OPS", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(device_type, "_GFX101X_TRITON_WORKAROUND_APPLIED", False)
    assert device_type.gfx101x_triton_workaround_applied() is False
    assert device_type.apply_gfx101x_triton_workaround() is True
    assert device_type.gfx101x_triton_workaround_applied() is True
    first = {
        k: os.environ[k]
        for k in ("AMDGCN_USE_BUFFER_OPS", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
    }
    os.environ.pop("TORCHINDUCTOR_CACHE_DIR")  # what patch_torch_compile does
    assert device_type.apply_gfx101x_triton_workaround() is True
    assert {k: os.environ[k] for k in first} == first


def test_a_dict_environ_does_not_mark_the_process():
    before = device_type.gfx101x_triton_workaround_applied()
    device_type.apply_gfx101x_triton_workaround({}, triton_home = "/th")
    assert device_type.gfx101x_triton_workaround_applied() is before
