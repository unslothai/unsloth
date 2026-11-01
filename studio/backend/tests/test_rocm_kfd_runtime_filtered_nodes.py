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

"""KFD sysfs is not namespaced, so a container handed only some /dev/dri/renderD* nodes still lists every GPU node,
while the ROCm thunk drops nodes whose render device it cannot open and renumbers the rest. The context-free reader
must then decline instead of reporting another card's VRAM for HIP ordinal 0."""

import pytest

from utils.hardware import amd, hardware


def _node(root, idx, *, simd, minor):
    d = root / str(idx)
    d.mkdir(parents = True)
    lines = [f"simd_count {simd}", "vendor_id 4098" if simd else "vendor_id 0"]
    if minor is not None:
        lines.append(f"drm_render_minor {minor}")
    (d / "properties").write_text("\n".join(lines) + "\n")


@pytest.fixture
def two_boards(monkeypatch, tmp_path):
    nodes = tmp_path / "nodes"
    dri = tmp_path / "dri"
    dri.mkdir()
    _node(nodes, 0, simd = 0, minor = None)  # CPU node
    _node(nodes, 1, simd = 96, minor = 128)  # board A, 20/24 GiB free
    _node(nodes, 2, simd = 32, minor = 129)  # board B, 2/8 GiB free
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_KFD_TOPOLOGY_NODES", str(nodes), raising = False)
    monkeypatch.setattr(hardware, "_DRI_DIR", str(dri), raising = False)
    monkeypatch.setattr(hardware, "_rocm_kfd_gpu_pci_ids", lambda: ["0000:03:00.0", "0000:04:00.0"])
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda *a, **k: ["gfx1100", "gfx1102"])
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_amdgpu_cards",
        lambda: [("0000:03:00.0", 0, "x"), ("0000:04:00.0", 1, "x")],
    )
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {"0000:03:00.0": (4.0, 24.0), "0000:04:00.0": (6.0, 8.0)},
    )
    return dri


def test_every_render_node_present_keeps_the_kfd_order(two_boards):
    (two_boards / "renderD128").write_text("")
    (two_boards / "renderD129").write_text("")
    devices = hardware._rocm_context_free_visible_devices()
    assert [(d["visible_ordinal"], d["vram_total_gb"]) for d in devices] == [(0, 24.0), (1, 8.0)]


def test_a_render_node_missing_from_the_container_declines(two_boards):
    # Only board B was passed in (docker --device /dev/dri/renderD129): HIP ordinal 0 is the 8 GiB card.
    (two_boards / "renderD129").write_text("")
    assert hardware._rocm_context_free_visible_devices() == []
