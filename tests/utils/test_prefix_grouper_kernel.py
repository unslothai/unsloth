# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for PrefixGrouper segment metadata construction and QKV sequence padding."""

import pytest
import torch

from unsloth.utils.prefix_grouper_kernel import (
    _PAD_GROUP,
    _pad_labels,
    build_seg_info_multigroup,
)


def _reference_seg_info_multigroup(group_specs, device):
    group_of_list, is_prefix_list, suffix_of_list = [], [], []
    suffix_counter = 0
    for gid, (P, R_list) in enumerate(group_specs):
        group_of_list.append(torch.full((P,), gid, dtype = torch.long, device = device))
        is_prefix_list.append(torch.ones(P, dtype = torch.bool, device = device))
        suffix_of_list.append(torch.full((P,), -1, dtype = torch.long, device = device))
        for r in R_list:
            group_of_list.append(torch.full((r,), gid, dtype = torch.long, device = device))
            is_prefix_list.append(torch.zeros(r, dtype = torch.bool, device = device))
            suffix_of_list.append(torch.full((r,), suffix_counter, dtype = torch.long, device = device))
            suffix_counter += 1
    return _pad_labels(
        torch.cat(group_of_list),
        torch.cat(is_prefix_list),
        torch.cat(suffix_of_list),
        device,
    )


@pytest.mark.parametrize(
    "group_specs",
    [
        [(128, [64, 96, 32, 128])],
        [(64, [64])],
        [(50, [0, 10, 0, 25]), (80, [15, 20, 5])],
        [(256, [100, 120, 80, 90, 110, 75, 130, 95]) for _ in range(4)],
    ],
)
def test_build_seg_info_multigroup_matches_reference(group_specs):
    device = torch.device("cpu")
    seg, meta = build_seg_info_multigroup(group_specs, device)
    want_g, want_p, want_s, want_T, want_T_pad = _reference_seg_info_multigroup(group_specs, device)
    assert seg.T == want_T
    assert seg.T_pad == want_T_pad
    assert seg.signature == (
        "multi",
        want_T_pad,
        tuple((P, tuple(R_list)) for P, R_list in group_specs),
    )
    assert torch.equal(seg.group_of_kv, want_g)
    assert torch.equal(seg.is_prefix, want_p)
    assert torch.equal(seg.suffix_of_kv, want_s)
    if want_T_pad > want_T:
        assert (seg.group_of_kv[want_T:] == _PAD_GROUP).all()
        assert (~seg.is_prefix[want_T:]).all()
        assert (seg.suffix_of_kv[want_T:] == _PAD_GROUP).all()
    assert len(meta) == len(group_specs)
