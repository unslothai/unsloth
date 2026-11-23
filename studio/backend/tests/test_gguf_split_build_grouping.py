# SPDX-License-Identifier: AGPL-3.0-only

from itertools import permutations

from hub.utils.gguf import group_gguf_variant_files

ALPHA_FIRST = "alpha-Q4_K_M-00001-of-00002.gguf"
ALPHA_SECOND = "alpha-Q4_K_M-00002-of-00002.gguf"
BETA_FIRST = "beta-Q4_K_M-00001-of-00002.gguf"
BETA_SECOND = "beta-Q4_K_M-00002-of-00002.gguf"
ENTRIES = [(ALPHA_FIRST, 10), (ALPHA_SECOND, 20), (BETA_FIRST, 30), (BETA_SECOND, 40)]
EXPECTED = {"Q4_K_M": (ALPHA_FIRST, 30), "beta-Q4_K_M": (BETA_FIRST, 70)}


def test_split_build_keys_keep_totals_separate_for_every_entry_order():
    keys = {
        ALPHA_FIRST: "Q4_K_M",
        ALPHA_SECOND: "Q4_K_M",
        BETA_FIRST: "beta-Q4_K_M",
        BETA_SECOND: "beta-Q4_K_M",
    }
    for entries in permutations(ENTRIES):
        assert group_gguf_variant_files(entries, keys) == EXPECTED


def test_partial_key_overrides_keep_the_unmapped_builds_legacy_key():
    keys = {BETA_FIRST: "beta-Q4_K_M", BETA_SECOND: "beta-Q4_K_M"}
    assert group_gguf_variant_files(iter(reversed(ENTRIES)), keys) == EXPECTED


def test_omitted_and_empty_overrides_keep_one_legacy_shard_family():
    expected = {"Q4_K_M": (ALPHA_FIRST, 30)}
    assert group_gguf_variant_files(ENTRIES) == expected
    assert group_gguf_variant_files(ENTRIES, None) == expected
    assert group_gguf_variant_files(ENTRIES, {}) == expected
