# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A downloaded repo's README is the publisher's: listing cached models reads only its top."""

import tracemalloc

from hub.services.models import cache_inventory

FRONT_MATTER = (
    "---\npipeline_tag: text-generation\nlibrary_name: transformers\ntags:\n- unsloth\n---\n"
)


def test_huge_readme_is_read_only_up_to_the_cap(tmp_path):
    readme = tmp_path / "README.md"
    with open(readme, "w", encoding = "utf-8") as handle:
        handle.write(FRONT_MATTER)
        line = "x" * 1023 + "\n"
        for _ in range(64 * 1024):  # 64 MiB of body
            handle.write(line)

    tracemalloc.start()
    try:
        card = cache_inventory._read_model_card_frontmatter(readme)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert card == {
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["unsloth"],
    }
    assert peak < 16 * 1024 * 1024, f"reading the card held {peak / 2**20:.0f} MiB"


def test_small_cards_read_as_before(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(FRONT_MATTER.replace("\n", "\r\n") + "# Model\n", encoding = "utf-8")
    assert cache_inventory._read_model_card_frontmatter(readme)["pipeline_tag"] == "text-generation"

    bad = tmp_path / "BAD.md"
    bad.write_bytes(b"---\npipeline_tag: \xff\n---\n")
    assert cache_inventory._read_model_card_frontmatter(bad) == {}
