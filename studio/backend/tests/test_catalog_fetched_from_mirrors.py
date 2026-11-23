# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Images / Video catalog labels a vendor-id row with the repo it downloads from (``fetchedFrom``), so that label
must be exactly the mirror the backend swaps in at its fetch sites, and every vendor id the backend mirrors must say
so (a row without it shows the vendor id while the files come from unsloth/*)."""

import re
from pathlib import Path

from core.inference.diffusion_families import mirror_repo

_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/features/model-picker/components/model-selector/model-catalog.ts"
)
# bf16Mirror rows already carry the unsloth id; matched only so their options are not read into the row before.
_HELPER_CALL = re.compile(
    r"\b(bf16Pipeline|bf16Mirror|bf16Single|fp8Pipeline|bnb4bit|gguf)\(\s*\"([^\"]+)\""
)


def _media_rows() -> list[tuple[str, dict[str, str]]]:
    """(repoId, options) per IMAGE / VIDEO artifact, options read up to the next artifact helper call."""
    source = _CATALOG.read_text(encoding = "utf-8")
    start = source.index("export const IMAGE_CATALOG")
    end = source.index("export const AUDIO_CATALOG")
    body = source[start:end]
    calls = list(_HELPER_CALL.finditer(body))
    rows = []
    for i, call in enumerate(calls):
        chunk = body[call.end() : calls[i + 1].start() if i + 1 < len(calls) else len(body)]
        options = dict(re.findall(r"\b(fetchedFrom|gated):\s*(\"[^\"]+\"|true|false)", chunk))
        if call.group(1) != "bf16Mirror":
            rows.append((call.group(2), {k: v.strip('"') for k, v in options.items()}))
    return rows


def test_the_catalog_parse_sees_the_media_rows():
    ids = {repo for repo, _ in _media_rows()}
    assert {
        "black-forest-labs/flux-3-image-klein",
        "unsloth/flux-3-image-klein-GGUF",
        "Lightricks/LTX-2.3",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    } <= ids


def test_fetched_from_is_the_mirror_the_backend_downloads():
    rows = [(repo, opts["fetchedFrom"]) for repo, opts in _media_rows() if "fetchedFrom" in opts]
    assert rows
    for repo, fetched in rows:
        assert fetched.startswith("unsloth/"), repo
        assert (mirror_repo(repo) or "").lower() == fetched.lower(), repo


def test_every_mirrored_vendor_row_names_its_mirror():
    # A row still marked gated keeps fetchedFrom unset (the network check would judge the open mirror against its
    # gate); its toast is renamed from the plan's entries instead.
    missing = [
        repo
        for repo, opts in _media_rows()
        if not repo.startswith("unsloth/")
        and mirror_repo(repo)
        and "fetchedFrom" not in opts
        and opts.get("gated") != "true"
    ]
    assert missing == []
