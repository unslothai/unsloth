# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every registered image and video family needs a label in the frontend family picker.

``studio/frontend/tests/family-override.test.ts`` checks the same thing, but only Frontend CI runs it and Frontend CI
does not trigger on a backend-only change. A family added to the registry without a label then turns the frontend
job red for every later frontend PR instead of the PR that added it. This copy runs with the backend tests, where the
family is added.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.inference import diffusion_families, video_families

FAMILY_OVERRIDE_TS = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "model-picker"
    / "components"
    / "model-selector"
    / "family-override.ts"
)


def _frontend_labels(source: str) -> set[str]:
    block = re.search(r"^const LABELS\b[^=]*= \{\n(.*?)^\};", source, re.M | re.S)
    assert block, f"{FAMILY_OVERRIDE_TS.name}: no LABELS table"
    return {
        quoted or bare
        for quoted, bare in re.findall(r'^\s+(?:"([^"]+)"|([A-Za-z_]\w*)):', block[1], re.M)
    }


def test_label_parser_reads_quoted_and_bare_keys():
    source = 'const LABELS: Readonly<Record<string, string>> = {\n  "flux.1": "FLUX.1",\n  sdxl: "SDXL",\n};\n'
    assert _frontend_labels(source) == {"flux.1", "sdxl"}


def test_every_registered_family_has_a_frontend_label():
    names = {family.name for family in diffusion_families._FAMILIES + video_families._FAMILIES}
    assert {"flux.1", "ltx-2", "cosmos3-nano"} <= names
    labels = _frontend_labels(FAMILY_OVERRIDE_TS.read_text(encoding = "utf-8"))
    missing = sorted(names - labels)
    assert not missing, f"add these families to LABELS in {FAMILY_OVERRIDE_TS.name}: {missing}"
