# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""studio-tauri-native-ci.yml runs only on what the Windows/macOS Rust jobs compile or read, so every
file the Rust code embeds or opens outside studio/src-tauri must match its path filters."""

import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TAURI = REPO / "studio" / "src-tauri"
WORKFLOW = REPO / ".github" / "workflows" / "studio-tauri-native-ci.yml"

# include_str!("..") / include_bytes!("..") are relative to the source file; a literal may sit on
# the line after the macro's opening parenthesis.
_INCLUDE = re.compile(r'include_(?:str|bytes)!\(\s*"([^"]+)"', re.S)
# Path::new(env!("CARGO_MANIFEST_DIR")).join("..") reads relative to studio/src-tauri.
_MANIFEST_JOIN = re.compile(r'env!\("CARGO_MANIFEST_DIR"\)\s*\)\s*\.join\(\s*"([^"]+)"', re.S)


def _covered(path: str, patterns) -> bool:
    for pattern in patterns:
        rx = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        if re.fullmatch(rx, path):
            return True
    return False


def _outside_inputs() -> set[str]:
    found = set()
    sources = list((TAURI / "src").rglob("*.rs")) + [TAURI / "build.rs"]
    for source in sources:
        text = source.read_text(encoding = "utf-8")
        for rel in _INCLUDE.findall(text):
            found.add((source.parent / rel).resolve())
        for rel in _MANIFEST_JOIN.findall(text):
            found.add((TAURI / rel).resolve())
    # Bundle resources ("../../install.ps1": "install.ps1") are copied by `tauri build`, relative
    # to studio/src-tauri.
    for conf in TAURI.glob("tauri*.conf.json"):
        resources = json.loads(conf.read_text(encoding = "utf-8")).get("bundle", {}).get("resources")
        if isinstance(resources, dict):
            resources = list(resources)
        for rel in resources or []:
            found.add((TAURI / rel).resolve())
    # `npm run build` (the frontendDist both jobs build) runs scripts/<name>.mjs first.
    frontend = REPO / "studio" / "frontend"
    build = json.loads((frontend / "package.json").read_text(encoding = "utf-8"))["scripts"]["build"]
    for rel in re.findall(r"node\s+(\S+\.mjs)", build):
        found.add((frontend / rel).resolve())
    outside = set()
    for path in found:
        try:
            path.relative_to(TAURI.resolve())
            continue
        except ValueError:
            pass
        if path.is_file():
            outside.add(path.relative_to(REPO.resolve()).as_posix())
    return outside


def test_the_scan_finds_the_known_outside_inputs():
    """A scan that matched nothing would pass the check below."""
    inputs = _outside_inputs()
    for expected in (
        "studio/frontend/src/features/auth/api.ts",
        "studio/backend/tests/fixtures/access_log_records.json",
        "unsloth_cli/_system_dir_guard.py",
        "install.ps1",
        "install.sh",
        "studio/frontend/scripts/typecheck.mjs",
    ):
        assert expected in inputs, f"{expected} is no longer found by the scan: {sorted(inputs)}"


def test_every_file_the_rust_code_reads_outside_src_tauri_triggers_the_workflow():
    on = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))[True]
    for trigger in ("pull_request", "push"):
        patterns = on[trigger]["paths"]
        missing = sorted(p for p in _outside_inputs() if not _covered(p, patterns))
        assert not missing, (
            f"studio/src-tauri reads these files, but no {trigger} path of {WORKFLOW.name} matches "
            f"them, so editing one would skip the Windows/macOS Rust tests: {missing}"
        )
