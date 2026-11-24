# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""studio-tauri-native-ci.yml runs only on what the Windows/macOS Rust jobs compile or read, so every
file the Rust code embeds or opens outside studio/src-tauri must match its path filters."""

import ast
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TAURI = REPO / "studio" / "src-tauri"
WORKFLOW = REPO / ".github" / "workflows" / "studio-tauri-native-ci.yml"

# include_str!("..") / include_bytes!("..") are relative to the source file; a literal may sit on
# the line after the macro's opening parenthesis.
_INCLUDE = re.compile(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"', re.S)
# Path::new(env!("CARGO_MANIFEST_DIR")).join("..") reads relative to studio/src-tauri.
_MANIFEST_JOIN = re.compile(r'env!\("CARGO_MANIFEST_DIR"\)\s*\)\s*\.join\(\s*"([^"]+)"', re.S)


def _glob(pattern: str) -> re.Pattern:
    return re.compile(re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*"))


def _covered(path: str, patterns) -> bool:
    """GitHub path filters: the last pattern that matches decides, and `!` excludes."""
    covered = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if _glob(pattern.lstrip("!")).fullmatch(path):
            covered = not negated
    return covered


def _outside_inputs() -> set[str]:
    found = set()
    # Every crate target cargo test / check builds: src/, tests/, examples/, benches/, build.rs.
    sources = [p for p in TAURI.rglob("*.rs") if "target" not in p.relative_to(TAURI).parts]
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
    # Modules the vite config files import from outside themselves run in `vite build` too.
    for config in frontend.glob("vite*.ts"):
        for rel in re.findall(r'from\s+"(\./[^"]+)"', config.read_text(encoding = "utf-8")):
            target = (frontend / rel).resolve()
            if not target.name.startswith("vite"):
                found.add(target if target.suffix else target.with_suffix(".ts"))
    # The pre-install lockfile audit both jobs run reads its default lockfiles.
    audit = REPO / "scripts" / "lockfile_supply_chain_audit.py"
    for node in ast.walk(ast.parse(audit.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("DEFAULT_") and t.id.endswith("LOCKFILES")
            for t in node.targets
        ):
            for rel in ast.literal_eval(node.value):
                found.add((REPO / rel).resolve())
    outside = set()
    for path in found:
        try:
            path.relative_to(TAURI.resolve())
            continue
        except ValueError:
            pass
        # A referenced file that was deleted still has to trigger: the deleting commit breaks the build.
        if not path.is_dir():
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
        "studio/frontend/src/i18n/locale-tags.ts",
        "studio/backend/core/data_recipe/oxc-validator/package-lock.json",
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


def test_a_later_exclusion_uncovers_a_path():
    """GitHub applies the last matching pattern, so a trailing `!x` removes x."""
    assert _covered("a/b.css", ["a/**"])
    assert not _covered("a/b.css", ["a/**", "!a/b.css"])
    assert _covered("a/b.css", ["a/**", "!a/b.css", "a/b.css"])
