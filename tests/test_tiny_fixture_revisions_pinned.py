# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every tiny Hub repo the gloo TP / PP builders read has a pinned revision, and the builders use it."""

import ast
import re
from pathlib import Path

from tests._tiny_fixture_revisions import REVISIONS

TESTS = Path(__file__).resolve().parent
BUILDERS = ("test_tp_native_oracle.py", "test_pp_native_gloo.py")
_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


def _module_repo_constants(path):
    found = {}
    for node in ast.parse(path.read_text(encoding = "utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _REPO.match(node.value.value)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
        ):
            found[node.targets[0].id] = node.value.value
    return found


def test_every_builder_fixture_repo_has_a_pinned_revision():
    missing = sorted(
        f"{name}: {name_repo[1]}"
        for name in BUILDERS
        for name_repo in _module_repo_constants(TESTS / name).items()
        if name_repo[1] not in REVISIONS
    )
    assert not missing, f"pin these in tests/_tiny_fixture_revisions.py: {missing}"


def test_pins_are_full_commit_ids():
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in REVISIONS.values())


def test_the_builders_load_the_config_at_the_pinned_revision():
    for name in BUILDERS:
        source = (TESTS / name).read_text(encoding = "utf-8")
        assert "AutoConfig.from_pretrained(repo, revision = revision(repo))" in source, name
