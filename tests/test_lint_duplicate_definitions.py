# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Compare mode of the duplicate-definition gate, driven over a throwaway git repo.

`--self-test` already covers the AST rule. What is only reachable through git is the
question compare mode answers: did THIS diff introduce the duplicate, or was it already
there? Getting that wrong in either direction is how the gate stops being useful, so each
case here runs the real script against real commits.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_duplicate_definitions.py"

DUPLICATE = "def go():\n    return 1\n\n\ndef go():\n    return 2\n"
SINGLE = "def go():\n    return 1\n"


def _git(repo, *args):
    done = subprocess.run(["git", *args], cwd = repo, capture_output = True, text = True, check = True)
    return done.stdout.strip()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", ".")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


def _run(repo, before, after, *paths):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--before", before, "--after", after, *paths],
        cwd = repo,
        capture_output = True,
        text = True,
    )


def test_duplicate_inserted_above_the_original_is_blocked(repo):
    """The copy lands first, so the finding is reported at the OLD, unchanged line.

    This is the shape that made the gate silent: judging by which lines the diff added
    classifies it as pre-existing even though the branch is what created it.
    """
    (repo / "mod.py").write_text(SINGLE)
    before = _commit(repo, "base")
    (repo / "mod.py").write_text("def go():\n    return 2\n\n\ndef go():\n    return 1\n")
    after = _commit(repo, "duplicate above")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 1, result.stdout
    assert "go is defined twice" in result.stdout


def test_pre_existing_duplicate_does_not_block_an_unrelated_edit(repo):
    (repo / "mod.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    (repo / "mod.py").write_text(DUPLICATE + "\n\ndef other():\n    return 3\n")
    after = _commit(repo, "unrelated edit")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 0, result.stdout
    assert "not failing" in result.stdout


def test_duplicate_in_a_file_the_branch_adds_is_blocked(repo):
    (repo / "keep.py").write_text(SINGLE)
    before = _commit(repo, "base")
    (repo / "added.py").write_text(DUPLICATE)
    after = _commit(repo, "add a file")

    result = _run(repo, before, after, "added.py")
    assert result.returncode == 1, result.stdout


def test_rename_carrying_a_pre_existing_duplicate_does_not_block(repo):
    """The sweep reports a rename under its NEW name, which does not exist at `before`."""
    (repo / "mod.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    _git(repo, "mv", "mod.py", "moved.py")
    (repo / "moved.py").write_text(DUPLICATE + "\n\ndef other():\n    return 3\n")
    after = _commit(repo, "rename and edit")

    result = _run(repo, before, after, "moved.py")
    assert result.returncode == 0, result.stdout


def test_rename_that_also_adds_a_duplicate_still_blocks(repo):
    (repo / "mod.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    _git(repo, "mv", "mod.py", "moved.py")
    (repo / "moved.py").write_text(
        DUPLICATE + "\n\ndef nu():\n    return 1\n\n\ndef nu():\n    return 2\n"
    )
    after = _commit(repo, "rename and duplicate")

    result = _run(repo, before, after, "moved.py")
    assert result.returncode == 1, result.stdout
    assert "nu is defined twice" in result.stdout


def test_a_scoped_duplicate_is_not_absorbed_by_a_module_level_one(repo):
    """The finding identity has to keep the SCOPE and the SOURCE apart.

    Concatenated, a module-level `from A.m import x` and a `from m import x` inside `class A`
    both spell `A.m:x`, so a branch that deletes the first and writes the second had its new
    finding charged to the old one's counter and passed. They are different duplicates in
    different scopes and only a delimiter can say so.
    """
    (repo / "mod.py").write_text("from A.m import x\nfrom A.m import x\n")
    before = _commit(repo, "module-level duplicate")
    (repo / "mod.py").write_text("class A:\n    from m import x\n    from m import x\n")
    after = _commit(repo, "swap it for a scoped one")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 1, result.stdout
    assert "A.x is imported twice" in result.stdout


def test_a_file_declaring_a_non_utf8_source_encoding_is_read_not_crashed_on(repo):
    """PEP 263 says the file's own declaration decides how it decodes.

    Reading the blob with the runner's UTF-8 locale raised UnicodeDecodeError out of
    `subprocess` and took the whole run down -- on a file the parser, `compileall` and ruff all
    accept, and on an edit that had nothing to do with it. Decoded properly the file is ordinary
    source, so the duplicate in it is found like any other.
    """
    (repo / "mod.py").write_bytes(b'# coding: cp1252\nS = "caf\xe9"\n\n\ndef go():\n    return 1\n')
    before = _commit(repo, "base")
    (repo / "mod.py").write_bytes(
        b'# coding: cp1252\nS = "caf\xe9"\n\n\ndef go():\n    return 1\n\n\ndef go():\n    return 2\n'
    )
    after = _commit(repo, "duplicate it")

    result = _run(repo, before, after, "mod.py")
    assert "UnicodeDecodeError" not in result.stderr, result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert "go is defined twice" in result.stdout


def test_a_rename_from_a_non_python_file_is_treated_as_an_addition(repo):
    """Renaming `mod.txt` to `mod.py` is what makes those definitions active code.

    The before side is not an eligible Python file, so its duplicates were never live and
    cannot be inherited. Following the rename onto it consumed them as pre-existing and let
    the branch turn a duplicate-carrying text file into a duplicate-carrying module for free.
    """
    (repo / "mod.txt").write_text(DUPLICATE)
    before = _commit(repo, "base with a duplicate in a text file")
    _git(repo, "mv", "mod.txt", "mod.py")
    after = _commit(repo, "promote the text file to a module")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 1, result.stdout
    assert "go is defined twice" in result.stdout


def test_a_second_copy_of_an_already_duplicated_name_is_blocked(repo):
    """Two copies before, three after. Counting by identity has to notice the third."""
    (repo / "mod.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    (repo / "mod.py").write_text(DUPLICATE + "\n\ndef go():\n    return 3\n")
    after = _commit(repo, "a third copy")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 1, result.stdout


def test_a_new_plain_import_duplicate_is_not_absorbed_by_a_different_one(repo):
    """Same bound name, different source: two distinct duplicates, not one carried over.

    Both findings would read as `import:None:x` if the identity dropped the module the
    alias came from, so the counter for the pre-existing `alpha` pair would absorb the
    `beta` pair this diff introduces and the gate would pass.
    """
    (repo / "mod.py").write_text("import alpha as x\nimport alpha as x\n")
    before = _commit(repo, "base with a duplicate import")
    (repo / "mod.py").write_text("import beta as x\nimport beta as x\n")
    after = _commit(repo, "swap the source, keep the duplication")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 1, result.stdout
    assert "x is imported" in result.stdout


def test_rename_of_a_non_ascii_path_is_matched_to_its_old_name(repo):
    """core.quotePath escapes the path, and the escaped spelling matches nothing.

    The before side then looks absent, every finding in the file reads as introduced, and
    a branch that only moves a file gets blocked for a duplicate already on main.
    """
    (repo / "café.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    _git(repo, "mv", "café.py", "resumé.py")
    (repo / "resumé.py").write_text(DUPLICATE + "\n\ndef other():\n    return 3\n")
    after = _commit(repo, "rename and edit")

    result = _run(repo, before, after, "resumé.py")
    assert result.returncode == 0, result.stdout
    assert "not failing" in result.stdout


def test_file_deleted_at_head_is_skipped(repo):
    (repo / "mod.py").write_text(DUPLICATE)
    before = _commit(repo, "base with duplicate")
    (repo / "mod.py").unlink()
    (repo / "keep.py").write_text(SINGLE)
    after = _commit(repo, "delete it")

    result = _run(repo, before, after, "mod.py")
    assert result.returncode == 0, result.stdout
