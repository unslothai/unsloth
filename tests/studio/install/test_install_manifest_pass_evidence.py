# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The evidence the idempotent dependency pass records and reads back.

Every skip in install_python_stack.py is gated on three things: the previous run
recorded that it did this exact work, the inputs are byte-identical, and a cheap
on-disk check of the OUTPUT still passes. This file covers the pieces of that in
install_manifest.py -- the additive manifest keys, the digests, the constraint
check and the sidecar predicate -- because a false "already done" here is an
install nobody can tell apart from a finished one.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import sysconfig

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "studio" / "install_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_evidence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_module()


# -- digests -------------------------------------------------------------------


def test_pass_inputs_are_a_superset_of_the_tracked_requirements() -> None:
    """The pass reads more files than verify_install fingerprints.

    They stay two lists: verify_install compares the whole `requirement_files`
    dict, so widening THAT one reports every install in the field as
    `studio_install_requirements_changed` and buys each of them a repair pass.
    """
    assert im.PASS_INPUT_FILES[: len(im.TRACKED_REQUIREMENT_FILES)] == (
        im.TRACKED_REQUIREMENT_FILES
    )
    for extra in (
        "diffusers-pin.txt",
        "triton-kernels.txt",
        "single-env/constraints.txt",
    ):
        assert extra in im.PASS_INPUT_FILES
        assert extra not in im.TRACKED_REQUIREMENT_FILES


def test_the_shipped_requirements_tree_supplies_every_pass_input() -> None:
    """A name nothing on disk answers is evidence that can never match."""
    digests = im.pass_input_digests(REPO_ROOT / "studio" / "backend" / "requirements")
    assert set(digests) == set(im.PASS_INPUT_FILES)


def test_an_absent_or_unreadable_input_has_no_digest(tmp_path: pathlib.Path) -> None:
    """None never equals a recorded digest, so the step that reads it runs."""
    assert im.digest_file(tmp_path / "nope.txt") is None
    assert im.digest_file(tmp_path) is None
    target = tmp_path / "there.txt"
    target.write_bytes(b"payload")
    assert im.digest_file(target) == im.digest_file(str(target))
    assert (tmp_path / "nope.txt").name not in im.pass_input_digests(tmp_path)


def test_a_one_byte_edit_changes_the_digest(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "studio.txt"
    target.write_text("structlog\n", encoding = "utf-8")
    before = im.digest_file(target)
    target.write_text("structlog \n", encoding = "utf-8")
    assert im.digest_file(target) != before


# -- additive manifest keys ----------------------------------------------------


def _payload(root: pathlib.Path) -> dict:
    return json.loads((root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))


def test_extra_keys_are_written_and_schema_stays_1(tmp_path: pathlib.Path) -> None:
    im.write_manifest(
        root = tmp_path,
        req_root = tmp_path,
        package_name = "pytest",
        extra = {"pass_inputs": {"studio.txt": "abc"}, "pip_check_ok": True},
    )
    payload = _payload(tmp_path)
    assert payload["schema"] == im.MANIFEST_SCHEMA == 1
    assert payload["pass_inputs"] == {"studio.txt": "abc"}
    assert payload["pip_check_ok"] is True


def test_a_none_extra_is_left_out_entirely(tmp_path: pathlib.Path) -> None:
    """Absent means unknown for these keys too, exactly as for no_torch."""
    im.write_manifest(
        root = tmp_path, req_root = tmp_path, package_name = "pytest", extra = {"mlx_health": None}
    )
    assert "mlx_health" not in _payload(tmp_path)


def test_extras_cannot_shadow_a_field_verify_install_reads(tmp_path: pathlib.Path) -> None:
    """`extra` is evidence, never authority: package_version and requirement_files
    are what decides whether this install is complete."""
    im.write_manifest(
        root = tmp_path,
        req_root = tmp_path,
        package_name = "pytest",
        extra = {"package_version": "9.9.9", "requirement_files": {"studio.txt": "no"}, "schema": 7},
    )
    payload = _payload(tmp_path)
    assert payload["schema"] == 1
    assert payload["requirement_files"] == im.requirement_digests(tmp_path)
    assert payload["package_version"] != "9.9.9"


def test_update_manifest_merges_without_touching_the_rest(tmp_path: pathlib.Path) -> None:
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    before = _payload(tmp_path)
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is True
    after = _payload(tmp_path)
    assert after["mlx_health"] == {"ok": True}
    assert after["completed_at_ms"] == before["completed_at_ms"]
    assert after["requirement_files"] == before["requirement_files"]


def test_update_manifest_never_creates_one(tmp_path: pathlib.Path) -> None:
    """The manifest's presence means the install finished; a post-manifest probe
    must not be able to claim that on its own."""
    assert im.update_manifest(root = tmp_path, mlx_health = {"ok": True}) is False
    assert not (tmp_path / im.MANIFEST_NAME).exists()


def test_update_manifest_with_nothing_to_say_is_a_no_op(tmp_path: pathlib.Path) -> None:
    im.write_manifest(root = tmp_path, req_root = tmp_path, package_name = "pytest")
    assert im.update_manifest(root = tmp_path) is False
    assert im.update_manifest(root = tmp_path, mlx_health = None) is False


def test_update_manifest_survives_a_corrupt_manifest(tmp_path: pathlib.Path) -> None:
    (tmp_path / im.MANIFEST_NAME).write_text("{not json", encoding = "utf-8")
    assert im.update_manifest(root = tmp_path, pip_check_ok = True) is False
    assert (tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8") == "{not json"


# -- constraints ---------------------------------------------------------------


def test_an_absent_distribution_does_not_violate_a_constraint(tmp_path: pathlib.Path) -> None:
    """A constraints file never asks for an install, so nothing to install is
    nothing to repair -- otherwise every pass would run forever."""
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("nonexistent-dist-xyz<2\n", encoding = "utf-8")
    assert im.violated_constraints(constraints, installed = {}) == []


def test_a_resident_version_outside_the_window_is_reported(tmp_path: pathlib.Path) -> None:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        "# a comment\n\npytest>=99\nanyio<1  # inline\n-r other.txt\n", encoding = "utf-8"
    )
    violated = im.violated_constraints(constraints, installed = {"pytest": "8.0.0", "anyio": "0.1"})
    assert violated == ["pytest"]


def test_an_unpinned_or_unreadable_constraints_file_reports_nothing(tmp_path: pathlib.Path) -> None:
    bare = tmp_path / "bare.txt"
    bare.write_text("pytest\n", encoding = "utf-8")
    assert im.violated_constraints(bare, installed = {"pytest": "8.0.0"}) == []
    assert im.violated_constraints(tmp_path / "gone.txt", installed = {}) == []


def test_the_shipped_constraints_file_parses() -> None:
    """Not an assertion about this machine's venv, only that the real file is
    readable by the parser the skip gate relies on."""
    real = REPO_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
    assert isinstance(im.violated_constraints(real, installed = {}), list)


# -- sidecar predicate ---------------------------------------------------------


def _sidecar(
    root: pathlib.Path,
    name: str,
    version: str,
    *,
    files = ("__init__.py",),
) -> None:
    """A flat `pip --target` tree: package dir, dist-info, RECORD with sizes."""
    package = root / name.replace("-", "_")
    package.mkdir(parents = True, exist_ok = True)
    rows = []
    for relative in files:
        target = package / relative
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 32)
        rows.append(f"{package.name}/{relative},sha256=deadbeef,32")
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8"
    )
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding = "utf-8")


@pytest.fixture
def sidecar(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / ".venv_t5_530"
    root.mkdir()
    (root / im.SIDECAR_OWNED_MARKER).write_text("", encoding = "utf-8")
    _sidecar(root, "transformers", "5.3.0")
    _sidecar(root, "huggingface_hub", "1.8.0")
    _sidecar(root, "hf_xet", "1.4.2")
    _sidecar(root, "tiktoken", "0.9.0")
    return root


PINS = ("transformers==5.3.0", "huggingface_hub==1.8.0", "hf_xet==1.4.2", "tiktoken")


def test_a_complete_sidecar_is_current(sidecar: pathlib.Path) -> None:
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_a_missing_or_empty_directory_is_not_current(tmp_path: pathlib.Path) -> None:
    assert im.sidecar_is_current(tmp_path / "gone", PINS)[0] is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert im.sidecar_is_current(empty, PINS) == (False, "empty")


def test_an_unowned_directory_is_never_current(sidecar: pathlib.Path) -> None:
    """Rebuilding starts with rm -rf, so "current" has to mean "and ours"."""
    (sidecar / im.SIDECAR_OWNED_MARKER).unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and im.SIDECAR_OWNED_MARKER in reason


def test_a_wrong_version_names_itself(sidecar: pathlib.Path) -> None:
    current, reason = im.sidecar_is_current(sidecar, ("transformers==5.5.0",) + PINS[1:])
    assert current is False
    assert "transformers==5.3.0" in reason and "5.5.0" in reason


def test_an_unpinned_name_only_has_to_be_present(sidecar: pathlib.Path) -> None:
    assert im.sidecar_is_current(sidecar, ("tiktoken",)) == (True, "")
    current, reason = im.sidecar_is_current(sidecar, ("regex",))
    assert current is False and "regex" in reason


def test_metadata_without_its_package_tree_is_not_current(sidecar: pathlib.Path) -> None:
    """The failure mode the version check alone cannot see: an interrupted pip
    leaves METADATA behind and takes the module with it."""
    import shutil

    shutil.rmtree(sidecar / "transformers")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "directory missing" in reason


def test_a_truncated_recorded_file_forces_a_rebuild(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").write_bytes(b"")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False
    assert "0 bytes, expected 32" in reason


def test_a_deleted_recorded_file_forces_a_rebuild(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").unlink()
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "is missing" in reason


def test_a_larger_file_is_a_collision_not_damage(sidecar: pathlib.Path) -> None:
    (sidecar / "transformers" / "__init__.py").write_bytes(b"x" * 64)
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_an_extension_built_for_another_interpreter_is_rejected(
    tmp_path: pathlib.Path, sidecar: pathlib.Path
) -> None:
    """A sidecar survives a Python upgrade intact and stops importing."""
    tag = "{}{}{}".format(
        sys.version_info.major,
        sys.version_info.minor,
        "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "",
    )
    foreign = "990" if not tag.startswith("99") else "980"
    _sidecar(
        sidecar,
        "regex",
        "2024.1.1",
        files = ("__init__.py", f"_regex.cpython-{foreign}-x86_64-linux-gnu.so"),
    )
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False
    assert f"targets cp{foreign}" in reason and f"interpreter is cp{tag}" in reason


def test_a_tag_in_a_directory_name_says_nothing_about_the_binary(sidecar: pathlib.Path) -> None:
    """Wiping several hundred MB over a directory name is the false positive this
    check must not have."""
    _sidecar(sidecar, "regex", "2024.1.1", files = ("__init__.py", "regex.cp312.libs/libhelper.so"))
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_every_distribution_is_scanned_not_only_the_pinned_ones(sidecar: pathlib.Path) -> None:
    """The whole directory goes on sys.path, so a truncated regex/ shadows the
    base install and breaks `import transformers` just as a truncated
    transformers/ does."""
    _sidecar(sidecar, "regex", "2024.1.1")
    (sidecar / "regex" / "__init__.py").write_bytes(b"")
    current, reason = im.sidecar_is_current(sidecar, PINS)
    assert current is False and "regex" in reason


def test_a_sidecar_with_no_record_is_left_alone(sidecar: pathlib.Path) -> None:
    """An absent RECORD says nothing about damage, and the answer costs a
    several-hundred-MB refetch."""
    (sidecar / "transformers-5.3.0.dist-info" / "RECORD").unlink()
    assert im.sidecar_is_current(sidecar, PINS) == (True, "")


def test_the_scan_is_bounded(sidecar: pathlib.Path) -> None:
    """No installer wraps this in a timeout, so a stalled mount must not wedge setup."""
    assert im.sidecar_is_current(sidecar, PINS, budget_seconds = 0.0)[0] is True


# -- the CLI shim both shells call ---------------------------------------------


def _shim(*args: str):
    import subprocess
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), *args],
        capture_output = True,
        text = True,
        encoding = "utf-8",
    )


def test_the_shim_reports_current(sidecar: pathlib.Path) -> None:
    result = _shim("sidecar", str(sidecar), *PINS)
    assert result.returncode == 0
    assert result.stdout.strip() == "sidecar: current"


def test_the_shim_reports_the_reason_and_exits_1(sidecar: pathlib.Path) -> None:
    result = _shim("sidecar", str(sidecar), "transformers==5.5.0")
    assert result.returncode == 1
    assert result.stdout.startswith("sidecar: ")
    assert "5.5.0" in result.stdout


def test_the_shim_marks_its_own_output(sidecar: pathlib.Path) -> None:
    """An install_manifest.py predating the shim has no __main__ block at all, so
    running it exits 0 with no output. Both shells require the marker line before
    believing exit 0, or that silence reads as "current" and no sidecar is ever
    rebuilt again."""
    assert im._SIDECAR_CLI_MARKER == "sidecar:"
    for args in ((), ("sidecar",), ("nonsense",), ("sidecar", str(sidecar))):
        result = _shim(*args)
        if result.returncode == 2:
            assert not result.stdout.strip().startswith(im._SIDECAR_CLI_MARKER)


def test_the_shim_refuses_what_it_does_not_implement() -> None:
    for args in ((), ("nonsense",), ("sidecar",)):
        assert _shim(*args).returncode == 2
