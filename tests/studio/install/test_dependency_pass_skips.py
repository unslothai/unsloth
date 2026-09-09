# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dependency pass must be idempotent: a second `unsloth studio update` does no
network work and leaves the install byte-identical.

Every skip is legal only when (a) the previous run recorded that it did this exact work,
(b) the inputs are byte-identical, and (c) a cheap on-disk check of the output passes.
This file is the unit half of that; the end-to-end half is
tests/studio/install/test_update_idempotency.py, which runs a real install.

The direction that matters is asymmetric. A needless install costs seconds. A wrong skip
ships a venv that answers `-h` and dies on `import structlog`, which is exactly the
failure the manifest was added to catch -- so every case below that cannot prove the
work was done asserts that the work runs.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import platform
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STACK_PATH = REPO_ROOT / "studio" / "install_python_stack.py"


def _load():
    sys.path.insert(0, str(REPO_ROOT / "studio"))
    try:
        spec = importlib.util.spec_from_file_location("studio_stack_pass_skips", STACK_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT / "studio"):
            sys.path.pop(0)
    return module


stack = _load()


# -- the plan ------------------------------------------------------------------


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    """A manifest and requirements tree that _plan_pass would accept."""
    req_root = tmp_path / "requirements"
    (req_root / "single-env").mkdir(parents = True)
    for name in stack.install_manifest.PASS_INPUT_FILES:
        (req_root / name).write_text(f"# {name}\n", encoding = "utf-8")
    monkeypatch.setattr(stack, "REQ_ROOT", req_root)
    monkeypatch.setattr(stack, "CONSTRAINTS", req_root / "single-env" / "constraints.txt")
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "IS_MAC_ARM", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "PLATFORM_LACKS_TORCHCODEC_WHEEL", False)
    monkeypatch.setattr(stack, "_expected_torch_flavor_tag", lambda: "cpu")
    monkeypatch.setattr(stack, "_recordable_torch_flavor_tag", lambda tag: tag)
    payload = {
        "schema": stack.install_manifest.MANIFEST_SCHEMA,
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "no_torch": False,
        "expected_torch_tag": "cpu",
        "pass_inputs": stack.install_manifest.pass_input_digests(req_root),
        "step_results": {name: "ran" for name in stack.install_manifest.PASS_INPUT_FILES},
    }
    monkeypatch.setattr(stack.install_manifest, "read_manifest", lambda *a, **k: dict(payload))
    monkeypatch.setattr(
        stack.install_manifest, "verify_install", lambda *a, **k: {"ok": True, "reason": None}
    )
    return payload, req_root


def _plan(**kwargs):
    return stack._plan_pass(
        kwargs.pop("package_name", "unsloth"), kwargs.pop("local_repo", ""), kwargs.pop("ci", "")
    )


def test_a_clean_manifest_plans_a_skippable_pass(manifest) -> None:
    plan = _plan()
    assert plan is not None
    assert plan["pass_inputs"] == manifest[0]["pass_inputs"]


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_full_deps_forces_everything(monkeypatch, manifest, value) -> None:
    """The escape hatch. A skip nobody can turn off is a bug nobody can work around."""
    monkeypatch.setenv(stack._FULL_DEPS_ENV, value)
    assert _plan() is None


@pytest.mark.parametrize("value", ["0", "false", "", "maybe"])
def test_a_non_true_full_deps_value_changes_nothing(monkeypatch, manifest, value) -> None:
    monkeypatch.setenv(stack._FULL_DEPS_ENV, value)
    assert _plan() is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_repo": "/checkout"},
        {"ci": "1"},
        {"package_name": "unsloth-nightly"},
    ],
)
def test_dev_shapes_never_skip(manifest, kwargs) -> None:
    """An editable overlay or another package name means the tree on disk is not the
    tree the manifest describes, and its digests describe neither."""
    assert _plan(**kwargs) is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": 2},
        {"pass_inputs": None},
        {"step_results": None},
        {"pass_inputs": "not a dict"},
        {"python": "3.0.0"},
        {"platform": "sunos-vax"},
        {"no_torch": True},
        {"no_torch": None},
        {"expected_torch_tag": "cu128"},
        {"expected_torch_tag": None},
    ],
)
def test_a_moved_input_forces_a_full_pass(monkeypatch, manifest, mutation) -> None:
    payload, _req_root = manifest
    payload.update(mutation)
    monkeypatch.setattr(stack.install_manifest, "read_manifest", lambda *a, **k: dict(payload))
    assert _plan() is None


def test_no_manifest_at_all_forces_a_full_pass(monkeypatch, manifest) -> None:
    monkeypatch.setattr(stack.install_manifest, "read_manifest", lambda *a, **k: None)
    assert _plan() is None


def test_a_damaged_install_forces_a_full_pass(monkeypatch, manifest) -> None:
    """verify_install(deep=True) is the only check that walks the filesystem, and it is
    the one that catches a payload quarantined after the manifest was written."""
    monkeypatch.setattr(
        stack.install_manifest,
        "verify_install",
        lambda *a, **k: {"ok": False, "reason": "studio_install_damaged"},
    )
    assert _plan() is None


def test_a_probe_that_raises_forces_a_full_pass(monkeypatch, manifest) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("the venv is unreadable")

    monkeypatch.setattr(stack.install_manifest, "verify_install", boom)
    assert _plan() is None
    monkeypatch.setattr(stack, "_expected_torch_flavor_tag", boom)
    assert _plan() is None


# -- one step's evidence -------------------------------------------------------


@pytest.fixture
def gated(monkeypatch, manifest):
    payload, req_root = manifest
    monkeypatch.setattr(stack, "_PASS_EVIDENCE", _plan())
    monkeypatch.setattr(stack, "_CONSTRAINTS_CACHE", None)
    monkeypatch.setattr(stack, "_INSTALL_ACTIONS", 0)
    monkeypatch.setattr(stack.install_manifest, "missing_requirements", lambda *a, **k: [])
    monkeypatch.setattr(stack.install_manifest, "violated_constraints", lambda *a, **k: [])
    return payload, req_root


def test_a_step_with_unchanged_inputs_is_skipped(gated) -> None:
    _payload, req_root = gated
    assert stack._requirements_satisfied(req_root / "studio.txt") is True


def test_nothing_is_skipped_without_evidence(monkeypatch, gated) -> None:
    _payload, req_root = gated
    monkeypatch.setattr(stack, "_PASS_EVIDENCE", None)
    assert stack._requirements_satisfied(req_root / "studio.txt") is False


def test_a_step_the_last_run_never_reached_is_not_skipped(monkeypatch, gated) -> None:
    """The manifest is written at the end, so a pass killed after step 8 records
    nothing for step 9. Absent must never read as done."""
    evidence = dict(stack._PASS_EVIDENCE)
    evidence["step_results"] = {
        key: value for key, value in evidence["step_results"].items() if key != "studio.txt"
    }
    monkeypatch.setattr(stack, "_PASS_EVIDENCE", evidence)
    assert stack._requirements_satisfied(gated[1] / "studio.txt") is False


def test_a_one_byte_edit_to_the_file_runs_the_step(gated) -> None:
    _payload, req_root = gated
    (req_root / "studio.txt").write_text("# studio.txt \n", encoding = "utf-8")
    assert stack._requirements_satisfied(req_root / "studio.txt") is False


def test_a_moved_constraint_runs_every_constrained_step(gated) -> None:
    """A constraint that moved under an unchanged requirements file is exactly what
    digest equality on the file alone cannot see."""
    _payload, req_root = gated
    (req_root / "single-env" / "constraints.txt").write_text("numpy<2\n", encoding = "utf-8")
    assert stack._requirements_satisfied(req_root / "studio.txt") is False
    # ...but not the unconstrained one, which never reads that file.
    assert stack._requirements_satisfied(req_root / "triton-kernels.txt", constrain = False) is True


def test_the_mac_overrides_file_counts_on_mac_arm(monkeypatch, gated) -> None:
    """It reaches uv through UV_OVERRIDE rather than the command line, so nothing else
    in the recorded inputs would notice it moving."""
    _payload, req_root = gated
    monkeypatch.setattr(stack, "IS_MAC_ARM", True)
    assert stack._requirements_satisfied(req_root / "studio.txt") is True
    (req_root / "single-env" / "overrides-darwin-arm64.txt").write_text("x\n", encoding = "utf-8")
    assert stack._requirements_satisfied(req_root / "studio.txt") is False


def test_an_uninstalled_requirement_runs_the_step(monkeypatch, gated) -> None:
    """(c): the output has to still be on disk. A venv edited after the install is the
    case a recorded digest cannot see."""
    _payload, req_root = gated
    monkeypatch.setattr(stack.install_manifest, "missing_requirements", lambda *a, **k: ["rich"])
    assert stack._requirements_satisfied(req_root / "studio.txt") is False


def test_a_violated_constraint_runs_the_step(monkeypatch, gated) -> None:
    _payload, req_root = gated
    monkeypatch.setattr(stack.install_manifest, "violated_constraints", lambda *a, **k: ["numpy"])
    monkeypatch.setattr(stack, "_CONSTRAINTS_CACHE", None)
    assert stack._requirements_satisfied(req_root / "studio.txt") is False


def test_a_file_outside_the_requirements_tree_is_never_skipped(gated) -> None:
    """--local points base.txt somewhere else; there is no manifest key for it."""
    _payload, req_root = gated
    assert stack._requirements_satisfied(req_root.parent / "elsewhere.txt") is False


def test_the_constraint_scan_is_redone_after_an_install(monkeypatch, gated) -> None:
    """Cached per pass, because every gated step asks -- but a step that installed
    something can have satisfied or broken a constraint the next step reads."""
    answers = iter([["numpy"], []])
    monkeypatch.setattr(
        stack.install_manifest, "violated_constraints", lambda *a, **k: next(answers)
    )
    monkeypatch.setattr(stack, "_CONSTRAINTS_CACHE", None)
    assert stack._violated_constraints() == ["numpy"]
    assert stack._violated_constraints() == ["numpy"]  # cached
    stack._count_install_action()
    assert stack._violated_constraints() == []


def test_the_progress_slot_is_spent_either_way(monkeypatch, gated) -> None:
    """The denominator cannot depend on how much of the install was already there."""
    _payload, req_root = gated
    seen: list[str] = []
    monkeypatch.setattr(stack, "_progress", seen.append)
    assert stack._skip_step(req_root / "studio.txt", "studio deps") is True
    monkeypatch.setattr(stack, "_PASS_EVIDENCE", None)
    assert stack._skip_step(req_root / "studio.txt", "studio deps") is False
    assert seen == ["studio deps (satisfied, skipped)", "studio deps"]


def test_an_extra_check_can_veto_a_skip(monkeypatch, gated) -> None:
    _payload, req_root = gated
    monkeypatch.setattr(stack, "_progress", lambda *_a: None)
    assert (
        stack._skip_step(req_root / "studio.txt", "studio deps", extra_check = lambda: False) is False
    )
    assert (
        stack._skip_step(req_root / "studio.txt", "studio deps", extra_check = lambda: True) is True
    )


# -- the effective requirements file -------------------------------------------


def test_the_gate_audits_the_file_the_install_would_use(monkeypatch, tmp_path) -> None:
    """extras.txt names torchcodec and three platforms filter it out. Auditing the raw
    file would report it missing on every one of them and never skip anything again."""
    req = tmp_path / "extras.txt"
    req.write_text("rich\ntorchcodec\n", encoding = "utf-8")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "PLATFORM_LACKS_TORCHCODEC_WHEEL", True)
    effective, temps = stack._effective_requirements(req)
    try:
        assert effective != req
        assert effective.read_text(encoding = "utf-8") == "rich\n"
    finally:
        for temp in temps:
            temp.unlink(missing_ok = True)


def test_an_unfiltered_file_is_used_as_is(monkeypatch, tmp_path) -> None:
    req = tmp_path / "extras.txt"
    req.write_text("rich\n", encoding = "utf-8")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "PLATFORM_LACKS_TORCHCODEC_WHEEL", False)
    effective, temps = stack._effective_requirements(req)
    assert effective == req and temps == []


def test_pip_install_and_the_gate_share_one_filter() -> None:
    """Two copies of the filter list is how the two would disagree, and a disagreement
    here reads as "already installed" for a package the install never had."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    body = source[source.index("def pip_install(") : source.index("def download_file(")]
    assert "_effective_requirements(req)" in body
    assert "_filter_requirements(" not in body


# -- the git requirement -------------------------------------------------------


def test_a_git_requirement_is_matched_by_ref_not_version(monkeypatch, tmp_path) -> None:
    """The same version is published from every branch, so a release/3.6.x pin is
    satisfied on paper by a build from main. direct_url.json is the only place the ref
    survives the install."""
    req = tmp_path / "triton-kernels.txt"
    req.write_text(
        "# comment\n"
        "triton_kernels @ git+https://example.invalid/triton.git@release/3.6.x"
        "#subdirectory=python/triton_kernels\n",
        encoding = "utf-8",
    )
    assert stack._direct_reference_in_requirements(req) == (
        "https://example.invalid/triton.git",
        "release/3.6.x",
        "python/triton_kernels",
    )

    recorded = {
        "url": "https://example.invalid/triton.git",
        "subdirectory": "python/triton_kernels",
        "vcs_info": {"vcs": "git", "requested_revision": "release/3.6.x"},
    }

    class _Dist:
        def __init__(self, payload):
            self._payload = payload

        def read_text(self, _name):
            return json.dumps(self._payload) if self._payload is not None else None

    import importlib.metadata

    def _distribution(payload):
        return lambda _name: _Dist(payload)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution(recorded))
    assert stack._direct_reference_is_installed(req, "triton_kernels") is True

    for broken in (
        {**recorded, "vcs_info": {"vcs": "git", "requested_revision": "main"}},
        {**recorded, "url": "https://example.invalid/other.git"},
        {**recorded, "subdirectory": "python/other"},
        {"url": recorded["url"]},
        None,
    ):
        monkeypatch.setattr(importlib.metadata, "distribution", _distribution(broken))
        assert stack._direct_reference_is_installed(req, "triton_kernels") is False


def test_userinfo_in_a_git_url_is_not_mistaken_for_a_ref(tmp_path) -> None:
    req = tmp_path / "t.txt"
    req.write_text("pkg @ git+ssh://git@example.invalid/repo.git\n", encoding = "utf-8")
    assert stack._direct_reference_in_requirements(req) == (
        "ssh://git@example.invalid/repo.git",
        "",
        "",
    )


def test_the_shipped_triton_requirement_still_parses() -> None:
    parsed = stack._direct_reference_in_requirements(
        REPO_ROOT / "studio" / "backend" / "requirements" / "triton-kernels.txt"
    )
    assert parsed is not None
    url, revision, subdirectory = parsed
    assert url.startswith("https://") and revision and subdirectory


# -- local plugins -------------------------------------------------------------


def test_a_plugin_digest_is_stable_and_content_addressed(tmp_path) -> None:
    """A directory listing has no order, so a digest that depended on one would differ
    between two byte-identical trees and reinstall both seed plugins every update."""
    first = tmp_path / "a"
    (first / "src").mkdir(parents = True)
    (first / "pyproject.toml").write_text("name = 'x'\n", encoding = "utf-8")
    (first / "src" / "mod.py").write_text("VALUE = 1\n", encoding = "utf-8")
    second = tmp_path / "b"
    (second / "src").mkdir(parents = True)
    (second / "src" / "mod.py").write_text("VALUE = 1\n", encoding = "utf-8")
    (second / "pyproject.toml").write_text("name = 'x'\n", encoding = "utf-8")

    assert stack._local_plugin_digest(first) == stack._local_plugin_digest(second)
    (second / "src" / "mod.py").write_text("VALUE = 2\n", encoding = "utf-8")
    assert stack._local_plugin_digest(first) != stack._local_plugin_digest(second)


def test_a_renamed_file_changes_the_plugin_digest(tmp_path) -> None:
    """Path and bytes both, so moving a file between two names is not a no-op."""
    plugin = tmp_path / "p"
    plugin.mkdir()
    (plugin / "one.py").write_text("x\n", encoding = "utf-8")
    before = stack._local_plugin_digest(plugin)
    (plugin / "one.py").rename(plugin / "two.py")
    assert stack._local_plugin_digest(plugin) != before


def test_the_shipped_plugins_digest(tmp_path) -> None:
    for plugin in (stack.LOCAL_DD_UNSTRUCTURED_PLUGIN, stack.LOCAL_DD_GITHUB_PLUGIN):
        assert stack._local_plugin_digest(plugin)


def test_an_unreadable_plugin_has_no_digest(tmp_path) -> None:
    assert stack._local_plugin_digest(tmp_path / "gone") is None


# -- the pip bootstrap ---------------------------------------------------------


def test_the_uv_probe_resolves_nothing(monkeypatch) -> None:
    """`uv pip install --dry-run pip` answered "can this uv address this interpreter"
    by resolving pip against the index -- one PyPI round trip on every installer run,
    and a hard failure on an offline host uv could have served from its cache."""
    seen: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(stack.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(stack.subprocess, "run", lambda cmd, **_k: (seen.append(cmd), _Result())[1])
    assert stack._bootstrap_uv() is True
    assert seen == [["uv", "pip", "freeze", "--python", sys.executable]]
    assert not any("--dry-run" in cmd for cmd in seen)
    assert not any("install" in cmd for cmd in seen)


def test_the_uv_probe_still_retries_with_system(monkeypatch) -> None:
    seen: list[list[str]] = []
    codes = iter([1, 0])

    class _Result:
        def __init__(self, code):
            self.returncode = code

    monkeypatch.setattr(stack.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        stack.subprocess, "run", lambda cmd, **_k: (seen.append(cmd), _Result(next(codes)))[1]
    )
    monkeypatch.setattr(stack, "UV_NEEDS_SYSTEM", False)
    assert stack._bootstrap_uv() is True
    assert stack.UV_NEEDS_SYSTEM is True
    assert seen[1] == ["uv", "pip", "freeze", "--system"]


def test_a_broken_uv_still_falls_back_to_pip(monkeypatch) -> None:
    class _Result:
        returncode = 2

    monkeypatch.setattr(stack.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(stack.subprocess, "run", lambda cmd, **_k: _Result())
    assert stack._bootstrap_uv() is False


def test_a_venv_that_already_has_pip_does_not_reinstall_it(monkeypatch) -> None:
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: "24.2")
    assert stack._venv_pip_is_usable() is True
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: "22.0.4")
    assert stack._venv_pip_is_usable() is False
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: None)
    assert stack._venv_pip_is_usable() is False
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: "not-a-version")
    assert stack._venv_pip_is_usable() is False


def test_a_fresh_uv_venv_still_bootstraps(monkeypatch) -> None:
    """uv venvs omit pip entirely, and that is the case this step exists for."""
    monkeypatch.setattr(stack.importlib.util, "find_spec", lambda _n: None)
    assert stack._venv_pip_is_usable() is False


# -- MLX and torchcodec --------------------------------------------------------


def test_the_mlx_stack_is_current_only_when_all_four_hold(monkeypatch) -> None:
    versions = {"mlx": "0.32.1", "mlx-metal": "0.32.1", "mlx-lm": "0.31.3", "mlx-vlm": "0.5.0"}
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda name: versions.get(name))
    assert stack._mlx_stack_is_current() is True
    for name, wrong in (
        ("mlx", "0.32.0"),
        ("mlx-metal", None),
        ("mlx-lm", "0.31.2"),
        ("mlx-vlm", "0.8.0"),
        ("mlx-vlm", None),
    ):
        broken = dict(versions)
        broken[name] = wrong
        monkeypatch.setattr(stack, "_installed_distribution_version", lambda n, b = broken: b.get(n))
        assert stack._mlx_stack_is_current() is False, name


def test_the_mlx_pins_match_the_runtime_repair() -> None:
    """One stack, two callers. The self-heal at startup reinstalls whatever this
    installed, so a pin that drifts here is a repair loop there."""
    names = {spec.partition("==")[0] for spec in stack._MLX_PINS} | {"mlx-vlm"}
    assert set(stack._MLX_NAMES) == names
    assert stack._MLX_VLM_SPEC.startswith("mlx-vlm")


@pytest.mark.parametrize(
    "spec,installed,expected",
    [
        ("torchcodec==0.8.0", "0.8.0", True),
        ("torchcodec==0.8.0", "0.8.0+cu130", True),
        ("torchcodec==0.8.0", "0.7.0", False),
        ("torchcodec>=0.7,<0.9", "0.8.0+cpu", True),
        ("torchcodec>=0.7,<0.9", "0.9.1", False),
        ("torchcodec==0.8.0", "", False),
    ],
)
def test_a_resident_codec_inside_the_window_needs_no_install(spec, installed, expected) -> None:
    """The local tag is ignored here on purpose: provenance is the caller's separate
    question, and a +cu130 suffix puts the version outside a PyPI-shaped specifier it
    in fact satisfies."""
    assert stack._codec_spec_is_satisfied(spec, installed) is expected


# -- the pip fallback ----------------------------------------------------------


def test_the_pip_fallback_never_names_one_project_twice() -> None:
    """pip refuses `mlx==0.32.1 mlx` outright with "Double requirement given", which
    would fail the very step the fallback exists to rescue."""
    cmd = stack._build_pip_cmd(
        (
            "--upgrade-package",
            "mlx",
            "--upgrade-package",
            "mlx-vlm",
            "mlx==0.32.1",
            "mlx-vlm>=0.4.4,<0.7.0",
        )
    )
    assert "--upgrade" in cmd and "--upgrade-package" not in cmd
    assert cmd.count("mlx") == 0 and cmd.count("mlx-vlm") == 0
    assert "mlx==0.32.1" in cmd and "mlx-vlm>=0.4.4,<0.7.0" in cmd


def test_a_package_named_only_by_the_flag_is_still_passed() -> None:
    """pip would otherwise upgrade nothing at all."""
    cmd = stack._build_pip_cmd(("--upgrade-package", "future-pkg", "other"))
    assert "future-pkg" in cmd and "other" in cmd


# -- accounting ----------------------------------------------------------------


def test_every_install_entry_point_is_counted() -> None:
    """`pip check` and the metadata patch are gated on this counter, so a new install
    site that does not increment it makes both skip a venv that just changed."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    counted = {"pip_install", "pip_install_try", "_uninstall_distribution"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in counted:
            body = ast.dump(node)
            assert "_count_install_action" in body, node.name
            counted.discard(node.name)
    assert not counted, f"install entry points are gone: {counted}"


def test_the_pass_state_is_reset_per_call() -> None:
    """The suites call install_python_stack() several times in one interpreter, and a
    counter that survived would make the second call skip what the first installed."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    entry = source[source.index("def install_python_stack() -> int:") :]
    head = entry[: entry.index("# --package installs")]
    for name in ("_INSTALL_ACTIONS = 0", "_PASS_EVIDENCE = None", "_STEP_RESULTS.clear()"):
        assert name in head, name


def test_the_plan_is_read_before_the_manifest_is_dropped() -> None:
    """remove_manifest deletes the only copy of what the last run did."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    plan = source.index("_PASS_EVIDENCE = _plan_pass(")
    drop = source.index("install_manifest.remove_manifest()")
    assert plan < drop


# -- the finalizing tail -------------------------------------------------------
#
# Three fixed costs at the end of every pass that a settled install has no reason to
# pay: a subprocess that rewrites three METADATA files it already rewrote, a full
# metadata resolve of the venv by `pip check` whose result is discarded, and an out of
# process probe that imports torch, mlx, mlx_lm and mlx_vlm to reprint the same verdict.


class _FakePatchModule:
    """Stands in for requirements/single-env/patch_metadata.py."""

    TARGETS = ("data-designer",)
    PATCHES = ((re.compile(r"^Requires-Dist: huggingface-hub<2,>=1\.0\.1$", re.MULTILINE), "x"),)

    def __init__(
        self,
        path,
        *,
        raise_on_lookup = False,
        raise_on_main = False,
    ):
        self._path = path
        self._raise_on_lookup = raise_on_lookup
        self._raise_on_main = raise_on_main
        self.main_calls = 0

    def metadata_path(self, _name):
        if self._raise_on_lookup:
            raise RuntimeError("distribution metadata is unreadable")
        return self._path

    def main(self):
        self.main_calls += 1
        if self._raise_on_main:
            raise RuntimeError("cannot patch in process")
        return 0


PATCHED = "Requires-Dist: huggingface-hub<2,>=0.34.0\n"
UNPATCHED = "Requires-Dist: huggingface-hub<2,>=1.0.1\n"


def _patch_module(
    monkeypatch,
    tmp_path,
    text = PATCHED,
    **kwargs,
):
    metadata = tmp_path / "METADATA"
    if text is not None:
        metadata.write_text(text, encoding = "utf-8")
    fake = _FakePatchModule(metadata if text is not None else None, **kwargs)
    monkeypatch.setattr(stack, "SINGLE_ENV", tmp_path)
    # The real import is cached in sys.modules after the first pass, so this is also
    # how the second install_python_stack() call in one interpreter sees it.
    monkeypatch.setitem(sys.modules, "patch_metadata", fake)
    return fake


def test_a_settled_install_does_not_re_run_the_metadata_patch(monkeypatch, tmp_path) -> None:
    _patch_module(monkeypatch, tmp_path, PATCHED)
    assert stack._patch_metadata_is_pending() is False


def test_an_unpatched_metadata_file_is_still_pending(monkeypatch, tmp_path) -> None:
    _patch_module(monkeypatch, tmp_path, UNPATCHED)
    assert stack._patch_metadata_is_pending() is True


def test_a_distribution_that_is_not_installed_needs_no_patch(monkeypatch, tmp_path) -> None:
    _patch_module(monkeypatch, tmp_path, None)
    assert stack._patch_metadata_is_pending() is False


@pytest.mark.parametrize("kwargs", [{"raise_on_lookup": True}])
def test_anything_it_cannot_answer_runs_the_patch(monkeypatch, tmp_path, kwargs) -> None:
    """Unknown means do the work: this replaces an unconditional run, so the failure
    mode of the question must be the old behaviour, not a skip."""
    _patch_module(monkeypatch, tmp_path, PATCHED, **kwargs)
    assert stack._patch_metadata_is_pending() is True


def test_an_unimportable_patch_module_runs_the_patch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stack, "SINGLE_ENV", tmp_path / "gone")
    monkeypatch.delitem(sys.modules, "patch_metadata", raising = False)
    assert stack._patch_metadata_is_pending() is True


def test_the_patch_applies_in_process(monkeypatch, tmp_path) -> None:
    fake = _patch_module(monkeypatch, tmp_path, UNPATCHED)
    ran = []
    monkeypatch.setattr(stack, "run", lambda *a, **k: ran.append(a))
    stack._run_patch_metadata()
    assert fake.main_calls == 1
    assert ran == []


def test_the_patch_still_falls_back_to_the_subprocess(monkeypatch, tmp_path) -> None:
    """It is a supported standalone entry point, and an import failure inside the
    installer must not turn into a failed install."""
    fake = _patch_module(monkeypatch, tmp_path, UNPATCHED, raise_on_main = True)
    ran = []
    monkeypatch.setattr(stack, "run", lambda *a, **k: ran.append(a))
    stack._run_patch_metadata()
    assert fake.main_calls == 1
    assert len(ran) == 1 and str(tmp_path / "patch_metadata.py") in [str(x) for x in ran[0][1]]


@pytest.mark.parametrize("fn", ["_patch_metadata_is_pending", "_run_patch_metadata"])
def test_neither_helper_leaves_single_env_on_sys_path(monkeypatch, tmp_path, fn) -> None:
    """sys.path.insert(0, SINGLE_ENV) that outlives the call shadows stdlib names for
    the rest of the install."""
    _patch_module(monkeypatch, tmp_path, UNPATCHED)
    monkeypatch.setattr(stack, "run", lambda *a, **k: None)
    before = list(sys.path)
    getattr(stack, fn)()
    assert sys.path == before


def test_the_metadata_patch_still_runs_whenever_data_designer_did() -> None:
    """The pending scan reads the METADATA of packages the data-designer steps just
    wrote, so a fresh install must not be gated on it at all."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    assert "_finalize_ran = _dd_deps_ran or _dd_ran or _patch_metadata_is_pending()" in source


def test_pip_check_is_gated_on_this_pass_having_changed_something() -> None:
    source = STACK_PATH.read_text(encoding = "utf-8")
    gate = source.index('_pip_check_ok = (_PASS_EVIDENCE or {}).get("pip_check_ok")')
    tail = source[gate : gate + 400]
    # Both halves: a pass that installed anything re-checks, and so does one whose last
    # recorded answer was not a clean True (missing, False, or never recorded).
    assert "if _INSTALL_ACTIONS > 0 or _pip_check_ok is not True:" in tail
    assert '"pip_check_ok": _pip_check_ok,' in source


# -- the MLX verdict -----------------------------------------------------------


class _FakeSubprocess:
    def __init__(self, stdout = "[]"):
        self.calls = 0
        self._stdout = stdout

    def run(self, *args, **kwargs):
        self.calls += 1
        return type("R", (), {"stdout": self._stdout, "stderr": "", "returncode": 0})()


@pytest.fixture
def mlx(monkeypatch):
    fake = _FakeSubprocess()
    monkeypatch.setattr(stack, "subprocess", fake)
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: "0.4.5")
    steps: list[tuple[str, str]] = []
    monkeypatch.setattr(stack, "_step", lambda label, value, *a: steps.append((label, value)))
    written: list[dict] = []
    monkeypatch.setattr(
        stack.install_manifest,
        "update_manifest",
        lambda **kw: written.append(kw.get("mlx_health")) or True,
    )
    healthy = {**stack._mlx_health_fingerprint(), "ok": True}
    monkeypatch.setattr(stack, "_PASS_EVIDENCE", {"mlx_health": healthy})
    return fake, steps, written, healthy


def test_a_recorded_healthy_stack_is_not_re_probed(mlx) -> None:
    fake, steps, written, _ = mlx
    stack._report_mlx_stack_health(skipped = True)
    assert fake.calls == 0
    assert steps == [("mlx", "training stack ready")]
    # Rewritten so the record does not age out of the manifest this pass just wrote.
    assert written and written[0]["ok"] is True


def test_a_rebuilt_mlx_stack_is_always_probed(mlx) -> None:
    fake, steps, _written, _ = mlx
    stack._report_mlx_stack_health(skipped = False)
    assert fake.calls == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"ok": False},
        {"ok": None},
        {"pins": ["mlx==0.0.1"]},
        {"python": "39"},
        {"mlx_vlm": "0.4.4"},
    ],
)
def test_a_verdict_that_no_longer_describes_this_install_is_re_probed(mlx, mutation) -> None:
    fake, _steps, _written, healthy = mlx
    stack._PASS_EVIDENCE = {"mlx_health": {**healthy, **mutation}}
    stack._report_mlx_stack_health(skipped = True)
    assert fake.calls == 1


def test_no_evidence_at_all_is_probed(mlx, monkeypatch) -> None:
    fake, _steps, _written, _ = mlx
    for evidence in (None, {}, {"mlx_health": "yes"}, {"mlx_health": {}}):
        fake.calls = 0
        monkeypatch.setattr(stack, "_PASS_EVIDENCE", evidence)
        stack._report_mlx_stack_health(skipped = True)
        assert fake.calls == 1, evidence


def test_the_fingerprint_names_everything_a_verdict_depends_on(monkeypatch) -> None:
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: "0.4.5")
    fingerprint = stack._mlx_health_fingerprint()
    assert fingerprint["pins"] == list(stack._MLX_PINS) + [stack._MLX_VLM_SPEC]
    assert fingerprint["python"] == stack._installer_python_tag()
    # mlx-vlm floats inside a range, so the pin string alone does not identify what is
    # installed -- and it is the package whose half-install the probe exists to catch.
    assert fingerprint["mlx_vlm"] == "0.4.5"
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _n: None)
    assert stack._mlx_health_fingerprint()["mlx_vlm"] == ""


def test_a_probe_verdict_is_recorded_for_next_time(mlx) -> None:
    fake, steps, written, _ = mlx
    stack._report_mlx_stack_health(skipped = False)
    assert written and written[0]["ok"] is True
    fake._stdout = '["mlx-lm is not importable"]'
    written.clear()
    stack._report_mlx_stack_health(skipped = False)
    assert written and written[0]["ok"] is False
    assert ("", "mlx-lm is not importable") in steps


def test_a_probe_that_cannot_answer_records_nothing(mlx) -> None:
    fake, _steps, written, _ = mlx
    fake._stdout = "null"
    stack._report_mlx_stack_health(skipped = False)
    assert written == []


def test_the_verdict_is_recorded_after_the_manifest_is_written() -> None:
    """The probe has a 180 s timeout. Writing through write_manifest would make a kill
    during it lose a finished install; update_manifest never creates one."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    # rindex: the definition comes first in the file, the call site is what is ordered.
    assert source.index("install_manifest.write_manifest(") < source.rindex(
        "_report_mlx_stack_health("
    )
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_report_mlx_stack_health":
            called = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            }
            assert "write_manifest" not in called
            assert "update_manifest" in called
            break
    else:  # pragma: no cover - the function is the subject of this file
        raise AssertionError("_report_mlx_stack_health is gone")


def test_the_escape_hatch_reaches_the_pip_bootstrap_skip() -> None:
    """UNSLOTH_STUDIO_FULL_DEPS is what a user is told to set when the install is
    behaving oddly. A step it cannot turn off is a step they cannot work around, and the
    pip bootstrap skip is not gated on the manifest, so nothing else would reach it."""
    source = STACK_PATH.read_text(encoding = "utf-8")
    assert "if not _full_deps_requested() and _venv_pip_is_usable():" in source
