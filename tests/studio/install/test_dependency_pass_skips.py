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
