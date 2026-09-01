# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``check_transformers_dependency_versions`` in ``unsloth/import_fixes.py``.

A ``--no-deps`` install of transformers from git main leaves pip enforcing nothing,
so the break lands at import with the wrong remedy (``pip install transformers -U``)
for someone deliberately on main. These tests drive the real functions and pin: the
requirement set comes from the installed distribution's metadata, only genuine
violations are reported, the remedy names the DEPENDENCY, and nothing raises.

Runs under the GPU-free ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging

import pytest

from unsloth import import_fixes as IF


# Base (no-extras) requirements as declared by two real transformers releases.
REQUIRES_4_57_6 = [
    "filelock",
    "huggingface-hub<1.0,>=0.34.0",
    "numpy>=1.17",
    "packaging>=20.0",
    "pyyaml>=5.1",
    "regex!=2019.12.17",
    "requests",
    "tokenizers<=0.23.0,>=0.22.0",
    "safetensors>=0.4.3",
    "tqdm>=4.27",
    'torch>=2.2; extra == "torch"',
    'accelerate>=0.26.0; extra == "torch"',
    'fugashi>=1.0; extra == "ja"',
]
REQUIRES_5_14_1 = [
    "huggingface-hub<2.0,>=1.5.0",
    "numpy>=1.17",
    "packaging>=20.0",
    "pyyaml>=5.1",
    "regex>=2025.10.22",
    "tokenizers<=0.23.0,>=0.22.0",
    "typer",
    "safetensors>=0.8.0",
    "tqdm>=4.60",
]


class _Missing(Exception):
    """Stands in for importlib.metadata.PackageNotFoundError."""


def _install_env(monkeypatch, requires, installed):
    """Point the check at a synthetic environment.

    ``requires`` is what transformers declares (or an exception to raise);
    ``installed`` maps distribution name -> version, anything absent raises.
    """

    def fake_requires(name):
        if name != "transformers":
            raise importlib.metadata.PackageNotFoundError(name)
        if isinstance(requires, Exception):
            raise requires
        return requires

    def fake_version(name):
        key = name.lower().replace("_", "-")
        if key in installed:
            return installed[key]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "requires", fake_requires)
    monkeypatch.setattr(IF, "importlib_version", fake_version)
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_DEPENDENCY_CHECK", raising = False)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _run_check(caplog):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger = IF.logger.name):
        IF.check_transformers_dependency_versions()
    return _warnings(caplog)


@pytest.mark.parametrize(
    "requires, installed",
    [
        (
            REQUIRES_4_57_6,
            {
                "filelock": "3.16.1",
                "huggingface-hub": "0.36.2",
                "numpy": "2.1.3",
                "packaging": "24.2",
                "pyyaml": "6.0.2",
                "regex": "2025.11.3",
                "requests": "2.32.4",
                "tokenizers": "0.22.2",
                "safetensors": "0.7.0",
                "tqdm": "4.67.3",
            },
        ),
        (
            REQUIRES_5_14_1,
            {
                "huggingface-hub": "1.27.0",
                "numpy": "2.1.3",
                "packaging": "24.2",
                "pyyaml": "6.0.2",
                "regex": "2025.11.3",
                "tokenizers": "0.22.2",
                "typer": "0.15.1",
                "safetensors": "0.8.0",
                "tqdm": "4.67.3",
            },
        ),
    ],
    ids = ["transformers-4.57.6", "transformers-5.14.1"],
)
def test_satisfied_requirements_are_silent(monkeypatch, caplog, requires, installed):
    """A fully satisfied requirement set says nothing, on 4.57.x and on 5.x."""
    _install_env(monkeypatch, requires, installed)
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_violated_floor_is_reported_and_names_the_dependency(monkeypatch, caplog):
    """The Kaggle LFM2 break: transformers main wants safetensors>=0.8.0, the
    image ships 0.7.0. The remedy must upgrade safetensors, not transformers."""
    installed = {
        "transformers": "5.15.0.dev0",
        "huggingface-hub": "1.27.0",
        "numpy": "2.1.3",
        "packaging": "24.2",
        "pyyaml": "6.0.2",
        "regex": "2025.11.3",
        "tokenizers": "0.22.2",
        "typer": "0.15.1",
        "safetensors": "0.7.0",
        "tqdm": "4.67.3",
    }
    _install_env(monkeypatch, REQUIRES_5_14_1, installed)

    assert IF._unsatisfied_transformers_requirements() == [("safetensors", ">=0.8.0", "0.7.0")]

    warnings = _run_check(caplog)
    assert len(warnings) == 1
    message = warnings[0]

    # Names the dependency, its floor and what is actually installed.
    assert "safetensors>=0.8.0 is required, but found safetensors==0.7.0" in message
    # The remedy upgrades the DEPENDENCY.
    assert 'pip install --upgrade "safetensors>=0.8.0"' in message
    assert "Upgrade the dependencies, not transformers" in message
    # And explicitly contradicts transformers' own misleading advice.
    assert "pip install transformers -U" in message
    assert "Ignore that" in message
    # Nothing satisfied gets dragged in.
    for satisfied in ("tqdm", "typer", "numpy", "huggingface-hub"):
        assert f"{satisfied}==" not in message


def test_string_comparison_trap_is_not_a_false_positive(monkeypatch, caplog):
    """``0.10.0`` sorts before ``0.8.0`` as a string but satisfies >=0.8.0."""
    _install_env(
        monkeypatch,
        ["safetensors>=0.8.0"],
        {"transformers": "5.15.0.dev0", "safetensors": "0.10.0"},
    )
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_multiple_violations_are_all_listed_in_one_command(monkeypatch, caplog):
    _install_env(
        monkeypatch,
        ["huggingface-hub<2.0,>=1.5.0", "safetensors>=0.8.0", "tqdm>=4.60"],
        {
            "transformers": "5.15.0.dev0",
            "huggingface-hub": "0.36.2",
            "safetensors": "0.7.0",
            "tqdm": "4.67.3",
        },
    )
    reported = {name for name, _, _ in IF._unsatisfied_transformers_requirements()}
    assert reported == {"huggingface-hub", "safetensors"}

    message = _run_check(caplog)[0]
    assert 'pip install --upgrade "huggingface-hub<2.0,>=1.5.0" "safetensors>=0.8.0"' in message


def test_prerelease_dependency_satisfying_the_floor_is_not_reported(monkeypatch, caplog):
    _install_env(
        monkeypatch,
        ["safetensors>=0.8.0"],
        {"transformers": "5.15.0.dev0", "safetensors": "0.9.0rc1"},
    )
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_inapplicable_environment_markers_are_skipped(monkeypatch, caplog):
    """Extras and python_version gates that do not apply must not be checked,
    even when a violating version of the named package is installed."""
    _install_env(
        monkeypatch,
        [
            'torch>=99.0; extra == "torch"',
            'accelerate>=99.0; extra == "accelerate"',
            'fugashi>=99.0; extra == "ja"',
            'numpy>=99.0; python_version < "3.0"',
            'requests>=99.0; sys_platform == "definitely-not-a-real-platform"',
        ],
        {
            "transformers": "5.15.0.dev0",
            "torch": "2.9.0",
            "accelerate": "1.2.0",
            "fugashi": "1.3.0",
            "numpy": "2.1.3",
            "requests": "2.32.4",
        },
    )
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_applicable_environment_marker_is_still_checked(monkeypatch, caplog):
    """The marker filter must not swallow requirements whose marker DOES apply."""
    _install_env(
        monkeypatch,
        ['safetensors>=0.8.0; python_version >= "3.0"'],
        {"transformers": "5.15.0.dev0", "safetensors": "0.7.0"},
    )
    assert IF._unsatisfied_transformers_requirements() == [("safetensors", ">=0.8.0", "0.7.0")]
    assert "safetensors" in _run_check(caplog)[0]


def test_an_absent_base_requirement_is_reported_like_a_stale_one(monkeypatch, caplog):
    """`--no-deps` leaves a dependency missing as often as it leaves it old.

    transformers checks its base requirements at its own root import and raises
    PackageNotFoundError carrying the same misleading `pip install transformers -U`
    hint, so skipping the absent case left the user with only that message.
    """
    _install_env(
        monkeypatch,
        ["safetensors>=0.8.0", "typer", "tqdm>=4.27", 'fugashi>=1.0; extra == "ja"'],
        {"transformers": "5.15.0.dev0", "tqdm": "4.67.3"},
    )
    assert IF._unsatisfied_transformers_requirements() == [
        ("safetensors", ">=0.8.0", None),
        ("typer", "", None),
    ]
    warning = "\n".join(_run_check(caplog))
    assert "safetensors>=0.8.0 is required, but it is not installed" in warning
    assert "typer is required, but it is not installed" in warning
    assert 'pip install --upgrade "safetensors>=0.8.0" "typer"' in warning
    assert "Install or upgrade the dependencies, not transformers" in warning
    # An extras-only requirement stays out of it:
    assert "fugashi" not in warning
    # Satisfied requirements stay out of it too.
    assert "tqdm" not in warning


def test_an_absent_extras_only_package_is_not_reported(monkeypatch, caplog):
    """Optional extras are opt-in; a package the user is right not to have is silent."""
    _install_env(
        monkeypatch,
        ['torch>=2.2; extra == "torch"', 'jax>=0.4.1; extra == "flax"'],
        {"transformers": "5.15.0.dev0"},
    )
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_unreadable_metadata_is_still_silent(monkeypatch, caplog):
    """Only PackageNotFoundError counts as absent; any other metadata error means we
    cannot tell, and guessing would warn about a working install."""
    _install_env(monkeypatch, ["safetensors>=0.8.0"], {"transformers": "5.15.0.dev0"})

    def broken_version(name):
        raise OSError("dist-info unreadable")

    monkeypatch.setattr(IF, "importlib_version", broken_version)
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


@pytest.mark.parametrize(
    "requires, installed",
    [
        # Unparseable requirement line.
        (["safetensors>=@@@not-a-specifier"], {"safetensors": "0.7.0"}),
        (["=== nonsense ==="], {"safetensors": "0.7.0"}),
        (["safetensors>=0.8.0; extra ==="], {"safetensors": "0.7.0"}),
        # Parseable specifier, but the installed version is not PEP 440.
        (["safetensors>=0.8.0"], {"safetensors": "not-a-version"}),
        # Undecidable marker.
        (['safetensors>=0.8.0; nonexistent_marker == "x"'], {"safetensors": "0.7.0"}),
    ],
    ids = ["bad-specifier", "garbage-line", "bad-marker", "bad-installed-version", "unknown-marker"],
)
def test_unparseable_input_is_silent_and_never_raises(monkeypatch, caplog, requires, installed):
    installed = {"transformers": "5.15.0.dev0", **installed}
    _install_env(monkeypatch, requires, installed)
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


@pytest.mark.parametrize(
    "requires",
    [
        importlib.metadata.PackageNotFoundError("transformers"),
        None,  # dist-info present but declares nothing
        [],
    ],
    ids = ["metadata-missing", "requires-none", "requires-empty"],
)
def test_missing_transformers_metadata_is_silent_and_never_raises(monkeypatch, caplog, requires):
    _install_env(monkeypatch, requires, {})
    assert IF._unsatisfied_transformers_requirements() == []
    assert _run_check(caplog) == []


def test_transformers_not_installed_is_silent(monkeypatch, caplog):
    real_find_spec = IF.importlib.util.find_spec
    monkeypatch.setattr(
        IF.importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name == "transformers" else real_find_spec(name, *a, **kw),
    )
    _install_env(monkeypatch, REQUIRES_5_14_1, {})
    assert _run_check(caplog) == []


def test_env_var_silences_the_check(monkeypatch, caplog):
    _install_env(
        monkeypatch,
        ["safetensors>=0.8.0"],
        {"transformers": "5.15.0.dev0", "safetensors": "0.7.0"},
    )
    monkeypatch.setenv("UNSLOTH_SKIP_TRANSFORMERS_DEPENDENCY_CHECK", "1")
    assert _run_check(caplog) == []


def test_runs_against_the_real_environment_without_raising(caplog):
    """No monkeypatching: the real installed transformers, read from real metadata."""
    result = IF._unsatisfied_transformers_requirements()
    assert isinstance(result, list)
    for entry in result:
        assert len(entry) == 3
    IF.check_transformers_dependency_versions()


def test_check_is_registered_in_gpu_init():
    """The check is worthless if nothing calls it at import time."""
    from pathlib import Path

    source = Path(IF.__file__).with_name("_gpu_init.py").read_text(encoding = "utf-8")
    assert "check_transformers_dependency_versions," in source, "not imported"
    assert "check_transformers_dependency_versions()" in source, "imported but never called"


def test_check_warns_rather_than_raises_on_a_violation(monkeypatch, caplog):
    """Deliberate contract: transformers' own (misleading) message still reaches the
    user, ours lands just before it. A metadata floor must not become a hard stop."""
    _install_env(
        monkeypatch,
        ["safetensors>=0.8.0"],
        {"transformers": "5.15.0.dev0", "safetensors": "0.7.0"},
    )
    IF.check_transformers_dependency_versions()  # must not raise
    assert len(_run_check(caplog)) == 1


def test_a_transformers_stub_in_sys_modules_does_not_break_the_import(monkeypatch, caplog):
    """`find_spec` RAISES on a module in sys.modules whose `__spec__` is None or unset,
    rather than returning None (documented behaviour, CPython Lib/importlib/util.py).
    An unguarded probe there turns a warn-only check into a failed `import unsloth`.
    """
    import sys
    import types

    _install_env(monkeypatch, ["safetensors>=0.8.0"], {"transformers": "5.15.0.dev0"})
    for stub in (types.ModuleType("transformers"), types.SimpleNamespace()):
        monkeypatch.setitem(sys.modules, "transformers", stub)
        with pytest.raises(ValueError):
            importlib.util.find_spec("transformers")  # __spec__ None / not set must not raise
        IF.check_transformers_dependency_versions()  # must not raise
        assert _warnings(caplog) == []


def test_check_also_runs_on_the_mlx_branch():
    """Apple Silicon never reaches `_gpu_init`.

    `unsloth/__init__.py` splits on `_IS_MLX` and only the `else` arm imports
    `_gpu_init`, while the MLX arm imports transformers itself, so registering on one
    arm leaves MLX users with transformers' own wrong remedy. The call belongs in the
    `if _IS_MLX:` body, next to the torchao fixes that are there for the same reason.
    """
    import ast
    from pathlib import Path

    source = Path(IF.__file__).with_name("__init__.py").read_text(encoding = "utf-8")
    branch = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_IS_MLX"
    )
    body = ast.unparse(ast.Module(body = branch.body, type_ignores = []))
    assert "check_transformers_dependency_versions" in body, "not called on the MLX path"
    # And the GPU arm still reaches it through _gpu_init.
    assert "_gpu_init" in ast.unparse(ast.Module(body = branch.orelse, type_ignores = []))
