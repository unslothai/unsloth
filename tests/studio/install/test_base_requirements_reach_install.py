# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Core-package skipping must not skip shared base requirements."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import install_manifest
import install_python_stack as ips

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACK = _REPO_ROOT / "studio" / "install_python_stack.py"
_REQ_ROOT = _REPO_ROOT / "studio" / "backend" / "requirements"
_EXTRA_PIN = "studio-extra @ https://example.invalid/studio-extra.zip " '; python_version >= "3.10"'


def _install_function() -> ast.FunctionDef:
    tree = ast.parse(_STACK.read_text(encoding = "utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_python_stack"
    )


def _core_branch() -> ast.If:
    return next(
        node
        for node in _install_function().body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "skip_base"
    )


def _shared_base_branch() -> ast.If:
    return next(
        node
        for node in _install_function().body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "base_requirements"
    )


class TestSharedBaseSelection:
    def _select(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        body: str,
        *,
        no_torch: bool = False,
    ) -> Path | None:
        req = tmp_path / "base.txt"
        req.write_text(textwrap.dedent(body).lstrip(), encoding = "utf-8")
        monkeypatch.setattr(ips, "REQ_ROOT", tmp_path)
        monkeypatch.setattr(ips, "NO_TORCH", no_torch)
        return ips._shared_base_requirements()

    def test_comments_only_add_no_subprocess(self, tmp_path, monkeypatch):
        assert self._select(tmp_path, monkeypatch, "# reserved for shared requirements\n\n") is None

    @pytest.mark.parametrize(
        "entry",
        [
            _EXTRA_PIN,
            "-r child.txt",
            "--requirement=https://example.invalid/base.txt",
            "-c constraints.txt",
        ],
    )
    def test_any_pip_entry_uses_the_original_file_unchanged(self, tmp_path, monkeypatch, entry):
        selected = self._select(tmp_path, monkeypatch, f"# shared\n{entry}\n")
        assert selected == tmp_path / "base.txt"
        assert selected.read_text(encoding = "utf-8") == f"# shared\n{entry}\n"
        assert not list(tmp_path.glob(".*-filtered-*.txt"))

    def test_no_torch_keeps_its_own_runtime_list(self, tmp_path, monkeypatch):
        selected = self._select(tmp_path, monkeypatch, _EXTRA_PIN, no_torch = True)
        assert selected is None

    def test_current_base_file_adds_no_install_step(self):
        assert ips._shared_base_requirements() is None

    def test_a_bom_does_not_read_as_content(self, tmp_path, monkeypatch):
        """PowerShell 5.1 redirection and some Windows editors prepend a UTF-8 BOM."""
        req = tmp_path / "base.txt"
        req.write_text("# shared\n", encoding = "utf-8-sig")
        monkeypatch.setattr(ips, "REQ_ROOT", tmp_path)
        monkeypatch.setattr(ips, "NO_TORCH", False)
        assert ips._shared_base_requirements() is None

    def test_crlf_entry_is_still_seen(self, tmp_path, monkeypatch):
        (tmp_path / "base.txt").write_bytes(b"# shared\r\n" + _EXTRA_PIN.encode() + b"\r\n")
        monkeypatch.setattr(ips, "REQ_ROOT", tmp_path)
        monkeypatch.setattr(ips, "NO_TORCH", False)
        assert ips._shared_base_requirements() == tmp_path / "base.txt"

    @pytest.mark.parametrize("mode", ["missing", "unreadable"])
    def test_an_unusable_file_does_not_take_down_the_install(self, tmp_path, monkeypatch, mode):
        """This runs before the manifest is dropped, so raising here aborts with a traceback."""
        if mode == "unreadable":
            req = tmp_path / "base.txt"
            req.write_text(_EXTRA_PIN, encoding = "utf-8")
            req.chmod(0o000)
        monkeypatch.setattr(ips, "REQ_ROOT", tmp_path)
        monkeypatch.setattr(ips, "NO_TORCH", False)
        try:
            assert ips._shared_base_requirements() is None
        finally:
            if mode == "unreadable":
                req.chmod(0o644)


class TestSharedBasePhase:
    def _run(self, req: Path | None, *, skip_base: bool) -> tuple[list[Path], list[str]]:
        installs: list[Path] = []
        progress: list[str] = []

        def record_install(
            _label,
            *_args,
            req = None,
            **_kwargs,
        ):
            installs.append(req)

        module = ast.Module(body = [_shared_base_branch()], type_ignores = [])
        namespace = {
            "base_requirements": req,
            "skip_base": skip_base,
            "_progress": progress.append,
            "_step": lambda *_args, **_kwargs: None,
            "_LABEL": "python",
            "pip_install": record_install,
        }
        exec(compile(module, "<shared base phase>", "exec"), namespace)
        return installs, progress

    @pytest.mark.parametrize("skip_base", [False, True])
    def test_shared_file_is_applied_once_on_both_core_paths(self, tmp_path, skip_base):
        req = tmp_path / "base.txt"
        req.write_text(_EXTRA_PIN + "\n", encoding = "utf-8")
        installs, _progress = self._run(req, skip_base = skip_base)
        assert installs == [req]

    def test_shell_handoff_owns_the_progress_slot_when_shared_work_exists(self, tmp_path):
        req = tmp_path / "base.txt"
        req.write_text(_EXTRA_PIN + "\n", encoding = "utf-8")
        _installs, progress = self._run(req, skip_base = True)
        assert progress == ["base requirements"]

    def test_no_selected_file_installs_nothing(self):
        assert self._run(None, skip_base = True) == ([], [])

    def test_shared_phase_is_after_and_outside_the_core_branch(self):
        core = _core_branch()
        shared = _shared_base_branch()
        between = _STACK.read_text(encoding = "utf-8").splitlines()[core.end_lineno : shared.lineno]
        assert "        base_requirements = _shared_base_requirements()" in between
        assert all(isinstance(node, ast.Pass) for node in core.body)


class TestCorePackageOwnership:
    def test_update_calls_name_both_core_distributions_directly(self):
        calls = [
            node
            for node in ast.walk(_install_function())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pip_install"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "Updating core packages"
        ]
        assert len(calls) == 2
        for call in calls:
            values = [arg.value for arg in call.args if isinstance(arg, ast.Constant)]
            assert "unsloth" in values and "unsloth-zoo" in values
            assert not any(keyword.arg == "req" for keyword in call.keywords)

    def test_base_file_does_not_own_core_distributions(self):
        names = []
        for line in (_REQ_ROOT / "base.txt").read_text(encoding = "utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if text and not text.startswith("-"):
                names.append(canonicalize_name(Requirement(text).name))
        assert {"unsloth", "unsloth-zoo"}.isdisjoint(names)


class TestInstallerHandoff:
    @pytest.mark.parametrize(
        "path, needle",
        [
            ("install.sh", "_SKIP_BASE=1"),
            ("install.ps1", '$env:SKIP_STUDIO_BASE = "1"'),
        ],
    )
    def test_both_installers_delegate_the_core_skip(self, path, needle):
        assert needle in (_REPO_ROOT / path).read_text(encoding = "utf-8")

    @pytest.mark.parametrize("path", ["install.sh", "install.ps1"])
    def test_python_stack_owns_shared_base_requirements(self, path):
        assert "base.txt" not in (_REPO_ROOT / path).read_text(encoding = "utf-8")

    def test_base_changes_invalidate_the_install_manifest(self):
        assert "base.txt" in install_manifest.TRACKED_REQUIREMENT_FILES
