# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""base.txt must reach the venv on the install.sh / install.ps1 paths too.

install.sh and install.ps1 install unsloth + unsloth-zoo inline and then export
SKIP_STUDIO_BASE=1 so setup.sh / setup.ps1 do not install them a second time.
install_python_stack.py used to read that flag as "skip base.txt", which was the
same thing only for as long as base.txt held nothing but those two names. Add a
third, pinned entry to base.txt and it reached no fresh install on any platform:
the installers never read the file, and the one branch that does was skipped.

These tests pin the distinction: the core packages stay skipped, everything else
in base.txt is applied.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

import install_python_stack as ips

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PINNED = (
    'diffusers @ https://github.com/huggingface/diffusers/archive/'
    'f53d552036a0d1bd5570782a39cd40cfabf112bc.zip ; python_version >= "3.10"'
)


class TestRequirementProjectName:
    """The name parser the core-package filter keys off."""

    @pytest.mark.parametrize(
        "line, expected",
        [
            ("unsloth\n", "unsloth"),
            ("unsloth-zoo\n", "unsloth-zoo"),
            ("unsloth_zoo\n", "unsloth-zoo"),          # PEP 503 normalisation
            ("  unsloth  \n", "unsloth"),
            ("unsloth>=2026.8.9\n", "unsloth"),
            ("unsloth[all]==1.0\n", "unsloth"),
            ("diffusers ; python_version < '3.10'\n", "diffusers"),
            # PEP 508 allows the marker and the URL with no surrounding space
            ('diffusers;python_version<"3.10"\n', "diffusers"),
            ("diffusers@https://example.invalid/d.zip\n", "diffusers"),
            (_PINNED + "\n", "diffusers"),             # direct URL + marker
            ("# a comment\n", ""),
            ("\n", ""),
            ("--no-deps\n", ""),
            ("torch  # trailing comment\n", "torch"),
        ],
    )
    def test_parses(self, line, expected):
        assert ips._requirement_project_name(line) == expected


class TestRequirementsBeyond:
    def _write(self, tmp_path: Path, body: str) -> Path:
        req = tmp_path / "base.txt"
        req.write_text(textwrap.dedent(body).lstrip(), encoding = "utf-8")
        return req

    def test_core_only_file_yields_nothing_to_install(self, tmp_path):
        """Today's base.txt: the filter must return None, not an empty file.

        Handing pip a requirements file with no requirements in it is a pointless
        subprocess on every single install.
        """
        req = self._write(tmp_path, """
            # Core unsloth packages
            unsloth-zoo
            unsloth
        """)
        assert ips._requirements_beyond(req, ips._CORE_BASE_PACKAGES) is None

    def test_pinned_entry_survives_and_core_packages_do_not(self, tmp_path):
        req = self._write(tmp_path, f"""
            unsloth-zoo
            unsloth
            {_PINNED}
        """)
        out = ips._requirements_beyond(req, ips._CORE_BASE_PACKAGES)
        assert out is not None
        try:
            text = out.read_text(encoding = "utf-8")
            assert _PINNED in text
            names = [
                ips._requirement_project_name(line) for line in text.splitlines()
            ]
            assert "unsloth" not in names and "unsloth-zoo" not in names
        finally:
            out.unlink(missing_ok = True)

    def test_a_prefixed_package_is_not_swallowed(self, tmp_path):
        """`unsloth-something` is not `unsloth`.

        _filter_requirements matches on startswith(), which would drop this line.
        """
        req = self._write(tmp_path, """
            unsloth
            unsloth-zoo
            unsloth-studio-extras==1.2.3
        """)
        out = ips._requirements_beyond(req, ips._CORE_BASE_PACKAGES)
        assert out is not None
        try:
            assert "unsloth-studio-extras==1.2.3" in out.read_text(encoding = "utf-8")
        finally:
            out.unlink(missing_ok = True)


def _skip_base_branch_body() -> list[ast.stmt]:
    """The shipped `if skip_base:` body from install_python_stack()."""
    tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "install_python_stack"
    )
    branch = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "skip_base"
        and any(isinstance(h, ast.If) for h in n.orelse)  # the base-packages chain
    )
    return branch.body


def _run_skip_base_branch(req_root: Path, *, no_torch: bool) -> list[dict]:
    """Execute that branch verbatim, recording what it hands to pip."""
    calls: list[dict] = []

    def _record_pip_install(label, *args, req = None, constrain = True):
        calls.append({
            "label": label,
            "args": args,
            "req_text": Path(req).read_text(encoding = "utf-8") if req else None,
            "constrain": constrain,
        })

    module = ast.Module(body = _skip_base_branch_body(), type_ignores = [])
    namespace = {
        "NO_TORCH": no_torch,
        "REQ_ROOT": req_root,
        "_CORE_BASE_PACKAGES": ips._CORE_BASE_PACKAGES,
        "_requirements_beyond": ips._requirements_beyond,
        "_step": lambda *a, **k: None,
        "_LABEL": "python",
        "pip_install": _record_pip_install,
    }
    exec(compile(module, "<skip_base branch>", "exec"), namespace)
    return calls


class TestSkipStudioBaseStillAppliesBaseTxt:
    """The regression itself."""

    def _req_root(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "base.txt").write_text(textwrap.dedent(body).lstrip(), encoding = "utf-8")
        return tmp_path

    def test_a_pinned_entry_is_installed_under_skip_studio_base(self, tmp_path):
        """The MiniMax-H3 case: a pinned revision must survive a fresh install."""
        root = self._req_root(tmp_path, f"""
            unsloth-zoo
            unsloth
            {_PINNED}
        """)
        calls = _run_skip_base_branch(root, no_torch = False)
        assert len(calls) == 1, "the pinned base requirements were never installed"
        assert _PINNED in calls[0]["req_text"]

    def test_the_core_packages_are_still_not_reinstalled(self, tmp_path):
        """SKIP_STUDIO_BASE=1 exists to avoid that; keep it working."""
        root = self._req_root(tmp_path, f"""
            unsloth-zoo
            unsloth
            {_PINNED}
        """)
        calls = _run_skip_base_branch(root, no_torch = False)
        names = [
            ips._requirement_project_name(line)
            for line in calls[0]["req_text"].splitlines()
        ]
        assert "unsloth" not in names and "unsloth-zoo" not in names

    def test_todays_core_only_base_txt_installs_nothing(self, tmp_path):
        """No new subprocess on installs that have nothing extra to apply."""
        root = self._req_root(tmp_path, """
            # Core unsloth packages
            unsloth-zoo
            unsloth
        """)
        assert _run_skip_base_branch(root, no_torch = False) == []

    def test_no_torch_mode_is_left_to_its_own_requirements_file(self, tmp_path):
        """no-torch installs apply no-torch-runtime.txt inline; base.txt is torch-bound."""
        root = self._req_root(tmp_path, f"""
            unsloth-zoo
            unsloth
            {_PINNED}
        """)
        assert _run_skip_base_branch(root, no_torch = True) == []

    def test_the_real_base_txt_round_trips(self):
        """Whatever base.txt currently holds, the branch must not raise on it."""
        _run_skip_base_branch(
            _REPO_ROOT / "studio" / "backend" / "requirements", no_torch = False,
        )


class TestInstallersStillHandOverTheFlag:
    """The precondition the branch above is written against."""

    @pytest.mark.parametrize(
        "path, needle",
        [
            ("install.sh", "_SKIP_BASE=1"),
            ("install.ps1", '$env:SKIP_STUDIO_BASE = "1"'),
        ],
    )
    def test_both_installers_set_skip_studio_base(self, path, needle):
        assert needle in (_REPO_ROOT / path).read_text(encoding = "utf-8")

    @pytest.mark.parametrize("path", ["install.sh", "install.ps1"])
    def test_neither_installer_applies_base_txt_itself(self, path):
        """If one ever does, this branch would install it twice."""
        assert "base.txt" not in (_REPO_ROOT / path).read_text(encoding = "utf-8")

    def test_editing_base_txt_forces_a_dependency_pass(self):
        """Covers `unsloth studio update`'s fast path, which skips the stack entirely."""
        import install_manifest

        assert "base.txt" in install_manifest.TRACKED_REQUIREMENT_FILES
