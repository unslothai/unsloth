# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a Windows-on-ARM install has to recover when it is not install.ps1.

install.ps1 hands its resolver decisions to setup.ps1 through process-scoped
environment variables. A direct `unsloth studio update` runs in a fresh shell,
where all of them are gone, so each has to be recoverable from disk: the CUDA
torch index (below) and the generated requirement overrides
(TestResolverEnvironmentRestore). Neither can be re-derived from the host.

Windows on ARM is the only platform whose CUDA torch wheels live nowhere on
download.pytorch.org, so a later `unsloth studio update` -- a fresh shell, with
install.ps1's handover variable long gone -- cannot re-derive the index from the
driver the way every other host can. write_manifest records it.

That is a deliberate exception to the rule the module documents for itself: the
FLAVOR, never the index URL it came from, because a pinned index can carry a
token in its userinfo, query or fragment, and this file is read back by
verify-install and desktop-capabilities. The exception only holds while the
guard does, so both halves are tested here -- the Python that writes it and the
PowerShell that reads it back, each of which must refuse independently. A
hand-edited manifest must not be able to redirect a torch install to an
arbitrary host, and a mirror the user pinned must not be copied into a file
other tooling prints.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST_PY = PACKAGE_ROOT / "studio" / "install_manifest.py"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"
STACK_LLAMA = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_woa", MANIFEST_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_manifest_module()


# (url, may_be_persisted, why)
CANDIDATES = (
    ("https://pypi.nvidia.com/nvtorch_oot", True, "the GA channel install.ps1 probes"),
    ("https://pypi.nvidia.com/nvtorch_oot_nightly/", True, "the nightly channel, trailing slash"),
    ("https://user:token@pypi.nvidia.com/nvtorch_oot", False, "userinfo is exactly the leak"),
    ("https://pypi.nvidia.com/nvtorch_oot?token=abc", False, "a token in the query"),
    ("https://pypi.nvidia.com/nvtorch_oot#token=abc", False, "a token in the fragment"),
    ("https://mirror.corp.example/whl", False, "a mirror the user pinned"),
    ("http://pypi.nvidia.com/nvtorch_oot", False, "plaintext, so not our channel"),
    ("https://pypi.nvidia.com.evil.example/whl", False, "a host that merely starts the same"),
    ("https://evilpypi.nvidia.com/whl", False, "a host that merely ends the same"),
    ("", False, "empty"),
    (None, False, "absent"),
)


class TestWriteSide:
    """studio/install_manifest.py: what is allowed into the file at all."""

    @pytest.mark.parametrize("url, allowed, why", CANDIDATES)
    def test_only_a_credential_free_nvidia_channel_is_recorded(
        self, tmp_path: pathlib.Path, url, allowed: bool, why: str
    ):
        path = im.write_manifest(root = tmp_path, req_root = tmp_path, woa_torch_index = url)
        assert path is not None
        payload = json.loads(pathlib.Path(path).read_text(encoding = "utf-8"))
        recorded = "woa_torch_index" in payload
        assert recorded is allowed, (
            f"{url!r} ({why}) was {'dropped' if allowed else 'persisted'}; "
            "the manifest is printed back by verify-install and read by setup.ps1"
        )
        if allowed:
            assert payload["woa_torch_index"] == str(url).strip().rstrip("/")

    def test_the_key_is_absent_when_no_index_was_chosen(self, tmp_path: pathlib.Path):
        """Every other host, and every WoA host that stayed on the x64 stack."""
        im.write_manifest(root = tmp_path, req_root = tmp_path)
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload

    def test_the_addition_is_backwards_compatible(self, tmp_path: pathlib.Path):
        """An additive optional key: older readers see the schema they already parse."""
        im.write_manifest(
            root = tmp_path,
            req_root = tmp_path,
            woa_torch_index = "https://pypi.nvidia.com/nvtorch_oot",
        )
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert payload["schema"] == 1, "the key is additive; bumping the schema is not"
        state = im.verify_install(root = tmp_path, req_root = tmp_path)
        assert state["manifest_ok"] is True, state["reason"]

    def test_the_installer_passes_the_handover_variable_through(self):
        """install.ps1 exports it; nothing else supplies this value."""
        source = STACK_PY.read_text(encoding = "utf-8")
        assert re.search(
            r"woa_torch_index\s*=\s*os\.environ\.get\(\s*[\"']UNSLOTH_WOA_SELECTED_TORCH_INDEX[\"']",
            source,
        ), "install_python_stack.py no longer forwards the index install.ps1 selected"


PWSH = shutil.which("pwsh")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "pwsh not available")


def _function_source(text: str, name: str) -> str:
    """Extract a PowerShell function by matching balanced braces."""
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    assert match, f"{name} is not defined in setup.ps1"
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


class TestReadSide:
    """studio/setup.ps1: what Get-PersistedWoaTorchIndex hands back to the resolver."""

    @requires_pwsh
    @pytest.mark.parametrize("url, allowed, why", [c for c in CANDIDATES if c[0]])
    def test_a_hand_edited_manifest_cannot_redirect_the_install(
        self, tmp_path: pathlib.Path, url: str, allowed: bool, why: str
    ):
        """
        The write guard is not enough on its own: the file sits in the user's venv and
        anything can put a line in it. Whatever is on disk, only NVIDIA's own channel
        may come back out, because the return value becomes --extra-index-url.
        """
        (tmp_path / "unsloth_install_manifest.json").write_text(
            json.dumps({"schema": 1, "woa_torch_index": url}),
            encoding = "utf-8",
        )
        got = self._invoke(tmp_path)
        assert got == (
            url.strip().rstrip("/") if allowed else ""
        ), f"{url!r} ({why}) came back as {got!r} and would be passed to uv"

    @requires_pwsh
    def test_a_missing_or_unreadable_manifest_is_empty_not_an_error(self, tmp_path: pathlib.Path):
        """Older installs have no such key, and a truncated file must not throw."""
        assert self._invoke(tmp_path) == "", "no manifest at all"
        path = tmp_path / "unsloth_install_manifest.json"
        path.write_text('{"schema": 1, "torch_flavor": "cu130"}', encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "an older manifest without the key"
        path.write_text('{"schema": 1, "woa_torch_ind', encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "a manifest truncated by a killed installer"
        path.write_text("", encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "an empty manifest"

    @staticmethod
    def _invoke(venv: pathlib.Path) -> str:
        body = _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Get-PersistedWoaTorchIndex")
        script = f"{body}\nWrite-Output (Get-PersistedWoaTorchIndex -VenvPath '{venv}')"
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()


class TestResolverEnvironmentRestore:
    """The other half of what a fresh shell loses.

    install.ps1 writes StudioHome\\woa\\overrides.txt and stages a win_arm64 wheelhouse
    beside it, then exports both through UV_OVERRIDE / UV_FIND_LINKS / PIP_FIND_LINKS.
    Those exports are process-scoped, so a direct `unsloth studio update` starts without
    them -- and the dependency pass resolves `ddgs`, which requires httpx[brotli], which
    requires Brotli on CPython, which publishes no win_arm64 wheel. Without the overrides
    the resolver reaches for the sdist and tries to build a C extension on a host that
    exists to avoid exactly that.
    """

    def _invoke(
        self,
        tmp_path: pathlib.Path,
        *,
        is_woa: bool = True,
        preset: str = "",
    ) -> dict:
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "$script:Warnings = @()",
                "function substep { param($m, $c) $script:Warnings += ,$m }",
                f"$StudioHome = '{tmp_path}'",
                f"function Test-WinArm64Venv {{ ${str(is_woa).lower()} }}",
                preset,
                _function_source(setup, "Get-UvSafePath"),
                _function_source(setup, "Restore-WoaResolverEnvironment"),
                "Restore-WoaResolverEnvironment",
                "[pscustomobject]@{",
                "  ov = $env:UV_OVERRIDE; uvfl = $env:UV_FIND_LINKS; pipfl = $env:PIP_FIND_LINKS",
                "  warned = ($script:Warnings -join ' ')",
                "} | ConvertTo-Json -Compress",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
            # The function assigns real environment variables; keep them out of the parent.
            env = {**os.environ, "UV_OVERRIDE": "", "UV_FIND_LINKS": "", "PIP_FIND_LINKS": ""},
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    @staticmethod
    def _stage(tmp_path: pathlib.Path, *, wheels: bool = True) -> pathlib.Path:
        woa = tmp_path / "woa"
        woa.mkdir(parents = True, exist_ok = True)
        overrides = woa / "overrides.txt"
        overrides.write_text(
            "# Generated by install.ps1 for Windows on ARM (win_arm64).\n"
            'brotli ; platform_machine == "AMD64"\n'
            'brotlicffi ; platform_machine == "AMD64"\n'
            "torch>=2.4\n",
            encoding = "utf-8",
        )
        if wheels:
            (woa / "wheels").mkdir(exist_ok = True)
        return overrides

    @requires_pwsh
    def test_a_native_venv_gets_its_overrides_back(self, tmp_path: pathlib.Path):
        overrides = self._stage(tmp_path)
        got = self._invoke(tmp_path)
        assert got["ov"] == str(overrides), "the drop list install.ps1 generated"
        assert got["uvfl"] == str(tmp_path / "woa" / "wheels")
        assert got["pipfl"] == str(tmp_path / "woa" / "wheels")

    @requires_pwsh
    def test_no_wheelhouse_still_restores_the_overrides(self, tmp_path: pathlib.Path):
        """The drops are what stop the brotli sdist; the wheelhouse is a separate favour."""
        overrides = self._stage(tmp_path, wheels = False)
        got = self._invoke(tmp_path)
        assert got["ov"] == str(overrides)
        assert not got["uvfl"] and not got["pipfl"]

    @requires_pwsh
    def test_every_other_host_is_untouched(self, tmp_path: pathlib.Path):
        """An x64 venv resolves brotli from a win_amd64 wheel, as it always has."""
        self._stage(tmp_path)
        got = self._invoke(tmp_path, is_woa = False)
        assert not got["ov"] and not got["uvfl"] and not got["pipfl"]
        assert not got["warned"], "and it says nothing about a platform it is not on"

    @requires_pwsh
    def test_a_caller_that_already_set_them_keeps_their_file(self, tmp_path: pathlib.Path):
        """
        The caller's own override file is never dropped. It is no longer the WHOLE answer
        though: the win_arm64 drop list is added beside it, because standing down entirely
        sent the dependency pass at a Brotli sdist. Disjoint files, so both are passed.
        """
        overrides = self._stage(tmp_path)
        got = self._invoke(
            tmp_path,
            preset = "$env:UV_OVERRIDE = 'C:\\caller\\ov.txt'",
        )
        assert "C:\\caller\\ov.txt" in got["ov"], "the caller's file survives"
        assert str(overrides) in got["ov"], "and ours is there too"

    @requires_pwsh
    @pytest.mark.parametrize("held", ["UV_FIND_LINKS", "PIP_FIND_LINKS"])
    def test_an_unrelated_find_links_does_not_cost_the_exclusions(
        self, tmp_path: pathlib.Path, held: str
    ):
        """
        The three are restored independently. Grouping them meant a shell carrying a
        corporate wheel mirror in PIP_FIND_LINKS silently lost the brotli exclusions and
        got the sdist build back -- a setting with nothing to do with the drop list.
        """
        overrides = self._stage(tmp_path)
        got = self._invoke(tmp_path, preset = f"$env:{held} = 'https://mirror.example/whl'")
        assert got["ov"] == str(overrides), f"{held} is unrelated to the overrides"
        # Round 16: ours is PREPENDED rather than skipped, because standing down left the
        # staged win_arm64 wheels out of the search entirely. The caller's entry still
        # has to survive, which is what this test was written to protect.
        value = got[{"UV_FIND_LINKS": "uvfl", "PIP_FIND_LINKS": "pipfl"}[held]]
        assert "https://mirror.example/whl" in value, "the caller's own value survives"
        assert value.endswith("https://mirror.example/whl"), "and ours goes in front of it"

    @requires_pwsh
    def test_a_deleted_overrides_file_says_so_rather_than_guessing(self, tmp_path: pathlib.Path):
        """Which packages were dropped depends on what the wheelhouse turned out to hold."""
        (tmp_path / "woa").mkdir()
        got = self._invoke(tmp_path)
        assert not got["ov"]
        assert "missing" in got["warned"] and "install.ps1" in got["warned"]

    def test_the_helper_is_a_faithful_copy_of_install_ps1s(self):
        """
        Get-UvSafePath exists in both scripts because neither can dot-source the other.
        A copy that drifts is worse than no copy: setup.ps1 would hand uv a path
        install.ps1 had already decided uv cannot read.
        """

        def normalized(source: str) -> str:
            lines = [
                line.rstrip()
                for line in source.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            indent = min(len(line) - len(line.lstrip()) for line in lines)
            return "\n".join(line[indent:] for line in lines)

        install = _function_source(
            (PACKAGE_ROOT / "install.ps1").read_text(encoding = "utf-8"),
            "Get-UvSafePath",
        )
        setup = _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Get-UvSafePath")
        assert normalized(install) == normalized(setup)

    def test_the_dependency_that_makes_this_necessary_is_still_there(self):
        """
        If studio.txt ever drops ddgs, this restore stops being load-bearing for brotli.
        It is still correct, but the reason recorded above would be stale, so fail loudly
        rather than let the comment rot.
        """
        studio_txt = PACKAGE_ROOT / "studio" / "backend" / "requirements" / "studio.txt"
        assert "ddgs" in studio_txt.read_text(encoding = "utf-8")

    def test_the_restore_runs_before_the_dependency_pass(self):
        """After it, the brotli resolve has already been attempted."""
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        restore = setup.index("\nRestore-WoaResolverEnvironment")
        stack = setup.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert restore < stack


class TestTheRecoveryReachesEveryModeThatNeedsIt:
    """Placement, which is what decided whether the two recoveries above fire at all.

    Both were originally written next to the torch index work, inside setup.ps1's
    `if (-not $NoTorchMode)` guard. Neither is about torch: install_python_stack.py
    installs studio.txt in every mode -- that is where ddgs resolves -- and it rewrites
    the manifest in every mode too.
    """

    @staticmethod
    def _setup() -> str:
        return SETUP_PS1.read_text(encoding = "utf-8")

    @staticmethod
    def _enclosing_blocks(text: str, needle: str) -> list:
        """The `{`-opening lines still unclosed where `needle` appears."""
        target = text.index(needle)
        stack = []
        for index, char in enumerate(text[:target]):
            if char == "{":
                stack.append(text.rfind("\n", 0, index) + 1)
            elif char == "}" and stack:
                stack.pop()
        return [text[start : text.index("\n", start)].strip() for start in stack]

    def test_the_restore_is_not_trapped_in_the_no_torch_guard(self):
        blocks = self._enclosing_blocks(self._setup(), "\nRestore-WoaResolverEnvironment")
        assert not any("NoTorchMode" in b for b in blocks), (
            "UNSLOTH_NO_TORCH=1 still installs studio.txt, and ddgs -> httpx[brotli] -> "
            f"Brotli has no win_arm64 wheel. Enclosing blocks: {blocks}"
        )

    def test_the_index_re_export_is_not_trapped_either(self):
        blocks = self._enclosing_blocks(
            self._setup(),
            "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $WinArm64TorchIndexUrl",
        )
        assert not any(
            "NoTorchMode" in b for b in blocks
        ), f"the manifest is rewritten in no-torch mode too. Enclosing blocks: {blocks}"

    def test_the_recovered_index_is_put_back_in_the_environment(self):
        """
        The bug this guards: recovering the index into a local variable only. The
        dependency pass rewrites the manifest from UNSLOTH_WOA_SELECTED_TORCH_INDEX, so a
        fresh-shell update would write one with no index at all -- erasing, on the first
        update, the only record of the one thing that cannot be re-derived from the host.
        """
        text = self._setup()
        assign = text.index("$WinArm64TorchIndexUrl = if (")
        export = text.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $WinArm64TorchIndexUrl")
        stack = text.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert assign < export < stack, "recovered, re-exported, then read by the stack"
        assert (
            "if ($WinArm64TorchIndexUrl) {" in text[export - 120 : export]
        ), "guarded: an empty recovery must not export an empty value"

    def test_studio_txt_is_installed_in_no_torch_mode(self):
        """
        The premise of the placement test above. If the studio.txt pass ever moves under
        a NO_TORCH guard, the reasoning changes and this should be revisited rather than
        quietly left stale.
        """
        source = STACK_PY.read_text(encoding = "utf-8")
        call = source.index('req = REQ_ROOT / "studio.txt"')
        line_start = source.rfind("\n", 0, source.rindex("pip_install(", 0, call)) + 1
        indent = len(source[line_start:]) - len(source[line_start:].lstrip())
        assert (
            indent == 4
        ), "the studio.txt install is no longer unconditional inside install_python_stack()"
        skip_list = source[source.index("NO_TORCH_SKIP_PACKAGES = {") :][:400]
        assert "ddgs" not in skip_list, "ddgs is still installed when NO_TORCH is set"


class TestTheRecoveryHappensWhileThereIsStillSomethingToRead:
    """Ordering against the manifest drop, which decides whether any of this works.

    setup.ps1 removes unsloth_install_manifest.json before it replaces pip, torch and
    triton, so a run killed in those leaves the venv marked half-built. The recovery reads
    that same file. Placed after the drop it read a file that no longer existed and
    returned empty every time -- on precisely the fresh-shell path the manifest exists to
    serve, and silently, because empty is also the legitimate answer for an older install.
    """

    def test_the_index_is_read_before_the_manifest_is_deleted(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        read = text.index("Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        drop = text.index("install_manifest.remove_manifest()")
        assert read < drop, (
            "the recovery reads unsloth_install_manifest.json; after the drop there is "
            "nothing left to read and the fresh-shell path silently gets no index"
        )

    def test_both_still_sit_inside_the_dependency_guard(self):
        """No point recovering for a run that installs nothing."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        guard = text.index("if (-not $SkipPythonDeps) {")
        read = text.index("Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        restore = text.index("\nRestore-WoaResolverEnvironment")
        assert guard < read and guard < restore


class TestThePublishedIndexIsTheOneTorchCameFrom:
    """What install_python_stack.py is told to repair from.

    The native Windows-on-ARM install resolves torch from the channel install.ps1 probed,
    while $TorchInstallIndexUrl still names the driver-derived family. Publishing the
    latter pointed any repair at an index with no win_arm64 CUDA wheel, and named the
    flavor cu130 when the installed wheel is +cu134.
    """

    def test_the_publish_block_reads_the_effective_index(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$_expectedLeaf = Get-TorchIndexLeaf $_effectiveTorchIndexUrl" in text
        assert "$env:UNSLOTH_TORCH_INSTALL_INDEX_URL = $_effectiveTorchIndexUrl" in text

    def test_it_defaults_to_the_old_value_and_only_torch_moves_it(self):
        """Every non-CUDA path must publish exactly what it published before."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        default = text.index("$_effectiveTorchIndexUrl = $TorchInstallIndexUrl")
        assign = text.index("$_effectiveTorchIndexUrl = $_cudaIndexUrl")
        publish = text.index("$_expectedLeaf = Get-TorchIndexLeaf $_effectiveTorchIndexUrl")
        assert default < assign < publish, "default, then the install, then the publish"
        assert text.count("$_effectiveTorchIndexUrl = ") == 2, "only the CUDA install may move it"

    def test_the_nvidia_channel_publishes_no_flavor_tag(self):
        """
        Get-TorchIndexLeaf on the NVIDIA channel yields `nvtorch_oot`, which is not a CUDA
        family name, so the tag resolves to $null and nothing is published. That is the
        honest answer -- the installed wheel is +cu134, which this vocabulary cannot name
        -- and it is what stops a repair from "correcting" the venv to cu130.
        """
        if PWSH is None:
            pytest.skip("pwsh not available")
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Get-TorchIndexLeaf"),
                _function_source(text, "Test-CudaFamilyLeaf"),
                "$leaf = Get-TorchIndexLeaf 'https://pypi.nvidia.com/nvtorch_oot'",
                'Write-Output "$leaf|$(Test-CudaFamilyLeaf $leaf)"',
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        leaf, is_cuda = done.stdout.strip().splitlines()[-1].split("|")
        assert leaf == "nvtorch_oot"
        assert is_cuda == "False", "a CUDA family leaf here would publish a wrong flavor"


class TestTheLlamaArm64CudaOptOut:
    """UNSLOTH_LLAMA_ARM64_CUDA=0, honoured on every path that can reach the branch."""

    @staticmethod
    def _arm64_nvidia_branches() -> list:
        """Every `if ...has_usable_nvidia...` whose enclosing branches select ARM64.

        Parsed rather than grepped: the same attribute guards the x64, Linux and macOS
        paths, which must NOT be gated, so a textual sweep either misses branches or
        demands gates where they would be wrong.
        """
        import ast

        tree = ast.parse(STACK_LLAMA.read_text(encoding = "utf-8"))
        found = []

        def uses(node, name: str) -> bool:
            return any(isinstance(n, ast.Attribute) and n.attr == name for n in ast.walk(node))

        def uses_positively(node, name: str) -> bool:
            """`not host.has_usable_nvidia` selects the CPU path and is not our business."""
            negated = {
                id(n.operand)
                for n in ast.walk(node)
                if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
            }
            return any(
                isinstance(n, ast.Attribute) and n.attr == name and id(n) not in negated
                for n in ast.walk(node)
            )

        def visit(node, arm64: bool):
            if isinstance(node, ast.If):
                # `elif` is an If inside orelse, so each one re-decides for itself.
                here = arm64 or (
                    uses(node.test, "is_arm64")
                    and not uses(node.test, "is_linux")
                    and not uses(node.test, "is_macos")
                )
                if here and uses_positively(node.test, "has_usable_nvidia"):
                    found.append((node.lineno, ast.unparse(node.test)))
                for stmt in node.body:
                    visit(stmt, here)
                for stmt in node.orelse:
                    visit(stmt, arm64)
                return
            for child in ast.iter_child_nodes(node):
                visit(child, arm64)

        visit(tree, False)
        return found

    def test_every_arm64_cuda_branch_is_gated(self):
        """
        Three entry points reach ARM64 CUDA independently: direct_upstream_release_plan,
        resolve_upstream_asset_choice (via resolve_asset_choice's fallbacks), and
        resolve_asset_choice's own published-artifact branch. An escape hatch honoured on
        some of them is worse than none, because nothing tells the user which path ran.
        """
        branches = self._arm64_nvidia_branches()
        assert len(branches) >= 3, branches
        ungated = [b for b in branches if "_upstream_arm64_cuda_allowed" not in b[1]]
        assert not ungated, f"ungated ARM64 CUDA branch(es): {ungated}"

    def test_the_x64_paths_are_not_gated_by_the_arm64_opt_out(self):
        """The negative control: this flag must not disable CUDA on ordinary hardware."""
        import ast

        tree = ast.parse(STACK_LLAMA.read_text(encoding = "utf-8"))
        arm64_lines = {line for line, _ in self._arm64_nvidia_branches()}
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.lineno not in arm64_lines:
                test = ast.unparse(node.test)
                if "has_usable_nvidia" in test:
                    assert "_upstream_arm64_cuda_allowed" not in test, test

    def test_the_published_artifact_branch_is_gated_too(self):
        """
        A cliff rather than a bug visible today: the published windows-arm64-cuda branch
        returns before the unverified-upstream tail, so gating only the tail meant
        UNSLOTH_LLAMA_ARM64_CUDA=0 would keep working right up until the fork published an
        approved artifact, then silently stop.
        """
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def resolve_asset_choice(")
        body = source[start:]
        marker = body.index("host.is_windows and host.is_arm64")
        branch = body[marker : marker + 4000]
        gate = branch.index("if host.has_usable_nvidia")
        published = branch.index("published_windows_cuda_attempts(")
        assert "_upstream_arm64_cuda_allowed()" in branch[gate : gate + 120]
        assert gate < published, "the gate must precede the published lookup"

    def test_the_now_unreachable_inner_check_is_gone(self):
        """
        With the branch gated, a second test inside it could only ever be true. Left in,
        it reads as though a CPU fallback still lives there and invites someone to
        "fix" the outer gate away.
        """
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def resolve_asset_choice(")
        assert "if _upstream_arm64_cuda_allowed():" not in source[start:]

    def test_the_docstring_matches_the_scope(self):
        """The helper documented itself as upstream-only; it now gates every bundle."""
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def _upstream_arm64_cuda_allowed(")
        doc = source[start : source.index('"""', source.index('"""', start) + 3)]
        assert "published or upstream" in doc

    def test_the_upstream_resolver_branch_specifically(self):
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def resolve_upstream_asset_choice(")
        end = source.index("\ndef ", start + 10)
        body = source[start:end]
        marker = body.index("if host.is_windows and host.is_arm64:")
        arm64_block = body[marker : marker + 900]
        assert (
            "_upstream_arm64_cuda_allowed()" in arm64_block
        ), "the Windows ARM64 CUDA branch of resolve_upstream_asset_choice is ungated"


class TestAMigratedX64VenvIsRebuiltAsArm64:
    """install.ps1: an upgrade must not leave a WoA NVIDIA host on the emulated stack.

    The migration branches keep a healthy legacy ~/.unsloth/studio/.venv exactly as it is,
    and on this host those are x64 by design -- every Windows-on-ARM install predating the
    native stack bootstrapped an emulated x64 interpreter. The venv-platform guard then
    saw win-amd64 and stood down to that same x64 stack, which is the state this PR exists
    to replace. Only a SECOND installer run recovered, because migrating makes the layout
    "new" and the new-layout branch does preserve-and-recreate.
    """

    INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"

    @staticmethod
    def _text() -> str:
        return TestAMigratedX64VenvIsRebuiltAsArm64.INSTALL_PS1.read_text(encoding = "utf-8")

    def test_the_rebuild_runs_before_venv_creation(self):
        """
        It works by making $VenvPython absent, so the existing creation block builds an
        ARM64 venv. After that block it would be too late and would need its own copy.
        """
        text = self._text()
        rebuild = text.index("$script:WoaNativeCudaTorch -and $_Migrated")
        create = text.index("if (-not (Test-Path -LiteralPath $VenvPython)) {")
        assert rebuild < create

    def test_it_preserves_the_old_environment(self):
        """Same rollback the new-layout branch uses; the user's packages are recoverable."""
        text = self._text()
        block = text[text.index("$script:WoaNativeCudaTorch -and $_Migrated") :][:1800]
        assert "Start-StudioVenvRollback -ExistingDir $VenvDir" in block

    def test_a_failed_rollback_keeps_the_old_behaviour(self):
        """Losing the user's environment is never worth a native stack."""
        text = self._text()
        block = text[text.index("$script:WoaNativeCudaTorch -and $_Migrated") :][:1800]
        assert "} catch {" in block
        assert "using the x64 stack instead" in block

    def test_the_rebuild_clears_the_migrated_flag(self):
        """
        The regression that would otherwise follow: $_Migrated drives an upgrade-in-place
        far below, which installs unsloth with --no-deps and --reinstall-package. Against
        a freshly created, empty venv that produces an unsloth with no dependencies.
        """
        text = self._text()
        block = text[text.index("$script:WoaNativeCudaTorch -and $_Migrated") :][:1800]
        rollback = block.index("Start-StudioVenvRollback")
        cleared = block.index("$_Migrated = $false")
        assert rollback < cleared, "cleared only after the environment is safely moved"
        # And the flag really does still gate that path.
        assert "if ($_Migrated) {" in text

    def test_it_only_touches_a_venv_this_run_migrated(self):
        """
        A new-layout venv was already moved aside above, and a venv created moments ago
        came from the interpreter this run chose. Rebuilding either would be pointless
        work at best and a second rollback at worst.
        """
        text = self._text()
        line = text[text.index("if ($script:WoaNativeCudaTorch -and $_Migrated") :].split("\n")[0]
        assert "$_Migrated" in line
        assert "Test-Path -LiteralPath $VenvPython" in line

    def test_the_platform_guard_below_still_has_the_final_say(self):
        """
        Belt and braces: if the rollback failed, the venv is still x64 and the existing
        guard must still disable native mode rather than install win_arm64-only specs
        into it.
        """
        text = self._text()
        rebuild = text.index("$script:WoaNativeCudaTorch -and $_Migrated")
        guard = text.index('if ($_woaVenvPlatform -ne "win-arm64") {')
        assert rebuild < guard
        block = text[guard:][:600]
        assert "$script:WoaNativeCudaTorch = $false" in block
        assert "$script:WoaTorchIndexUrl = $null" in block


class TestTheOptOutBundleSurvivesTheKindCheck:
    """setup.ps1's mismatch check must expect the bundle the selector actually installs.

    With UNSLOTH_LLAMA_ARM64_CUDA=0 the selector installs a `windows-arm64` CPU bundle on
    an NVIDIA Windows-on-ARM host. Expecting only `windows-arm64-cuda` there called that
    correct install mismatched and deleted it on every setup and update -- and an update
    that then cannot download leaves the user with no llama.cpp at all.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "value, expected",
        [
            # Not opted out: CUDA is preferred, and the CPU bundle the selector falls
            # back to when no ARM64 CUDA asset exists is valid too.
            ("", "windows-arm64-cuda,windows-arm64"),
            ("1", "windows-arm64-cuda,windows-arm64"),
            ("true", "windows-arm64-cuda,windows-arm64"),
            ("0", "windows-arm64"),
            ("false", "windows-arm64"),
            ("no", "windows-arm64"),
            ("off", "windows-arm64"),
            ("OFF", "windows-arm64"),
            (" 0 ", "windows-arm64"),
        ],
    )
    def test_the_expected_kind_follows_the_opt_out(self, value: str, expected: str):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_arm64CudaOptOut =")
        end = text.index('} else { @("windows-cuda") }', start) + len(
            '} else { @("windows-cuda") }'
        )
        script = "\n".join(
            [
                "function Test-WinArm64Venv { $true }",
                text[start:end].strip(),
                "Write-Output ($_nvidiaKinds -join ',')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
            env = {**os.environ, "UNSLOTH_LLAMA_ARM64_CUDA": value},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected

    @requires_pwsh
    def test_an_x64_venv_is_unaffected_by_the_flag(self):
        """The flag is ARM64-only; an emulated x64 venv installs windows-cuda regardless."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_arm64CudaOptOut =")
        end = text.index('} else { @("windows-cuda") }', start) + len(
            '} else { @("windows-cuda") }'
        )
        for value in ("", "0"):
            script = "\n".join(
                [
                    "function Test-WinArm64Venv { $false }",
                    text[start:end].strip(),
                    "Write-Output ($_nvidiaKinds -join ',')",
                ]
            )
            done = subprocess.run(
                [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output = True,
                text = True,
                timeout = 120,
                env = {**os.environ, "UNSLOTH_LLAMA_ARM64_CUDA": value},
            )
            assert done.returncode == 0, done.stderr
            assert done.stdout.strip().splitlines()[-1] == "windows-cuda"

    @requires_pwsh
    def test_the_opt_out_arm_stays_exclusive(self):
        """
        A CUDA bundle installed before the flag was set must still be replaced by the one
        the flag asks for, so the opt-out arm expects the CPU kind INSTEAD of the CUDA
        kind. Only the not-opted-out arm accepts both.
        """
        assert self._kinds("0") == "windows-arm64", "opted out: CUDA is no longer valid"
        assert self._kinds("") == "windows-arm64-cuda windows-arm64"

    @staticmethod
    def _kinds(value: str) -> str:
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_arm64CudaOptOut =")
        tail = '} else { @("windows-cuda") }'
        end = text.index(tail, start) + len(tail)
        script = "\n".join(
            [
                "function Test-WinArm64Venv { $true }",
                text[start:end].strip(),
                "Write-Output ($_nvidiaKinds -join ' ')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
            env = {**os.environ, "UNSLOTH_LLAMA_ARM64_CUDA": value},
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1]

    def test_the_cpu_fallback_is_a_real_selector_outcome(self):
        """
        The premise of widening: resolve_asset_choice falls through to the published
        windows-arm64 bundle when no ARM64 CUDA asset is available on an NVIDIA host.
        """
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def resolve_asset_choice(")
        body = source[start:]
        marker = body.index("host.is_windows and host.is_arm64")
        assert (
            'published_asset_choice_for_kind(release, "windows-arm64")'
            in body[marker : marker + 4000]
        )

    def test_widening_does_not_strand_anyone_on_cpu(self):
        """
        The installer's already-satisfied short-circuit is per candidate, and CUDA is
        attempted first, so a CPU bundle accepted here is still replaced the day an ARM64
        CUDA asset appears. If that ever became a whole-plan check, this would need
        revisiting.
        """
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        raise_at = source.index("raise ExistingInstallSatisfied(attempt, tried_fallback)")
        window = source[max(0, raise_at - 1200) : raise_at]
        assert (
            "choice = attempt" in window
        ), "the reuse check is per attempt; a plan-level one would pin the user to CPU"

    def test_the_falsy_spellings_match_the_python_helper(self):
        """One vocabulary; the two must not drift apart."""
        source = STACK_LLAMA.read_text(encoding = "utf-8")
        start = source.index("def _upstream_arm64_cuda_allowed(")
        body = source[start : source.index("\ndef ", start + 10)]
        python_set = set(re.findall(r'"(0|false|no|off)"', body))
        ps_block = SETUP_PS1.read_text(encoding = "utf-8")
        ps_line = ps_block[ps_block.index("$_arm64CudaOptOut =") :].split("\n")[0]
        ps_set = set(re.findall(r'"(0|false|no|off)"', ps_line))
        assert python_set == ps_set == {"0", "false", "no", "off"}


INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"


def _ps_function(path: pathlib.Path, name: str) -> str:
    return _function_source(path.read_text(encoding = "utf-8"), name)


class TestTheCudaWheelProbeIsNotFooled:
    """install.ps1: what the probe accepts as proof of a win_arm64 CUDA wheel.

    Driven against synthetic PEP 503 pages with Invoke-RestMethod stubbed, so these are
    offline and deterministic. The live NVIDIA channels are exercised separately in
    temp/sim10282/probe_test.ps1.
    """

    @staticmethod
    def _probe(
        body: str,
        project: str = "torch",
        minor: str = "3.13",
    ) -> str:
        script = "\n".join(
            [
                "function Join-UrlPath { param([string]$Base,[string]$Path)",
                "  return ($Base.TrimEnd('/') + '/' + $Path.TrimStart('/')) }",
                f"function Invoke-RestMethod {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) return @'\n{body}\n'@ }}",
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Get-WoaCudaWheelVersion"),
                f"$v = Get-WoaCudaWheelVersion -IndexUrl 'https://x.test/i' -PythonMinor '{minor}' -Project '{project}'",
                'Write-Output "[$v]"',
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1][1:-1]

    @requires_pwsh
    def test_a_percent_encoded_cpu_wheel_is_rejected(self):
        """
        PEP 503 hrefs encode `+` as %2B. Matching a literal `\\+cpu` never fired on the
        encoded spelling, so a CPU-only mirror read as CUDA and the host went native on
        CPU torch -- worse than staying emulated, because the GPU then goes unused with
        no fallback.
        """
        body = '<a href="torch-2.14.0%2Bcpu-cp313-cp313-win_arm64.whl">t</a>'
        assert self._probe(body) == ""

    @requires_pwsh
    def test_an_untagged_wheel_is_rejected(self):
        """
        PyPI's own win_arm64 torch wheels carry no local version at all. `not +cpu`
        accepted them; CUDA has to be established positively.
        """
        assert self._probe('<a href="torch-2.14.0-cp313-cp313-win_arm64.whl">t</a>') == ""

    @requires_pwsh
    @pytest.mark.parametrize("spelling", ["2.14.0%2Bcu134", "2.14.0+cu134"])
    def test_both_spellings_of_a_cuda_wheel_are_accepted(self, spelling: str):
        body = f'<a href="torch-{spelling}-cp313-cp313-win_arm64.whl">t</a>'
        assert self._probe(body) == "2.14.0+cu134"

    @requires_pwsh
    def test_the_interpreter_tag_still_has_to_match(self):
        body = '<a href="torch-2.14.0%2Bcu134-cp311-cp311-win_arm64.whl">t</a>'
        assert self._probe(body, minor = "3.13") == ""
        assert self._probe(body, minor = "3.11") == "2.14.0+cu134"

    @requires_pwsh
    def test_the_platform_still_has_to_match(self):
        body = '<a href="torch-2.14.0%2Bcu134-cp313-cp313-win_amd64.whl">t</a>'
        assert self._probe(body) == ""

    @requires_pwsh
    def test_the_newest_release_wins(self):
        body = " ".join(
            f'<a href="torch-{v}-cp313-cp313-win_arm64.whl">t</a>'
            for v in ("2.9.0%2Bcu134", "2.14.0%2Bcu134", "2.11.0%2Bcu134")
        )
        assert self._probe(body) == "2.14.0+cu134"

    @requires_pwsh
    def test_a_dev_stamp_does_not_look_older_than_a_release(self):
        """2.15.0.dev... is newer than 2.14.0; a plain string sort would disagree."""
        body = " ".join(
            f'<a href="torch-{v}-cp313-cp313-win_arm64.whl">t</a>'
            for v in ("2.14.0%2Bcu134", "2.15.0.dev20260819%2Bcu134")
        )
        assert self._probe(body) == "2.15.0.dev20260819+cu134"

    @requires_pwsh
    def test_the_newest_dev_stamp_of_one_release_wins(self):
        body = " ".join(
            f'<a href="torch-{v}-cp313-cp313-win_arm64.whl">t</a>'
            for v in ("2.15.0.dev20260819%2Bcu134", "2.15.0.dev20260728%2Bcu134")
        )
        assert self._probe(body) == "2.15.0.dev20260819+cu134"

    @requires_pwsh
    def test_an_empty_or_broken_page_is_not_a_wheel(self):
        assert self._probe("") == ""
        assert self._probe("<html><body>nothing here</body></html>") == ""


class TestTorchaudioIsOnlyTakenAsAMatchedPair:
    """The GA channel publishes torch 2.14.0+cu134 beside torchaudio 2.11.0+cu134.

    torchaudio 2.11 dropped the exact `torch==` pin that used to make such a pair
    unresolvable (2.10.0 still had it), and the native specs are open-ended, so nothing
    else stops the resolver from installing them together and leaving torchaudio's
    extension to fail against a libtorch three minors newer.
    """

    @staticmethod
    def _match(torch_v: str, audio_v: str) -> bool:
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaAudioMatchesTorch"),
                f"Write-Output (Test-WoaAudioMatchesTorch -TorchVersion '{torch_v}' -AudioVersion '{audio_v}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1] == "True"

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch_v, audio_v, expected, why",
        [
            ("2.14.0+cu134", "2.11.0+cu134", False, "the pair the GA channel serves today"),
            ("2.14.0+cu134", "2.14.0+cu134", True, "what a matched channel would serve"),
            ("2.14.0+cu134", "2.14.1+cu134", True, "patch releases pair"),
            ("2.15.0.dev20260819+cu134", "2.11.0.dev20260819+cu134", False, "nightly, mismatched"),
            ("2.15.0.dev20260819+cu134", "2.15.0.dev20260728+cu134", True, "nightly, same minor"),
            ("2.14.0+cu134", "", False, "no audio wheel at all"),
            ("", "2.14.0+cu134", False, "no torch wheel at all"),
        ],
    )
    def test_only_a_matching_major_minor_enables_audio(
        self, torch_v: str, audio_v: str, expected: bool, why: str
    ):
        assert self._match(torch_v, audio_v) is expected, why

    def test_the_probe_compares_versions_rather_than_existence(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        # rindex, not index: the first occurrence is the reset at the top of the probe,
        # which must stay a plain $false so a re-probe cannot inherit an earlier answer.
        block = text[text.rindex("$script:WoaTorchAudio = ") :][:400]
        assert "Test-WoaAudioMatchesTorch" in block, (
            "torchaudio was enabled on existence alone, which is how the mismatched "
            "GA pair became installable"
        )


class TestPrereleasesAreOnlyForTheNightlyChannel:
    """setup.ps1 must gate --prerelease=allow the way install.ps1 already does.

    `allow` means every prerelease, and it rides on a command that also carries
    unsafe-best-match and public PyPI, so on the GA channel a prerelease of torch or of
    any shared dependency could outrank the stable build this host exists to install.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "index, expect_pre",
        [
            ("https://pypi.nvidia.com/nvtorch_oot", False),
            ("https://pypi.nvidia.com/nvtorch_oot_nightly", True),
            ("", False),
        ],
    )
    def test_the_flag_follows_the_channel(self, index: str, expect_pre: bool):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64IndexArgs = if (")
        end = text.index("} else { @() }", start) + len("} else { @() }")
        script = "\n".join(
            [
                "$WinArm64Venv = $true",
                "$UseUv = $true",
                f"$WinArm64TorchIndexUrl = '{index}'",
                text[start:end],
                "Write-Output ($WinArm64IndexArgs -join ' ')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        out = done.stdout.strip().splitlines()[-1]
        assert ("--prerelease=allow" in out) is expect_pre, out
        assert "unsafe-best-match" in out, "the other flags are unconditional"

    @requires_pwsh
    def test_every_other_host_gets_no_flags_at_all(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64IndexArgs = if (")
        end = text.index("} else { @() }", start) + len("} else { @() }")
        script = "\n".join(
            [
                "$WinArm64Venv = $false",
                "$UseUv = $true",
                "$WinArm64TorchIndexUrl = 'https://pypi.nvidia.com/nvtorch_oot_nightly'",
                text[start:end],
                "Write-Output \"[$($WinArm64IndexArgs -join ' ')]\"",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "[]"

    def test_both_scripts_gate_on_the_same_thing(self):
        """One rule; two files. Drift here is invisible until a resolve goes wrong."""
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert re.search(r"-match 'nightly'", text), f"{path.name} lost the gate"
            for line in text.splitlines():
                if "--prerelease=allow" in line and "#" not in line.split("--prerelease")[0]:
                    assert "@(" in line, f"{path.name}: unexpected shape: {line.strip()}"


class TestManifestWriterAndReaderAcceptTheSameSet:
    """A value the writer persists and the reader refuses is worse than none at all.

    urlsplit().hostname strips the port, so `https://pypi.nvidia.com:443/nvtorch_oot`
    passed the writer's host test while setup.ps1's pattern, which allows no port,
    rejected it -- and the fresh-shell update silently lost the index it had recorded.
    """

    PORTED = "https://pypi.nvidia.com:443/nvtorch_oot"

    def test_the_writer_refuses_a_url_the_reader_cannot_read(self, tmp_path: pathlib.Path):
        im.write_manifest(root = tmp_path, req_root = tmp_path, woa_torch_index = self.PORTED)
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload

    @requires_pwsh
    def test_and_the_reader_still_refuses_it(self, tmp_path: pathlib.Path):
        (tmp_path / "unsloth_install_manifest.json").write_text(
            json.dumps({"schema": 1, "woa_torch_index": self.PORTED}),
            encoding = "utf-8",
        )
        assert TestReadSide._invoke(tmp_path) == ""

    @requires_pwsh
    @pytest.mark.parametrize(
        "url",
        [
            "https://pypi.nvidia.com/nvtorch_oot",
            "https://pypi.nvidia.com/nvtorch_oot_nightly",
        ],
    )
    def test_the_two_agree_on_what_is_acceptable(self, tmp_path: pathlib.Path, url: str):
        """The pair that matters: written, then read back unchanged."""
        written = tmp_path / "w"
        written.mkdir()
        im.write_manifest(root = written, req_root = written, woa_torch_index = url)
        payload = json.loads((written / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert payload["woa_torch_index"] == url
        assert TestReadSide._invoke(written) == url


class TestTheSuppliedPyarrowWheelIsValidated:
    """install.ps1: UNSLOTH_PYARROW_WHEEL decides whether the native path is taken.

    Every other branch of Get-WoaPyarrowSource checks the interpreter and platform tags.
    This one accepted any existing file, so an x64 wheel, a wheel for another minor, or a
    truncated download selected the native stack -- and the staging step trusts the .whl
    name without opening it, so the run failed at resolution having already given up the
    working x64 path.
    """

    # A REAL archive. The check opens the file rather than reading its first two bytes,
    # because a PK header proves nothing about an interrupted download -- which is now
    # one of the cases below.
    ZIP = "zip"
    HEADER_ONLY = b"PK\x03\x04" + b"\0" * 64

    @staticmethod
    def _write(path: pathlib.Path, content) -> None:
        if content == "zip":
            import zipfile
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("pyarrow/__init__.py", "")
        else:
            path.write_bytes(content)

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, content, expected, why",
        [
            ("pyarrow-21.0.0-cp313-cp313-win_arm64.whl", ZIP, "local", "the wheel this is for"),
            ("pyarrow-21.0.0-cp312-cp312-win_arm64.whl", ZIP, "", "another interpreter minor"),
            ("pyarrow-21.0.0-cp313-cp313-win_amd64.whl", ZIP, "", "an x64 wheel"),
            ("pyarrow-21.0.0-cp313-cp313-win_arm64.whl", b"not a zip", "", "a truncated download"),
            (
                "pyarrow-21.0.0-cp313-cp313-win_arm64.whl",
                HEADER_ONLY,
                "",
                "an interrupted one that still carries the PK signature",
            ),
            ("numpy-2.0.0-cp313-cp313-win_arm64.whl", ZIP, "", "a wheel for another project"),
            ("pyarrow-21.0.0.tar.gz", ZIP, "", "an sdist, which cannot be staged"),
        ],
    )
    def test_only_a_matching_readable_wheel_selects_native(
        self, tmp_path: pathlib.Path, name: str, content, expected: str, why: str
    ):
        wheel = tmp_path / name
        self._write(wheel, content)
        assert self._probe(str(wheel)) == expected, why

    @requires_pwsh
    def test_a_missing_file_is_ignored_rather_than_fatal(self, tmp_path: pathlib.Path):
        assert self._probe(str(tmp_path / "nope.whl")) == ""

    @staticmethod
    def _probe(wheel: str) -> str:
        """Get-WoaPyarrowSource with its network branches stubbed out."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "function substep { param($m, $c) }",
                "function Join-UrlPath { param($Base, $Path) return $Base }",
                "function Test-WoaWheelhouseIsLocal { $false }",
                "function Invoke-RestMethod { throw 'no network in this test' }",
                "$script:WoaWheelhouse = 'https://example.test/wheels'",
                _function_source(text, "Test-WoaWheelTags"),
                # The supplied-wheel branch opens the archive now, so its helper is
                # needed too; PowerShell does not hoist.
                _function_source(text, "Test-ZipArchiveReadable"),
                _function_source(text, "Get-WoaPyarrowSource"),
                f"$env:UNSLOTH_PYARROW_WHEEL = '{wheel}'",
                "Write-Output \"[$(Get-WoaPyarrowSource -PythonMinor '3.13')]\"",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1][1:-1]


class TestCallerResolverConfigurationSurvives:
    """install.ps1 must not discard what its own purge block just chose to keep.

    The purge above removes only variables pointing into this StudioHome's woa directory,
    precisely so a caller's corporate wheel source is left alone -- and then the three
    assignments overwrote them anyway.
    """

    def test_the_overrides_are_kept_or_folded_never_dropped(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$_woaOwnNames = @{}") :][:3200]
        assert "$env:UV_OVERRIDE -split" in block, "the caller's files are read"
        assert "$_woaKeepFiles += $_woaOvFull" in block, (
            "a file that names none of our packages is passed to uv where it is, which "
            "keeps its relative -r and wheel paths resolving"
        )
        assert (
            "$WoaOverrideLines += (Resolve-WoaOverrideLine" in block
        ), "and a conflicting one is folded line by line, rebased as it goes"
        assert "$_woaOwnNames.ContainsKey($_woaOvName)" in block, (
            "minus the packages this file declares -- uv combines override files and "
            "errors on a duplicate package, so a blind append could fail the resolve"
        )

    def test_the_find_links_are_appended_with_the_right_separators(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            '$env:UV_FIND_LINKS = if ($_woaCallerUvLinks) { "$WoaWheelDir,$_woaCallerUvLinks" }'
            in text
        ), "UV_FIND_LINKS is comma-separated"
        assert (
            '"$_woaSafeWheelDir $_woaCallerPipLinks"' in text
        ), "PIP_FIND_LINKS is split on whitespace, and ours must be the 8.3-safe form"

    def test_ours_is_searched_first(self):
        """A win_arm64 wheel staged for this host must win a tie against the same name."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert '"$WoaWheelDir,$_woaCallerUvLinks"' in text
        assert '"$_woaCallerUvLinks,$WoaWheelDir"' not in text

    def test_the_python_side_can_read_an_appended_value(self):
        """
        install_python_stack.py split find-links on os.pathsep alone, which would have
        read "dirA,dirB" as one unusable path now that appending is possible.
        """
        source = STACK_PY.read_text(encoding = "utf-8")
        assert 're.split(r"[,\\s" + re.escape(os.pathsep) + r"]+"' in source


class TestTheProbeAsksForTheInterpretersAbi:
    """install.ps1 keyed its wheel search on the minor alone.

    A free-threaded build installs cp313-cp313t, not cp313-cp313, so probing for the GIL
    tag found the ordinary wheels on the index, enabled the native stack, and left the
    resolve to fail on wheels the venv cannot use -- after the x64 fallback had been given
    up. Reachable because Find-CompatiblePython's PATH scan and its `py -0p` enumeration
    both accept any interpreter whose --version matches, and the native path ranks ARM64
    builds first.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "minor, free_threaded, expected",
        [("3.13", False, "cp313"), ("3.13", True, "cp313t"), ("3.11", True, "cp311t")],
    )
    def test_the_abi_tag_follows_the_build(self, minor: str, free_threaded: bool, expected: str):
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Get-WoaAbiTag"),
                f"Write-Output (Get-WoaAbiTag -PythonMinor '{minor}' "
                f"-FreeThreaded ${str(free_threaded).lower()})",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected

    @requires_pwsh
    @pytest.mark.parametrize(
        "abi, wheel_abi, found",
        [
            ("", "cp313", True),  # a GIL interpreter, unchanged
            ("cp313t", "cp313", False),  # free-threaded must not take a GIL wheel
            ("cp313t", "cp313t", True),  # and does take its own
        ],
    )
    def test_only_wheels_of_that_abi_are_found(self, abi: str, wheel_abi: str, found: bool):
        body = f'<a href="torch-2.14.0%2Bcu134-cp313-{wheel_abi}-win_arm64.whl">t</a>'
        script = "\n".join(
            [
                "function Join-UrlPath { param([string]$Base,[string]$Path)",
                "  return ($Base.TrimEnd('/') + '/' + $Path.TrimStart('/')) }",
                f"function Invoke-RestMethod {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) return @'\n{body}\n'@ }}",
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Get-WoaCudaWheelVersion"),
                f"$v = Get-WoaCudaWheelVersion -IndexUrl 'https://x.test/i' -PythonMinor '3.13' -AbiTag '{abi}'",
                'Write-Output "[$v]"',
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        got = done.stdout.strip().splitlines()[-1][1:-1]
        assert bool(got) is found, got

    def test_the_probe_takes_the_flag_and_the_call_sites_supply_it(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "function Initialize-WoaNativeCudaTorch {\n        param([string]$PythonMinor, [bool]$FreeThreaded = $false)"
            in text
        )
        # Both re-probes know which interpreter was chosen, so both must answer for it.
        # The flag is read into a variable first, so the guard can compare it as well as
        # the minor -- a same-minor free-threaded build has to re-probe too.
        assert text.count("Test-PythonFreeThreaded -PythonExe $DetectedPython.Path") == 2
        assert "-FreeThreaded $WoaDetectedFreeThreaded" in text
        assert "-FreeThreaded $_woaNewFreeThreaded" in text

    def test_the_staging_scan_uses_the_venv_abi(self):
        """
        The other half: keyed on the python tag, staging kept the cp313-cp313 wheels a
        free-threaded venv cannot install and discarded the cp313-cp313t ones it can.
        """
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$WoaWheelAbi = Get-WoaAbiTag -PythonMinor $WoaVenvMinor" in text
        assert "($abiTags -contains $WoaWheelAbi)" in text
        assert (
            "($WoaWheelStable -and ($abiTags -contains 'abi3'))" in text
        ), "free-threaded builds do not implement the stable ABI"
        assert (
            "$script:WoaVenvFreeThreaded = Test-PythonFreeThreaded -PythonExe $VenvPython" in text
        )

    @requires_pwsh
    def test_an_unknown_interpreter_answers_gil(self):
        """The historical assumption: unknown must not turn a working GIL host free-threaded."""
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-PythonFreeThreaded"),
                "Write-Output (Test-PythonFreeThreaded -PythonExe 'C:\\nope\\python.exe')",
                "Write-Output (Test-PythonFreeThreaded -PythonExe '')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["False", "False"]

    @requires_pwsh
    def test_it_reads_the_real_interpreter_correctly(self):
        """Executed against the interpreter running this suite, whichever build that is."""
        import sysconfig

        expected = "True" if sysconfig.get_config_var("Py_GIL_DISABLED") else "False"
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-PythonFreeThreaded"),
                f"Write-Output (Test-PythonFreeThreaded -PythonExe '{sys.executable}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected


class TestTheAbiReprobeFiresOnAMatchingMinor:
    """The hole left by keying the re-probe on the minor alone.

    The first probe runs before an interpreter exists, so it has to assume a GIL build.
    A free-threaded 3.13t selected for a 3.13 request then matched on minor, skipped the
    only call that passes -FreeThreaded, and left the cp313 answer standing -- so native
    mode was enabled on wheels the resulting cp313t venv cannot install.
    """

    def test_the_guard_compares_the_abi_as_well_as_the_minor(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$WoaProbedFreeThreaded = $false" in text
        assert (
            "($WoaDetectedFreeThreaded -ne $WoaProbedFreeThreaded)" in text
        ), "a 3.13t selected for a 3.13 request matches on minor and must still re-probe"

    def test_the_second_reprobe_compares_it_too(self):
        """After Install-PythonFromPythonOrg the ABI can change without the minor doing so."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "($_woaNewFreeThreaded -ne $WoaProbedFreeThreaded)" in text

    def test_the_detection_is_scoped_to_this_host(self):
        """A subprocess per run on every Windows x64 host would buy nothing."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$WoaDetectedFreeThreaded = $false") :][:600]
        assert '(Get-HostMachineArch) -eq "arm64"' in block

    @requires_pwsh
    @pytest.mark.parametrize(
        "probed_minor, probed_ft, minor, ft, should_reprobe, why",
        [
            ("3.13", False, "3.13", False, False, "nothing changed"),
            ("3.13", False, "3.12", False, True, "a different minor, as before"),
            ("3.13", False, "3.13", True, True, "THE BUG: same minor, free-threaded"),
            ("3.13", True, "3.13", False, True, "and back again, after a GIL install"),
        ],
    )
    def test_the_guard_decides_correctly(
        self, probed_minor, probed_ft, minor, ft, should_reprobe, why
    ):
        script = "\n".join(
            [
                f"$WoaProbedMinor = '{probed_minor}'",
                f"$WoaProbedFreeThreaded = ${str(probed_ft).lower()}",
                f"$DetectedPython = @{{ Version = '{minor}' }}",
                f"$WoaDetectedFreeThreaded = ${str(ft).lower()}",
                "if ($DetectedPython -and (",
                "        ($DetectedPython.Version -ne $WoaProbedMinor) -or",
                "        ($WoaDetectedFreeThreaded -ne $WoaProbedFreeThreaded))) {",
                "  Write-Output 'REPROBE' } else { Write-Output 'SKIP' }",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        got = done.stdout.strip().splitlines()[-1]
        assert (got == "REPROBE") is should_reprobe, why


class TestAnExplicitPinOutranksThePersistedIndex:
    """The recovery is a memory of what install.ps1 chose, not a decision.

    Letting it win meant a user who set UNSLOTH_TORCH_INDEX_URL or
    UNSLOTH_TORCH_INDEX_FAMILY to move to another CUDA mirror was silently still served
    by the previously recorded NVIDIA channel.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, woa, install_url, expected, why",
        [
            (
                "",
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://d.pytorch.org/whl/cu130",
                "https://pypi.nvidia.com/nvtorch_oot",
                "unpinned fresh shell: the recovery",
            ),
            (
                "https://mirror.test/cu129",
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://mirror.test/cu129",
                "https://mirror.test/cu129",
                "an explicit pin wins",
            ),
            (
                "",
                "",
                "https://d.pytorch.org/whl/cu130",
                "https://d.pytorch.org/whl/cu130",
                "no recovery, no pin: unchanged",
            ),
        ],
    )
    def test_the_pin_wins(self, pinned, woa, install_url, expected, why):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_cudaIndexUrl = if ($PinnedTorchIndexUrl)")
        end = text.index("else { $TorchInstallIndexUrl }", start) + len(
            "else { $TorchInstallIndexUrl }"
        )
        script = "\n".join(
            [
                f"$PinnedTorchIndexUrl = '{pinned}'",
                f"$WinArm64TorchIndexUrl = '{woa}'",
                f"$TorchInstallIndexUrl = '{install_url}'",
                text[start:end].strip(),
                "Write-Output $_cudaIndexUrl",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, why


class TestTheRestoreMergesRatherThanStandsDown:
    """A caller override must not cost the win_arm64 drop list.

    studio.txt installs ddgs, whose HTTP stack asks for httpx[brotli], and Brotli has no
    win_arm64 wheel -- so skipping the generated file because UV_OVERRIDE happened to be
    set sent the update at an sdist it cannot build.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "line, expected_name",
        [
            ('Brotli ; platform_machine == "AMD64"', "brotli"),
            ("brotli_cffi>=1.0", "brotli-cffi"),
            ("torch>=2.4", "torch"),
            ("pyarrow==21.0.0", "pyarrow"),
            ("# a comment", ""),
            ("", ""),
            ("-r other.txt", ""),
        ],
    )
    def test_requirement_names_are_canonical(self, line: str, expected_name: str):
        """PEP 503 normalisation, so Brotli and brotli_cffi compare as one name."""
        script = "\n".join(
            [
                _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Get-RequirementName"),
                f"Write-Output \"[$(Get-RequirementName -Line '{line}')]\"",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == f"[{expected_name}]"

    def test_disjoint_files_are_both_passed_and_conflicts_are_merged(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$_woaOursNames = Get-RequirementNames") :][:2600]
        assert '$env:UV_OVERRIDE = "$safeOverrides $($env:UV_OVERRIDE)"' in block, (
            "disjoint files need no rewriting, which keeps each file's relative "
            "references resolving against its own directory"
        )
        assert "overrides.merged.txt" in block, "only an actual conflict is merged"
        assert "(Get-RequirementName -Line $_woaOvLine) -in $_woaOursNames" not in block

    def test_the_caller_no_longer_suppresses_the_drop_list(self):
        """The regression: the whole restore used to sit under `if (-not $env:UV_OVERRIDE)`."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        body = _function_source(text, "Restore-WoaResolverEnvironment")
        # The drop list is read whatever the caller set; only the LAST step -- assigning
        # ours alone versus combining -- is allowed to look at UV_OVERRIDE.
        overrides_at = body.index("$safeOverrides = Get-UvSafePath $overrides")
        head = body[:overrides_at]
        assert head.count("$env:UV_OVERRIDE") <= 1, (
            "reaching the drop list must not depend on the caller having set nothing; "
            f"head still tests it: {head[-300:]}"
        )
        assert (
            "elseif (-not $env:UV_OVERRIDE) {" in body
        ), "the no-caller case still assigns ours alone"


class TestThePurgeKeepsWhatIsNotOurs:
    """The purge drops a PREVIOUS run's resolver settings, not the caller's.

    Since the assignments started PREPENDING this StudioHome's wheelhouse to whatever the
    caller had, a value that merely starts with an owned path still carries the caller's
    own entries behind it -- and removing the whole variable took an air-gapped mirror
    with it on the second `irm | iex` of one shell.
    """

    @staticmethod
    def _purge_block() -> str:
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('$_woaOwnedPrefix = Join-Path $StudioHome "woa"')
        end = text.index("if ($script:WoaNativeCudaTorch) {", start)
        return text[start:end]

    def test_the_block_works_entry_by_entry(self):
        block = self._purge_block()
        assert "$_woaKept" in block, "entries are kept, not just the whole value dropped"
        assert "-split '[,\\s]+'" in block, (
            "every separator the three variables are read with: uv takes UV_FIND_LINKS "
            "comma-separated and UV_OVERRIDE space-separated, pip splits on whitespace"
        )
        assert '"UV_FIND_LINKS" = ","' in block, "and each is rejoined with its own"

    @requires_pwsh
    @pytest.mark.parametrize(
        "var, value, expected, why",
        [
            ("UV_FIND_LINKS", "{home}/woa/wheels", "", "ours alone still goes"),
            (
                "UV_FIND_LINKS",
                "{home}/woa/wheels,/mnt/mirror",
                "/mnt/mirror",
                "the caller's mirror survives the comma form",
            ),
            (
                "PIP_FIND_LINKS",
                "{home}/woa/wheels /mnt/mirror",
                "/mnt/mirror",
                "and the whitespace form",
            ),
            (
                "PIP_FIND_LINKS",
                "/a {home}/woa/wheels /b",
                "/a /b",
                "ours is removed from the middle without disturbing the rest",
            ),
            (
                "UV_FIND_LINKS",
                "/mnt/mirror",
                "/mnt/mirror",
                "a value that is entirely the caller's is untouched",
            ),
            ("UV_OVERRIDE", "{home}/woa/overrides.txt", "", "ours alone still goes"),
            (
                "UV_OVERRIDE",
                "{home}/woa/overrides.txt /etc/ov.txt",
                "/etc/ov.txt",
                "a caller's override file survives",
            ),
        ],
    )
    def test_only_the_owned_entries_are_removed(self, var, value, expected, why):
        home = "/home/u/AppData/Local/unsloth"
        script = "\n".join(
            [
                f"$StudioHome = '{home}'",
                "function Get-UvSafePath { param([string]$p) return $p }",
                f"$env:{var} = '{value.format(home = home)}'",
                self._purge_block(),
                f"Write-Output ('[' + $env:{var} + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == f"[{expected}]", why


class TestAMalformedHandoffUrlDoesNotCostTheManifest:
    """write_manifest documents that it never raises.

    UNSLOTH_WOA_SELECTED_TORCH_INDEX comes in from the environment -- setup.ps1 forwards
    it as received -- and urlsplit raises ValueError on a malformed authority, before the
    write-side try block, losing the whole manifest.
    """

    @pytest.mark.parametrize(
        "bad",
        ["https://[", "https://pypi.nvidia.com:notaport/x", "https://[::1", "://"],
    )
    def test_the_manifest_is_still_written(self, tmp_path, bad):
        module = _load_manifest_module()
        written = module.write_manifest(tmp_path, woa_torch_index = bad)
        assert written is not None, "a bad URL must not take the manifest with it"
        payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload, "and it is certainly not persisted"

    def test_a_good_url_is_still_persisted(self, tmp_path):
        module = _load_manifest_module()
        written = module.write_manifest(
            tmp_path,
            woa_torch_index = "https://pypi.nvidia.com/nvtorch_oot/",
        )
        payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert payload["woa_torch_index"] == "https://pypi.nvidia.com/nvtorch_oot"


class TestAStaleTorchaudioIsRemoved:
    """A fresh-shell update on Windows on ARM cannot see UNSLOTH_WOA_HAS_TORCHAUDIO, so it
    installs torch/torchvision alone. uv leaves an already-installed torchaudio exactly
    where it is, and a channel that has moved torch to a new minor leaves that wheel's
    compiled extension linked against the previous libtorch.
    """

    @staticmethod
    def _block() -> str:
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("if ($WinArm64Venv -and $WinArm64NoAudio) {")
        end = text.index("# Triton for Windows enables torch.compile", start)
        return text[start:end]

    def test_the_removal_is_conditional_on_a_mismatch(self):
        block = self._block()
        assert "Fast-Uninstall torchaudio" in block
        assert (
            "$_woaAudioMm -ne $_woaTorchMm" in block
        ), "a matching audio wheel is the one this venv was installed with and stays"
        assert (
            "$_woaAudioProbe.Ok" in block
        ), "a probe that did not answer says nothing; it must not trigger a removal"

    def test_it_only_runs_where_audio_was_dropped(self):
        block = self._block()
        assert block.startswith("if ($WinArm64Venv -and $WinArm64NoAudio) {"), (
            "on every other host, and on a WoA index that DOES publish torchaudio, the "
            "trio asked for it and the resolver already matched the pair"
        )

    def test_it_runs_after_the_install_succeeded(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        failed = text.index('Exit-SetupFailure "PyTorch CUDA installation failed')
        assert (
            text.index("if ($WinArm64Venv -and $WinArm64NoAudio) {") > failed
        ), "removing audio before knowing torch installed would strip a working venv"


class TestTheWheelTagsAreMatchedAsFields:
    """`*cp313-cp313*` also matches cp313-cp313t.

    On an ordinary GIL 3.13 that let a free-threaded wheel select the native path, and the
    staging block then picked the same wheel -- uv rejected it only after the installer had
    already committed to the ARM64 venv and given up the working x64 stack.
    """

    def test_no_substring_match_is_left(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$tag-$AbiTag*" not in text, "every probe goes through the field parser"
        assert text.count("Test-WoaWheelTags") >= 7, (
            "the pyarrow probe (supplied wheel, PyPI, local wheelhouse, remote index), "
            "the CUDA wheel scan, and both staging sites"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, py, abi, expected, why",
        [
            (
                "pyarrow-21.0.0-cp313-cp313t-win_arm64.whl",
                "cp313",
                "cp313",
                False,
                "the regression: a free-threaded wheel on a GIL interpreter",
            ),
            ("pyarrow-21.0.0-cp313-cp313-win_arm64.whl", "cp313", "cp313", True, "its own"),
            (
                "pyarrow-21.0.0-cp313-cp313t-win_arm64.whl",
                "cp313",
                "cp313t",
                True,
                "and the free-threaded interpreter still finds its own",
            ),
            (
                "pyarrow-21.0.0-cp313-cp313-win_arm64.whl",
                "cp313",
                "cp313t",
                False,
                "which is not the GIL one",
            ),
            ("pyarrow-21.0.0-cp311-cp311-win_arm64.whl", "cp313", "cp313", False, "another minor"),
            (
                "torch-2.14.0+cu134-cp312.cp313-cp312.cp313-win_arm64.whl",
                "cp313",
                "cp313",
                True,
                "a dot-separated tag set is expanded, as PEP 425 says",
            ),
            (
                "pyarrow-21.0.0-1-cp313-cp313-win_arm64.whl",
                "cp313",
                "cp313",
                True,
                "an optional build tag does not shift the last three fields",
            ),
            ("garbage.whl", "cp313", "cp313", False, "unparseable is not installable"),
        ],
    )
    def test_the_matcher(self, name, py, abi, expected, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Test-WoaWheelTags"),
                f"Write-Output (Test-WoaWheelTags -Name '{name}' -PyTag '{py}' -AbiTag '{abi}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == str(expected), why


class TestThePurgeNeedsAPathBoundary:
    """<StudioHome>\\woa-mirror is a caller's own wheel source, not ours."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "value, expected, why",
        [
            ("{home}/woa-mirror", "{home}/woa-mirror", "a sibling directory survives"),
            ("{home}/woa-custom.txt", "{home}/woa-custom.txt", "and a sibling file"),
            ("{home}/woa", "", "the prefix itself is ours"),
            ("{home}/woa/wheels", "", "and anything under it"),
            (
                "{home}/woa/wheels,{home}/woa-mirror",
                "{home}/woa-mirror",
                "ours goes, the sibling stays",
            ),
        ],
    )
    def test_only_the_prefix_or_its_descendants_are_owned(self, value, expected, why):
        home = "/home/u/AppData/Local/unsloth"
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('$_woaOwnedPrefix = Join-Path $StudioHome "woa"')
        end = text.index("if ($script:WoaNativeCudaTorch) {", start)
        script = "\n".join(
            [
                f"$StudioHome = '{home}'",
                "function Get-UvSafePath { param([string]$p) return $p }",
                f"$env:UV_FIND_LINKS = '{value.format(home = home)}'",
                text[start:end],
                "Write-Output ('[' + $env:UV_FIND_LINKS + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        want = f"[{expected.format(home = home)}]"
        assert done.stdout.strip().splitlines()[-1] == want, why


class TestAHostedDropCandidateMustMeetItsFloor:
    """The override drop list recorded names only.

    Without the override the RELEASED unsloth metadata applies, so a hosted
    xformers-0.0.22 against `xformers>=0.0.22.post7` clears the drop and then fails the
    resolve on a win_arm64 build that does not exist.
    """

    def test_versions_are_recorded_and_the_floor_is_consulted(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$WoaWheelNames[$_woaWheelKey] += $parts[1]" in text, "versions are kept"
        assert '$WoaDropFloors = @{ "xformers" = "0.0.22.post7" }' in text
        assert "Test-WoaVersionAtLeast -Version $_woaHostedVer -Floor $_woaFloor" in text

    def test_the_floor_still_matches_the_metadata(self):
        """The one duplicated constant, pinned to its source so it cannot drift."""
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
        assert (
            "xformers>=0.0.22.post7 ; (sys_platform == 'win32')" in pyproject
        ), "if this floor moves, $WoaDropFloors in install.ps1 moves with it"

    @requires_pwsh
    @pytest.mark.parametrize(
        "have, floor, expected, why",
        [
            ("0.0.22", "0.0.22.post7", "False", "the regression: post7 outranks the release"),
            ("0.0.22.post7", "0.0.22.post7", "True", "the floor itself"),
            ("0.0.22.post8", "0.0.22.post7", "True", "above it"),
            ("0.0.23", "0.0.22.post7", "True", "a later release"),
            ("0.0.30", "0.0.22.post7", "True", "compared numerically, not as text"),
            ("0.0.21", "0.0.22.post7", "False", "an earlier release"),
            ("0.0.23+cu134", "0.0.22.post7", "True", "a local tag is not part of the order"),
            ("garbage", "0.0.22.post7", "False", "unreadable keeps the drop"),
        ],
    )
    def test_the_comparison(self, have, floor, expected, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Test-WoaVersionAtLeast"),
                f"Write-Output (Test-WoaVersionAtLeast -Version '{have}' -Floor '{floor}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, why

    @requires_pwsh
    @pytest.mark.parametrize(
        "hosted, dropped, why",
        [
            ("'0.0.22'", True, "the regression: below the released floor, the drop stays"),
            ("'0.0.22.post7'", False, "at the floor, the wheelhouse wheel is usable"),
            ("'0.0.23'", False, "above it"),
            ("'0.0.22','0.0.23'", False, "one satisfying version among several is enough"),
            ("", True, "nothing hosted at all"),
        ],
    )
    def test_the_loop_keeps_the_drop(self, hosted, dropped, why):
        """Executed, not just read: the version has to reach the decision."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("        foreach ($candidate in $WoaDropCandidates) {")
        end = text.index('$WoaOverrideLines += "$candidate ; platform_machine', start)
        end = text.index("}", text.index("\n", end)) + 1
        wheel_names = "@{}" if not hosted else "@{ 'xformers' = @(%s) }" % hosted
        script = "\n".join(
            [
                _function_source(text, "Test-WoaVersionAtLeast"),
                "function substep { param($m, $c) }",
                "$WoaDropCandidates = @('xformers')",
                '$WoaDropFloors = @{ "xformers" = "0.0.22.post7" }',
                f"$WoaWheelNames = {wheel_names}",
                "$WoaOverrideLines = @()",
                "$WoaReported = @{}",
                text[start:end],
                "Write-Output ('[' + ($WoaOverrideLines -join '|') + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        line = done.stdout.strip().splitlines()[-1]
        assert ("xformers" in line) is dropped, f"{why}: {line}"


class TestAFreeThreadedInterpreterIsPreflightedForAv:
    """torch and pyarrow were not the whole story.

    constraints.txt requires av>=17.1.0 on this platform, and PyAV's win_arm64 wheel is
    cp311-abi3 -- which covers every GIL build from 3.11 up and no free-threaded one,
    since those do not implement the stable ABI. PyAV does publish cp314-cp314t, so the
    answer differs by minor and is asked rather than assumed.
    """

    def test_the_probe_gates_native_mode(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$pyarrowSource = Get-WoaPyarrowSource") :][:2200]
        assert '$FreeThreaded -and -not (Test-WoaWheelAvailable -Project "av"' in block, (
            "asked only on a free-threaded build; a GIL one takes the abi3 wheel and "
            "must not gain a network call or a new way to fail"
        )
        assert block.index("Test-WoaWheelAvailable") < block.index(
            "$script:WoaNativeCudaTorch = $true"
        ), "before the commit, not after: the point is to keep the x64 fallback"

    def test_the_constraint_that_makes_this_matter_is_still_there(self):
        constraints = (
            PACKAGE_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
        ).read_text(encoding = "utf-8")
        assert 'av>=17.1.0; sys_platform == "win32" and platform_machine == "ARM64"' in constraints

    @requires_pwsh
    @pytest.mark.parametrize(
        "listing, abi, expected, why",
        [
            (
                "av-17.1.0-cp311-abi3-win_arm64.whl",
                "cp313t",
                "False",
                "the regression: abi3 is not installable on a free-threaded build",
            ),
            (
                "av-17.1.0-cp314-cp314t-win_arm64.whl",
                "cp314t",
                "True",
                "PyAV does publish a free-threaded wheel, for 3.14t",
            ),
            ("av-17.1.0-cp314-cp314t-win_arm64.whl", "cp313t", "False", "but not for 3.13t"),
            ("", "cp313t", "False", "nothing published at all"),
        ],
    )
    def test_the_probe(self, listing, abi, expected, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        minor = "3.14" if "314" in abi else "3.13"
        body = f'<a href="{listing}">a</a>' if listing else "<html></html>"
        script = "\n".join(
            [
                f"function Invoke-RestMethod {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) return @'\n{body}\n'@ }}",
                "$script:WoaWheelhouse = $null",
                _function_source(text, "Test-WoaWheelTags"),
                _function_source(text, "Test-WoaWheelAvailable"),
                f"Write-Output (Test-WoaWheelAvailable -Project 'av' -PythonMinor '{minor}' -AbiTag '{abi}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, why

    @requires_pwsh
    def test_an_unreachable_index_answers_no(self):
        """The x64 stack still works; a native venv that cannot build PyAV does not."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "function Invoke-RestMethod { param([Parameter(ValueFromRemainingArguments=$true)]$a) throw 'offline' }",
                "$script:WoaWheelhouse = $null",
                _function_source(text, "Test-WoaWheelTags"),
                _function_source(text, "Test-WoaWheelAvailable"),
                "Write-Output (Test-WoaWheelAvailable -Project 'av' -PythonMinor '3.13' -AbiTag 'cp313t')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "False"


class TestACallerOverrideFileKeepsItsOwnDirectory:
    """uv resolves a nested -r, and a relative wheel path, against the file that contains
    the line. Copying such a line into $StudioHome\\woa moved its base directory and every
    later native resolve died on a missing file.
    """

    def test_a_non_conflicting_file_is_passed_through(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$_woaKeepFiles = @()") :][:2600]
        assert "$_woaKeepFiles += $_woaOvFull" in block, "kept where it is"
        assert "$_woaOvConflicts" in block, "only a package clash forces a rewrite"
        assert (
            "Resolve-WoaOverrideLine -Line $_woaOvLine -BaseDir $_woaOvDir" in block
        ), "and a folded line has its relative references made absolute"

    def test_the_kept_files_reach_uv(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "foreach ($_woaKeepFile in $_woaKeepFiles) { $_woaOverrideValue += (Get-UvSafePath $_woaKeepFile) }"
            in text
        )
        assert (
            '$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")' in text
        ), "uv splits UV_OVERRIDE on whitespace and combines the files"

    @requires_pwsh
    @pytest.mark.parametrize(
        "line, expected, why",
        [
            ("-r nested.txt", "-r /opt/corp/ov/nested.txt", "a nested include"),
            ("--requirement=sub/n.txt", "--requirement=/opt/corp/ov/sub/n.txt", "long form"),
            ("-c cons.txt", "-c /opt/corp/ov/cons.txt", "a constraint file"),
            ("-f wheels", "-f /opt/corp/ov/wheels", "a find-links directory"),
            ("-r /etc/n.txt", "-r /etc/n.txt", "an absolute path is already right"),
            ("-r https://x.test/n.txt", "-r https://x.test/n.txt", "so is a URL"),
            ("brotli==1.1.0", "brotli==1.1.0", "an ordinary requirement is untouched"),
            (
                'foo ; platform_machine == "AMD64"',
                'foo ; platform_machine == "AMD64"',
                "and so is a marker",
            ),
            ("foo @ https://x.test/a.whl", "foo @ https://x.test/a.whl", "a direct URL"),
            ("foo @ file:dist/a.whl", "foo @ file:/opt/corp/ov/dist/a.whl", "a relative file: URL"),
            ("dist/a.whl", "/opt/corp/ov/dist/a.whl", "a bare relative wheel path"),
            ("a.whl", "a.whl", "a bare name with no directory is a requirement, not a path"),
            ("# note", "# note", "a comment"),
        ],
    )
    def test_the_rewriter(self, line, expected, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Resolve-WoaOverrideLine"),
                f"Write-Output ('[' + (Resolve-WoaOverrideLine -Line '{line}' -BaseDir '/opt/corp/ov') + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == f"[{expected}]", why

    @requires_pwsh
    @pytest.mark.parametrize(
        "caller_lines, folded, why",
        [
            (
                ["-r nested.txt"],
                False,
                "the regression: no package clash, so the file stays where it was written",
            ),
            (["brotli==1.1.0"], False, "an unrelated package is still no clash"),
            (
                ["torch==2.9.0", "-r nested.txt"],
                True,
                "torch is one of ours, so this file has to be folded, rebased as it goes",
            ),
        ],
    )
    def test_the_block_end_to_end(self, tmp_path, caller_lines, folded, why):
        """Executed: a source-level assertion cannot tell a live branch from a dead one."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("        $_woaOwnNames = @{}")
        end = text.index('$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")', start)
        end = text.index("\n", end)
        caller_dir = tmp_path / "corp"
        caller_dir.mkdir()
        (caller_dir / "nested.txt").write_text("idna==3.10\n", encoding = "utf-8")
        caller = caller_dir / "ov.txt"
        caller.write_text("\n".join(caller_lines) + "\n", encoding = "utf-8")
        managed = tmp_path / "woa.txt"
        script = "\n".join(
            [
                _function_source(text, "Resolve-WoaOverrideLine"),
                "function Get-UvSafePath { param([string]$p) return $p }",
                "$WoaOverrideLines = @('# generated', 'torch>=2.4', 'torchvision>=0.19')",
                f"$WoaOverrides = '{managed}'",
                f"$env:UV_OVERRIDE = '{caller}'",
                text[start:end],
                'Write-Output ("OVERRIDE=" + $env:UV_OVERRIDE)',
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        value = [line for line in done.stdout.splitlines() if line.startswith("OVERRIDE=")][-1][
            len("OVERRIDE=") :
        ].split()
        written = managed.read_text(encoding = "utf-8")
        if folded:
            assert value == [str(managed)], why
            assert (
                str(caller_dir / "nested.txt") in written
            ), "the include is rebased onto the directory it was written for"
            assert "torch==2.9.0" not in written, "our own declaration still wins"
        else:
            assert value == [str(managed), str(caller)], why
            assert "-r nested.txt" not in written, "nothing was copied, so nothing moved"


class TestTheRecoveryPrependsRatherThanStandsDown:
    """A caller's own find-links must not cost the staged win_arm64 wheels.

    find-links are additional search locations with no conflict semantics, so ours and a
    corporate mirror coexist. Skipping ours because the caller had set something left
    pyarrow -- which exists nowhere but that directory on this platform -- out of the
    search, and the update reached for an sdist that cannot build here.
    """

    @staticmethod
    def _block() -> str:
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("    if (Test-Path -LiteralPath $wheels -PathType Container) {")
        return text[start : text.index("\n}", start)]

    def test_it_prepends(self):
        block = self._block()
        assert (
            '$env:UV_FIND_LINKS = "$wheels,$($env:UV_FIND_LINKS)"' in block
        ), "UV_FIND_LINKS is comma-separated, and ours goes first"
        assert (
            '$env:PIP_FIND_LINKS = "$_woaSafeWheels $($env:PIP_FIND_LINKS)"' in block
        ), "PIP_FIND_LINKS is split on whitespace, and ours must be the 8.3-safe form"
        assert "-notcontains" in block, "a second run must not keep prepending"

    @requires_pwsh
    @pytest.mark.parametrize(
        "var, before, expected, why",
        [
            ("UV_FIND_LINKS", "", "/home/u/woa/wheels", "nothing set: ours alone"),
            (
                "UV_FIND_LINKS",
                "/mnt/mirror",
                "/home/u/woa/wheels,/mnt/mirror",
                "the regression: a caller mirror no longer suppresses ours",
            ),
            (
                "UV_FIND_LINKS",
                "/home/u/woa/wheels,/mnt/mirror",
                "/home/u/woa/wheels,/mnt/mirror",
                "already first: unchanged, not doubled",
            ),
            (
                "PIP_FIND_LINKS",
                "/mnt/mirror",
                "/home/u/woa/wheels /mnt/mirror",
                "whitespace for pip",
            ),
            (
                "PIP_FIND_LINKS",
                "/home/u/woa/wheels",
                "/home/u/woa/wheels",
                "already present: unchanged",
            ),
        ],
    )
    def test_the_block(self, var, before, expected, why):
        script = "\n".join(
            [
                "function Get-UvSafePath { param([string]$p) return $p }",
                "$wheels = '/home/u/woa/wheels'",
                "Remove-Item Env:UV_FIND_LINKS,Env:PIP_FIND_LINKS -ErrorAction SilentlyContinue",
                (f"$env:{var} = '{before}'" if before else ""),
                self._block().replace(
                    "if (Test-Path -LiteralPath $wheels -PathType Container) {",
                    "if ($true) {",
                    1,
                ),
                f"Write-Output ('[' + $env:{var} + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == f"[{expected}]", why


class TestTheMergedOverrideFileIsRebasedToo:
    """install.ps1 rebases a folded line; setup.ps1's merge had the same problem."""

    def test_the_merge_rebases(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$_woaMerged = Join-Path $woaDir") :][:1600]
        assert "Resolve-WoaOverrideLine -Line $_woaLine -BaseDir" in block
        assert "$_woaLines += $_woaLine" not in block, "no line is copied verbatim"

    def test_the_helper_is_a_faithful_copy_of_install_ps1s(self):
        """Neither script can dot-source the other, so the copy is pinned instead."""

        def normalized(source: str) -> str:
            lines = [
                line.rstrip()
                for line in source.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            indent = min(len(line) - len(line.lstrip()) for line in lines)
            return "\n".join(line[indent:] for line in lines)

        install = _function_source(
            (PACKAGE_ROOT / "install.ps1").read_text(encoding = "utf-8"),
            "Resolve-WoaOverrideLine",
        )
        setup = _function_source(
            SETUP_PS1.read_text(encoding = "utf-8"),
            "Resolve-WoaOverrideLine",
        )
        assert normalized(install) == normalized(setup)


class TestAWheelhouseThatIsTheStagingDirectory:
    """UNSLOTH_WOA_WHEELHOUSE may BE $StudioHome\\woa\\wheels -- that is how an offline
    run reuses the installer's own cache. Copy-Item refuses to overwrite an item with
    itself, and under $ErrorActionPreference = "Stop" that aborted the whole install.
    """

    def test_both_staging_copies_are_guarded(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "if (-not (Test-WoaSamePath $found.FullName $_woaDest)) {" in text
        ), "the pyarrow copy, which is not inside a try and so was fatal"
        assert (
            "if (-not (Test-WoaSamePath $wheel.FullName $_woaExtraDest)) {" in text
        ), "and the extra-wheel loop, which swallowed the error but miscounted"

    @requires_pwsh
    @pytest.mark.parametrize(
        "a, b, expected, why",
        [
            ("/x/woa/wheels/a.whl", "/x/woa/wheels/a.whl", "True", "the same file"),
            (
                "/x/woa/wheels/a.whl",
                "/x/woa/wheels/../wheels/a.whl",
                "True",
                "the same file spelled differently",
            ),
            (
                "/x/woa/wheels/a.whl",
                "/X/WOA/WHEELS/A.WHL",
                "True",
                "Windows paths are case-insensitive, and this only runs there",
            ),
            ("/x/mirror/a.whl", "/x/woa/wheels/a.whl", "False", "different files"),
            ("", "/x/woa/wheels/a.whl", "False", "nothing is not a path"),
        ],
    )
    def test_the_comparison(self, a, b, expected, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Test-WoaSamePath"),
                f"Write-Output (Test-WoaSamePath '{a}' '{b}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, why

    @requires_pwsh
    def test_a_self_copy_would_otherwise_be_fatal(self, tmp_path):
        """The behaviour this guards, executed, so the reason cannot go stale."""
        wheel = tmp_path / "a.whl"
        wheel.write_text("x", encoding = "utf-8")
        done = subprocess.run(
            [
                PWSH,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                '$ErrorActionPreference = "Stop"; '
                f"try {{ Copy-Item -LiteralPath '{wheel}' -Destination '{wheel}' -Force; "
                "Write-Output 'OK' } catch { Write-Output 'THREW' }",
            ],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "THREW"


class TestTheSuppliedWheelIsOpenedNotSniffed:
    """A truncated download still starts with "PK".

    Accepting one selected the native path, staged the broken file, and failed the pyarrow
    resolve with the working x64 stack already given up.
    """

    def test_the_zip_helper_is_used(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("if ($env:UNSLOTH_PYARROW_WHEEL) {") :][:1800]
        assert 'if (Test-ZipArchiveReadable -Path $_paWheel) { return "local" }' in block
        assert "ReadByte()" not in block, "the two-byte signature sniff is gone"

    def test_the_helper_is_defined_before_this_runs(self):
        """PowerShell does not hoist: a call above the definition is a runtime error."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert text.index("function Test-ZipArchiveReadable") < text.index(
            "function Get-WoaPyarrowSource"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "payload, expected, why",
        [
            (
                b"PK\x03\x04" + b"\x00" * 64,
                "False",
                "the regression: a PK header with no central directory",
            ),
            (b"not a zip at all", "False", "nothing zip-like"),
            (b"", "False", "an empty file"),
            (None, "True", "a real archive"),
        ],
    )
    def test_the_check(self, tmp_path, payload, expected, why):
        import zipfile

        wheel = tmp_path / "pyarrow-21.0.0-cp313-cp313-win_arm64.whl"
        if payload is None:
            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr("pyarrow/__init__.py", "")
        else:
            wheel.write_bytes(payload)
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Test-ZipArchiveReadable"),
                f"Write-Output (Test-ZipArchiveReadable -Path '{wheel}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, why


class TestAConfiguredWoaMirrorSurvivesAFreshShell:
    """write_manifest persists only NVIDIA's own channels, because any other URL could
    carry a credential. So a corporate win_arm64 mirror is recoverable nowhere but the
    variable the user set, and ignoring it sent a fresh-shell update to the
    driver-derived download.pytorch.org family, which has no win_arm64 CUDA wheel at all.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "configured, handover, persisted, expected, why",
        [
            (
                "https://mirror.corp/woa",
                "",
                "",
                "https://mirror.corp/woa",
                "the regression: a fresh shell with only the user's own variable",
            ),
            (
                "https://mirror.corp/woa/",
                "",
                "",
                "https://mirror.corp/woa",
                "trailing slash trimmed, as the other branches do",
            ),
            (
                "https://mirror.corp/woa",
                "https://pypi.nvidia.com/oot",
                "",
                "https://mirror.corp/woa",
                "the user's channel outranks the handover",
            ),
            (
                "",
                "https://pypi.nvidia.com/oot",
                "",
                "https://pypi.nvidia.com/oot",
                "unchanged when it is not set",
            ),
            (
                "",
                "",
                "https://pypi.nvidia.com/oot",
                "https://pypi.nvidia.com/oot",
                "and the manifest still answers when nothing else does",
            ),
        ],
    )
    def test_the_precedence(self, configured, handover, persisted, expected, why):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64TorchIndexUrl = if ($WinArm64Venv")
        end = text.index('} else { "" }', start) + len('} else { "" }')
        script = "\n".join(
            [
                "$WinArm64Venv = $true",
                "$VenvDir = '/nonexistent'",
                f"function Get-PersistedWoaTorchIndex {{ param($VenvPath) return '{persisted}' }}",
                f"$env:UNSLOTH_WOA_TORCH_INDEX_URL = '{configured}'",
                f"$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = '{handover}'",
                text[start:end],
                "Write-Output ('[' + $WinArm64TorchIndexUrl + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == f"[{expected}]", why

    @requires_pwsh
    def test_a_non_arm64_venv_still_reads_nothing(self):
        """Every other host must see exactly the index choice it saw before."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64TorchIndexUrl = if ($WinArm64Venv")
        end = text.index('} else { "" }', start) + len('} else { "" }')
        script = "\n".join(
            [
                "$WinArm64Venv = $false",
                "$VenvDir = '/nonexistent'",
                "function Get-PersistedWoaTorchIndex { param($VenvPath) throw 'must not be called' }",
                "$env:UNSLOTH_WOA_TORCH_INDEX_URL = 'https://mirror.corp/woa'",
                "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = 'https://pypi.nvidia.com/oot'",
                text[start:end],
                "Write-Output ('[' + $WinArm64TorchIndexUrl + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "[]"

    def test_install_ps1_still_does_not_write_that_variable(self):
        """It is the user's INPUT. Writing it would make a second run read its own answer
        back as an instruction, which is what the handover variable is for."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert not re.search(r"\$env:UNSLOTH_WOA_TORCH_INDEX_URL\s*=", text)

    def test_a_mirror_is_still_not_persisted(self):
        """The reason this branch has to exist; if the manifest ever took one, the
        credential rule would have been weakened instead."""
        module = _load_manifest_module()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            written = module.write_manifest(
                pathlib.Path(tmp),
                woa_torch_index = "https://mirror.corp/woa",
            )
            payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload
