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


def _persistence_block(text: str) -> str:
    """The live `if (...) { ... }` that writes both index records.

    Sliced out of setup.ps1 rather than restated here: a copy of the block passes forever
    after the original stops matching it, which is exactly the failure these tests exist
    to catch.
    """
    start = text.index("$_woaPinnedIndex = if ($WinArm64Venv)")
    guard = text.index("if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {", start)
    depth = 0
    for index in range(text.index("{", guard), len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError("unbalanced braces in the persistence block")


class TestReadSide:
    """studio/setup.ps1: what Get-PersistedWoaTorchIndex hands back to the resolver."""

    @requires_pwsh
    @pytest.mark.parametrize("url, allowed, why", [c for c in CANDIDATES if c[0]])
    def test_a_hand_edited_manifest_cannot_redirect_the_install(
        self, tmp_path: pathlib.Path, url: str, allowed: bool, why: str
    ):
        """The write guard is not enough on its own: the file sits in the user's venv and
        anything can put a line in it.
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
        """The caller's own override file is never dropped."""
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
        """The three are restored independently."""
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
        """Get-UvSafePath exists in both scripts because neither can dot-source the other."""

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
        """If studio.txt ever drops ddgs, this restore stops being load-bearing for brotli."""
        studio_txt = PACKAGE_ROOT / "studio" / "backend" / "requirements" / "studio.txt"
        assert "ddgs" in studio_txt.read_text(encoding = "utf-8")

    def test_the_restore_runs_before_the_dependency_pass(self):
        """After it, the brotli resolve has already been attempted."""
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        restore = setup.index("\nRestore-WoaResolverEnvironment")
        stack = setup.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert restore < stack


class TestTheRecoveryReachesEveryModeThatNeedsIt:
    """Placement, which is what decided whether the two recoveries above fire at all."""

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
            "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex",
        )
        assert not any(
            "NoTorchMode" in b for b in blocks
        ), f"the manifest is rewritten in no-torch mode too. Enclosing blocks: {blocks}"

    def test_the_recovered_index_is_put_back_in_the_environment(self):
        """The bug this guards: recovering the index into a local variable only."""
        text = self._setup()
        assign = text.index("$WinArm64TorchIndexUrl = if (")
        export = text.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        stack = text.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert assign < export < stack, "recovered, re-exported, then read by the stack"
        block = text.rindex("if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {", 0, export)
        assert (
            "$_woaMarkerIndex = $_woaPinnedIndex" in text[block:export]
        ), "guarded: neither record present must not export an empty value"

    def test_studio_txt_is_installed_in_no_torch_mode(self):
        """The premise of the placement test above."""
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
    """Ordering against the manifest drop, which decides whether any of this works."""

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
    """What install_python_stack.py is told to repair from."""

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
        """Get-TorchIndexLeaf on the NVIDIA channel yields `nvtorch_oot`, which is not a
        CUDA family name, so the tag resolves to $null and nothing is published.
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
        """Every `if ...has_usable_nvidia...` whose enclosing branches select ARM64."""
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
        """Three entry points reach ARM64 CUDA independently: direct_upstream_release_plan,
        resolve_upstream_asset_choice (via resolve_asset_choice's fallbacks), and
        resolve_asset_choice's own published-artifact branch.
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
        """A cliff rather than a bug visible today: the published windows-arm64-cuda branch
        returns before the unverified-upstream tail, so gating only the tail meant
        UNSLOTH_LLAMA_ARM64_CUDA=0 would keep working right up until the fork published
        an approved artifact, then silently stop.
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
        """With the branch gated, a second test inside it could only ever be true."""
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
    """install.ps1: an upgrade must not leave a WoA NVIDIA host on the emulated stack."""

    INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"

    @staticmethod
    def _text() -> str:
        return TestAMigratedX64VenvIsRebuiltAsArm64.INSTALL_PS1.read_text(encoding = "utf-8")

    def test_the_rebuild_runs_before_venv_creation(self):
        """It works by making $VenvPython absent, so the existing creation block builds an
        ARM64 venv.
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
        """The regression that would otherwise follow: $_Migrated drives an upgrade-in-place
        far below, which installs unsloth with --no-deps and --reinstall-package.
        """
        text = self._text()
        block = text[text.index("$script:WoaNativeCudaTorch -and $_Migrated") :][:1800]
        rollback = block.index("Start-StudioVenvRollback")
        cleared = block.index("$_Migrated = $false")
        assert rollback < cleared, "cleared only after the environment is safely moved"
        # And the flag really does still gate that path.
        assert "if ($_Migrated) {" in text

    def test_it_only_touches_a_venv_this_run_migrated(self):
        """A new-layout venv was already moved aside above, and a venv created moments ago
        came from the interpreter this run chose.
        """
        text = self._text()
        line = text[text.index("if ($script:WoaNativeCudaTorch -and $_Migrated") :].split("\n")[0]
        assert "$_Migrated" in line
        assert "Test-Path -LiteralPath $VenvPython" in line

    def test_the_platform_guard_below_still_has_the_final_say(self):
        """Belt and braces: if the rollback failed, the venv is still x64 and the existing
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
    """setup.ps1's mismatch check must expect the bundle the selector actually installs."""

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
        """A CUDA bundle installed before the flag was set must still be replaced by the one
        the flag asks for, so the opt-out arm expects the CPU kind INSTEAD of the CUDA
        kind.
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
        """The premise of widening: resolve_asset_choice falls through to the published
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
        """The installer's already-satisfied short-circuit is per candidate, and CUDA is
        attempted first, so a CPU bundle accepted here is still replaced the day an ARM64
        CUDA asset appears.
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
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
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
        """PyPI's own win_arm64 torch wheels carry no local version at all."""
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
    """The GA channel publishes torch 2.14.0+cu134 beside torchaudio 2.11.0+cu134."""

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

    The URL spelling is now the SECOND signal: install.ps1 reads the answer off the wheel it
    probed and hands it over, because a mirror of a prerelease-only channel need not have
    "nightly" anywhere in its address.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "index, handover, expect_pre",
        [
            ("https://pypi.nvidia.com/nvtorch_oot", "0", False),
            ("https://pypi.nvidia.com/nvtorch_oot_nightly", "0", True),
            ("", "0", False),
            ("https://mirror.test/simple", "1", True),
            ("https://mirror.test/simple", "0", False),
        ],
    )
    def test_the_flag_follows_the_channel(self, index: str, handover: str, expect_pre: bool):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64IndexArgs = if (")
        end = text.index("} else { @() }", start) + len("} else { @() }")
        script = "\n".join(
            [
                "$WinArm64Venv = $true",
                "$UseUv = $true",
                f"$WinArm64TorchIndexUrl = '{index}'",
                f"$WinArm64EffectiveTorchIndexUrl = '{index}'",
                f"$WinArm64HandoffApplies = ${bool(index)}",
                f"$env:UNSLOTH_WOA_TORCH_PRERELEASE = '{handover}'",
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
                "$WinArm64EffectiveTorchIndexUrl = $WinArm64TorchIndexUrl",
                "$WinArm64HandoffApplies = $false",
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
    """A value the writer persists and the reader refuses is worse than none at all."""

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
    """install.ps1: UNSLOTH_PYARROW_WHEEL decides whether the native path is taken."""

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
                # Test-WoaPyarrowWheelUsable gates on Test-WoaWheelTagsUsable now, and a
                # helper the prelude does not lift is a command-not-found that aborts the
                # statement rather than answering false.
                _function_source(text, "Test-WoaWheelTagsUsable"),
                _function_source(text, "Test-WoaVersionAtLeast"),
                # Every pyarrow candidate is floored against constraints.txt now.
                '$script:WoaPyarrowFloor = "21.0.0"',
                _function_source(text, "Test-WoaPyarrowWheelUsable"),
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
    """install.ps1 must not discard what its own purge block just chose to keep."""

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

    def test_the_python_side_can_read_an_appended_value(self, tmp_path):
        """install_python_stack.py split find-links on os.pathsep alone, which would have
        read "dirA,dirB" as one unusable path now that appending is possible.

        Asserted as behaviour rather than as a regex, because the separator is now
        per-variable: a shared class that also broke on whitespace tore a directory whose
        name contains a space into two paths that do not exist.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ips_findlinks_split", STACK_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        first = tmp_path / "a"
        second = tmp_path / "private wheels"
        for directory in (first, second):
            directory.mkdir()
        (first / "alpha-1.0.0-py3-none-any.whl").write_bytes(b"")
        (second / "beta-2.0.0-py3-none-any.whl").write_bytes(b"")

        os.environ["UV_FIND_LINKS"] = f"{first},{second}"
        os.environ.pop("PIP_FIND_LINKS", None)
        try:
            module._find_links_wheel_versions.cache_clear()
            found = module._find_links_wheel_versions()
        finally:
            os.environ.pop("UV_FIND_LINKS", None)
            module._find_links_wheel_versions.cache_clear()
        assert "alpha" in found, "an appended comma-separated entry is still read"
        assert "beta" in found, (
            "a UV_FIND_LINKS directory whose name contains a space was split into "
            "fragments, so every wheel an air-gapped user hosted there went unseen"
        )


class TestTheProbeAsksForTheInterpretersAbi:
    """install.ps1 keyed its wheel search on the minor alone."""

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
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
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
        """The other half: keyed on the python tag, staging kept the cp313-cp313 wheels a
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
    """The hole left by keying the re-probe on the minor alone."""

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
    """The recovery is a memory of what install.ps1 chose, not a decision."""

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
    """A caller override must not cost the win_arm64 drop list."""

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
    """The purge drops a PREVIOUS run's resolver settings, not the caller's."""

    @staticmethod
    def _purge_block() -> str:
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('$_woaOwnedPrefix = Join-Path $StudioHome "woa"')
        end = text.index("if ($script:WoaNativeCudaTorch) {", start)
        return text[start:end]

    def test_the_block_works_entry_by_entry(self):
        block = self._purge_block()
        assert "$_woaKept" in block, "entries are kept, not just the whole value dropped"
        assert '"UV_FIND_LINKS" = ","' in block, "each is rejoined with its own separator"
        assert "$_woaSplitOn" in block, (
            "and SPLIT with that same one: uv takes UV_FIND_LINKS comma-separated, so a "
            "shared whitespace split tore a path with a space into fragments and then "
            "rejoined them with commas"
        )

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
    """write_manifest documents that it never raises."""

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
    installs torch/torchvision alone.
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
    """`*cp313-cp313*` also matches cp313-cp313t."""

    def test_no_substring_match_is_left(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$tag-$AbiTag*" not in text, "every probe goes through the field parser"
        # The six pyarrow sites now go through Test-WoaPyarrowWheelUsable, which is a floor
        # check wrapped around this one, so they still reach the field parser.
        assert (text.count("Test-WoaWheelTags") + text.count("Test-WoaPyarrowWheelUsable")) >= 7, (
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
                # Test-WoaPyarrowWheelUsable gates on Test-WoaWheelTagsUsable now, and a
                # helper the prelude does not lift is a command-not-found that aborts the
                # statement rather than answering false.
                _function_source(text, "Test-WoaWheelTagsUsable"),
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
    """The override drop list recorded names only."""

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
            # PEP 440 hangs .devN off whatever precedes it, so a development release sorts
            # BELOW that segment. Reading .dev as a pre-release of the RELEASE got the post
            # case backwards: the wheel cleared the floor, the drop override was omitted, and
            # the released requirement then rejected it with nothing left to install.
            ("0.0.22.post7.dev0", "0.0.22.post7", "False", "a dev of post7 is below post7"),
            ("0.0.22.post8.dev0", "0.0.22.post7", "True", "but still above post6 and post7"),
            ("0.0.22.post7", "0.0.22.post7.dev0", "True", "and the release outranks its dev"),
            ("0.0.23.dev0", "0.0.22.post7", "True", "a later release line wins outright"),
            ("0.0.22.dev0", "0.0.22.post7", "False", "a dev of the release is below post7"),
            ("0.0.22.post7.dev1", "0.0.22.post7.dev0", "True", "dev stamps still order"),
            ("0.0.22.post7.dev0", "0.0.22.post7.dev1", "False", "and order in both directions"),
            ("0.0.23rc1", "0.0.23", "False", "rc still sorts below its release"),
            ("0.0.23", "0.0.23rc1", "True", "and the release above the rc"),
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
    """torch and pyarrow were not the whole story."""

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
                # Test-WoaPyarrowWheelUsable gates on Test-WoaWheelTagsUsable now, and a
                # helper the prelude does not lift is a command-not-found that aborts the
                # statement rather than answering false.
                _function_source(text, "Test-WoaWheelTagsUsable"),
                _function_source(text, "Test-WoaWheelTagsUsable"),
                _function_source(text, "Test-WoaVersionAtLeast"),
                _function_source(text, "Test-WoaPyPIWheel"),
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
                # Test-WoaPyarrowWheelUsable gates on Test-WoaWheelTagsUsable now, and a
                # helper the prelude does not lift is a command-not-found that aborts the
                # statement rather than answering false.
                _function_source(text, "Test-WoaWheelTagsUsable"),
                _function_source(text, "Test-WoaWheelTagsUsable"),
                _function_source(text, "Test-WoaVersionAtLeast"),
                _function_source(text, "Test-WoaPyPIWheel"),
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
    the line.
    """

    def test_a_non_conflicting_file_is_passed_through(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        block = text[text.index("$_woaKeepFiles = @()") :][:2600]
        assert "$_woaKeepFiles += $_woaOvFull" in block, "kept where it is"
        assert "$_woaOvConflicts" in block, "only a package clash forces a rewrite"
        assert (
            "Resolve-WoaOverrideLine -Line $_woaOvEntry.Line -BaseDir $_woaOvEntry.BaseDir" in block
        ), "and a folded line has its relative references made absolute, against the file it came from"

    def test_the_kept_files_reach_uv(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "foreach ($_woaKeepFile in $_woaKeepFiles) { $_woaOverrideValue += (Get-UvSafePath $_woaKeepFile) }"
            in text
        )
        assert (
            '$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")' in text
        ), "uv splits UV_OVERRIDE on whitespace and combines the files"

    # The rewriter calls [System.IO.Path]::GetFullPath, so on Windows "/opt/corp/ov" comes
    # back as "C:\\opt\\corp\\ov" with backslashes. os.path.abspath applies the same rule
    # independently, which keeps the expected value right on every host instead of only on
    # the POSIX one these were first written on.
    BASE = "/opt/corp/ov"

    @staticmethod
    def _rebased(relative: str) -> str:
        return os.path.abspath(
            os.path.join(TestACallerOverrideFileKeepsItsOwnDirectory.BASE, relative)
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "line, prefix, rebased, why",
        [
            ("-r nested.txt", "-r ", "nested.txt", "a nested include"),
            ("--requirement=sub/n.txt", "--requirement=", "sub/n.txt", "long form"),
            ("-c cons.txt", "-c ", "cons.txt", "a constraint file"),
            ("-f wheels", "-f ", "wheels", "a find-links directory"),
            ("foo @ file:dist/a.whl", "foo @ file:", "dist/a.whl", "a relative file: URL"),
            ("dist/a.whl", "", "dist/a.whl", "a bare relative wheel path"),
        ],
    )
    def test_a_relative_reference_is_rebased(self, line, prefix, rebased, why):
        assert self._run(line) == prefix + self._rebased(rebased), why

    @requires_pwsh
    @pytest.mark.parametrize(
        "line, why",
        [
            ("-r /etc/n.txt", "an absolute path is already right"),
            ("-r https://x.test/n.txt", "so is a URL"),
            ("brotli==1.1.0", "an ordinary requirement is untouched"),
            ('foo ; platform_machine == "AMD64"', "and so is a marker"),
            ("foo @ https://x.test/a.whl", "a direct URL"),
            ("a.whl", "a bare name with no directory is a requirement, not a path"),
            ("# note", "a comment"),
        ],
    )
    def test_everything_else_is_returned_unchanged(self, line, why):
        assert self._run(line) == line, why

    @staticmethod
    def _run(line: str) -> str:
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _function_source(text, "Resolve-WoaOverrideLine"),
                "Write-Output ('[' + (Resolve-WoaOverrideLine -Line '{}' -BaseDir '{}') + ']')".format(
                    line,
                    TestACallerOverrideFileKeepsItsOwnDirectory.BASE,
                ),
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
                # PowerShell does not hoist, so the scanner the block calls has to be here too.
                _function_source(text, "Get-WoaRequirementEntries"),
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
            # The include is FLATTENED as it folds, not copied as an "-r" line: its contents
            # arrive rebased against the directory the include was written in, which is the
            # only way a conflict discovered one level down can also be removed.
            assert (
                "idna==3.10" in written
            ), "the include's own lines did not come across, so folding dropped them"
            assert "-r " not in written, "an include line copied verbatim would move its base"
            assert "torch==2.9.0" not in written, "our own declaration still wins"
        else:
            assert value == [str(managed), str(caller)], why
            assert "-r nested.txt" not in written, "nothing was copied, so nothing moved"


class TestTheRecoveryPrependsRatherThanStandsDown:
    """A caller's own find-links must not cost the staged win_arm64 wheels."""

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
        assert "Resolve-WoaOverrideLine -Line $_woaEntry.Line -BaseDir $_woaEntry.BaseDir" in block
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
    r"""UNSLOTH_WOA_WHEELHOUSE may BE $StudioHome\woa\wheels -- that is how an offline run
    reuses the installer's own cache.
    """

    def test_every_staging_copy_is_guarded(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "if (-not (Test-WoaSamePath $found.FullName $_woaDest)) {" in text
        ), "the wheelhouse pyarrow copy, which is not inside a try and so was fatal"
        assert (
            "if (-not (Test-WoaSamePath $wheel.FullName $_woaExtraDest)) {" in text
        ), "the extra-wheel loop, which swallowed the error but miscounted"
        assert "if (-not (Test-WoaSamePath $srcWheel $_woaLocalDest)) {" in text, (
            "and the supplied-wheel copy, where the failure was caught but disabled "
            "native mode after the ARM64 venv had already been chosen"
        )

    def test_no_staging_copy_is_left_unguarded(self):
        """Counted, so a fourth copy added later cannot quietly skip the guard."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        # Anchored on code, not on a comment: a comment pass must not be able to break this.
        start = text.index('if ($script:WoaPyarrowSource -eq "local") {')
        staging = text[start : text.index("$WoaOverrides = Join-Path $WoaDir", start)]
        copies = staging.count("Copy-Item -LiteralPath")
        guards = staging.count("Test-WoaSamePath")
        assert copies == guards == 3, f"{copies} copies, {guards} guards"

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
    """A truncated download still starts with "PK"."""

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
    """write_manifest persists only NVIDIA's own channels, because any other URL could carry
    a credential.
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
        """It is the user's INPUT."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert not re.search(r"\$env:UNSLOTH_WOA_TORCH_INDEX_URL\s*=", text)

    def test_a_mirror_is_still_not_persisted(self):
        """The reason this branch has to exist; if the manifest ever took one, the
        credential rule would have been weakened instead.
        """
        module = _load_manifest_module()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            written = module.write_manifest(
                pathlib.Path(tmp),
                woa_torch_index = "https://mirror.corp/woa",
            )
            payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload


class TestAWheelhousePyarrowMustClearTheFloor:
    """A tag-compatible pyarrow is not enough: it also has to satisfy the ARM64 constraint.

    Staging turns whichever wheel it picks into an exact `pyarrow==<version>` override, and
    single-env/constraints.txt floors the ARM64 row at 21.0.0. A 19.x wheel in the wheelhouse
    therefore selected the native path and then made the dependency pass unsatisfiable, after
    the ARM64 venv had already been built.
    """

    def test_the_floor_matches_the_constraint(self):
        """Two places state it, so a bump to one that skips the other is caught here."""
        floor = re.search(
            r'\$script:WoaPyarrowFloor\s*=\s*"([^"]+)"',
            INSTALL_PS1.read_text(encoding = "utf-8"),
        )
        assert floor, "install.ps1 no longer declares the pyarrow floor"
        constraints = (
            PACKAGE_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
        ).read_text(encoding = "utf-8")
        pinned = re.search(
            r'(?m)^pyarrow>=([0-9.]+);\s*sys_platform == "win32" and platform_machine == "ARM64"',
            constraints,
        )
        assert pinned, "the ARM64 pyarrow row is gone from constraints.txt"
        assert floor.group(1) == pinned.group(1), (
            f"install.ps1 floors pyarrow at {floor.group(1)} but constraints.txt requires "
            f">={pinned.group(1)}, so the staged wheel would not satisfy the resolve"
        )

    def test_every_pyarrow_candidate_goes_through_the_floor(self):
        """Counted: a seventh candidate site added later cannot skip the check.

        Four in the preflight (supplied wheel, PyPI, wheelhouse directory, wheelhouse index)
        and two in staging, which picks the file the override is written from.
        """
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert text.count("Test-WoaPyarrowWheelUsable") == 7, (
            "one definition and six call sites; a pyarrow candidate is being accepted on "
            "its tags alone somewhere"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, expected, why",
        [
            ("pyarrow-21.0.0-cp313-cp313-win_arm64.whl", "True", "at the floor"),
            ("pyarrow-23.0.1-cp313-cp313-win_arm64.whl", "True", "above it"),
            # The shape upstream is actually going to ship. apache/arrow#48539 is held behind
            # apache/arrow#50398, whose plan is an abi3 floor of 3.11, so the first win_arm64
            # pyarrow on PyPI will be cp311-abi3. Rejecting it would keep us on our own
            # 24.0.0.dev260 silently and forever.
            ("pyarrow-26.0.0-cp311-abi3-win_arm64.whl", "True", "abi3 from below this venv"),
            ("pyarrow-26.0.0-cp39-abi3-win_arm64.whl", "True", "abi3 from further below"),
            # abi3 reaches forward from what it was built against, never backward.
            ("pyarrow-26.0.0-cp314-abi3-win_arm64.whl", "False", "abi3 built for a newer one"),
            # A floor still applies to an abi3 wheel; the tag is not a bypass.
            ("pyarrow-19.0.0-cp311-abi3-win_arm64.whl", "False", "abi3 below the floor"),
            ("pyarrow-19.0.1-cp313-cp313-win_arm64.whl", "False", "below it"),
            ("pyarrow-21.0.0-cp312-cp312-win_arm64.whl", "False", "wrong interpreter"),
            ("pyarrow-notaversion-cp313-cp313-win_arm64.whl", "False", "unreadable version"),
        ],
    )
    def test_the_floor_is_applied(self, name, expected, why):
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                '$script:WoaPyarrowFloor = "21.0.0"',
                _ps_function(INSTALL_PS1, "Test-WoaPyarrowWheelUsable"),
                f"Write-Output ([bool](Test-WoaPyarrowWheelUsable -Name '{name}' "
                "-PyTag 'cp313' -AbiTag 'cp313'))",
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


class TestThePrereleaseAnswerComesFromTheWheel:
    """Not from the URL: a mirror of a prerelease-only channel need not say "nightly".

    Without --prerelease=allow, uv takes the stable win_arm64 CPU torch from the PyPI extra
    index instead of the 2.15.0.dev CUDA build the probe just proved, so the install
    completes CPU-only on a machine that was bought for its GPU.
    """

    def test_the_installer_reads_the_probed_version(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert re.search(
            r"\$script:WoaTorchIsPrerelease\s*=\s*\[bool\]\(\$_woaTorchVersion\s*-match",
            text,
        ), "install.ps1 no longer derives the prerelease answer from the probed wheel"
        assert (
            "if ($script:WoaTorchIsPrerelease -or ($script:WoaTorchIndexUrl -match 'nightly'))"
            in text
        ), "the --prerelease=allow gate is back to testing only the URL spelling"

    def test_the_answer_is_handed_to_setup(self):
        """setup.ps1 cannot probe the index itself, so install.ps1 has to tell it."""
        assert "UNSLOTH_WOA_TORCH_PRERELEASE" in INSTALL_PS1.read_text(encoding = "utf-8")
        assert "UNSLOTH_WOA_TORCH_PRERELEASE" in SETUP_PS1.read_text(encoding = "utf-8")

    @requires_pwsh
    @pytest.mark.parametrize(
        "version, expected",
        [
            ("2.15.0.dev20260101+cu134", "True"),
            ("2.14.0rc1+cu134", "True"),
            ("2.14.0a1+cu134", "True"),
            ("2.14.0+cu134", "False"),
            ("", "False"),
        ],
    )
    def test_the_version_test_itself(self, version, expected):
        script = (
            f"$v = '{version}'\nWrite-Output ([bool]($v -match '(?i)\\d(a|b|rc)\\d|\\.dev\\d'))"
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expected, version


class TestAChangedPinInvalidatesTheHandover:
    """install.ps1's flags describe the index IT chose, not the one this run will use.

    Change UNSLOTH_TORCH_INDEX_URL and re-run `unsloth studio update` in the same shell and
    the pin outranks the handover for the install itself, while the torchaudio and prerelease
    answers still came from the previous channel: the trio asks a new index for an audio
    wheel it does not publish, and the whole torch update aborts.
    """

    def test_the_handover_index_is_read_before_it_is_overwritten(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        capture = text.index("$_woaHandoffIndex = if ($env:UNSLOTH_WOA_SELECTED_TORCH_INDEX)")
        rewrite = text.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        assert capture < rewrite, (
            "the handover value is read after setup.ps1 overwrites it, so the comparison "
            "always succeeds and the staleness check does nothing"
        )

    def test_both_flags_are_gated_on_it(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            '$WinArm64NoAudio = $WinArm64Venv -and -not ($WinArm64HandoffApplies -and $env:UNSLOTH_WOA_HAS_TORCHAUDIO -eq "1")'
            in text
        )
        assert '($WinArm64HandoffApplies -and $env:UNSLOTH_WOA_TORCH_PRERELEASE -eq "1")' in text

    def test_the_effective_index_prefers_the_pin(self):
        """The same order the install itself uses, or the two would disagree."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$WinArm64EffectiveTorchIndexUrl = if ($PinnedTorchIndexUrl)" in text
        assert "$_cudaIndexUrl = if ($PinnedTorchIndexUrl) { $TorchInstallIndexUrl }" in text

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, handoff, audio, expect_no_audio, why",
        [
            (
                "",
                "https://pypi.nvidia.com/nvtorch_oot",
                "1",
                "False",
                "unpinned and unchanged: the handover still describes this index",
            ),
            (
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://pypi.nvidia.com/nvtorch_oot",
                "1",
                "False",
                "pinned to the same index: still current",
            ),
            (
                "https://pypi.nvidia.com/nvtorch_oot/",
                "https://pypi.nvidia.com/nvtorch_oot",
                "1",
                "False",
                "a trailing slash is not a different index",
            ),
            (
                "https://mirror.test/simple",
                "https://pypi.nvidia.com/nvtorch_oot",
                "1",
                "True",
                "pinned elsewhere: the audio answer belongs to the old channel",
            ),
            ("", "", "1", "True", "no handover to trust"),
            ("", "https://pypi.nvidia.com/nvtorch_oot", "0", "True", "the handover says no audio"),
        ],
    )
    def test_the_staleness_rule(self, pinned, handoff, audio, expect_no_audio, why):
        script = "\n".join(
            [
                f"$_woaHandoffIndex = '{handoff}'",
                f"$PinnedTorchIndexUrl = '{pinned}'",
                "$WinArm64TorchIndexUrl = $_woaHandoffIndex",
                "$WinArm64Venv = $true",
                f"$env:UNSLOTH_WOA_HAS_TORCHAUDIO = '{audio}'",
                "$WinArm64EffectiveTorchIndexUrl = if ($PinnedTorchIndexUrl) { ([string]$PinnedTorchIndexUrl).Trim().TrimEnd('/') }",
                "                                  elseif ($WinArm64TorchIndexUrl) { $WinArm64TorchIndexUrl }",
                "                                  else { '' }",
                "$WinArm64HandoffApplies = [bool]($WinArm64EffectiveTorchIndexUrl -and $_woaHandoffIndex -and",
                "    $WinArm64EffectiveTorchIndexUrl.Equals($_woaHandoffIndex, [System.StringComparison]::OrdinalIgnoreCase))",
                '$WinArm64NoAudio = $WinArm64Venv -and -not ($WinArm64HandoffApplies -and $env:UNSLOTH_WOA_HAS_TORCHAUDIO -eq "1")',
                "Write-Output ([bool]$WinArm64NoAudio)",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == expect_no_audio, why


class TestAnOverrideConflictCanHideInAnInclude:
    """uv follows a nested -r inside an override file, so the top-level scan was not enough.

    Two override files naming the same package is an error to uv, so calling them disjoint
    and handing over both turns a working install into a resolution failure -- or, when uv
    accepts the pair, the caller's capped torch wins over the managed ARM64 override.
    """

    @staticmethod
    def _names(tmp_path, install: bool):
        source = INSTALL_PS1 if install else SETUP_PS1
        name = "Get-WoaRequirementEntries" if install else "Get-RequirementEntries"
        top = (tmp_path / "top.txt").as_posix()
        script = "\n".join(
            [
                _ps_function(source, name),
                f"$e = @({name} -Path '{top}')",
                "foreach ($x in $e) { Write-Output ($x.Line.Trim() + '|' + [System.IO.Path]::GetFileName($x.BaseDir)) }",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return [line for line in done.stdout.strip().splitlines() if line]

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    def test_an_included_file_is_read(self, tmp_path, install):
        nested = tmp_path / "managed"
        nested.mkdir()
        (nested / "nested.txt").write_text("torch<2.9\n", encoding = "utf-8")
        (tmp_path / "top.txt").write_text(
            "# a comment\nrich>=13\n-r managed/nested.txt\n", encoding = "utf-8"
        )
        lines = self._names(tmp_path, install)
        assert any(
            line.startswith("torch<2.9|managed") for line in lines
        ), f"the include was not followed, so a torch conflict reads as disjoint: {lines}"
        assert any(line.startswith("rich>=13|") for line in lines)
        assert not any(
            line.startswith("-r ") for line in lines
        ), "the include line survived as a line"

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    def test_a_cycle_terminates(self, tmp_path, install):
        (tmp_path / "top.txt").write_text("-r other.txt\nrich>=13\n", encoding = "utf-8")
        (tmp_path / "other.txt").write_text("-r top.txt\ntorch<2.9\n", encoding = "utf-8")
        lines = self._names(tmp_path, install)
        assert any(line.startswith("torch<2.9|") for line in lines)
        assert any(line.startswith("rich>=13|") for line in lines)

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    def test_a_missing_include_is_not_fatal(self, tmp_path, install):
        (tmp_path / "top.txt").write_text("-r gone.txt\nrich>=13\n", encoding = "utf-8")
        lines = self._names(tmp_path, install)
        assert [line.split("|")[0] for line in lines] == ["rich>=13"]

    def test_the_scan_and_the_fold_read_the_same_thing(self):
        """Detecting a conflict one level down and then folding only the top file would
        drop the line that caused the conflict, which is worse than not folding at all.
        """
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert text.count("$_woaOvEntries") == 3, "the conflict scan and the fold have diverged"
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        assert "foreach ($_woaEntry in (Get-RequirementEntries -Path $_woaFile))" in setup


class TestAFloorIsPep440AboutPrereleases:
    """21.0.0rc1 does not satisfy >=21.0.0, and staging writes an exact == from what it picks.

    Accepting a candidate the constraint would then reject is the failure this whole floor
    exists to prevent, so the ordering has to place a pre-release below its own release. A
    LARGER release is untouched: a wheelhouse nightly like the pyarrow 24.0.0.dev260 an
    end-to-end GB10 run staged still clears a 21.0.0 floor, which is what makes hosting a
    wheel the only step needed to enable a feature on that host.
    """

    CASES = [
        ("24.0.0.dev260", "21.0.0", True, "a nightly of a later release clears the floor"),
        ("22.0.0rc1", "21.0.0", True, "and so does an rc of a later release"),
        ("21.0.0", "21.0.0", True, "the release itself clears its own floor"),
        ("21.0.1", "21.0.0", True, "a later patch clears it"),
        ("21.0.0rc1", "21.0.0", False, "an rc sorts below the release it is for"),
        ("21.0.0.dev1", "21.0.0", False, "and so does a dev build"),
        ("19.0.1", "21.0.0", False, "plainly below the floor"),
        ("0.0.22.post7", "0.0.22.post7", True, "the xformers drop floor still holds"),
        ("0.0.22", "0.0.22.post7", False, "a bare release is below its own post"),
        ("0.0.23", "0.0.22.post7", True, "a later release outranks a post"),
        ("nonsense", "21.0.0", False, "unreadable compares as too old, keeping the drop"),
    ]

    @requires_pwsh
    @pytest.mark.parametrize(
        "version, floor, expected, why",
        CASES,
        ids = [f"{v}_vs_{f}" for v, f, _, _ in CASES],
    )
    def test_the_ordering(self, version, floor, expected, why):
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                f"Write-Output ([bool](Test-WoaVersionAtLeast -Version '{version}' -Floor '{floor}'))",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert (done.stdout.strip().splitlines()[-1] == "True") is expected, why

    def test_the_table_agrees_with_packaging(self):
        """The PowerShell cannot import packaging, so the expectations are checked against it
        here instead of being asserted from memory. Skipped rather than guessed if absent.
        """
        specifiers = pytest.importorskip("packaging.specifiers")
        for version, floor, expected, why in self.CASES:
            try:
                reference = specifiers.SpecifierSet(f">={floor}").contains(
                    version,
                    prereleases = True,
                )
            except Exception:
                continue  # "nonsense" is not a version; the PowerShell rule stands alone
            assert reference is expected, (
                f"{version} >= {floor}: packaging says {reference}, the table says "
                f"{expected} ({why})"
            )

    @requires_pwsh
    def test_abi3_is_refused_on_a_free_threaded_venv(self):
        """Free-threaded CPython has no stable ABI (CPython #111506), so abi3 is not an option
        there. Accepting usable tags must not have loosened this."""
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                '$script:WoaPyarrowFloor = "21.0.0"',
                _ps_function(INSTALL_PS1, "Test-WoaPyarrowWheelUsable"),
                "Write-Output ([bool](Test-WoaPyarrowWheelUsable "
                "-Name 'pyarrow-26.0.0-cp311-abi3-win_arm64.whl' "
                "-PyTag 'cp313' -AbiTag 'cp313t'))",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "False", (
            "an abi3 wheel was accepted on a free-threaded venv, where it cannot import"
        )

    @requires_pwsh
    def test_the_wheel_the_gb10_run_staged_is_still_accepted(self):
        """Named explicitly: a floor that rejected it would break a verified install."""
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                '$script:WoaPyarrowFloor = "21.0.0"',
                _ps_function(INSTALL_PS1, "Test-WoaPyarrowWheelUsable"),
                "Write-Output ([bool](Test-WoaPyarrowWheelUsable "
                "-Name 'pyarrow-24.0.0.dev260-cp313-cp313-win_arm64.whl' "
                "-PyTag 'cp313' -AbiTag 'cp313'))",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "True"


class TestAFindLinksPathWithASpaceSurvivesThePurge:
    """UV_FIND_LINKS is comma-separated to uv, so "C:\\private wheels" is ONE directory.

    Splitting it on whitespace as well tore it into two fragments, dropped the managed
    entry, and rejoined the pieces with commas -- leaving an air-gapped user pointed at two
    paths that do not exist, with nothing to say the mirror had been lost.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "var, value, expected, why",
        [
            (
                "UV_FIND_LINKS",
                r"C:\private wheels,{owned}",
                r"C:\private wheels",
                "the caller's spaced directory survives whole and ours is dropped",
            ),
            (
                "UV_FIND_LINKS",
                r"C:\a,C:\b",
                r"C:\a,C:\b",
                "two unrelated entries are both kept, unchanged",
            ),
            (
                "PIP_FIND_LINKS",
                r"C:\a {owned}",
                r"C:\a",
                "pip splits on whitespace, so that is how its value is read",
            ),
            (
                "UV_OVERRIDE",
                r"C:\a.txt {owned}\overrides.txt",
                r"C:\a.txt",
                "uv splits UV_OVERRIDE on whitespace, which is why 8.3 exists for it",
            ),
        ],
    )
    def test_the_purge_reads_each_variable_its_own_way(self, tmp_path, var, value, expected, why):
        owned = str(tmp_path / "woa")
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('$_woaOwnedPrefix = Join-Path $StudioHome "woa"')
        end = text.index("if ($script:WoaNativeCudaTorch) {", start)
        script = "\n".join(
            [
                "function Get-UvSafePath { param([string]$p) return $p }",
                f"$StudioHome = '{tmp_path}'",
                f"$env:{var} = '{value.format(owned = owned)}'",
                text[start:end],
                f"Write-Output ('[' + [Environment]::GetEnvironmentVariable('{var}') + ']')",
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
        assert got == expected.format(owned = owned), why


class TestThePipFallbackKeepsTheIndexArguments:
    """When uv cannot be obtained at all, Fast-Install uses pip -- and pip needs these.

    The NVIDIA channel publishes only torch, torchvision and torchaudio, so an install given
    just --index-url has nowhere to resolve their shared dependencies. Remove-UvOnlyResolverFlags
    is what makes handing pip the same list safe: it drops --index-strategy and rewrites
    --prerelease=allow as --pre.
    """

    def test_the_arguments_are_not_gated_on_uv(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "$WinArm64IndexArgs = if ($WinArm64Venv) {" in text
        ), "gating on $UseUv means the pip fallback gets no --extra-index-url at all"

    @requires_pwsh
    @pytest.mark.parametrize(
        "use_uv, pre, expect_pre",
        [(True, "1", True), (False, "1", True), (False, "0", False)],
    )
    def test_pip_receives_a_translated_list(self, use_uv, pre, expect_pre):
        """Executed end to end: build the list, then run it through the pip translation."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$WinArm64IndexArgs = if (")
        end = text.index("} else { @() }", start) + len("} else { @() }")
        script = "\n".join(
            [
                _function_source(text, "Remove-UvOnlyResolverFlags"),
                "$WinArm64Venv = $true",
                f"$UseUv = ${str(use_uv).lower()}",
                "$WinArm64TorchIndexUrl = 'https://pypi.nvidia.com/nvtorch_oot'",
                "$WinArm64EffectiveTorchIndexUrl = $WinArm64TorchIndexUrl",
                "$WinArm64HandoffApplies = $true",
                f"$env:UNSLOTH_WOA_TORCH_PRERELEASE = '{pre}'",
                text[start:end],
                "$pipArgs = Remove-UvOnlyResolverFlags -Arguments $WinArm64IndexArgs",
                "Write-Output ('[' + ($pipArgs -join ' ') + ']')",
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
        assert (
            "--extra-index-url https://pypi.org/simple" in got
        ), f"pip cannot resolve the trio's shared dependencies without it: {got!r}"
        assert "--index-strategy" not in got, "a uv-only flag would make pip print usage"
        assert "--prerelease" not in got, "likewise the uv spelling"
        assert ("--pre" in got) is expect_pre, got


class TestTheWoaIndexOutlivesTheManifest:
    """The dependency pass deletes the manifest before rebuilding it.

    A run that dies in that window used to leave nothing behind: the next update finds no
    handover and no manifest, falls back to the driver-derived cu130, and fails on an index
    with no win_arm64 wheel -- on every retry, until install.ps1 is run again by hand.
    """

    def test_the_marker_is_written_before_the_manifest_is_dropped(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        saved = text.index("Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex")
        dropped = text.index("$_ManifestDropped = $true")
        assert saved < dropped, (
            "written after the drop, the marker would not survive the very interruption "
            "it exists for"
        )

    def test_the_manifest_is_still_preferred(self):
        """The marker is a fallback, not a replacement: the manifest is rewritten each run."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$_woaFromManifest = Get-PersistedWoaTorchIndex -VenvPath $VenvDir" in text
        assert (
            "if ($_woaFromManifest) { $_woaFromManifest } else { Get-WoaTorchIndexMarker }" in text
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "url, persisted, why",
        [
            ("https://pypi.nvidia.com/nvtorch_oot", True, "NVIDIA's own channel"),
            ("https://pypi.nvidia.com/nvtorch_oot_nightly", True, "and its nightly"),
            ("https://mirror.corp.test/simple", False, "a pinned mirror is not persisted"),
            ("https://user:tok@pypi.nvidia.com/x", False, "userinfo could carry a token"),
            ("https://pypi.nvidia.com/x?token=abc", False, "nor may a query"),
            ("https://pypi.nvidia.com/x#f", False, "nor a fragment"),
            ("http://pypi.nvidia.com/x", False, "https only"),
            ("https://pypi.nvidia.com.evil.test/x", False, "a lookalike host"),
        ],
    )
    def test_the_marker_persists_only_what_the_manifest_would(self, tmp_path, url, persisted, why):
        """Same set as write_manifest, so this file cannot become the softer way in."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                _function_source(text, "Get-WoaTorchIndexMarker"),
                f"Save-WoaTorchIndexMarker -IndexUrl '{url}'",
                "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
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
        assert (got == url.rstrip("/")) is persisted, f"{why}: got {got!r}"

    @requires_pwsh
    def test_a_hand_edited_marker_cannot_redirect_the_install(self, tmp_path):
        """Checked on read as well as on write, exactly as the manifest is."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        woa = tmp_path / "woa"
        woa.mkdir()
        (woa / "torch-index.txt").write_text("https://evil.test/whl", encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Get-WoaTorchIndexMarker"),
                "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
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

    @requires_pwsh
    def test_the_marker_lives_where_the_dependency_pass_does_not_reach(self, tmp_path):
        """Beside overrides.txt, which survives the pass for the same reason."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                "Write-Output ('[' + (Get-WoaTorchIndexMarkerPath) + ']')",
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
        assert got.endswith("torch-index.txt")
        assert (tmp_path / "woa").name in got, "the woa directory, not the venv"

    @requires_pwsh
    @pytest.mark.parametrize(
        "url",
        [
            "https://user:s3cret@pypi.nvidia.com/nvtorch_oot",
            "https://pypi.nvidia.com/nvtorch_oot?token=s3cret",
            "https://mirror.corp.test/simple?token=s3cret",
        ],
    )
    def test_a_credential_never_reaches_the_disk(self, tmp_path, url):
        """The FILE, not the round-trip: a reader that refuses the value afterwards is no
        help at all if the token was written out in the first place. This is the guard the
        manifest already applies, and it has to hold on the write side by itself.
        """
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                f"Save-WoaTorchIndexMarker -IndexUrl '{url}'",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        marker = tmp_path / "woa" / "torch-index.txt"
        written = marker.read_text(encoding = "utf-8") if marker.exists() else ""
        assert "s3cret" not in written, f"the marker file holds a credential: {written!r}"
        assert written == "", "nothing unpersistable should be written at all"


class TestTheTorchMergeRebasesWhatItFolds:
    """Two override files is the NORMAL case on the native path, not an edge one.

    A non-conflicting caller file is deliberately kept where it sits, so UV_OVERRIDE names it
    alongside the generated one. The merge used to write itself into the caller's directory to
    keep relative references working, which only helped when there was exactly ONE directory --
    with two it fell to %TEMP% and every relative reference resolved against nothing.
    """

    @staticmethod
    def _merge(tmp_path, override_files):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        fake_py = tmp_path / "fakepython"
        fake_py.write_text(
            "#!/usr/bin/env bash\nprintf 'torch==2.11.0+cu130\\ntorchvision==0.26.0+cu130\\n'\n",
            encoding = "ascii",
        )
        fake_py.chmod(0o755)
        script = "\n".join(
            [
                "$SkipTorch = $false",
                _ps_function(INSTALL_PS1, "Get-WoaRequirementEntries"),
                _ps_function(INSTALL_PS1, "Resolve-WoaOverrideLine"),
                _ps_function(INSTALL_PS1, "New-UnslothTorchOverridesFile"),
                "$env:UV_OVERRIDE = '{}'".format(" ".join(str(f) for f in override_files)),
                f"$m = New-UnslothTorchOverridesFile -PythonExe '{fake_py}'",
                "Write-Output ('<<<' + [System.IO.File]::ReadAllText($m) + '>>>')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 180,
        )
        assert done.returncode == 0, done.stderr
        out = done.stdout
        return out[out.index("<<<") + 3 : out.rindex(">>>")]

    @requires_pwsh
    def test_two_override_directories_still_rebase(self, tmp_path):
        first = tmp_path / "managed"
        second = tmp_path / "corp"
        for directory in (first, second):
            directory.mkdir()
        (first / "nested.txt").write_text("idna==3.6\n", encoding = "utf-8")
        (first / "a.txt").write_text("-r nested.txt\nrich>=13\n", encoding = "utf-8")
        (second / "b.txt").write_text("./local.whl\nplainpkg==2.0\n", encoding = "utf-8")

        merged = self._merge(tmp_path, [first / "a.txt", second / "b.txt"])
        assert "idna==3.6" in merged, "the include one directory down was not followed"
        assert "-r " not in merged, "a relative include survived as a relative line"
        assert str(second / "local.whl") in merged.replace(
            "/", os.sep
        ), f"the bare relative wheel path was not rebased onto its own directory: {merged!r}"
        assert "rich>=13" in merged and "plainpkg==2.0" in merged

    @requires_pwsh
    def test_the_frozen_trio_still_wins(self, tmp_path):
        """Rebasing must not disturb what the merge is FOR: pinning the installed trio."""
        caller = tmp_path / "corp"
        caller.mkdir()
        (caller / "a.txt").write_text("torch==1.0\ntorchvision==0.1\nrich>=13\n", encoding = "utf-8")
        merged = self._merge(tmp_path, [caller / "a.txt"])
        assert merged.lstrip().startswith("torch==2.11.0+cu130")
        assert "torch==1.0" not in merged and "torchvision==0.1" not in merged
        assert "rich>=13" in merged


class TestThePipFallbackIsRefusedOnTheNativeStack:
    """pip has no override mechanism, so falling back to it does not recover here.

    The WoA overrides lift the released torch cap -- no win_arm64 CUDA wheel satisfies it --
    and drop the packages with no win_arm64 build at all. Constraints cannot stand in: they
    narrow a requirement, they cannot replace one. Running pip anyway downgrades a working
    CUDA torch or fails later with nothing to say why, so it is refused with a reason.
    """

    @pytest.fixture
    def ips(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ips_pip_fallback", STACK_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.mark.parametrize(
        "arm64, override, expected, why",
        [
            (True, "C:/s/woa/overrides.txt", True, "the native stack, with overrides in force"),
            (True, "", False, "an ARM64 run that configured none is not this case"),
            (False, "C:/s/woa/overrides.txt", False, "x64 keeps the fallback it always had"),
            (False, "", False, "and so does every other host"),
            (True, "   ", False, "a blank value is not an override file"),
        ],
    )
    def test_when_the_refusal_applies(self, ips, monkeypatch, arm64, override, expected, why):
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: arm64)
        if override:
            monkeypatch.setenv("UV_OVERRIDE", override)
        else:
            monkeypatch.delenv("UV_OVERRIDE", raising = False)
        assert ips._woa_overrides_are_load_bearing() is expected, why

    def test_both_fallback_paths_are_covered(self):
        """uv failing and uv never being available reach pip by different routes."""
        source = STACK_PY.read_text(encoding = "utf-8")
        assert source.count("_woa_overrides_are_load_bearing()") == 3, (
            "one definition and both fallback sites; a route that skips the check would "
            "silently resolve the wrong stack"
        )
        after_uv_failed = source.index("if _woa_overrides_are_load_bearing():")
        pip_build = source.index("pip_cmd = _build_pip_cmd(args)")
        assert after_uv_failed < pip_build, "the check has to precede the pip command"

    def test_the_message_names_the_remedy(self):
        source = STACK_PY.read_text(encoding = "utf-8")
        assert (
            "Install uv and re-run" in source
        ), "a refusal with no way forward is worse than the silent fallback it replaces"


class TestAnAnnotatedIncludeStillOpens:
    """`-r nested.txt # corporate pins` is a valid line, and the comment is not the path.

    Capturing it meant the include never opened, so a torch conflict inside it went unseen
    and the two override files were handed to uv as disjoint -- which uv then rejected,
    having followed the include itself.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "include_line, hashed_name, why",
        [
            ("-r nested.txt # corporate pins", False, "an inline comment is not the path"),
            ("-r nested.txt\t# tab before the hash", False, "any whitespace opens one"),
            ("--requirement nested.txt  # long form", False, "and the long spelling too"),
            ("-r nested.txt", False, "the plain case is unchanged"),
            ("-r a#b.txt", True, "a hash with no space before it belongs to the filename"),
        ],
    )
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    def test_the_target_is_read_without_its_comment(
        self, tmp_path, install, include_line, hashed_name, why
    ):
        source = INSTALL_PS1 if install else SETUP_PS1
        name = "Get-WoaRequirementEntries" if install else "Get-RequirementEntries"
        target = "a#b.txt" if hashed_name else "nested.txt"
        (tmp_path / target).write_text("idna==3.10\n", encoding = "utf-8")
        (tmp_path / "top.txt").write_text(f"{include_line}\nrich>=13\n", encoding = "utf-8")
        script = "\n".join(
            [
                _ps_function(source, name),
                f"$e = @({name} -Path '{(tmp_path / 'top.txt').as_posix()}')",
                "foreach ($x in $e) { Write-Output $x.Line.Trim() }",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        lines = [line for line in done.stdout.strip().splitlines() if line]
        assert "idna==3.10" in lines, f"{why}: the include did not open ({lines})"


class TestAnUnrecordableIndexInheritsNothing:
    """A marker we may not overwrite must not be left saying something else.

    An install that first used NVIDIA's channel and later moved to a credentialed corporate
    mirror kept the old marker, because the new URL is deliberately not persistable. The
    manifest does not record the mirror either, so the next fresh-shell update read the
    stale marker and put torch back on the channel the user had moved off.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "second, expect_left, why",
        [
            ("https://mirror.corp.test/simple", "", "a private mirror clears it"),
            ("https://user:tok@pypi.nvidia.com/x", "", "so does a credentialed one"),
            ("https://pypi.nvidia.com/x?token=a", "", "and one carrying a query"),
            (
                "https://pypi.nvidia.com/nvtorch_oot_nightly",
                "https://pypi.nvidia.com/nvtorch_oot_nightly",
                "a recordable index simply replaces it",
            ),
        ],
    )
    def test_the_marker_does_not_outlive_its_index(self, tmp_path, second, expect_left, why):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                _function_source(text, "Get-WoaTorchIndexMarker"),
                "Save-WoaTorchIndexMarker -IndexUrl 'https://pypi.nvidia.com/nvtorch_oot'",
                f"Save-WoaTorchIndexMarker -IndexUrl '{second}'",
                "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expect_left, why

    @requires_pwsh
    def test_clearing_removes_the_file_rather_than_blanking_it(self, tmp_path):
        """A zero-byte marker would read as empty anyway, but leaving one behind invites
        the next reader to treat "present" as meaningful.
        """
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                "Save-WoaTorchIndexMarker -IndexUrl 'https://pypi.nvidia.com/nvtorch_oot'",
                "Save-WoaTorchIndexMarker -IndexUrl 'https://mirror.corp.test/simple'",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert not (tmp_path / "woa" / "torch-index.txt").exists()


class TestALocalWheelIsOpenedBeforeItCounts:
    """The wheelhouse mirror trusted a filename; the resolver then trusted the mirror.

    _find_links_wheel_versions reads names, so a truncated wheel copied into the managed
    directory took its package off the ARM64 skip list, and uv failed the whole dependency
    pass on the corrupt archive instead of leaving one optional feature disabled.
    """

    def test_both_staging_branches_validate(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("$WoaExtraStaged = 0")
        end = text.index("$WoaOverrides = Join-Path $WoaDir", start)
        block = text[start:end]
        assert block.count("Test-ZipArchiveReadable") >= 3, (
            "the local mirror, the reused download and the fresh download all have to "
            f"open what they count: {block.count('Test-ZipArchiveReadable')}"
        )
        local = block[: block.index("} else {")]
        assert (
            "Test-ZipArchiveReadable" in local
        ), "the local branch counted a wheel on its filename alone"

    def test_the_check_precedes_the_copy(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("$WoaExtraStaged = 0")
        local = text[start : text.index("} else {", start)]
        assert local.index("Test-ZipArchiveReadable") < local.index("Copy-Item -LiteralPath"), (
            "validating after the copy would still put a corrupt wheel in the cache, "
            "where the resolver reads it"
        )


class TestTheMandatoryPyarrowWheelIsOpened:
    """pyarrow decides the ROUTE, so a truncated one must not select native mode.

    Staging writes an exact pyarrow== override from whichever file it picks, so a wheel that
    only looks right took the native path and then failed the resolve, with x64 already
    given up. The optional-wheel mirror validates for a milder reason: one feature stays
    disabled. This one costs the whole install.
    """

    def test_the_probe_and_the_staging_agree(self):
        """Different filters would let the probe clear one file and staging take another."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            text.count(
                "(Test-WoaPyarrowWheelUsable -Name $_.Name -PyTag $tag -AbiTag $AbiTag) -and"
            )
            == 2
        ), "the local probe and the local staging both filter on tags AND readability"
        assert text.count("(Test-ZipArchiveReadable -Path $_.FullName)") == 2

    @requires_pwsh
    @pytest.mark.parametrize(
        "content, expected, why",
        [
            ("zip", "wheelhouse", "a readable archive selects the native path"),
            ("truncated", "", "a truncated one does not, so the x64 path is kept"),
            ("empty", "", "nor does an empty file"),
        ],
    )
    def test_a_local_wheel_is_opened_before_native_is_chosen(
        self, tmp_path, content, expected, why
    ):
        import zipfile

        wheel = tmp_path / "pyarrow-24.0.0-cp313-cp313-win_arm64.whl"
        if content == "zip":
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("pyarrow/__init__.py", "")
        elif content == "truncated":
            wheel.write_bytes(b"PK\x03\x04truncated")
        else:
            wheel.write_bytes(b"")

        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "function substep { param($m, $c) }",
                "function Join-UrlPath { param($Base, $Path) return $Base }",
                "function Test-WoaWheelhouseIsLocal { $true }",
                "function Invoke-RestMethod { throw 'no network in this test' }",
                f"$script:WoaWheelhouse = '{tmp_path}'",
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                '$script:WoaPyarrowFloor = "21.0.0"',
                _ps_function(INSTALL_PS1, "Test-WoaPyarrowWheelUsable"),
                _ps_function(INSTALL_PS1, "Test-ZipArchiveReadable"),
                _ps_function(INSTALL_PS1, "Get-WoaPyarrowSource"),
                "Write-Output ('[' + (Get-WoaPyarrowSource -PythonMinor '3.13') + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 180,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expected, why


class TestARebasedOptionPathKeepsItsQuoting:
    """-r/-c/-f take ONE file argument, so an unquoted space truncates the path.

    Two ways in, and only one of them is about quoting: a caller who quoted the value had
    the quotes stripped and not put back, and a caller who had no reason to quote a plain
    relative name gets a space anyway when it rebases onto a directory that has one. Both
    end as a dependency pass failing to open a path cut at its first space.
    """

    @staticmethod
    def _rebase(source, line, base):
        # Here-strings, because the values under test contain both quote characters and
        # spaces -- interpolating them into a single-quoted argument is how a harness ends
        # up testing its own escaping instead of the function.
        script = "\n".join(
            [
                _ps_function(source, "Resolve-WoaOverrideLine"),
                f"$l = @'\n{line}\n'@",
                f"$b = @'\n{base}\n'@",
                "Write-Output ('[' + (Resolve-WoaOverrideLine -Line $l -BaseDir $b) + ']')",
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
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    @pytest.mark.parametrize(
        "line, base, quoted, why",
        [
            (
                '-c "corp pins/constraints.txt"',
                "/opt/corp",
                True,
                "the value was quoted because it needed to be",
            ),
            (
                "-r nested.txt",
                "/opt/my corp",
                True,
                "and here the base is what brings the space in",
            ),
            ("-f wheels", "/opt/my corp", True, "find-links too"),
            (
                "--constraint 'corp pins/c.txt'",
                "/opt/corp",
                True,
                "a single-quoted value is unwrapped and re-quoted the same way",
            ),
            (
                "-r nested.txt",
                "/opt/corp",
                False,
                "nothing with a space stays unquoted, as it always was",
            ),
            (
                "-c /already/absolute.txt",
                "/opt/corp",
                False,
                "an absolute path is returned untouched",
            ),
        ],
    )
    def test_the_result_is_quoted_exactly_when_it_has_to_be(self, install, line, base, quoted, why):
        source = INSTALL_PS1 if install else SETUP_PS1
        got = self._rebase(source, line, base)
        assert (
            got.split(None, 1)[0] == line.split(None, 1)[0]
        ), f"the option itself was dropped, leaving a bare path: {got!r}"
        has_quotes = '"' in got
        assert has_quotes is quoted, f"{why}: {got!r}"
        if quoted:
            # The quotes wrap the WHOLE path, or they solve nothing.
            inner = got[got.index('"') + 1 : got.rindex('"')]
            assert " " in inner, f"quoted but the space is outside them: {got!r}"
            assert (
                got.rindex('"') == len(got.rstrip()) - 1
            ), f"the closing quote has to end the value: {got!r}"

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    def test_a_rebased_line_still_names_one_file(self, install, tmp_path):
        """Read back the way a resolver reads it: split the option's argument on spaces
        and the path must still exist.
        """
        source = INSTALL_PS1 if install else SETUP_PS1
        base = tmp_path / "my corp"
        base.mkdir()
        (base / "constraints.txt").write_text("idna==3.10\n", encoding = "utf-8")
        got = self._rebase(source, "-c constraints.txt", base.as_posix())
        argument = got.split(None, 1)[1].strip()
        assert argument.startswith('"') and argument.endswith('"'), got
        assert pathlib.Path(
            argument[1:-1]
        ).exists(), f"the rebased path does not resolve to the file it names: {got!r}"

    def test_the_two_copies_stay_identical(self):
        """setup.ps1 carries a parity copy; a fix applied to one is a bug in the other."""

        def normalized(source: str) -> str:
            lines = [
                line.rstrip()
                for line in source.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            indent = min(len(line) - len(line.lstrip()) for line in lines)
            return "\n".join(line[indent:] for line in lines)

        assert normalized(
            _function_source(INSTALL_PS1.read_text(encoding = "utf-8"), "Resolve-WoaOverrideLine")
        ) == normalized(
            _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Resolve-WoaOverrideLine")
        )


class TestTheMarkerRecordsTheIndexActuallyUsed:
    """A generic pin never reached the marker, only the WoA chain did.

    $_cudaIndexUrl prefers $PinnedTorchIndexUrl, but $WinArm64TorchIndexUrl -- the value
    that was being saved -- consults only UNSLOTH_WOA_TORCH_INDEX_URL, the handover, the
    manifest and the marker. So a run pinned elsewhere installed from the pin and then
    recorded an NVIDIA channel it had not used, and the next fresh shell went back to it.
    """

    def test_the_saved_value_prefers_the_pin(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$_woaMarkerIndex = $_woaPinnedIndex" in text
        assert "else { $_woaMarkerIndex = $WinArm64TorchIndexUrl }" in text
        assert "Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex" in text
        # The same value the marker gets, or the manifest shadows the marker on the next
        # fresh shell: install_python_stack.py writes this export into woa_torch_index, and
        # the read chain prefers the manifest.
        assert "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex" in text

    def test_the_pin_is_read_before_it_is_used(self):
        """Get-PinnedTorchIndexUrl is a function, so only its DEFINITION has to precede this."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert text.index("function Get-PinnedTorchIndexUrl") < text.index(
            "$_woaPinnedIndex = if ($WinArm64Venv) { Get-PinnedTorchIndexUrl }"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, chain, expected, why",
        [
            (
                "",
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://pypi.nvidia.com/nvtorch_oot",
                "no pin: the WoA chain is what the run uses, so it is what is recorded",
            ),
            (
                "https://pypi.nvidia.com/nvtorch_oot_nightly",
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://pypi.nvidia.com/nvtorch_oot_nightly",
                "a pin to another NVIDIA channel replaces the recorded one",
            ),
            (
                "https://download.pytorch.org/whl/cu130",
                "https://pypi.nvidia.com/nvtorch_oot",
                "",
                "a pin to a recognised non-NVIDIA index clears it rather than leaving a lie",
            ),
            (
                "https://mirror.corp.test/simple",
                "https://pypi.nvidia.com/nvtorch_oot",
                "",
                "and so does a private mirror",
            ),
        ],
    )
    def test_what_ends_up_on_disk(self, tmp_path, pinned, chain, expected, why):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                f"function Get-PinnedTorchIndexUrl {{ return '{pinned}' }}",
                f"$WinArm64TorchIndexUrl = '{chain}'",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                _function_source(text, "Get-WoaTorchIndexMarker"),
                # A previous run recorded the public channel; this run may not inherit it.
                "Save-WoaTorchIndexMarker -IndexUrl 'https://pypi.nvidia.com/nvtorch_oot'",
                "$WinArm64Venv = $true",
                "$_woaHandoffIndex = ''",
                _persistence_block(text),
                "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expected, why


class TestTheManifestRecordsTheSameIndexAsTheMarker:
    """The marker was corrected; the manifest kept the old answer and outranked it.

    install_python_stack.py writes $env:UNSLOTH_WOA_SELECTED_TORCH_INDEX into the manifest as
    woa_torch_index, and $WinArm64TorchIndexUrl above reads the manifest BEFORE the marker. So
    exporting the WoA chain while saving the pin left the two records disagreeing, and the
    losing one was the correct one.
    """

    def test_the_export_and_the_save_carry_one_value(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        export = text.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        save = text.index("Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex")
        resolve = text.index("$_woaMarkerIndex = $_woaPinnedIndex")
        assert resolve < export < save, "resolved once, then written to both records"

    def test_the_stack_writes_that_variable_into_the_manifest(self):
        """The premise: without this read the export would reach nothing."""
        source = STACK_PY.read_text(encoding = "utf-8")
        assert "UNSLOTH_WOA_SELECTED_TORCH_INDEX" in source
        assert "woa_torch_index" in source

    def test_the_manifest_is_preferred_over_the_marker_on_read(self):
        """Which is why the two must agree rather than the marker being enough."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        chain = text.index("$_woaFromManifest = Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        assert (
            "if ($_woaFromManifest) { $_woaFromManifest } else { Get-WoaTorchIndexMarker }"
            in text[chain : chain + 300]
        )

    def test_a_moved_pin_drops_the_probed_indexs_flags(self):
        """torchaudio and prerelease were measured on the index install.ps1 probed."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        guard = text.index("if ($_woaMarkerIndex -ne $_woaHandoffIndex) {")
        body = text[guard : text.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX", guard)]
        assert "Remove-Item Env:UNSLOTH_WOA_HAS_TORCHAUDIO" in body
        assert "Remove-Item Env:UNSLOTH_WOA_TORCH_PRERELEASE" in body
        assert guard < text.index(
            "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex"
        ), "compared before the overwrite, or the two are always equal"


class TestThePypiPyarrowWheelIsPinnedToo:
    """ "PyPI has a compatible wheel" is not "the newest release is one".

    The probe cleared the native route on a wheel it then forgot, and the override was
    emitted only for a staged file. With just pyarrow>=21.0.0 in force, uv takes the newest
    release, and if that one ships only an sdist for this interpreter the resolve builds
    Arrow from source -- the outcome this preflight exists to prevent.
    """

    def test_the_probe_records_what_it_matched(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("function Get-WoaPyarrowSource")
        end = text.index("function Test-WoaNvidiaPresent", start)
        body = text[start:end]
        assert (
            "$script:WoaPyarrowWheelName = $match.Value" in body
        ), "the PyPI branch returns without naming the wheel it cleared"
        assert (
            "$script:WoaPyarrowWheelName = $null" in body
        ), "and it clears the name on entry, so a re-probe cannot inherit the first answer"

    @requires_pwsh
    @pytest.mark.parametrize(
        "body, expected_pin, why",
        [
            (
                "pyarrow-24.0.0-cp313-cp313-win_arm64.whl",
                "24.0.0",
                "the wheel that cleared the route is the one pinned",
            ),
            (
                "pyarrow-19.0.1-cp313-cp313-win_arm64.whl",
                "",
                "below the floor: nothing is cleared, so nothing is pinned",
            ),
        ],
    )
    def test_the_recorded_name_yields_the_pin(self, body, expected_pin, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "function substep { param($m, $c) }",
                "function Join-UrlPath { param($Base, $Path) return $Base }",
                "function Test-WoaWheelhouseIsLocal { $false }",
                f"function Invoke-RestMethod {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) return @'\n{body}\n'@ }}",
                "$script:WoaWheelhouse = ''",
                "function Test-WoaResolveReachesPyPI { $true }",
                _ps_function(INSTALL_PS1, "Test-WoaWheelTags"),
                _ps_function(INSTALL_PS1, "Test-WoaWheelTagsUsable"),
                _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
                '$script:WoaPyarrowFloor = "21.0.0"',
                _ps_function(INSTALL_PS1, "Test-WoaPyarrowWheelUsable"),
                _ps_function(INSTALL_PS1, "Test-ZipArchiveReadable"),
                _ps_function(INSTALL_PS1, "Get-WoaPyarrowSource"),
                "$null = Get-WoaPyarrowSource -PythonMinor '3.13'",
                # The emission, verbatim from the override block.
                "$pin = ''",
                "if ($script:WoaPyarrowWheelName -and $script:WoaPyarrowWheelName -match '^pyarrow-([^-]+)-') {",
                "    $pin = $Matches[1] }",
                "Write-Output ('[' + $pin + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 180,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expected_pin, why

    def test_the_override_is_emitted_for_every_source(self):
        """The pin is keyed on the recorded name, which all three routes now set."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert text.count("$script:WoaPyarrowWheelName = ") == 6, (
            "the script-level init, the per-probe reset, and one per source: pypi, the "
            "supplied UNSLOTH_PYARROW_WHEEL, the wheelhouse directory, the wheelhouse index"
        )


class TestEveryPyarrowRouteOpensWhatItKeeps:
    """Four ways in, and the last one was trusting a 200 response.

    A mirror can serve a truncated body with a successful status, and Invoke-WebRequest
    reports success for it. This wheel decides the route and the exact pyarrow== override is
    written from it, so an unreadable download kept native mode and then failed uv on the
    override -- after x64 had been given up.
    """

    def test_every_mandatory_route_validates(self):
        """Staging opens the two files it chooses itself; the probe opens the other two.

        Staging trusts the probe's answer for UNSLOTH_PYARROW_WHEEL, which is why that one
        is checked there and not again here.
        """
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('if ($script:WoaPyarrowSource -eq "local") {')
        end = text.index("$WoaExtraStaged = 0", start)
        block = text[start:end]
        assert block.count("Test-ZipArchiveReadable") == 2, (
            "the wheelhouse directory selection and the download: "
            f"{block.count('Test-ZipArchiveReadable')}"
        )
        probe = text[
            text.index("function Get-WoaPyarrowSource") : text.index(
                "function Test-WoaNvidiaPresent"
            )
        ]
        assert probe.count("Test-ZipArchiveReadable") == 2, (
            "the probe opens the supplied wheel and the local wheelhouse candidate before "
            f"it clears the native route: {probe.count('Test-ZipArchiveReadable')}"
        )

    def test_a_bad_download_is_removed_not_left_in_the_cache(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("Invoke-WebRequest -Uri (Join-UrlPath $script:WoaWheelhouse $wheelName)")
        block = text[start : start + 900]
        assert "Remove-Item -LiteralPath $_woaPaDest" in block, (
            "a corrupt wheel left in the managed directory is read by the resolver on "
            "every later run, and by _find_links_wheel_versions as proof of availability"
        )
        assert block.index("Remove-Item") < block.index("throw"), "removed before it gives up"

    def test_the_failure_falls_back_rather_than_continuing(self):
        """It throws into the existing catch, which is what disables the native route."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("Invoke-WebRequest -Uri (Join-UrlPath $script:WoaWheelhouse $wheelName)")
        block = text[start : start + 1400]
        assert "$script:WoaNativeCudaTorch = $false" in block
        assert block.index("throw") < block.index("$script:WoaNativeCudaTorch = $false")

    @requires_pwsh
    @pytest.mark.parametrize(
        "readable, expect_native, why",
        [
            (True, "True", "a readable download keeps the native route"),
            (False, "False", "a truncated one gives it up rather than failing later"),
        ],
    )
    def test_the_branch_end_to_end(self, tmp_path, readable, expect_native, why):
        """Executed, because a source-level assertion cannot tell a live branch from a dead
        one, and because the point is what is left on disk afterwards.
        """
        import zipfile

        served = tmp_path / "served.whl"
        if readable:
            with zipfile.ZipFile(served, "w") as archive:
                archive.writestr("pyarrow/__init__.py", "")
        else:
            served.write_bytes(b"PK\x03\x04truncated")
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()

        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index("                if ($wheelName) {")
        end = text.index(
            "                } else {\n                    $script:WoaNativeCudaTorch = $false",
            start,
        )
        script = "\n".join(
            [
                "function substep { param($m, $c) }",
                "function Join-UrlPath { param($Base, $Path) return $Path }",
                "function Invoke-WebRequest {",
                "  param([Parameter(ValueFromRemainingArguments=$true)]$a)",
                f"  Copy-Item -LiteralPath '{served.as_posix()}' -Destination $a[$a.IndexOf('-OutFile') + 1] -Force }}",
                _ps_function(INSTALL_PS1, "Test-ZipArchiveReadable"),
                f"$WoaWheelDir = '{wheel_dir.as_posix()}'",
                "$script:WoaWheelhouse = 'https://mirror.test/wheels'",
                "$script:WoaNativeCudaTorch = $true",
                "$script:WoaPyarrowWheelName = $null",
                "$wheelName = 'pyarrow-24.0.0-cp313-cp313-win_arm64.whl'",
                # The slice stops before the "} else {" that follows, so the brace closing
                # `if ($wheelName) {` is not in it.
                text[start:end] + "\n                }",
                "Write-Output ('[' + [bool]$script:WoaNativeCudaTorch + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 180,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expect_native, why
        staged = list(wheel_dir.glob("*.whl"))
        assert (
            bool(staged) is readable
        ), f"a rejected wheel must not stay in the managed directory: {staged}"


class TestAnExplicitPinIsPersistedWithoutAnOldRecord:
    """The persistence block was gated on the WoA chain alone.

    A native venv installed through a credentialed corporate mirror has nothing to recover:
    write_manifest keeps only NVIDIA's channels and Save-WoaTorchIndexMarker clears rather
    than lie, so the chain is empty. Pin UNSLOTH_TORCH_INDEX_URL at an NVIDIA channel on a
    later direct update and the torch install used it while the guard skipped both records,
    leaving the next fresh shell to fall back to the driver-derived download.pytorch index,
    which publishes no win_arm64 CUDA wheel.
    """

    def test_either_record_opens_the_block(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {" in text

    def test_the_pin_is_only_consulted_on_a_native_venv(self):
        """Every other host must reach the block exactly as it did before."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "$_woaPinnedIndex = if ($WinArm64Venv) { Get-PinnedTorchIndexUrl } else { $null }"
            in text
        )

    def test_the_pin_is_resolved_once(self):
        """Two reads of the getter are two chances to disagree about what was installed."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$_woaMarkerIndex = $_woaPinnedIndex" in text
        opens = text.index("if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {")
        closes = text.index("Restore-WoaResolverEnvironment", opens)
        assert (
            "Get-PinnedTorchIndexUrl" not in text[opens:closes]
        ), "the block calls the getter again instead of using the value the guard tested"

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, chain, expected, why",
        [
            (
                "https://pypi.nvidia.com/nvtorch_oot",
                "",
                "https://pypi.nvidia.com/nvtorch_oot",
                "nothing to recover, but the pin is what the install used",
            ),
            (
                "",
                "https://pypi.nvidia.com/nvtorch_oot",
                "https://pypi.nvidia.com/nvtorch_oot",
                "no pin: the recovered chain, exactly as before",
            ),
            ("", "", "", "neither: the block does not run at all"),
        ],
    )
    def test_what_the_guard_lets_through(self, tmp_path, pinned, chain, expected, why):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                f"$StudioHome = '{tmp_path}'",
                "$WinArm64Venv = $true",
                f"function Get-PinnedTorchIndexUrl {{ return '{pinned}' }}",
                f"$WinArm64TorchIndexUrl = '{chain}'",
                "$_woaHandoffIndex = ''",
                _function_source(text, "Get-WoaTorchIndexMarkerPath"),
                _function_source(text, "Test-WoaPersistableIndex"),
                _function_source(text, "Save-WoaTorchIndexMarker"),
                _function_source(text, "Get-WoaTorchIndexMarker"),
                _persistence_block(text),
                "Write-Output ('[' + $env:UNSLOTH_WOA_SELECTED_TORCH_INDEX + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expected, why


class TestTheProbedCudaWheelIsWhatGetsInstalled:
    """A floor plus unsafe-best-match is not a request for the wheel that was probed.

    uv documents unsafe-best-match as selecting the best version from the combined
    candidate set of every index, and the PyPI extra index is there because NVIDIA's
    channel publishes only the trio. So the moment PyPI's stable win_arm64 CPU torch is
    one release ahead of this channel, `torch>=2.4` takes it: the native GPU path is
    replaced by a CPU build that imports perfectly and runs everything on the CPU.
    """

    @staticmethod
    def _block() -> str:
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        start = text.index('if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {')
        return text[start : text.index("# Release preservation cannot run here", start)]

    def test_the_trio_is_pinned_to_the_probed_versions(self):
        block = self._block()
        for spec in (
            '"torch==$($script:WoaTorchWheelVersion)"',
            '"torchvision==$($script:WoaVisionWheelVersion)"',
            '"torchaudio==$($script:WoaAudioWheelVersion)"',
        ):
            assert spec in block, f"{spec} is not what gets installed"

    def test_an_unreadable_version_keeps_the_old_floor(self):
        """Strictly better than before is the bar; no worse is the floor."""
        block = self._block()
        assert 'else { "torch>=2.4" }' in block
        assert 'else { "torchvision>=0.19" }' in block
        assert 'else { "torchaudio>=2.4" }' in block

    def test_the_versions_come_from_the_probe(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        for name, project in (
            ("WoaTorchWheelVersion", None),
            ("WoaAudioWheelVersion", None),
            ("WoaVisionWheelVersion", "torchvision"),
        ):
            assert f"$script:{name} = " in text, f"{name} is never set"
        assert '-Project "torchvision"' in text, "torchvision is never probed"

    def test_the_pin_is_set_before_the_install_reads_it(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert text.index("$script:WoaVisionWheelVersion = ") < text.index(
            '"torchvision==$($script:WoaVisionWheelVersion)"'
        )

    def test_audio_still_needs_the_pairing(self):
        """The exact pin does not replace Test-WoaAudioMatchesTorch: a paired version is
        what makes torchaudio installable at all, and this only fixes which one lands."""
        block = self._block()
        audio = block.index("torchaudio==")
        assert "if ($script:WoaTorchAudio) {" in block[:audio]

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch_v, vision_v, audio_v, has_audio, expected, why",
        [
            (
                "2.15.0.dev20260101+cu134",
                "0.26.0.dev20260101+cu134",
                "2.11.0+cu134",
                "$true",
                "torch==2.15.0.dev20260101+cu134 torchvision==0.26.0.dev20260101+cu134"
                " torchaudio==2.11.0+cu134",
                "every probed version pinned, local tag included",
            ),
            (
                "2.14.0+cu134",
                "0.25.0+cu134",
                "",
                "$false",
                "torch==2.14.0+cu134 torchvision==0.25.0+cu134",
                "no paired audio: the trio is a pair, as before",
            ),
            (
                "",
                "",
                "",
                "$false",
                "torch>=2.4 torchvision>=0.19",
                "nothing readable: exactly the specs this used to send",
            ),
            (
                "2.14.0+cu134",
                "",
                "",
                "$false",
                "torch==2.14.0+cu134 torchvision>=0.19",
                "a half-readable probe pins what it read and floors the rest",
            ),
        ],
    )
    def test_what_the_specs_come_out_as(self, torch_v, vision_v, audio_v, has_audio, expected, why):
        script = "\n".join(
            [
                "function substep { param($m, $c) }",
                '$VenvPlatform = "win-arm64"',
                "$script:WoaNativeCudaTorch = $true",
                f"$script:WoaTorchWheelVersion = '{torch_v}'",
                f"$script:WoaVisionWheelVersion = '{vision_v}'",
                f"$script:WoaAudioWheelVersion = '{audio_v}'",
                f"$script:WoaTorchAudio = {has_audio}",
                "$script:WoaTorchIsPrerelease = $false",
                "$script:WoaTorchIndexUrl = 'https://pypi.nvidia.com/nvtorch_oot'",
                self._block(),
                "Write-Output ($_torchSpecs -join ' ')",
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


class TestTheCompanionWheelsArePairedWithTorch:
    """Newest-of-each is not a pair on a channel that publishes on separate schedules.

    NVIDIA's nightly channel stamps each project independently, and nightly torchvision
    metadata pins its exact torch. Maximizing the two separately and then pinning both
    exactly can therefore name a pair no index can satisfy -- after the installer has
    already committed to the ARM64 path, so there is nothing left to fall back to.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch_v, other_v, pairs, why",
        [
            ("2.15.0.dev20260101+cu134", "0.26.0.dev20260101+cu134", True, "same stamp and tag"),
            ("2.15.0.dev20260101+cu134", "0.26.0.dev20260102+cu134", False, "staggered nightly"),
            ("2.15.0.dev20260101+cu134", "0.26.0+cu134", False, "a release is not that build"),
            ("2.14.0+cu134", "0.29.0+cu134", True, "GA: torchvision 0.(M+15) pairs with torch 2.M"),
            (
                "2.14.0+cu134",
                "0.25.0+cu134",
                False,
                "GA, same tag, another release line: not a pair",
            ),
            ("2.14.0+cu134", "0.29.0+cu130", False, "a different CUDA build"),
            ("2.14.0+cu134", "", False, "nothing to pair with"),
        ],
    )
    def test_what_counts_as_a_pair(self, torch_v, other_v, pairs, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaWheelPairsWithTorch"),
                f"Write-Output (Test-WoaWheelPairsWithTorch -TorchVersion '{torch_v}'"
                f" -OtherVersion '{other_v}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert (done.stdout.strip().splitlines()[-1] == "True") is pairs, why

    def test_both_companions_are_probed_as_a_pair(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        for project in ("torchvision", "torchaudio"):
            line = [
                l
                for l in text.splitlines()
                if f'-Project "{project}"' in l and "Get-WoaCudaWheelVersion" in l
            ]
            assert line, f"{project} is not probed"
            assert all(
                "-PairWith $_woaTorchVersion" in l for l in line
            ), f"{project} is still maximized independently of torch"

    def test_an_unpaired_companion_falls_back_rather_than_pinning(self):
        """A pin the index cannot satisfy is worse than the floor it replaced."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "if (-not $script:WoaVisionWheelVersion) {" in text
        assert 'else { "torchvision>=0.19" }' in text


class TestTheRepairPathPinsTheSameWayTheInstallDoes:
    """setup.ps1's forced repair carries the same flags, so it had the same defect.

    Changing the index pin or repairing an unimportable torch enables --force-reinstall,
    and those specs are resolved with unsafe-best-match against a public PyPI extra index:
    the repair replaces the CUDA stack with CPU wheels the moment PyPI is one release ahead.
    """

    def test_the_repair_probes_the_effective_index(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "if ($WinArm64Venv -and $WinArm64EffectiveTorchIndexUrl) {" in text
        assert '$WinArm64TorchSpec = "torch==$_woaTorchV"' in text

    def test_the_companions_are_paired_here_too(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        for project in ("torchvision", "torchaudio"):
            assert f'-Project "{project}" -PairWith $_woaTorchV' in text

    def test_an_unanswered_index_leaves_every_spec_alone(self):
        """Best effort: no probe, no change, which is exactly today's behaviour."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index('$WinArm64TorchSpec = "torch>=2.4"')
        block = text[start : text.index("$_tritonSpec = ", start)]
        assert block.index('"torch>=2.4"') < block.index(
            "Get-WoaCudaWheelVersionParity"
        ), "the floor must be the default the probe overrides, not the other way round"
        assert "if ($_woaTorchV) {" in block, "an empty probe must not pin anything"

    @pytest.mark.parametrize(
        "install_fn, setup_fn",
        [
            ("Test-WoaWheelTags", "Test-WoaWheelTagsParity"),
            ("Test-WoaWheelPairsWithTorch", "Test-WoaPairsWithTorchParity"),
        ],
    )
    def test_the_parity_copies_have_not_drifted(self, install_fn, setup_fn):
        """Two copies of a rule is two chances for one of them to be wrong."""
        original = _ps_function(INSTALL_PS1, install_fn)
        copy = _function_source(SETUP_PS1.read_text(encoding = "utf-8"), setup_fn)

        def body(text: str) -> list:
            lines = text.split("\n")[1:]
            return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]

        assert body(original) == body(copy), f"{setup_fn} has drifted from {install_fn}"

    @requires_pwsh
    @pytest.mark.parametrize(
        "pair_with, expected, why",
        [
            ("", "0.26.0.dev20260102+cu134", "unpaired: the newest wheel on the index"),
            (
                "2.15.0.dev20260101+cu134",
                "0.26.0.dev20260101+cu134",
                "paired: the newest wheel from THIS torch build, not the newest overall",
            ),
            ("2.15.0.dev20251231+cu134", "", "no wheel from that build: nothing to pin"),
        ],
    )
    def test_the_parity_probe_pairs_when_asked(self, pair_with, expected, why):
        """The signatures differ, so the whole-body comparison above cannot cover this one:
        dropping the filter here leaves setup.ps1 pinning an unpairable companion."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        listing = " ".join(
            f'<a href="{name}">{name}</a>'
            for name in (
                "torchvision-0.26.0.dev20260101%2Bcu134-cp313-cp313-win_arm64.whl",
                "torchvision-0.26.0.dev20260102%2Bcu134-cp313-cp313-win_arm64.whl",
                # NEWER, and tagged for another interpreter: it must lose on the tag alone,
                # so dropping the tag filter cannot pass by picking the same version anyway.
                "torchvision-0.27.0.dev20260103%2Bcu134-cp312-cp312-win_arm64.whl",
            )
        )
        script = "\n".join(
            [
                f"function Invoke-RestMethod {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) return @'\n{listing}\n'@ }}",
                _function_source(text, "Test-WoaWheelTagsParity"),
                _function_source(text, "Test-WoaPairsWithTorchParity"),
                _function_source(text, "Get-WoaCudaWheelVersionParity"),
                "$v = Get-WoaCudaWheelVersionParity -IndexUrl 'https://pypi.nvidia.com/nvtorch_oot'"
                f" -PyTag 'cp313' -AbiTag 'cp313' -Project 'torchvision' -PairWith '{pair_with}'",
                "Write-Output ('[' + $v + ']')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1][1:-1] == expected, why


class TestTheOverrideFileDoesNotOutrankTheTorchPin:
    """uv's --overrides replace a version even for a requirement named on the command line.

    Verified against uv 0.10.7: an override of `packaging>=20` beat a CLI `packaging==24.0`.
    The generated file carries torch>=2.4 and torchvision>=0.19, so it discarded the exact
    CUDA pins the probe had just selected, and best-match then took PyPI's newer CPU wheel --
    the whole native GPU path replaced by a CPU build that imports.
    """

    def test_the_trio_is_dropped_for_that_one_command(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "function New-WoaTorchStepOverrideValue {" in text
        body = _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue")
        assert '@("torch", "torchvision", "torchaudio") -contains $name' in body

    def test_it_is_applied_around_the_native_install_only(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        swap = text.index("$_woaStep = New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved")
        guard = text.rindex(
            'if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64" -and $env:UV_OVERRIDE) {',
            0,
            swap,
        )
        assert guard < swap, "every other host must keep the overrides it had"

    def test_the_original_value_is_restored(self):
        """The later unsloth resolve still needs the drop list."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        swap = text.index("$_woaStep = New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved")
        tail = text[swap : swap + 1200]
        assert "} finally {" in tail
        assert "if ($_woaOverrideSwapped) { $env:UV_OVERRIDE = $_woaOverrideSaved }" in tail

    @requires_pwsh
    @pytest.mark.parametrize(
        "lines, expect_kept, why",
        [
            (
                ["torch>=2.4", "torchvision>=0.19", 'hf-transfer ; platform_machine == "AMD64"'],
                ["hf-transfer"],
                "the trio goes, the drop list stays",
            ),
            (
                ['hf-transfer ; platform_machine == "AMD64"', "pyarrow==21.0.0"],
                ["hf-transfer", "pyarrow"],
                "nothing to drop: the file is passed through untouched",
            ),
            (
                ["torch_geometric>=2.0"],
                ["torch_geometric"],
                "a different package that starts with torch",
            ),
        ],
    )
    def test_what_survives_the_filter(self, tmp_path, lines, expect_kept, why):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        src = tmp_path / "ovr.txt"
        src.write_text("\n".join(lines) + "\n", encoding = "utf-8")
        script = "\n".join(
            [
                "function Get-UvSafePath { param([string]$Path) return $Path }",
                _ps_function(INSTALL_PS1, "Get-WoaRequirementEntries"),
                _ps_function(INSTALL_PS1, "Resolve-WoaOverrideLine"),
                _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue"),
                f"$v = (New-WoaTorchStepOverrideValue -Value '{src}' -Dir '{tmp_path}').Value",
                "Get-Content -LiteralPath $v | ForEach-Object { Write-Output $_ }",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        out = done.stdout
        for name in expect_kept:
            assert name in out, f"{name} was dropped: {why}"
        for name in ("torch>=", "torchvision>=", "torchaudio>="):
            assert name not in out, f"{name} survived: {why}"


class TestATransientProbeFailureKeepsTheCudaBundle:
    """nvidia-smi is a probe, and one that did not answer is not evidence the GPU is gone.

    During a direct update of a native ARM64 CUDA install, a transiently missing nvidia-smi
    dropped windows-arm64-cuda from the expected kinds, deleted the working llama.cpp tree,
    and ran the selector with no NVIDIA evidence, which installs the CPU bundle instead.
    """

    def test_the_persisted_cuda_index_counts_as_evidence(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$_nvidiaEvidence = $HasNvidiaSmi -or ((Test-WinArm64Venv)" in text
        assert "elseif ($_nvidiaEvidence) { $_nvidiaKinds }" in text

    def test_only_a_persistable_index_counts(self):
        """Test-WoaPersistableIndex passes only NVIDIA's own channels, so a /cpu pin cannot
        claim to be NVIDIA evidence."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_nvidiaEvidence = ")
        assert "Test-WoaPersistableIndex $_woaEvidenceIndex" in text[start : start + 400]

    def test_rocm_still_wins_the_branch(self):
        """The ROCm arm is first and unchanged: this only widens the NVIDIA one."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        line = [l for l in text.splitlines() if "$expectedKinds = if (" in l][0]
        assert line.index("$HasROCm") < line.index("$_nvidiaEvidence")


class TestStableCompanionsPairByReleaseLine:
    """Every stable release has an empty dev stamp, so the CUDA tag alone paired a companion
    from any release the index still served, and the exact-pin install then asked for a pair
    that does not exist. torchvision 0.(M+15) requires torch 2.M exactly (PyPI metadata:
    0.25.0 -> torch==2.10.0, 0.19.0 -> torch==2.4.0); torchaudio agrees on major.minor.
    """

    @requires_pwsh
    @pytest.mark.parametrize(
        "project, torch_v, other_v, pairs, why",
        [
            ("torchvision", "2.10.0+cu134", "0.25.0+cu134", True, "the PyPI-documented pair"),
            ("torchvision", "2.10.0+cu134", "0.26.0+cu134", False, "the next GA, published early"),
            (
                "torchvision",
                "2.10.0+cu134",
                "0.24.0+cu134",
                False,
                "the previous one, still served",
            ),
            ("torchaudio", "2.10.0+cu134", "2.10.1+cu134", True, "audio: major.minor"),
            ("torchaudio", "2.14.0+cu134", "2.11.0+cu134", False, "the GA mismatch round 9 found"),
            (
                "torchvision",
                "2.15.0.dev20260101+cu134",
                "0.30.0.dev20260101+cu134",
                True,
                "nightly: the stamp still decides",
            ),
            (
                "torchvision",
                "2.15.0.dev20260101+cu134",
                "0.26.0.dev20260101+cu134",
                True,
                "nightly: the release offset is NOT applied to a stamped build",
            ),
        ],
    )
    def test_the_pairing(self, project, torch_v, other_v, pairs, why):
        script = "\n".join(
            [
                _ps_function(INSTALL_PS1, "Test-WoaWheelPairsWithTorch"),
                f"Write-Output (Test-WoaWheelPairsWithTorch -TorchVersion '{torch_v}'"
                f" -OtherVersion '{other_v}' -Project '{project}')",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        assert (done.stdout.strip().splitlines()[-1] == "True") is pairs, why

    def test_the_probe_passes_the_project_through(self):
        for path, fn in (
            (INSTALL_PS1, "Test-WoaWheelPairsWithTorch"),
            (SETUP_PS1, "Test-WoaPairsWithTorchParity"),
        ):
            text = path.read_text(encoding = "utf-8")
            assert (
                f"{fn} -TorchVersion $PairWith -OtherVersion $version -Project $Project" in text
            ), path.name


class TestTheFilteredOverrideIsUvSafeAndShortLived:
    """GetTempFileName() lands in %TEMP%, which follows the profile: a spaced one produced a
    quoted path in UV_OVERRIDE, and this file documents that uv rejects quoting there, so the
    torch command failed before installing anything. And the copies were never deleted: every
    native run made at least one, and a flattened caller file can carry an authenticated URL.
    """

    @requires_pwsh
    def test_the_copy_lands_in_the_given_directory_and_is_reported(self, tmp_path):
        src = tmp_path / "ovr.txt"
        src.write_text('torch>=2.4\nhf-transfer ; platform_machine == "AMD64"\n', encoding = "utf-8")
        woa = tmp_path / "woa"
        woa.mkdir()
        script = "\n".join(
            [
                "function Get-UvSafePath { param([string]$Path) return $Path }",
                _ps_function(INSTALL_PS1, "Get-WoaRequirementEntries"),
                _ps_function(INSTALL_PS1, "Resolve-WoaOverrideLine"),
                _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue"),
                f"$r = New-WoaTorchStepOverrideValue -Value '{src}' -Dir '{woa}'",
                "Write-Output ('VALUE=' + $r.Value)",
                "Write-Output ('TEMPS=' + ($r.Temps -join ';'))",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        out = dict(l.split("=", 1) for l in done.stdout.strip().splitlines() if "=" in l)
        assert out["TEMPS"], "the created copy is not reported, so nothing can delete it"
        assert out["TEMPS"].startswith(str(woa)), "the copy must live under the uv-safe directory"
        assert out["VALUE"] == out["TEMPS"]

    def test_every_path_goes_through_the_uv_safe_helper(self):
        body = _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue")
        assert "$safe = foreach ($f in $files) { Get-UvSafePath $f }" in body

    def test_the_caller_deletes_the_copies_on_every_exit(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        swap = text.index(
            "New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved -Dir $script:WoaDir"
        )
        tail = text[swap : swap + 1200]
        fin = tail.index("} finally {")
        assert (
            "foreach ($_woaTmp in $_woaOverrideTemps) { Remove-Item -LiteralPath $_woaTmp"
            in tail[fin:]
        )

    def test_the_woa_directory_is_published_to_script_scope(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "$script:WoaDir = $WoaDir" in text


class TestSetupSwapsTheOverrideAroundItsOwnTorchInstall:
    """Restore-WoaResolverEnvironment puts the generated overrides.txt back before setup's CUDA
    trio is installed, so the same floors undid the same exact pins there."""

    def test_the_swap_wraps_the_install_and_restores_in_finally(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        restore = text.index("\nRestore-WoaResolverEnvironment")
        swap = text.index("New-WoaTorchStepOverrideValueParity -Value $_woaStepSaved")
        assert (
            restore < swap
        ), "the swap must come after the overrides are restored, or it swaps nothing"
        tail = text[swap : swap + 1600]
        assert "Fast-Install @_cudaTrio" in tail
        fin = tail.index("} finally {")
        assert "if ($_woaStepSwapped) { $env:UV_OVERRIDE = $_woaStepSaved }" in tail[fin:]
        assert (
            "foreach ($_woaTmp in $_woaStepTemps) { Remove-Item -LiteralPath $_woaTmp" in tail[fin:]
        )

    def test_only_a_native_venv_swaps(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "if ($WinArm64Venv -and $env:UV_OVERRIDE) {" in text

    @requires_pwsh
    def test_the_parity_helper_drops_the_trio(self, tmp_path):
        src = tmp_path / "ovr.txt"
        src.write_text("torch>=2.4\ntorchvision>=0.19\npyarrow==21.0.0\n", encoding = "utf-8")
        text = SETUP_PS1.read_text(encoding = "utf-8")
        script = "\n".join(
            [
                "function Get-UvSafePath { param([string]$Path) return $Path }",
                _function_source(text, "Get-RequirementEntries"),
                _function_source(text, "Resolve-WoaOverrideLine"),
                _function_source(text, "New-WoaTorchStepOverrideValueParity"),
                f"$r = New-WoaTorchStepOverrideValueParity -Value '{src}' -Dir '{tmp_path}'",
                "Get-Content -LiteralPath $r.Value | ForEach-Object { Write-Output $_ }",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        lines = [l for l in done.stdout.splitlines() if l.strip()]
        assert lines == ["pyarrow==21.0.0"], lines


class TestNvidiaEvidenceSurvivesTheFastPath:
    """When the manifest verifies, $SkipPythonDeps skips the whole dependency block, so
    $WinArm64EffectiveTorchIndexUrl is never set and the llama.cpp check read "no evidence":
    a transient nvidia-smi failure on a no-op update then deleted the working CUDA bundle."""

    def test_the_index_is_read_at_the_check_when_the_pass_did_not_run(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        start = text.index("$_woaEvidenceIndex = if ($WinArm64EffectiveTorchIndexUrl)")
        block = text[start : text.index("$_nvidiaEvidence = ", start)]
        assert "Get-PinnedTorchIndexUrl" in block
        assert "Get-PersistedWoaTorchIndex -VenvPath $VenvDir" in block
        assert "Get-WoaTorchIndexMarker" in block
        assert (
            block.index("Get-PinnedTorchIndexUrl")
            < block.index("Get-PersistedWoaTorchIndex")
            < block.index("Get-WoaTorchIndexMarker")
        ), "same order as the dependency pass"

    def test_the_evidence_uses_that_index(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "(Test-WinArm64Venv) -and $_woaEvidenceIndex -and" in text
        assert "Test-WoaPersistableIndex $_woaEvidenceIndex" in text

    def test_the_check_sits_outside_the_dependency_guard(self):
        """The premise: if it were inside, the fast path would never reach it at all."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        guard = text.index("\nif (-not $SkipPythonDeps) {")
        depth = 0
        for i in range(guard + 1, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                break
        assert text.index("$_nvidiaEvidence = ") > i


class TestThePyPIProbeHonoursUvConfiguration:
    """A direct HTTP probe can see pypi.org while uv, under a uv.toml with no-index or an
    exclusive default-index, cannot. "pypi" then skipped a usable wheelhouse wheel for one the
    resolve would never fetch."""

    def test_the_pyarrow_probe_is_gated(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        probe = text.index('Invoke-RestMethod -Uri "https://pypi.org/simple/pyarrow/"')
        assert "if (Test-WoaResolveReachesPyPI) { try {" in text[probe - 400 : probe]

    @staticmethod
    def _reaches(tmp_path, files: dict, env: dict) -> str:
        for name, body in files.items():
            (tmp_path / name).parent.mkdir(parents = True, exist_ok = True)
            (tmp_path / name).write_text(body, encoding = "utf-8")
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        setenv = "\n".join(f"$env:{k} = '{v}'" for k, v in env.items())
        script = "\n".join(
            [
                "foreach ($n in 'UV_OFFLINE','PIP_NO_INDEX','UV_DEFAULT_INDEX','UV_INDEX_URL','PIP_INDEX_URL','UV_NO_CONFIG','UV_CONFIG_FILE') { Remove-Item Env:$n -ErrorAction SilentlyContinue }",
                f"$env:APPDATA = '{tmp_path / 'appdata'}'",
                f"$env:ProgramData = '{tmp_path / 'programdata'}'",
                f"Set-Location -LiteralPath '{tmp_path / 'proj'}'",
                setenv,
                _ps_function(INSTALL_PS1, "Test-WoaUrlIsPublicPyPI"),
                # Read-WoaUvTomlIndexKeys scans for quotes now, so its two scanners come with
                # it. Omitting one is a command-not-found, which aborts the statement instead
                # of answering, and the caller then reports PyPI as reachable.
                _ps_function(INSTALL_PS1, "Remove-WoaTomlComment"),
                _ps_function(INSTALL_PS1, "Split-WoaTomlKey"),
                _ps_function(INSTALL_PS1, "Read-WoaUvTomlIndexKeys"),
                _ps_function(INSTALL_PS1, "Get-WoaUvConfigIndexPolicy"),
                _ps_function(INSTALL_PS1, "Test-WoaResolveReachesPyPI"),
                "Write-Output (Test-WoaResolveReachesPyPI)",
            ]
        )
        (tmp_path / "proj").mkdir(exist_ok = True)
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1]

    @requires_pwsh
    @pytest.mark.parametrize(
        "files, env, expected, why",
        [
            ({}, {}, "True", "nothing configured: PyPI"),
            ({"proj/uv.toml": "no-index = true\n"}, {}, "False", "project uv.toml no-index"),
            (
                {"proj/uv.toml": 'default-index = "https://pypi.corp.test/simple"\n'},
                {},
                "False",
                "exclusive default-index",
            ),
            (
                {"proj/uv.toml": 'index-url = "https://pypi.corp.test/simple"\n'},
                {},
                "False",
                "the older spelling",
            ),
            (
                {"proj/uv.toml": "[pip]\nno-index = true\n"},
                {},
                "False",
                "under [pip], which uv pip reads",
            ),
            (
                {
                    "proj/uv.toml": '[[index]]\nurl = "https://pypi.corp.test/simple"\ndefault = true\n'
                },
                {},
                "False",
                "an [[index]] with default = true replaces PyPI",
            ),
            (
                {"proj/uv.toml": '[[index]]\nurl = "https://pypi.corp.test/simple"\n'},
                {},
                "True",
                "an extra index leaves PyPI in play",
            ),
            (
                {"proj/pyproject.toml": "[tool.uv]\nno-index = true\n"},
                {},
                "False",
                "pyproject [tool.uv]",
            ),
            (
                {"proj/pyproject.toml": '[project]\nname = "x"\n'},
                {},
                "True",
                "a pyproject without [tool.uv] is ignored",
            ),
            ({"uv.toml": "no-index = true\n"}, {}, "False", "found in a parent directory"),
            ({"appdata/uv/uv.toml": "no-index = true\n"}, {}, "False", "the user file"),
            (
                {"proj/uv.toml": "no-index = false\n", "appdata/uv/uv.toml": "no-index = true\n"},
                {},
                "True",
                "project outranks user for a scalar",
            ),
            (
                {"proj/uv.toml": "no-index = true\n"},
                {"UV_NO_CONFIG": "1"},
                "True",
                "UV_NO_CONFIG discovers nothing",
            ),
            (
                {"proj/uv.toml": "no-index = true\n"},
                {"UV_DEFAULT_INDEX": "https://pypi.org/simple"},
                "True",
                "an index in the environment outranks every file",
            ),
            (
                {"other.toml": "no-index = true\n"},
                {"UV_CONFIG_FILE": "__TMP__/other.toml"},
                "False",
                "UV_CONFIG_FILE names the one file read",
            ),
            (
                {
                    "proj/uv.toml": 'index = [{ url = "https://pypi.corp.test/simple", default = true }]\n'
                },
                {},
                "False",
                "an inline table this parser does not model is not guessed at",
            ),
            # The host, not a substring: a lookalike that merely contains the name is not PyPI.
            (
                {},
                {"UV_DEFAULT_INDEX": "https://pypi.org.corp.example/simple"},
                "False",
                "a subdomain lookalike in the environment",
            ),
            (
                {},
                {"UV_DEFAULT_INDEX": "https://packages.example/api/pypi/pypi.org/simple"},
                "False",
                "the name in the path",
            ),
            (
                {},
                {"UV_INDEX_URL": "HTTPS://PYPI.ORG/simple/"},
                "True",
                "case does not matter for a host",
            ),
            (
                {},
                {"PIP_INDEX_URL": "https://user:token@pypi.org/simple"},
                "True",
                "credentials do not hide the host",
            ),
            (
                {},
                {"UV_DEFAULT_INDEX": "https://test.pypi.org/simple"},
                "False",
                "TestPyPI does not carry these packages",
            ),
            (
                {"proj/uv.toml": 'default-index = "https://pypi.org.corp.example/simple"\n'},
                {},
                "False",
                "a subdomain lookalike in a config file",
            ),
            (
                {"proj/uv.toml": 'default-index = "https://pypi.org/simple"\n'},
                {},
                "True",
                "public PyPI named explicitly in a config file",
            ),
            # uv pip: [pip] scalars outrank the top-level ones whatever their order in the file,
            # and an [[index]] entry with default = true outranks [pip].index-url (uv 0.10.7).
            (
                {"proj/uv.toml": "no-index = false\n[pip]\nno-index = true\n"},
                {},
                "False",
                "[pip].no-index = true beats the top-level false",
            ),
            (
                {"proj/uv.toml": "no-index = true\n[pip]\nno-index = false\n"},
                {},
                "True",
                "[pip].no-index = false beats the top-level true",
            ),
            (
                {"proj/uv.toml": "[pip]\nno-index = true\n\n[other]\nx = 1\nno-index = false\n"},
                {},
                "False",
                "a later section does not reopen the top level",
            ),
            (
                {
                    "proj/uv.toml": 'index-url = "https://pypi.org/simple"\n[pip]\nindex-url = "https://pypi.corp.test/simple"\n'
                },
                {},
                "False",
                "[pip].index-url beats the top-level index-url",
            ),
            (
                {
                    "proj/uv.toml": 'index-url = "https://pypi.corp.test/simple"\n[pip]\nindex-url = "https://pypi.org/simple"\n'
                },
                {},
                "True",
                "the other way round",
            ),
            (
                {
                    "proj/uv.toml": '[[index]]\nurl = "https://pypi.corp.test/simple"\ndefault = true\n[pip]\nindex-url = "https://pypi.org/simple"\n'
                },
                {},
                "False",
                "[[index]] default = true beats [pip].index-url",
            ),
            (
                {
                    "proj/uv.toml": '[pip]\nindex-url = "https://pypi.corp.test/simple"\n[[index]]\nurl = "https://pypi.org/simple"\ndefault = true\n'
                },
                {},
                "True",
                "and still does when it comes later in the file",
            ),
            (
                {"proj/uv.toml": 'no-index = true\n[pip]\nindex-url = "https://pypi.org/simple"\n'},
                {},
                "False",
                "no-index disables every registry, [pip].index-url included",
            ),
            (
                {
                    "proj/pyproject.toml": "[tool.uv]\nno-index = false\n[tool.uv.pip]\nno-index = true\n"
                },
                {},
                "False",
                "the same under [tool.uv.pip]",
            ),
        ],
    )
    def test_where_the_resolve_will_look(self, tmp_path, files, env, expected, why):
        env = {k: v.replace("__TMP__", str(tmp_path)) for k, v in env.items()}
        assert self._reaches(tmp_path, files, env) == expected, why


class TestARedundantWheelLeavesTheManagedDirectoryToo:
    """Skipping the copy was not enough: the managed directory is prepended to UV_FIND_LINKS,
    so a copy already there (the offline-cache mode points the wheelhouse AT that directory,
    and an earlier install may have staged one) still wins the tie over the upstream wheel."""

    def _redundant_branches(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        out = []
        for marker in (
            'substep "windows on arm: PyPI publishes $($wheel.Name) itself -- taking it from there, not the wheelhouse."',
            'substep "windows on arm: PyPI publishes $name itself -- taking it from there, not the wheelhouse."',
        ):
            i = text.index(marker)
            out.append(text[i : text.index("continue", i)])
        return out

    def test_both_wheelhouse_modes_remove_the_managed_copy(self):
        local, url = self._redundant_branches()
        assert "Remove-Item -LiteralPath (Join-Path $WoaWheelDir $wheel.Name) -Force" in local
        assert "Remove-Item -LiteralPath (Join-Path $WoaWheelDir $name) -Force" in url

    def test_only_the_managed_copy_goes(self):
        """An external wheelhouse file is never the target: the path is built from $WoaWheelDir."""
        for branch in self._redundant_branches():
            assert "$wheel.FullName" not in branch.split("Remove-Item")[1]
            assert "$script:WoaWheelhouse" not in branch.split("Remove-Item")[1]

    @requires_pwsh
    def test_the_removal_line_deletes_a_stale_copy_and_leaves_the_source(self, tmp_path):
        src = tmp_path / "house"
        src.mkdir()
        managed = tmp_path / "wheels"
        managed.mkdir()
        (src / "tiktoken-0.9.0-cp313-cp313-win_arm64.whl").write_bytes(b"x")
        (managed / "tiktoken-0.9.0-cp313-cp313-win_arm64.whl").write_bytes(b"x")
        local, _ = self._redundant_branches()
        line = [l.strip() for l in local.splitlines() if l.strip().startswith("Remove-Item")][0]
        script = "\n".join(
            [
                f"$WoaWheelDir = '{managed}'",
                f"$wheel = Get-Item -LiteralPath '{src / 'tiktoken-0.9.0-cp313-cp313-win_arm64.whl'}'",
                line,
                "Write-Output ('SRC=' + (Test-Path -LiteralPath $wheel.FullName))",
                f"Write-Output ('MANAGED=' + (Test-Path -LiteralPath '{managed / 'tiktoken-0.9.0-cp313-cp313-win_arm64.whl'}'))",
            ]
        )
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert done.returncode == 0, done.stderr
        assert "SRC=True" in done.stdout and "MANAGED=False" in done.stdout, done.stdout
