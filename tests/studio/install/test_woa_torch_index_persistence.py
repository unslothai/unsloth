# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a Windows-on-ARM install has to recover when it is not install.ps1.

install.ps1 hands its resolver decisions to setup.ps1 through process-scoped environment
variables, and a direct `unsloth studio update` runs in a fresh shell where all of them are
gone. Windows on ARM is the only platform whose CUDA torch wheels live nowhere on
download.pytorch.org, so nothing here can be re-derived from the host: the index has to come
off disk (the manifest and the marker below) and so do the generated requirement overrides
(TestResolverEnvironmentRestore).

Recording the index at all is a deliberate exception to the rule the manifest documents for
itself -- the FLAVOR, never the URL it came from, because a pinned index can carry a token in
its userinfo, query or fragment and this file is printed back by verify-install. The exception
only holds while the guard does, so both halves are tested: the Python that writes it and the
PowerShell that reads it back, each of which must refuse independently.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from woa_ps_harness import (
    CONSTRAINTS_SRC,
    CORP_INDEX,
    INSTALL_PS1,
    INSTALL_SRC,
    INVOKE_RESTMETHOD_NO_NETWORK,
    INVOKE_RESTMETHOD_OFFLINE,
    JOIN_URL_PATH,
    JOIN_URL_RETURNS_BASE,
    JOIN_URL_RETURNS_PATH,
    LLAMA_SRC,
    MANIFEST_PY,
    MARKER_FUNCS,
    NV_GA,
    NV_NIGHTLY,
    PACKAGE_ROOT,
    PWSH,
    PYARROW_FLOOR,
    PYARROW_USABLE_FUNCS,
    PYPI,
    SETUP_PS1,
    SETUP_SRC,
    STACK_PY,
    STACK_SRC,
    SUBSTEP_NOOP,
    UV_INDEX_ENV,
    UV_ONLY_INDEX_ENV,
    UV_POLICY_ENV,
    UV_SAFE_PATH,
    WHEEL_TAG_FUNCS,
    _function_source,
    _ps,
    _ps_copies,
    _ps_function,
    _ps_kv,
    _ps_last,
    _ps_ok,
    _script,
    clear_env,
    functions,
    invoke_restmethod,
    native_probe_script,
    pyarrow_source_script,
    requires_pwsh,
    slice_between,
    substep_collector,
)


# PowerShell's Join-Path uses the HOST separator, so a hardcoded POSIX home only ever exercises
# POSIX and fails on Windows. Both sides are built the way the host builds them.
FAKE_HOME = (
    "C:\\Users\\u\\AppData\\Local\\unsloth" if os.name == "nt" else "/home/u/AppData/Local/unsloth"
)


def _native_path_value(template: str) -> str:
    """Fill in the fake home and speak the host's separator, entries and expectations alike."""
    return template.format(home = FAKE_HOME).replace("/", os.sep)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_woa", MANIFEST_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_manifest_module()

# TestTheDependencyIndexFollowsTheResolverPolicy shadows PYPI with the --extra-index-url form
# it expects on a command line, so the bare URL keeps a second name for use inside it.
PYPI_URL = PYPI
WORKFLOW = PACKAGE_ROOT / ".github" / "workflows" / "windows-arm64-ci.yml"


def _persistence_block(text: str) -> str:
    """The live `if (...) { ... }` that writes both index records, sliced out rather than
    restated: a copy of it passes forever after the original stops matching it."""
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


# (url, may_be_persisted, why)
CANDIDATES = (
    (NV_GA, True, "the GA channel install.ps1 probes"),
    (NV_NIGHTLY + "/", True, "the nightly channel, trailing slash"),
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

    @pytest.mark.parametrize(
        "url, persisted",
        [
            ("HTTPS://PYPI.NVIDIA.COM/nvtorch_oot", NV_GA),
            ("https://PyPI.nvidia.com/nvtorch_oot_nightly/", NV_NIGHTLY),
            ("https://pypi.nvidia.com/NVTORCH_oot", "https://pypi.nvidia.com/NVTORCH_oot"),
        ],
    )
    def test_scheme_and_host_case_is_normalised_and_the_path_kept(self, tmp_path, url, persisted):
        """RFC 3986: scheme and host compare case-insensitively; a path does not."""
        path = im.write_manifest(root = tmp_path, req_root = tmp_path, woa_torch_index = url)
        payload = json.loads(pathlib.Path(path).read_text(encoding = "utf-8"))
        assert payload["woa_torch_index"] == persisted

    def test_the_key_is_absent_when_no_index_was_chosen(self, tmp_path: pathlib.Path):
        """Every other host, and every WoA host that stayed on the x64 stack."""
        im.write_manifest(root = tmp_path, req_root = tmp_path)
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload

    def test_the_addition_is_backwards_compatible(self, tmp_path: pathlib.Path):
        """An additive optional key: older readers see the schema they already parse."""
        im.write_manifest(root = tmp_path, req_root = tmp_path, woa_torch_index = NV_GA)
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert payload["schema"] == 1, "the key is additive; bumping the schema is not"
        state = im.verify_install(root = tmp_path, req_root = tmp_path)
        assert state["manifest_ok"] is True, state["reason"]

    def test_the_installer_passes_the_handover_variable_through(self):
        """install.ps1 exports it; nothing else supplies this value."""
        assert re.search(
            r"woa_torch_index\s*=\s*os\.environ\.get\(\s*[\"']UNSLOTH_WOA_SELECTED_TORCH_INDEX[\"']",
            STACK_SRC,
        ), "install_python_stack.py no longer forwards the index install.ps1 selected"


class TestReadSide:
    """studio/setup.ps1: what Get-PersistedWoaTorchIndex hands back to the resolver."""

    @requires_pwsh
    @pytest.mark.parametrize("url, allowed, why", [c for c in CANDIDATES if c[0]])
    def test_a_hand_edited_manifest_cannot_redirect_the_install(
        self, tmp_path: pathlib.Path, url: str, allowed: bool, why: str
    ):
        """The write guard is not enough on its own: the file sits in the user's venv and
        anything can put a line in it."""
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
        body = _function_source(SETUP_SRC, "Get-PersistedWoaTorchIndex")
        script = f"{body}\nWrite-Output (Get-PersistedWoaTorchIndex -VenvPath '{venv}')"
        return _ps_ok(script).stdout.strip()


class TestResolverEnvironmentRestore:
    """The other half of what a fresh shell loses.

    install.ps1 writes StudioHome\\woa\\overrides.txt and stages a win_arm64 wheelhouse beside
    it, then exports both through UV_OVERRIDE / UV_FIND_LINKS / PIP_FIND_LINKS. Those exports
    are process-scoped, so a direct `unsloth studio update` starts without them -- and the
    dependency pass resolves `ddgs`, which requires httpx[brotli], which requires Brotli on
    CPython, which publishes no win_arm64 wheel. Without the overrides the resolver reaches for
    the sdist and builds a C extension on a host that exists to avoid exactly that.
    """

    def _invoke(
        self,
        tmp_path: pathlib.Path,
        *,
        is_woa: bool = True,
        preset: str = "",
    ) -> dict:
        script = _script(
            substep_collector("Warnings"),
            f"$StudioHome = '{tmp_path}'",
            f"function Test-WinArm64Venv {{ ${str(is_woa).lower()} }}",
            preset,
            functions(SETUP_SRC, "Get-UvSafePath", "Restore-WoaResolverEnvironment"),
            "Restore-WoaResolverEnvironment",
            "[pscustomobject]@{",
            "  ov = $env:UV_OVERRIDE; uvfl = $env:UV_FIND_LINKS; pipfl = $env:PIP_FIND_LINKS",
            "  warned = ($script:Warnings -join ' ')",
            "} | ConvertTo-Json -Compress",
        )
        # The function assigns real environment variables; keep them out of the parent.
        done = _ps_ok(
            script,
            env = {**os.environ, "UV_OVERRIDE": "", "UV_FIND_LINKS": "", "PIP_FIND_LINKS": ""},
        )
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
        got = self._invoke(tmp_path, preset = "$env:UV_OVERRIDE = 'C:\\caller\\ov.txt'")
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
        # Ours is PREPENDED rather than skipped, and the caller's entry still has to survive.
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
        install, setup = _ps_copies("Get-UvSafePath")
        assert install == setup

    def test_the_dependency_that_makes_this_necessary_is_still_there(self):
        """If studio.txt ever drops ddgs, this restore stops being load-bearing for brotli."""
        studio_txt = PACKAGE_ROOT / "studio" / "backend" / "requirements" / "studio.txt"
        assert "ddgs" in studio_txt.read_text(encoding = "utf-8")

    def test_the_restore_runs_before_the_dependency_pass(self):
        """After it, the brotli resolve has already been attempted."""
        restore = SETUP_SRC.index("\nRestore-WoaResolverEnvironment")
        stack = SETUP_SRC.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert restore < stack


class TestTheRecoveryReachesEveryModeThatNeedsIt:
    """Placement, which is what decided whether the two recoveries above fire at all."""

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
        blocks = self._enclosing_blocks(SETUP_SRC, "\nRestore-WoaResolverEnvironment")
        assert not any("NoTorchMode" in b for b in blocks), (
            "UNSLOTH_NO_TORCH=1 still installs studio.txt, and ddgs -> httpx[brotli] -> "
            f"Brotli has no win_arm64 wheel. Enclosing blocks: {blocks}"
        )

    def test_the_index_re_export_is_not_trapped_either(self):
        blocks = self._enclosing_blocks(
            SETUP_SRC, "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex"
        )
        assert not any(
            "NoTorchMode" in b for b in blocks
        ), f"the manifest is rewritten in no-torch mode too. Enclosing blocks: {blocks}"

    def test_the_recovered_index_is_put_back_in_the_environment(self):
        """The bug this guards: recovering the index into a local variable only."""
        assign = SETUP_SRC.index("$WinArm64TorchIndexUrl = if (")
        export = SETUP_SRC.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        stack = SETUP_SRC.index('python "$PSScriptRoot\\install_python_stack.py"')
        assert assign < export < stack, "recovered, re-exported, then read by the stack"
        block = SETUP_SRC.rindex("if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {", 0, export)
        assert (
            "$_woaMarkerIndex = $_woaPinnedIndex" in SETUP_SRC[block:export]
        ), "guarded: neither record present must not export an empty value"

    def test_studio_txt_is_installed_in_no_torch_mode(self):
        """The premise of the placement test above."""
        call = STACK_SRC.index('req = REQ_ROOT / "studio.txt"')
        line_start = STACK_SRC.rfind("\n", 0, STACK_SRC.rindex("pip_install(", 0, call)) + 1
        indent = len(STACK_SRC[line_start:]) - len(STACK_SRC[line_start:].lstrip())
        assert (
            indent == 4
        ), "the studio.txt install is no longer unconditional inside install_python_stack()"
        skip_list = STACK_SRC[STACK_SRC.index("NO_TORCH_SKIP_PACKAGES = {") :][:400]
        assert "ddgs" not in skip_list, "ddgs is still installed when NO_TORCH is set"


class TestTheRecoveryHappensWhileThereIsStillSomethingToRead:
    """Ordering against the manifest drop, which decides whether any of this works."""

    def test_the_index_is_read_before_the_manifest_is_deleted(self):
        read = SETUP_SRC.index("Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        drop = SETUP_SRC.index("install_manifest.remove_manifest()")
        assert read < drop, (
            "the recovery reads unsloth_install_manifest.json; after the drop there is "
            "nothing left to read and the fresh-shell path silently gets no index"
        )

    def test_both_still_sit_inside_the_dependency_guard(self):
        """No point recovering for a run that installs nothing."""
        guard = SETUP_SRC.index("if (-not $SkipPythonDeps) {")
        read = SETUP_SRC.index("Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        restore = SETUP_SRC.index("\nRestore-WoaResolverEnvironment")
        assert guard < read and guard < restore


class TestThePublishedIndexIsTheOneTorchCameFrom:
    """What install_python_stack.py is told to repair from."""

    def test_the_publish_block_reads_the_effective_index(self):
        assert "$_expectedLeaf = Get-TorchIndexLeaf $_effectiveTorchIndexUrl" in SETUP_SRC
        assert "$env:UNSLOTH_TORCH_INSTALL_INDEX_URL = $_effectiveTorchIndexUrl" in SETUP_SRC

    def test_it_defaults_to_the_old_value_and_only_torch_moves_it(self):
        """Every non-CUDA path must publish exactly what it published before."""
        default = SETUP_SRC.index("$_effectiveTorchIndexUrl = $TorchInstallIndexUrl")
        assign = SETUP_SRC.index("$_effectiveTorchIndexUrl = $_cudaIndexUrl")
        publish = SETUP_SRC.index("$_expectedLeaf = Get-TorchIndexLeaf $_effectiveTorchIndexUrl")
        assert default < assign < publish, "default, then the install, then the publish"
        assert (
            SETUP_SRC.count("$_effectiveTorchIndexUrl = ") == 2
        ), "only the CUDA install may move it"

    def test_the_nvidia_channel_publishes_no_flavor_tag(self):
        """Get-TorchIndexLeaf on the NVIDIA channel yields `nvtorch_oot`, which is not a CUDA
        family name, so the tag resolves to $null and nothing is published."""
        if PWSH is None:
            pytest.skip("pwsh not available")
        script = _script(
            functions(SETUP_SRC, "Get-TorchIndexLeaf", "Test-CudaFamilyLeaf"),
            f"$leaf = Get-TorchIndexLeaf '{NV_GA}'",
            'Write-Output "$leaf|$(Test-CudaFamilyLeaf $leaf)"',
        )
        leaf, is_cuda = _ps_last(script).split("|")
        assert leaf == "nvtorch_oot"
        assert is_cuda == "False", "a CUDA family leaf here would publish a wrong flavor"


class TestTheLlamaArm64CudaOptOut:
    """UNSLOTH_LLAMA_ARM64_CUDA=0, honoured on every path that can reach the branch."""

    @staticmethod
    def _arm64_nvidia_branches() -> list:
        """Every `if ...has_usable_nvidia...` whose enclosing branches select ARM64."""
        import ast

        tree = ast.parse(LLAMA_SRC)
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
        resolve_upstream_asset_choice, and resolve_asset_choice's published-artifact branch."""
        branches = self._arm64_nvidia_branches()
        assert len(branches) >= 3, branches
        ungated = [b for b in branches if "_upstream_arm64_cuda_allowed" not in b[1]]
        assert not ungated, f"ungated ARM64 CUDA branch(es): {ungated}"

    def test_the_x64_paths_are_not_gated_by_the_arm64_opt_out(self):
        """The negative control: this flag must not disable CUDA on ordinary hardware."""
        import ast

        tree = ast.parse(LLAMA_SRC)
        arm64_lines = {line for line, _ in self._arm64_nvidia_branches()}
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.lineno not in arm64_lines:
                test = ast.unparse(node.test)
                if "has_usable_nvidia" in test:
                    assert "_upstream_arm64_cuda_allowed" not in test, test

    def test_the_published_artifact_branch_is_gated_too(self):
        """The published windows-arm64-cuda branch returns before the unverified-upstream tail,
        so gating only the tail meant the opt-out worked right up until the fork published an
        approved artifact, then silently stopped."""
        start = LLAMA_SRC.index("def resolve_asset_choice(")
        body = LLAMA_SRC[start:]
        marker = body.index("host.is_windows and host.is_arm64")
        branch = body[marker : marker + 4000]
        gate = branch.index("if host.has_usable_nvidia")
        published = branch.index("published_windows_cuda_attempts(")
        assert "_upstream_arm64_cuda_allowed()" in branch[gate : gate + 120]
        assert gate < published, "the gate must precede the published lookup"

    def test_the_now_unreachable_inner_check_is_gone(self):
        """With the branch gated, a second test inside it could only ever be true."""
        start = LLAMA_SRC.index("def resolve_asset_choice(")
        assert "if _upstream_arm64_cuda_allowed():" not in LLAMA_SRC[start:]

    def test_the_docstring_matches_the_scope(self):
        """The helper documented itself as upstream-only; it now gates every bundle."""
        start = LLAMA_SRC.index("def _upstream_arm64_cuda_allowed(")
        doc = LLAMA_SRC[start : LLAMA_SRC.index('"""', LLAMA_SRC.index('"""', start) + 3)]
        assert "published or upstream" in doc

    def test_the_upstream_resolver_branch_specifically(self):
        start = LLAMA_SRC.index("def resolve_upstream_asset_choice(")
        end = LLAMA_SRC.index("\ndef ", start + 10)
        body = LLAMA_SRC[start:end]
        marker = body.index("if host.is_windows and host.is_arm64:")
        arm64_block = body[marker : marker + 900]
        assert (
            "_upstream_arm64_cuda_allowed()" in arm64_block
        ), "the Windows ARM64 CUDA branch of resolve_upstream_asset_choice is ungated"


class TestAMigratedX64VenvIsRebuiltAsArm64:
    """install.ps1: an upgrade must not leave a WoA NVIDIA host on the emulated stack."""

    REBUILD = "$script:WoaNativeCudaTorch -and $_Migrated"

    def _rebuild_block(self) -> str:
        return INSTALL_SRC[INSTALL_SRC.index(self.REBUILD) :][:1800]

    def test_the_rebuild_runs_before_venv_creation(self):
        """It works by making $VenvPython absent, so the existing creation block builds an
        ARM64 venv."""
        rebuild = INSTALL_SRC.index(self.REBUILD)
        create = INSTALL_SRC.index("if (-not (Test-Path -LiteralPath $VenvPython)) {")
        assert rebuild < create

    def test_it_preserves_the_old_environment(self):
        """Same rollback the new-layout branch uses; the user's packages are recoverable."""
        assert "Start-StudioVenvRollback -ExistingDir $VenvDir" in self._rebuild_block()

    def test_a_failed_rollback_keeps_the_old_behaviour(self):
        """Losing the user's environment is never worth a native stack."""
        block = self._rebuild_block()
        assert "} catch {" in block
        assert "using the x64 stack instead" in block

    def test_the_rebuild_clears_the_migrated_flag(self):
        """The regression that would otherwise follow: $_Migrated drives an upgrade-in-place far
        below, which installs unsloth with --no-deps and --reinstall-package."""
        block = self._rebuild_block()
        rollback = block.index("Start-StudioVenvRollback")
        cleared = block.index("$_Migrated = $false")
        assert rollback < cleared, "cleared only after the environment is safely moved"
        # And the flag really does still gate that path.
        assert "if ($_Migrated) {" in INSTALL_SRC

    def test_it_only_touches_a_venv_this_run_migrated(self):
        """A new-layout venv was already moved aside above, and a venv created moments ago came
        from the interpreter this run chose."""
        line = INSTALL_SRC[INSTALL_SRC.index(f"if ({self.REBUILD}") :].split("\n")[0]
        assert "$_Migrated" in line
        assert "Test-Path -LiteralPath $VenvPython" in line

    def test_the_platform_guard_below_still_has_the_final_say(self):
        """Belt and braces: if the rollback failed the venv is still x64, and the existing guard
        must disable native mode rather than install win_arm64-only specs into it."""
        rebuild = INSTALL_SRC.index(self.REBUILD)
        guard = INSTALL_SRC.index('if ($_woaVenvPlatform -ne "win-arm64") {')
        assert rebuild < guard
        block = INSTALL_SRC[guard:][:600]
        assert "$script:WoaNativeCudaTorch = $false" in block
        assert "$script:WoaTorchIndexUrl = $null" in block


# The live `if`/`else` that decides which llama.cpp bundles the mismatch check will accept.
OPT_OUT_KINDS = slice_between(
    SETUP_SRC,
    "$_arm64CudaOptOut =",
    '} else { @("windows-cuda", "windows-vulkan") }',
    include_end = True,
).strip()

CUDA = "windows-arm64-cuda,windows-arm64,windows-vulkan"
NO_CUDA = "windows-arm64,windows-vulkan"


class TestTheOptOutBundleSurvivesTheKindCheck:
    """setup.ps1's mismatch check must expect the bundle the selector actually installs."""

    @staticmethod
    def _kinds(
        value: str,
        *,
        arm64: bool = True,
        separator: str = ",",
    ) -> str:
        script = _script(
            f"function Test-WinArm64Venv {{ ${str(arm64).lower()} }}",
            OPT_OUT_KINDS,
            f"Write-Output ($_nvidiaKinds -join '{separator}')",
        )
        return _ps_last(script, env = {**os.environ, "UNSLOTH_LLAMA_ARM64_CUDA": value})

    @requires_pwsh
    @pytest.mark.parametrize(
        "value, expected",
        [
            # Not opted out: CUDA is preferred, and the CPU fallback bundle is valid too.
            ("", CUDA),
            ("1", CUDA),
            ("true", CUDA),
            ("0", NO_CUDA),
            ("false", NO_CUDA),
            ("no", NO_CUDA),
            ("off", NO_CUDA),
            ("OFF", NO_CUDA),
            (" 0 ", NO_CUDA),
        ],
    )
    def test_the_expected_kind_follows_the_opt_out(self, value: str, expected: str):
        assert self._kinds(value) == expected

    @requires_pwsh
    def test_an_x64_venv_is_unaffected_by_the_flag(self):
        """The flag is ARM64-only; an emulated x64 venv installs windows-cuda regardless."""
        for value in ("", "0"):
            assert self._kinds(value, arm64 = False) == "windows-cuda,windows-vulkan"

    @requires_pwsh
    def test_the_opt_out_arm_stays_exclusive(self):
        """A CUDA bundle installed before the flag was set must still be replaced by the one the
        flag asks for, so the opt-out arm expects the CPU kind INSTEAD of the CUDA kind."""
        assert (
            self._kinds("0", separator = " ") == "windows-arm64 windows-vulkan"
        ), "opted out: CUDA is no longer valid"
        assert self._kinds("", separator = " ") == "windows-arm64-cuda windows-arm64 windows-vulkan"

    def test_the_cpu_fallback_is_a_real_selector_outcome(self):
        """The premise of widening: resolve_asset_choice falls through to the published
        windows-arm64 bundle when no ARM64 CUDA asset is available on an NVIDIA host."""
        start = LLAMA_SRC.index("def resolve_asset_choice(")
        body = LLAMA_SRC[start:]
        marker = body.index("host.is_windows and host.is_arm64")
        assert (
            'published_asset_choice_for_kind(release, "windows-arm64")'
            in body[marker : marker + 5200]
        )

    def test_widening_does_not_strand_anyone_on_cpu(self):
        """The installer's already-satisfied short-circuit is per candidate, and CUDA is
        attempted first, so a CPU bundle accepted here is still replaced the day an ARM64 CUDA
        asset appears."""
        raise_at = LLAMA_SRC.index("raise ExistingInstallSatisfied(attempt, tried_fallback)")
        window = LLAMA_SRC[max(0, raise_at - 1200) : raise_at]
        assert (
            "choice = attempt" in window
        ), "the reuse check is per attempt; a plan-level one would pin the user to CPU"

    def test_the_falsy_spellings_match_the_python_helper(self):
        """One vocabulary; the two must not drift apart."""
        start = LLAMA_SRC.index("def _upstream_arm64_cuda_allowed(")
        body = LLAMA_SRC[start : LLAMA_SRC.index("\ndef ", start + 10)]
        python_set = set(re.findall(r'"(0|false|no|off)"', body))
        ps_line = SETUP_SRC[SETUP_SRC.index("$_arm64CudaOptOut =") :].split("\n")[0]
        ps_set = set(re.findall(r'"(0|false|no|off)"', ps_line))
        assert python_set == ps_set == {"0", "false", "no", "off"}


# The CUDA wheel scan, with the tag matcher it calls. Driven against synthetic PEP 503 pages
# with Invoke-RestMethod stubbed, so these are offline and deterministic; the live NVIDIA
# channels are exercised separately in temp/sim10282/probe_test.ps1.
CUDA_PROBE_FUNCS = _script(
    WHEEL_TAG_FUNCS, _function_source(INSTALL_SRC, "Get-WoaCudaWheelVersion")
)


def _wheel_links(*names: str) -> str:
    """A PEP 503 index page listing exactly these wheels."""
    return " ".join(f'<a href="{name}">t</a>' for name in names)


def _torch_links(
    *versions: str,
    py: str = "cp313",
    abi: str = "cp313",
) -> str:
    return _wheel_links(*(f"torch-{v}-{py}-{abi}-win_arm64.whl" for v in versions))


class TestTheCudaWheelProbeIsNotFooled:
    """install.ps1: what the probe accepts as proof of a win_arm64 CUDA wheel."""

    @staticmethod
    def _probe(
        body: str,
        project: str = "torch",
        minor: str = "3.13",
    ) -> str:
        script = _script(
            JOIN_URL_PATH,
            invoke_restmethod(body),
            CUDA_PROBE_FUNCS,
            f"$v = Get-WoaCudaWheelVersion -IndexUrl 'https://x.test/i' -PythonMinor '{minor}'"
            f" -Project '{project}'",
            'Write-Output "[$v]"',
        )
        return _ps_last(script)[1:-1]

    @requires_pwsh
    def test_a_percent_encoded_cpu_wheel_is_rejected(self):
        """PEP 503 hrefs encode `+` as %2B. Matching a literal `\\+cpu` never fired on the
        encoded spelling, so a CPU-only mirror read as CUDA and the host went native on CPU
        torch -- worse than staying emulated, because the GPU then goes unused."""
        assert self._probe(_torch_links("2.14.0%2Bcpu")) == ""

    @requires_pwsh
    def test_an_untagged_wheel_is_rejected(self):
        """PyPI's own win_arm64 torch wheels carry no local version at all."""
        assert self._probe(_torch_links("2.14.0")) == ""

    @requires_pwsh
    @pytest.mark.parametrize("spelling", ["2.14.0%2Bcu134", "2.14.0+cu134"])
    def test_both_spellings_of_a_cuda_wheel_are_accepted(self, spelling: str):
        assert self._probe(_torch_links(spelling)) == "2.14.0+cu134"

    @requires_pwsh
    def test_the_interpreter_tag_still_has_to_match(self):
        body = _torch_links("2.14.0%2Bcu134", py = "cp311", abi = "cp311")
        assert self._probe(body, minor = "3.13") == ""
        assert self._probe(body, minor = "3.11") == "2.14.0+cu134"

    @requires_pwsh
    def test_the_platform_still_has_to_match(self):
        body = '<a href="torch-2.14.0%2Bcu134-cp313-cp313-win_amd64.whl">t</a>'
        assert self._probe(body) == ""

    @requires_pwsh
    def test_the_newest_release_wins(self):
        body = _torch_links("2.9.0%2Bcu134", "2.14.0%2Bcu134", "2.11.0%2Bcu134")
        assert self._probe(body) == "2.14.0+cu134"

    @requires_pwsh
    def test_a_dev_stamp_does_not_look_older_than_a_release(self):
        """2.15.0.dev... is newer than 2.14.0; a plain string sort would disagree."""
        body = _torch_links("2.14.0%2Bcu134", "2.15.0.dev20260819%2Bcu134")
        assert self._probe(body) == "2.15.0.dev20260819+cu134"

    @requires_pwsh
    def test_the_newest_dev_stamp_of_one_release_wins(self):
        body = _torch_links("2.15.0.dev20260819%2Bcu134", "2.15.0.dev20260728%2Bcu134")
        assert self._probe(body) == "2.15.0.dev20260819+cu134"

    @requires_pwsh
    def test_an_empty_or_broken_page_is_not_a_wheel(self):
        assert self._probe("") == ""
        assert self._probe("<html><body>nothing here</body></html>") == ""


class TestTorchaudioIsOnlyTakenAsAMatchedPair:
    """The GA channel publishes torch 2.14.0+cu134 beside torchaudio 2.11.0+cu134."""

    @staticmethod
    def _match(torch_v: str, audio_v: str) -> bool:
        script = _script(
            _ps_function(INSTALL_PS1, "Test-WoaAudioMatchesTorch"),
            f"Write-Output (Test-WoaAudioMatchesTorch -TorchVersion '{torch_v}'"
            f" -AudioVersion '{audio_v}')",
        )
        return _ps_last(script) == "True"

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
        # rindex, not index: the first occurrence is the reset at the top of the probe.
        block = INSTALL_SRC[INSTALL_SRC.rindex("$script:WoaTorchAudio = ") :][:400]
        assert "Test-WoaAudioMatchesTorch" in block, (
            "torchaudio was enabled on existence alone, which is how the mismatched "
            "GA pair became installable"
        )


# The live `if` that builds the resolver flags for the WoA torch install.
INDEX_ARGS_BLOCK = slice_between(
    SETUP_SRC, "$WinArm64IndexArgs = if (", "} else { @() }", include_end = True
)


class TestPrereleasesAreOnlyForTheNightlyChannel:
    """setup.ps1 must gate --prerelease=allow the way install.ps1 already does. The URL spelling
    is the SECOND signal now: install.ps1 reads the answer off the wheel it probed and hands it
    over, because a mirror of a prerelease-only channel need not say "nightly"."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "index, handover, expect_pre",
        [
            (NV_GA, "0", False),
            (NV_NIGHTLY, "0", True),
            ("", "0", False),
            ("https://mirror.test/simple", "1", True),
            ("https://mirror.test/simple", "0", False),
        ],
    )
    def test_the_flag_follows_the_channel(self, index: str, handover: str, expect_pre: bool):
        script = _script(
            "$WinArm64Venv = $true",
            "$UseUv = $true",
            f"$WinArm64TorchIndexUrl = '{index}'",
            f"$WinArm64EffectiveTorchIndexUrl = '{index}'",
            f"$WinArm64HandoffApplies = ${bool(index)}",
            f"$env:UNSLOTH_WOA_TORCH_PRERELEASE = '{handover}'",
            INDEX_ARGS_BLOCK,
            "Write-Output ($WinArm64IndexArgs -join ' ')",
        )
        out = _ps_last(script)
        assert ("--prerelease=allow" in out) is expect_pre, out
        assert "unsafe-best-match" in out, "the other flags are unconditional"

    @requires_pwsh
    def test_every_other_host_gets_no_flags_at_all(self):
        script = _script(
            "$WinArm64Venv = $false",
            "$UseUv = $true",
            f"$WinArm64TorchIndexUrl = '{NV_NIGHTLY}'",
            "$WinArm64EffectiveTorchIndexUrl = $WinArm64TorchIndexUrl",
            "$WinArm64HandoffApplies = $false",
            INDEX_ARGS_BLOCK,
            "Write-Output \"[$($WinArm64IndexArgs -join ' ')]\"",
        )
        assert _ps_last(script) == "[]"

    def test_both_scripts_gate_on_the_same_thing(self):
        """One rule; two files. Drift here is invisible until a resolve goes wrong."""
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert re.search(r"-match 'nightly'", text), f"{path.name} lost the gate"
            for line in text.splitlines():
                if "--prerelease=allow" not in line or "#" in line.split("--prerelease")[0]:
                    continue
                # Producing the flag is what has to stay behind the gate, so only an @(...)
                # will do. Comparing against it does not produce it: that is
                # Remove-UvOnlyResolverFlags translating an argument the gate already allowed
                # into pip's spelling, and it runs after the decision, not instead of it.
                if "-eq '--prerelease=allow'" in line:
                    continue
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
    @pytest.mark.parametrize("url", [NV_GA, NV_NIGHTLY])
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

    # A REAL archive: a PK header proves nothing about an interrupted download.
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
        return _ps_last(
            pyarrow_source_script(
                wheelhouse = "'https://example.test/wheels'",
                preamble = (f"$env:UNSLOTH_PYARROW_WHEEL = '{wheel}'",),
            )
        )[1:-1]


class TestCallerResolverConfigurationSurvives:
    """install.ps1 must not discard what its own purge block just chose to keep."""

    def test_the_overrides_are_kept_or_folded_never_dropped(self):
        block = INSTALL_SRC[INSTALL_SRC.index("$_woaOwnNames = @{}") :][:3200]
        assert "$env:UV_OVERRIDE -split" in block, "the caller's files are read"
        assert "$_woaKeepFiles += $_woaOvFull" in block, (
            "a file that names none of our packages is passed to uv where it is, which "
            "keeps its relative -r and wheel paths resolving"
        )
        assert (
            "$_woaSessionLines += (Resolve-WoaOverrideLine" in block
        ), "and a conflicting one is folded line by line, rebased as it goes"
        assert "$_woaOwnNames.ContainsKey($_woaOvName)" in block, (
            "minus the packages this file declares -- uv combines override files and "
            "errors on a duplicate package, so a blind append could fail the resolve"
        )

    def test_the_find_links_are_appended_with_the_right_separators(self):
        assert (
            '$env:UV_FIND_LINKS = if ($_woaCallerUvLinks) { "$WoaWheelDir,$_woaCallerUvLinks" }'
            in INSTALL_SRC
        ), "UV_FIND_LINKS is comma-separated"
        assert (
            '"$_woaSafeWheelDir $_woaCallerPipLinks"' in INSTALL_SRC
        ), "PIP_FIND_LINKS is split on whitespace, and ours must be the 8.3-safe form"

    def test_ours_is_searched_first(self):
        """A win_arm64 wheel staged for this host must win a tie against the same name."""
        assert '"$WoaWheelDir,$_woaCallerUvLinks"' in INSTALL_SRC
        assert '"$_woaCallerUvLinks,$WoaWheelDir"' not in INSTALL_SRC

    def test_the_python_side_can_read_an_appended_value(self, tmp_path):
        """install_python_stack.py split find-links on os.pathsep alone, which would have read
        "dirA,dirB" as one unusable path now that appending is possible. Asserted as behaviour
        because the separator is per-variable: a shared split that also broke on whitespace tore
        a directory whose name contains a space into two paths that do not exist."""
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
        script = _script(
            _ps_function(INSTALL_PS1, "Get-WoaAbiTag"),
            f"Write-Output (Get-WoaAbiTag -PythonMinor '{minor}' "
            f"-FreeThreaded ${str(free_threaded).lower()})",
        )
        assert _ps_last(script) == expected

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
        script = _script(
            JOIN_URL_PATH,
            invoke_restmethod(_torch_links("2.14.0%2Bcu134", abi = wheel_abi)),
            CUDA_PROBE_FUNCS,
            "$v = Get-WoaCudaWheelVersion -IndexUrl 'https://x.test/i' -PythonMinor '3.13'"
            f" -AbiTag '{abi}'",
            'Write-Output "[$v]"',
        )
        got = _ps_last(script)[1:-1]
        assert bool(got) is found, got

    def test_the_probe_takes_the_flag_and_the_call_sites_supply_it(self):
        assert (
            "function Initialize-WoaNativeCudaTorch {\n        param([string]$PythonMinor,"
            " [bool]$FreeThreaded = $false)" in INSTALL_SRC
        )
        # Both re-probes know which interpreter was chosen, so both must answer for it.
        # The flag is read into a variable first: a same-minor free-threaded build re-probes too.
        assert INSTALL_SRC.count("Test-PythonFreeThreaded -PythonExe $DetectedPython.Path") == 2
        assert "-FreeThreaded $WoaDetectedFreeThreaded" in INSTALL_SRC
        assert "-FreeThreaded $_woaNewFreeThreaded" in INSTALL_SRC

    def test_the_staging_scan_uses_the_venv_abi(self):
        """Keyed on the python tag, staging kept the cp313-cp313 wheels a free-threaded venv
        cannot install and discarded the cp313-cp313t ones it can."""
        assert "$WoaWheelAbi = Get-WoaAbiTag -PythonMinor $WoaVenvMinor" in INSTALL_SRC
        assert "($abiTags -contains $WoaWheelAbi)" in INSTALL_SRC
        assert (
            "($WoaWheelStable -and ($abiTags -contains 'abi3'))" in INSTALL_SRC
        ), "free-threaded builds do not implement the stable ABI"
        assert (
            "$script:WoaVenvFreeThreaded = Test-PythonFreeThreaded -PythonExe $VenvPython"
            in INSTALL_SRC
        )

    @requires_pwsh
    def test_an_unknown_interpreter_answers_gil(self):
        """The historical assumption: unknown must not turn a working GIL host free-threaded."""
        script = _script(
            _ps_function(INSTALL_PS1, "Test-PythonFreeThreaded"),
            "Write-Output (Test-PythonFreeThreaded -PythonExe 'C:\\nope\\python.exe')",
            "Write-Output (Test-PythonFreeThreaded -PythonExe '')",
        )
        assert _ps_ok(script).stdout.split() == ["False", "False"]

    @requires_pwsh
    def test_it_reads_the_real_interpreter_correctly(self):
        """Executed against the interpreter running this suite, whichever build that is."""
        import sysconfig

        expected = "True" if sysconfig.get_config_var("Py_GIL_DISABLED") else "False"
        script = _script(
            _ps_function(INSTALL_PS1, "Test-PythonFreeThreaded"),
            f"Write-Output (Test-PythonFreeThreaded -PythonExe '{sys.executable}')",
        )
        assert _ps_last(script) == expected


class TestTheAbiReprobeFiresOnAMatchingMinor:
    """The hole left by keying the re-probe on the minor alone."""

    def test_the_guard_compares_the_abi_as_well_as_the_minor(self):
        assert "$WoaProbedFreeThreaded = $false" in INSTALL_SRC
        assert (
            "($WoaDetectedFreeThreaded -ne $WoaProbedFreeThreaded)" in INSTALL_SRC
        ), "a 3.13t selected for a 3.13 request matches on minor and must still re-probe"

    def test_the_second_reprobe_compares_it_too(self):
        """After Install-PythonFromPythonOrg the ABI can change without the minor doing so."""
        assert "($_woaNewFreeThreaded -ne $WoaProbedFreeThreaded)" in INSTALL_SRC

    def test_the_detection_is_scoped_to_this_host(self):
        """A subprocess per run on every Windows x64 host would buy nothing."""
        block = INSTALL_SRC[INSTALL_SRC.index("$WoaDetectedFreeThreaded = $false") :][:600]
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
        script = _script(
            f"$WoaProbedMinor = '{probed_minor}'",
            f"$WoaProbedFreeThreaded = ${str(probed_ft).lower()}",
            f"$DetectedPython = @{{ Version = '{minor}' }}",
            f"$WoaDetectedFreeThreaded = ${str(ft).lower()}",
            "if ($DetectedPython -and (",
            "        ($DetectedPython.Version -ne $WoaProbedMinor) -or",
            "        ($WoaDetectedFreeThreaded -ne $WoaProbedFreeThreaded))) {",
            "  Write-Output 'REPROBE' } else { Write-Output 'SKIP' }",
        )
        assert (_ps_last(script) == "REPROBE") is should_reprobe, why


class TestAnExplicitPinOutranksThePersistedIndex:
    """The recovery is a memory of what install.ps1 chose, not a decision."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, woa, install_url, expected, why",
        [
            ("", NV_GA, "https://d.pytorch.org/whl/cu130", NV_GA, "unpinned fresh shell"),
            (
                "https://mirror.test/cu129",
                NV_GA,
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
        script = _script(
            f"$PinnedTorchIndexUrl = '{pinned}'",
            f"$WinArm64TorchIndexUrl = '{woa}'",
            f"$TorchInstallIndexUrl = '{install_url}'",
            slice_between(
                SETUP_SRC,
                "$_cudaIndexUrl = if ($PinnedTorchIndexUrl)",
                "else { $TorchInstallIndexUrl }",
                include_end = True,
            ).strip(),
            "Write-Output $_cudaIndexUrl",
        )
        assert _ps_last(script) == expected, why


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
        script = _script(
            _function_source(SETUP_SRC, "Get-RequirementName"),
            f"Write-Output \"[$(Get-RequirementName -Line '{line}')]\"",
        )
        assert _ps_last(script) == f"[{expected_name}]"

    def test_disjoint_files_are_both_passed_and_conflicts_are_merged(self):
        block = SETUP_SRC[SETUP_SRC.index("$_woaOursNames = Get-RequirementNames") :][:2600]
        assert '$env:UV_OVERRIDE = "$safeOverrides $($env:UV_OVERRIDE)"' in block, (
            "disjoint files need no rewriting, which keeps each file's relative "
            "references resolving against its own directory"
        )
        assert "overrides.merged.txt" in block, "only an actual conflict is merged"
        assert "(Get-RequirementName -Line $_woaOvLine) -in $_woaOursNames" not in block

    def test_the_caller_no_longer_suppresses_the_drop_list(self):
        """The regression: the whole restore used to sit under `if (-not $env:UV_OVERRIDE)`."""
        body = _function_source(SETUP_SRC, "Restore-WoaResolverEnvironment")
        # The drop list is read whatever the caller set; only the LAST step reads UV_OVERRIDE.
        head = body[: body.index("$safeOverrides = Get-UvSafePath $overrides")]
        assert head.count("$env:UV_OVERRIDE") <= 1, (
            "reaching the drop list must not depend on the caller having set nothing; "
            f"head still tests it: {head[-300:]}"
        )
        assert (
            "elseif (-not $env:UV_OVERRIDE) {" in body
        ), "the no-caller case still assigns ours alone"


# The live purge that drops a PREVIOUS run's resolver settings but not the caller's.
PURGE_BLOCK = slice_between(
    INSTALL_SRC,
    '$_woaOwnedPrefix = Join-Path $StudioHome "woa"',
    "if ($script:WoaNativeCudaTorch) {",
)


class TestThePurgeKeepsWhatIsNotOurs:
    """The purge drops a PREVIOUS run's resolver settings, not the caller's."""

    def test_the_block_works_entry_by_entry(self):
        assert "$_woaKept" in PURGE_BLOCK, "entries are kept, not just the whole value dropped"
        assert '"UV_FIND_LINKS" = ","' in PURGE_BLOCK, "each is rejoined with its own separator"
        assert "$_woaSplitOn" in PURGE_BLOCK, (
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
                "the whitespace form",
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
        script = _script(
            f"$StudioHome = '{FAKE_HOME}'",
            UV_SAFE_PATH,
            f"$env:{var} = '{_native_path_value(value)}'",
            PURGE_BLOCK,
            f"Write-Output ('[' + $env:{var} + ']')",
        )
        assert _ps_last(script) == f"[{_native_path_value(expected)}]", why


class TestAMalformedHandoffUrlDoesNotCostTheManifest:
    """write_manifest documents that it never raises."""

    @pytest.mark.parametrize(
        "bad", ["https://[", "https://pypi.nvidia.com:notaport/x", "https://[::1", "://"]
    )
    def test_the_manifest_is_still_written(self, tmp_path, bad):
        module = _load_manifest_module()
        written = module.write_manifest(tmp_path, woa_torch_index = bad)
        assert written is not None, "a bad URL must not take the manifest with it"
        payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload, "and it is certainly not persisted"

    def test_a_good_url_is_still_persisted(self, tmp_path):
        module = _load_manifest_module()
        written = module.write_manifest(tmp_path, woa_torch_index = NV_GA + "/")
        payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert payload["woa_torch_index"] == NV_GA


class TestAStaleTorchaudioIsRemoved:
    """A fresh-shell update on Windows on ARM cannot see UNSLOTH_WOA_HAS_TORCHAUDIO, so it
    installs torch/torchvision alone."""

    @staticmethod
    def _block() -> str:
        # Anchored on the code that follows rather than on its comment.
        return slice_between(
            SETUP_SRC,
            "if ($WinArm64Venv -and $WinArm64NoAudio) {",
            'substep "installing Triton for Windows..."',
        )

    def test_the_removal_is_conditional_on_a_mismatch(self):
        block = self._block()
        assert "Fast-Uninstall torchaudio" in block
        assert (
            "Test-WoaPairsWithTorchParity -TorchVersion $_woaTorchVer -OtherVersion $_woaAudioVer"
            in block
            and "Test-WoaAudioMatchesTorchParity -TorchVersion $_woaTorchVer"
            " -AudioVersion $_woaAudioVer"
            in block
        ), "a matching audio wheel is the one this venv was installed with and stays"
        assert (
            "$_woaAudioProbe.Ok" in block
        ), "a probe that did not answer says nothing; it must not trigger a removal"

    def test_it_only_runs_where_audio_was_dropped(self):
        assert self._block().startswith("if ($WinArm64Venv -and $WinArm64NoAudio) {"), (
            "on every other host, and on a WoA index that DOES publish torchaudio, the "
            "trio asked for it and the resolver already matched the pair"
        )

    def test_it_runs_after_the_install_succeeded(self):
        failed = SETUP_SRC.index('Exit-SetupFailure "PyTorch CUDA installation failed')
        assert (
            SETUP_SRC.index("if ($WinArm64Venv -and $WinArm64NoAudio) {") > failed
        ), "removing audio before knowing torch installed would strip a working venv"


class TestTheWheelTagsAreMatchedAsFields:
    """`*cp313-cp313*` also matches cp313-cp313t."""

    def test_no_substring_match_is_left(self):
        assert "$tag-$AbiTag*" not in INSTALL_SRC, "every probe goes through the field parser"
        # The six pyarrow sites go through Test-WoaPyarrowWheelUsable, a floor check around this.
        assert (
            INSTALL_SRC.count("Test-WoaWheelTags") + INSTALL_SRC.count("Test-WoaPyarrowWheelUsable")
        ) >= 7, (
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
        script = _script(
            WHEEL_TAG_FUNCS,
            f"Write-Output (Test-WoaWheelTags -Name '{name}' -PyTag '{py}' -AbiTag '{abi}')",
        )
        assert _ps_last(script) == str(expected), why


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
        script = _script(
            f"$StudioHome = '{FAKE_HOME}'",
            UV_SAFE_PATH,
            f"$env:UV_FIND_LINKS = '{_native_path_value(value)}'",
            PURGE_BLOCK,
            "Write-Output ('[' + $env:UV_FIND_LINKS + ']')",
        )
        assert _ps_last(script) == f"[{_native_path_value(expected)}]", why


class TestAHostedDropCandidateMustMeetItsFloor:
    """The override drop list recorded names only."""

    def test_versions_are_recorded_and_the_floor_is_consulted(self):
        assert "$WoaWheelNames[$_woaWheelKey] += $parts[1]" in INSTALL_SRC, "versions are kept"
        assert '$WoaDropFloors = @{ "xformers" = "0.0.22.post7" }' in INSTALL_SRC
        assert "Test-WoaVersionAtLeast -Version $_woaHostedVer -Floor $_woaFloor" in INSTALL_SRC

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
            # PEP 440 hangs .devN off whatever precedes it, so a development release sorts BELOW
            # that segment. Read as a pre-release of the RELEASE it got the post case backwards.
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
        script = _script(
            _function_source(INSTALL_SRC, "Test-WoaVersionAtLeast"),
            f"Write-Output (Test-WoaVersionAtLeast -Version '{have}' -Floor '{floor}')",
        )
        assert _ps_last(script) == expected, why

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
        start = INSTALL_SRC.index("        foreach ($candidate in $WoaDropCandidates) {")
        end = INSTALL_SRC.index('$WoaOverrideLines += "$candidate ; platform_machine', start)
        end = INSTALL_SRC.index("}", INSTALL_SRC.index("\n", end)) + 1
        wheel_names = "@{}" if not hosted else "@{ 'xformers' = @(%s) }" % hosted
        script = _script(
            _function_source(INSTALL_SRC, "Test-WoaVersionAtLeast"),
            SUBSTEP_NOOP,
            "$WoaDropCandidates = @('xformers')",
            '$WoaDropFloors = @{ "xformers" = "0.0.22.post7" }',
            f"$WoaWheelNames = {wheel_names}",
            "$WoaOverrideLines = @()",
            "$WoaReported = @{}",
            INSTALL_SRC[start:end],
            "Write-Output ('[' + ($WoaOverrideLines -join '|') + ']')",
        )
        line = _ps_last(script)
        assert ("xformers" in line) is dropped, f"{why}: {line}"


class TestAFreeThreadedInterpreterIsPreflightedForAv:
    """torch and pyarrow were not the whole story."""

    def test_the_probe_gates_native_mode(self):
        block = INSTALL_SRC[INSTALL_SRC.index("$pyarrowSource = Get-WoaPyarrowSource") :][:2200]
        assert '$FreeThreaded -and -not (Test-WoaWheelAvailable -Project "av"' in block, (
            "asked only on a free-threaded build; a GIL one takes the abi3 wheel and "
            "must not gain a network call or a new way to fail"
        )
        assert block.index("Test-WoaWheelAvailable") < block.index(
            "$script:WoaNativeCudaTorch = $true"
        ), "before the commit, not after: the point is to keep the x64 fallback"

    def test_the_constraint_that_makes_this_matter_is_still_there(self):
        assert (
            'av>=17.0.0; sys_platform == "win32" and platform_machine == "ARM64"' in CONSTRAINTS_SRC
        )

    @staticmethod
    def _available(rest_method: str, minor: str, abi: str) -> str:
        script = _script(
            rest_method,
            "$script:WoaWheelhouse = $null",
            functions(
                INSTALL_SRC,
                "Test-WoaWheelTags",
                "Test-WoaWheelTagsUsable",
                "Test-WoaVersionAtLeast",
                "Test-WoaPyPIWheel",
            ),
            "function Test-WoaResolveReachesPyPI { $true }",
            _function_source(INSTALL_SRC, "Test-WoaWheelAvailable"),
            f"Write-Output (Test-WoaWheelAvailable -Project 'av' -PythonMinor '{minor}'"
            f" -AbiTag '{abi}')",
        )
        return _ps_last(script)

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
        body = f'<a href="{listing}">a</a>' if listing else "<html></html>"
        minor = "3.14" if "314" in abi else "3.13"
        assert self._available(invoke_restmethod(body), minor, abi) == expected, why

    @requires_pwsh
    def test_an_unreachable_index_answers_no(self):
        """The x64 stack still works; a native venv that cannot build PyAV does not."""
        assert self._available(INVOKE_RESTMETHOD_OFFLINE, "3.13", "cp313t") == "False"


class TestACallerOverrideFileKeepsItsOwnDirectory:
    """uv resolves a nested -r, and a relative wheel path, against the file that contains the
    line."""

    def test_a_non_conflicting_file_is_passed_through(self):
        block = INSTALL_SRC[INSTALL_SRC.index("$_woaKeepFiles = @()") :][:2600]
        assert "$_woaKeepFiles += $_woaOvFull" in block, "kept where it is"
        assert "$_woaOvConflicts" in block, "only a package clash forces a rewrite"
        assert (
            "Resolve-WoaOverrideLine -Line $_woaOvEntry.Line -BaseDir $_woaOvEntry.BaseDir" in block
        ), "and a folded line has its relative references made absolute, against its own file"

    def test_the_kept_files_reach_uv(self):
        assert (
            "foreach ($_woaKeepFile in $_woaKeepFiles) { $_woaOverrideValue += "
            "(Get-UvSafePath $_woaKeepFile) }" in INSTALL_SRC
        )
        assert (
            '$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")' in INSTALL_SRC
        ), "uv splits UV_OVERRIDE on whitespace and combines the files"

    # The rewriter calls GetFullPath, and os.path.abspath applies the same rule on every host.
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
            ("dist/a.whl", "", "dist/a.whl", "a bare relative wheel path"),
        ],
    )
    def test_a_relative_reference_is_rebased(self, line, prefix, rebased, why):
        assert self._run(line) == prefix + self._rebased(rebased), why

    @requires_pwsh
    def test_a_relative_file_url_is_rebased_as_a_uri(self):
        """The file: rows cannot be checked by pasting a path after "file://": a rebased Windows
        path is C:\\opt\\corp\\ov\\dist\\a.whl, whose URL is file:///C:/opt/corp/ov/dist/a.whl.
        Concatenating only happened to match where the separator is already "/"."""
        expected = pathlib.Path(self._rebased("dist/a.whl")).as_uri()
        assert self._run("foo @ file:dist/a.whl") == f"foo @ {expected}"

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
        script = _script(
            _function_source(INSTALL_SRC, "Resolve-WoaOverrideLine"),
            "Write-Output ('[' + (Resolve-WoaOverrideLine -Line '{}' -BaseDir '{}') + ']')".format(
                line, TestACallerOverrideFileKeepsItsOwnDirectory.BASE
            ),
        )
        return _ps_last(script)[1:-1]

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
        start = INSTALL_SRC.index("        $_woaOwnNames = @{}")
        end = INSTALL_SRC.index('$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")', start)
        end = INSTALL_SRC.index("\n", end)
        caller_dir = tmp_path / "corp"
        caller_dir.mkdir()
        (caller_dir / "nested.txt").write_text("idna==3.10\n", encoding = "utf-8")
        caller = caller_dir / "ov.txt"
        caller.write_text("\n".join(caller_lines) + "\n", encoding = "utf-8")
        managed = tmp_path / "woa.txt"
        script = _script(
            # PowerShell does not hoist, so the scanner the block calls has to be here too.
            functions(INSTALL_SRC, "Resolve-WoaOverrideLine", "Get-WoaRequirementEntries"),
            UV_SAFE_PATH,
            "$WoaOverrideLines = @('# generated', 'torch>=2.4', 'torchvision>=0.19')",
            f"$WoaOverrides = '{managed}'",
            f"$env:UV_OVERRIDE = '{caller}'",
            INSTALL_SRC[start:end],
            'Write-Output ("OVERRIDE=" + $env:UV_OVERRIDE)',
        )
        done = _ps_ok(script)
        value = [line for line in done.stdout.splitlines() if line.startswith("OVERRIDE=")][-1][
            len("OVERRIDE=") :
        ].split()
        written = managed.read_text(encoding = "utf-8")
        session = tmp_path / "overrides.session.txt"
        if folded:
            assert value == [str(managed), str(session)], why
            # The include is FLATTENED as it folds, rebased against its own directory: the only
            # way a conflict one level down can be removed. It lands in the per-run file.
            folded_text = session.read_text(encoding = "utf-8")
            assert (
                "idna==3.10" in folded_text
            ), "the include's own lines did not come across, so folding dropped them"
            assert "-r " not in folded_text, "an include line copied verbatim would move its base"
            assert "torch==2.9.0" not in folded_text, "our own declaration still wins"
            assert "idna" not in written, "the persistent file carries only our own lines"
        else:
            assert value == [str(managed), str(caller)], why
            assert "-r nested.txt" not in written, "nothing was copied, so nothing moved"
            assert not session.exists(), "nothing folded, so no per-run file"


class TestTheRecoveryPrependsRatherThanStandsDown:
    """A caller's own find-links must not cost the staged win_arm64 wheels."""

    @staticmethod
    def _block() -> str:
        start = SETUP_SRC.index("    if (Test-Path -LiteralPath $wheels -PathType Container) {")
        return SETUP_SRC[start : SETUP_SRC.index("\n}", start)]

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
            ("PIP_FIND_LINKS", "/mnt/mirror", "/home/u/woa/wheels /mnt/mirror", "pip whitespace"),
            (
                "PIP_FIND_LINKS",
                "/home/u/woa/wheels",
                "/home/u/woa/wheels",
                "already present: unchanged",
            ),
        ],
    )
    def test_the_block(self, var, before, expected, why):
        script = _script(
            UV_SAFE_PATH,
            "$wheels = '/home/u/woa/wheels'",
            "Remove-Item Env:UV_FIND_LINKS,Env:PIP_FIND_LINKS -ErrorAction SilentlyContinue",
            (f"$env:{var} = '{before}'" if before else ""),
            self._block().replace(
                "if (Test-Path -LiteralPath $wheels -PathType Container) {", "if ($true) {", 1
            ),
            f"Write-Output ('[' + $env:{var} + ']')",
        )
        assert _ps_last(script) == f"[{expected}]", why


class TestTheMergedOverrideFileIsRebasedToo:
    """install.ps1 rebases a folded line; setup.ps1's merge had the same problem."""

    def test_the_merge_rebases(self):
        block = SETUP_SRC[SETUP_SRC.index("$_woaMerged = Join-Path $woaDir") :][:1600]
        assert "Resolve-WoaOverrideLine -Line $_woaEntry.Line -BaseDir $_woaEntry.BaseDir" in block
        assert "$_woaLines += $_woaLine" not in block, "no line is copied verbatim"

    def test_the_helper_is_a_faithful_copy_of_install_ps1s(self):
        """Neither script can dot-source the other, so the copy is pinned instead."""
        install, setup = _ps_copies("Resolve-WoaOverrideLine")
        assert install == setup


class TestAWheelhouseThatIsTheStagingDirectory:
    r"""UNSLOTH_WOA_WHEELHOUSE may BE $StudioHome\woa\wheels -- that is how an offline run
    reuses the installer's own cache."""

    def test_every_staging_copy_is_guarded(self):
        assert (
            "if (-not (Test-WoaSamePath $found.FullName $_woaDest)) {" in INSTALL_SRC
        ), "the wheelhouse pyarrow copy, which is not inside a try and so was fatal"
        assert (
            "if (-not (Test-WoaSamePath $wheel.FullName $_woaExtraDest)) {" in INSTALL_SRC
        ), "the extra-wheel loop, which swallowed the error but miscounted"
        assert "if (-not (Test-WoaSamePath $srcWheel $_woaLocalDest)) {" in INSTALL_SRC, (
            "and the supplied-wheel copy, where the failure was caught but disabled "
            "native mode after the ARM64 venv had already been chosen"
        )

    def test_no_staging_copy_is_left_unguarded(self):
        """Counted, so a fourth copy added later cannot quietly skip the guard."""
        # Anchored on code, not on a comment: a comment pass must not be able to break this.
        staging = slice_between(
            INSTALL_SRC,
            'if ($script:WoaPyarrowSource -eq "local") {',
            "$WoaOverrides = Join-Path $WoaDir",
        )
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
        script = _script(
            _function_source(INSTALL_SRC, "Test-WoaSamePath"),
            f"Write-Output (Test-WoaSamePath '{a}' '{b}')",
        )
        assert _ps_last(script) == expected, why

    @requires_pwsh
    def test_a_self_copy_would_otherwise_be_fatal(self, tmp_path):
        """The behaviour this guards, executed, so the reason cannot go stale."""
        wheel = tmp_path / "a.whl"
        wheel.write_text("x", encoding = "utf-8")
        done = _ps(
            '$ErrorActionPreference = "Stop"; '
            f"try {{ Copy-Item -LiteralPath '{wheel}' -Destination '{wheel}' -Force; "
            "Write-Output 'OK' } catch { Write-Output 'THREW' }"
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "THREW"


class TestTheSuppliedWheelIsOpenedNotSniffed:
    """A truncated download still starts with "PK"."""

    def test_the_zip_helper_is_used(self):
        block = INSTALL_SRC[INSTALL_SRC.index("if ($env:UNSLOTH_PYARROW_WHEEL) {") :][:1800]
        assert 'if (Test-ZipArchiveReadable -Path $_paWheel) { return "local" }' in block
        assert "ReadByte()" not in block, "the two-byte signature sniff is gone"

    def test_the_helper_is_defined_before_this_runs(self):
        """PowerShell does not hoist: a call above the definition is a runtime error."""
        assert INSTALL_SRC.index("function Test-ZipArchiveReadable") < INSTALL_SRC.index(
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
        script = _script(
            _function_source(INSTALL_SRC, "Test-ZipArchiveReadable"),
            f"Write-Output (Test-ZipArchiveReadable -Path '{wheel}')",
        )
        assert _ps_last(script) == expected, why


# The live chain that decides which index a WoA venv installs torch from.
WOA_INDEX_CHAIN = slice_between(
    SETUP_SRC, "$WinArm64TorchIndexUrl = if ($WinArm64Venv", '} else { "" }', include_end = True
)
CORP_WOA = "https://mirror.corp/woa"
NV_OOT = "https://pypi.nvidia.com/oot"


class TestAConfiguredWoaMirrorSurvivesAFreshShell:
    """write_manifest persists only NVIDIA's own channels, because any other URL could carry a
    credential, so UNSLOTH_WOA_TORCH_INDEX_URL has to be consulted on its own."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "configured, handover, persisted, expected, why",
        [
            (CORP_WOA, "", "", CORP_WOA, "the regression: a fresh shell with only the user's own"),
            (CORP_WOA + "/", "", "", CORP_WOA, "trailing slash trimmed, as the other branches do"),
            (CORP_WOA, NV_OOT, "", CORP_WOA, "the user's channel outranks the handover"),
            ("", NV_OOT, "", NV_OOT, "unchanged when it is not set"),
            ("", "", NV_OOT, NV_OOT, "and the manifest still answers when nothing else does"),
        ],
    )
    def test_the_precedence(self, configured, handover, persisted, expected, why):
        script = _script(
            "$WinArm64Venv = $true",
            "$VenvDir = '/nonexistent'",
            f"function Get-PersistedWoaTorchIndex {{ param($VenvPath) return '{persisted}' }}",
            f"$env:UNSLOTH_WOA_TORCH_INDEX_URL = '{configured}'",
            f"$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = '{handover}'",
            WOA_INDEX_CHAIN,
            "Write-Output ('[' + $WinArm64TorchIndexUrl + ']')",
        )
        assert _ps_last(script) == f"[{expected}]", why

    @requires_pwsh
    def test_a_non_arm64_venv_still_reads_nothing(self):
        """Every other host must see exactly the index choice it saw before."""
        script = _script(
            "$WinArm64Venv = $false",
            "$VenvDir = '/nonexistent'",
            "function Get-PersistedWoaTorchIndex { param($VenvPath) throw 'must not be called' }",
            f"$env:UNSLOTH_WOA_TORCH_INDEX_URL = '{CORP_WOA}'",
            f"$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = '{NV_OOT}'",
            WOA_INDEX_CHAIN,
            "Write-Output ('[' + $WinArm64TorchIndexUrl + ']')",
        )
        assert _ps_last(script) == "[]"

    def test_install_ps1_still_does_not_write_that_variable(self):
        """It is the user's INPUT."""
        assert not re.search(r"\$env:UNSLOTH_WOA_TORCH_INDEX_URL\s*=", INSTALL_SRC)

    def test_a_mirror_is_still_not_persisted(self):
        """The reason this branch has to exist; if the manifest ever took one, the credential
        rule would have been weakened instead."""
        module = _load_manifest_module()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            written = module.write_manifest(pathlib.Path(tmp), woa_torch_index = CORP_WOA)
            payload = json.loads(pathlib.Path(written).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload


class TestAWheelhousePyarrowMustClearTheFloor:
    """A tag-compatible pyarrow is not enough: staging turns whichever wheel it picks into an
    exact `pyarrow==<version>` override, and constraints.txt floors the ARM64 row at 21.0.0, so
    a 19.x wheel selected the native path and then made the dependency pass unsatisfiable."""

    def test_the_floor_matches_the_constraint(self):
        """Two places state it, so a bump to one that skips the other is caught here."""
        floor = re.search(r'\$script:WoaPyarrowFloor\s*=\s*"([^"]+)"', INSTALL_SRC)
        assert floor, "install.ps1 no longer declares the pyarrow floor"
        pinned = re.search(
            r'(?m)^pyarrow>=([0-9.]+);\s*sys_platform == "win32" and platform_machine == "ARM64"',
            CONSTRAINTS_SRC,
        )
        assert pinned, "the ARM64 pyarrow row is gone from constraints.txt"
        assert floor.group(1) == pinned.group(1), (
            f"install.ps1 floors pyarrow at {floor.group(1)} but constraints.txt requires "
            f">={pinned.group(1)}, so the staged wheel would not satisfy the resolve"
        )

    def test_every_pyarrow_candidate_goes_through_the_floor(self):
        """Counted: four in the preflight (supplied wheel, PyPI, wheelhouse directory, wheelhouse
        index) and two in staging, so a seventh site added later cannot skip the check."""
        assert INSTALL_SRC.count("Test-WoaPyarrowWheelUsable") == 7, (
            "one definition and six call sites; a pyarrow candidate is being accepted on "
            "its tags alone somewhere"
        )

    @staticmethod
    def _usable(
        name: str,
        py: str = "cp313",
        abi: str = "cp313",
    ) -> str:
        script = _script(
            PYARROW_USABLE_FUNCS,
            f"Write-Output ([bool](Test-WoaPyarrowWheelUsable -Name '{name}' "
            f"-PyTag '{py}' -AbiTag '{abi}'))",
        )
        return _ps_last(script)

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, expected, why",
        [
            ("pyarrow-21.0.0-cp313-cp313-win_arm64.whl", "True", "at the floor"),
            ("pyarrow-23.0.1-cp313-cp313-win_arm64.whl", "True", "above it"),
            # The shape upstream is going to ship: cp311-abi3 (apache/arrow#48539).
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
        assert self._usable(name) == expected, why


class TestThePrereleaseAnswerComesFromTheWheel:
    """Not from the URL: a mirror of a prerelease-only channel need not say "nightly". Without
    --prerelease=allow uv takes the stable win_arm64 CPU torch from the PyPI extra index instead
    of the 2.15.0.dev CUDA build the probe just proved."""

    def test_the_installer_reads_the_probed_version(self):
        assert re.search(
            r"\$script:WoaTorchIsPrerelease\s*=\s*\[bool\]\(\$_woaTorchVersion\s*-match",
            INSTALL_SRC,
        ), "install.ps1 no longer derives the prerelease answer from the probed wheel"
        assert (
            "if ($script:WoaTorchIsPrerelease -or ($script:WoaTorchIndexUrl -match 'nightly'))"
            in INSTALL_SRC
        ), "the --prerelease=allow gate is back to testing only the URL spelling"

    def test_the_answer_is_handed_to_setup(self):
        """setup.ps1 cannot probe the index itself, so install.ps1 has to tell it."""
        assert "UNSLOTH_WOA_TORCH_PRERELEASE" in INSTALL_SRC
        assert "UNSLOTH_WOA_TORCH_PRERELEASE" in SETUP_SRC

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
        assert _ps_last(script) == expected, version


class TestAChangedPinInvalidatesTheHandover:
    """install.ps1's flags describe the index IT chose. Change UNSLOTH_TORCH_INDEX_URL and re-run
    in the same shell and the pin outranks the handover for the install itself, while the
    torchaudio and prerelease answers still came from the previous channel: the trio asks a new
    index for an audio wheel it does not publish and the whole torch update aborts."""

    def test_the_handover_index_is_read_before_it_is_overwritten(self):
        capture = SETUP_SRC.index("$_woaHandoffIndex = if ($env:UNSLOTH_WOA_SELECTED_TORCH_INDEX)")
        rewrite = SETUP_SRC.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        assert capture < rewrite, (
            "the handover value is read after setup.ps1 overwrites it, so the comparison "
            "always succeeds and the staleness check does nothing"
        )

    def test_both_flags_are_gated_on_it(self):
        assert (
            "$WinArm64NoAudio = $WinArm64Venv -and -not ($WinArm64HandoffApplies -and"
            ' $env:UNSLOTH_WOA_HAS_TORCHAUDIO -eq "1")' in SETUP_SRC
        )
        assert (
            '($WinArm64HandoffApplies -and $env:UNSLOTH_WOA_TORCH_PRERELEASE -eq "1")' in SETUP_SRC
        )

    def test_the_effective_index_prefers_the_pin(self):
        """The same order the install itself uses, or the two would disagree."""
        assert "$WinArm64EffectiveTorchIndexUrl = if ($PinnedTorchIndexUrl)" in SETUP_SRC
        assert "$_cudaIndexUrl = if ($PinnedTorchIndexUrl) { $TorchInstallIndexUrl }" in SETUP_SRC

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, handoff, audio, expect_no_audio, why",
        [
            ("", NV_GA, "1", "False", "unpinned and unchanged: the handover describes this index"),
            (NV_GA, NV_GA, "1", "False", "pinned to the same index: still current"),
            (NV_GA + "/", NV_GA, "1", "False", "a trailing slash is not a different index"),
            (
                "https://mirror.test/simple",
                NV_GA,
                "1",
                "True",
                "pinned elsewhere: the audio answer belongs to the old channel",
            ),
            ("", "", "1", "True", "no handover to trust"),
            ("", NV_GA, "0", "True", "the handover says no audio"),
        ],
    )
    def test_the_staleness_rule(self, pinned, handoff, audio, expect_no_audio, why):
        script = _script(
            f"$_woaHandoffIndex = '{handoff}'",
            f"$PinnedTorchIndexUrl = '{pinned}'",
            "$WinArm64TorchIndexUrl = $_woaHandoffIndex",
            "$WinArm64Venv = $true",
            f"$env:UNSLOTH_WOA_HAS_TORCHAUDIO = '{audio}'",
            "$WinArm64EffectiveTorchIndexUrl = if ($PinnedTorchIndexUrl)"
            " { ([string]$PinnedTorchIndexUrl).Trim().TrimEnd('/') }",
            "                                  elseif ($WinArm64TorchIndexUrl)"
            " { $WinArm64TorchIndexUrl }",
            "                                  else { '' }",
            "$WinArm64HandoffApplies = [bool]($WinArm64EffectiveTorchIndexUrl -and"
            " $_woaHandoffIndex -and",
            "    $WinArm64EffectiveTorchIndexUrl.Equals($_woaHandoffIndex,"
            " [System.StringComparison]::OrdinalIgnoreCase))",
            "$WinArm64NoAudio = $WinArm64Venv -and -not ($WinArm64HandoffApplies -and"
            ' $env:UNSLOTH_WOA_HAS_TORCHAUDIO -eq "1")',
            "Write-Output ([bool]$WinArm64NoAudio)",
        )
        assert _ps_last(script) == expect_no_audio, why


class TestAnOverrideConflictCanHideInAnInclude:
    """uv follows a nested -r inside an override file, so the top-level scan was not enough: two
    override files naming the same package is an error to uv, so calling them disjoint turns a
    working install into a resolution failure."""

    @staticmethod
    def _names(tmp_path, install: bool):
        source = INSTALL_PS1 if install else SETUP_PS1
        name = "Get-WoaRequirementEntries" if install else "Get-RequirementEntries"
        top = (tmp_path / "top.txt").as_posix()
        script = _script(
            _ps_function(source, name),
            f"$e = @({name} -Path '{top}')",
            "foreach ($x in $e) { Write-Output ($x.Line.Trim() + '|'"
            " + [System.IO.Path]::GetFileName($x.BaseDir)) }",
        )
        return [line for line in _ps_ok(script).stdout.strip().splitlines() if line]

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
        """Detecting a conflict one level down and then folding only the top file would drop the
        line that caused the conflict, which is worse than not folding at all."""
        assert (
            INSTALL_SRC.count("$_woaOvEntries") == 3
        ), "the conflict scan and the fold have diverged"
        assert "foreach ($_woaEntry in (Get-RequirementEntries -Path $_woaFile))" in SETUP_SRC


class TestAFloorIsPep440AboutPrereleases:
    """21.0.0rc1 does not satisfy >=21.0.0, and staging writes an exact == from what it picks, so
    the ordering has to place a pre-release below its own release. A LARGER release is untouched:
    a wheelhouse nightly like the pyarrow 24.0.0.dev260 a GB10 run staged still clears 21.0.0,
    which is what makes hosting a wheel the only step needed to enable a feature there."""

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
        "version, floor, expected, why", CASES, ids = [f"{v}_vs_{f}" for v, f, _, _ in CASES]
    )
    def test_the_ordering(self, version, floor, expected, why):
        script = _script(
            _ps_function(INSTALL_PS1, "Test-WoaVersionAtLeast"),
            f"Write-Output ([bool](Test-WoaVersionAtLeast -Version '{version}' -Floor '{floor}'))",
        )
        assert (_ps_last(script) == "True") is expected, why

    def test_the_table_agrees_with_packaging(self):
        """The PowerShell cannot import packaging, so the expectations are checked against it here
        instead of being asserted from memory."""
        specifiers = pytest.importorskip("packaging.specifiers")
        for version, floor, expected, why in self.CASES:
            try:
                reference = specifiers.SpecifierSet(f">={floor}").contains(
                    version, prereleases = True
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
        got = TestAWheelhousePyarrowMustClearTheFloor._usable(
            "pyarrow-26.0.0-cp311-abi3-win_arm64.whl", abi = "cp313t"
        )
        assert (
            got == "False"
        ), "an abi3 wheel was accepted on a free-threaded venv, where it cannot import"

    @requires_pwsh
    def test_the_wheel_the_gb10_run_staged_is_still_accepted(self):
        """Named explicitly: a floor that rejected it would break a verified install."""
        got = TestAWheelhousePyarrowMustClearTheFloor._usable(
            "pyarrow-24.0.0.dev260-cp313-cp313-win_arm64.whl"
        )
        assert got == "True"


class TestAFindLinksPathWithASpaceSurvivesThePurge:
    """UV_FIND_LINKS is comma-separated to uv, so "C:\\private wheels" is ONE directory. Splitting
    it on whitespace as well tore it into two fragments, dropped the managed entry and rejoined
    the pieces with commas, leaving an air-gapped user pointed at paths that do not exist."""

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
            (r"UV_FIND_LINKS", r"C:\a,C:\b", r"C:\a,C:\b", "two unrelated entries, unchanged"),
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
        script = _script(
            UV_SAFE_PATH,
            f"$StudioHome = '{tmp_path}'",
            f"$env:{var} = '{value.format(owned = owned)}'",
            PURGE_BLOCK,
            f"Write-Output ('[' + [Environment]::GetEnvironmentVariable('{var}') + ']')",
        )
        assert _ps_last(script)[1:-1] == expected.format(owned = owned), why


class TestThePipFallbackKeepsTheIndexArguments:
    """When uv cannot be obtained at all, Fast-Install uses pip -- and pip needs these. The NVIDIA
    channel publishes only the trio, so an install given just --index-url has nowhere to resolve
    their shared dependencies. Remove-UvOnlyResolverFlags is what makes handing pip the same list
    safe: it drops --index-strategy and rewrites --prerelease=allow as --pre."""

    def test_the_arguments_are_not_gated_on_uv(self):
        assert (
            "$WinArm64IndexArgs = if ($WinArm64Venv) {" in SETUP_SRC
        ), "gating on $UseUv means the pip fallback gets no --extra-index-url at all"

    @staticmethod
    def _index_args_block():
        return INDEX_ARGS_BLOCK

    @requires_pwsh
    @pytest.mark.parametrize(
        "use_uv, pre, expect_pre, wheels",
        [
            (True, "1", True, True),
            (False, "1", True, True),
            (False, "0", False, True),
            (False, "1", True, False),
        ],
    )
    def test_pip_receives_a_translated_list(self, tmp_path, use_uv, pre, expect_pre, wheels):
        """Executed end to end: build the list, then run it through the pip translation."""
        if wheels:
            (tmp_path / "woa" / "wheels").mkdir(parents = True)
        script = _script(
            _function_source(SETUP_SRC, "Remove-UvOnlyResolverFlags"),
            # The dependency index follows the resolver policy; with none configured it is PyPI.
            clear_env(UV_INDEX_ENV),
            "$env:UV_NO_CONFIG = '1'",
            UV_SAFE_PATH,
            functions(SETUP_SRC, "Get-WoaUvConfigIndexPolicy", "Get-WoaDependencyIndexArgs"),
            f"$StudioHome = '{tmp_path}'",
            "$WinArm64Venv = $true",
            f"$UseUv = ${str(use_uv).lower()}",
            f"$WinArm64TorchIndexUrl = '{NV_GA}'",
            "$WinArm64EffectiveTorchIndexUrl = $WinArm64TorchIndexUrl",
            "$WinArm64HandoffApplies = $true",
            f"$env:UNSLOTH_WOA_TORCH_PRERELEASE = '{pre}'",
            INDEX_ARGS_BLOCK,
            "$pipArgs = Remove-UvOnlyResolverFlags -Arguments $WinArm64IndexArgs",
            "Write-Output ('[' + ($pipArgs -join ' ') + ']')",
        )
        got = _ps_last(script)[1:-1]
        assert (
            f"--extra-index-url {PYPI}" in got
        ), f"pip cannot resolve the trio's shared dependencies without it: {got!r}"
        assert "--index-strategy" not in got, "a uv-only flag would make pip print usage"
        assert "--prerelease" not in got, "likewise the uv spelling"
        assert ("--pre" in got) is expect_pre, got
        # Fast-Install clears PIP_FIND_LINKS beside UV_FIND_LINKS, so pip is told on the line too.
        if wheels:
            assert f"--find-links {tmp_path / 'woa' / 'wheels'}" in got, got
        else:
            assert (
                "--find-links" not in got
            ), "uv fails outright on a --find-links directory that is missing"

    # Both call sites spell it --prerelease=allow, so the rest of the grammar is untested by the
    # run above. uv accepts a space-separated value too, and five values, only one of which means
    # what --pre means. The first spelling used to drop both tokens and lose the permission;
    # every other value used to become --pre and invert it.
    @requires_pwsh
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["--prerelease=allow", "numpy"], ["--pre", "numpy"]),
            (["--prerelease", "allow", "numpy"], ["--pre", "numpy"]),
            (["--prerelease=disallow", "numpy"], ["numpy"]),
            (["--prerelease", "disallow", "numpy"], ["numpy"]),
            (["--prerelease=if-necessary", "numpy"], ["numpy"]),
            (["--prerelease", "explicit", "numpy"], ["numpy"]),
            (["--index-strategy=unsafe-best-match", "numpy"], ["numpy"]),
            (["--index-strategy", "unsafe-best-match", "numpy"], ["numpy"]),
            # The value is only swallowed by the flag it belongs to.
            (["numpy", "allow"], ["numpy", "allow"]),
            (
                ["--prerelease", "allow", "--index-strategy", "first-index", "numpy"],
                ["--pre", "numpy"],
            ),
        ],
        ids = lambda v: " ".join(v),
    )
    def test_every_spelling_of_the_uv_only_flags(self, argv, expected):
        quoted = ", ".join("'" + a + "'" for a in argv)
        script = _script(
            _function_source(SETUP_SRC, "Remove-UvOnlyResolverFlags"),
            f"$out = Remove-UvOnlyResolverFlags -Arguments @({quoted})",
            "Write-Output ('[' + ($out -join ' ') + ']')",
        )
        assert _ps_last(script)[1:-1] == " ".join(expected)


class TestTheWoaIndexOutlivesTheManifest:
    """The dependency pass deletes the manifest before rebuilding it. A run that dies in that
    window used to leave nothing behind: the next update finds no handover and no manifest, falls
    back to the driver-derived cu130, and fails on an index with no win_arm64 wheel."""

    def test_the_marker_is_written_before_the_manifest_is_dropped(self):
        saved = SETUP_SRC.index("Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex")
        dropped = SETUP_SRC.index("$_ManifestDropped = $true")
        assert saved < dropped, (
            "written after the drop, the marker would not survive the very interruption "
            "it exists for"
        )

    def test_the_manifest_is_still_preferred(self):
        """The marker is a fallback, not a replacement: the manifest is rewritten each run."""
        assert "$_woaFromManifest = Get-PersistedWoaTorchIndex -VenvPath $VenvDir" in SETUP_SRC
        assert (
            "if ($_woaFromManifest) { $_woaFromManifest } else { Get-WoaTorchIndexMarker }"
            in SETUP_SRC
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "url, persisted, why",
        [
            (NV_GA, True, "NVIDIA's own channel"),
            (NV_NIGHTLY, True, "and its nightly"),
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
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            MARKER_FUNCS,
            f"Save-WoaTorchIndexMarker -IndexUrl '{url}'",
            "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
        )
        got = _ps_last(script)[1:-1]
        assert (got == url.rstrip("/")) is persisted, f"{why}: got {got!r}"

    @requires_pwsh
    def test_a_hand_edited_marker_cannot_redirect_the_install(self, tmp_path):
        """Checked on read as well as on write, exactly as the manifest is."""
        woa = tmp_path / "woa"
        woa.mkdir()
        (woa / "torch-index.txt").write_text("https://evil.test/whl", encoding = "utf-8")
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            MARKER_FUNCS,
            "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
        )
        assert _ps_last(script) == "[]"

    @requires_pwsh
    def test_the_marker_lives_where_the_dependency_pass_does_not_reach(self, tmp_path):
        """Beside overrides.txt, which survives the pass for the same reason."""
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            _function_source(SETUP_SRC, "Get-WoaTorchIndexMarkerPath"),
            "Write-Output ('[' + (Get-WoaTorchIndexMarkerPath) + ']')",
        )
        got = _ps_last(script)[1:-1]
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
        """The FILE, not the round-trip: a reader that refuses the value afterwards is no help at
        all if the token was written out in the first place."""
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            MARKER_FUNCS,
            f"Save-WoaTorchIndexMarker -IndexUrl '{url}'",
        )
        _ps_ok(script)
        marker = tmp_path / "woa" / "torch-index.txt"
        written = marker.read_text(encoding = "utf-8") if marker.exists() else ""
        assert "s3cret" not in written, f"the marker file holds a credential: {written!r}"
        assert written == "", "nothing unpersistable should be written at all"


class TestTheTorchMergeRebasesWhatItFolds:
    """Two override files is the NORMAL case on the native path. A non-conflicting caller file is
    kept where it sits, so UV_OVERRIDE names it alongside the generated one; the merge used to
    write itself into the caller's directory to keep relative references working, which only
    helped when there was exactly ONE directory -- with two it fell to %TEMP%."""

    @staticmethod
    def _merge(tmp_path, override_files):
        # A shebang script is not executable on Windows, so the merge would fail about an empty
        # path rather than about rebasing. A .cmd stub runs everywhere.
        if os.name == "nt":
            fake_py = tmp_path / "fakepython.cmd"
            fake_py.write_text(
                "@echo off\r\necho torch==2.11.0+cu130\r\necho torchvision==0.26.0+cu130\r\n",
                encoding = "ascii",
            )
        else:
            fake_py = tmp_path / "fakepython"
            fake_py.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'torch==2.11.0+cu130\\ntorchvision==0.26.0+cu130\\n'\n",
                encoding = "ascii",
            )
            fake_py.chmod(0o755)
        script = _script(
            "$SkipTorch = $false",
            functions(
                INSTALL_SRC,
                "Get-WoaRequirementEntries",
                "Resolve-WoaOverrideLine",
                "New-UnslothTorchOverridesFile",
            ),
            "$env:UV_OVERRIDE = '{}'".format(" ".join(str(f) for f in override_files)),
            f"$m = New-UnslothTorchOverridesFile -PythonExe '{fake_py}'",
            "Write-Output ('<<<' + [System.IO.File]::ReadAllText($m) + '>>>')",
        )
        out = _ps_ok(script, timeout = 180).stdout
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
    """pip has no override mechanism, so falling back to it does not recover here: the WoA
    overrides lift the released torch cap and drop the packages with no win_arm64 build at all,
    and constraints cannot stand in. Running pip anyway downgrades a working CUDA torch or fails
    later with nothing to say why, so it is refused with a reason."""

    @pytest.fixture
    def ips(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ips_pip_fallback", STACK_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    HEADER = "# Generated by install.ps1 for Windows on ARM (win_arm64). See the WoA block there."

    @pytest.mark.parametrize(
        "arm64, files, expected, why",
        [
            (True, ["ours"], True, "the native stack, with overrides in force"),
            (True, ["merged"], True, "setup.ps1's merge copies the generated file first"),
            (True, ["caller", "ours"], True, "ours in force behind a caller's file"),
            (True, ["missing", "ours"], True, "and behind a path that does not open"),
            (True, ["bom"], True, "an editor's BOM does not hide the header"),
            (True, ["caller"], False, "a caller's own override never configured the stack"),
            (True, ["missing"], False, "a path uv cannot open is not one of ours"),
            (True, [], False, "an ARM64 run that configured none is not this case"),
            (True, ["blank"], False, "a blank value is not an override file"),
            (False, ["ours"], False, "x64 keeps the fallback it always had"),
            (False, [], False, "and so does every other host"),
        ],
    )
    def test_when_the_refusal_applies(
        self, ips, monkeypatch, tmp_path, arm64, files, expected, why
    ):
        texts = {
            "ours": self.HEADER + "\ntorch>=2.4\n",
            "merged": self.HEADER + "\ntorch>=2.4\nrich>=13\n",
            "bom": "\ufeff" + self.HEADER + "\n",
            "caller": "rich>=13\n",
        }
        paths = []
        for kind in files:
            if kind == "blank":
                paths.append("   ")
                continue
            path = tmp_path / f"{kind}.txt"
            if kind != "missing":
                path.write_text(texts[kind], encoding = "utf-8")
            paths.append(str(path))
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: arm64)
        if paths:
            monkeypatch.setenv("UV_OVERRIDE", " ".join(paths))
        else:
            monkeypatch.delenv("UV_OVERRIDE", raising = False)
        assert ips._woa_overrides_are_load_bearing() is expected, why

    def test_the_header_is_the_one_install_ps1_writes(self):
        assert f"'{self.HEADER}'" in INSTALL_SRC
        assert self.HEADER.startswith(
            re.search(r'WOA_OVERRIDES_HEADER = "([^"]+)"', STACK_SRC).group(1)
        )

    def test_both_fallback_paths_are_covered(self):
        """uv failing and uv never being available reach pip by different routes."""
        assert STACK_SRC.count("_woa_overrides_are_load_bearing()") == 3, (
            "one definition and both fallback sites; a route that skips the check would "
            "silently resolve the wrong stack"
        )
        after_uv_failed = STACK_SRC.index("if _woa_overrides_are_load_bearing():")
        pip_build = STACK_SRC.index("pip_cmd = _build_pip_cmd(args)")
        assert after_uv_failed < pip_build, "the check has to precede the pip command"

    def test_the_message_names_the_remedy(self):
        assert (
            "Install uv and re-run" in STACK_SRC
        ), "a refusal with no way forward is worse than the silent fallback it replaces"


class TestAnAnnotatedIncludeStillOpens:
    """`-r nested.txt # corporate pins` is a valid line, and the comment is not the path. Capturing
    it meant the include never opened, so a torch conflict inside it went unseen and the two
    override files were handed to uv as disjoint -- which uv then rejected."""

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
        script = _script(
            _ps_function(source, name),
            f"$e = @({name} -Path '{(tmp_path / 'top.txt').as_posix()}')",
            "foreach ($x in $e) { Write-Output $x.Line.Trim() }",
        )
        lines = [line for line in _ps_ok(script).stdout.strip().splitlines() if line]
        assert "idna==3.10" in lines, f"{why}: the include did not open ({lines})"


class TestAnUnrecordableIndexInheritsNothing:
    """A marker we may not overwrite must not be left saying something else. An install that first
    used NVIDIA's channel and later moved to a credentialed corporate mirror kept the old marker,
    so the next fresh-shell update put torch back on the channel the user had moved off."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "second, expect_left, why",
        [
            ("https://mirror.corp.test/simple", "", "a private mirror clears it"),
            ("https://user:tok@pypi.nvidia.com/x", "", "so does a credentialed one"),
            ("https://pypi.nvidia.com/x?token=a", "", "and one carrying a query"),
            (NV_NIGHTLY, NV_NIGHTLY, "a recordable index simply replaces it"),
        ],
    )
    def test_the_marker_does_not_outlive_its_index(self, tmp_path, second, expect_left, why):
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            MARKER_FUNCS,
            f"Save-WoaTorchIndexMarker -IndexUrl '{NV_GA}'",
            f"Save-WoaTorchIndexMarker -IndexUrl '{second}'",
            "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
        )
        assert _ps_last(script)[1:-1] == expect_left, why

    @requires_pwsh
    def test_clearing_removes_the_file_rather_than_blanking_it(self, tmp_path):
        """A zero-byte marker would read as empty anyway, but leaving one behind invites the next
        reader to treat "present" as meaningful."""
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            MARKER_FUNCS,
            f"Save-WoaTorchIndexMarker -IndexUrl '{NV_GA}'",
            "Save-WoaTorchIndexMarker -IndexUrl 'https://mirror.corp.test/simple'",
        )
        _ps_ok(script)
        assert not (tmp_path / "woa" / "torch-index.txt").exists()


class TestWheelsAnEarlierWheelhouseLeftArePruned:
    """The managed woa\\wheels directory persists across runs, so after UNSLOTH_WOA_WHEELHOUSE
    changed a wheel the earlier wheelhouse staged (tiktoken, say) stayed in it, the scan below
    read it as hosted now, and UV_FIND_LINKS installed it from a source no longer configured.
    Reconciled against the current listing; kept when the listing could not be read (offline
    reuse). The managed directory as its own wheelhouse lists exactly what it holds."""

    STALE = "tiktoken-0.9.0-cp313-cp313-win_arm64.whl"
    HOSTED = "hf_transfer-0.1.9-cp313-cp313-win_arm64.whl"
    PYARROW = "pyarrow-25.0.1-cp313-cp313-win_arm64.whl"

    @classmethod
    def _run(
        cls,
        tmp_path,
        wheelhouse,
        extra_stubs = (),
    ):
        managed = tmp_path / "woa" / "wheels"
        managed.mkdir(parents = True)
        (managed / cls.STALE).write_text("")
        (managed / cls.PYARROW).write_text("")
        script = _script(
            substep_collector(),
            functions(INSTALL_SRC, "Test-WoaWheelhouseIsLocal", "Test-WoaSamePath", "Join-UrlPath"),
            "function Get-WoaAbiTag { param($PythonMinor, $FreeThreaded) 'cp313' }",
            "function Test-WoaWheelhouseWheelIsRedundant { param($Name, $PyTag, $AbiTag) $false }",
            "function Test-ZipArchiveReadable { param($Path) $true }",
            "function Invoke-WebRequest { param($Uri, $OutFile, [switch]$UseBasicParsing, $TimeoutSec, $ErrorAction) Set-Content -Path $OutFile -Value 'x' }",
            *extra_stubs,
            "$WoaVenvMinor = '3.13'",
            "$script:WoaVenvFreeThreaded = $false",
            "$script:WoaPyPIProvided = @{}",
            "$script:WoaPyPIMatchedVersion = $null",
            f"$script:WoaPyarrowWheelName = '{cls.PYARROW}'",
            f"$WoaWheelDir = '{managed}'",
            f"$script:WoaWheelhouse = '{wheelhouse}'",
            slice_between(
                INSTALL_SRC, "$WoaExtraStaged = 0", "        if ($WoaExtraStaged -gt 0) {"
            ),
            "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
        )
        out = _ps_ok(script).stdout
        return sorted(p.name for p in managed.glob("*.whl")), out

    @requires_pwsh
    def test_a_local_wheelhouse_prunes_what_it_no_longer_hosts(self, tmp_path):
        house = tmp_path / "house"
        house.mkdir()
        (house / self.HOSTED).write_text("")
        names, out = self._run(tmp_path, house)
        assert names == sorted([self.HOSTED, self.PYARROW]), names
        assert "removed 1 wheel(s) an earlier wheelhouse left" in out

    @requires_pwsh
    def test_a_url_wheelhouse_prunes_against_its_index(self, tmp_path):
        names, out = self._run(
            tmp_path,
            "https://wheels.test/woa",
            [invoke_restmethod(f"{self.HOSTED}\n{self.PYARROW}\n")],
        )
        assert names == sorted([self.HOSTED, self.PYARROW]), names

    @requires_pwsh
    def test_the_selected_pyarrow_is_kept_even_when_the_listing_lacks_it(self, tmp_path):
        """UNSLOTH_PYARROW_WHEEL supplies a wheel no wheelhouse lists."""
        names, _ = self._run(
            tmp_path, "https://wheels.test/woa", [invoke_restmethod(f"{self.HOSTED}\n")]
        )
        assert self.PYARROW in names, names
        assert self.STALE not in names, names

    @requires_pwsh
    def test_an_unreadable_listing_keeps_the_offline_copies(self, tmp_path):
        names, out = self._run(tmp_path, "https://wheels.test/woa", [INVOKE_RESTMETHOD_OFFLINE])
        assert self.STALE in names, "offline, the staged copies are the only source"
        assert "earlier wheelhouse" not in out

    @requires_pwsh
    def test_the_managed_directory_as_its_own_wheelhouse_is_left_alone(self, tmp_path):
        names, out = self._run(tmp_path, tmp_path / "woa" / "wheels")
        assert self.STALE in names, "self-sourced: every wheel there is the wheelhouse"
        assert "earlier wheelhouse" not in out


class TestALocalWheelIsOpenedBeforeItCounts:
    """The wheelhouse mirror trusted a filename; the resolver then trusted the mirror.
    _find_links_wheel_versions reads names, so a truncated wheel copied into the managed directory
    took its package off the ARM64 skip list and uv failed the whole dependency pass on it."""

    @staticmethod
    def _staging() -> str:
        return slice_between(
            INSTALL_SRC, "$WoaExtraStaged = 0", "$WoaOverrides = Join-Path $WoaDir"
        )

    def test_both_staging_branches_validate(self):
        block = self._staging()
        assert block.count("Test-ZipArchiveReadable") >= 3, (
            "the local mirror, the reused download and the fresh download all have to "
            f"open what they count: {block.count('Test-ZipArchiveReadable')}"
        )
        local = block[: block.index("} else {")]
        assert (
            "Test-ZipArchiveReadable" in local
        ), "the local branch counted a wheel on its filename alone"

    def test_the_check_precedes_the_copy(self):
        local = slice_between(INSTALL_SRC, "$WoaExtraStaged = 0", "} else {")
        assert local.index("Test-ZipArchiveReadable") < local.index("Copy-Item -LiteralPath"), (
            "validating after the copy would still put a corrupt wheel in the cache, "
            "where the resolver reads it"
        )


class TestTheMandatoryPyarrowWheelIsOpened:
    """pyarrow decides the ROUTE, so a truncated one must not select native mode: staging writes
    an exact pyarrow== override from whichever file it picks, so a wheel that only looks right
    took the native path and then failed the resolve, with x64 already given up."""

    def test_the_probe_and_the_staging_agree(self):
        """Different filters would let the probe clear one file and staging take another."""
        assert (
            INSTALL_SRC.count(
                "(Test-WoaPyarrowWheelUsable -Name $_.Name -PyTag $tag -AbiTag $AbiTag) -and"
            )
            == 2
        ), "the local probe and the local staging both filter on tags AND readability"
        assert INSTALL_SRC.count("(Test-ZipArchiveReadable -Path $_.FullName)") == 2

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

        script = pyarrow_source_script(wheelhouse = f"'{tmp_path}'", local = "$true")
        assert _ps_last(script, timeout = 180)[1:-1] == expected, why


class TestARebasedOptionPathKeepsItsQuoting:
    """-r/-c/-f take ONE file argument, so an unquoted space truncates the path. Two ways in: a
    caller who quoted the value had the quotes stripped and not put back, and a caller who had no
    reason to quote a plain relative name gets a space anyway when it rebases onto a directory
    that has one."""

    @staticmethod
    def _rebase(source, line, base):
        # Here-strings, because the values under test contain both quote characters and spaces.
        script = _script(
            _ps_function(source, "Resolve-WoaOverrideLine"),
            f"$l = @'\n{line}\n'@",
            f"$b = @'\n{base}\n'@",
            "Write-Output ('[' + (Resolve-WoaOverrideLine -Line $l -BaseDir $b) + ']')",
        )
        return _ps_last(script)[1:-1]

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
            ("-r nested.txt", "/opt/my corp", True, "and here the base brings the space in"),
            ("-f wheels", "/opt/my corp", True, "find-links too"),
            (
                "--constraint 'corp pins/c.txt'",
                "/opt/corp",
                True,
                "a single-quoted value is unwrapped and re-quoted the same way",
            ),
            ("-r nested.txt", "/opt/corp", False, "nothing with a space stays unquoted, as ever"),
            ("-c /already/absolute.txt", "/opt/corp", False, "an absolute path is untouched"),
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
        """Read back the way a resolver reads it: split the option's argument on spaces and the
        path must still exist."""
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
        install, setup = _ps_copies("Resolve-WoaOverrideLine")
        assert install == setup


class TestALocalDirectoryRequirementIsRebasedToo:
    """The line forms the fold moved and left behind. New-UnslothTorchOverridesFile writes to
    %TEMP% now and rebases each line on the way, but only for the forms Resolve-WoaOverrideLine
    knows: ``-e ./pkg`` and a bare ``./pkg`` are requirements pip and uv accept, and both pointed
    at nothing after the move. The fold runs for every Windows host with UV_OVERRIDE set, so the
    regression reached hosts this feature never touches."""

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    @pytest.mark.parametrize(
        "line, expected_suffix, why",
        [
            ("-e ./localproj", "ovdir/localproj", "editable, forward slashes"),
            ("--editable ..\\sibling", "sibling", "the long spelling and a parent segment"),
            ("./localproj", "ovdir/localproj", "a bare relative directory"),
            (".", "ovdir", "the current directory, which is the commonest spelling of all"),
            ("-e .[dev]", "ovdir[dev]", "extras stay outside the path"),
            ("./localproj[dev]", "ovdir/localproj[dev]", "and on the bare form too"),
        ],
    )
    def test_a_local_project_path_is_made_absolute(
        self, install, line, expected_suffix, why, tmp_path
    ):
        source = INSTALL_PS1 if install else SETUP_PS1
        base = tmp_path / "ovdir"
        base.mkdir()
        got = TestARebasedOptionPathKeepsItsQuoting._rebase(source, line, base.as_posix())
        # The value, not the option token in front of it, and without any extras suffix.
        value = got.split(None, 1)[1] if got.startswith("-") else got
        assert os.path.isabs(
            re.sub(r"\[[^\]]*\]$", "", value.strip())
        ), f"{why}: the line was left relative and now resolves against %TEMP%: {got!r}"
        assert got.rstrip().replace(os.sep, "/").endswith(expected_suffix), f"{why}: {got!r}"

    @requires_pwsh
    @pytest.mark.parametrize("install", [True, False], ids = ["install.ps1", "setup.ps1"])
    @pytest.mark.parametrize(
        "line",
        [
            "packaging>=20",
            "torch==2.6.0",
            "some-pkg",
            "pkg ; python_version < '3.11'",
            "-e git+https://example.test/p.git#egg=p",
            "-e /already/absolute",
        ],
    )
    def test_a_line_with_no_relative_path_is_untouched(self, install, line):
        """A package name may not begin with a dot, which is the whole of the test above's licence
        to rewrite one. Nothing else may move."""
        source = INSTALL_PS1 if install else SETUP_PS1
        assert TestARebasedOptionPathKeepsItsQuoting._rebase(source, line, "/opt/corp") == line


class TestTheMarkerRecordsTheIndexActuallyUsed:
    """A generic pin never reached the marker, only the WoA chain did: $_cudaIndexUrl prefers
    $PinnedTorchIndexUrl, while $WinArm64TorchIndexUrl consults only UNSLOTH_WOA_TORCH_INDEX_URL,
    the handover, the manifest and the marker. So a run pinned elsewhere installed from the pin
    and recorded an NVIDIA channel it had not used."""

    def test_the_saved_value_prefers_the_pin(self):
        assert "$_woaMarkerIndex = $_woaPinnedIndex" in SETUP_SRC
        assert "else { $_woaMarkerIndex = $WinArm64TorchIndexUrl }" in SETUP_SRC
        assert "Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex" in SETUP_SRC
        # The same value the marker gets, or the manifest shadows it on the next fresh shell.
        assert "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex" in SETUP_SRC

    def test_the_pin_is_read_before_it_is_used(self):
        """Get-PinnedTorchIndexUrl is a function, so only its DEFINITION has to precede this."""
        assert SETUP_SRC.index("function Get-PinnedTorchIndexUrl") < SETUP_SRC.index(
            "$_woaPinnedIndex = if ($WinArm64Venv) { Get-PinnedTorchIndexUrl }"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, chain, expected, why",
        [
            ("", NV_GA, NV_GA, "no pin: the WoA chain is what the run uses, so it is recorded"),
            (NV_NIGHTLY, NV_GA, NV_NIGHTLY, "a pin to another NVIDIA channel replaces it"),
            (
                "https://download.pytorch.org/whl/cu130",
                NV_GA,
                "",
                "a pin to a recognised non-NVIDIA index clears it rather than leaving a lie",
            ),
            ("https://mirror.corp.test/simple", NV_GA, "", "and so does a private mirror"),
        ],
    )
    def test_what_ends_up_on_disk(self, tmp_path, pinned, chain, expected, why):
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            f"function Get-PinnedTorchIndexUrl {{ return '{pinned}' }}",
            f"$WinArm64TorchIndexUrl = '{chain}'",
            MARKER_FUNCS,
            # A previous run recorded the public channel; this run may not inherit it.
            f"Save-WoaTorchIndexMarker -IndexUrl '{NV_GA}'",
            "$WinArm64Venv = $true",
            "$_woaHandoffIndex = ''",
            _persistence_block(SETUP_SRC),
            "Write-Output ('[' + (Get-WoaTorchIndexMarker) + ']')",
        )
        assert _ps_last(script)[1:-1] == expected, why


class TestTheManifestRecordsTheSameIndexAsTheMarker:
    """The marker was corrected; the manifest kept the old answer and outranked it, because
    $WinArm64TorchIndexUrl reads the manifest BEFORE the marker. Exporting the WoA chain while
    saving the pin left the two records disagreeing, and the losing one was the correct one."""

    def test_the_export_and_the_save_carry_one_value(self):
        export = SETUP_SRC.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex")
        save = SETUP_SRC.index("Save-WoaTorchIndexMarker -IndexUrl $_woaMarkerIndex")
        resolve = SETUP_SRC.index("$_woaMarkerIndex = $_woaPinnedIndex")
        assert resolve < export < save, "resolved once, then written to both records"

    def test_the_stack_writes_that_variable_into_the_manifest(self):
        """The premise: without this read the export would reach nothing."""
        assert "UNSLOTH_WOA_SELECTED_TORCH_INDEX" in STACK_SRC
        assert "woa_torch_index" in STACK_SRC

    def test_the_manifest_is_preferred_over_the_marker_on_read(self):
        """Which is why the two must agree rather than the marker being enough."""
        chain = SETUP_SRC.index("$_woaFromManifest = Get-PersistedWoaTorchIndex -VenvPath $VenvDir")
        assert (
            "if ($_woaFromManifest) { $_woaFromManifest } else { Get-WoaTorchIndexMarker }"
            in SETUP_SRC[chain : chain + 300]
        )

    def test_a_moved_pin_drops_the_probed_indexs_flags(self):
        """torchaudio and prerelease were measured on the index install.ps1 probed."""
        guard = SETUP_SRC.index("if ($_woaMarkerIndex -ne $_woaHandoffIndex) {")
        body = SETUP_SRC[guard : SETUP_SRC.index("$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX", guard)]
        assert "Remove-Item Env:UNSLOTH_WOA_HAS_TORCHAUDIO" in body
        assert "Remove-Item Env:UNSLOTH_WOA_TORCH_PRERELEASE" in body
        assert guard < SETUP_SRC.index(
            "$env:UNSLOTH_WOA_SELECTED_TORCH_INDEX = $_woaMarkerIndex"
        ), "compared before the overwrite, or the two are always equal"


class TestThePypiPyarrowWheelIsPinnedToo:
    """ "PyPI has a compatible wheel" is not "the newest release is one". The probe cleared the
    native route on a wheel it then forgot, and with just pyarrow>=21.0.0 in force uv takes the
    newest release -- which, if it ships only an sdist for this interpreter, builds Arrow from
    source, the outcome this preflight exists to prevent."""

    def test_the_probe_records_what_it_matched(self):
        body = slice_between(
            INSTALL_SRC, "function Get-WoaPyarrowSource", "function Test-WoaNvidiaPresent"
        )
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
        script = pyarrow_source_script(
            wheelhouse = "''",
            rest_method = invoke_restmethod(body),
            reaches_pypi = "$true",
            tail = (
                "$null = Get-WoaPyarrowSource -PythonMinor '3.13'",
                # The emission, verbatim from the override block.
                "$pin = ''",
                "if ($script:WoaPyarrowWheelName -and $script:WoaPyarrowWheelName"
                " -match '^pyarrow-([^-]+)-') {",
                "    $pin = $Matches[1] }",
                "Write-Output ('[' + $pin + ']')",
            ),
        )
        assert _ps_last(script, timeout = 180)[1:-1] == expected_pin, why

    def test_the_override_is_emitted_for_every_source(self):
        """The pin is keyed on the recorded name, which all three routes now set."""
        assert INSTALL_SRC.count("$script:WoaPyarrowWheelName = ") == 6, (
            "the script-level init, the per-probe reset, and one per source: pypi, the "
            "supplied UNSLOTH_PYARROW_WHEEL, the wheelhouse directory, the wheelhouse index"
        )


class TestEveryPyarrowRouteOpensWhatItKeeps:
    """Four ways in, and the last one was trusting a 200 response. A mirror can serve a truncated
    body with a successful status, and this wheel decides the route and the exact pyarrow==
    override, so an unreadable download kept native mode and then failed uv on the override."""

    DOWNLOAD = "Invoke-WebRequest -Uri (Join-UrlPath $script:WoaWheelhouse $wheelName)"

    def test_every_mandatory_route_validates(self):
        """Staging opens the two files it chooses itself; the probe opens the other two. Staging
        trusts the probe's answer for UNSLOTH_PYARROW_WHEEL, which is why that one is checked
        there and not again here."""
        block = slice_between(
            INSTALL_SRC, 'if ($script:WoaPyarrowSource -eq "local") {', "$WoaExtraStaged = 0"
        )
        assert block.count("Test-ZipArchiveReadable") == 2, (
            "the wheelhouse directory selection and the download: "
            f"{block.count('Test-ZipArchiveReadable')}"
        )
        probe = slice_between(
            INSTALL_SRC, "function Get-WoaPyarrowSource", "function Test-WoaNvidiaPresent"
        )
        assert probe.count("Test-ZipArchiveReadable") == 2, (
            "the probe opens the supplied wheel and the local wheelhouse candidate before "
            f"it clears the native route: {probe.count('Test-ZipArchiveReadable')}"
        )

    def test_a_bad_download_is_removed_not_left_in_the_cache(self):
        start = INSTALL_SRC.index(self.DOWNLOAD)
        block = INSTALL_SRC[start : start + 900]
        assert "Remove-Item -LiteralPath $_woaPaDest" in block, (
            "a corrupt wheel left in the managed directory is read by the resolver on "
            "every later run, and by _find_links_wheel_versions as proof of availability"
        )
        assert block.index("Remove-Item") < block.index("throw"), "removed before it gives up"

    def test_the_failure_falls_back_rather_than_continuing(self):
        """It throws into the existing catch, which is what disables the native route."""
        start = INSTALL_SRC.index(self.DOWNLOAD)
        block = INSTALL_SRC[start : start + 1400]
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
        """Executed, because a source-level assertion cannot tell a live branch from a dead one,
        and because the point is what is left on disk afterwards."""
        import zipfile

        served = tmp_path / "served.whl"
        if readable:
            with zipfile.ZipFile(served, "w") as archive:
                archive.writestr("pyarrow/__init__.py", "")
        else:
            served.write_bytes(b"PK\x03\x04truncated")
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()

        branch = slice_between(
            INSTALL_SRC,
            "                if ($wheelName) {",
            "                } else {\n                    $script:WoaNativeCudaTorch = $false",
        )
        script = _script(
            SUBSTEP_NOOP,
            JOIN_URL_RETURNS_PATH,
            "function Invoke-WebRequest {",
            "  param([Parameter(ValueFromRemainingArguments=$true)]$a)",
            f"  Copy-Item -LiteralPath '{served.as_posix()}'"
            " -Destination $a[$a.IndexOf('-OutFile') + 1] -Force }",
            _ps_function(INSTALL_PS1, "Test-ZipArchiveReadable"),
            f"$WoaWheelDir = '{wheel_dir.as_posix()}'",
            "$script:WoaWheelhouse = 'https://mirror.test/wheels'",
            "$script:WoaNativeCudaTorch = $true",
            "$script:WoaPyarrowWheelName = $null",
            "$wheelName = 'pyarrow-24.0.0-cp313-cp313-win_arm64.whl'",
            # The slice stops before the "} else {", so that closing brace is not in it.
            branch + "\n                }",
            "Write-Output ('[' + [bool]$script:WoaNativeCudaTorch + ']')",
        )
        assert _ps_last(script, timeout = 180)[1:-1] == expect_native, why
        staged = list(wheel_dir.glob("*.whl"))
        assert (
            bool(staged) is readable
        ), f"a rejected wheel must not stay in the managed directory: {staged}"


class TestAnExplicitPinIsPersistedWithoutAnOldRecord:
    """The persistence block was gated on the WoA chain alone. A native venv installed through a
    credentialed mirror has nothing to recover, so the chain is empty; pin UNSLOTH_TORCH_INDEX_URL
    at an NVIDIA channel on a later direct update and the torch install used it while the guard
    skipped both records, leaving the next fresh shell on an index with no win_arm64 wheel."""

    def test_either_record_opens_the_block(self):
        assert "if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {" in SETUP_SRC

    def test_the_pin_is_only_consulted_on_a_native_venv(self):
        """Every other host must reach the block exactly as it did before."""
        assert (
            "$_woaPinnedIndex = if ($WinArm64Venv) { Get-PinnedTorchIndexUrl } else { $null }"
            in SETUP_SRC
        )

    def test_the_pin_is_resolved_once(self):
        """Two reads of the getter are two chances to disagree about what was installed."""
        assert "$_woaMarkerIndex = $_woaPinnedIndex" in SETUP_SRC
        opens = SETUP_SRC.index("if ($WinArm64TorchIndexUrl -or $_woaPinnedIndex) {")
        closes = SETUP_SRC.index("Restore-WoaResolverEnvironment", opens)
        assert (
            "Get-PinnedTorchIndexUrl" not in SETUP_SRC[opens:closes]
        ), "the block calls the getter again instead of using the value the guard tested"

    @requires_pwsh
    @pytest.mark.parametrize(
        "pinned, chain, expected, why",
        [
            (NV_GA, "", NV_GA, "nothing to recover, but the pin is what the install used"),
            ("", NV_GA, NV_GA, "no pin: the recovered chain, exactly as before"),
            ("", "", "", "neither: the block does not run at all"),
        ],
    )
    def test_what_the_guard_lets_through(self, tmp_path, pinned, chain, expected, why):
        script = _script(
            f"$StudioHome = '{tmp_path}'",
            "$WinArm64Venv = $true",
            f"function Get-PinnedTorchIndexUrl {{ return '{pinned}' }}",
            f"$WinArm64TorchIndexUrl = '{chain}'",
            "$_woaHandoffIndex = ''",
            MARKER_FUNCS,
            _persistence_block(SETUP_SRC),
            "Write-Output ('[' + $env:UNSLOTH_WOA_SELECTED_TORCH_INDEX + ']')",
        )
        assert _ps_last(script)[1:-1] == expected, why


class TestTheProbedCudaWheelIsWhatGetsInstalled:
    """A floor plus unsafe-best-match is not a request for the wheel that was probed: uv selects
    the best version from the combined candidate set of every index, and the PyPI extra index is
    there because NVIDIA's channel publishes only the trio. So the moment PyPI's stable win_arm64
    CPU torch is one release ahead, `torch>=2.4` takes it and the native GPU path is replaced by a
    CPU build that imports perfectly."""

    @staticmethod
    def _block() -> str:
        # Anchored on the code that follows rather than on its comment.
        return slice_between(
            INSTALL_SRC,
            'if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {',
            "if ($script:PrevTorchPin -and $script:WoaNativeCudaTorch -and $VenvPlatform -eq"
            ' "win-arm64") {',
        )

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
        for name in ("WoaTorchWheelVersion", "WoaAudioWheelVersion", "WoaVisionWheelVersion"):
            assert f"$script:{name} = " in INSTALL_SRC, f"{name} is never set"
        assert '-Project "torchvision"' in INSTALL_SRC, "torchvision is never probed"

    def test_the_pin_is_set_before_the_install_reads_it(self):
        assert INSTALL_SRC.index("$script:WoaVisionWheelVersion = ") < INSTALL_SRC.index(
            '"torchvision==$($script:WoaVisionWheelVersion)"'
        )

    def test_audio_still_needs_the_pairing(self):
        """The exact pin does not replace Test-WoaAudioMatchesTorch: a paired version is what
        makes torchaudio installable at all, and this only fixes which one lands."""
        block = self._block()
        assert "if ($script:WoaTorchAudio) {" in block[: block.index("torchaudio==")]

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
        script = _script(
            SUBSTEP_NOOP,
            '$VenvPlatform = "win-arm64"',
            "$script:WoaNativeCudaTorch = $true",
            f"$script:WoaTorchWheelVersion = '{torch_v}'",
            f"$script:WoaVisionWheelVersion = '{vision_v}'",
            f"$script:WoaAudioWheelVersion = '{audio_v}'",
            f"$script:WoaTorchAudio = {has_audio}",
            "$script:WoaTorchIsPrerelease = $false",
            f"$script:WoaTorchIndexUrl = '{NV_GA}'",
            self._block(),
            "Write-Output ($_torchSpecs -join ' ')",
        )
        assert _ps_last(script) == expected, why


class TestTheCompanionWheelsArePairedWithTorch:
    """Newest-of-each is not a pair on a channel that publishes on separate schedules. NVIDIA's
    nightly channel stamps each project independently and nightly torchvision metadata pins its
    exact torch, so maximizing the two separately and pinning both exactly can name a pair no
    index can satisfy -- after the installer has committed to the ARM64 path."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch_v, other_v, pairs, why",
        [
            ("2.15.0.dev20260101+cu134", "0.26.0.dev20260101+cu134", True, "same stamp and tag"),
            ("2.15.0.dev20260101+cu134", "0.26.0.dev20260102+cu134", False, "staggered nightly"),
            ("2.15.0.dev20260101+cu134", "0.26.0+cu134", False, "a release is not that build"),
            ("2.14.0+cu134", "0.29.0+cu134", True, "GA: torchvision 0.(M+15) pairs with torch 2.M"),
            ("2.14.0+cu134", "0.25.0+cu134", False, "GA, same tag, another release line"),
            ("2.14.0+cu134", "0.29.0+cu130", False, "a different CUDA build"),
            ("2.14.0+cu134", "", False, "nothing to pair with"),
        ],
    )
    def test_what_counts_as_a_pair(self, torch_v, other_v, pairs, why):
        script = _script(
            _ps_function(INSTALL_PS1, "Test-WoaWheelPairsWithTorch"),
            f"Write-Output (Test-WoaWheelPairsWithTorch -TorchVersion '{torch_v}'"
            f" -OtherVersion '{other_v}')",
        )
        assert (_ps_last(script) == "True") is pairs, why

    def test_both_companions_are_probed_as_a_pair(self):
        for project in ("torchvision", "torchaudio"):
            line = [
                l
                for l in INSTALL_SRC.splitlines()
                if f'-Project "{project}"' in l and "Get-WoaCudaWheelVersion" in l
            ]
            assert line, f"{project} is not probed"
            assert all(
                "-PairWith $_woaTorchVersion" in l for l in line
            ), f"{project} is still maximized independently of torch"

    def test_an_unpaired_torchvision_is_a_gate_not_a_floor(self):
        """A pin the index cannot satisfy is worse than the floor it replaced, and the floor
        resolved against the wrong torch: the pairing now decides whether the index is used at
        all, so the trio only ever sees a paired pin."""
        fn = _function_source(INSTALL_SRC, "Initialize-WoaNativeCudaTorch")
        assert "if (-not $_woaVisionVersion) {" in fn
        assert "$script:WoaVisionWheelVersion = $_woaVisionVersion" in fn


class TestTheRepairPathPinsTheSameWayTheInstallDoes:
    """setup.ps1's forced repair carries the same flags, so it had the same defect: those specs
    are resolved with unsafe-best-match against a public PyPI extra index, and the repair replaces
    the CUDA stack with CPU wheels the moment PyPI is one release ahead."""

    def test_the_repair_probes_the_effective_index(self):
        assert "if ($WinArm64Venv -and $WinArm64EffectiveTorchIndexUrl) {" in SETUP_SRC
        assert '$WinArm64TorchSpec = "torch==$_woaTorchV"' in SETUP_SRC

    def test_the_companions_are_paired_here_too(self):
        for project in ("torchvision", "torchaudio"):
            assert f'-Project "{project}" -PairWith $_woaTorchV' in SETUP_SRC

    def test_an_unanswered_index_leaves_every_spec_alone(self):
        """Best effort: no probe, no change, which is exactly today's behaviour."""
        block = slice_between(SETUP_SRC, '$WinArm64TorchSpec = "torch>=2.4"', "$_tritonSpec = ")
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
        copy = _function_source(SETUP_SRC, setup_fn)

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
        listing = " ".join(
            f'<a href="{name}">{name}</a>'
            for name in (
                "torchvision-0.26.0.dev20260101%2Bcu134-cp313-cp313-win_arm64.whl",
                "torchvision-0.26.0.dev20260102%2Bcu134-cp313-cp313-win_arm64.whl",
                # NEWER, and tagged for another interpreter: it must lose on the tag alone.
                "torchvision-0.27.0.dev20260103%2Bcu134-cp312-cp312-win_arm64.whl",
            )
        )
        script = _script(
            invoke_restmethod(listing),
            functions(
                SETUP_SRC,
                "Test-WoaWheelTagsParity",
                "Test-WoaPairsWithTorchParity",
                "Get-WoaCudaWheelVersionParity",
            ),
            f"$v = Get-WoaCudaWheelVersionParity -IndexUrl '{NV_GA}'"
            f" -PyTag 'cp313' -AbiTag 'cp313' -Project 'torchvision' -PairWith '{pair_with}'",
            "Write-Output ('[' + $v + ']')",
        )
        assert _ps_last(script)[1:-1] == expected, why


class TestTheOverrideFileDoesNotOutrankTheTorchPin:
    """uv's --overrides replace a version even for a requirement named on the command line
    (verified against uv 0.10.7). The generated file carries torch>=2.4 and torchvision>=0.19, so
    it discarded the exact CUDA pins the probe had just selected and best-match took PyPI's newer
    CPU wheel."""

    SWAP = "$_woaStep = New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved"

    def test_the_trio_is_dropped_for_that_one_command(self):
        assert "function New-WoaTorchStepOverrideValue {" in INSTALL_SRC
        body = _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue")
        assert '@("torch", "torchvision", "torchaudio") -contains $name' in body

    def test_it_is_applied_around_the_native_install_only(self):
        swap = INSTALL_SRC.index(self.SWAP)
        guard = INSTALL_SRC.rindex(
            'if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64" -and'
            " $env:UV_OVERRIDE) {",
            0,
            swap,
        )
        assert guard < swap, "every other host must keep the overrides it had"

    def test_the_original_value_is_restored(self):
        """The later unsloth resolve still needs the drop list."""
        swap = INSTALL_SRC.index(self.SWAP)
        tail = INSTALL_SRC[swap : swap + 2600]
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
        src = tmp_path / "ovr.txt"
        src.write_text("\n".join(lines) + "\n", encoding = "utf-8")
        script = _script(
            UV_SAFE_PATH,
            functions(
                INSTALL_SRC,
                "Get-WoaRequirementEntries",
                "Resolve-WoaOverrideLine",
                "New-WoaTorchStepOverrideValue",
            ),
            f"$v = (New-WoaTorchStepOverrideValue -Value '{src}' -Dir '{tmp_path}').Value",
            "Get-Content -LiteralPath $v | ForEach-Object { Write-Output $_ }",
        )
        out = _ps_ok(script).stdout
        for name in expect_kept:
            assert name in out, f"{name} was dropped: {why}"
        for name in ("torch>=", "torchvision>=", "torchaudio>="):
            assert name not in out, f"{name} survived: {why}"


class TestATransientProbeFailureKeepsTheCudaBundle:
    """nvidia-smi is a probe, and one that did not answer is not evidence the GPU is gone. During
    a direct update of a native ARM64 CUDA install, a transiently missing nvidia-smi dropped
    windows-arm64-cuda from the expected kinds, deleted the working llama.cpp tree, and ran the
    selector with no NVIDIA evidence, which installs the CPU bundle instead."""

    def test_the_persisted_cuda_index_counts_as_evidence(self):
        assert "$_nvidiaEvidence = $HasNvidiaSmi -or ((Test-WinArm64Venv)" in SETUP_SRC
        assert "elseif ($_nvidiaEvidence) { $_nvidiaKinds }" in SETUP_SRC

    def test_only_a_persistable_index_counts(self):
        """Test-WoaPersistableIndex passes only NVIDIA's own channels, so a /cpu pin cannot claim
        to be NVIDIA evidence."""
        start = SETUP_SRC.index("$_nvidiaEvidence = ")
        assert "Test-WoaPersistableIndex $_woaEvidenceIndex" in SETUP_SRC[start : start + 400]

    def test_rocm_still_wins_the_branch(self):
        """The ROCm arm is first and unchanged: this only widens the NVIDIA one."""
        line = [l for l in SETUP_SRC.splitlines() if "$expectedKinds = if (" in l][0]
        assert line.index("$HasROCm") < line.index("$_nvidiaEvidence")


class TestStableCompanionsPairByReleaseLine:
    """Every stable release has an empty dev stamp, so the CUDA tag alone paired a companion from
    any release the index still served and the exact-pin install then asked for a pair that does
    not exist. torchvision 0.(M+15) requires torch 2.M exactly (PyPI metadata: 0.25.0 ->
    torch==2.10.0, 0.19.0 -> torch==2.4.0); torchaudio agrees on major.minor."""

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
        script = _script(
            _ps_function(INSTALL_PS1, "Test-WoaWheelPairsWithTorch"),
            f"Write-Output (Test-WoaWheelPairsWithTorch -TorchVersion '{torch_v}'"
            f" -OtherVersion '{other_v}' -Project '{project}')",
        )
        assert (_ps_last(script) == "True") is pairs, why

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
    """GetTempFileName() lands in %TEMP%, which follows the profile: a spaced one produced a quoted
    path in UV_OVERRIDE, and uv rejects quoting there, so the torch command failed before
    installing anything. And the copies were never deleted, and a flattened caller file can carry
    an authenticated URL."""

    @requires_pwsh
    def test_the_copy_lands_in_the_given_directory_and_is_reported(self, tmp_path):
        src = tmp_path / "ovr.txt"
        src.write_text('torch>=2.4\nhf-transfer ; platform_machine == "AMD64"\n', encoding = "utf-8")
        woa = tmp_path / "woa"
        woa.mkdir()
        script = _script(
            UV_SAFE_PATH,
            functions(
                INSTALL_SRC,
                "Get-WoaRequirementEntries",
                "Resolve-WoaOverrideLine",
                "New-WoaTorchStepOverrideValue",
            ),
            f"$r = New-WoaTorchStepOverrideValue -Value '{src}' -Dir '{woa}'",
            "Write-Output ('VALUE=' + $r.Value)",
            "Write-Output ('TEMPS=' + ($r.Temps -join ';'))",
        )
        out = _ps_kv(script)
        assert out["TEMPS"], "the created copy is not reported, so nothing can delete it"
        assert out["TEMPS"].startswith(str(woa)), "the copy must live under the uv-safe directory"
        assert out["VALUE"] == out["TEMPS"]

    def test_every_path_goes_through_the_uv_safe_helper(self):
        body = _ps_function(INSTALL_PS1, "New-WoaTorchStepOverrideValue")
        assert "$safe = foreach ($f in $files) { Get-UvSafePath $f }" in body

    def test_the_caller_deletes_the_copies_on_every_exit(self):
        swap = INSTALL_SRC.index(
            "New-WoaTorchStepOverrideValue -Value $_woaOverrideSaved -Dir $script:WoaDir"
        )
        tail = INSTALL_SRC[swap : swap + 2600]
        fin = tail.index("} finally {")
        assert (
            "foreach ($_woaTmp in $_woaOverrideTemps) { Remove-Item -LiteralPath $_woaTmp"
            in tail[fin:]
        )

    def test_the_woa_directory_is_published_to_script_scope(self):
        assert "$script:WoaDir = $WoaDir" in INSTALL_SRC


class TestSetupSwapsTheOverrideAroundItsOwnTorchInstall:
    """Restore-WoaResolverEnvironment puts the generated overrides.txt back before setup's CUDA
    trio is installed, so the same floors undid the same exact pins there."""

    def test_the_swap_wraps_the_install_and_restores_in_finally(self):
        restore = SETUP_SRC.index("\nRestore-WoaResolverEnvironment")
        swap = SETUP_SRC.index("New-WoaTorchStepOverrideValueParity -Value $_woaStepSaved")
        assert (
            restore < swap
        ), "the swap must come after the overrides are restored, or it swaps nothing"
        tail = SETUP_SRC[swap : swap + 3000]
        assert "Fast-Install @_cudaTrio" in tail
        fin = tail.index("} finally {")
        assert "if ($_woaStepSwapped) { $env:UV_OVERRIDE = $_woaStepSaved }" in tail[fin:]
        assert (
            "foreach ($_woaTmp in $_woaStepTemps) { Remove-Item -LiteralPath $_woaTmp" in tail[fin:]
        )

    def test_only_a_native_venv_swaps(self):
        assert "if ($WinArm64Venv -and $env:UV_OVERRIDE) {" in SETUP_SRC

    @requires_pwsh
    def test_the_parity_helper_drops_the_trio(self, tmp_path):
        src = tmp_path / "ovr.txt"
        src.write_text("torch>=2.4\ntorchvision>=0.19\npyarrow==21.0.0\n", encoding = "utf-8")
        script = _script(
            UV_SAFE_PATH,
            functions(
                SETUP_SRC,
                "Get-RequirementEntries",
                "Resolve-WoaOverrideLine",
                "New-WoaTorchStepOverrideValueParity",
            ),
            f"$r = New-WoaTorchStepOverrideValueParity -Value '{src}' -Dir '{tmp_path}'",
            "Get-Content -LiteralPath $r.Value | ForEach-Object { Write-Output $_ }",
        )
        lines = [l for l in _ps_ok(script).stdout.splitlines() if l.strip()]
        assert lines == ["pyarrow==21.0.0"], lines


class TestNvidiaEvidenceSurvivesTheFastPath:
    """When the manifest verifies, $SkipPythonDeps skips the whole dependency block, so
    $WinArm64EffectiveTorchIndexUrl is never set and the llama.cpp check read "no evidence": a
    transient nvidia-smi failure on a no-op update then deleted the working CUDA bundle."""

    def test_the_index_is_read_at_the_check_when_the_pass_did_not_run(self):
        block = slice_between(
            SETUP_SRC,
            "$_woaEvidenceIndex = if ($WinArm64EffectiveTorchIndexUrl)",
            "$_nvidiaEvidence = ",
        )
        assert "Get-PinnedTorchIndexUrl" in block
        assert "Get-PersistedWoaTorchIndex -VenvPath $VenvDir" in block
        assert "Get-WoaTorchIndexMarker" in block
        assert (
            block.index("Get-PinnedTorchIndexUrl")
            < block.index("Get-PersistedWoaTorchIndex")
            < block.index("Get-WoaTorchIndexMarker")
        ), "same order as the dependency pass"

    def test_the_evidence_uses_that_index(self):
        assert "(Test-WinArm64Venv) -and $_woaEvidenceIndex -and" in SETUP_SRC
        assert "Test-WoaPersistableIndex $_woaEvidenceIndex" in SETUP_SRC

    def test_the_check_sits_outside_the_dependency_guard(self):
        """The premise: if it were inside, the fast path would never reach it at all."""
        guard = SETUP_SRC.index("\nif (-not $SkipPythonDeps) {")
        depth = 0
        for i in range(guard + 1, len(SETUP_SRC)):
            depth += (SETUP_SRC[i] == "{") - (SETUP_SRC[i] == "}")
            if depth == 0:
                break
        assert SETUP_SRC.index("$_nvidiaEvidence = ") > i


def _uv_toml(body: str, where: str = "proj/uv.toml") -> dict:
    """One configuration file, written where uv would discover it."""
    return {where: body}


class TestThePyPIProbeHonoursUvConfiguration:
    """A direct HTTP probe can see pypi.org while uv, under a uv.toml with no-index or an
    exclusive default-index, cannot. "pypi" then skipped a usable wheelhouse wheel for one the
    resolve would never fetch."""

    def test_the_pyarrow_probe_is_gated(self):
        probe = INSTALL_SRC.index('Invoke-RestMethod -Uri "https://pypi.org/simple/pyarrow/"')
        assert "if (Test-WoaResolveReachesPyPI) { try {" in INSTALL_SRC[probe - 400 : probe]

    @staticmethod
    def _reaches(tmp_path, files: dict, env: dict) -> str:
        for name, body in files.items():
            (tmp_path / name).parent.mkdir(parents = True, exist_ok = True)
            (tmp_path / name).write_text(body, encoding = "utf-8")
        script = _script(
            clear_env(UV_POLICY_ENV),
            f"$env:APPDATA = '{tmp_path / 'appdata'}'",
            f"$env:ProgramData = '{tmp_path / 'programdata'}'",
            f"Set-Location -LiteralPath '{tmp_path / 'proj'}'",
            "\n".join(f"$env:{k} = '{v}'" for k, v in env.items()),
            # Read-WoaUvTomlIndexKeys scans for quotes, so its two scanners come with it.
            functions(
                INSTALL_SRC,
                "Test-WoaUrlIsPublicPyPI",
                "Remove-WoaTomlComment",
                "Split-WoaTomlKey",
                "Read-WoaUvTomlIndexKeys",
                "Get-WoaUvConfigIndexPolicy",
                "Test-WoaResolveReachesPyPI",
            ),
            "Write-Output (Test-WoaResolveReachesPyPI)",
        )
        (tmp_path / "proj").mkdir(exist_ok = True)
        return _ps_last(script)

    @requires_pwsh
    @pytest.mark.parametrize(
        "files, env, expected, why",
        [
            ({}, {}, "True", "nothing configured: PyPI"),
            (_uv_toml("no-index = true\n"), {}, "False", "project uv.toml no-index"),
            (_uv_toml(f'default-index = "{CORP_INDEX}"\n'), {}, "False", "exclusive default-index"),
            (_uv_toml(f'index-url = "{CORP_INDEX}"\n'), {}, "False", "the older spelling"),
            (_uv_toml("[pip]\nno-index = true\n"), {}, "False", "under [pip], which uv pip reads"),
            (
                _uv_toml(f'[[index]]\nurl = "{CORP_INDEX}"\ndefault = true\n'),
                {},
                "False",
                "an [[index]] with default = true replaces PyPI",
            ),
            (
                _uv_toml(f'[[index]]\nurl = "{CORP_INDEX}"\n'),
                {},
                "True",
                "an extra index leaves PyPI in play",
            ),
            (
                _uv_toml("[tool.uv]\nno-index = true\n", "proj/pyproject.toml"),
                {},
                "False",
                "pyproject [tool.uv]",
            ),
            (
                _uv_toml('[project]\nname = "x"\n', "proj/pyproject.toml"),
                {},
                "True",
                "a pyproject without [tool.uv] is ignored",
            ),
            (_uv_toml("no-index = true\n", "uv.toml"), {}, "False", "found in a parent directory"),
            (
                _uv_toml("no-index = true\n", "appdata/uv/uv.toml"),
                {},
                "False",
                "the user file",
            ),
            (
                {
                    **_uv_toml("no-index = false\n"),
                    **_uv_toml("no-index = true\n", "appdata/uv/uv.toml"),
                },
                {},
                "True",
                "project outranks user for a scalar",
            ),
            (
                _uv_toml("no-index = true\n"),
                {"UV_NO_CONFIG": "1"},
                "True",
                "UV_NO_CONFIG discovers nothing",
            ),
            (
                _uv_toml("no-index = true\n"),
                {"UV_DEFAULT_INDEX": PYPI},
                "True",
                "an index in the environment outranks every file",
            ),
            (
                _uv_toml("no-index = true\n", "other.toml"),
                {"UV_CONFIG_FILE": "__TMP__/other.toml"},
                "False",
                "UV_CONFIG_FILE names the one file read",
            ),
            (
                _uv_toml(f'index = [{{ url = "{CORP_INDEX}", default = true }}]\n'),
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
            ({}, {"UV_INDEX_URL": "HTTPS://PYPI.ORG/simple/"}, "True", "case does not matter"),
            (
                {},
                {"UV_INDEX_URL": "https://user:token@pypi.org/simple"},
                "True",
                "credentials do not hide the host",
            ),
            (
                {},
                {"PIP_INDEX_URL": CORP_INDEX},
                "True",
                "pip's variable: uv, which resolves here, never reads it",
            ),
            (
                {},
                {"UV_INDEX_URL": CORP_INDEX, "PIP_EXTRA_INDEX_URL": PYPI},
                "False",
                "a pip extra does not put PyPI back for uv",
            ),
            ({}, {"UV_NO_INDEX": "1"}, "False", "uv's no-index"),
            (
                {},
                {"UV_DEFAULT_INDEX": "https://test.pypi.org/simple"},
                "False",
                "TestPyPI does not carry these packages",
            ),
            (
                _uv_toml('default-index = "https://pypi.org.corp.example/simple"\n'),
                {},
                "False",
                "a subdomain lookalike in a config file",
            ),
            (
                _uv_toml(f'default-index = "{PYPI}"\n'),
                {},
                "True",
                "public PyPI named explicitly in a config file",
            ),
            # uv pip 0.10.7: [pip] scalars outrank top-level whatever the file order, and an
            # [[index]] with default = true outranks [pip].index-url.
            (
                _uv_toml("no-index = false\n[pip]\nno-index = true\n"),
                {},
                "False",
                "[pip].no-index = true beats the top-level false",
            ),
            (
                _uv_toml("no-index = true\n[pip]\nno-index = false\n"),
                {},
                "True",
                "[pip].no-index = false beats the top-level true",
            ),
            (
                _uv_toml("[pip]\nno-index = true\n\n[other]\nx = 1\nno-index = false\n"),
                {},
                "False",
                "a later section does not reopen the top level",
            ),
            (
                _uv_toml(f'index-url = "{PYPI}"\n[pip]\nindex-url = "{CORP_INDEX}"\n'),
                {},
                "False",
                "[pip].index-url beats the top-level index-url",
            ),
            (
                _uv_toml(f'index-url = "{CORP_INDEX}"\n[pip]\nindex-url = "{PYPI}"\n'),
                {},
                "True",
                "the other way round",
            ),
            (
                _uv_toml(
                    f'[[index]]\nurl = "{CORP_INDEX}"\ndefault = true\n'
                    f'[pip]\nindex-url = "{PYPI}"\n'
                ),
                {},
                "False",
                "[[index]] default = true beats [pip].index-url",
            ),
            (
                _uv_toml(
                    f'[pip]\nindex-url = "{CORP_INDEX}"\n'
                    f'[[index]]\nurl = "{PYPI}"\ndefault = true\n'
                ),
                {},
                "True",
                "and still does when it comes later in the file",
            ),
            (
                _uv_toml(f'no-index = true\n[pip]\nindex-url = "{PYPI}"\n'),
                {},
                "False",
                "no-index disables every registry, [pip].index-url included",
            ),
            (
                _uv_toml(
                    "[tool.uv]\nno-index = false\n[tool.uv.pip]\nno-index = true\n",
                    "proj/pyproject.toml",
                ),
                {},
                "False",
                "the same under [tool.uv.pip]",
            ),
            # An extra index adds to the default rather than replacing it, so PyPI named as one
            # is consulted.
            (
                {},
                {"UV_INDEX_URL": CORP_INDEX, "UV_EXTRA_INDEX_URL": PYPI},
                "True",
                "UV_EXTRA_INDEX_URL names PyPI beside a corporate default",
            ),
            (
                {},
                {"UV_DEFAULT_INDEX": CORP_INDEX, "UV_INDEX": f"https://mirror.test/simple {PYPI}"},
                "True",
                "UV_INDEX, space-separated",
            ),
            (
                {},
                {"PIP_INDEX_URL": CORP_INDEX, "PIP_EXTRA_INDEX_URL": PYPI},
                "True",
                "pip's spelling",
            ),
            (
                {},
                {
                    "UV_INDEX_URL": CORP_INDEX,
                    "UV_EXTRA_INDEX_URL": "https://pypi.org.corp.example/simple",
                },
                "False",
                "an extra that is not PyPI changes nothing",
            ),
            (
                _uv_toml(f'index-url = "{CORP_INDEX}"\nextra-index-url = ["{PYPI}"]\n'),
                {},
                "True",
                "extra-index-url in a config file",
            ),
            (
                _uv_toml(
                    f'[[index]]\nurl = "{CORP_INDEX}"\ndefault = true\n\n'
                    f'[[index]]\nurl = "{PYPI}"\n'
                ),
                {},
                "True",
                "a second [[index]] without default = true is an extra",
            ),
            (
                _uv_toml(f'no-index = true\nextra-index-url = ["{PYPI}"]\n'),
                {},
                "False",
                "no-index disables extras too",
            ),
            (
                _uv_toml(f'index-url = "{CORP_INDEX}"\n[pip]\nextra-index-url = ["{PYPI}"]\n'),
                {},
                "True",
                "under [pip]",
            ),
        ],
    )
    def test_where_the_resolve_will_look(self, tmp_path, files, env, expected, why):
        env = {k: v.replace("__TMP__", str(tmp_path)) for k, v in env.items()}
        assert self._reaches(tmp_path, files, env) == expected, why


class TestARedundantWheelLeavesTheManagedDirectoryToo:
    """Skipping the copy was not enough: the managed directory is prepended to UV_FIND_LINKS, so a
    copy already there still wins the tie over the upstream wheel."""

    def _redundant_branches(self):
        out = []
        for marker in (
            'substep "windows on arm: PyPI publishes $($wheel.Name) itself -- taking it from'
            ' there, not the wheelhouse."',
            'substep "windows on arm: PyPI publishes $name itself -- taking it from there, not'
            ' the wheelhouse."',
        ):
            i = INSTALL_SRC.index(marker)
            out.append(INSTALL_SRC[i : INSTALL_SRC.index("continue", i)])
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
        script = _script(
            f"$WoaWheelDir = '{managed}'",
            f"$wheel = Get-Item -LiteralPath '{src / 'tiktoken-0.9.0-cp313-cp313-win_arm64.whl'}'",
            line,
            "Write-Output ('SRC=' + (Test-Path -LiteralPath $wheel.FullName))",
            "Write-Output ('MANAGED=' + (Test-Path -LiteralPath"
            f" '{managed / 'tiktoken-0.9.0-cp313-cp313-win_arm64.whl'}'))",
        )
        done = _ps_ok(script, timeout = 60)
        assert "SRC=True" in done.stdout and "MANAGED=False" in done.stdout, done.stdout


class TestTheEarlyNvidiaProbesAreBounded:
    """Initialize-WoaNativeCudaTorch runs before the main GPU detection, and its two probes called
    nvidia-smi unbounded: a wedged driver hung the installer before the bounded probe it already
    had was ever reached. PowerShell does not hoist, so the helper must also be DEFINED first."""

    def test_both_probes_use_the_bounded_helper(self):
        for name in ("Test-WoaNvidiaPresent", "Get-WoaDriverCudaVersion"):
            body = _function_source(INSTALL_SRC, name)
            assert "Invoke-NvidiaSmiBounded $exe" in body, name
            assert "& $exe" not in body, f"{name} still calls nvidia-smi unbounded"

    def test_the_leaf_reads_the_shared_parse(self):
        body = _function_source(INSTALL_SRC, "Get-WoaDriverCudaLeaf")
        assert "Get-WoaDriverCudaVersion" in body and "& $exe" not in body

    def test_the_helper_is_defined_before_its_first_early_caller(self):
        assert INSTALL_SRC.index("function Invoke-NvidiaSmiBounded") < INSTALL_SRC.index(
            "function Test-WoaNvidiaPresent"
        )

    @staticmethod
    def _fake_nvidia_smi(directory, sh_body: str, cmd_body: str) -> None:
        """A fake nvidia-smi the host will actually execute.

        Windows has no shebang handling and PATHEXT covers no extensionless name, so a /bin/sh
        script here is not an executable at all and the probe finds nothing. .cmd is what PATHEXT
        does cover, and it is the idiom the rest of this suite already uses.
        """
        if os.name == "nt":
            path = directory / "nvidia-smi.cmd"
            path.write_text("@echo off\n" + cmd_body, encoding = "utf-8")
            return
        path = directory / "nvidia-smi"
        path.write_text("#!/bin/sh\n" + sh_body, encoding = "utf-8")
        path.chmod(0o755)

    @requires_pwsh
    def test_a_listing_with_a_gpu_row_is_present(self, tmp_path):
        self._fake_nvidia_smi(
            tmp_path,
            "echo 'GPU 0: NVIDIA RTX (UUID: GPU-1)'\n",
            "echo GPU 0: NVIDIA RTX (UUID: GPU-1)\n",
        )
        script = _script(
            _function_source(INSTALL_SRC, "Invoke-NvidiaSmiBounded"),
            # The probe resolves its executable through this; without it the call is
            # unresolved and the probe answers False for a reason the test is not about.
            _function_source(INSTALL_SRC, "Get-WoaNvidiaSmiPath"),
            _function_source(INSTALL_SRC, "Test-WoaNvidiaPresent"),
            f"$env:PATH = '{tmp_path}' + [System.IO.Path]::PathSeparator + $env:PATH",
            "Write-Output (Test-WoaNvidiaPresent)",
        )
        assert _ps_last(script) == "True"

    @requires_pwsh
    def test_a_hung_nvidia_smi_returns_within_the_bound(self, tmp_path):
        # ping, not timeout: timeout /t needs a console and fails when stdin is redirected.
        self._fake_nvidia_smi(tmp_path, "sleep 30\n", "ping -n 31 127.0.0.1 > nul\n")
        # The bound is the helper's default; the assertion is that the call comes back at all.
        script = _script(
            _function_source(INSTALL_SRC, "Invoke-NvidiaSmiBounded").replace(
                "[int]$TimeoutSec = 10", "[int]$TimeoutSec = 2"
            ),
            # Required, and easy to miss: an unresolved lookup makes Get-WoaDriverCudaVersion
            # return $null before it ever calls nvidia-smi, so the "[]" below would pass
            # without the timeout this test exists to bound ever being exercised.
            _function_source(INSTALL_SRC, "Get-WoaNvidiaSmiPath"),
            _function_source(INSTALL_SRC, "Get-WoaDriverCudaVersion"),
            f"$env:PATH = '{tmp_path}' + [System.IO.Path]::PathSeparator + $env:PATH",
            "$v = Get-WoaDriverCudaVersion",
            "Write-Output ('[' + ($v -join '.') + ']')",
        )
        assert _ps_last(script, timeout = 60) == "[]"


class TestTheAvProbeHonoursResolverPolicy:
    """A free-threaded install probed public PyPI for PyAV directly, so under UV_OFFLINE, no-index
    or an exclusive index the native path was selected for an av the resolve could never fetch.
    The pyarrow probe was already gated the same way."""

    @requires_pwsh
    @pytest.mark.parametrize("reaches, expected", [("$false", "False"), ("$true", "True")])
    def test_pypi_counts_only_where_the_resolve_looks(self, reaches, expected):
        script = _script(
            f"function Test-WoaResolveReachesPyPI {{ {reaches} }}",
            "function Test-WoaPyPIWheel"
            " { param([Parameter(ValueFromRemainingArguments=$true)]$a) $true }",
            "$script:WoaWheelhouse = $null",
            _function_source(INSTALL_SRC, "Test-WoaWheelAvailable"),
            "Write-Output (Test-WoaWheelAvailable -Project 'av' -PythonMinor '3.14'"
            " -AbiTag 'cp314t')",
        )
        assert _ps_last(script) == expected


class TestARedundantWheelMeansTheSameVersionOnPyPI:
    """The guard compared with >=, so a wheelhouse tiktoken 0.13.0 was dropped once PyPI published
    0.14.0; extras.txt pins tiktoken==0.13.0, which 0.14.0 cannot satisfy, and the dependency pass
    then filtered tiktoken out. A newer upstream release leaves the staged copy alone."""

    @staticmethod
    def _redundant(pypi_versions):
        listing = "\n".join(
            f'<a href="tiktoken-{v}-cp313-cp313-win_arm64.whl">x</a>' for v in pypi_versions
        )
        script = _script(
            invoke_restmethod(listing),
            "function Test-WoaResolveReachesPyPI { $true }",
            functions(
                INSTALL_SRC,
                "Test-WoaVersionAtLeast",
                "Test-WoaWheelTags",
                "Test-WoaWheelTagsUsable",
                "Test-WoaPyPIWheel",
                "Test-WoaWheelhouseWheelIsRedundant",
            ),
            "Write-Output (Test-WoaWheelhouseWheelIsRedundant"
            " -Name 'tiktoken-0.13.0-cp313-cp313-win_arm64.whl' -PyTag 'cp313' -AbiTag 'cp313')",
        )
        return _ps_last(script)

    @requires_pwsh
    @pytest.mark.parametrize(
        "pypi, expected, why",
        [
            (["0.13.0"], "True", "the same version: upstream's copy is the one to use"),
            (["0.14.0"], "False", "newer only: an exact pin on 0.13.0 still needs ours"),
            (["0.12.0"], "False", "older only"),
            (["0.12.0", "0.13.0", "0.14.0"], "True", "the same version among others"),
        ],
    )
    def test_the_verdict(self, pypi, expected, why):
        assert self._redundant(pypi) == expected, why

    def test_the_guard_asks_for_the_exact_version(self):
        body = _function_source(INSTALL_SRC, "Test-WoaWheelhouseWheelIsRedundant")
        assert "-Floor $fields[1] -AllowAgnostic -Exact" in body


class TestTheManagedScanReadsThePlatformTag:
    """The offline cache is a directory the user fills, and a win_amd64 wheel there passed the
    Python and ABI checks alone: its project left the drop list, uv rejected the foreign platform,
    and the resolve fell through to an unbuildable ARM64 sdist."""

    def test_the_platform_is_checked_before_the_abi(self):
        block = slice_between(INSTALL_SRC, "$WoaWheelNames = @{}", "$WoaDropCandidates = @(")
        plat = block.index("$platTags = $parts[-1] -split")
        assert "($platTags -contains 'win_arm64') -or ($platTags -contains 'any')" in block
        assert plat < block.index("$abiTags = $parts[-2]")

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, listed",
        [
            ("tiktoken-0.13.0-cp313-cp313-win_arm64.whl", True),
            ("tiktoken-0.13.0-cp313-cp313-win_amd64.whl", False),
            ("hf_transfer-0.1.9-cp38-abi3-win_arm64.whl", True),
            ("brotli-1.1.0-py3-none-any.whl", True),
            ("brotli-1.1.0-py3-none-manylinux_2_17_aarch64.whl", False),
        ],
    )
    def test_the_scan_executed(self, tmp_path, name, listed):
        (tmp_path / name).write_text("", encoding = "utf-8")
        # Anchored on the code that follows rather than on its comment.
        scan = slice_between(
            INSTALL_SRC,
            "        $WoaWheelNames = @{}",
            "        foreach ($_woaProvided in $script:WoaPyPIProvided.Keys) {",
        )
        script = _script(
            f"$WoaWheelDir = '{tmp_path}'",
            "$WoaWheelTag = 'cp313'; $WoaWheelAbi = 'cp313'; $WoaWheelStable = $true;"
            " $WoaWheelMinor = 13",
            scan,
            "Write-Output ('KEYS=' + (($WoaWheelNames.Keys | Sort-Object) -join ','))",
        )
        keys = _ps_kv(script)["KEYS"]
        project = name.split("-")[0].replace("_", "-").lower()
        assert (project in keys.split(",")) is listed, (name, keys)


class TestAFinalBuildOutranksADevelopmentBuildOfTheSameRelease:
    """On equal numeric releases the tie was broken by a raw string comparison, where '.' sorts
    after '+', so 2.15.0.dev20260901+cu134 beat 2.15.0+cu134 and the installer pinned a
    development build and enabled prereleases for it. PEP 440 orders the dev build first."""

    @staticmethod
    def _pick(source, fn, versions):
        parity = fn != "Get-WoaCudaWheelVersion"
        lines = [
            invoke_restmethod(_torch_links(*versions)),
            JOIN_URL_RETURNS_BASE,
            functions(
                source,
                "Test-WoaWheelTagsParity" if parity else "Test-WoaWheelTags",
                "Test-WoaPairsWithTorchParity" if parity else "Test-WoaWheelPairsWithTorch",
                fn,
            ),
        ]
        if parity:
            lines.append(f"$v = {fn} -IndexUrl 'https://i.test' -PyTag 'cp313' -AbiTag 'cp313'")
        else:
            lines.append(
                f"$v = {fn} -IndexUrl 'https://i.test' -PythonMinor '3.13' -AbiTag 'cp313'"
            )
        lines.append("Write-Output ('[' + $v + ']')")
        return _ps_last(_script(*lines))[1:-1]

    @requires_pwsh
    @pytest.mark.parametrize(
        "fn, source",
        [("Get-WoaCudaWheelVersion", "INSTALL"), ("Get-WoaCudaWheelVersionParity", "SETUP")],
    )
    @pytest.mark.parametrize(
        "versions, expected, why",
        [
            (
                ["2.15.0.dev20260901%2Bcu134", "2.15.0%2Bcu134"],
                "2.15.0+cu134",
                "the final build, listed second",
            ),
            (
                ["2.15.0%2Bcu134", "2.15.0.dev20260901%2Bcu134"],
                "2.15.0+cu134",
                "the final build, listed first",
            ),
            (
                ["2.15.0.dev20260901%2Bcu134", "2.15.0.dev20260904%2Bcu134"],
                "2.15.0.dev20260904+cu134",
                "among dev builds the later stamp",
            ),
            (
                ["2.15.0.dev20260904%2Bcu134", "2.15.0.dev20260901%2Bcu134"],
                "2.15.0.dev20260904+cu134",
                "in either order",
            ),
            (
                ["2.14.0%2Bcu134", "2.15.0.dev20260901%2Bcu134"],
                "2.15.0.dev20260901+cu134",
                "a newer release still wins as a dev build",
            ),
            (
                ["2.15.0rc1%2Bcu134", "2.15.0%2Bcu134"],
                "2.15.0+cu134",
                "a release candidate ranks below the final build",
            ),
            (["2.15.0%2Bcu134", "2.15.0rc1%2Bcu134"], "2.15.0+cu134", "in either order"),
            (
                ["2.15.0a1%2Bcu134", "2.15.0b1%2Bcu134", "2.15.0rc1%2Bcu134"],
                "2.15.0rc1+cu134",
                "rc above beta above alpha",
            ),
            (
                ["2.15.0rc2%2Bcu134", "2.15.0rc1%2Bcu134"],
                "2.15.0rc2+cu134",
                "a later number wins within a kind",
            ),
            (
                ["2.15.0rc1%2Bcu134", "2.15.0.dev20260901%2Bcu134"],
                "2.15.0rc1+cu134",
                "and a dev build sits below every prerelease",
            ),
            (
                ["2.15.0b1%2Bcu134", "2.15.0a1%2Bcu134"],
                "2.15.0b1+cu134",
                "beta above alpha, listed first",
            ),
            (
                ["2.15.0a2%2Bcu134", "2.15.0b1%2Bcu134"],
                "2.15.0b1+cu134",
                "beta above a later alpha",
            ),
        ],
    )
    def test_the_pick(self, fn, source, versions, expected, why):
        src = INSTALL_SRC if source == "INSTALL" else SETUP_SRC
        assert self._pick(src, fn, versions) == expected, why


class TestTheWheelsCudaMajorMustNotExceedTheDrivers:
    """The NVIDIA channel was accepted on interpreter tags alone, so a CUDA 12 driver selected the
    +cu134 wheel: the install finished and torch could not initialise CUDA. A CUDA 13 runtime does
    not run on a CUDA 12 driver on Windows."""

    @staticmethod
    def _native(driver, wheel = "2.14.0+cu134"):
        script = native_probe_script(
            driver = "$null" if driver is None else "@(" + ", ".join(str(d) for d in driver) + ")",
            driver_leaf = "'cu128'",
            stubs = (
                f"$script:WoaNvidiaTorchIndexUrls = @('{NV_GA}')",
                "function Test-WoaCudaWheel { param($IndexUrl, $PythonMinor, $AbiTag, $Project)"
                " $IndexUrl -like '*nvidia*' }",
                "function Get-WoaCudaWheelVersion"
                f" {{ param([Parameter(ValueFromRemainingArguments=$true)]$a) '{wheel}' }}",
            ),
        )
        out = _ps_kv(script)
        return out["NATIVE"], out.get("MSG", "")

    @requires_pwsh
    @pytest.mark.parametrize(
        "driver, wheel, native, why",
        [
            ((12, 8), "2.14.0+cu134", "False", "a CUDA 12.8 driver cannot run a cu134 wheel"),
            ((13, 0), "2.14.0+cu134", "True", "a CUDA 13.0 driver can"),
            ((13, 4), "2.14.0+cu134", "True", "and a matching one"),
            ((12, 8), "2.10.0+cu128", "True", "the same major is fine"),
            ((11, 8), "2.10.0+cu128", "False", "cu128 on an 11.8 driver is the classic case"),
            (None, "2.14.0+cu134", "True", "no driver answer: the probe decides"),
        ],
    )
    def test_the_gate(self, driver, wheel, native, why):
        got, msg = self._native(driver, wheel)
        assert got == native, (why, msg)
        if native == "False":
            assert "Update the NVIDIA driver" in msg


class TestAStagingFailureAfterTheVenvIsAStop:
    """Once the native ARM64 venv exists, clearing the flag and continuing sent the torch step to
    the driver-derived index, which has no win_arm64 wheel: a guaranteed failure three steps later
    with nothing to say why. Now it stops here and says what to do."""

    def test_the_block_exits_with_the_reason(self):
        start = INSTALL_SRC.index(
            "        if (-not $script:WoaNativeCudaTorch) {\n"
            "            # The venv is already native ARM64"
        )
        block = INSTALL_SRC[start : start + 900]
        assert 'return (Exit-InstallFailure "windows on arm: pyarrow could not be staged")' in block
        assert "UNSLOTH_WOA_NATIVE=0" in block
        assert "continuing without the native stack" not in INSTALL_SRC


# The cutoff swap that wraps install.ps1's own trio command.
INSTALL_CUTOFF_SWAP_START = (
    '                if ($script:WoaNativeCudaTorch -and $VenvPlatform -eq "win-arm64") {\n'
    "                    # The probe read the index page"
)
CUTOFF_RESTORE = (
    "foreach ($_woaCutoffName in @($_woaCutoffSaved.Keys))"
    ' { Set-Item "Env:$_woaCutoffName" $_woaCutoffSaved[$_woaCutoffName] }'
)


class TestTheExactCudaPinIgnoresAnUploadCutoff:
    """UV_EXCLUDE_NEWER limits candidates by upload time. The probe reads the index page, which
    carries no dates, and pins exactly, so an inherited cutoff rejected the selected wheel and the
    native install aborted. Removed for that one command and restored after, in both installers;
    an installed package audits fine under a cutoff, so the later passes are safe."""

    @pytest.mark.parametrize(
        "src, marker",
        [
            ("INSTALL", "-Dir $script:WoaDir"),
            ("SETUP", "New-WoaTorchStepOverrideValueParity -Value $_woaStepSaved"),
        ],
    )
    def test_removed_before_and_restored_in_finally(self, src, marker):
        text = INSTALL_SRC if src == "INSTALL" else SETUP_SRC
        tail = text[text.index(marker) :][:3000]
        fin = tail.index("} finally {")
        assert (
            'foreach ($_woaCutoffName in @("UV_EXCLUDE_NEWER", "UV_EXCLUDE_NEWER_PACKAGE"))'
            in tail[:fin]
        )
        assert 'Remove-Item "Env:$_woaCutoffName"' in tail[:fin]
        assert CUTOFF_RESTORE in tail[fin:]

    @requires_pwsh
    def test_the_swap_executed(self):
        """The two loops, lifted out and run: the variable is gone inside and back after."""
        script = _script(
            SUBSTEP_NOOP,
            "$script:WoaNativeCudaTorch = $true; $VenvPlatform = 'win-arm64';"
            " $_woaCutoffSaved = @{}",
            "$env:UV_EXCLUDE_NEWER = '2026-01-01'",
            slice_between(INSTALL_SRC, INSTALL_CUTOFF_SWAP_START, "                try {"),
            "Write-Output ('INSIDE=[' + $env:UV_EXCLUDE_NEWER + ']')",
            CUTOFF_RESTORE,
            "Write-Output ('AFTER=[' + $env:UV_EXCLUDE_NEWER + ']')",
        )
        out = _ps_ok(script).stdout
        assert "INSIDE=[]" in out and "AFTER=[2026-01-01]" in out


class TestARebasedFileReferenceIsAUri:
    """`pkg @ file:../wheels/p.whl` was rebased to `file:` plus a raw path, and under a spaced
    profile that URL ended at the space and every later uv resolution failed."""

    @requires_pwsh
    @pytest.mark.parametrize("src", ["INSTALL", "SETUP"])
    def test_a_space_in_the_base_survives(self, tmp_path, src):
        base = tmp_path / "First Last" / "woa"
        base.mkdir(parents = True)
        text = INSTALL_SRC if src == "INSTALL" else SETUP_SRC
        script = _script(
            _function_source(text, "Resolve-WoaOverrideLine"),
            "Write-Output ('[' + (Resolve-WoaOverrideLine -Line 'pkg @ file:../wheels/p.whl'"
            f" -BaseDir '{base}') + ']')",
        )
        got = _ps_last(script)[1:-1]
        expected = "pkg @ " + (tmp_path / "First Last" / "wheels" / "p.whl").resolve().as_uri()
        assert got == expected

    @requires_pwsh
    @pytest.mark.parametrize("src", ["INSTALL", "SETUP"])
    @pytest.mark.parametrize(
        "line, marker, why",
        [
            (
                'pkg @ file:../wheels/p.whl ; python_version < "3.13"',
                ' ; python_version < "3.13"',
                "a marker rides after the URI",
            ),
            (
                "pkg @ file:../wheels/p.whl;python_version",
                "",
                "no whitespace before the semicolon: part of the path (PEP 508)",
            ),
        ],
    )
    def test_a_marker_is_kept_aside_while_the_path_is_rebased(
        self, tmp_path, src, line, marker, why
    ):
        """The marker was captured with the target and handed to GetFullPath and System.Uri, so it
        was either encoded into the URL or made the fallback rebase against the wrong base."""
        base = tmp_path / "woa"
        base.mkdir(parents = True)
        text = INSTALL_SRC if src == "INSTALL" else SETUP_SRC
        script = _script(
            _function_source(text, "Resolve-WoaOverrideLine"),
            f"Write-Output ('[' + (Resolve-WoaOverrideLine -Line '{line}'"
            f" -BaseDir '{base}') + ']')",
        )
        got = _ps_last(script)[1:-1]
        leaf = "p.whl" if marker else "p.whl;python_version"
        # pathlib percent-encodes the ";" that System.Uri keeps in the path (a sub-delim).
        uri = (tmp_path / "wheels" / leaf).resolve().as_uri().replace("%3B", ";")
        assert got == "pkg @ " + uri + marker, why

    @requires_pwsh
    def test_a_marked_url_reference_is_left_alone(self, tmp_path):
        script = _script(
            _function_source(INSTALL_SRC, "Resolve-WoaOverrideLine"),
            "Write-Output ('[' + (Resolve-WoaOverrideLine"
            " -Line 'pkg @ https://x.test/a.whl ; os_name == \"nt\"'"
            f" -BaseDir '{tmp_path}') + ']')",
        )
        assert _ps_last(script)[1:-1] == 'pkg @ https://x.test/a.whl ; os_name == "nt"'

    def test_the_two_copies_are_identical(self):
        install, setup = _ps_copies("Resolve-WoaOverrideLine")
        assert install == setup


class TestTheMergedOverrideFileDoesNotOutliveTheRun:
    """It copies caller lines, which can carry credentials, into $StudioHome\\woa. It is now
    removed when the run ends, on failure too, and any stale copy before the next is written."""

    def test_the_lifecycle_is_wired(self):
        assert "function Remove-WoaMergedOverrides" in SETUP_SRC
        restore = _function_source(SETUP_SRC, "Restore-WoaResolverEnvironment")
        assert (
            'Remove-Item -LiteralPath (Join-Path $woaDir "overrides.merged.txt") -Force' in restore
        ), "a stale copy is removed first"
        assert (
            "$script:WoaMergedOverrides = $_woaMerged" in restore
        ), "the written one is recorded for removal"
        assert "Remove-WoaMergedOverrides" in _function_source(SETUP_SRC, "Exit-SetupFailure")
        assert SETUP_SRC.rstrip().endswith(
            "Remove-WoaMergedOverrides"
        ), "the last statement of the script"

    @requires_pwsh
    def test_the_removal(self, tmp_path):
        merged = tmp_path / "overrides.merged.txt"
        merged.write_text("secret @ https://user:token@x.test/w.whl\n", encoding = "utf-8")
        script = _script(
            _function_source(SETUP_SRC, "Remove-WoaMergedOverrides"),
            f"$script:WoaMergedOverrides = '{merged}'",
            "Remove-WoaMergedOverrides",
            f"Write-Output (Test-Path -LiteralPath '{merged}')",
        )
        assert _ps_last(script) == "False"

    @requires_pwsh
    def test_a_failure_exit_still_reports_and_removes(self, tmp_path):
        """Exit-SetupFailure itself, not a stub: the cleanup call sits after the param block (a
        statement before it turns `param` into a command and the function into a crash), so the
        desktop app still gets its [TAURI:ERROR] line and the file is gone."""
        merged = tmp_path / "overrides.merged.txt"
        merged.write_text("secret @ https://user:token@x.test/w.whl\n", encoding = "utf-8")
        done = _ps(
            _script(
                functions(SETUP_SRC, "Remove-WoaMergedOverrides", "Exit-SetupFailure"),
                f"$script:WoaMergedOverrides = '{merged}'",
                '$env:UNSLOTH_TAURI_MODE = "1"',
                'Exit-SetupFailure "boom" -Code 3',
                'Write-Output "REACHED_UNREACHABLE"',
            )
        )
        assert done.returncode == 3, done.stderr
        assert "[TAURI:ERROR] boom" in done.stdout, done.stdout + done.stderr
        assert "REACHED_UNREACHABLE" not in done.stdout
        assert not merged.exists()


class TestNativeNeedsAPairedTorchvision:
    """torchvision is part of the stack. An index whose torch had no torchvision paired with it
    still took the native path and left torchvision to a floor, so the exact torch pin and an
    unpaired torchvision resolved against each other after the ARM64 venv existed. The pairing is
    part of the gate now: the next index is tried, and with none pairing the x64 stack is kept."""

    GA = NV_GA
    NIGHTLY = NV_NIGHTLY

    @classmethod
    def _native(cls, torch_by_index, vision_by_index):
        def table(d):
            return "@{ " + "; ".join(f"'{k}' = '{v}'" for k, v in d.items()) + " }"

        script = native_probe_script(
            driver = "@(13, 4)",
            stubs = (
                f"$script:WoaNvidiaTorchIndexUrls = @('{cls.GA}', '{cls.NIGHTLY}')",
                f"$script:Torch = {table(torch_by_index)}",
                f"$script:Vision = {table(vision_by_index)}",
                "function Test-WoaCudaWheel { param($IndexUrl, $PythonMinor, $AbiTag, $Project)"
                " [bool]$script:Torch[$IndexUrl] }",
                "function Get-WoaCudaWheelVersion { param($IndexUrl, $PythonMinor, $AbiTag,"
                " $Project, $PairWith)",
                "  if ($Project -eq 'torchvision') { return $script:Vision[$IndexUrl] }",
                "  if ($Project -eq 'torchaudio') { return '' }",
                "  return $script:Torch[$IndexUrl] }",
            ),
            outputs = (
                "Write-Output ('INDEX=' + $script:WoaTorchIndexUrl)",
                "Write-Output ('TORCH=' + $script:WoaTorchWheelVersion)",
                "Write-Output ('VISION=' + $script:WoaVisionWheelVersion)",
            ),
        )
        return _ps_kv(script)

    @requires_pwsh
    def test_a_paired_index_is_taken_with_both_pins(self):
        out = self._native({self.GA: "2.14.0+cu134"}, {self.GA: "0.29.0+cu134"})
        assert out["NATIVE"] == "True"
        assert out["INDEX"] == self.GA
        assert (out["TORCH"], out["VISION"]) == ("2.14.0+cu134", "0.29.0+cu134")

    @requires_pwsh
    def test_the_first_paired_index_wins(self):
        """Order is the maintenance story: the official index retires the NVIDIA one by
        existing."""
        out = self._native(
            {self.GA: "2.14.0+cu134", self.NIGHTLY: "2.15.0.dev20260905+cu134"},
            {self.GA: "0.29.0+cu134", self.NIGHTLY: "0.30.0.dev20260905+cu134"},
        )
        assert out["INDEX"] == self.GA
        assert (out["TORCH"], out["VISION"]) == ("2.14.0+cu134", "0.29.0+cu134")

    @requires_pwsh
    def test_an_unpaired_index_yields_to_the_next(self):
        out = self._native(
            {self.GA: "2.14.0+cu134", self.NIGHTLY: "2.15.0.dev20260905+cu134"},
            {self.GA: "", self.NIGHTLY: "0.30.0.dev20260905+cu134"},
        )
        assert out["NATIVE"] == "True"
        assert out["INDEX"] == self.NIGHTLY, out["MSG"]
        assert (out["TORCH"], out["VISION"]) == (
            "2.15.0.dev20260905+cu134",
            "0.30.0.dev20260905+cu134",
        )
        assert "no torchvision paired with it; trying the next index" in out["MSG"]

    @requires_pwsh
    def test_with_no_pairing_index_the_x64_stack_is_kept(self):
        out = self._native(
            {self.GA: "2.14.0+cu134", self.NIGHTLY: "2.15.0.dev20260905+cu134"},
            {self.GA: "", self.NIGHTLY: ""},
        )
        assert out["NATIVE"] == "False"
        assert out["INDEX"] == ""
        assert "no index pairs a torchvision" in out["MSG"] and "x64 stack" in out["MSG"]

    def test_the_vision_pin_is_the_gates_answer(self):
        """One probe, not two: the pin the trio installs is the version the gate accepted."""
        fn = _function_source(INSTALL_SRC, "Initialize-WoaNativeCudaTorch")
        assert fn.count('-Project "torchvision"') == 1
        assert "$script:WoaVisionWheelVersion = $_woaVisionVersion" in fn
        assert "leaving torchvision unpinned" not in fn


class TestAnUpdateKeepsTheInstalledPairWhenTheIndexLags:
    """setup.ps1's fresh-shell probe pinned the index's newest torch and left torchvision at its
    floor when no build paired with it. The venv already exists there, so the installed pair is
    kept instead; with nothing installed to keep, the floor stays and the log says why."""

    @staticmethod
    def _block():
        # Anchored on the code that follows rather than on its comment.
        return slice_between(
            SETUP_SRC,
            '$WinArm64TorchSpec = "torch>=2.4"',
            '$_tritonSpec = if ($WinArm64Venv) { "triton-windows>=3.8.0.post28" }',
        )

    @staticmethod
    def _venv_with(tmp_path, installed):
        """A REAL venv, with real .dist-info for whatever `installed` names.

        The block runs `<VenvDir>/Scripts/python.exe -c ...` by that exact name, and a /bin/sh
        script called python.exe is not an executable on Windows, so a fake came back empty there
        and all three cases asserted the floor rather than the branch they were written for. On
        POSIX the interpreter lands in bin/, so Scripts/python.exe is linked to it.
        """
        venv = tmp_path / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv)],
            check = True,
            capture_output = True,
            timeout = 300,
        )
        sites = list(venv.glob("Lib/site-packages")) + list(venv.glob("lib/*/site-packages"))
        assert sites, f"no site-packages under {venv}"
        for name, version in installed.items():
            info = sites[0] / f"{name}-{version}.dist-info"
            info.mkdir(parents = True)
            (info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8"
            )
        if os.name != "nt":
            scripts = venv / "Scripts"
            scripts.mkdir(exist_ok = True)
            (scripts / "python.exe").symlink_to(venv / "bin" / "python")
        return venv

    def _run(self, tmp_path, installed):
        venv = self._venv_with(tmp_path, installed)
        script = _script(
            substep_collector(),
            f"$VenvDir = '{venv}'",
            "$WinArm64Venv = $true",
            "$WinArm64NoAudio = $true",
            "$WinArm64EffectiveTorchIndexUrl = 'https://i.test'",
            "function Get-WoaCudaWheelVersionParity { param($IndexUrl, $PyTag, $AbiTag, $Project,"
            " $PairWith)",
            "  if ($Project -eq 'torchvision') { return '' }",
            "  if ($Project -eq 'torchaudio') { return '' }",
            "  return '2.15.0.dev20260905+cu134' }",
            functions(SETUP_SRC, "Test-WoaPairsWithTorchParity", "Test-WoaAudioMatchesTorchParity"),
            self._block(),
            "Write-Output ('TORCH=' + $WinArm64TorchSpec)",
            "Write-Output ('VISION=' + $WinArm64VisionSpec)",
            "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
        )
        return _ps_kv(script)

    @requires_pwsh
    def test_the_installed_pair_is_kept(self, tmp_path):
        out = self._run(tmp_path, {"torch": "2.14.0+cu134", "torchvision": "0.29.0+cu134"})
        assert out["TORCH"] == "torch==2.14.0+cu134", out["MSG"]
        assert out["VISION"] == "torchvision==0.29.0+cu134"
        assert "keeping the installed torch 2.14.0+cu134 and torchvision 0.29.0+cu134" in out["MSG"]

    @requires_pwsh
    def test_an_installed_pair_that_does_not_pair_is_not_kept(self, tmp_path):
        out = self._run(tmp_path, {"torch": "2.14.0+cu134", "torchvision": "0.28.0+cu134"})
        assert out["TORCH"] == "torch==2.15.0.dev20260905+cu134"
        assert out["VISION"] == "torchvision>=0.19"
        assert "no installed pair can be kept" in out["MSG"]

    @requires_pwsh
    def test_nothing_installed_leaves_the_floor_and_says_so(self, tmp_path):
        out = self._run(tmp_path, {})
        assert out["VISION"] == "torchvision>=0.19"
        assert "no installed pair can be kept" in out["MSG"]


class TestASuppliedWheelUnderAnotherNameIsReadFromItsArchive:
    """UNSLOTH_PYARROW_WHEEL saved as .bin or with no extension was rejected by the probe on its
    file name, although staging reads the wheel name from the archive and accepts that shape. The
    probe now reconstructs the name the same way before the project and tag checks."""

    @staticmethod
    def _archive(path, dist_info, tag):
        import zipfile
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                f"{dist_info}.dist-info/WHEEL",
                f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: {tag}\n",
            )
            zf.writestr(f"{dist_info}.dist-info/METADATA", "Metadata-Version: 2.1\nName: pyarrow\n")
        return path

    @staticmethod
    def _probe(wheel):
        return _ps_last(
            pyarrow_source_script(
                wheelhouse = "'https://example.test/wheels'",
                reaches_pypi = "$false",
                lifts = ("Get-WheelFileNameFromArchive",),
                preamble = (f"$env:UNSLOTH_PYARROW_WHEEL = '{wheel}'",),
            )
        )[1:-1]

    @requires_pwsh
    @pytest.mark.parametrize(
        "name, tag, expected, why",
        [
            ("pyarrow.bin", "cp313-cp313-win_arm64", "local", "a .bin download of the right wheel"),
            ("pyarrow_wheel", "cp313-cp313-win_arm64", "local", "no extension at all"),
            (
                "pyarrow.bin",
                "cp313-cp313-win_amd64",
                "",
                "the archive says x64, whatever the file is called",
            ),
            ("pyarrow.bin", "cp312-cp312-win_arm64", "", "or another interpreter"),
        ],
    )
    def test_the_name_comes_from_the_archive(self, tmp_path, name, tag, expected, why):
        wheel = self._archive(tmp_path / name, "pyarrow-21.0.0", tag)
        assert self._probe(str(wheel)) == expected, why

    @requires_pwsh
    def test_a_file_that_is_not_a_wheel_archive_is_still_ignored(self, tmp_path):
        bogus = tmp_path / "pyarrow.bin"
        bogus.write_bytes(b"not a zip at all")
        assert self._probe(str(bogus)) == ""

    @requires_pwsh
    def test_a_real_wheel_name_still_passes(self, tmp_path):
        wheel = self._archive(
            tmp_path / "pyarrow-21.0.0-cp313-cp313-win_arm64.whl",
            "pyarrow-21.0.0",
            "cp313-cp313-win_arm64",
        )
        assert self._probe(str(wheel)) == "local"


class TestAReselectedInterpreterIsProbedBeforeItIsTaken:
    """After the re-probe flips native mode, Find-CompatiblePython runs again with the new arch
    preference and ranks the requested minor first. It could hand back an ARM64 3.12 whose probe
    had failed while the flip came from 3.13, and the 3.13 answers were then carried into a 3.12
    venv. The reselected interpreter is probed too, and the accepted answer restored."""

    @staticmethod
    def _block():
        # Anchored on the code either side rather than on the comments that head them.
        return slice_between(
            INSTALL_SRC,
            "    $WoaProbedMinor = $PythonVersion",
            "    if ($script:WoaNativeCudaTorch -and $DetectedPython -and $DetectedPython.Arch -eq"
            ' "x86_64") {',
        )

    def _run(self, detected, reselected, native_minors):
        natives = ", ".join(f"'{m}'" for m in native_minors)
        script = _script(
            "$PythonVersion = '3.12'",
            substep_collector(),
            "$script:Probes = @()",
            "function step { param($a, $b, $c) }",
            "function Get-HostMachineArch { 'arm64' }",
            "function Remove-IndexUrlCredentials { param($u) $u }",
            "function Test-PythonFreeThreaded { param($PythonExe) $false }",
            "function Remove-SkippedPython { param($p) $p }",
            f"$script:Natives = @({natives})",
            "$script:WoaNativeCudaTorch = $false",
            "function Initialize-WoaNativeCudaTorch { param($PythonMinor, $FreeThreaded)",
            "  $script:Probes += $PythonMinor",
            "  $script:WoaNativeCudaTorch = [bool]($script:Natives -contains $PythonMinor)",
            "  $script:WoaTorchIndexUrl = if ($script:WoaNativeCudaTorch) { 'https://i.test' }"
            " else { $null } }",
            f"$DetectedPython = @{{ Version = '{detected[0]}'; Path = 'p{detected[0]}';"
            f" Arch = '{detected[1]}' }}",
            f"function Find-CompatiblePython {{ @{{ Version = '{reselected[0]}';"
            f" Path = 'p{reselected[0]}'; Arch = '{reselected[1]}' }} }}",
            self._block(),
            "Write-Output ('PY=' + $DetectedPython.Version)",
            "Write-Output ('NATIVE=' + $script:WoaNativeCudaTorch)",
            "Write-Output ('PROBED=' + $WoaProbedMinor)",
            "Write-Output ('PROBES=' + ($script:Probes -join ','))",
            "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
        )
        return _ps_kv(script)

    @requires_pwsh
    def test_a_reselected_minor_without_a_stack_is_not_taken(self):
        # Requested 3.12 (no stack); detected 3.13 goes native; the reselection offers ARM64 3.12.
        out = self._run(("3.13", "arm64"), ("3.12", "arm64"), ["3.13"])
        assert out["PY"] == "3.13", out["MSG"]
        assert out["NATIVE"] == "True" and out["PROBED"] == "3.13"
        assert (
            out["PROBES"] == "3.13,3.12,3.13"
        ), "probed the offer, then restored the accepted answer"
        assert "keeping Python 3.13, which has one" in out["MSG"]

    @requires_pwsh
    def test_a_reselected_minor_with_a_stack_is_taken(self):
        out = self._run(("3.13", "arm64"), ("3.12", "arm64"), ["3.13", "3.12"])
        assert out["PY"] == "3.12" and out["NATIVE"] == "True" and out["PROBED"] == "3.12"
        assert out["PROBES"] == "3.13,3.12"

    @requires_pwsh
    def test_the_same_interpreter_back_is_not_probed_again(self):
        out = self._run(("3.13", "x86_64"), ("3.13", "arm64"), ["3.13"])
        assert out["PY"] == "3.13" and out["NATIVE"] == "True"
        assert out["PROBES"] == "3.13"


class TestTheResolverVariablesDoNotOutliveTheInstaller:
    """Under `irm | iex` the process-scoped UV_OVERRIDE, UV_FIND_LINKS and PIP_FIND_LINKS set for
    the native stack were the caller's own session variables and stayed set, so every later
    `uv pip` in that shell resolved with Studio's override file and wheelhouse. They are
    snapshotted before the first assignment and put back in the script-level finally."""

    def test_the_snapshot_precedes_the_first_assignment(self):
        snap = INSTALL_SRC.index("$script:WoaResolverEnvSaved = @{")
        first = INSTALL_SRC.index('$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")')
        assert snap < first
        block = INSTALL_SRC[snap : snap + 200]
        for name in ("UV_OVERRIDE", "UV_FIND_LINKS", "PIP_FIND_LINKS"):
            assert f"{name} = $env:{name}" in block, name

    def test_the_restore_is_in_the_script_level_finally(self):
        run = INSTALL_SRC.index("    Install-UnslothStudio @args\n} finally {")
        assert "$script:WoaResolverEnvSaved" in INSTALL_SRC[run:]
        reset = INSTALL_SRC.rindex("$script:WoaResolverEnvSaved = $null\n", 0, run)
        assert (
            INSTALL_SRC.index("try {", reset) < run
        ), "cleared right before the run, so an earlier session value cannot leak in"

    @staticmethod
    def _restore_block():
        # Anchored on the code either side rather than on the comments that head them.
        return slice_between(
            INSTALL_SRC,
            "    if ($script:WoaResolverEnvSaved) {",
            "\n    Remove-Item Env:UNSLOTH_KEPT_TORCH -ErrorAction SilentlyContinue",
        )

    @requires_pwsh
    def test_the_callers_values_come_back_and_absent_ones_are_removed(self):
        script = _script(
            "$env:UV_OVERRIDE = 'C:\\studio\\overrides.txt'",
            "$env:UV_FIND_LINKS = 'C:\\studio\\wheels,C:\\mine'",
            "$env:PIP_FIND_LINKS = 'C:\\studio\\wheels'",
            "$script:WoaResolverEnvSaved = @{ UV_OVERRIDE = $null;"
            " UV_FIND_LINKS = 'C:\\mine'; PIP_FIND_LINKS = $null }",
            self._restore_block(),
            "Write-Output ('OV=' + [string]$env:UV_OVERRIDE)",
            "Write-Output ('UV=' + [string]$env:UV_FIND_LINKS)",
            "Write-Output ('PIP=' + [string]$env:PIP_FIND_LINKS)",
            "Write-Output ('SAVED=' + [string]($null -eq $script:WoaResolverEnvSaved))",
        )
        assert _ps_kv(script) == {"OV": "", "UV": "C:\\mine", "PIP": "", "SAVED": "True"}

    @requires_pwsh
    def test_nothing_snapshotted_touches_nothing(self):
        script = _script(
            "$env:UV_FIND_LINKS = 'C:\\mine'",
            "$script:WoaResolverEnvSaved = $null",
            self._restore_block(),
            "Write-Output ('UV=' + [string]$env:UV_FIND_LINKS)",
        )
        assert _ps_last(script) == "UV=C:\\mine"


class TestTheDependencyIndexFollowsTheResolverPolicy:
    """The trio install passed a hard-coded public PyPI as the extra index for torch's shared
    dependencies, while Invoke-InstallCommand clears every inherited index setting whenever
    --default-index is given. A caller with an exclusive corporate index or no-index therefore had
    public PyPI searched on their behalf, and a network that blocks it failed after the ARM64 venv
    existed. The dependency index is now what the policy names."""

    @classmethod
    def _args(
        cls,
        tmp_path,
        files,
        env,
        source = None,
        resolver = "uv",
    ):
        src = INSTALL_SRC if source is None else source
        for name, body in files.items():
            (tmp_path / name).parent.mkdir(parents = True, exist_ok = True)
            (tmp_path / name).write_text(body, encoding = "utf-8")
        (tmp_path / "proj").mkdir(exist_ok = True)
        script = _script(
            clear_env(UV_POLICY_ENV),
            f"$env:APPDATA = '{tmp_path / 'appdata'}'",
            f"$env:ProgramData = '{tmp_path / 'programdata'}'",
            f"Set-Location -LiteralPath '{tmp_path / 'proj'}'",
            "\n".join(f"$env:{k} = '{v}'" for k, v in env.items()),
            functions(
                src,
                "Remove-WoaTomlComment",
                "Split-WoaTomlKey",
                "Read-WoaUvTomlIndexKeys",
                "Get-WoaUvConfigIndexPolicy",
                "Get-WoaDependencyIndexArgs",
            ),
            f"Write-Output ('[' + ((Get-WoaDependencyIndexArgs -Resolver '{resolver}')"
            " -join '|') + ']')",
        )
        return _ps_last(script)[1:-1]

    PYPI = "--extra-index-url|https://pypi.org/simple"
    CORP = "--extra-index-url|https://pypi.corp.test/simple"

    @requires_pwsh
    @pytest.mark.parametrize(
        "body, expected, why",
        [
            (
                f'[[index]]\nurl = "{CORP_INDEX}"\nexplicit = true\n',
                "--extra-index-url|https://pypi.org/simple",
                "an explicit index is not handed to the trio resolve; PyPI stays the default",
            ),
            (
                f'[[index]]\nurl = "{CORP_INDEX}"\nexplicit = true\ndefault = true\n',
                "",
                "explicit and default is doubt, and doubt names nothing",
            ),
        ],
    )
    def test_an_explicit_index_is_never_an_extra_index_url(self, tmp_path, body, expected, why):
        assert self._args(tmp_path, {"proj/uv.toml": body}, {}) == expected, why

    @requires_pwsh
    @pytest.mark.parametrize(
        "files, env, expected, why",
        [
            ({}, {}, PYPI, "nothing configured: public PyPI"),
            ({}, {"UV_NO_INDEX": "1"}, "", "uv no-index: the wheelhouse is the whole source"),
            ({}, {"PIP_NO_INDEX": "true"}, PYPI, "pip's variable, which uv never reads"),
            ({}, {"UV_NO_INDEX": "0"}, PYPI, "a false flag is not set"),
            (
                {},
                {"UV_DEFAULT_INDEX": CORP_INDEX},
                CORP,
                "an exclusive default replaces PyPI",
            ),
            (
                {},
                {"UV_INDEX_URL": CORP_INDEX + "/"},
                CORP + "/",
                "the older spelling, kept as written",
            ),
            ({}, {"PIP_INDEX_URL": CORP_INDEX}, PYPI, "pip's default is not uv's"),
            (
                {},
                {"UV_INDEX_URL": CORP_INDEX, "PIP_EXTRA_INDEX_URL": PYPI_URL},
                CORP,
                "a pip extra does not put PyPI back for uv",
            ),
            (
                {},
                {"UV_DEFAULT_INDEX": CORP_INDEX, "UV_EXTRA_INDEX_URL": PYPI_URL},
                CORP + "|" + PYPI,
                "PyPI named as an extra stays in play, after the default",
            ),
            (
                {},
                {"UV_EXTRA_INDEX_URL": CORP_INDEX},
                PYPI + "|" + CORP,
                "an extra alone adds to the PyPI default",
            ),
            (
                {},
                {"UV_INDEX": "https://a.test/simple https://b.test/simple"},
                PYPI + "|--extra-index-url|https://a.test/simple"
                "|--extra-index-url|https://b.test/simple",
                "UV_INDEX is a space-separated list",
            ),
            (_uv_toml("no-index = true\n"), {}, "", "project uv.toml no-index"),
            (
                _uv_toml(f'default-index = "{CORP_INDEX}"\n'),
                {},
                CORP,
                "project default-index",
            ),
            (
                _uv_toml(f'default-index = "{CORP_INDEX}"\nextra-index-url = ["{PYPI_URL}"]\n'),
                {},
                CORP + "|" + PYPI,
                "config default plus config extra",
            ),
            (
                _uv_toml(
                    f'[[index]]\nurl = "{CORP_INDEX}"\ndefault = true\n\n'
                    '[[index]]\nurl = "https://extra.test/simple"\n'
                ),
                {},
                CORP + "|--extra-index-url|https://extra.test/simple",
                "[[index]] entries",
            ),
            (
                _uv_toml(f'default-index = "{CORP_INDEX}"\n'),
                {"UV_DEFAULT_INDEX": "https://env.test/simple"},
                "--extra-index-url|https://env.test/simple",
                "the environment beats the file",
            ),
            (
                _uv_toml(f'default-index = "{CORP_INDEX}"\n'),
                {"UV_NO_CONFIG": "1"},
                PYPI,
                "UV_NO_CONFIG hides the file, as it does for uv",
            ),
            (
                {},
                {"UV_DEFAULT_INDEX": PYPI_URL, "UV_EXTRA_INDEX_URL": PYPI_URL},
                PYPI,
                "a duplicate is listed once",
            ),
        ],
    )
    def test_the_index_arguments(self, tmp_path, files, env, expected, why):
        assert self._args(tmp_path, files, env) == expected, why

    @requires_pwsh
    @pytest.mark.parametrize(
        "files, env, expected, why",
        [
            ({}, {}, PYPI, "nothing configured: public PyPI"),
            ({}, {"PIP_NO_INDEX": "1"}, "", "pip no-index"),
            ({}, {"PIP_INDEX_URL": CORP_INDEX}, CORP, "pip's default replaces PyPI"),
            (
                {},
                {"PIP_INDEX_URL": CORP_INDEX, "PIP_EXTRA_INDEX_URL": PYPI_URL},
                CORP + "|" + PYPI,
                "pip's extra",
            ),
            ({}, {"UV_NO_INDEX": "1"}, PYPI, "uv's variable, which pip never reads"),
            ({}, {"UV_INDEX_URL": CORP_INDEX}, PYPI, "uv's default is not pip's"),
            (_uv_toml("no-index = true\n"), {}, PYPI, "uv's configuration files are not pip's"),
        ],
    )
    def test_the_pip_fallback_reads_pips_policy(self, tmp_path, files, env, expected, why):
        assert self._args(tmp_path, files, env, resolver = "pip") == expected, why

    @requires_pwsh
    def test_setup_answers_the_same(self, tmp_path):
        files = _uv_toml(f'default-index = "{CORP_INDEX}"\nextra-index-url = ["{PYPI_URL}"]\n')
        assert self._args(tmp_path, files, {}, SETUP_SRC) == self.CORP + "|" + self.PYPI
        assert self._args(tmp_path, {}, {"UV_NO_INDEX": "1"}, SETUP_SRC) == ""
        assert self._args(tmp_path, {}, {"PIP_NO_INDEX": "1"}, SETUP_SRC, resolver = "pip") == ""

    @pytest.mark.parametrize(
        "name",
        [
            "Remove-WoaTomlComment",
            "Split-WoaTomlKey",
            "Read-WoaUvInlineIndexArray",
            "Read-WoaUvTomlIndexKeys",
            "Get-WoaUvConfigIndexPolicy",
            "Test-WoaUvIndexPolicyUnreadable",
            "Get-WoaDependencyIndexArgs",
        ],
    )
    def test_the_two_copies_match(self, name):
        install, setup = _ps_copies(name)
        assert install == setup

    def test_both_install_paths_use_it_and_neither_hard_codes_pypi(self):
        trio = INSTALL_SRC[INSTALL_SRC.index("# NVIDIA's index publishes only the trio") :][:900]
        assert "$_woaDependencyIndexArgs = @(Get-WoaDependencyIndexArgs)" in trio
        assert f'"--extra-index-url", "{PYPI_URL}"' not in trio
        # The whole block, not a fixed slice of it: a guard added ahead of the call pushed the
        # call past a 500-character window and failed an assertion that was still true.
        shared = SETUP_SRC[SETUP_SRC.index("$WinArm64IndexArgs = if ($WinArm64Venv) {") :]
        shared = shared[: shared.index("\n} else {")]
        assert (
            '$_woaResolver = if ($UseUv) { "uv" } else { "pip" }' in shared
        ), "the resolver that runs the install"
        assert "@(Get-WoaDependencyIndexArgs -Resolver $_woaResolver)" in shared
        assert f'"--extra-index-url", "{PYPI_URL}"' not in shared


class TestTheArmJobRunsForEveryRequirementsInput:
    """The path filter named constraints.txt alone, although the selected tests read pyproject.toml
    and the other files under studio/backend/requirements/. A change to one of those skipped the
    workflow, so a marker or pin regression merged without the ARM checks."""

    def test_both_filters_name_the_requirements_tree_and_pyproject(self):
        text = WORKFLOW.read_text(encoding = "utf-8")
        head = text[: text.index("workflow_dispatch:")]
        push = head[head.index("  push:") : head.index("  pull_request:")]
        pull = head[head.index("  pull_request:") :]
        for name, block in (("push", push), ("pull_request", pull)):
            assert "- 'pyproject.toml'" in block, name
            assert "- 'studio/backend/requirements/**'" in block, name


class TestATerminatingErrorStillRemovesTheMergedOverrides:
    """Exit-SetupFailure and the last statement of the script both remove the merged file, but a
    throw after the restore (the unsupported-Vulkan one, for instance) reaches neither. A trap at
    script scope removes it and rethrows."""

    def test_the_trap_precedes_the_restore(self):
        trap = SETUP_SRC.index("trap { Remove-WoaMergedOverrides; break }")
        assert trap < SETUP_SRC.index("\nRestore-WoaResolverEnvironment\n")

    @requires_pwsh
    def test_a_throw_after_the_restore_removes_the_file_and_still_fails(self, tmp_path):
        merged = tmp_path / "overrides.merged.txt"
        merged.write_text("secret @ https://user:token@x.test/w.whl\n", encoding = "utf-8")
        done = _ps(
            _script(
                _function_source(SETUP_SRC, "Remove-WoaMergedOverrides"),
                f"$script:WoaMergedOverrides = '{merged}'",
                "trap { Remove-WoaMergedOverrides; break }",
                'throw "Vulkan was requested, but no Windows ARM64 Vulkan bundle is published."',
                'Write-Output "REACHED_UNREACHABLE"',
            )
        )
        assert done.returncode != 0
        assert "REACHED_UNREACHABLE" not in done.stdout
        assert (
            "Vulkan was requested" in done.stderr + done.stdout
        ), "the error is rethrown, not swallowed"
        assert not merged.exists()


class TestAnUnwritableWheelDirectoryIsAStop:
    """The catch stood native mode down after the ARM64 venv existed, and the torch step then went
    to the driver-derived index, which has no win_arm64 wheel. It stops with the reason now, as the
    pyarrow staging failure does."""

    @staticmethod
    def _block():
        return slice_between(
            INSTALL_SRC,
            '    if ($script:WoaNativeCudaTorch) {\n        $WoaDir = Join-Path $StudioHome "woa"',
            "    if ($script:WoaNativeCudaTorch) {\n"
            '        if ($script:WoaPyarrowSource -eq "local") {',
        )

    def test_the_catch_exits_with_the_reason(self):
        block = self._block()
        assert (
            'return (Exit-InstallFailure "windows on arm: could not create $WoaWheelDir")' in block
        )
        assert "UNSLOTH_WOA_NATIVE=0" in block
        assert "falling back to the x64 stack" not in block

    @requires_pwsh
    def test_a_file_in_the_way_stops_the_install(self, tmp_path):
        (tmp_path / "woa").write_text("not a directory", encoding = "utf-8")
        script = _script(
            "$script:Lines = @()",
            "function Write-StudioLine { param($m, $ForegroundColor) $script:Lines += $m }",
            "function substep { param($m, $c) $script:Lines += $m }",
            'function Exit-InstallFailure { param($m, $c) return "STOPPED: $m" }',
            "function Install-Probe {",
            "  $script:WoaNativeCudaTorch = $true",
            f"  $StudioHome = '{tmp_path}'",
            self._block(),
            "  return 'CONTINUED'",
            "}",
            "$r = Install-Probe",
            "Write-Output ('RESULT=' + $r)",
            "Write-Output ('NATIVE=' + $script:WoaNativeCudaTorch)",
            "Write-Output ('MSG=' + ($script:Lines -join ' | '))",
        )
        out = _ps_kv(script)
        assert out["RESULT"].startswith("STOPPED: windows on arm: could not create"), out
        assert out["NATIVE"] == "True", "not silently stood down"
        assert "[ERROR]" in out["MSG"] and "UNSLOTH_WOA_NATIVE=0" in out["MSG"]


class TestANoIndexNativeTrioStillSeesItsSources:
    """Under UV_NO_INDEX the trio command carried --default-index, so Invoke-InstallCommand cleared
    UV_FIND_LINKS while --no-index stayed on: neither the CUDA index nor the staged wheelhouse was
    visible and the exact pin failed. The wheelhouse now rides on the command line, and UV_NO_INDEX
    yields for this one command and is put back after it."""

    @staticmethod
    def _extra_args(tmp_path, env):
        # Anchored on the code either side rather than on the comments that head them.
        block = slice_between(
            INSTALL_SRC,
            "                $_woaDependencyIndexArgs = @(Get-WoaDependencyIndexArgs)",
            "                if ($script:WoaTorchIsPrerelease -or"
            " ($script:WoaTorchIndexUrl -match 'nightly')) {",
        )
        script = _script(
            clear_env(UV_ONLY_INDEX_ENV),
            "$env:UV_NO_CONFIG = '1'",
            "\n".join(f"$env:{k} = '{v}'" for k, v in env.items()),
            SUBSTEP_NOOP,
            functions(
                INSTALL_SRC,
                "Get-UvSafePath",
                "Get-WoaUvConfigIndexPolicy",
                "Get-WoaDependencyIndexArgs",
            ),
            f"$script:WoaDir = '{tmp_path / 'woa'}'",
            block,
            "Write-Output ('[' + ($_torchExtraArgs -join '|') + ']')",
        )
        return _ps_last(script)[1:-1].split("|")

    @requires_pwsh
    def test_the_wheelhouse_is_named_on_the_command_line(self, tmp_path):
        args = self._extra_args(tmp_path, {})
        i = args.index("--find-links")
        assert args[i + 1] == str(tmp_path / "woa" / "wheels")
        assert "--extra-index-url" in args, "with no policy the dependencies come from PyPI as well"

    @requires_pwsh
    def test_under_no_index_the_wheelhouse_is_the_only_dependency_source(self, tmp_path):
        args = self._extra_args(tmp_path, {"UV_NO_INDEX": "1"})
        assert "--find-links" in args
        assert "--extra-index-url" not in args

    @staticmethod
    def _swap_block():
        return slice_between(
            INSTALL_SRC,
            INSTALL_CUTOFF_SWAP_START,
            "                try {\n                    $torchInstallExit ="
            ' Invoke-InstallCommandRetry -Label "install PyTorch"',
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "value, yields", [("1", True), ("true", True), ("0", False), ("false", False)]
    )
    def test_no_index_yields_for_the_command_and_is_put_back(self, value, yields):
        script = _script(
            substep_collector(),
            "$script:WoaNativeCudaTorch = $true",
            "$VenvPlatform = 'win-arm64'",
            f"$env:UV_NO_INDEX = '{value}'",
            "Remove-Item Env:UV_EXCLUDE_NEWER -ErrorAction SilentlyContinue",
            "Remove-Item Env:UV_EXCLUDE_NEWER_PACKAGE -ErrorAction SilentlyContinue",
            "$_woaCutoffSaved = @{}",
            self._swap_block(),
            "Write-Output ('DURING=' + [string]$env:UV_NO_INDEX)",
            CUTOFF_RESTORE,
            "Write-Output ('AFTER=' + [string]$env:UV_NO_INDEX)",
            "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
        )
        out = _ps_kv(script)
        assert out["DURING"] == ("" if yields else value)
        assert out["AFTER"] == value, "the caller's value comes back either way"
        assert ("UV_NO_INDEX yields" in out["MSG"]) is yields


class TestANoIndexNativeTrioStillSeesItsSourcesInSetup:
    """setup.ps1's trio step had the same gap as install.ps1's: Fast-Install clears UV_FIND_LINKS
    and PIP_FIND_LINKS whenever --index-url is given and leaves UV_NO_INDEX alone, so under
    --no-index the update saw neither the wheelhouse nor the CUDA index."""

    @staticmethod
    def _index_args(tmp_path, env):
        (tmp_path / "woa" / "wheels").mkdir(parents = True)
        script = _script(
            clear_env(UV_INDEX_ENV),
            "$env:UV_NO_CONFIG = '1'",
            "\n".join(f"$env:{k} = '{v}'" for k, v in env.items()),
            UV_SAFE_PATH,
            functions(SETUP_SRC, "Get-WoaUvConfigIndexPolicy", "Get-WoaDependencyIndexArgs"),
            f"$StudioHome = '{tmp_path}'",
            "$WinArm64Venv = $true",
            "$UseUv = $true",
            f"$WinArm64TorchIndexUrl = '{NV_GA}'",
            "$WinArm64EffectiveTorchIndexUrl = $WinArm64TorchIndexUrl",
            "$WinArm64HandoffApplies = $true",
            "$env:UNSLOTH_WOA_TORCH_PRERELEASE = '0'",
            INDEX_ARGS_BLOCK,
            "Write-Output ('[' + ($WinArm64IndexArgs -join '|') + ']')",
        )
        return _ps_last(script)[1:-1].split("|")

    @requires_pwsh
    def test_the_wheelhouse_is_named_on_the_command_line(self, tmp_path):
        args = self._index_args(tmp_path, {})
        i = args.index("--find-links")
        assert args[i + 1] == str(tmp_path / "woa" / "wheels")
        assert "--extra-index-url" in args, "with no policy the dependencies come from PyPI as well"

    @requires_pwsh
    def test_under_no_index_the_wheelhouse_is_the_only_dependency_source(self, tmp_path):
        args = self._index_args(tmp_path, {"UV_NO_INDEX": "1"})
        assert "--find-links" in args
        assert "--extra-index-url" not in args

    def test_the_setting_fast_install_clears_is_the_one_put_on_the_command_line(self):
        body = _function_source(SETUP_SRC, "Fast-Install")
        assert "'UV_FIND_LINKS'" in body and "'PIP_FIND_LINKS'" in body
        assert "--find-links" in INDEX_ARGS_BLOCK

    @staticmethod
    def _swap_block():
        return slice_between(
            SETUP_SRC,
            "        if ($WinArm64Venv) {\n"
            "            # The pins are exact and the index page carries no upload dates",
            "        try {\n            if ($script:UnslothVerbose) {\n"
            "                Fast-Install @_cudaTrio",
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "value, yields", [("1", True), ("true", True), ("0", False), ("false", False)]
    )
    def test_no_index_yields_for_the_command_and_is_put_back(self, value, yields):
        script = _script(
            substep_collector(),
            "$WinArm64Venv = $true",
            f"$env:UV_NO_INDEX = '{value}'",
            "$env:UV_EXCLUDE_NEWER = '2026-01-01'",
            "Remove-Item Env:UV_EXCLUDE_NEWER_PACKAGE -ErrorAction SilentlyContinue",
            "$_woaCutoffSaved = @{}",
            self._swap_block(),
            "Write-Output ('DURING=' + [string]$env:UV_NO_INDEX)",
            "Write-Output ('CUTOFF=' + [string]$env:UV_EXCLUDE_NEWER)",
            CUTOFF_RESTORE,
            "Write-Output ('AFTER=' + [string]$env:UV_NO_INDEX)",
            "Write-Output ('CUTOFFAFTER=' + [string]$env:UV_EXCLUDE_NEWER)",
            "Write-Output ('MSG=' + ($script:Messages -join ' | '))",
        )
        out = _ps_kv(script)
        assert out["DURING"] == ("" if yields else value)
        assert out["AFTER"] == value, "the caller's value comes back either way"
        assert ("UV_NO_INDEX yields" in out["MSG"]) is yields
        assert (
            out["CUTOFF"] == "" and out["CUTOFFAFTER"] == "2026-01-01"
        ), "the cutoff swap is unchanged"

    @requires_pwsh
    def test_off_arm64_nothing_is_touched(self):
        script = _script(
            SUBSTEP_NOOP,
            "$WinArm64Venv = $false",
            "$env:UV_NO_INDEX = '1'",
            "$_woaCutoffSaved = @{}",
            self._swap_block(),
            "Write-Output ('DURING=' + [string]$env:UV_NO_INDEX + ' SAVED='"
            " + $_woaCutoffSaved.Count)",
        )
        assert _ps_last(script) == "DURING=1 SAVED=0"


class TestProbeWarningsDoNotPrintIndexCredentials:
    """Two warnings printed the candidate index URL as configured. An authenticated mirror that
    lags on torchvision, or whose torch outruns the driver, put its token in the installer output
    and the Tauri log."""

    URL = "https://user:s3cret@mirror.test/simple?token=abc"

    def _native(self, driver, vision):
        script = native_probe_script(
            driver = f"@({driver[0]}, {driver[1]})",
            stubs = (
                f"$env:UNSLOTH_TORCH_INDEX_URL = '{self.URL}'",
                "function Test-WoaCudaWheel"
                " { param($IndexUrl, $PythonMinor, $AbiTag, $Project) $true }",
                "function Get-WoaCudaWheelVersion { param($IndexUrl, $PythonMinor, $AbiTag,"
                " $Project, $PairWith)",
                f"  if ($Project -eq 'torchvision') {{ return '{vision}' }}",
                "  if ($Project -eq 'torchaudio') { return '' }",
                "  return '2.14.0+cu134' }",
            ),
            lifts = ("Remove-IndexUrlCredentials",),
        )
        return _ps_kv(script)

    @requires_pwsh
    def test_the_unpaired_torchvision_warning(self):
        out = self._native((13, 4), "")
        assert out["NATIVE"] == "False"
        assert "mirror.test" in out["MSG"] and "no torchvision paired" in out["MSG"]
        assert "s3cret" not in out["MSG"] and "token=abc" not in out["MSG"]

    @requires_pwsh
    def test_the_driver_warning(self):
        out = self._native((12, 8), "0.29.0+cu134")
        assert out["NATIVE"] == "False"
        assert "mirror.test" in out["MSG"] and "Update the NVIDIA driver" in out["MSG"]
        assert "s3cret" not in out["MSG"] and "token=abc" not in out["MSG"]


class TestTheNoAudioDecisionFollowsTheProbe:
    """On a fresh-shell update there is no handoff, so $WinArm64NoAudio was fixed to true even when
    the effective index pairs a torchaudio with the torch it selects: the trio was installed
    without it and an older audio build removed."""

    def test_the_probe_revises_the_decision(self):
        start = SETUP_SRC.index("$_woaAudioV = Get-WoaCudaWheelVersionParity")
        block = SETUP_SRC[start : start + 1400]
        assert (
            "$_woaProbeHasAudio = [bool]($_woaAudioV -and (Test-WoaAudioMatchesTorchParity"
            " -TorchVersion $_woaTorchV -AudioVersion $_woaAudioV))" in block
        )
        assert "$WinArm64NoAudio = -not $_woaProbeHasAudio" in block

    def test_the_parity_copy_has_not_drifted(self):
        def normalized(source, name):
            lines = [
                l.strip()
                for l in _function_source(source, name).splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            return "\n".join(lines[1:])  # the signature line carries the name

        assert normalized(INSTALL_SRC, "Test-WoaAudioMatchesTorch") == normalized(
            SETUP_SRC, "Test-WoaAudioMatchesTorchParity"
        )

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch, audio, matches",
        [
            ("2.14.0+cu134", "2.11.0+cu134", "False"),
            ("2.15.0.dev20260904+cu134", "2.15.0.dev20260904+cu134", "True"),
            ("2.11.0+cu134", "2.11.0+cu134", "True"),
        ],
    )
    def test_the_parity_predicate(self, torch, audio, matches):
        script = _script(
            _function_source(SETUP_SRC, "Test-WoaAudioMatchesTorchParity"),
            f"Write-Output (Test-WoaAudioMatchesTorchParity -TorchVersion '{torch}'"
            f" -AudioVersion '{audio}')",
        )
        assert _ps_last(script) == matches


class TestARetainedTorchaudioMustPairWithTheNewTorch:
    """The retention check compared major.minor only, while selection requires the dev stamp and
    CUDA tag to match too: a nightly torchaudio one stamp behind the replaced torch was kept,
    linked against a libtorch that is gone."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "torch, audio, kept",
        [
            ("2.15.0.dev20260904+cu134", "2.15.0.dev20260904+cu134", True),
            ("2.15.0.dev20260904+cu134", "2.15.0.dev20260902+cu134", False),
            ("2.14.0+cu134", "2.14.0+cu130", False),
            ("2.14.0+cu134", "2.11.0+cu134", False),
        ],
    )
    def test_the_predicate_pair(self, torch, audio, kept):
        script = _script(
            functions(SETUP_SRC, "Test-WoaAudioMatchesTorchParity", "Test-WoaPairsWithTorchParity"),
            f"$a = Test-WoaAudioMatchesTorchParity -TorchVersion '{torch}'"
            f" -AudioVersion '{audio}'",
            f"$b = Test-WoaPairsWithTorchParity -TorchVersion '{torch}' -OtherVersion '{audio}'"
            " -Project 'torchaudio'",
            "Write-Output ($a -and $b)",
        )
        assert _ps_last(script) == str(kept)


class TestTheArmJobFailsWhenARequiredTestSkips:
    """-rs only reports skipped cases, so a guard that stopped a test running on the very row it
    exists for still reported green."""

    def test_the_summary_line_is_read_back(self):
        text = WORKFLOW.read_text(encoding = "utf-8")
        start = text.index("- name: WoA installer and wheelhouse tests")
        step = text[start : text.index("- name: Uninstaller tests", start)]
        assert "-q -rs" in step
        assert "if ($rc -ne 0) { exit $rc }" in step
        assert r"if ($summary -match '(\d+) skipped')" in step
        assert "exit 1" in step[step.index("skipped") :]


class TestBothNvidiaSmiProbesSearchTheSameLocations:
    """The presence probe searched PATH, System32 and NVSMI while the version probe searched PATH
    and System32 only. On a host carrying nvidia-smi.exe under NVSMI alone, Test-WoaNvidiaPresent
    said yes and Get-WoaDriverCudaVersion returned $null, so Initialize-WoaNativeCudaTorch skipped
    the CUDA-major guard entirely. A guard that silently does not run is the failure this covers."""

    LOCATIONS = (
        r"$env:SystemRoot\System32\nvidia-smi.exe",
        r"$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    )

    def test_the_shared_helper_lists_every_supported_location(self):
        body = _function_source(INSTALL_SRC, "Get-WoaNvidiaSmiPath")
        assert "Get-Command nvidia-smi" in body
        for location in self.LOCATIONS:
            assert location in body, location

    @pytest.mark.parametrize("name", ("Test-WoaNvidiaPresent", "Get-WoaDriverCudaVersion"))
    def test_neither_probe_keeps_its_own_candidate_list(self, name):
        body = _function_source(INSTALL_SRC, name)
        assert "$exe = Get-WoaNvidiaSmiPath" in body, f"{name} does not use the shared lookup"
        # The point of the shared helper: a second list is what let the two disagree.
        for location in self.LOCATIONS:
            assert location not in body, f"{name} still hardcodes {location}"

    def test_the_helper_is_defined_before_both_callers(self):
        # PowerShell does not hoist, so definition order is load-bearing here.
        helper = INSTALL_SRC.index("function Get-WoaNvidiaSmiPath")
        for name in ("Test-WoaNvidiaPresent", "Get-WoaDriverCudaVersion"):
            assert helper < INSTALL_SRC.index(f"function {name}"), name

    def test_every_composed_script_injects_the_helper_its_bodies_call(self):
        """The pwsh tests paste real function bodies into a bare script, so a body that gains a
        call to a helper the composition does not also inject leaves that call unresolved.
        PowerShell writes an error and carries on with $null, so Get-WoaDriverCudaVersion returns
        $null BEFORE reaching nvidia-smi and the "[]" assertion still passes: a silent false pass,
        and on a host without pwsh the whole test skips. This guard needs no pwsh."""
        callers = [
            name
            for name in ("Test-WoaNvidiaPresent", "Get-WoaDriverCudaVersion")
            if "Get-WoaNvidiaSmiPath" in _function_source(INSTALL_SRC, name)
        ]
        assert callers, "neither probe routes through the shared lookup any more"
        own = pathlib.Path(__file__).read_text(encoding = "utf-8")
        for name in callers:
            for match in re.finditer(rf'_function_source\(INSTALL_SRC, "{re.escape(name)}"\)', own):
                # The _script(...) call this appears in, back to its opening paren.
                start = own.rindex("_script(", 0, match.start())
                block = own[start : own.index("\n        )", match.end())]
                assert '_function_source(INSTALL_SRC, "Get-WoaNvidiaSmiPath")' in block, (
                    f"a composed script injects {name}, which calls Get-WoaNvidiaSmiPath, "
                    "without injecting that helper"
                )

    def test_the_guard_this_protects_is_still_there(self):
        # If the CUDA-major check ever goes away, this whole class is pointless; say so loudly.
        body = _function_source(INSTALL_SRC, "Initialize-WoaNativeCudaTorch")
        assert "$_woaDriver = Get-WoaDriverCudaVersion" in body
        assert "if ($_woaDriver -and $_woaTorchVersion -match '\\+cu(\\d+)')" in body


class TestAnExplicitBlockIndexIsNotAGeneralExtra:
    """The block form `[[index]]` read url and default only, so `explicit = true` was flushed as a
    general extra and Get-WoaDependencyIndexArgs handed the trio resolve an --extra-index-url
    the user had restricted to explicitly pinned packages. Both readers, both scripts."""

    A = "https://a.corp.test/simple"
    B = "https://b.corp.test/simple"

    @staticmethod
    def _read(src, tmp_path, body, top):
        cfg = tmp_path / ("pyproject.toml" if top else "uv.toml")
        cfg.write_text(body, encoding = "utf-8")
        script = _script(
            functions(
                src,
                "Remove-WoaTomlComment",
                "Split-WoaTomlKey",
                "Read-WoaUvInlineIndexArray",
                "Read-WoaUvTomlIndexKeys",
            ),
            f"$p = Read-WoaUvTomlIndexKeys -Path '{cfg}' -Top '{top}'",
            "if ($null -eq $p) { Write-Output 'NULL' } else { "
            "Write-Output (([string]$p.DefaultIndex) + '|' + (@($p.ExtraIndexes) -join ',')) }",
        )
        return _ps_last(script)

    @requires_pwsh
    @pytest.mark.parametrize("src", [INSTALL_SRC, SETUP_SRC], ids = ["install.ps1", "setup.ps1"])
    @pytest.mark.parametrize(
        "body, top, expected, why",
        [
            (f'[[index]]\nurl = "{A}"\nexplicit = true\n', "", "|", "explicit alone: nothing"),
            (
                f'[[index]]\nurl = "{A}"\nexplicit = true\n\n[[index]]\nurl = "{B}"\n',
                "",
                f"|{B}",
                "the other entry is still an extra",
            ),
            (
                f'[[index]]\nurl = "{A}"\ndefault = true\n\n[[index]]\nurl = "{B}"\nexplicit = true\n',
                "",
                f"{A}|",
                "an explicit entry beside the default",
            ),
            (
                f'[[index]]\nurl = "{A}"\nexplicit = false\n',
                "",
                f"|{A}",
                "explicit = false is an extra",
            ),
            (
                f'[[tool.uv.index]]\nurl = "{A}"\nexplicit = true\n',
                "tool.uv",
                "|",
                "pyproject [[tool.uv.index]]",
            ),
            (
                f'[[index]]\nurl = "{A}"\nexplicit = true\ndefault = true\n',
                "",
                "NULL",
                "explicit AND default removes PyPI as the default: not modelled, so doubt",
            ),
        ],
    )
    def test_the_block_form(self, tmp_path, src, body, top, expected, why):
        assert self._read(src, tmp_path, body, top) == expected, why


class TestTheInlineIndexSpellingIsRead:
    """`index = [{ url = "...", default = true }]` is valid, documented uv config, and the parser
    refused it outright with `return $null`. Every downstream disagreement about Unreadable was a
    symptom: a corporate mirror written this way read as "cannot know", and
    Get-WoaDependencyIndexArgs then substituted public PyPI for it."""

    @staticmethod
    def _funcs(src):
        return functions(
            src,
            "Remove-WoaTomlComment",
            "Split-WoaTomlKey",
            "Read-WoaUvInlineIndexArray",
            "Read-WoaUvTomlIndexKeys",
        )

    @pytest.mark.parametrize(
        "value, want_default, want_extras",
        [
            (f'[{{ url = "{CORP_INDEX}", default = true }}]', CORP_INDEX, []),
            (f'[{{ url = "{PYPI}", default = true }}]', PYPI, []),
            (
                '[{ url = "https://a/simple" }, { url = "https://b/simple", default = true }]',
                "https://b/simple",
                ["https://a/simple"],
            ),
            (
                '[{ name = "corp", url = "https://c/simple", default = true }]',
                "https://c/simple",
                [],
            ),
            ("[]", None, []),
            # uv: explicit = true serves only packages pinned via [tool.uv.sources], so it is
            # neither the default nor an extra for the trio's dependencies.
            ('[{ url = "https://a/simple", explicit = true }]', None, []),
            (
                '[{ url = "https://a/simple", explicit = true }, { url = "https://b/simple" }]',
                None,
                ["https://b/simple"],
            ),
            ('[{ url = "https://a/simple", explicit = false }]', None, ["https://a/simple"]),
        ],
    )
    @pytest.mark.parametrize("src", [INSTALL_SRC, SETUP_SRC], ids = ["install.ps1", "setup.ps1"])
    @requires_pwsh
    def test_a_flat_inline_array_is_read(self, tmp_path, src, value, want_default, want_extras):
        cfg = tmp_path / "uv.toml"
        cfg.write_text(f"index = {value}\n", encoding = "utf-8")
        script = _script(
            self._funcs(src),
            f"$p = Read-WoaUvTomlIndexKeys -Path '{cfg}' -Top ''",
            "if ($null -eq $p) { Write-Output 'NULL' } else { "
            "Write-Output (([string]$p.DefaultIndex) + '|' + (@($p.ExtraIndexes) -join ',')) }",
        )
        assert _ps_last(script) == f"{want_default or ''}|{','.join(want_extras)}"

    @pytest.mark.parametrize(
        "value",
        [
            # Ambiguity that must stay Unreadable rather than be guessed at.
            # explicit AND default also removes PyPI as the default (uv docs): not modelled.
            '[{ url = "https://a/simple", explicit = true, default = true }]',
            '[{ url = "https://a/simple", explicit = "yes" }]',  # not a bool
            "[{ default = true }]",  # no url at all
            '[{ url = "https://a/simple", default = "yes" }]',  # not a bool
            '[{ url = { host = "a" } }]',  # nested table
            '[{ url = "https://a/simple" }, "https://b"]',  # a bare entry beside a table
            '[{ url = "https://a/simple" }',  # unbalanced / continues next line
        ],
    )
    @pytest.mark.parametrize("src", [INSTALL_SRC, SETUP_SRC], ids = ["install.ps1", "setup.ps1"])
    @requires_pwsh
    def test_anything_ambiguous_stays_unreadable(self, tmp_path, src, value):
        cfg = tmp_path / "uv.toml"
        cfg.write_text(f"index = {value}\n", encoding = "utf-8")
        script = _script(
            self._funcs(src),
            f"$p = Read-WoaUvTomlIndexKeys -Path '{cfg}' -Top ''",
            "Write-Output $(if ($null -eq $p) { 'NULL' } else { 'READ' })",
        )
        assert _ps_last(script) == "NULL"

    def test_the_parser_no_longer_refuses_the_key_outright(self):
        """No pwsh needed, so this runs everywhere. `return $null` on sight of the key was the
        defect; it must now go through the array reader, which still answers $null on doubt."""
        for src, label in ((INSTALL_SRC, "install.ps1"), (SETUP_SRC, "setup.ps1")):
            body = _function_source(src, "Read-WoaUvTomlIndexKeys")
            assert "if ($key -eq 'index') { return $null }" not in body, label
            assert "Read-WoaUvInlineIndexArray -Value $val" in body, label
            # The conservative answer is still reachable from the caller.
            assert "if ($null -eq $inline) { return $null }" in body, label

    def test_an_unreadable_policy_never_substitutes_public_pypi(self):
        """The consumer half. Get-WoaDependencyIndexArgs read NoIndex, DefaultIndex and
        ExtraIndexes but not Unreadable, so it fell through to the public PyPI default and
        silently overrode the mirror the file configured."""
        for src, label in ((INSTALL_SRC, "install.ps1"), (SETUP_SRC, "setup.ps1")):
            body = _function_source(src, "Get-WoaDependencyIndexArgs")
            assert "if ($cfg.Unreadable -and -not $default) { return @() }" in body, label
            # Ordering is the whole point: the guard must precede the PyPI fallback.
            assert body.index("$cfg.Unreadable") < body.index(f'"{PYPI}"'), label

    def test_the_site_that_cannot_read_config_stops_instead_of_guessing(self):
        """setup.ps1's trio runs under Fast-Install with --index-url, which sets UV_NO_CONFIG and
        scrubs UV_*, so uv cannot read the file itself and an empty answer would leave the trio
        index as the only source. install.ps1 calls uv directly and can take the empty answer."""
        block = SETUP_SRC[SETUP_SRC.index("$WinArm64IndexArgs = if ($WinArm64Venv) {") :]
        block = block[: block.index("\n} else {")]
        assert "Test-WoaUvIndexPolicyUnreadable" in block
        assert "Exit-SetupFailure" in block
        # The message has to be actionable: the file, the spelling, and both ways out.
        assert "UnreadablePath" in block
        assert "[[index]]" in block
        assert "UV_DEFAULT_INDEX" in block
        # install.ps1's trio step deliberately does NOT stop; it lets uv read the config.
        assert "Test-WoaUvIndexPolicyUnreadable" not in _function_source(
            INSTALL_SRC, "Get-WoaDependencyIndexArgs"
        )


class TestThePyPIProvidedHandoverIsExported:
    """What install.ps1 learned about PyPI serving a wheelhouse wheel reaches
    install_python_stack.py only through UNSLOTH_WOA_PYPI_PROVIDED."""

    @requires_pwsh
    @pytest.mark.parametrize(
        "table, expected, why",
        [
            (
                '@{ tiktoken = @("0.12.0"); "hf-transfer" = @("0.1.9", "0.1.9", "") }',
                "hf-transfer==0.1.9 tiktoken==0.12.0",
                "one name==version per finding, sorted, blanks and repeats dropped",
            ),
            ("@{}", "", "nothing found: assigned blank, never left over from an earlier run"),
        ],
    )
    def test_the_value_install_python_stack_reads(self, table, expected, why):
        script = _script(
            f"$script:WoaPyPIProvided = {table}",
            '$env:UNSLOTH_WOA_PYPI_PROVIDED = "stale==0.0.0"',
            slice_between(
                INSTALL_SRC,
                "$_woaProvidedPairs = @()",
                '$env:UNSLOTH_WOA_PYPI_PROVIDED = ($_woaProvidedPairs -join " ")',
                include_end = True,
            ),
            'Write-Output "[$env:UNSLOTH_WOA_PYPI_PROVIDED]"',
        )
        assert _ps_last(script) == f"[{expected}]", why


class TestFoldedCallerOverridesDoNotOutliveTheRun:
    """A caller file that clashes with one of ours is folded line by line, and those lines were
    written into woa\\overrides.txt: the file setup.ps1 restores on every fresh-shell update. A
    credentialed direct URL or a private index policy the caller set once was then on disk for
    good and applied to every later update. They go to a per-run file instead."""

    @staticmethod
    def _block() -> str:
        start = INSTALL_SRC.index("        $_woaOwnNames = @{}")
        end = INSTALL_SRC.index('$env:UV_OVERRIDE = ($_woaOverrideValue -join " ")', start)
        return INSTALL_SRC[start : INSTALL_SRC.index("\n", end)]

    def _run(
        self,
        tmp_path,
        caller_lines,
        stale = None,
    ):
        caller = tmp_path / "ov.txt"
        caller.write_text("\n".join(caller_lines) + "\n", encoding = "utf-8")
        managed = tmp_path / "overrides.txt"
        session = tmp_path / "overrides.session.txt"
        if stale is not None:
            session.write_text(stale, encoding = "utf-8")
        script = _script(
            functions(INSTALL_SRC, "Resolve-WoaOverrideLine", "Get-WoaRequirementEntries"),
            UV_SAFE_PATH,
            "$WoaOverrideLines = @('# generated', 'torch>=2.4')",
            f"$WoaOverrides = '{managed}'",
            f"$env:UV_OVERRIDE = '{caller}'",
            self._block(),
            'Write-Output ("OVERRIDE=" + $env:UV_OVERRIDE)',
            'Write-Output ("SESSION=" + $script:WoaSessionOverrides)',
        )
        out = _ps_ok(script).stdout.splitlines()
        value = [l for l in out if l.startswith("OVERRIDE=")][-1][len("OVERRIDE=") :].split()
        recorded = [l for l in out if l.startswith("SESSION=")][-1][len("SESSION=") :]
        return managed, session, value, recorded

    SECRET = "corp-pkg @ https://user:s3cret@pypi.corp.test/corp_pkg-1.0-py3-none-any.whl"

    @requires_pwsh
    def test_a_folded_credential_never_reaches_the_persistent_file(self, tmp_path):
        managed, session, value, recorded = self._run(tmp_path, ["torch==2.9.0", self.SECRET])
        assert "s3cret" not in managed.read_text(encoding = "utf-8")
        assert self.SECRET in session.read_text(encoding = "utf-8"), "still applied to this run"
        assert value == [str(managed), str(session)], "both files reach uv"
        assert recorded == str(session), "recorded, so the exit path can remove it"

    @requires_pwsh
    def test_a_stale_per_run_file_is_removed_even_when_nothing_folds(self, tmp_path):
        managed, session, value, recorded = self._run(
            tmp_path, ["brotli==1.1.0"], stale = "leftover==1\n"
        )
        assert not session.exists(), "an earlier interrupted run's copy would be re-read by uv"
        assert value[0] == str(managed) and str(session) not in value
        assert recorded == ""

    def test_the_exit_path_removes_it(self):
        """Beside the torch overrides file, which is deleted on exit for the same reason."""
        tail = INSTALL_SRC[INSTALL_SRC.index("try {\n    Install-UnslothStudio @args") :]
        assert (
            "Remove-Item -LiteralPath $script:WoaSessionOverrides -Force -ErrorAction SilentlyContinue"
            in tail
        )
        head = INSTALL_SRC[: INSTALL_SRC.index("try {\n    Install-UnslothStudio @args")]
        assert head.rstrip().endswith(
            "$script:WoaSessionOverrides = $null\n$script:TorchOverridesFile = $null"
        ), "reset before the outer try: under irm | iex an earlier value must not leak"

    def test_setup_restores_only_the_persistent_file(self):
        body = _function_source(SETUP_SRC, "Restore-WoaResolverEnvironment")
        assert 'Join-Path $woaDir "overrides.txt"' in body
        assert (
            'Remove-Item -LiteralPath (Join-Path $woaDir "overrides.session.txt")' in body
        ), "an interrupted install's copy is cleared, never restored"
        assert body.count("overrides.session.txt") == 1, "and referenced nowhere else"
