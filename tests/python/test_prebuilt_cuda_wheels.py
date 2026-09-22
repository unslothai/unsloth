# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The wheels we build ourselves, and the resolver that has to find them.

Upstream publishes prebuilt flash-attn / causal-conv1d / mamba-ssm wheels against torch 2.10 and
2.11, reused through 2.12. torch 2.13 broke the extension ABI (materialize_cow_storage, the
c10_cuda_check_implementation signature) and 2.14 broke it again, so from 2.13 on the wheels come
from .github/workflows/prebuilt-cuda-wheels.yml and a release on this repository.

Two halves have to agree for that to work at all, and they live in different languages in
different directories: the workflow decides what a wheel is CALLED, and wheel_utils decides what
the installer ASKS FOR. A disagreement between them is a 404 and a five-hour source build, and
neither half can notice it alone. So the central test here builds the filename both ways for
every cell the workflow can produce and asserts they are the same string.

The rest pins what must not move:

* 2.4 through 2.12 keep resolving to upstream, unchanged, including the 2.11/2.12 reuse;
* the override is Linux x86_64, cu13, cxx11abiTRUE only, so Windows, macOS, aarch64, CUDA 12 and
  a non-C++11-ABI torch keep exactly the behaviour they have today;
* the workflow stays dispatch-only, keeps `contents: write` on the publish job alone, and keeps
  every action pinned to a SHA, because it signs and publishes binaries under our identity.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "prebuilt-cuda-wheels.yml"
SCRIPTS = REPO / ".github" / "scripts"

sys.path.insert(0, str(REPO / "studio"))
sys.path.insert(0, str(REPO / "studio" / "backend"))

from utils import wheel_utils  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prebuilt_wheels = _load("prebuilt_wheels", SCRIPTS / "prebuilt_wheels.py")


def env(
    *,
    torch_mm: str = "2.13",
    cuda_major: str = "13",
    cxx11abi: str = "TRUE",
    python_tag: str = "cp313",
    platform_tag: str = "linux_x86_64",
) -> dict[str, str]:
    """The shape probe_torch_wheel_env() returns, with the fields the resolvers read."""
    return {
        "torch_mm": torch_mm,
        "cuda_major": cuda_major,
        "cxx11abi": cxx11abi,
        "python_tag": python_tag,
        "platform_tag": platform_tag,
        "torch_version": f"{torch_mm}.0+cu{cuda_major}0",
        "cuda_version": "13.0",
    }


OUR_BASE = "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"


def triggers(doc: dict) -> dict:
    """PyYAML is YAML 1.1, where a bare `on:` key parses as the boolean True."""
    return doc.get(True) if True in doc else doc.get("on")


# ── The two halves agree ──────────────────────────────────────────────────────


class TestWorkflowAndResolverAgree:
    def test_every_cell_the_workflow_builds_is_what_the_resolver_asks_for(self):
        """The one test this whole file exists for.

        The workflow renames its build output to prebuilt_wheels.wheel_name(...); the installer
        downloads the basename of unsloth_prebuilt_wheel_url(...). Nothing else checks that
        those are the same string, and if they ever differ the symptom is a 404 on a user's
        machine rather than a red run here.
        """
        checked = 0
        for package, spec in prebuilt_wheels.SPECS.items():
            for torch_version in prebuilt_wheels.TORCH_VERSIONS:
                for python_version in prebuilt_wheels.PYTHON_VERSIONS:
                    built = prebuilt_wheels.wheel_name(package, torch_version, python_version)
                    url = wheel_utils.unsloth_prebuilt_wheel_url(
                        filename_prefix = spec["dist"],
                        env = env(
                            torch_mm = prebuilt_wheels.torch_minor(torch_version),
                            python_tag = prebuilt_wheels.python_tag(python_version),
                        ),
                    )
                    assert url is not None, (package, torch_version, python_version)
                    assert url.rsplit("/", 1)[1] == built
                    assert url.startswith(OUR_BASE)
                    checked += 1
        # 3 packages x 2 torch minors x 3 interpreters. A silently emptied table would
        # otherwise pass this class without asserting anything.
        assert checked == 18

    def test_the_versions_published_are_the_versions_resolved(self):
        for package, spec in prebuilt_wheels.SPECS.items():
            assert wheel_utils._UNSLOTH_PREBUILT_VERSIONS[spec["dist"]] == spec["version"], package

    def test_the_two_torch_tables_are_the_same_set(self):
        built = {prebuilt_wheels.torch_minor(version) for version in prebuilt_wheels.TORCH_VERSIONS}
        assert built == set(wheel_utils._UNSLOTH_PREBUILT_TORCH_MM)

    def test_the_release_tag_the_workflow_defaults_to_is_the_one_we_resolve(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
        default = triggers(workflow)["workflow_dispatch"]["inputs"]["release_tag"]["default"]
        assert default == wheel_utils.UNSLOTH_PREBUILT_RELEASE_TAG


# ── The override fires exactly where it should ────────────────────────────────


class TestOverrideScope:
    @pytest.mark.parametrize("torch_mm", ["2.13", "2.14"])
    @pytest.mark.parametrize(
        "prefix,version",
        [("flash_attn", "2.8.4"), ("causal_conv1d", "1.7.0"), ("mamba_ssm", "2.3.2.post1")],
    )
    def test_exact_url(self, torch_mm, prefix, version):
        url = wheel_utils.unsloth_prebuilt_wheel_url(
            filename_prefix = prefix, env = env(torch_mm = torch_mm)
        )
        assert url == (
            f"{OUR_BASE}{prefix}-{version}+cu13torch{torch_mm}"
            "cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )

    @pytest.mark.parametrize("torch_mm", ["2.4", "2.9", "2.10", "2.11", "2.12", "2.15", "3.0"])
    def test_other_torch_minors_are_not_ours(self, torch_mm):
        assert (
            wheel_utils.unsloth_prebuilt_wheel_url(
                filename_prefix = "flash_attn", env = env(torch_mm = torch_mm)
            )
            is None
        )

    @pytest.mark.parametrize("platform_tag", ["linux_aarch64", "win_amd64", "macosx_11_0_arm64"])
    def test_only_linux_x86_64(self, platform_tag):
        assert (
            wheel_utils.unsloth_prebuilt_wheel_url(
                filename_prefix = "mamba_ssm", env = env(platform_tag = platform_tag)
            )
            is None
        )

    @pytest.mark.parametrize("cuda_major", ["11", "12", "14", ""])
    def test_only_cuda_13(self, cuda_major):
        assert (
            wheel_utils.unsloth_prebuilt_wheel_url(
                filename_prefix = "mamba_ssm", env = env(cuda_major = cuda_major)
            )
            is None
        )

    def test_only_cxx11_abi_true(self):
        assert (
            wheel_utils.unsloth_prebuilt_wheel_url(
                filename_prefix = "flash_attn", env = env(cxx11abi = "FALSE")
            )
            is None
        )

    def test_unknown_package_is_not_ours(self):
        assert wheel_utils.unsloth_prebuilt_wheel_url(filename_prefix = "xformers", env = env()) is None

    def test_no_env_is_not_ours(self):
        assert (
            wheel_utils.unsloth_prebuilt_wheel_url(filename_prefix = "flash_attn", env = None) is None
        )

    @pytest.mark.parametrize("python_tag", ["cp311", "cp312", "cp313"])
    def test_every_interpreter_we_build(self, python_tag):
        url = wheel_utils.unsloth_prebuilt_wheel_url(
            filename_prefix = "flash_attn", env = env(python_tag = python_tag)
        )
        assert url is not None and f"-{python_tag}-{python_tag}-" in url


# ── Nothing that worked before changed ────────────────────────────────────────


class TestBackwardsCompatible:
    def test_torch_210_still_resolves_upstream_flash_attn(self):
        url = wheel_utils.flash_attn_wheel_url(env(torch_mm = "2.10"))
        assert url == (
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/"
            "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )

    @pytest.mark.parametrize("torch_mm", ["2.11", "2.12"])
    def test_the_torch210_reuse_window_is_untouched(self, torch_mm):
        assert wheel_utils.prebuilt_wheel_torch_mm(torch_mm) == "2.10"
        url = wheel_utils.flash_attn_wheel_url(env(torch_mm = torch_mm))
        assert "torch2.10" in url
        assert "Dao-AILab" in url

    def test_torch_29_still_resolves_upstream(self):
        url = wheel_utils.flash_attn_wheel_url(env(torch_mm = "2.9"))
        assert "Dao-AILab/flash-attention/releases/download/v2.8.3/" in url
        assert "torch2.9" in url

    def test_upstream_causal_conv1d_url_is_unchanged_on_212(self):
        url = wheel_utils.direct_wheel_url(
            filename_prefix = "causal_conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
            env = env(torch_mm = "2.12"),
        )
        assert url == (
            "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/"
            "causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )

    def test_213_takes_over_the_same_call(self):
        """Same call site, same arguments, different answer only for the new minors."""
        url = wheel_utils.direct_wheel_url(
            filename_prefix = "causal_conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
            env = env(torch_mm = "2.13"),
        )
        assert (
            url
            == f"{OUR_BASE}causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )

    def test_a_windows_env_still_reaches_upstream_for_a_non_our_package(self):
        """The override must not become a Windows gate for anything else that uses this path."""
        url = wheel_utils.direct_wheel_url(
            filename_prefix = "something_else",
            package_version = "1.0",
            release_tag = "v1.0",
            release_base_url = "https://example.invalid/download",
            env = env(torch_mm = "2.13", platform_tag = "win_amd64"),
        )
        assert url == (
            "https://example.invalid/download/v1.0/"
            "something_else-1.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
        )

    def test_no_cuda_still_resolves_to_nothing(self):
        assert (
            wheel_utils.direct_wheel_url(
                filename_prefix = "flash_attn",
                package_version = "2.8.1",
                release_tag = "v2.8.1",
                release_base_url = "https://example.invalid/download",
                env = env(cuda_major = ""),
            )
            is None
        )


# ── The build plan ────────────────────────────────────────────────────────────


class TestMatrix:
    def test_defaults_are_both_torch_minors_on_cp313(self):
        include = prebuilt_wheels.build_matrix()
        assert len(include) == 6
        assert {cell["torch"] for cell in include} == {"2.13.0", "2.14.0"}
        assert {cell["python_tag"] for cell in include} == {"cp313"}
        assert {cell["package"] for cell in include} == set(prebuilt_wheels.SPECS)

    def test_extra_interpreters_are_separate_cells(self):
        include = prebuilt_wheels.build_matrix(pythons = "3.11,3.12,3.13")
        assert len(include) == 18
        assert len({cell["wheel_name"] for cell in include}) == 18

    def test_one_torch_minor_at_a_time(self):
        include = prebuilt_wheels.build_matrix(torches = "2.14.0")
        assert {cell["torch_mm"] for cell in include} == {"2.14"}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"packages": "flash-attn; rm -rf /"},
            {"packages": "torchvision"},
            {"torches": "2.12.0"},
            {"torches": "$(whoami)"},
            {"pythons": "3.9"},
            {"pythons": "3.13 && curl evil"},
        ],
    )
    def test_anything_not_on_the_allowlist_is_refused(self, kwargs):
        with pytest.raises(SystemExit):
            prebuilt_wheels.build_matrix(**kwargs)

    def test_every_cell_carries_what_the_steps_read(self):
        needed = {
            "package",
            "dist",
            "version",
            "repo",
            "ref",
            "submodules",
            "patch",
            "torch",
            "torch_mm",
            "python",
            "python_tag",
            "cuda_tag",
            "abi",
            "max_jobs",
            "nvcc_threads",
            "build_timeout",
            "build_env",
            "import_names",
            "wheel_name",
            "label",
        }
        for cell in prebuilt_wheels.build_matrix(pythons = "3.11,3.12,3.13"):
            assert needed <= set(cell), needed - set(cell)

    def test_sources_are_pinned_to_a_commit(self):
        """A branch or a floating tag would make a rebuild silently different from the last one,
        which for a signed binary is the one property worth keeping."""
        for package, spec in prebuilt_wheels.SPECS.items():
            assert re.fullmatch(r"[0-9a-f]{40}", spec["ref"]), (package, spec["ref"])

    def test_the_build_env_is_name_equals_value_pairs_only(self):
        """The workflow `export`s these. Anything with a space or a shell metacharacter in it
        would be an injection with a constant on the other end, which is still an injection."""
        for cell in prebuilt_wheels.build_matrix():
            for pair in cell["build_env"].split(" "):
                assert re.fullmatch(r"[A-Z0-9_]+=[A-Za-z0-9_;.,-]+", pair), pair

    def test_force_build_is_set_for_every_package(self):
        """Without it each setup.py downloads an upstream wheel and repackages it, which for
        torch 2.13 means publishing the very binary that does not load."""
        for package, spec in prebuilt_wheels.SPECS.items():
            forced = [key for key in spec["env"] if key.endswith("FORCE_BUILD")]
            assert len(forced) == 1, package
            assert spec["env"][forced[0]] == "TRUE", package

    def test_mamba_keeps_the_cuda_kernels(self):
        """MAMBA_KEEP_CUDA_BUILD is opt-in upstream, and selective_scan_cuda -- the extension
        whose missing symbols are the entire problem -- is absent from the wheel without it."""
        assert prebuilt_wheels.SPECS["mamba-ssm"]["env"]["MAMBA_KEEP_CUDA_BUILD"] == "TRUE"
        assert "selective_scan_cuda" in prebuilt_wheels.SPECS["mamba-ssm"]["import_names"]

    def test_flash_attn_archs_cover_what_we_claim(self):
        archs = prebuilt_wheels.SPECS["flash-attn"]["env"]["FLASH_ATTN_CUDA_ARCHS"].split(";")
        # 86 and 89 are covered by the 80 cubin, which is forward compatible across the minor
        # versions of its major. Those two must NOT be listed: each one is a full extra pass
        # over every kernel for nothing.
        assert archs == ["80", "90", "100", "120"]


class TestWheelNames:
    def test_the_local_version_segment_is_upstreams(self):
        """Byte for byte what Dao-AILab and state-spaces put in their own release assets, so a
        wheel of ours drops into any tooling that already pattern-matches theirs."""
        assert (
            prebuilt_wheels.wheel_name("flash-attn", "2.13.0", "3.13")
            == "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )
        assert (
            prebuilt_wheels.wheel_name("mamba-ssm", "2.14.0", "3.12")
            == "mamba_ssm-2.3.2.post1+cu13torch2.14cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
        )
        assert (
            prebuilt_wheels.wheel_name("causal-conv1d", "2.13.0", "3.11")
            == "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        )

    def test_round_trip(self):
        for package in prebuilt_wheels.SPECS:
            name = prebuilt_wheels.wheel_name(package, "2.14.0", "3.13")
            parsed = prebuilt_wheels.parse_wheel_name(name)
            assert parsed["package"] == package
            assert parsed["torch"] == "2.14"
            assert parsed["python"] == "cp313"
            assert parsed["abi"] == "TRUE"

    @pytest.mark.parametrize(
        "name",
        [
            "SHA256SUMS",
            "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl.sigstore.json",
            "flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
            "torch-2.13.0-cp313-cp313-linux_x86_64.whl",
        ],
    )
    def test_things_that_are_not_our_wheels_are_not_parsed_as_wheels(self, name):
        assert prebuilt_wheels.parse_wheel_name(name) is None


class TestReleaseNotes:
    def _notes(self):
        entries = [
            (f"{index:064x}", prebuilt_wheels.wheel_name(package, torch, "3.13"))
            for index, (package, torch) in enumerate(
                (package, torch)
                for torch in prebuilt_wheels.TORCH_VERSIONS
                for package in prebuilt_wheels.SPECS
            )
        ]
        entries.append(("deadbeef", "SHA256SUMS"))
        return prebuilt_wheels.render_notes(
            entries, tag = "prebuilt-wheels-cu13", repo = "unslothai/unsloth"
        )

    def test_every_wheel_is_listed_with_its_digest(self):
        notes = self._notes()
        for torch in prebuilt_wheels.TORCH_VERSIONS:
            for package in prebuilt_wheels.SPECS:
                assert f"`{prebuilt_wheels.wheel_name(package, torch, '3.13')}`" in notes
        assert "| 6 wheels" not in notes
        assert "6 wheels, each with a `.sigstore.json` bundle beside it." in notes

    def test_non_wheel_assets_are_not_rows(self):
        assert "| `SHA256SUMS` |" not in self._notes()

    def test_the_verification_commands_are_the_ones_the_workflow_signs_with(self):
        notes = self._notes()
        assert "python -m sigstore verify identity" in notes
        assert (
            "https://github.com/unslothai/unsloth/.github/workflows/prebuilt-cuda-wheels.yml"
            in notes
        )
        assert "--cert-oidc-issuer https://token.actions.githubusercontent.com" in notes
        assert "gh attestation verify <wheel> --repo unslothai/unsloth" in notes
        assert "sha256sum -c SHA256SUMS" in notes

    def test_it_says_why_these_exist_with_the_actual_symbols(self):
        notes = self._notes()
        assert "materialize_cow_storage" in notes
        assert "c10_cuda_check_implementation" in notes
        assert "Linux x86_64 only" in notes
        assert "Not covered: Windows, macOS, ROCm" in notes

    def test_an_empty_release_does_not_render_an_empty_table(self):
        notes = prebuilt_wheels.render_notes([], tag = "t", repo = "o/r")
        assert "No wheels are attached" in notes
        assert "| Wheel | Package |" not in notes


class TestMambaPatch:
    SOURCE = """
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "-std=c++17"],
}
other = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "-std=c++17"],
}
"""

    def _run(self, tmp_path, source):
        setup = tmp_path / "setup.py"
        setup.write_text(source, encoding = "utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "patch_mamba_cxx20.py"), str(setup)],
            capture_output = True,
            text = True,
        )
        return result, setup

    def test_it_patches_all_four(self, tmp_path):
        result, setup = self._run(tmp_path, self.SOURCE)
        assert result.returncode == 0, result.stderr
        patched = setup.read_text(encoding = "utf-8")
        assert patched.count('"-std=c++20"') == 4
        assert '"-std=c++17"' not in patched

    def test_it_is_idempotent(self, tmp_path):
        result, setup = self._run(tmp_path, self.SOURCE)
        assert result.returncode == 0
        again = subprocess.run(
            [sys.executable, str(SCRIPTS / "patch_mamba_cxx20.py"), str(setup)],
            capture_output = True,
            text = True,
        )
        assert again.returncode == 0, again.stderr
        assert "nothing to do" in again.stdout
        assert setup.read_text(encoding = "utf-8").count('"-std=c++20"') == 4

    def test_an_upstream_change_fails_loudly(self, tmp_path):
        """If upstream makes this change itself, or moves to c++23, the run must stop rather
        than build something nobody chose."""
        result, _ = self._run(tmp_path, self.SOURCE.replace('"-std=c++17"', '"-std=c++23"', 2))
        assert result.returncode == 1
        assert "expected 4 occurrences" in result.stderr

    def test_a_missing_file_fails(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "patch_mamba_cxx20.py"), str(tmp_path / "nope.py")],
            capture_output = True,
            text = True,
        )
        assert result.returncode == 1


# ── The workflow itself ───────────────────────────────────────────────────────


@pytest.fixture(scope = "module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


class TestWorkflow:
    def test_dispatch_only(self, workflow):
        """It signs binaries and writes a release under our identity. A pull_request or push
        trigger would put both behind whatever a fork can propose."""
        assert set(triggers(workflow)) == {"workflow_dispatch"}

    def test_publishing_is_opt_in(self, workflow):
        """A run that only builds, smoke tests and signs is the default, so the expensive and
        irreversible half has to be asked for."""
        inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
        assert inputs["publish"]["default"] is False
        assert inputs["gpu_smoke"]["default"] is False
        assert workflow["jobs"]["publish"]["if"] == "inputs.publish"
        assert workflow["jobs"]["gpu-smoke"]["if"] == "inputs.gpu_smoke"

    def test_only_publish_can_write_contents(self, workflow):
        assert workflow["permissions"] == {"contents": "read"}
        for name, job in workflow["jobs"].items():
            permissions = job.get("permissions", {})
            if name == "publish":
                assert permissions == {"contents": "write"}
            else:
                assert permissions.get("contents", "read") == "read", name

    def test_only_the_sign_job_gets_an_oidc_token(self, workflow):
        for name, job in workflow["jobs"].items():
            permissions = job.get("permissions", {})
            expected = "write" if name == "sign" else None
            assert permissions.get("id-token") == expected, name
            assert permissions.get("attestations") == expected, name

    def test_the_signing_job_names_the_gateable_environment(self, workflow):
        assert workflow["jobs"]["sign"]["environment"] == "release-signing"

    def test_the_publish_job_builds_nothing(self, workflow):
        """contents: write and a compiler in the same job is how a poisoned build reaches a
        release. The publish job only downloads an artifact and uploads it."""
        steps = workflow["jobs"]["publish"]["steps"]
        body = json.dumps(steps)
        assert "setup.py" not in body
        assert "pip install" not in body

    def test_it_never_becomes_the_latest_release(self, workflow):
        """The repository's latest release is what the installers and the README resolve to.
        A wheelhouse tag must not take it."""
        body = json.dumps(workflow["jobs"]["publish"]["steps"])
        assert body.count("--latest=false") == 3
        assert "gh release create" in body
        assert "--clobber" in body

    def test_every_action_is_pinned_to_a_sha(self, workflow):
        for name, job in workflow["jobs"].items():
            for step in job["steps"]:
                uses = step.get("uses")
                if uses is None:
                    continue
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), (name, uses)

    def test_it_signs_with_sigstore_and_attests_provenance(self, workflow):
        uses = [step.get("uses", "") for step in workflow["jobs"]["sign"]["steps"]]
        assert any(u.startswith("sigstore/gh-action-sigstore-python@") for u in uses)
        assert any(u.startswith("actions/attest-build-provenance@") for u in uses)

    def test_the_signature_is_verified_in_the_same_job(self, workflow):
        """A bundle nobody checked is a bundle nobody can rely on, and it must verify against
        the identity the release notes tell people to use."""
        step = next(
            step
            for step in workflow["jobs"]["sign"]["steps"]
            if step.get("uses", "").startswith("sigstore/")
        )
        assert step["with"]["verify"] is True
        identity = step["with"]["verify-cert-identity"]
        assert ".github/workflows/prebuilt-cuda-wheels.yml@" in identity
        assert step["with"]["verify-oidc-issuer"] == "https://token.actions.githubusercontent.com"

    def test_the_build_job_gates_on_the_import(self, workflow):
        """The import check is the point of the release. If it ever becomes informational this
        workflow publishes exactly the broken wheels it exists to replace."""
        smoke = next(
            step
            for step in workflow["jobs"]["build"]["steps"]
            if step.get("name", "").startswith("Smoke test")
        )
        assert "continue-on-error" not in smoke
        assert "importlib.import_module" in smoke["run"]

    def test_the_build_runs_on_the_older_ubuntu(self, workflow):
        """The wheels are tagged linux_x86_64, which pip installs without a glibc check, so the
        runner's glibc is the real compatibility floor."""
        assert workflow["jobs"]["build"]["runs-on"] == "ubuntu-22.04"

    def test_the_gpu_job_uses_a_label_that_already_exists(self, workflow):
        """Never invent a runner label: a job with one queues until the run is cancelled."""
        existing = set()
        for path in (REPO / ".github" / "workflows").glob("*.yml"):
            for job in (
                (yaml.safe_load(path.read_text(encoding = "utf-8")) or {}).get("jobs", {}).values()
            ):
                runs_on = job.get("runs-on")
                if isinstance(runs_on, list):
                    existing.add(tuple(runs_on))
        assert tuple(workflow["jobs"]["gpu-smoke"]["runs-on"]) in existing

    def test_the_build_cannot_outlive_githubs_job_limit(self, workflow):
        """A job that hits the 6 h cap is reported as cancelled with no logs. Both the step's
        own timeout and the job's have to sit below it."""
        assert workflow["jobs"]["build"]["timeout-minutes"] < 360
        for spec in prebuilt_wheels.SPECS.values():
            minutes = int(spec["build_timeout"].removesuffix("m"))
            assert minutes < workflow["jobs"]["build"]["timeout-minutes"]

    def test_the_sign_job_refuses_a_short_matrix(self, workflow):
        """A missing leg publishes a set with a hole in it, which downstream reads as "no wheel"
        and sends the user into a source build."""
        body = json.dumps(workflow["jobs"]["sign"]["steps"])
        assert "expected $EXPECTED wheels" in body

    def test_concurrency_queues_rather_than_cancels(self, workflow):
        assert workflow["concurrency"]["cancel-in-progress"] is False
