# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""docker/Dockerfile.studio-rocm ships JupyterLab, the notebooks and key-only sshd
beside Unsloth Studio, as unsloth/unsloth:studio does on CUDA. Dockerfile.studio
inherits JupyterLab and the notebook tooling from the CUDA core image; the ROCm base
carries none of that, so the ROCm file installs it itself, and the two can drift
apart without any build noticing. These pin each piece to the CUDA file it mirrors,
and the entrypoint hooks the services depend on. Static reads: no AMD GPU, no
Docker, no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
CUDA_BASE = DOCKER / "Dockerfile"
CUDA_STUDIO = DOCKER / "Dockerfile.studio"
ROCM_BASE = DOCKER / "Dockerfile.rocm"
ROCM_STUDIO = DOCKER / "Dockerfile.studio-rocm"
ENTRYPOINT = DOCKER / "entrypoint-rocm.sh"
CUDA_ENTRYPOINT = DOCKER / "entrypoint.sh"
SUPERVISORD = DOCKER / "supervisord.conf"
LAUNCH = DOCKER / "studio_launch.sh"
DOCKERIGNORE = DOCKER / ".dockerignore"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish-rocm.yml"
LABEXT_PKG = DOCKER / "jupyter" / "unsloth_labext" / "package.json"
BRANDING = DOCKER / "jupyter" / "unsloth_branding.py"

VENV = "/opt/unsloth-venv"


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def _logical_lines(text: str) -> list[str]:
    """Dockerfile instructions with their backslash continuations joined."""
    return [ln.strip() for ln in re.sub(r"\\\r?\n", " ", text).splitlines()]


def _instructions(path: Path, name: str) -> list[str]:
    return [
        ln[len(name) :].strip() for ln in _logical_lines(_read(path)) if ln.startswith(name + " ")
    ]


def _copies(path: Path) -> list[tuple[str, list[str], str]]:
    """(--from stage or '', sources, destination) for every COPY."""
    out = []
    for args in _instructions(path, "COPY"):
        words = args.split()
        stage = ""
        if words and words[0].startswith("--from="):
            stage = words.pop(0)[len("--from=") :]
        out.append((stage, words[:-1], words[-1]))
    return out


def _env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for args in _instructions(path, "ENV"):
        for m in re.finditer(r"(\w+)=(\S+)", args):
            env[m.group(1)] = m.group(2)
    return env


def _pins(text: str, packages) -> dict[str, set[str]]:
    return {pkg: set(re.findall(rf'"{re.escape(pkg)}==([0-9][^"]*)"', text)) for pkg in packages}


def _branding_module():
    spec = importlib.util.spec_from_file_location("unsloth_branding_under_test", BRANDING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── JupyterLab itself ────────────────────────────────────────────────────────

JUPYTER_PINS = ("jupyterlab", "notebook", "ipywidgets")


def test_jupyterlab_is_pinned_to_the_cuda_core_image():
    cuda = _pins(_read(CUDA_BASE), JUPYTER_PINS)
    rocm = _pins(_read(ROCM_STUDIO), JUPYTER_PINS)
    for pkg in JUPYTER_PINS:
        assert len(cuda[pkg]) == 1, f"{pkg} is pinned {cuda[pkg] or 'nowhere'} in docker/Dockerfile"
        assert rocm[pkg] == cuda[pkg], (
            f"{pkg}=={rocm[pkg] or '(unpinned)'} in Dockerfile.studio-rocm but "
            f"{cuda[pkg]} in docker/Dockerfile: the two images would ship different notebook stacks"
        )
    # the labext-builder stage builds the extension against the jupyterlab it will run under
    (jl,) = cuda["jupyterlab"]
    assert (
        _read(ROCM_STUDIO).count(f'"jupyterlab=={jl}"') == 2
    ), "the labext-builder stage and the final stage must install the same jupyterlab"


def test_jupyterlab_goes_into_the_base_venv_and_leaves_torch_alone():
    """The notebook kernel has to be the venv with the ROCm torch, and a resolve
    against pypi alone must not be allowed to replace that torch with a CUDA one."""
    (install,) = [r for r in _instructions(ROCM_STUDIO, "RUN") if '"notebook==' in r]
    assert f"{VENV}/bin/uv pip install --python {VENV}/bin/python" in install
    assert (
        "BASE_TORCH=" in install and "version('torch')" in install
    ), "the install must assert the base venv's torch is the same before and after"


# ── the labextension, theme and branding chain ───────────────────────────────


def test_the_labextension_lands_where_the_branding_guard_looks():
    branding = _branding_module()
    # the guard joins with os.path, and it only ever runs inside the (Linux) image
    paths = {
        key: value.replace("\\", "/") if isinstance(value, str) else value
        for key, value in branding.resolve_paths(
            venv_share = f"{VENV}/share/jupyter", jupyter_server_dir = "/unused", config_dirs = []
        ).items()
    }
    output_dir = json.loads(_read(LABEXT_PKG))["jupyterlab"]["outputDir"]

    copies = _copies(ROCM_STUDIO)
    (labext_src, labext_dest) = next(
        (src[0], dest) for stage, src, dest in copies if stage == "labext-builder"
    )
    (staged_src,) = [dest for stage, src, dest in copies if src == ["jupyter/unsloth_labext"]]
    assert (
        labext_src == f"{staged_src}/{output_dir}"
    ), "the --from copy must take the labextension from where jlpm build:prod writes it"
    assert labext_dest == paths["labext_dir"]
    assert paths["overrides"] in [
        dest for _, src, dest in copies if src == ["jupyter/overrides.json"]
    ]
    text = _read(ROCM_STUDIO)
    assert paths["license"] in text, "the AGPLv3 text must be staged where the guard reads it"
    assert "-m unsloth_branding --verify" in text, "the build must run the branding guard"


def test_the_branding_chain_matches_the_cuda_studio_image():
    """Same assets, same destinations, same disable/lock pairs: the ROCm image is
    the same product with a different torch, and the guard checks the same paths."""

    def branding(path: Path):
        copies = {
            (tuple(src), dest)
            for stage, src, dest in _copies(path)
            if not stage and all(s.startswith("jupyter/") for s in src)
        }
        locks = sorted(re.findall(r"jupyter labextension (?:disable|lock) \S+", _read(path)))
        return copies, locks

    assert branding(ROCM_STUDIO) == branding(CUDA_STUDIO)


# ── the notebooks and their tooling ──────────────────────────────────────────


def test_the_notebook_tooling_matches_the_cuda_core_image():
    def helpers(path: Path) -> set[str]:
        (src,) = [src for _, src, dest in _copies(path) if dest == "/opt/unsloth-nb/"]
        return set(src)

    assert helpers(ROCM_STUDIO) == helpers(
        CUDA_BASE
    ), "a notebook helper added to one image and not the other"
    cuda_env, rocm_env = _env(CUDA_BASE), _env(ROCM_STUDIO)
    assert rocm_env["IPYTHONDIR"] == cuda_env["IPYTHONDIR"]
    for env in (cuda_env, rocm_env):
        assert env["PATH"].startswith(
            "/opt/unsloth-nb/bin:"
        ), "the pip/uv shim has to sit ahead of the venv on PATH or install cells clobber torch"


def test_the_notebooks_are_baked_where_the_sync_script_looks():
    text = _read(ROCM_STUDIO)
    assert "https://github.com/unslothai/notebooks" in text
    assert "/opt/unsloth-notebooks/.unsloth_template_commit" in text
    assert "/opt/unsloth-notebooks" in _read(DOCKER / "unsloth_sync_notebooks.sh")
    assert "ARG UNSLOTH_NOTEBOOKS_REF" in text, "CI has to be able to pin the notebooks commit"
    # the AMD-* set is the point of the image, so an upstream ref without one fails the build
    assert "grep -c '^AMD-'" in text


# ── the three services ───────────────────────────────────────────────────────


def test_every_supervisord_program_is_installed_by_the_dockerfile():
    conf = _read(SUPERVISORD)
    commands = re.findall(r"^command=(\S+)", conf, re.M)
    assert commands, "supervisord.conf lost its programs"
    dests = {dest for _, _, dest in _copies(ROCM_STUDIO)}
    chmod = " ".join(
        r for r in _instructions(ROCM_STUDIO, "RUN") if r.startswith("chmod +x /usr/local/bin/")
    )
    apt = " ".join(r for r in _instructions(ROCM_STUDIO, "RUN") if "apt-get install" in r)
    for command in commands:
        if command.startswith("/usr/local/bin/"):
            assert (
                command in dests
            ), f"supervisord runs {command}, which the Dockerfile never copies"
            assert command in chmod, f"{command} is copied but not made executable"
        elif command == "/usr/sbin/sshd":
            assert "openssh-server" in apt
        elif command == "jupyter":
            pass  # the venv's, pinned above
        else:
            raise AssertionError(f"unexpected supervisord command {command}")
    assert "supervisor" in apt.split()
    (conf_dest,) = [dest for _, src, dest in _copies(ROCM_STUDIO) if src == ["supervisord.conf"]]
    assert f"exec supervisord -c {conf_dest}" in _read(LAUNCH)


def test_the_launcher_is_the_command_and_the_ports_are_exposed():
    (cmd,) = _instructions(ROCM_STUDIO, "CMD")
    assert json.loads(cmd) == ["/usr/local/bin/unsloth-studio-launch"]
    assert "unsloth-studio-home" not in cmd, "the home link moved into the entrypoint"
    env = _env(ROCM_STUDIO)
    (expose,) = _instructions(ROCM_STUDIO, "EXPOSE")
    assert set(expose.split()) == {env["UNSLOTH_STUDIO_PORT"], env["JUPYTER_PORT"], "22"}
    # supervisord.conf expands these before the launcher has exported anything
    for name in ("JUPYTER_PORT", "UNSLOTH_ENABLE_SSHD", "UNSLOTH_STUDIO_STOP_WAIT_S"):
        assert name in env, f"supervisord's %(ENV_{name})s needs an image default"


def test_ssh_login_shells_keep_the_rocm_variables():
    """studio_launch.sh writes the container's env into /etc/profile.d for SSH
    sessions, filtered by prefix. The image's ROCBLAS_USE_HIPBLASLT and a user's
    HSA_OVERRIDE_GFX_VERSION have to make it through, or an SSH shell trains on a
    different ROCm configuration than the Studio and Jupyter processes."""
    match = re.search(r'keep\s*=\s*re\.compile\(r"(.*?)"\)', _read(LAUNCH))
    assert match, "the profile.d keep pattern moved"
    keep = re.compile(match.group(1))
    for var in (
        "ROCBLAS_USE_HIPBLASLT",
        "HSA_OVERRIDE_GFX_VERSION",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ROCM_HOME",
        "CUDA_VISIBLE_DEVICES",
        "PATH",
    ):
        assert keep.search(var), f"{var} would not reach an SSH login shell"
    assert not keep.search("HOME") and not keep.search("LANG")


# ── the entrypoint hooks the services depend on ──────────────────────────────


def test_the_entrypoint_links_the_studio_home_before_anything_reads_it():
    """supervisord starts Studio from $UNSLOTH_STUDIO_HOME/bin/unsloth, a link into
    the app dir that unsloth-studio-home creates; a volume mounted on the home hides
    the build-time link, so the entrypoint has to run the linker on every start,
    before the GPU checks that may exit. Mirrors entrypoint.sh on the CUDA image."""
    body = _read(ENTRYPOINT)
    linker = body.index("/usr/local/bin/unsloth-studio-home")
    assert linker < body.index("Check 1"), "the home link has to precede the GPU checks"
    assert "/usr/local/bin/unsloth-studio-home" in _read(CUDA_ENTRYPOINT)
    # the studio image reinstalls the entrypoint, since the published base predates the hooks
    (base_dest,) = [dest for _, src, dest in _copies(ROCM_BASE) if src == ["entrypoint-rocm.sh"]]
    assert (base_dest,) == tuple(
        dest for _, src, dest in _copies(ROCM_STUDIO) if src == ["entrypoint-rocm.sh"]
    )
    assert not _instructions(ROCM_STUDIO, "ENTRYPOINT"), "the base ENTRYPOINT is inherited"


def test_the_entrypoint_syncs_the_notebooks_before_every_exec():
    body = _read(ENTRYPOINT)
    assert "/usr/local/bin/unsloth-sync-notebooks" in body
    assert "/usr/local/bin/unsloth-sync-notebooks" in _read(CUDA_ENTRYPOINT)
    execs = [m.start() for m in re.finditer(r'^\s*exec "\$@"\s*$', body, re.M)]
    assert len(execs) >= 2, "the skip path and the checked path both exec the command"
    for pos in execs:
        preceding = body[:pos].rstrip().splitlines()[-1].strip()
        assert (
            preceding == "sync_notebooks"
        ), f"exec at offset {pos} is not preceded by sync_notebooks but by {preceding!r}"


# ── the publisher ────────────────────────────────────────────────────────────


def test_the_publisher_passes_every_build_arg_the_final_stage_declares():
    """Each ARG after the final FROM is a ref a RUN layer is keyed on. One the
    publisher leaves at its default bakes a mutable 'main' that docker matches on
    the next run, so the published image would carry the first build's bits."""
    import yaml

    text = _read(ROCM_STUDIO)
    final = text[text.rindex("\nFROM ") :]
    declared = {m.group(1) for m in re.finditer(r"^ARG (\w+)=", final, re.M)}
    assert declared, "no ARG after the final FROM"

    wf = yaml.safe_load(_read(WORKFLOW))
    step = next(s for s in wf["jobs"]["build-studio"]["steps"] if s.get("id") == "build")
    assert step["with"]["file"] == "./docker/Dockerfile.studio-rocm"
    passed = dict(ln.split("=", 1) for ln in step["with"]["build-args"].splitlines() if ln)
    assert declared <= set(passed), declared - set(passed)
    # the base by digest, so a newer run's :latest cannot slip under this build
    assert "@${{ needs.build.outputs.digest }}" in passed["BASE_IMAGE"]
    for name in declared:
        assert passed[name].startswith("${{ needs.prepare.outputs."), (name, passed[name])


# ── the build context ────────────────────────────────────────────────────────


def test_every_copy_source_is_allowed_by_the_dockerignore():
    """docker/.dockerignore denies everything and allow-lists by name, so a file
    COPY'd here but not listed there fails the build with 'not found'."""
    allowed = [ln[1:].strip() for ln in _read(DOCKERIGNORE).splitlines() if ln.startswith("!")]

    def is_allowed(source: str) -> bool:
        for pattern in allowed:
            if pattern == source:
                return True
            if pattern.endswith("/**") and source.startswith(pattern[:-3] + "/"):
                return True
        return False

    for stage, sources, _ in _copies(ROCM_STUDIO):
        if stage:
            continue
        for source in sources:
            assert is_allowed(source), f"{source} is not allow-listed in docker/.dockerignore"
