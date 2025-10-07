from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


def _which_uv() -> str:
    from shutil import which
    uv = which("uv")
    if not uv:
        pytest.skip("uv not found on PATH. Install uv first: https://docs.astral.sh/uv/")
    return uv


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _rm_tree_force(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, p, exc):
        # Windows: clear read-only bits then retry
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_onerror)


@pytest.mark.slow
def test_uv_fresh_env_many_imports(
    per_test_dir: Path,
    project_root: Path,
    run_many,
    assert_all_ok,
    workers: int,
):
    """
    Creates a brand-new venv with uv, installs Unsloth into it with NO cache,
    fans out many subprocesses that 'import unsloth', then deletes the env.
    """
    uv = _which_uv()
    vdir = per_test_dir / "uv-venv"

    _rm_tree_force(vdir)

    env = os.environ.copy()
    env["UV_NO_CACHE"] = "1"
    env["UV_NO_CONFIG"] = "1"

    try:
        create_cmd = [uv, "--no-cache", "--no-config", "venv", str(vdir)]
        proc = subprocess.run(create_cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            pytest.fail(
                "uv venv failed\n\n"
                f"$ {' '.join(create_cmd)}\n\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        py = _venv_python(vdir)

        install_cmd = [uv, "--no-cache", "--no-config", "pip", "install", "--python", str(py)]
        uninstall_cmd = [uv, "pip", "uninstall", "--python", str(py)]
        install_dev_cmd = [uv, "--no-cache", "--no-config", "pip", "install", "--python", str(py)]
        editable = os.environ.get("UNSLOTH_EDITABLE") == "1"
        spec = os.environ.get("UNSLOTH_PIP_SPEC") or ["unsloth", "xformers<=0.0.28.post3", "vllm<=0.9.1", "transformers<=4.49.0"]
        uninstall_spec = ["unsloth", "unsloth_zoo"]
        dev_install_spec = ["https://github.com/mmathew23/unsloth.git@locks", "https://github.com/mmathew23/unsloth_zoo.git@locks"]

        index_url = os.environ.get("UNSLOTH_INDEX_URL")
        extra_index_url = os.environ.get("UNSLOTH_EXTRA_INDEX_URL")
        if index_url:
            install_cmd += ["--index-url", index_url]
        if extra_index_url:
            install_cmd += ["--extra-index-url", extra_index_url]

        pip_extra = os.environ.get("UNSLOTH_PIP_EXTRA", "")
        if pip_extra:
            install_cmd += pip_extra.split()

        if editable:
            install_cmd += ["-e", str(project_root)]
        else:
            install_cmd += spec

        uninstall_cmd += uninstall_spec
        install_dev_cmd += dev_install_spec
        for icmd in [install_cmd, uninstall_cmd, install_dev_cmd]:
            proc = subprocess.run(icmd, capture_output=True, text=True, env=env, cwd=project_root)
            if proc.returncode != 0:
                pytest.fail(
                    "uv pip install failed\n\n"
                    f"$ {' '.join(install_cmd)}\n\n"
                    f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                )

        payloads = [{"args": [], "kwargs": {}} for _ in range(workers)]
        results = run_many(
            "tests.pytest.filelocks.workers:import_unsloth",
            payloads,
            python=str(py),
            cwd=project_root,
            timeout=120.0,
        )
        assert_all_ok(results)

    finally:
        # 4) Always delete the environment so nothing persists on disk
        _rm_tree_force(vdir)
