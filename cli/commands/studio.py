# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import typer

studio_app = typer.Typer(help = "Unsloth Studio commands.")

STUDIO_HOME = Path.home() / ".unsloth" / "studio"


def _get_repo_root() -> Optional[Path]:
    """Find the git clone repo root, or None if pure pip install."""
    # Check 1: __file__ is in the repo (editable install)
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "studio" / "setup.sh"
    ).is_file():
        return candidate
    # Check 2: CWD is the repo (non-editable wheel, running from repo dir)
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file() and (cwd / "studio" / "setup.sh").is_file():
        return cwd
    return None


def _is_git_clone() -> bool:
    return _get_repo_root() is not None


def _studio_venv_python() -> Optional[Path]:
    """Return the studio venv Python binary, or None if not set up."""
    if platform.system() == "Windows":
        p = STUDIO_HOME / ".venv" / "Scripts" / "python.exe"
    else:
        p = STUDIO_HOME / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def _find_run_py() -> Optional[Path]:
    """Find studio/backend/run.py."""
    # 1. Repo root (git clone / editable)
    repo = _get_repo_root()
    if repo:
        run_py = repo / "studio" / "backend" / "run.py"
        if run_py.is_file():
            return run_py
    # 2. Studio venv's site-packages
    for match in (STUDIO_HOME / ".venv").glob(
        "lib/python*/site-packages/studio/backend/run.py"
    ):
        return match
    # 3. Current package's site-packages
    run_py = (
        Path(__file__).resolve().parent.parent.parent / "studio" / "backend" / "run.py"
    )
    return run_py if run_py.is_file() else None


def _find_install_script() -> Optional[Path]:
    """Find studio/install_python_stack.py."""
    # 1. Repo root
    repo = _get_repo_root()
    if repo:
        s = repo / "studio" / "install_python_stack.py"
        if s.is_file():
            return s
    # 2. Relative to __file__ (in site-packages)
    s = (
        Path(__file__).resolve().parent.parent.parent
        / "studio"
        / "install_python_stack.py"
    )
    return s if s.is_file() else None


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
):
    """Launch the Unsloth Studio server."""
    if ctx.invoked_subcommand is not None:
        return

    # Always use the studio venv if it exists and we're not already in it
    studio_venv_dir = STUDIO_HOME / ".venv"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))

    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if studio_python and run_py:
            if not silent:
                typer.echo("Launching with studio venv...")
            args = [
                str(studio_python),
                str(run_py),
                "--host",
                host,
                "--port",
                str(port),
            ]
            if frontend:
                args.extend(["--frontend", str(frontend)])
            if silent:
                args.append("--silent")
            os.execvp(str(studio_python), args)
        else:
            typer.echo("Studio not set up. Run 'unsloth studio setup' first.")
            raise typer.Exit(1)

    from studio.backend.run import run_server

    if not silent:
        from studio.backend.run import _resolve_external_ip

        display_host = _resolve_external_ip() if host == "0.0.0.0" else host
        typer.echo(f"Starting Unsloth Studio on http://{display_host}:{port}")

    run_server(
        host = host,
        port = port,
        frontend_path = frontend,
        silent = silent,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\nShutting down...")


# ── unsloth studio setup ─────────────────────────────────────────────


@studio_app.command()
def setup():
    """Run one-time Studio environment setup."""
    if _is_git_clone():
        _dev_setup()
    else:
        _pip_setup()


def _dev_setup():
    """Git-clone: run setup.sh / setup.ps1."""
    repo_root = _get_repo_root()
    studio_dir = repo_root / "studio"
    if platform.system() == "Windows":
        script = studio_dir / "setup.ps1"
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check = True,
        )
    else:
        script = studio_dir / "setup.sh"
        subprocess.run(["bash", str(script)], check = True)


def _pip_setup():
    """Pip-install: create studio venv, install all deps, build extras."""
    import venv as _venv

    venv_dir = STUDIO_HOME / ".venv"
    venv_t5_dir = STUDIO_HOME / ".venv_t5"

    if platform.system() == "Windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"

    typer.echo("Setting up Unsloth Studio...")

    # 1. Create venv
    if not venv_python.is_file():
        typer.echo(f"  Creating venv at {venv_dir}...")
        STUDIO_HOME.mkdir(parents = True, exist_ok = True)
        _venv.create(str(venv_dir), with_pip = True)

    # 2. Install all Python deps via install_python_stack.py
    install_script = _find_install_script()
    if install_script:
        typer.echo("  Installing Python dependencies...")
        subprocess.run([str(venv_python), str(install_script)], check = True)
    else:
        typer.echo("Error: Could not find install_python_stack.py")
        raise typer.Exit(1)

    # 3. Pre-install transformers 5.x overlay
    if venv_t5_dir.is_dir() and any(venv_t5_dir.iterdir()):
        typer.echo(f"  Transformers 5.x overlay already at {venv_t5_dir}")
    else:
        typer.echo("  Installing transformers 5.x overlay...")
        venv_t5_dir.mkdir(parents = True, exist_ok = True)
        subprocess.run(
            [
                str(venv_pip),
                "install",
                "--target",
                str(venv_t5_dir),
                "--no-deps",
                "transformers==5.2.0",
            ],
            check = True,
        )
        subprocess.run(
            [
                str(venv_pip),
                "install",
                "--target",
                str(venv_t5_dir),
                "--no-deps",
                "huggingface_hub==1.3.0",
            ],
            check = True,
        )
        typer.echo(f"  Installed to {venv_t5_dir}")

    # 4. Build llama.cpp
    _build_llama_cpp()

    typer.echo("")
    typer.echo("Setup complete! Run 'unsloth studio' to start.")


def _build_llama_cpp():
    """Build llama.cpp at ~/.unsloth/llama.cpp/."""
    import shutil

    unsloth_home = Path.home() / ".unsloth"
    llama_dir = unsloth_home / "llama.cpp"

    if not shutil.which("cmake"):
        typer.echo("  cmake not found — skipping llama.cpp build")
        return
    if not shutil.which("git"):
        typer.echo("  git not found — skipping llama.cpp build")
        return

    typer.echo("  Building llama.cpp for GGUF inference...")

    if llama_dir.exists():
        shutil.rmtree(llama_dir)
    unsloth_home.mkdir(parents = True, exist_ok = True)

    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_dir),
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    if result.returncode != 0:
        typer.echo("  Failed to clone llama.cpp")
        return

    cmake_args = []
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path and Path("/usr/local/cuda/bin/nvcc").is_file():
        nvcc_path = "/usr/local/cuda/bin/nvcc"
    if nvcc_path:
        typer.echo(f"  Building with CUDA (nvcc: {nvcc_path})...")
        cmake_args.append("-DGGML_CUDA=ON")
    else:
        typer.echo("  Building CPU-only...")

    build_dir = llama_dir / "build"
    result = subprocess.run(
        ["cmake", "-S", str(llama_dir), "-B", str(build_dir)] + cmake_args,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    if result.returncode != 0:
        typer.echo("  cmake configure failed")
        return

    ncpu = str(os.cpu_count() or 4)
    result = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            f"-j{ncpu}",
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    if result.returncode != 0:
        typer.echo("  llama-server build failed")
        return

    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-quantize",
            f"-j{ncpu}",
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )

    server_bin = build_dir / "bin" / "llama-server"
    if server_bin.is_file():
        typer.echo(f"  llama-server built at {server_bin}")
    else:
        typer.echo("  llama-server binary not found after build")
