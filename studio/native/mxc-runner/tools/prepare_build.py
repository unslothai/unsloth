#!/usr/bin/env python3
"""Prepare and build the pinned, extended MXC runner without ambient source state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


MXC_REPOSITORY = "https://github.com/microsoft/mxc.git"
MXC_REVISION = "ca7ea12ac6bd9f5420d6adecb37e32a8158da476"
MXC_PATCH_SHA256 = "4741ee9db1f389e6c43f8badcbf2c53076e3d7e5e546d6abd474459c30fff77c"
MXC_PATCHED_TREE = "2c5f5245a4676173f5e4c9e03576eb68d02c47d2"
RUNNER_LOCK_SHA256 = "b1bd3b7d83352de7f8ceb5fc85cbca2ed0a273fd4680f12b7846b3d3391ff25f"
PATCHED_LOCK_SHA256 = "1313685ae6b926cde46a96d1c481e326e059b3b649f96205bb78674075a8c594"
RUNNER_PROTOCOL_VERSION = 1
RUNTIME_MANIFEST_VERSION = 1
MXC_SCHEMA_VERSION = "0.8.0-alpha"
PROFILE_ID = "unsloth-mxc-windows-basecontainer-v1"
TARGET = "x86_64-pc-windows-msvc"
FEATURES = "mxc-no-dacl-api"


class PreparationError(RuntimeError):
    """A pinned source or build input failed integrity validation."""


def _run(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    command = [os.fspath(value) for value in argv]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise PreparationError(f"could not run {command[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stdout.strip()
        raise PreparationError(
            f"command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise PreparationError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PreparationError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def runner_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_build_inputs(root: Path) -> None:
    _require_digest(
        root / "upstream" / "mxc-ca7ea12-no-dacl-tier.patch",
        MXC_PATCH_SHA256,
        "approved MXC patch",
    )
    _require_digest(root / "Cargo.lock", RUNNER_LOCK_SHA256, "ordinary Cargo.lock")
    _require_digest(
        root / "upstream" / "Cargo.patched.lock",
        PATCHED_LOCK_SHA256,
        "patched Cargo.lock",
    )


def verify_upstream_checkout(source: Path, expected_revision: str = MXC_REVISION) -> None:
    if not (source / ".git").exists():
        raise PreparationError(f"MXC source is not a Git checkout: {source}")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=source)
    if revision != expected_revision:
        raise PreparationError(
            f"MXC revision mismatch: expected {expected_revision}, got {revision}"
        )
    dirty = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=source)
    if dirty:
        raise PreparationError("the supplied MXC checkout is dirty; refusing ambient source state")


def apply_approved_patch(source: Path, patch: Path, expected_tree: str) -> None:
    """Apply one patch atomically and prove the entire resulting Git tree."""
    _run(["git", "apply", "--check", "--whitespace=error-all", patch], cwd=source)
    _run(["git", "apply", "--whitespace=error-all", patch], cwd=source)
    _run(["git", "diff", "--check", "HEAD"], cwd=source)
    _run(["git", "add", "--all"], cwd=source)
    tree = _run(["git", "write-tree"], cwd=source)
    if tree != expected_tree:
        raise PreparationError(f"patched MXC tree mismatch: expected {expected_tree}, got {tree}")


def prepare_source(*, source: str, destination: Path, root: Path | None = None) -> Path:
    """Clone one clean pinned tree, apply the approved patch, and publish atomically."""
    root = root or runner_root()
    verify_build_inputs(root)
    patch = root / "upstream" / "mxc-ca7ea12-no-dacl-tier.patch"
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PreparationError(f"prepared source destination already exists: {destination}")

    local_source = Path(source).resolve() if Path(source).exists() else None
    if local_source is not None:
        verify_upstream_checkout(local_source)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=os.fspath(destination.parent))
    )
    shutil.rmtree(stage)
    try:
        clone_source = os.fspath(local_source) if local_source is not None else source
        _run(["git", "clone", "--quiet", "--no-checkout", "--no-hardlinks", clone_source, stage])
        _run(["git", "checkout", "--quiet", "--detach", MXC_REVISION], cwd=stage)
        verify_upstream_checkout(stage)
        apply_approved_patch(stage, patch, MXC_PATCHED_TREE)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def build_runner(
    *,
    source: str,
    output: Path,
    root: Path | None = None,
    cargo: str = "cargo",
) -> Path:
    """Build with a path override in an isolated staging crate and emit a strict manifest."""
    root = (root or runner_root()).resolve()
    verify_build_inputs(root)
    output = output.resolve()
    if output.exists():
        raise PreparationError(f"artifact output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="unsloth-mxc-build-"))
    prepared = work / "mxc"
    staged_runner = work / "runner"
    artifact_stage = work / "artifact"
    try:
        prepare_source(source=source, destination=prepared, root=root)
        shutil.copytree(root / "src", staged_runner / "src")
        shutil.copy2(root / "Cargo.toml", staged_runner / "Cargo.toml")
        shutil.copy2(root / "upstream" / "Cargo.patched.lock", staged_runner / "Cargo.lock")
        manifest_path = staged_runner / "Cargo.toml"
        with manifest_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                '\n[patch."https://github.com/microsoft/mxc.git"]\n'
                f'mxc-sdk = {{ path = "{(prepared / "src/core/mxc-sdk").as_posix()}" }}\n'
            )

        environment = dict(os.environ)
        runner_prefix = staged_runner.as_posix()
        source_prefix = prepared.as_posix()
        existing = environment.get("RUSTFLAGS", "").strip()
        remap = (
            f"--remap-path-prefix={runner_prefix}=/unsloth-mxc-runner "
            f"--remap-path-prefix={source_prefix}=/mxc"
        )
        environment["RUSTFLAGS"] = f"{existing} {remap}".strip()
        environment.setdefault("SOURCE_DATE_EPOCH", "0")
        runner_source = hashlib.sha256()
        for path in sorted(
            (root / "src").rglob("*"), key=lambda value: value.relative_to(root).as_posix()
        ):
            if path.is_file():
                runner_source.update(path.relative_to(root).as_posix().encode("utf-8"))
                runner_source.update(b"\0")
                runner_source.update(path.read_bytes())
                runner_source.update(b"\0")
        runner_source.update((root / "Cargo.toml").read_bytes())
        runner_source_identity = runner_source.hexdigest()
        environment["UNSLOTH_MXC_PATCH_SHA256"] = MXC_PATCH_SHA256
        environment["UNSLOTH_MXC_PATCHED_TREE"] = MXC_PATCHED_TREE
        environment["UNSLOTH_MXC_RUNNER_SOURCE"] = runner_source_identity
        _run(
            [
                cargo,
                "build",
                "--release",
                "--locked",
                "--features",
                FEATURES,
                "--target",
                TARGET,
            ],
            cwd=staged_runner,
            env=environment,
        )
        _require_digest(staged_runner / "Cargo.lock", PATCHED_LOCK_SHA256, "built Cargo.lock")
        runner = staged_runner / "target" / TARGET / "release" / "unsloth-mxc-runner.exe"
        if not runner.is_file():
            raise PreparationError(f"Cargo did not produce the runner: {runner}")

        artifact_stage.mkdir()
        published_runner = artifact_stage / "unsloth-mxc-runner.exe"
        shutil.copy2(runner, published_runner)
        runner_digest = sha256_file(published_runner)
        rustc = _run(["rustc", "-Vv"])
        generation = f"mxc-{MXC_REVISION[:12]}-{runner_digest[:16]}"
        manifest = {
            "manifestVersion": RUNTIME_MANIFEST_VERSION,
            "runtimeVersion": "unsloth-mxc-preview-1",
            "generation": generation,
            "architecture": "x86_64",
            "target": TARGET,
            "protocolVersion": RUNNER_PROTOCOL_VERSION,
            "profileId": PROFILE_ID,
            "schemaVersion": MXC_SCHEMA_VERSION,
            "mxcRepository": MXC_REPOSITORY,
            "mxcRevision": MXC_REVISION,
            "mxcPatchSha256": MXC_PATCH_SHA256,
            "mxcPatchedTree": MXC_PATCHED_TREE,
            "runnerSourceIdentity": runner_source_identity,
            "cargoLockSha256": PATCHED_LOCK_SHA256,
            "features": [FEATURES],
            "rustc": rustc,
            "artifacts": {
                "runner": {
                    "path": "unsloth-mxc-runner.exe",
                    "sha256": runner_digest,
                    "size": published_runner.stat().st_size,
                }
            },
        }
        (artifact_stage / "runtime-manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(artifact_stage, output)
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-inputs")
    verify_parser.add_argument("--root", type=Path, default=runner_root())
    prepare_parser = subparsers.add_parser("prepare-source")
    prepare_parser.add_argument("--source", default=MXC_REPOSITORY)
    prepare_parser.add_argument("--destination", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, default=runner_root())
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", default=MXC_REPOSITORY)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--root", type=Path, default=runner_root())
    args = parser.parse_args()
    try:
        if args.command == "verify-inputs":
            verify_build_inputs(args.root.resolve())
        elif args.command == "prepare-source":
            prepare_source(
                source=args.source, destination=args.destination, root=args.root.resolve()
            )
        else:
            build_runner(source=args.source, output=args.output, root=args.root.resolve())
    except PreparationError as exc:
        parser.exit(2, f"MXC preparation refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
