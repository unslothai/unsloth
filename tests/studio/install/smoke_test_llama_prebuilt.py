#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"


def load_installer_module():
    spec = importlib.util.spec_from_file_location(
        "studio_install_llama_prebuilt", INSTALLER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load installer module from {INSTALLER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


installer = load_installer_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Run a real end-to-end prebuilt llama.cpp install into an isolated temporary "
            "directory on the current machine."
        )
    )
    parser.add_argument(
        "--llama-tag",
        default = "latest",
        help = "llama.cpp tag to resolve. Defaults to the latest usable published Unsloth release.",
    )
    parser.add_argument(
        "--published-repo",
        default = installer.DEFAULT_PUBLISHED_REPO,
        help = "Published bundle repository used for Linux CUDA selection.",
    )
    parser.add_argument(
        "--published-release-tag",
        default = installer.DEFAULT_PUBLISHED_TAG or "",
        help = "Optional published GitHub release tag to pin.",
    )
    parser.add_argument(
        "--work-dir",
        default = "",
        help = (
            "Optional directory under which the smoke install temp dir will be created. "
            "If omitted, defaults to ./.tmp/llama-prebuilt-smoke under the current directory."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action = "store_true",
        help = "Keep the temporary smoke install directory after success.",
    )
    return parser.parse_args()


def smoke_root_base(work_dir: str) -> Path:
    if work_dir:
        return Path(work_dir).expanduser().resolve()
    return (Path.cwd() / ".tmp" / "llama-prebuilt-smoke").resolve()


def make_smoke_root(base_dir: Path) -> Path:
    base_dir.mkdir(parents = True, exist_ok = True)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    return Path(tempfile.mkdtemp(prefix = f"run-{timestamp}-", dir = base_dir))


def main() -> int:
    args = parse_args()
    host = installer.detect_host()
    smoke_base = smoke_root_base(args.work_dir)
    smoke_root = make_smoke_root(smoke_base)
    install_dir = smoke_root / "install" / "llama.cpp"
    choice = None

    print(f"[smoke] host={host.system} machine={host.machine}")
    print(f"[smoke] temp_root={smoke_root}")

    try:
        requested_tag, resolved_tag, attempts, _approved_checksums = (
            installer.resolve_install_attempts(
                args.llama_tag,
                host,
                args.published_repo,
                args.published_release_tag,
            )
        )
        choice = attempts[0]
        print(f"[smoke] requested_tag={requested_tag}")
        print(f"[smoke] resolved_tag={resolved_tag}")
        print(f"[smoke] selected_asset={choice.name}")
        print(f"[smoke] selected_source={choice.source_label}")
        print(f"[smoke] install_dir={install_dir}")
        installer.install_prebuilt(
            install_dir = install_dir,
            llama_tag = args.llama_tag,
            published_repo = args.published_repo,
            published_release_tag = args.published_release_tag,
        )
        print(f"[smoke] PASS install_dir={install_dir}")
        print(
            "[smoke] note=This was a real prebuilt install into an isolated temp directory."
        )
        return installer.EXIT_SUCCESS
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else installer.EXIT_ERROR
        if code == installer.EXIT_FALLBACK:
            print(f"[smoke] FALLBACK install_dir={install_dir}")
            print(
                "[smoke] note=Prebuilt path failed and would fall back to source build in setup."
            )
            print(installer.collect_system_report(host, choice, install_dir))
        else:
            print(f"[smoke] ERROR exit_code={code} install_dir={install_dir}")
        return code
    except Exception as exc:
        print(f"[smoke] ERROR {exc}")
        print(installer.collect_system_report(host, choice, install_dir))
        return installer.EXIT_ERROR
    finally:
        if args.keep_temp:
            print(f"[smoke] keeping_temp_root={smoke_root}")
        elif smoke_root.exists():
            shutil.rmtree(smoke_root, ignore_errors = True)


if __name__ == "__main__":
    raise SystemExit(main())
