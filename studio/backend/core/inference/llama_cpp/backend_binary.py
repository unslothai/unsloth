# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: binary discovery."""

import os
import shutil
import sys
import time

from pathlib import Path
from typing import Optional

from .launch_support import _expanded_user_path
from .log import logger

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _BinaryDiscoveryMixin:
    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _resolved_studio_root_and_is_legacy() -> "tuple[Optional[Path], bool]":
        """Resolve the directory llama.cpp hangs off and classify it as the
        legacy ~/.unsloth/studio root vs. a custom (env/venv-inferred) root.

        Returns (resolved_root, is_legacy). On any import/resolution failure the
        root is treated as legacy and resolved_root is None -- callers must read
        resolved_root only when is_legacy is False. Shared by
        _find_llama_server_binary (discovery) and _kill_orphaned_servers
        (cleanup) so the two never disagree on which root is legacy.
        """
        try:
            from utils.paths.storage_roots import (  # noqa: WPS433
                studio_root as _sr,
                unsloth_home as _uh,
            )

            # llama.cpp is a sibling of studio/ under the master root, the path run.py and
            # main.py export, so a portable install is never the legacy layout.
            master = _uh()
            if master is not None:
                return master, False
            resolved = _sr()
            legacy_studio = Path.home() / ".unsloth" / "studio"
            try:
                is_legacy = resolved.resolve() == legacy_studio.resolve()
            except (OSError, ValueError):
                is_legacy = resolved == legacy_studio
            return (None if is_legacy else resolved), is_legacy
        except (ImportError, OSError, ValueError):
            return None, True

    @staticmethod
    def _find_llama_server_binary(*, include_denied: bool = False) -> Optional[str]:
        """
        Locate the llama-server binary.

        Search order:
        1.  LLAMA_SERVER_PATH environment variable (direct path to binary)
        1b. UNSLOTH_LLAMA_CPP_PATH env var (custom llama.cpp install dir)
        2.  Unsloth's custom llama.cpp folder setting
        3.  ~/.unsloth/llama.cpp/llama-server        (make build, root dir)
        4.  ~/.unsloth/llama.cpp/build/bin/llama-server  (cmake build, Linux)
        5.  ~/.unsloth/llama.cpp/build/bin/Release/llama-server.exe  (cmake build, Windows)
        6.  ./llama.cpp/llama-server                 (legacy: make build, root dir)
        7.  ./llama.cpp/build/bin/llama-server        (legacy: cmake in-tree build)
        8.  llama-server on PATH                     (system install)
        9.  ./bin/llama-server                       (legacy: extracted binary)
        """
        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        def _file_status(p: Path) -> str:
            # "file", "absent", "non_executable", or "denied". A denied
            # file stays access-denied across a short retry (Windows AV/ACL or
            # an install replace in flight). is_file() raises PermissionError
            # (WinError 5) instead of returning False for the locked case, so
            # never treat it as missing.
            for _ in range(5):
                try:
                    if not p.is_file():
                        return "absent"
                    if sys.platform != "win32" and not os.access(p, os.X_OK):
                        return "non_executable"
                    return "file"
                except PermissionError:
                    time.sleep(0.2)
                except OSError:
                    return "absent"
            return "denied"

        def _is_file(p: Path) -> bool:
            return _file_status(p) == "file"

        def _layout_candidates(d: Path) -> list:
            # Keep Settings validation and runtime discovery on one layout
            # contract so a folder accepted by the UI is always launchable.
            from utils.llama_cpp_path_settings import llama_server_candidates
            return list(llama_server_candidates(d))

        def _unavailable(p: object) -> None:
            # a pinned or managed binary that exists but is access-denied: report
            # it instead of silently downgrading to a lower-priority llama-server
            logger.warning(
                f"llama-server at {p} is not executable or is access-denied "
                "(permissions, antivirus, or an in-flight install); not falling "
                "back to another binary, retry once it is available"
            )
            return None

        def _scan_pinned(paths: list):
            # First executable candidate wins. A genuinely access-denied one
            # stops immediately; a known non-executable candidate is remembered
            # while sibling build layouts are checked. include_denied returns an
            # unavailable path instead: diffusion asset lookup only needs its dir.
            non_executable = None
            from utils.llama_cpp_path_settings import prefer_gpu_capable

            statuses = {p: _file_status(p) for p in paths}
            first = next((p for p in paths if statuses[p] == "file"), None)
            # A first hit proven CPU-only yields to a GPU-capable sibling build (#5941), unless
            # a denied candidate precedes it: that denial stops discovery, it is not skipped over.
            if first is not None and not any(
                statuses[p] == "denied" for p in paths[: paths.index(first)]
            ):
                paths = prefer_gpu_capable(paths, lambda p: statuses[p] == "file")
            for p in paths:
                st = statuses[p]
                if st == "file":
                    return str(p), None
                if st == "denied":
                    return (str(p), None) if include_denied else (None, p)
                if st == "non_executable" and non_executable is None:
                    # A source checkout may leave a stale root entrypoint beside a
                    # valid CMake build. Keep looking within this pinned layout,
                    # but do not fall through to a different runtime if none works.
                    non_executable = p
            if non_executable is not None:
                # include_denied is used only to preserve genuinely transient
                # access-denied paths. A missing execute bit needs repair, not a
                # retry, and must not pass the download preflight as available.
                return (None, None) if include_denied else (None, non_executable)
            return None, None

        # 1. Env var: direct path to binary
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path:
            hit, locked = _scan_pinned([Path(env_path)])
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 1b. UNSLOTH_LLAMA_CPP_PATH: custom llama.cpp install dir
        custom_llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
        managed_path_marker = os.environ.get("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH") == "1"
        if custom_llama_cpp and not managed_path_marker:
            # expanduser, like every other reader of this variable:
            # default_managed_llama_dir, get_stored_custom_llama_cpp_path and the
            # desktop's own pinning all expand it, and the literal lookup here was
            # the only one that did not. A "~/llama.cpp" written into a service
            # unit, a .env or the Windows environment dialog reaches the process
            # unexpanded, and the literal form made this search a folder named ~
            # beside the working directory, walk past it, and load a different
            # runtime than every other component was reporting on.
            # Not Path.expanduser: it raises RuntimeError on a name it cannot
            # resolve, so a stale ~deleted-user pin made discovery itself throw
            # instead of falling through to the rest of the order below.
            hit, locked = _scan_pinned(_layout_candidates(_expanded_user_path(custom_llama_cpp)))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 2. Unsloth setting: a deliberate pin, so a missing or inaccessible
        # selected build must not silently fall through to the bundled runtime.
        # The user would otherwise see their custom path selected while another
        # llama-server actually ran.
        try:
            from utils.llama_cpp_path_settings import get_stored_custom_llama_cpp_path
            studio_custom_llama_cpp = get_stored_custom_llama_cpp_path()
        except Exception:
            studio_custom_llama_cpp = None
        if studio_custom_llama_cpp is not None:
            hit, locked = _scan_pinned(_layout_candidates(studio_custom_llama_cpp))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit
            logger.warning(
                "The custom llama.cpp folder selected in Unsloth no longer contains "
                "%s; not falling back to another runtime",
                binary_name,
            )
            return None

        # 3-5. Match installer layout: env-mode -> $STUDIO_HOME/llama.cpp;
        # default/HOME-redirect -> ~/.unsloth/llama.cpp (sibling of studio).
        legacy_llama = Path.home() / ".unsloth" / "llama.cpp"
        _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
        if _is_legacy:
            search_roots = [legacy_llama]
        else:
            # _kill_orphaned_servers excludes the legacy root in custom mode;
            # discovery must match so we never spawn a server we then refuse to
            # clean up. UNSLOTH_LLAMA_CPP_PATH (handled earlier) is the explicit
            # way to share a build across roots.
            search_roots = [_resolved_sr / "llama.cpp"]
        for unsloth_home in search_roots:
            hit, locked = _scan_pinned(_layout_candidates(unsloth_home))
            if locked is not None:
                return _unavailable(locked)
            if hit:
                return hit

        # 5-6. Legacy: in-tree build (older setup.sh / setup.ps1). A fallback,
        # so a denied candidate here just continues (no no-fallback halt).
        project_root = Path(__file__).resolve().parents[5]
        from utils.llama_cpp_path_settings import prefer_gpu_capable

        for p in prefer_gpu_capable(_layout_candidates(project_root / "llama.cpp"), _is_file):
            if _is_file(p):
                return str(p)

        # 7. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 8. Legacy: extracted to bin/
        bin_path = project_root / "bin" / binary_name
        if _is_file(bin_path):
            return str(bin_path)

        return None
