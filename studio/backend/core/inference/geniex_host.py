#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Authenticated loopback Lemonade API shim driving Qualcomm GenieX on Hexagon NPU (HTP0).

Drop-in counterpart to Unsloth Studio's flm_host.py for Windows on Snapdragon (ARM64).
Started by LemonadeServer with the exact same CLI arguments and LEMONADE_API_KEY environment.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote

import httpx

_HIDDEN = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # CREATE_NO_WINDOW
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
    "authorization",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PCT_RE = re.compile(r"downloading\s+(\d{1,3})%", re.IGNORECASE)
_BRIDGE_DIR = Path(__file__).resolve().parent
_USER_MODELS_JSON = Path.home() / ".unsloth" / "qualcomm-npu-bridge" / "models.json"
_MODELS_JSON = (
    _USER_MODELS_JSON
    if _USER_MODELS_JSON.is_file()
    else (
        _BRIDGE_DIR / "qualcomm_npu_models.json"
        if (_BRIDGE_DIR / "qualcomm_npu_models.json").is_file()
        else _BRIDGE_DIR / "models.json"
    )
)

# Studio's npu_flm_args power modes -> GenieX --power-mode values
_POWER_MODE_MAP = {
    "powersaver": "power_saver",
    "balanced": "balanced",
    "performance": "high_performance",
    "turbo": "burst",
}
_SAC_EXIT_CODE = 0xC0E90002


def _format_exit_code(code: int) -> str:
    """Format a GenieX process exit code, surfacing Smart App Control blocks clearly."""
    if (code & 0xFFFFFFFF) == _SAC_EXIT_CODE:
        return (
            "Windows Smart App Control blocked unsigned geniex.exe/geniex.dll (0xC0E90002). "
            "Check Windows Security -> App & browser control -> Smart App Control, or update "
            "Qualcomm.GenieX.CLI once Authenticode-signed host binaries are published."
        )
    return f"exit code {code} (0x{code & 0xFFFFFFFF:08X})"


def log(message: str, tag: str = "GenieXHost") -> None:
    """Format log lines in lemond's timestamped format so LemonadeServer captures them."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    try:
        print(f"{stamp} [Info] ({tag}) {message}", flush=True)
    except (OSError, ValueError):
        pass


def safe_model_id(checkpoint: str) -> str:
    """Convert a GenieX checkpoint (which may contain '/' and ':') to a Studio-safe NPU id."""
    cleaned = checkpoint.strip().replace("\\", "--").replace("/", "--").replace(":", "-")
    if not cleaned.upper().endswith("-NPU"):
        cleaned = f"{cleaned}-NPU"
    return cleaned


def default_hf_repo(checkpoint: str) -> str:
    """Derive a display repo ending in -NPU so Studio's NPU name matcher recognizes it."""
    base = checkpoint.split(":", 1)[0].strip()
    if base.lower().endswith("-gguf"):
        base = base[:-5]
    if not base.upper().endswith("-NPU"):
        base = f"{base}-NPU"
    return base


def find_geniex_binary() -> Optional[Path]:
    """Locate geniex.exe across env override, PATH, App Paths registry, and Windows install locations."""
    env_bin = os.environ.get("UNSLOTH_GENIEX_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    which_bin = shutil.which("geniex")
    if which_bin and Path(which_bin).is_file():
        return Path(which_bin)
    local_app_data = Path(
        os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    )
    prog_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates: list[Path] = [
        local_app_data / "GenieX" / "geniex.exe",
        local_app_data / "Programs" / "GenieX" / "geniex.exe",
        prog_files / "GenieX" / "geniex.exe",
        local_app_data / "GenieX CLI" / "geniex.exe",
        local_app_data / "Programs" / "GenieX CLI" / "geniex.exe",
        local_app_data / "Microsoft" / "WinGet" / "Links" / "geniex.exe",
        prog_files / "GenieX CLI" / "geniex.exe",
    ]
    if sys.platform == "win32":
        try:
            import winreg

            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(
                        root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\geniex.exe"
                    ) as key:
                        reg_path, _ = winreg.QueryValueEx(key, None)
                        if isinstance(reg_path, str) and reg_path.strip():
                            candidates.insert(0, Path(reg_path.strip()))
                except OSError:
                    pass
        except ImportError:
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10.0,
                **_HIDDEN,
            )
        except Exception:
            pass
    try:
        proc.kill()
        proc.wait(timeout=5.0)
    except Exception:
        pass


class GenieX:
    """Wrapper around geniex.exe CLI for catalog queries, pulling, and removing models."""

    def __init__(self, cache_dir: Path) -> None:
        binary = find_geniex_binary()
        if binary is None:
            raise SystemExit(
                "Qualcomm GenieX CLI (geniex.exe) was not found. "
                "Install it with: winget install --id Qualcomm.GenieX.CLI -e"
            )
        self.binary = binary
        self.cache_dir = cache_dir
        self._cache_lock = threading.Lock()
        self._cached_models_cache: Optional[tuple[float, list[dict[str, Any]]]] = None
        self._exports_synced_at: Optional[float] = None

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._cached_models_cache = None
            self._exports_synced_at = None

    def run(
        self,
        *args: str,
        timeout: float = 600.0,
        stdin: str = "",
    ) -> tuple[int, str]:
        cmd = [str(self.binary), *args, "--skip-update"]
        try:
            done = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **_HIDDEN,
            )
        except OSError as exc:
            return 1, f"[ERROR] {exc}"
        output = (done.stdout or "") + (done.stderr or "")
        if done.returncode and (not output.strip() or (done.returncode & 0xFFFFFFFF) == _SAC_EXIT_CODE):
            detail = _format_exit_code(done.returncode)
            output = f"[ERROR] {detail}" + (f"\n{output}" if output.strip() else "")
        return done.returncode, output

    def version_info(self) -> tuple[int, str, Optional[str]]:
        code, output = self.run("--version", timeout=15.0)
        version = None
        for line in output.splitlines():
            if "GenieX CLI Version:" in line:
                version = line.split(":", 1)[1].strip().lstrip("v")
                break
        return code, output, version

    def cached_models(self, max_age: float = 5.0) -> list[dict[str, Any]]:
        with self._cache_lock:
            now = time.monotonic()
            if (
                max_age > 0
                and self._cached_models_cache is not None
                and (now - self._cached_models_cache[0]) < max_age
            ):
                return list(self._cached_models_cache[1])
            code, output = self.run("list", "--format", "json", timeout=60.0)
            start = output.find("[")
            if code or start < 0:
                return []
            try:
                data = json.loads(output[start:])
                rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
            except ValueError:
                return []
            self._cached_models_cache = (time.monotonic(), rows)
            return list(rows)

    def curated_presets(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(_MODELS_JSON.read_text(encoding="utf-8"))
            models = raw.get("models") if isinstance(raw, dict) else None
            return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []
        except (OSError, ValueError):
            return []

    def _geniex_data_dir(self) -> Path:
        env_dir = os.environ.get("GENIEX_DATADIR", "").strip()
        return Path(os.path.expanduser(env_dir)) if env_dir else (Path.home() / ".cache" / "geniex")

    def _precision_meta(
        self, name: str, precision: Optional[str], fallback_bytes: Any
    ) -> tuple[Optional[float], Optional[str]]:
        size_gb: Optional[float] = None
        mtp_path: Optional[str] = None
        parts = [p for p in name.strip().replace("\\", "/").split("/") if p]
        if len(parts) >= 2:
            model_dir = self._geniex_data_dir() / "models" / Path(*parts)
            for manifest_name in ("geniex.json", "metadata.json"):
                manifest = model_dir / manifest_name
                if not manifest.is_file():
                    continue
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if precision and size_gb is None:
                        mfile = (data.get("ModelFile") or {}).get(precision) or {}
                        p_size = mfile.get("Size")
                        if isinstance(p_size, (int, float)) and p_size > 0:
                            size_gb = round(float(p_size) / 1e9, 2)
                    if mtp_path is None:
                        mtp_obj = data.get("MtpFile") or {}
                        if isinstance(mtp_obj, dict) and mtp_obj.get("Name"):
                            cand = model_dir / str(mtp_obj["Name"])
                            if cand.is_file() and cand.stat().st_size > 0:
                                mtp_path = str(cand)
                except (OSError, ValueError, TypeError):
                    pass
            if mtp_path is None and model_dir.is_dir():
                for cand in sorted(model_dir.glob("mtp-*.gguf")):
                    if cand.is_file() and cand.stat().st_size > 0:
                        mtp_path = str(cand)
                        break
        if size_gb is None and isinstance(fallback_bytes, (int, float)) and fallback_bytes > 0:
            size_gb = round(float(fallback_bytes) / 1e9, 2)
        return size_gb, mtp_path

    def _precision_size_gb(self, name: str, precision: Optional[str], fallback_bytes: Any) -> Optional[float]:
        size_gb, _mtp_path = self._precision_meta(name, precision, fallback_bytes)
        return size_gb

    def _sync_studio_exports(self, max_age: float = 15.0) -> None:
        """Auto-discover Hexagon-compatible GGUFs under Unsloth Studio's exports/ folder."""
        with self._cache_lock:
            now = time.monotonic()
            if (
                max_age > 0
                and self._exports_synced_at is not None
                and (now - self._exports_synced_at) < max_age
            ):
                return
            self._exports_synced_at = now
        try:
            try:
                import convert_to_hexagon
            except ImportError:
                from backend.core.inference import convert_to_hexagon  # type: ignore[no-redef]

            convert_to_hexagon.scan_studio_exports(self._geniex_data_dir(), auto_quantize=False)
        except Exception:
            pass

    def catalog(self) -> list[dict[str, Any]]:
        self._sync_studio_exports()
        profile = detect_system_profile(self._geniex_data_dir())
        cached = self.cached_models()
        # Map "org/repo:QUANT" -> (ckpt_actual, prec, entry) and "org/repo" -> list[...]
        cached_by_checkpoint: dict[str, tuple[str, Optional[str], dict[str, Any]]] = {}
        cached_by_repo: dict[str, list[tuple[str, Optional[str], dict[str, Any]]]] = {}
        for entry in cached:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            precs = entry.get("precisions") or []
            if isinstance(precs, list) and precs:
                for prec in precs:
                    ckpt = f"{name}:{prec}"
                    cached_by_checkpoint[ckpt.lower()] = (ckpt, str(prec), entry)
                    cached_by_repo.setdefault(name.lower(), []).append((ckpt, str(prec), entry))
            else:
                cached_by_checkpoint[name.lower()] = (name, None, entry)
                cached_by_repo.setdefault(name.lower(), []).append((name, None, entry))

        rows: list[dict[str, Any]] = []
        seen_checkpoints: set[str] = set()

        def _attach_plan_fields(row: dict[str, Any]) -> dict[str, Any]:
            plan = plan_model_execution(row, profile, requested_ctx=row.get("max_context_window") or 8192)
            row["tier"] = plan["tier"]
            row["ssd_offload"] = plan["ssd_offload"]
            row["mtp_enabled"] = bool(plan["mtp_path"])
            return row

        for preset in self.curated_presets():
            ckpt = str(preset.get("checkpoint") or "").strip()
            if not ckpt:
                continue
            key = ckpt.lower()
            repo_key = ckpt.split(":", 1)[0].lower()
            hit = cached_by_checkpoint.get(key)
            if hit is None and repo_key in cached_by_repo and ":" not in ckpt:
                hit = cached_by_repo[repo_key][0]
            matched_entry = hit[2] if hit is not None else None
            matched_prec = hit[1] if hit is not None else None
            if hit is not None:
                ckpt = hit[0]
                key = ckpt.lower()
            seen_checkpoints.add(key)
            downloaded = matched_entry is not None
            size_gb = preset.get("size_gb")
            mtp_path: Optional[str] = None
            if downloaded:
                actual_gb, mtp_path = self._precision_meta(
                    str(matched_entry.get("name") or ""),
                    matched_prec,
                    matched_entry.get("size"),
                )
                if actual_gb is not None:
                    size_gb = actual_gb
            labels = [str(l) for l in (preset.get("labels") or ["chat"])]
            if "chat" not in labels:
                labels.insert(0, "chat")
            row = {
                "id": safe_model_id(ckpt),
                "checkpoint": ckpt,
                "hf_repo": str(preset.get("hf_repo") or default_hf_repo(ckpt)),
                "labels": labels,
                "size": size_gb,
                "n_layers": int(preset.get("n_layers") or 40),
                "min_ram_gb": int(preset.get("min_ram_gb") or 8),
                "mtp_file": preset.get("mtp_file"),
                "mtp_path": mtp_path,
                "downloaded": downloaded,
                "recipe": "flm",
                "max_context_window": int(preset.get("max_context_window") or 32768),
            }
            rows.append(_attach_plan_fields(row))

        # Include every remaining cached repo/precision discovered via geniex CLI or local exports
        for ckpt_lower, (ckpt, prec, entry) in cached_by_checkpoint.items():
            if ckpt_lower in seen_checkpoints:
                continue
            seen_checkpoints.add(ckpt_lower)
            size_gb, mtp_path = self._precision_meta(
                str(entry.get("name") or ""),
                prec,
                entry.get("size"),
            )
            mtype = str(entry.get("type") or "llm").lower()
            labels = ["chat", "vision"] if mtype == "vlm" else ["chat", "tool-calling"]
            row = {
                "id": safe_model_id(ckpt),
                "checkpoint": ckpt,
                "hf_repo": default_hf_repo(ckpt),
                "labels": labels,
                "size": size_gb,
                "n_layers": 40,
                "min_ram_gb": 8,
                "mtp_file": Path(mtp_path).name if mtp_path else None,
                "mtp_path": mtp_path,
                "downloaded": True,
                "recipe": "flm",
                "max_context_window": 32768,
            }
            rows.append(_attach_plan_fields(row))

        return rows


class Serving:
    """Manages the single active `geniex serve` child process."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.respawn_lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.model_id: Optional[str] = None
        self.tag: Optional[str] = None
        self.ctx: Optional[int] = None
        self.port: Optional[int] = None
        self.desired_load: Optional[dict[str, Any]] = None

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        with self.lock:
            process, self.process = self.process, None
            self.model_id = self.tag = self.ctx = self.port = None
        if process is not None:
            _kill_process_tree(process)


def detect_system_profile(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    """Return physical RAM and free NVMe/disk capacity (supports UNSLOTH_SIMULATE_RAM_GB)."""
    total_ram_gb = 16.0
    avail_ram_gb = 10.0

    sim_ram = os.environ.get("UNSLOTH_SIMULATE_RAM_GB", "").strip()
    if sim_ram:
        try:
            val = float(sim_ram)
            if val > 0:
                total_ram_gb = val
                avail_ram_gb = round(val * 0.72, 2)
                sim_used = True
            else:
                sim_used = False
        except ValueError:
            sim_used = False
    else:
        sim_used = False

    if not sim_used and sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
                avail_ram_gb = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass

    disk_free_gb = 0.0
    try:
        probe_path = cache_dir if cache_dir and cache_dir.exists() else Path.home()
        disk_free_gb = round(shutil.disk_usage(probe_path).free / (1024**3), 2)
    except Exception:
        pass

    return {
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "disk_free_gb": disk_free_gb,
    }


def plan_model_execution(
    row: dict[str, Any],
    profile: Optional[dict[str, Any]] = None,
    requested_ctx: int = 8192,
) -> dict[str, Any]:
    """Compute RAM-aware In-RAM NPU vs Hybrid NPU + Read-Only NVMe SSD mmap execution plan."""
    prof = profile or detect_system_profile()
    total_ram_gb = max(4.0, float(prof.get("total_ram_gb") or 16.0))
    raw_avail = prof.get("avail_ram_gb")
    avail_ram_gb = (
        max(0.0, float(raw_avail))
        if isinstance(raw_avail, (int, float))
        else (total_ram_gb * 0.72)
    )
    usable_npu_ram_gb = max(2.0, total_ram_gb * 0.65)
    size_gb = float(row.get("size") or 0.0)
    total_layers = max(8, int(row.get("n_layers") or 40))
    headroom_gb = avail_ram_gb - size_gb - 1.2

    disable_mtp = os.environ.get("UNSLOTH_DISABLE_MTP", "").strip().lower() in ("1", "true", "yes")
    disable_htp_tuning = os.environ.get("UNSLOTH_DISABLE_HEXAGON_TUNING", "").strip().lower() in ("1", "true", "yes")
    mtp_path = None if disable_mtp else (row.get("mtp_path") or None)

    ssd_offload = size_gb > usable_npu_ram_gb
    if total_ram_gb <= 10.0:
        max_ctx = 4096
    elif total_ram_gb <= 20.0:
        if ssd_offload:
            max_ctx = 4096
        elif headroom_gb < 1.5:
            max_ctx = 4096
        elif headroom_gb < 3.5:
            max_ctx = 8192
        else:
            max_ctx = 16384
    elif total_ram_gb <= 48.0:
        max_ctx = 8192 if (not ssd_offload and headroom_gb < 1.5) else 32768
    else:
        max_ctx = 16384 if (not ssd_offload and headroom_gb < 1.5) else 65536

    req_ctx = int(requested_ctx) if isinstance(requested_ctx, (int, float)) and int(requested_ctx) > 0 else 8192
    ctx = min(req_ctx, max_ctx)

    attempts: list[dict[str, Any]] = []
    if not ssd_offload:
        tier = "in_ram_npu"
        # GenieX v0.12.0+ (QAIRT 2.51): DMA64 + OPFUSION trimmed Moon-Beam warm-load 2.26s & lifted prefill
        env_overrides = (
            {}
            if disable_htp_tuning
            else {"GGML_HEXAGON_DMA64": "1", "GGML_HEXAGON_OPFUSION": "1"}
        )
        attempts.append({"compute": "npu", "ngl": -1, "use_mtp": bool(mtp_path), "ctx": ctx, "warm_timeout": 240.0})
        if mtp_path:
            attempts.append({"compute": "npu", "ngl": -1, "use_mtp": False, "ctx": ctx, "warm_timeout": 180.0})
        if ctx > 4096:
            attempts.append({"compute": "npu", "ngl": -1, "use_mtp": False, "ctx": 4096, "warm_timeout": 180.0})
        attempts.append({"compute": "cpu", "ngl": None, "use_mtp": False, "ctx": ctx, "warm_timeout": 180.0})
    else:
        tier = "hybrid_npu_ssd"
        # Stream remaining weight rows directly from read-only mmap NVMe without anonymous repacking
        env_overrides = {
            "GGML_HEXAGON_RT_REPACK": "0",
            "LLAMA_ARG_REPACK": "0",
        }
        tight_live_ram = avail_ram_gb < min(10.5, total_ram_gb * 0.68)
        if tight_live_ram:
            # Skip doomed high-ngl / MTP pinning probes so /v1/load hits working ngl=0 well within Studio's 900s budget
            attempts.append({"compute": "npu", "ngl": 0, "use_mtp": False, "ctx": ctx, "warm_timeout": 540.0})
            attempts.append({"compute": "cpu", "ngl": None, "use_mtp": False, "ctx": ctx, "warm_timeout": 240.0})
        else:
            ngl = max(4, min(total_layers - 2, int((usable_npu_ram_gb * 0.72 / size_gb) * total_layers)))
            half_ngl = max(4, ngl // 2)
            if mtp_path:
                attempts.append({"compute": "npu", "ngl": ngl, "use_mtp": True, "ctx": ctx, "warm_timeout": 90.0})
                attempts.append({"compute": "npu", "ngl": ngl, "use_mtp": False, "ctx": ctx, "warm_timeout": 90.0})
            else:
                attempts.append({"compute": "npu", "ngl": ngl, "use_mtp": False, "ctx": ctx, "warm_timeout": 100.0})
                if half_ngl < ngl:
                    attempts.append({"compute": "npu", "ngl": half_ngl, "use_mtp": False, "ctx": ctx, "warm_timeout": 90.0})
            attempts.append({"compute": "npu", "ngl": 0, "use_mtp": False, "ctx": ctx, "warm_timeout": 480.0})
            attempts.append({"compute": "cpu", "ngl": None, "use_mtp": False, "ctx": ctx, "warm_timeout": 160.0})

    return {
        "tier": tier,
        "ssd_offload": ssd_offload,
        "usable_npu_ram_gb": round(usable_npu_ram_gb, 2),
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": round(avail_ram_gb, 2),
        "headroom_gb": round(headroom_gb, 2),
        "ctx": ctx,
        "max_ctx": max_ctx,
        "mtp_path": mtp_path,
        "env_overrides": env_overrides,
        "attempts": attempts,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _relay(process: subprocess.Popen, tail: list[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        clean = _ANSI_RE.sub("", line).rstrip()
        if clean:
            tail.append(clean)
            del tail[:-80]
            log(clean, "Process")


def _translate_flm_args(flm_args: Optional[str]) -> list[str]:
    """Translate Studio's FLM flags (--pmode, speculative flags, etc.) to GenieX serve CLI flags."""
    if not isinstance(flm_args, str) or not flm_args.strip():
        return []
    tokens = flm_args.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--pmode" and i + 1 < len(tokens):
            mapped = _POWER_MODE_MAP.get(tokens[i + 1].strip().lower())
            if mapped:
                out.extend(["--power-mode", mapped])
            i += 2
            continue
        if tok in ("--spec-type", "--draft-model", "--draft-tokens", "--draft-min", "--draft-p-min") and i + 1 < len(tokens):
            out.extend([tok, tokens[i + 1]])
            i += 2
            continue
        if tok in ("--prefill-chunk-len", "--img-pre-resize") and i + 1 < len(tokens):
            # AMD FastFlowLM-specific tuning flags; safely ignored on Hexagon
            i += 2
            continue
        i += 1
    return out


def make_handler(
    geniex: GenieX,
    serving: Serving,
    api_key: str,
):
    client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(600.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=0),
    )

    def find(model_name: str) -> dict[str, Any]:
        wanted = model_name.strip()
        for row in geniex.catalog():
            if wanted in (row["id"], row["checkpoint"]) or wanted.lower() in (
                row["id"].lower(),
                row["checkpoint"].lower(),
            ):
                return row
        raise KeyError(f"{model_name} is not in the Qualcomm Hexagon NPU catalog.")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args) -> None:
            pass

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "server_error"}})

        def _authorized(self) -> bool:
            if self.path == "/live":
                return True
            sent = self.headers.get("Authorization", "").encode("utf-8")
            if api_key and not hmac.compare_digest(sent, f"Bearer {api_key}".encode("utf-8")):
                self._error(401, "Invalid API key.")
                return False
            return True

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw) if raw else {}

        def do_GET(self) -> None:
            if not self._authorized():
                return
            path = self.path.split("?", 1)[0]
            if path == "/live":
                self._json(200, {"status": "ok"})
            elif path == "/v1/health":
                loaded = []
                with serving.lock:
                    if serving.alive():
                        loaded.append(
                            {
                                "model_name": serving.model_id,
                                "checkpoint": serving.tag,
                                "device": "npu",
                                "recipe": "flm",
                                "loaded": True,
                                "recipe_options": {"ctx_size": serving.ctx},
                            }
                        )
                self._json(200, {"status": "ok", "all_models_loaded": loaded})
            elif path == "/v1/models":
                query = parse_qs(self.path.partition("?")[2])
                show_all = (query.get("show_all") or [""])[0].lower() in ("true", "1")
                try:
                    rows = geniex.catalog()
                    rows = rows if show_all else [r for r in rows if r["downloaded"]]
                    self._json(200, {"object": "list", "data": rows})
                except Exception as exc:  # noqa: BLE001
                    self._error(500, str(exc))
            elif path.startswith("/v1/models/") and path.endswith("/options"):
                self._json(200, {"effective": {"flm_args": "", "ctx_size": -1}, "saved": {}})
            else:
                self._proxy("GET")

        def do_POST(self) -> None:
            if not self._authorized():
                return
            path = self.path.split("?", 1)[0]
            routes = {
                "/internal/set": lambda body: self._json(200, {}),
                "/v1/install": self._install,
                "/v1/pull": self._pull,
                "/v1/load": self._load,
                "/v1/unload": self._unload,
                "/v1/delete": self._delete,
            }
            if path not in routes:
                self._proxy("POST")
                return
            try:
                body = self._body()
            except ValueError:
                self._error(400, "Invalid JSON.")
                return
            routes[path](body if isinstance(body, dict) else {})

        def _install(self, body: dict[str, Any]) -> None:
            code, output, _ver = geniex.version_info()
            if code:
                self._error(500, f"Qualcomm GenieX does not run: {output.strip()[-300:]}")
            else:
                self._json(200, {"status": "success"})

        def _pull(self, body: dict[str, Any]) -> None:
            try:
                row = find(str(body.get("model_name") or ""))
            except Exception as exc:  # noqa: BLE001
                self._error(404, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            cmd = [str(geniex.binary), "pull", row["checkpoint"], "--skip-update"]
            log(f"Pulling model: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    **_HIDDEN,
                )
            except OSError as exc:
                err_evt = {"error": f"geniex pull could not start: {exc}"}
                self.wfile.write(f"event: error\ndata: {json.dumps(err_evt)}\n\n".encode("utf-8"))
                self.wfile.flush()
                self.close_connection = True
                return

            last_pct = -1
            tail: list[str] = []
            buf = bytearray()
            try:
                assert proc.stdout is not None
                while True:
                    chunk = proc.stdout.read(256)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while True:
                        idx_r = buf.find(b"\r")
                        idx_n = buf.find(b"\n")
                        if idx_r < 0 and idx_n < 0:
                            break
                        cut = idx_r if (idx_n < 0 or (0 <= idx_r < idx_n)) else idx_n
                        raw_line = bytes(buf[:cut]).decode("utf-8", errors="replace")
                        del buf[: cut + 1]
                        clean = _ANSI_RE.sub("", raw_line).strip()
                        if not clean:
                            continue
                        m = _PCT_RE.search(clean)
                        if m:
                            pct = min(99, max(0, int(m.group(1))))
                            if pct > last_pct:
                                last_pct = pct
                                evt = {"status": "downloading", "percent": pct, "model": row["id"]}
                                self.wfile.write(
                                    f"event: progress\ndata: {json.dumps(evt)}\n\n".encode("utf-8")
                                )
                                self.wfile.flush()
                        else:
                            tail.append(clean)
                            del tail[:-30]
                            log(clean, "Process")
                code = proc.wait(timeout=30.0)
            except (BrokenPipeError, ConnectionResetError):
                _kill_process_tree(proc)
                self.close_connection = True
                return
            except Exception as exc:  # noqa: BLE001
                _kill_process_tree(proc)
                code = 1
                tail.append(str(exc))

            if code:
                geniex.invalidate_cache()
                reason = next((l for l in reversed(tail) if l), f"geniex pull exited {code}")
                self.wfile.write(
                    f"event: error\ndata: {json.dumps({'error': reason})}\n\n".encode("utf-8")
                )
            else:
                geniex.invalidate_cache()
                self.wfile.write(
                    f"event: complete\ndata: {json.dumps({'status': 'success', 'model': row['id'], 'percent': 100})}\n\n".encode(
                        "utf-8"
                    )
                )
            self.wfile.flush()
            self.close_connection = True

        def _start_serving(
            self,
            row: dict[str, Any],
            requested_ctx: int,
            flm_args: Optional[str],
            *,
            step_down_ctx: bool = False,
            prev_ctx: Optional[int] = None,
        ) -> tuple[Optional[dict[str, Any]], str]:
            # Stop previous child BEFORE probing available RAM so freed weights count as headroom
            serving.stop()
            eff_req_ctx = int(requested_ctx)
            if step_down_ctx:
                base_ctx = int(prev_ctx) if isinstance(prev_ctx, int) and prev_ctx > 0 else eff_req_ctx
                eff_req_ctx = max(4096, min(eff_req_ctx, base_ctx) // 2)

            profile = detect_system_profile(geniex._geniex_data_dir())
            plan = plan_model_execution(row, profile, requested_ctx=eff_req_ctx)
            plan_ctx = int(plan["ctx"])
            # Explicit user env vars always take precedence over default plan tuning overrides
            child_env = {**plan["env_overrides"], **os.environ}
            warm_timeout = 420.0 if plan["ssd_offload"] else 300.0
            flm_extra = _translate_flm_args(flm_args)
            disable_spec = os.environ.get("UNSLOTH_DISABLE_SPEC", "").strip().lower() in ("1", "true", "yes")
            env_spec_type = "" if disable_spec else os.environ.get("UNSLOTH_SPEC_TYPE", "").strip()
            env_draft_model = os.environ.get("UNSLOTH_DRAFT_MODEL", "").strip()
            env_draft_tokens = os.environ.get("UNSLOTH_DRAFT_TOKENS", "").strip()

            last_error = "Failed to start GenieX."
            for idx, attempt in enumerate(plan["attempts"]):
                attempt_ctx = int(attempt.get("ctx") or plan_ctx)
                attempt_warm_timeout = float(attempt.get("warm_timeout") or warm_timeout)
                port = _free_port()
                cmd = [
                    str(geniex.binary),
                    "serve",
                    "--host",
                    f"127.0.0.1:{port}",
                    "-c",
                    attempt["compute"],
                    "--model",
                    row["checkpoint"],
                    "--nctx",
                    str(attempt_ctx),
                    "--keepalive",
                    "3600",
                    "--skip-update",
                    *flm_extra,
                ]
                if plan["ssd_offload"] and attempt["compute"] in ("npu", "hybrid") and "--power-mode" not in flm_extra:
                    cmd.extend(["--power-mode", "burst"])
                if attempt.get("ngl") is not None:
                    cmd.extend(["--ngl", str(attempt["ngl"])])
                if attempt.get("use_mtp") and plan.get("mtp_path"):
                    cmd.extend(["--spec-type", "draft-mtp", "--draft-model", str(plan["mtp_path"])])
                elif env_spec_type and "--spec-type" not in flm_extra and attempt["compute"] == "npu":
                    cmd.extend(["--spec-type", env_spec_type])
                    if env_draft_model and "--draft-model" not in flm_extra:
                        cmd.extend(["--draft-model", env_draft_model])
                    if env_draft_tokens and "--draft-tokens" not in flm_extra:
                        cmd.extend(["--draft-tokens", env_draft_tokens])

                log(
                    f"Starting [{plan['tier']}] attempt {idx + 1}/{len(plan['attempts'])} "
                    f"(RAM={profile['total_ram_gb']}GB, avail={profile['avail_ram_gb']}GB, "
                    f"size={row.get('size')}GB, ctx={attempt_ctx}): {' '.join(cmd)}"
                )
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=child_env,
                        **_HIDDEN,
                    )
                except OSError as exc:
                    last_error = f"GenieX failed to start: {exc}"
                    continue

                tail: list[str] = []
                threading.Thread(target=_relay, args=(process, tail), daemon=True).start()
                deadline = time.monotonic() + 60.0
                http_ready = False
                early_err: Optional[str] = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        time.sleep(0.1)
                        code = process.returncode
                        if (code & 0xFFFFFFFF) == _SAC_EXIT_CODE:
                            reason = _format_exit_code(code)
                        else:
                            reason = next(
                                (l for l in reversed(tail) if l.strip()),
                                _format_exit_code(code),
                            )
                        early_err = f"GenieX exited early: {reason}"
                        break
                    try:
                        resp = client.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
                        if resp.status_code == 200:
                            http_ready = True
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)

                if not http_ready:
                    _kill_process_tree(process)
                    last_error = early_err or "GenieX did not start in time"
                    log(f"Attempt {idx + 1} failed ({last_error}); trying fallback if available...")
                    continue

                # Warm-load weights / mmap pages onto Hexagon HTP0 + host cache now
                try:
                    warm = client.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={
                            "model": row["checkpoint"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                            "stream": False,
                            "enable_think": False,
                        },
                        timeout=attempt_warm_timeout,
                    )
                    if warm.status_code != 200:
                        _kill_process_tree(process)
                        reason = warm.text.strip()[:300] or f"HTTP {warm.status_code}"
                        last_error = f"Warm-load failed: {reason}"
                        log(f"Attempt {idx + 1} failed ({last_error}); trying fallback if available...")
                        continue
                except Exception as exc:  # noqa: BLE001
                    _kill_process_tree(process)
                    last_error = f"Warm-load error: {exc}"
                    log(f"Attempt {idx + 1} failed ({last_error}); trying fallback if available...")
                    continue

                with serving.lock:
                    serving.process = process
                    serving.model_id = row["id"]
                    serving.tag = row["checkpoint"]
                    serving.ctx = attempt_ctx
                    serving.port = port
                    serving.desired_load = {
                        "row": row,
                        "requested_ctx": int(requested_ctx),
                        "flm_args": flm_args,
                        "last_ctx": attempt_ctx,
                    }
                log(
                    f"Loaded {row['checkpoint']} ({row['id']}) tier={plan['tier']} "
                    f"compute={attempt['compute']} ngl={attempt.get('ngl')} mtp={attempt.get('use_mtp')} "
                    f"on port {port} (ctx={attempt_ctx})"
                )
                return {"status": "success", "model_name": row["id"], "tier": plan["tier"]}, ""

            return None, last_error

        def _ensure_alive_or_respawn(
            self,
            *,
            step_down_ctx: bool = True,
            force: bool = False,
            expected_port: Optional[int] = None,
        ) -> tuple[Optional[int], Optional[str], Optional[str]]:
            with serving.lock:
                if not force and serving.alive():
                    return serving.port, serving.tag, None
                desired = serving.desired_load
            if desired is None:
                return None, None, None
            with serving.respawn_lock:
                with serving.lock:
                    if serving.alive() and (
                        not force or (expected_port is not None and serving.port != expected_port)
                    ):
                        return serving.port, serving.tag, None
                    desired = serving.desired_load
                if desired is None:
                    return None, None, None
                log(
                    f"GenieX child exited/unresponsive for {desired['row']['id']}; "
                    f"self-healing respawn (step_down_ctx={step_down_ctx}, force={force})..."
                )
                result, err = self._start_serving(
                    desired["row"],
                    int(desired["requested_ctx"]),
                    desired.get("flm_args"),
                    step_down_ctx=step_down_ctx,
                    prev_ctx=desired.get("last_ctx"),
                )
                if result is not None:
                    with serving.lock:
                        return serving.port, serving.tag, None
                return None, None, err

        def _load(self, body: dict[str, Any]) -> None:
            geniex.invalidate_cache()
            name = str(body.get("model_name") or "")
            try:
                row = find(name)
            except Exception as exc:  # noqa: BLE001
                self._error(404, str(exc))
                return

            requested_ctx = (
                body.get("ctx_size")
                if isinstance(body.get("ctx_size"), int) and body.get("ctx_size") > 0
                else (row.get("max_context_window") or 8192)
            )
            result, last_error = self._start_serving(
                row,
                int(requested_ctx),
                body.get("flm_args"),
                step_down_ctx=False,
            )
            if result is not None:
                self._json(200, result)
                return
            with serving.lock:
                serving.desired_load = None
            self._error(500, f"Failed to load model '{row['id']}': {last_error}")

        def _unload(self, body: dict[str, Any]) -> None:
            with serving.lock:
                serving.desired_load = None
            serving.stop()
            self._json(200, {"status": "success"})

        def _delete(self, body: dict[str, Any]) -> None:
            try:
                row = find(str(body.get("model_name") or ""))
            except Exception as exc:  # noqa: BLE001
                self._error(422, str(exc))
                return
            if not row.get("downloaded"):
                # npu_backend._download calls /v1/delete before /v1/pull when files is None
                self._json(200, {"status": "success"})
                return
            with serving.lock:
                desired_id = (serving.desired_load or {}).get("row", {}).get("id")
                if serving.model_id == row["id"] or desired_id == row["id"]:
                    serving.desired_load = None
            if serving.model_id == row["id"]:
                serving.stop()
            code, output = geniex.run("remove", row["checkpoint"], "-y", timeout=120.0)
            if code and ":" in row["checkpoint"]:
                code, output = geniex.run(
                    "remove", row["checkpoint"].split(":", 1)[0], "-y", timeout=120.0
                )
            geniex.invalidate_cache()
            if code:
                self._error(500, output.strip()[-300:] or "geniex remove failed")
            else:
                self._json(200, {"status": "success"})

        def _proxy(self, method: str) -> None:
            with serving.lock:
                port, tag = (serving.port, serving.tag) if serving.alive() else (None, None)
            if port is None:
                if method == "POST" and serving.desired_load is not None:
                    port, tag, respawn_err = self._ensure_alive_or_respawn(step_down_ctx=True)
                    if port is None:
                        self._error(
                            503,
                            respawn_err or f"No route {self.path}, and no NPU model is loaded.",
                        )
                        return
                else:
                    self._error(
                        404 if method == "GET" else 503,
                        f"No route {self.path}, and no NPU model is loaded.",
                    )
                    return
            raw = (
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if method == "POST"
                else b""
            )
            payload_obj: Optional[dict[str, Any]] = None
            is_stream = False
            if raw and "json" in (self.headers.get("Content-Type") or ""):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        payload_obj = payload
                        payload["model"] = tag
                        is_stream = bool(payload.get("stream"))
                        if "think" in payload:
                            think_flag = bool(payload.pop("think"))
                            payload["enable_think"] = think_flag
                            kwargs = payload.get("chat_template_kwargs")
                            if not isinstance(kwargs, dict):
                                kwargs = {}
                                payload["chat_template_kwargs"] = kwargs
                            kwargs.setdefault("enable_thinking", think_flag)
                        raw = json.dumps(payload).encode("utf-8")
                except ValueError:
                    pass
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
            t_start = time.monotonic()
            t_first: Optional[float] = None
            token_chunks = 0
            for req_attempt in range(2):
                headers_sent = False
                try:
                    with client.stream(
                        method,
                        f"http://127.0.0.1:{port}{self.path}",
                        content=raw or None,
                        headers=headers,
                    ) as upstream:
                        self.send_response(upstream.status_code)
                        for key, value in upstream.headers.items():
                            if key.lower() not in _HOP_HEADERS:
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        headers_sent = True
                        if not is_stream or upstream.status_code != 200:
                            for chunk in upstream.iter_raw():
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        else:
                            # Relay SSE lines and inject timings into the final finish_reason chunk
                            for line in upstream.iter_lines():
                                if not line:
                                    self.wfile.write(b"\n")
                                    self.wfile.flush()
                                    continue
                                if line.startswith("data:") and line != "data:[DONE]" and line != "data: [DONE]":
                                    now = time.monotonic()
                                    try:
                                        obj = json.loads(line[5:].strip())
                                    except ValueError:
                                        obj = None
                                    if isinstance(obj, dict):
                                        choices = obj.get("choices") or []
                                        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                                        delta = first_choice.get("delta") or {}
                                        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                                            if t_first is None:
                                                t_first = now
                                            token_chunks += 1
                                        if first_choice.get("finish_reason") is not None and "timings" not in obj:
                                            if t_first is None:
                                                t_first = now
                                            prompt_ms = max(1.0, (t_first - t_start) * 1000.0)
                                            pred_ms = max(1.0, (now - t_first) * 1000.0)
                                            n_pred = max(1, token_chunks)
                                            obj["timings"] = {
                                                "prompt_ms": round(prompt_ms, 2),
                                                "predicted_n": n_pred,
                                                "predicted_ms": round(pred_ms, 2),
                                                "predicted_per_second": round(n_pred / (pred_ms / 1000.0), 2),
                                            }
                                            line = "data: " + json.dumps(obj)
                                self.wfile.write((line + "\n").encode("utf-8"))
                                self.wfile.flush()
                    break
                except (BrokenPipeError, ConnectionResetError) as exc:
                    if (
                        not headers_sent
                        and req_attempt == 0
                        and method == "POST"
                        and serving.desired_load is not None
                    ):
                        new_port, new_tag, respawn_err = self._ensure_alive_or_respawn(
                            step_down_ctx=True, force=True, expected_port=port
                        )
                        if new_port is not None:
                            port, tag = new_port, new_tag
                            if payload_obj is not None and new_tag is not None:
                                payload_obj["model"] = new_tag
                                raw = json.dumps(payload_obj).encode("utf-8")
                            t_start = time.monotonic()
                            t_first = None
                            token_chunks = 0
                            continue
                        self._error(503, f"Qualcomm GenieX did not answer and respawn failed: {respawn_err}")
                    break
                except (httpx.HTTPError, OSError) as exc:
                    if (
                        not headers_sent
                        and req_attempt == 0
                        and method == "POST"
                        and serving.desired_load is not None
                    ):
                        child_healthy = serving.alive()
                        if child_healthy:
                            try:
                                probe = client.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.0)
                                child_healthy = probe.status_code == 200
                            except (httpx.HTTPError, OSError):
                                child_healthy = False
                        new_port, new_tag, respawn_err = self._ensure_alive_or_respawn(
                            step_down_ctx=True,
                            force=not child_healthy,
                            expected_port=port,
                        )
                        if new_port is not None:
                            port, tag = new_port, new_tag
                            if payload_obj is not None and new_tag is not None:
                                payload_obj["model"] = new_tag
                                raw = json.dumps(payload_obj).encode("utf-8")
                            t_start = time.monotonic()
                            t_first = None
                            token_chunks = 0
                            continue
                        self._error(503, f"Qualcomm GenieX did not answer and respawn failed: {respawn_err}")
                        break
                    if not headers_sent:
                        self._error(502, f"Qualcomm GenieX did not answer: {exc}")
                    break
            self.close_connection = True

    return Handler


def exit_with_parent(parent_pid: int, serving: Serving) -> None:
    """Stop geniex serve and exit if Unsloth Studio dies without stopping us."""
    if sys.platform != "win32" or parent_pid <= 0:
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE

    def wait() -> None:
        if handle:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        serving.stop()
        os._exit(0)

    threading.Thread(target=wait, daemon=True, name="parent-watch").start()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args, _ = parser.parse_known_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    geniex = GenieX(args.cache_dir)
    serving = Serving()
    exit_with_parent(os.getppid(), serving)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(geniex, serving, os.environ.get("LEMONADE_API_KEY", "")),
    )
    server.daemon_threads = True
    log(f"Qualcomm GenieX NPU host on 127.0.0.1:{args.port} using {geniex.binary}")
    try:
        server.serve_forever()
    finally:
        serving.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
