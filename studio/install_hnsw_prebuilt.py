#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installs the prebuilt usearch + numkong pair (RAG's HNSW tier) into a Studio interpreter; wheels are
sha256-checked against the in-tree hnsw_prebuilt_pins.json, whichever host served them; the pair is probed
in a child interpreter before it counts. Never builds from source. Exit 0 for every outcome (2 bad
arguments; 1 with --strict unless the pair is ready); --json prints one status object last."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PAIR = (("usearch", "2.26.6"), ("numkong", "7.8.6"))
PAIR_STR = " ".join(f"{n}=={v}" for n, v in PAIR)
CATALOG = Path(__file__).resolve().with_name("hnsw_prebuilt_pins.json")
MARKER = ".unsloth-hnsw.json"
CACHE = Path(".unsloth-hnsw") / "wheels"
OK_STATUSES = ("ready", "already")
_FALSY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}
_NET_TIMEOUT = 30.0
_PROBE_TIMEOUT = 60.0
_LOCK_STALE_S = 1800.0
# Same allowlist as backend/utils/s3_runtime.py: TLS / proxy, never credentials or index overrides.
_CHILD_ENV_KEYS = {
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR",
    "APPDATA", "LOCALAPPDATA", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy",
    "http_proxy", "all_proxy", "no_proxy", "PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY",
    "UV_NATIVE_TLS", "UV_SYSTEM_CERTS", "UV_CACHE_DIR",
}  # fmt: skip

# Run in the TARGET interpreter: its tags, prefix and fingerprint, never this process's.
_INFO_SCRIPT = r"""
import hashlib, json, platform, sys
from importlib import metadata
def ver(n):
    try:
        return metadata.version(n)
    except Exception:
        return None
def manual_tags():
    # uv venvs ship neither pip nor packaging: CPython tags only, which is all the catalog holds.
    import glob, os, re, sysconfig
    machine = platform.machine().lower()
    plats = []
    if sys.platform == "win32":
        plats = [sysconfig.get_platform().replace("-", "_").replace(".", "_")]
    elif sys.platform == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        rel = [int(p) for p in (platform.mac_ver()[0] or "11.0").split(".")[:2]] + [0]
        vers = [(m, 0) for m in range(max(rel[0], 11), 10, -1)]
        if arch == "x86_64":
            vers += [(10, m) for m in range(15, 8, -1)]
        for a, b in vers:
            plats += [f"macosx_{a}_{b}_{arch}", f"macosx_{a}_{b}_universal2"]
    else:
        arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
        lib, ver = platform.libc_ver()
        if lib == "glibc" and ver:
            minor = int(ver.split(".")[1])
            plats += [f"manylinux_2_{m}_{arch}" for m in range(minor, 16, -1)]
            plats += [f"manylinux2014_{arch}"] if minor >= 17 else []
        elif glob.glob("/lib/ld-musl-*"):
            plats += [f"musllinux_1_{m}_{arch}" for m in (2, 1)]
        plats.append(f"linux_{arch}")
    py = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    return [f"{py}-{abi}-{p}" for abi in (py, "abi3", "none") for p in plats]
tags = []
for mod in ("packaging.tags", "pip._vendor.packaging.tags"):
    try:
        t = __import__(mod, fromlist = ["sys_tags"])
        tags = [str(x) for x in t.sys_tags()]
        break
    except Exception:
        continue
if not tags:
    tags = manual_tags()
print(json.dumps({
    "tags": tags,
    "prefix": sys.prefix,
    "version": list(sys.version_info[:2]),
    "fp": hashlib.sha1(f"{sys.executable}|{sys.version}|{platform.machine()}".encode()).hexdigest()[:16],
    "platform": sys.platform,
    "machine": platform.machine(),
    "libc": list(platform.libc_ver()),
    "installed": {n: ver(n) for n in ("usearch", "numkong", "numpy", "tqdm")},
}))
"""

_PROBE_SCRIPT = r"""
import os, sys, tempfile
if len(sys.argv) > 2 and sys.argv[2]:
    sys.path.insert(0, sys.argv[2])
want = dict(p.split("==") for p in sys.argv[1].split())
import numpy as np
import numkong
from usearch.index import Index
from importlib import metadata
for name, version in want.items():
    got = metadata.version(name)
    assert got == version, f"{name} {got} != {version}"
rows = np.random.default_rng(0).standard_normal((64, 16)).astype(np.float32)
index = Index(ndim = 16, metric = "cos", dtype = "f16", connectivity = 16, expansion_add = 128)
index.add(np.arange(64, dtype = np.uint64), rows)
assert all(int(index.search(rows[i], 1).keys[0]) == i for i in range(0, 64, 7))
d = tempfile.mkdtemp()
path = os.path.join(d, "probe.usearch")
index.save(path)
view = Index.restore(path, view = True)
assert len(view) == 64 and int(view.search(rows[5], 1).keys[0]) == 5
del view, index
print("HNSW_PROBE_OK")
"""


def _env_flag(name: str, values: set[str]) -> bool:
    return os.environ.get(name, "").strip().lower() in values


def _disabled() -> bool:
    return _env_flag("RAG_VEC_ANN_AUTO_INSTALL", _FALSY) or _env_flag("RAG_VEC_ANN", _FALSY)


def _offline() -> bool:
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "UNSLOTH_OFFLINE",
        "UV_OFFLINE",
        "PIP_NO_INDEX",
    )
    return any(_env_flag(k, _TRUTHY) for k in keys)


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _CHILD_ENV_KEYS}
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def _hidden() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def _result(
    status: str,
    detail: str = "",
    info: dict | None = None,
    source: str | None = None,
) -> dict:
    installed = (info or {}).get("installed") or {}
    return {
        "status": status,
        "detail": detail,
        "usearch": installed.get("usearch"),
        "numkong": installed.get("numkong"),
        "source": source,
    }


def load_catalog(path: Path = CATALOG) -> dict:
    with open(path, encoding = "utf-8") as fh:
        return json.load(fh)


def wheel_tags(filename: str) -> set[str]:
    """PEP 425: each of python / abi / platform may be a "."-separated set."""
    py, abi, plat = filename[:-4].split("-")[-3:]
    return {f"{a}-{b}-{c}" for a in py.split(".") for b in abi.split(".") for c in plat.split(".")}


def select(tags: list[str], catalog: dict) -> dict[str, dict | None]:
    """{package: catalog wheel} by the interpreter's own tag preference; None where nothing fits."""
    out: dict[str, dict | None] = {}
    for name, pkg in catalog["packages"].items():
        by_tag: dict[str, dict] = {}
        for wheel in pkg["wheels"]:
            for tag in wheel_tags(wheel["filename"]):
                by_tag.setdefault(tag, wheel)
        out[name] = next((by_tag[t] for t in tags if t in by_tag), None)
    return out


def interpreter_info(python: str, timeout: float = 60.0) -> dict:
    proc = subprocess.run(
        [python, "-I", "-c", _INFO_SCRIPT],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = _child_env(),
        **_hidden(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"interpreter probe exited {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _marker_path(prefix: str) -> Path:
    return Path(prefix) / MARKER


def read_marker(prefix: str) -> dict | None:
    try:
        with open(_marker_path(prefix), encoding = "utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_marker(
    prefix: str,
    fp: str,
    ok: bool,
    reason: str = "",
) -> None:
    path = _marker_path(prefix)
    tmp = path.with_name(path.name + ".tmp")
    data = {
        "pair": PAIR_STR,
        "fingerprint": fp,
        "result": "ok" if ok else "failed",
        "reason": reason[:500],
        "time": time.time(),
    }
    try:
        with open(tmp, "w", encoding = "utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except OSError:
        pass


# A failure (WDAC block, missing VC++ runtime, probe timeout) is retried after this, or with --force.
FAILED_RETRY_S = 7 * 86400


def _marker_says(marker: dict | None, fp: str) -> str | None:
    if marker and marker.get("pair") == PAIR_STR and marker.get("fingerprint") == fp:
        return marker.get("result")
    return None


def _failure_is_fresh(marker: dict | None) -> bool:
    try:
        return time.time() - float((marker or {}).get("time", 0)) < FAILED_RETRY_S
    except (TypeError, ValueError):
        return False


def _verified(path: Path, wheel: dict) -> bool:
    try:
        if path.stat().st_size != wheel["size"]:
            return False
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest() == wheel["sha256"]
    except OSError:
        return False


def _sources(wheel: dict, catalog: dict) -> list[tuple[str, str]]:
    mirror = os.environ.get("UNSLOTH_HNSW_MIRROR", "").strip() or catalog.get("mirror", "")
    out = []
    if mirror:
        out.append(("mirror", mirror.rstrip("/") + "/" + wheel["filename"]))
    if not _env_flag("UNSLOTH_HNSW_PYPI", _FALSY):
        out.append(("pypi", wheel["url"]))
    return out


def _fetch(url: str, dest: Path, deadline: float) -> None:
    req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio-hnsw"})
    timeout = max(1.0, min(_NET_TIMEOUT, deadline - time.monotonic()))
    with urllib.request.urlopen(req, timeout = timeout) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 16):
            if time.monotonic() > deadline:
                raise TimeoutError("download exceeded its time budget")
            fh.write(chunk)


def obtain(
    wheel: dict, catalog: dict, cache: Path, deadline: float, network: bool
) -> tuple[Path, str]:
    """A verified local copy of ``wheel``: the cache first, then each allowed source. Bad bytes are deleted."""
    cache.mkdir(parents = True, exist_ok = True)
    path = cache / wheel["filename"]
    if path.exists():
        if _verified(path, wheel):
            return path, "cache"
        path.unlink(missing_ok = True)
    if not network:
        raise FileNotFoundError(f"{wheel['filename']} is not cached and downloads are off")
    errors = []
    for source, url in _sources(wheel, catalog):
        part = path.with_name(path.name + ".part")
        try:
            _fetch(url, part, deadline)
            if not _verified(part, wheel):
                raise ValueError("size or sha256 does not match the pinned catalog")
            os.replace(part, path)
            return path, source
        except Exception as exc:  # noqa: BLE001 - next source
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
        finally:
            part.unlink(missing_ok = True)
    raise RuntimeError(f"{wheel['filename']}: " + "; ".join(errors or ["no source allowed"]))


def _uv() -> str | None:
    name = "uv.exe" if os.name == "nt" else "uv"
    found = shutil.which("uv")
    if found:
        return found
    for cand in (
        Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / name,
        Path.home() / ".local" / "bin" / name,
        Path.home() / ".cargo" / "bin" / name,
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _pip_cmd(python: str, verb: str, args: list[str]) -> list[list[str]]:
    """uv first, then the interpreter's own pip."""
    cmds = []
    uv = _uv()
    if uv:
        extra = ["--no-deps", "--no-index", "--no-config"] if verb == "install" else ["--no-config"]
        cmds.append([uv, "pip", verb, "--python", python, *extra, *args])
    if verb == "install":
        cmds.append(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--no-user",
                "--disable-pip-version-check",
                *args,
            ]
        )
    else:
        cmds.append([python, "-m", "pip", "uninstall", "-y", "--disable-pip-version-check", *args])
    return cmds


def _run_first(cmds: list[list[str]], timeout: float) -> None:
    last = ""
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd, capture_output = True, text = True, timeout = timeout, env = _child_env(), **_hidden()
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last = f"{type(exc).__name__}"
            continue
        if proc.returncode == 0:
            return
        last = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
        last = last[0][:300]
    raise RuntimeError(last or "no installer available")


def probe(
    python: str,
    extra_path: str = "",
    timeout: float = _PROBE_TIMEOUT,
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [python, "-I", "-c", _PROBE_SCRIPT, PAIR_STR, extra_path],
            capture_output = True, text = True, timeout = timeout, env = _child_env(), **_hidden(),
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return False, "probe timed out"
    except OSError as exc:
        return False, f"probe could not start: {exc}"
    if proc.returncode == 0 and "HNSW_PROBE_OK" in proc.stdout:
        return True, ""
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
    return False, (tail[0] if tail else f"probe exited {proc.returncode}")[:300]


class _Lock:
    """Cross-process lock in the target prefix; a holder older than _LOCK_STALE_S is presumed dead."""

    def __init__(self, path: Path, timeout: float):
        self.path, self.timeout, self.fd = path, timeout, None

    def __enter__(self):
        end = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > _LOCK_STALE_S:
                        self.path.unlink(missing_ok = True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > end:
                    raise TimeoutError("another HNSW install holds the lock")
                time.sleep(0.5)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok = True)


def _pair_installed(info: dict) -> bool:
    return all(info["installed"].get(n) == v for n, v in PAIR)


def ensure(
    python: str = sys.executable,
    *,
    check_only: bool = False,
    timeout: float = 600.0,
    force: bool = False,
) -> dict:
    """Make the pinned pair importable in ``python``; returns {status, detail, usearch, numkong, source}."""
    deadline = time.monotonic() + timeout
    try:
        info = interpreter_info(python)
    except Exception as exc:  # noqa: BLE001
        return _result("failed", f"could not inspect {python}: {exc}")
    if tuple(info["version"]) < (3, 11):
        return _result("unsupported", "usearch needs Python 3.11 or newer", info)
    prefix, fp = info["prefix"], info["fp"]
    marker = read_marker(prefix)
    said = _marker_says(marker, fp)
    if said == "failed" and (check_only or (_failure_is_fresh(marker) and not force)):
        return _result("failed", (marker or {}).get("reason", ""), info)
    if _pair_installed(info):
        if said == "ok" or check_only:
            return _result("already", "" if said == "ok" else "not probed", info)
        ok, reason = probe(python)
        _write_marker(prefix, fp, ok, reason)
        return _result("already" if ok else "failed", reason, info)
    if _disabled():
        return _result("disabled", "RAG_VEC_ANN_AUTO_INSTALL=0 or RAG_VEC_ANN=0", info)
    if check_only:
        return _result("missing", "pair not installed", info)
    if not (info["installed"].get("numpy") and info["installed"].get("tqdm")):
        return _result("unsupported", "numpy and tqdm must already be installed", info)
    try:
        catalog = load_catalog()
    except (OSError, ValueError) as exc:
        return _result("failed", f"catalog unreadable: {exc}", info)
    chosen = select(info["tags"], catalog)
    missing = [n for n, w in chosen.items() if w is None]
    if missing:
        top = info["tags"][0] if info["tags"] else "no tags"
        return _result("unsupported", f"no prebuilt {', '.join(missing)} for {top}", info)
    network = not _offline()
    cache = Path(prefix) / CACHE
    if not network and not all(_verified(cache / w["filename"], w) for w in chosen.values()):
        return _result("offline", "offline and the wheels are not cached", info)
    try:
        lock = _Lock(
            Path(prefix) / ".unsloth-hnsw.lock", max(1.0, min(60.0, deadline - time.monotonic()))
        )
        with lock:
            info = interpreter_info(python)
            if _pair_installed(info) and _marker_says(read_marker(prefix), fp) == "ok":
                return _result("already", "", info)
            files, sources = [], set()
            for wheel in chosen.values():
                path, source = obtain(wheel, catalog, cache, deadline, network)
                files.append(str(path))
                sources.add(source)
            source = "+".join(sorted(sources))
            remaining = max(30.0, deadline - time.monotonic())
            _run_first(_pip_cmd(python, "install", files), remaining)
            ok, reason = probe(python)
            if not ok:
                try:
                    _run_first(_pip_cmd(python, "uninstall", [n for n, _ in PAIR]), 120.0)
                except Exception:  # noqa: BLE001 - the failed marker still keeps Studio off it
                    pass
            _write_marker(prefix, fp, ok, reason)
            info = interpreter_info(python)
            return _result("ready" if ok else "failed", reason, info, source)
    except TimeoutError as exc:
        status = "busy" if "lock" in str(exc) else "failed"
        return _result(status, str(exc), info)
    except Exception as exc:  # noqa: BLE001 - optional component: report, never raise
        return _result("failed", f"{type(exc).__name__}: {exc}"[:500], info)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--python", default = sys.executable, help = "interpreter to install into")
    ap.add_argument("--json", action = "store_true", help = "print one JSON status object last")
    ap.add_argument("--strict", action = "store_true", help = "exit 1 unless the pair is ready")
    ap.add_argument("--check-only", action = "store_true", help = "no download, no install")
    ap.add_argument("--timeout", type = float, default = 600.0)
    ap.add_argument("--force", action = "store_true", help = "retry even after a recent failure")
    args = ap.parse_args(argv)
    result = ensure(args.python, check_only = args.check_only, timeout = args.timeout, force = args.force)
    if args.json:
        print(json.dumps(result), flush = True)
    else:
        print(
            f"hnsw: {result['status']}" + (f" ({result['detail']})" if result["detail"] else ""),
            flush = True,
        )
    if args.strict and result["status"] not in OK_STATUSES:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
