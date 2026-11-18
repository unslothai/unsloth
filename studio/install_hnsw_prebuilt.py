#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installs the hash-pinned usearch + numkong pair (hnsw_prebuilt.txt), probed in a child interpreter.
Exit 0 for every outcome (2 bad arguments; 1 with --strict unless ready); --json prints one status object last."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PAIR = (("usearch", "2.26.6"), ("numkong", "7.8.6"))
PAIR_STR = " ".join(f"{n}=={v}" for n, v in PAIR)
REQUIREMENTS = Path(__file__).resolve().with_name("hnsw_prebuilt.txt")
MARKER = ".unsloth-hnsw.json"
OK_STATUSES = ("ready", "already")
_FALSY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}
_PROBE_TIMEOUT = 60.0
_LOCK_STALE_S = 1800.0
# Held from install to marker; the backend will not import the pair while it is live.
LOCK = ".unsloth-hnsw.lock"
# A failure (WDAC block, missing VC++ runtime, probe timeout) is retried after this, or with --force.
FAILED_RETRY_S = 7 * 86400
# TLS / proxy / index only: a user's UV_CONSTRAINT or uv.toml would break --require-hashes.
_CHILD_ENV_KEYS = {
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR",
    "APPDATA", "LOCALAPPDATA", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy",
    "http_proxy", "all_proxy", "no_proxy", "PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY",
    "PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_NATIVE_TLS", "UV_SYSTEM_CERTS",
}  # fmt: skip

# Run in the TARGET interpreter: its prefix, fingerprint and packages, never this process's.
_INFO_SCRIPT = r"""
import hashlib, json, platform, sys, sysconfig
from importlib import metadata
def dist(n):
    try:
        d = metadata.distribution(n)
        return d.version, str(d.locate_file(""))
    except Exception:
        return None, None
pkgs = {n: dist(n) for n in ("usearch", "numkong", "numpy", "tqdm")}
print(json.dumps({
    "prefix": sys.prefix,
    "version": list(sys.version_info[:2]),
    "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    "fp": hashlib.sha1(f"{sys.prefix}|{sys.version}|{platform.machine()}".encode()).hexdigest()[:16],
    "installed": {n: v for n, (v, _) in pkgs.items()},
    "sites": {n: s for n, (_, s) in pkgs.items() if s},
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
# One thread, as vec_cache builds: parallel inserts can strand a node, which would fail a working pair.
index.add(np.arange(64, dtype = np.uint64), rows, threads = 1)
missed = [i for i in range(0, 64, 7) if i not in index.search(rows[i], 4).keys.tolist()]
assert not missed, f"self search missed rows {missed}"
with tempfile.TemporaryDirectory(ignore_cleanup_errors = True) as d:
    path = os.path.join(d, "probe.usearch")
    index.save(path)
    view = Index.restore(path, view = True)
    assert len(view) == 64, f"restored {len(view)} of 64 rows"
    assert 5 in view.search(rows[5], 4).keys.tolist(), "restored index lost row 5"
    del view, index
print("HNSW_PROBE_OK")
"""


def _env_flag(name: str, values: set[str]) -> bool:
    return os.environ.get(name, "").strip().lower() in values


def _disabled() -> bool:
    # RAG_VEC_ANN keeps vec_cache's own reading: only "0" turns the tier off.
    return (
        _env_flag("RAG_VEC_ANN_AUTO_INSTALL", _FALSY)
        or os.environ.get("RAG_VEC_ANN", "").strip() == "0"
    )


def _offline() -> bool:
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "UNSLOTH_OFFLINE",
        "UV_OFFLINE",
        "UV_NO_INDEX",
        "PIP_NO_INDEX",
    )
    return any(_env_flag(k, _TRUTHY) for k in keys)  # fmt: skip


def _native_tls() -> bool:
    # The OS trust store where Studio's other downloads use it (native_tls.py): macOS, Windows, Desktop.
    flag = os.environ.get("UNSLOTH_STUDIO_NATIVE_TLS", "").strip().lower()
    if flag in ("1", "true", "yes") or flag in ("0", "false", "no"):
        return flag in ("1", "true", "yes")
    owned = os.environ.get("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", "") == "tauri"
    return sys.platform in ("darwin", "win32") or (sys.platform.startswith("linux") and owned)


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _CHILD_ENV_KEYS}
    if _native_tls():
        env.setdefault("UV_NATIVE_TLS", "1")
    env["PIP_CONFIG_FILE"] = os.devnull
    # Child output is decoded as UTF-8; make the Python children write it.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _hidden() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = timeout, env = _child_env(), **_hidden(),
    )  # fmt: skip


def _result(
    status: str,
    detail: str = "",
    info: dict | None = None,
) -> dict:
    installed = (info or {}).get("installed") or {}
    return {
        "status": status,
        "detail": detail,
        "usearch": installed.get("usearch"),
        "numkong": installed.get("numkong"),
    }


def interpreter_info(python: str, timeout: float = 60.0) -> dict:
    proc = _run([python, "-I", "-c", _INFO_SCRIPT], timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"interpreter probe exited {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def requirement_blocks(path: Path = REQUIREMENTS) -> dict[str, str]:
    """{package: its "name==version --hash=..." block} from the hash-pinned requirements file."""
    blocks, name = {}, None
    for line in path.read_text(encoding = "utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            name = line.split("==")[0].strip().lower()
            blocks[name] = ""
        blocks[name] += line + "\n"
    return blocks


def read_marker(prefix: str) -> dict | None:
    try:
        with open(Path(prefix) / MARKER, encoding = "utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_marker(
    prefix: str,
    fp: str,
    ok: bool | None,
    reason: str = "",
) -> None:
    path = Path(prefix) / MARKER
    tmp = path.with_name(path.name + ".tmp")
    data = {
        "pair": PAIR_STR,
        "fingerprint": fp,
        # "unverified": the probe hung or could not run; the backend will not import until one passes.
        "result": "unverified" if ok is None else ("ok" if ok else "failed"),
        "reason": reason[:500],
        "time": time.time(),
    }
    try:
        with open(tmp, "w", encoding = "utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _marker_says(marker: dict | None, fp: str) -> str | None:
    if marker and marker.get("pair") == PAIR_STR and marker.get("fingerprint") == fp:
        return marker.get("result")
    return None


def _failure_is_fresh(marker: dict | None) -> bool:
    try:
        return time.time() - float((marker or {}).get("time", 0)) < FAILED_RETRY_S
    except (TypeError, ValueError):
        return False


_SHF_ALLOC = 0x2
_SHF_INFO_LINK = 0x40
_SHT_NOBITS, _SHT_REL, _SHT_RELA = 8, 9, 4


def strip_elf_debug(data: bytes) -> bytes | None:
    """``data`` without its .debug_* sections (the PyPI Linux builds ship full DWARF: ~90% of numkong).

    Each run of debug bytes is cut by a whole multiple of the largest segment alignment and later
    bytes move down by that amount, so every segment keeps its offset-to-address congruence and no
    loaded byte changes. None when this is not a little-endian ELF64 it can rewrite safely."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    phoff, shoff = struct.unpack_from("<QQ", data, 0x20)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", data, 0x36)
    if shentsize != 64 or phentsize != 56 or not shnum or shstrndx >= shnum:
        return None
    if shoff + shnum * 64 > len(data) or phoff + phnum * 56 > len(data):
        return None
    secs = [list(struct.unpack_from("<IIQQQQIIQQ", data, shoff + k * 64)) for k in range(shnum)]
    phdrs = [list(struct.unpack_from("<IIQQQQQQ", data, phoff + k * 56)) for k in range(phnum)]
    names = secs[shstrndx]

    def name(sec) -> bytes:
        start = names[4] + sec[0]
        return data[start : data.index(b"\0", start)]

    drop = {
        k
        for k, sec in enumerate(secs)
        if k
        and not sec[2] & _SHF_ALLOC
        and sec[1] != _SHT_NOBITS
        and name(sec).startswith((b".debug", b".zdebug"))
    }
    for k, sec in enumerate(secs):  # relocations for a dropped section go with it
        if k and sec[1] in (_SHT_REL, _SHT_RELA) and not sec[2] & _SHF_ALLOC and sec[7] in drop:
            drop.add(k)
    if not drop or shstrndx in drop:
        return None
    if any(secs[k][6] in drop for k in range(shnum) if k not in drop):
        return None  # a kept section links into the debug info
    # Contiguous runs of dropped bytes; nothing kept (segment, section, headers) may sit inside one.
    spans = sorted((secs[k][4], secs[k][4] + secs[k][5]) for k in drop)
    runs = []
    for a, b in spans:
        if runs and a <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    kept = [(ph[2], ph[2] + ph[5]) for ph in phdrs if ph[5]]
    kept += [
        (sec[4], sec[4] + sec[5])
        for k, sec in enumerate(secs)
        if k and k not in drop and sec[1] != _SHT_NOBITS and sec[5]
    ]
    kept += [(0, 64), (phoff, phoff + phnum * 56)]
    if any(a < rb and ra < b for a, b in kept for ra, rb in runs):
        return None
    align = max([ph[7] for ph in phdrs if ph[0] == 1 and ph[7]] or [1])
    if align & (align - 1) or align > 1 << 21:
        return None
    cuts = []  # (offset, bytes removed there)
    for a, b in runs:
        n = (b - a) // align * align
        if n:
            cuts.append((a, n))
    old_table = (shoff, shoff + shnum * 64)
    if not cuts and old_table[1] != len(data):
        return None

    def moved(off: int) -> int:
        return off - sum(n for at, n in cuts if at + n <= off)

    out = bytearray()
    pos = 0
    for at, n in cuts:
        out += data[pos:at]
        pos = at + n
    end = len(data)
    if old_table[1] == end and all(old_table[0] >= at + n for at, n in cuts):
        end = old_table[0]  # the old section table is the file's tail: rewrite it in place
    out += data[pos:end]
    for k, ph in enumerate(phdrs):
        ph[2] = moved(ph[2])
        struct.pack_into("<IIQQQQQQ", out, moved(phoff) + k * 56, *ph)
    keep = [k for k in range(shnum) if k not in drop]
    index = {old: new for new, old in enumerate(keep)}
    out += b"\0" * (-len(out) % 8)
    new_shoff = len(out)
    for k in keep:
        sec = list(secs[k])
        if k:
            sec[4] = moved(sec[4])
        sec[6] = index.get(sec[6], 0)
        if sec[1] in (_SHT_REL, _SHT_RELA) or sec[2] & _SHF_INFO_LINK:
            sec[7] = index.get(sec[7], 0)
        out += struct.pack("<IIQQQQIIQQ", *sec)
    struct.pack_into("<QQ", out, 0x20, moved(phoff), new_shoff)
    struct.pack_into("<HH", out, 0x3C, len(keep), index[shstrndx])
    return bytes(out) if len(out) < len(data) else None


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _replace(path: Path, data: bytes) -> None:
    # A new inode: uv may hardlink installed files to its cache, which must not change.
    tmp = path.with_name(path.name + ".slim")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def slim_installed(site: str, name: str) -> int:
    """Strip the debug info from ``name``'s installed ELF libraries under ``site`` and keep its RECORD
    true, so uninstall and pip check still work. Returns bytes saved; macOS / Windows builds save 0."""
    saved = 0
    for record in Path(site).glob(f"{name}-*.dist-info/RECORD"):
        lines = record.read_text(encoding = "utf-8").splitlines()
        for i, line in enumerate(lines):
            rel = line.rsplit(",", 2)[0]
            path = Path(site) / rel
            try:
                with open(path, "rb") as fh:
                    if fh.read(4) != b"\x7fELF":
                        continue
                data = path.read_bytes()
            except OSError:
                continue
            slim = strip_elf_debug(data)
            if slim is None:
                continue
            _replace(path, slim)
            lines[i] = f"{rel},{_record_hash(slim)},{len(slim)}"
            saved += len(data) - len(slim)
        if saved:
            _replace(record, ("\n".join(lines) + "\n").encode("utf-8"))
    return saved


def _uv(prefix: str | None = None) -> str | None:
    # Studio venvs are made by uv without pip, and a Desktop launch can have a minimal PATH: also look
    # in Studio's private uv dir (<studio home>/uv, the venv's parent by default) and the installers' dirs.
    name = "uv.exe" if os.name == "nt" else "uv"
    found = shutil.which("uv")
    if found:
        return found
    homes = [os.environ.get(k, "").strip() for k in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME")]
    homes += [str(Path(p).parent) for p in (prefix, sys.prefix) if p]
    cands = [Path(h) / "uv" / name for h in homes if h]
    cands += [
        Path(p) / ("Scripts" if os.name == "nt" else "bin") / name
        for p in (prefix, sys.prefix)
        if p
    ]
    cands += [
        Path(d) / name
        for d in (os.environ.get("UV_INSTALL_DIR"), os.environ.get("XDG_BIN_HOME"))
        if d
    ]
    cands += [Path.home() / ".local" / "bin" / name, Path.home() / ".cargo" / "bin" / name]
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cands.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / name)
    cands += [Path("/opt/homebrew/bin/uv"), Path("/usr/local/bin/uv")]
    for cand in cands:
        try:
            if cand.is_file() and (os.name == "nt" or os.access(cand, os.X_OK)):
                return str(cand)
        except OSError:
            continue
    return None  # fmt: skip


def _pip_cmd(
    python: str,
    verb: str,
    args: list[str],
    prefix: str | None = None,
) -> list[list[str]]:
    """uv first (Studio venvs have no pip), then pip. Installs take only hash-pinned wheels."""
    pinned = ["--no-deps", "--only-binary", ":all:", "--require-hashes"]
    cmds = []
    uv = _uv(prefix)
    if uv:
        extra = [*pinned, "--no-config", "--no-cache"] if verb == "install" else ["--no-config"]
        cmds.append([uv, "pip", verb, "--python", python, *extra, *args])
    if verb == "install":
        cmds.append(
            [
                python,
                "-m",
                "pip",
                "install",
                *pinned,
                "--no-user",
                "--no-cache-dir",
                "--disable-pip-version-check",
                *args,
            ]
        )
    else:
        cmds.append([python, "-m", "pip", "uninstall", "-y", "--disable-pip-version-check", *args])
    return cmds  # fmt: skip


def _run_first(cmds: list[list[str]], timeout: float) -> None:
    errors = []
    for cmd in cmds:
        try:
            proc = _run(cmd, timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{os.path.basename(cmd[0])}: {type(exc).__name__}")
            continue
        if proc.returncode == 0:
            return
        out = (proc.stderr or proc.stdout).strip().splitlines()
        tail = next(
            (l for l in out if "no usable wheels" in l or "No matching distribution" in l), None
        )
        errors.append(
            f"{os.path.basename(cmd[0])}: {(tail or (out[-1] if out else f'exit {proc.returncode}'))[:300]}"
        )
    raise RuntimeError("; ".join(errors) or "no installer available")  # fmt: skip


def probe(
    python: str,
    extra_path: str = "",
    timeout: float = _PROBE_TIMEOUT,
    want: str = PAIR_STR,
) -> tuple[bool | None, str]:
    """(True, "") works, (False, why) broken, (None, why) inconclusive. A failure is confirmed by a rerun."""
    ok, why = _probe_once(python, extra_path, timeout, want)
    if ok is False:
        again, why_again = _probe_once(python, extra_path, timeout, want)
        if again is not False:
            return again, why_again
    return ok, why


def _probe_once(python: str, extra_path: str, timeout: float, want: str) -> tuple[bool | None, str]:
    try:
        proc = _run([python, "-I", "-c", _PROBE_SCRIPT, want, extra_path], timeout)
    except subprocess.TimeoutExpired:
        return None, "probe timed out"
    except OSError as exc:
        return None, f"probe could not start: {exc}"
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
    # Any installed pair is left alone and only probed: older Studio installs resolved numkong past the
    # pin, and a user's own usearch must not be replaced or, on a failed probe, uninstalled.
    return all(info["installed"].get(n) for n, _ in PAIR)


def _want(info: dict) -> str:
    return PAIR_STR if all(info["installed"].get(n) == v for n, v in PAIR) else ""


def _install(python: str, info: dict, timeout: float) -> dict:
    prefix, fp = info["prefix"], info["fp"]
    # Half a pair already here (the user's numkong, say): install only the other half, and never
    # replace or, on a failed probe, uninstall what this run did not put there.
    present = {n for n, _ in PAIR if info["installed"].get(n)}
    ours = [n for n, _ in PAIR if n not in present]
    blocks = requirement_blocks()
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "hnsw.txt"
        req.write_text("".join(blocks[n] for n in ours), encoding = "utf-8")
        # Until a probe passes, files from a failed, killed or timed-out install stay unimportable.
        _write_marker(prefix, fp, None, "install not finished")
        try:
            _run_first(_pip_cmd(python, "install", ["-r", str(req)], prefix), timeout)
        except Exception as exc:  # noqa: BLE001
            after = interpreter_info(python)
            reason = f"{type(exc).__name__}: {exc}"[:500]
            if not any(after["installed"].get(n) for n in ours):
                # Nothing landed: no wheel for this platform, or no network. Back off, do not retry hourly.
                _write_marker(prefix, fp, False, reason)
                status = (
                    "unsupported"
                    if "usable wheels" in reason or "No matching distribution" in reason
                    else "failed"
                )
                return _result(status, reason, after)  # fmt: skip
            return _result("failed", reason, after)
    after = interpreter_info(python)
    for name in ours:
        if after["sites"].get(name):
            slim_installed(after["sites"][name], name)
    ok, reason = probe(python, want = _want(after) if present else PAIR_STR)
    if ok is False:
        try:
            _run_first(_pip_cmd(python, "uninstall", ours, prefix), 120.0)
        except Exception:  # noqa: BLE001 - the failed marker still keeps Studio off it
            pass
    _write_marker(prefix, fp, ok, reason)
    return _result("ready" if ok else "failed", reason, interpreter_info(python))


def ensure(
    python: str = sys.executable,
    *,
    check_only: bool = False,
    timeout: float = 600.0,
    force: bool = False,
) -> dict:
    """Make the pinned pair importable in ``python``; returns {status, detail, usearch, numkong}."""
    deadline = time.monotonic() + timeout
    try:
        info = interpreter_info(python)
    except Exception as exc:  # noqa: BLE001
        return _result("failed", f"could not inspect {python}: {exc}")
    if tuple(info["version"]) < (3, 11) or info.get("free_threaded"):
        return _result("unsupported", "usearch needs a regular (GIL) CPython 3.11 or newer", info)
    prefix, fp = info["prefix"], info["fp"]
    marker = read_marker(prefix)
    said = _marker_says(marker, fp)
    if said == "failed" and (check_only or (_failure_is_fresh(marker) and not force)):
        return _result("failed", (marker or {}).get("reason", ""), info)
    if _pair_installed(info):
        if said == "ok":
            return _result("already", "", info)
        if check_only:
            if said == "unverified":
                return _result("missing", "probe inconclusive", info)
            return _result("already", "not probed", info)
        ok, reason = probe(python, want = _want(info))
        _write_marker(prefix, fp, ok, reason)
        return _result("already" if ok else "failed", reason, info)
    if _disabled():
        return _result("disabled", "RAG_VEC_ANN_AUTO_INSTALL=0 or RAG_VEC_ANN=0", info)
    if not (info["installed"].get("numpy") and info["installed"].get("tqdm")):
        return _result("unsupported", "numpy and tqdm must already be installed", info)
    if check_only:
        return _result("missing", "pair not installed", info)
    if _offline():
        return _result("offline", "downloads are off", info)
    try:
        with _Lock(Path(prefix) / LOCK, max(1.0, min(60.0, deadline - time.monotonic()))):
            info = interpreter_info(python)
            if _pair_installed(info) and _marker_says(read_marker(prefix), fp) == "ok":
                return _result("already", "", info)
            return _install(python, info, max(30.0, deadline - time.monotonic()))
    except TimeoutError as exc:
        return _result("busy" if "lock" in str(exc) else "failed", str(exc), info)
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
        detail = f" ({result['detail']})" if result["detail"] else ""
        print(f"hnsw: {result['status']}{detail}", flush = True)
    if args.strict and result["status"] not in OK_STATUSES:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
