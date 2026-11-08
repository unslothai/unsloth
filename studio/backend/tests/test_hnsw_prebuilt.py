# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studio/install_hnsw_prebuilt.py: wheel selection, verified download, install / probe / marker, and one
real install of the pinned pair for this interpreter."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "install_hnsw_prebuilt.py"
_spec = importlib.util.spec_from_file_location("install_hnsw_prebuilt", _PATH)
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

CATALOG = hp.load_catalog()
PYS = ("cp311", "cp312", "cp313", "cp314")


def _tags(py, plats):
    return [f"{py}-{abi}-{p}" for abi in (py, "abi3", "none") for p in plats]


def _glibc(minor, arch):
    return [f"manylinux_2_{m}_{arch}" for m in range(minor, 16, -1)] + [f"linux_{arch}"]


def _mac(major, arch):
    vers = [(m, 0) for m in range(major, 10, -1)] + (
        [(10, m) for m in range(15, 8, -1)] if arch == "x86_64" else []
    )
    return [p for a, b in vers for p in (f"macosx_{a}_{b}_{arch}", f"macosx_{a}_{b}_universal2")]


TARGETS = {
    "linux-x86_64": _glibc(31, "x86_64"),
    "linux-aarch64": _glibc(35, "aarch64"),
    "macos-arm64": _mac(12, "arm64"),
    "macos-x86_64": _mac(12, "x86_64"),
    "win-amd64": ["win_amd64"],
    "win-arm64": ["win_arm64"],
}


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("py", PYS)
def test_every_supported_target_has_both_wheels(target, py):
    chosen = hp.select(_tags(py, TARGETS[target]), CATALOG)
    assert chosen["usearch"] and chosen["numkong"], (target, py, chosen)
    for wheel in chosen.values():
        assert py in wheel["filename"] and len(wheel["sha256"]) == 64 and wheel["size"] > 0


def test_selection_follows_the_interpreter():
    pick = lambda plats, py = "cp312": {
        n: w and w["filename"] for n, w in hp.select(_tags(py, plats), CATALOG).items()
    }
    assert pick(["win_arm64"])["usearch"].endswith("win_arm64.whl")
    assert pick(["win_amd64"])["numkong"].endswith("win_amd64.whl")
    assert (
        "musllinux_1_2_x86_64" in pick(["musllinux_1_2_x86_64", "musllinux_1_1_x86_64"])["numkong"]
    )
    assert pick(_mac(14, "arm64"))["numkong"].endswith("macosx_11_0_arm64.whl")
    # numkong is manylinux_2_28 only: glibc 2.27 gets usearch but no numkong, so the pair is unsupported.
    old = pick(_glibc(27, "x86_64"))
    assert old["usearch"] and old["numkong"] is None
    assert pick(["win_amd64"], py = "cp310") == {"usearch": None, "numkong": None}


def test_wheel_tags_expand_compressed_sets():
    assert hp.wheel_tags("x-1-cp312-cp312-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl") == {
        "cp312-cp312-manylinux_2_26_x86_64",
        "cp312-cp312-manylinux_2_28_x86_64",
    }


def _fake_wheel(tmp_path, body = b"wheel-bytes"):
    return {
        "filename": "pkg-1-cp312-cp312-win_amd64.whl",
        "sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
        "url": "https://pypi.example/pkg.whl",
    }


def _serve(monkeypatch, responses):
    calls = []

    def fetch(url, dest, deadline):
        calls.append(url)
        body = responses.get(url)
        if body is None:
            raise OSError("HTTP Error 404")
        Path(dest).write_bytes(body)

    monkeypatch.setattr(hp, "_fetch", fetch)
    return calls


def test_pypi_comes_first_and_the_mirror_serves_when_pypi_is_blocked(tmp_path, monkeypatch):
    wheel = _fake_wheel(tmp_path)
    mirrored = "https://mirror.example/" + wheel["filename"]
    calls = _serve(monkeypatch, {mirrored: b"wheel-bytes"})
    path, source = hp.obtain(wheel, {"mirror": "https://mirror.example/"}, tmp_path, 1e18, True)
    assert source == "mirror" and path.read_bytes() == b"wheel-bytes"
    assert calls == [wheel["url"], mirrored]
    assert (
        hp.obtain(wheel, {"mirror": "https://mirror.example/"}, tmp_path, 1e18, False)[1] == "cache"
    )


def test_an_explicit_mirror_is_tried_before_pypi(tmp_path, monkeypatch):
    wheel = _fake_wheel(tmp_path)
    monkeypatch.setenv("UNSLOTH_HNSW_MIRROR", "https://corp.example/hnsw")
    calls = _serve(monkeypatch, {wheel["url"]: b"wheel-bytes"})
    assert hp.obtain(wheel, CATALOG, tmp_path, 1e18, True)[1] == "pypi"
    assert calls == ["https://corp.example/hnsw/" + wheel["filename"], wheel["url"]]


def test_bytes_that_do_not_match_the_pin_are_rejected_and_deleted(tmp_path, monkeypatch):
    wheel = _fake_wheel(tmp_path)
    _serve(
        monkeypatch,
        {wheel["url"]: b"tampered!!!", "https://mirror.example/" + wheel["filename"]: b"short"},
    )
    with pytest.raises(RuntimeError, match = "sha256"):
        hp.obtain(wheel, {"mirror": "https://mirror.example/"}, tmp_path, 1e18, True)
    assert not list(tmp_path.iterdir())
    (tmp_path / wheel["filename"]).write_bytes(b"corrupt-cache")
    with pytest.raises(FileNotFoundError):
        hp.obtain(wheel, {}, tmp_path, 1e18, False)
    assert not (tmp_path / wheel["filename"]).exists()


def test_pypi_can_be_turned_off(tmp_path, monkeypatch):
    wheel = _fake_wheel(tmp_path)
    monkeypatch.setenv("UNSLOTH_HNSW_PYPI", "0")
    monkeypatch.setenv("UNSLOTH_HNSW_MIRROR", "https://corp.example/hnsw")
    calls = _serve(monkeypatch, {wheel["url"]: b"wheel-bytes"})
    with pytest.raises(RuntimeError):
        hp.obtain(wheel, CATALOG, tmp_path, 1e18, True)
    assert calls == ["https://corp.example/hnsw/" + wheel["filename"]]


def _info(
    prefix,
    installed = None,
    version = (3, 12),
    tags = None,
):
    pkgs = {"usearch": None, "numkong": None, "numpy": "2.0", "tqdm": "4.0"}
    pkgs.update(installed or {})
    return {
        "tags": tags if tags is not None else _tags("cp312", ["win_amd64"]),
        "prefix": str(prefix),
        "version": list(version),
        "fp": "fp1",
        "installed": pkgs,
    }


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """ensure() with the interpreter, downloads, installer and probe faked; records what ran."""
    for k in ("RAG_VEC_ANN_AUTO_INSTALL", "RAG_VEC_ANN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
              "UNSLOTH_OFFLINE", "UV_OFFLINE", "PIP_NO_INDEX"):  # fmt: skip
        monkeypatch.delenv(k, raising = False)
    state = {"info": _info(tmp_path), "probe": (True, ""), "ran": [], "net": [], "wants": []}

    def info(python, timeout = 60.0):
        return json.loads(json.dumps(state["info"]))

    def run_first(cmds, timeout):
        verb = cmds[0][2] if cmds[0][1] == "pip" else cmds[0][3]
        state["ran"].append(verb)
        if verb == "install":
            state["info"]["installed"].update({n: v for n, v in hp.PAIR})
        else:
            state["info"]["installed"].update({n: None for n, _ in hp.PAIR})

    def obtain(wheel, catalog, cache, deadline, network):
        state["net"].append(network)
        return Path(cache) / wheel["filename"], "pypi"

    monkeypatch.setattr(hp, "interpreter_info", info)
    monkeypatch.setattr(hp, "_run_first", run_first)
    monkeypatch.setattr(hp, "obtain", obtain)

    def probe(
        python,
        extra_path = "",
        timeout = 60.0,
        want = hp.PAIR_STR,
    ):
        state["wants"].append(want)
        return state["probe"]

    monkeypatch.setattr(hp, "probe", probe)
    return state


def test_a_fresh_install_probes_and_marks_ready(flow, tmp_path):
    r = hp.ensure("py")
    assert r["status"] == "ready" and r["usearch"] == "2.26.6" and r["source"] == "pypi"
    assert flow["ran"] == ["install"]
    assert hp.read_marker(str(tmp_path))["result"] == "ok"
    assert hp.ensure("py")["status"] == "already" and flow["ran"] == ["install"]


def test_a_failed_probe_uninstalls_and_marks_failed(flow, tmp_path):
    flow["probe"] = (False, "ImportError: DLL load failed")
    r = hp.ensure("py")
    assert r["status"] == "failed" and "DLL" in r["detail"]
    assert flow["ran"] == ["install", "uninstall"]
    marker = hp.read_marker(str(tmp_path))
    assert (
        marker["result"] == "failed"
        and marker["fingerprint"] == "fp1"
        and marker["pair"] == hp.PAIR_STR
    )
    assert hp.ensure("py", check_only = True)["status"] == "failed"


def test_an_inconclusive_probe_keeps_the_install_but_marks_it_unverified(flow, tmp_path):
    flow["probe"] = (None, "probe timed out")
    r = hp.ensure("py")
    assert r["status"] == "failed" and r["detail"] == "probe timed out"
    assert flow["ran"] == ["install"] and hp.read_marker(str(tmp_path))["result"] == "unverified"
    # No week-long backoff: setup sees it as owed, and the next run probes again.
    assert hp.ensure("py", check_only = True)["status"] == "missing"
    flow["probe"] = (True, "")
    assert hp.ensure("py")["status"] == "already" and flow["ran"] == ["install"]
    assert hp.read_marker(str(tmp_path))["result"] == "ok"


def test_an_install_that_dies_halfway_leaves_the_pair_unimportable(flow, tmp_path, monkeypatch):
    def run_first(cmds, timeout):
        # Both wheels' files and metadata landed, then the installer timed out.
        flow["info"]["installed"].update({n: v for n, v in hp.PAIR})
        raise TimeoutError("uv pip install timed out")

    monkeypatch.setattr(hp, "_run_first", run_first)
    r = hp.ensure("py")
    assert r["status"] == "failed" and not flow["wants"]
    # vec_ann.usearch() refuses an unverified marker (test_rag_hnsw_runtime).
    assert hp.read_marker(str(tmp_path))["result"] == "unverified"
    # Setup's fast path does not call it installed; the next full run probes it.
    assert hp.ensure("py", check_only = True)["status"] == "missing"
    assert hp.ensure("py")["status"] == "already" and flow["wants"]
    assert hp.read_marker(str(tmp_path))["result"] == "ok"


@pytest.mark.parametrize("numkong", ["7.8.7", "8.0.0"])
def test_a_pair_from_an_older_install_is_probed_not_replaced(flow, tmp_path, numkong):
    flow["info"] = _info(tmp_path, installed = {"usearch": "2.26.6", "numkong": numkong})
    assert hp.ensure("py", check_only = True)["status"] == "already"
    assert hp.ensure("py")["status"] == "already"
    assert flow["wants"] == [""] and not flow["ran"] and not flow["net"]
    assert hp.read_marker(str(tmp_path))["result"] == "ok"


def test_a_users_own_pair_is_never_uninstalled(flow, tmp_path):
    flow["info"] = _info(tmp_path, installed = {"usearch": "2.30.0", "numkong": "8.1.0"})
    flow["probe"] = (False, "AttributeError: restore")
    assert hp.ensure("py")["status"] == "failed"
    assert not flow["ran"] and hp.read_marker(str(tmp_path))["result"] == "failed"


@pytest.mark.parametrize("installed", [False, True])
def test_a_recent_failure_is_not_retried_until_it_ages_or_is_forced(
    flow, tmp_path, monkeypatch, installed
):
    if installed:
        flow["info"] = _info(tmp_path, installed = dict(hp.PAIR))
    flow["probe"] = (False, "ImportError: DLL load failed")
    hp.ensure("py")
    ran, net = list(flow["ran"]), list(flow["net"])
    flow["probe"] = (True, "")
    assert hp.ensure("py")["status"] == "failed" and (flow["ran"], flow["net"]) == (ran, net)
    failed_at = hp.read_marker(str(tmp_path))["time"]
    monkeypatch.setattr(hp.time, "time", lambda: failed_at + hp.FAILED_RETRY_S + 1)
    assert hp.ensure("py")["status"] in ("ready", "already")
    assert hp.read_marker(str(tmp_path))["result"] == "ok"


def test_force_retries_a_recent_failure(flow, tmp_path):
    flow["probe"] = (False, "ImportError: DLL load failed")
    hp.ensure("py")
    flow["probe"] = (True, "")
    assert hp.ensure("py")["status"] == "failed"
    assert hp.ensure("py", force = True)["status"] == "ready"


def test_the_backend_fingerprint_matches_the_installer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from core.rag import vec_ann
    finally:
        sys.path.pop(0)
    out = subprocess.run(
        [sys.executable, str(_PATH), "--check-only", "--json"], capture_output = True, text = True
    )
    assert vec_ann.hnsw_fingerprint() == hp.interpreter_info(sys.executable)["fp"], out.stderr
    assert vec_ann.PAIR == hp.PAIR_STR and vec_ann.MARKER == hp.MARKER
    assert vec_ann.LOCK == hp.LOCK and vec_ann.LOCK_STALE_S == hp._LOCK_STALE_S


@pytest.mark.parametrize("env", ["RAG_VEC_ANN_AUTO_INSTALL", "RAG_VEC_ANN"])
def test_opting_out_downloads_nothing(flow, monkeypatch, env):
    monkeypatch.setenv(env, "0")
    assert hp.ensure("py")["status"] == "disabled" and not flow["net"] and not flow["ran"]


def test_offline_uses_only_the_cache(flow, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hp.ensure("py")["status"] == "offline" and not flow["ran"]
    monkeypatch.setattr(hp, "_verified", lambda path, wheel: True)
    assert hp.ensure("py")["status"] == "ready" and flow["net"] == [False, False]


def test_check_only_never_touches_the_network(flow, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert hp.ensure("py", check_only = True)["status"] == "missing"
    assert not flow["ran"] and not flow["net"]


@pytest.mark.parametrize(
    "change, detail",
    [
        ({"version": (3, 10)}, "3.11"),
        ({"installed": {"tqdm": None}}, "numpy and tqdm"),
        ({"tags": _tags("cp312", _glibc(27, "x86_64"))}, "numkong"),
    ],
)
def test_unsupported_interpreters_are_reported_not_built(flow, tmp_path, change, detail):
    flow["info"] = _info(tmp_path, **change)
    for check_only in (True, False):
        r = hp.ensure("py", check_only = check_only)
        assert r["status"] == "unsupported" and detail in r["detail"] and not flow["ran"]


def test_a_pair_installed_elsewhere_is_probed_once(flow, tmp_path):
    flow["info"] = _info(tmp_path, installed = dict(hp.PAIR))
    assert hp.ensure("py", check_only = True)["status"] == "already"
    assert hp.read_marker(str(tmp_path)) is None
    assert (
        hp.ensure("py")["status"] == "already" and hp.read_marker(str(tmp_path))["result"] == "ok"
    )
    assert not flow["ran"]


def test_strict_exit_codes(monkeypatch, capsys):
    for status, code in (("ready", 0), ("already", 0), ("failed", 1), ("unsupported", 1)):
        monkeypatch.setattr(hp, "ensure", lambda *a, **k: hp._result(status))
        assert hp.main(["--strict", "--json"]) == code
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == status
        assert hp.main(["--json"]) == 0


def test_a_held_lock_reports_busy(flow, tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "_LOCK_STALE_S", 1e9)
    (tmp_path / ".unsloth-hnsw.lock").write_text("1")
    monkeypatch.setattr(hp.time, "sleep", lambda s: None)
    r = hp.ensure("py", timeout = 0.5)
    assert r["status"] == "busy" and not flow["ran"]


def test_the_lock_times_out_while_another_holder_is_live(tmp_path):
    path = tmp_path / "l"
    with hp._Lock(path, 5):
        with pytest.raises(TimeoutError):
            with hp._Lock(path, 0.6):
                pass
    assert not path.exists()


def _network():
    if os.environ.get("CI"):
        return True
    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout = 5).close()
        return True
    except Exception:
        return False


@pytest.mark.allow_network
@pytest.mark.skipif(not _network(), reason = "needs network to fetch the pinned wheels")
def test_the_pinned_pair_installs_and_runs_on_this_interpreter(tmp_path):
    info = hp.interpreter_info(sys.executable)
    chosen = hp.select(info["tags"], CATALOG)
    assert all(chosen.values()), f"no pinned wheel for {info['tags'][:3]}"
    files = [
        str(
            hp.slim_wheel(
                hp.obtain(w, CATALOG, tmp_path / "cache", 1e18, True)[0], tmp_path / "slim"
            )
        )
        for w in chosen.values()
    ]
    target = tmp_path / "site"
    uv = hp._uv()
    if uv:
        cmd = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            *files,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            *files,
        ]
    subprocess.run(cmd, check = True, capture_output = True)
    ok, reason = hp.probe(sys.executable, extra_path = str(target))
    assert ok, reason
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    libs = [f for f in target.rglob("*") if f.is_file() and f.read_bytes()[:4] == b"\x7fELF"]
    debug = [f.name for f in libs if any(n.startswith(".debug") for n in _sections(f.read_bytes()))]
    shutil.rmtree(target, ignore_errors = True)
    # Unstripped, Linux x86_64 is ~60 MB on disk; every platform's slim pair is under 15 MB.
    assert size < 15e6, f"installed pair is {size / 1e6:.1f} MB"
    assert not debug, f"debug info left in {debug}"


def _elf(tail_segment: bool) -> bytes:
    """A tiny ELF64 shaped like the PyPI builds: code, then DWARF, then (as patchelf leaves them)
    another loaded segment and the symbol / string tables, section headers last."""
    import struct

    align, debug = 0x1000, 3 * 0x1000 + 123
    names = b"\0.text\0.debug_info\0.extra\0.symtab\0.strtab\0.shstrtab\0"
    off = {
        n: names.index(n.encode() + b"\0")
        for n in (".text", ".debug_info", ".extra", ".symtab", ".strtab", ".shstrtab")
    }
    body = bytearray(b"\0" * 0x1000)
    text = (0x1000, b"\xc3" * 64)
    body += text[1] + b"\0" * (0x1000 - 64)
    dbg_at = len(body)
    body += b"D" * debug
    body += b"\0" * (-len(body) % align)
    extra_at = len(body)
    body += b"E" * 256
    sym_at = len(body)
    body += b"\0" * 48
    str_at = len(body)
    body += b"\0x\0"
    shs_at = len(body)
    body += names
    body += b"\0" * (-len(body) % 8)
    shoff = len(body)
    phdrs = [(1, 5, 0, 0, 0, 0x1040, 0x1040, align)]
    if tail_segment:
        phdrs.append((1, 6, extra_at, 0x10000, 0x10000, 256, 256, align))
    secs = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (off[".text"], 1, 0x6, 0x1000, 0x1000, 64, 0, 0, 16, 0),
        (off[".debug_info"], 1, 0, 0, dbg_at, debug, 0, 0, 1, 0),
        (
            off[".extra"],
            1,
            0x3 if tail_segment else 0,
            0x10000 if tail_segment else 0,
            extra_at,
            256,
            0,
            0,
            8,
            0,
        ),
        (off[".symtab"], 2, 0, 0, sym_at, 48, 5, 1, 8, 24),
        (off[".strtab"], 3, 0, 0, str_at, 3, 0, 0, 1, 0),
        (off[".shstrtab"], 3, 0, 0, shs_at, len(names), 0, 0, 1, 0),
    ]
    for sec in secs:
        body += struct.pack("<IIQQQQIIQQ", *sec)
    struct.pack_into("<4sBBBB8xHHIQQQIHHHHHH", body, 0, b"\x7fELF", 2, 1, 1, 0, 3, 62, 1, 0, 64,
                     shoff, 0, 64, 56, len(phdrs), 64, len(secs), 6)  # fmt: skip
    for k, ph in enumerate(phdrs):
        struct.pack_into("<IIQQQQQQ", body, 64 + k * 56, *ph)
    return bytes(body)


def _sections(data: bytes) -> dict:
    import struct

    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shnum, shstrndx = struct.unpack_from("<HH", data, 0x3C)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + k * 64) for k in range(shnum)]
    base = secs[shstrndx][4]
    out = {}
    for sec in secs[1:]:
        nm = data[base + sec[0] : data.index(b"\0", base + sec[0])].decode()
        out[nm] = (sec, data[sec[4] : sec[4] + sec[5]])
    return out


@pytest.mark.parametrize("tail_segment", [False, True])
def test_debug_info_is_cut_without_moving_any_loaded_byte(tail_segment):
    import struct

    raw = _elf(tail_segment)
    slim = hp.strip_elf_debug(raw)
    assert slim is not None and len(slim) <= len(raw) - 3 * 0x1000
    before, after = _sections(raw), _sections(slim)
    assert ".debug_info" in before and ".debug_info" not in after
    for name in (".text", ".extra", ".symtab", ".strtab"):
        assert after[name][1] == before[name][1], name
    assert after[".symtab"][0][6] == list(after).index(".strtab") + 1  # sh_link remapped
    (phoff,) = struct.unpack_from("<Q", slim, 0x20)
    (phnum,) = struct.unpack_from("<H", slim, 0x38)
    for k in range(phnum):
        _, _, off, vaddr, _, filesz, _, align = struct.unpack_from(
            "<IIQQQQQQ", slim, phoff + k * 56
        )
        assert (off - vaddr) % align == 0
        src = _elf(tail_segment)
        o_off = struct.unpack_from("<IIQQQQQQ", src, 64 + k * 56)[2]
        if k:
            assert slim[off : off + filesz] == src[o_off : o_off + filesz]


@pytest.mark.parametrize(
    "data", [b"", b"MZ\x90\x00" * 40, b"\xcf\xfa\xed\xfe" * 40, b"\x7fELF" + b"\x01" * 60]
)
def test_non_elf64_or_unsafe_files_are_left_alone(data):
    assert hp.strip_elf_debug(data) is None


def test_a_kept_section_inside_the_debug_run_is_not_touched():
    import struct

    raw = bytearray(_elf(False))
    (shoff,) = struct.unpack_from("<Q", raw, 0x28)
    dbg = list(struct.unpack_from("<IIQQQQIIQQ", raw, shoff + 2 * 64))
    sym = list(struct.unpack_from("<IIQQQQIIQQ", raw, shoff + 4 * 64))
    sym[4] = dbg[4] + 8  # overlap: symtab inside the debug bytes
    struct.pack_into("<IIQQQQIIQQ", raw, shoff + 4 * 64, *sym)
    assert hp.strip_elf_debug(bytes(raw)) is None


def test_slim_wheel_rewrites_record_and_leaves_other_wheels_alone(tmp_path):
    import base64
    import hashlib
    import zipfile

    lib = _elf(True)
    src = tmp_path / "numkong-7.8.6-cp312-cp312-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("numkong/_numkong.so", lib)
        z.writestr("numkong/__init__.py", "x = 1\n")
        z.writestr("numkong-7.8.6.dist-info/RECORD",
                   "numkong/_numkong.so,sha256=old,%d\nnumkong/__init__.py,sha256=keep,6\n"
                   "numkong-7.8.6.dist-info/RECORD,,\n" % len(lib))  # fmt: skip
    out = hp.slim_wheel(src, tmp_path / "slim")
    assert out != src and out.name == src.name
    with zipfile.ZipFile(out) as z:
        so = z.read("numkong/_numkong.so")
        record = z.read("numkong-7.8.6.dist-info/RECORD").decode().splitlines()
        assert z.read("numkong/__init__.py") == b"x = 1\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(so).digest()).rstrip(b"=").decode()
    assert len(so) < len(lib) and record[0] == f"numkong/_numkong.so,sha256={digest},{len(so)}"
    assert record[1:] == ["numkong/__init__.py,sha256=keep,6", "numkong-7.8.6.dist-info/RECORD,,"]
    mac = tmp_path / "usearch-2.26.6-cp312-cp312-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(mac, "w") as z:
        z.writestr("usearch/compiled.so", b"\xcf\xfa\xed\xfe" + b"\0" * 100)
        z.writestr("usearch-2.26.6.dist-info/RECORD", "usearch/compiled.so,sha256=x,104\n")
    assert hp.slim_wheel(mac, tmp_path / "slim") == mac


def test_a_ready_install_leaves_no_wheels_behind_in_the_venv(flow, tmp_path):
    cache = tmp_path / hp.CACHE
    cache.mkdir(parents = True)
    (cache / "leftover.whl").write_bytes(b"x")
    assert hp.ensure("py")["status"] == "ready"
    assert not cache.parent.exists()


def test_a_probe_failure_must_repeat_before_it_counts(monkeypatch):
    runs = iter([(False, "AssertionError: self search missed rows [7]"), (True, "")])
    monkeypatch.setattr(hp, "_probe_once", lambda *a: next(runs))
    assert hp.probe("py") == (True, "")
    runs = iter([(False, "first"), (False, "second")])
    assert hp.probe("py") == (False, "first")


def test_uv_is_found_in_studios_private_dir_without_path(tmp_path, monkeypatch):
    for k in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UV_INSTALL_DIR", "XDG_BIN_HOME"):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setattr(hp.shutil, "which", lambda name: None)
    monkeypatch.setattr(hp.Path, "home", lambda: tmp_path / "nohome")
    name = "uv.exe" if os.name == "nt" else "uv"
    uv = tmp_path / "studio" / "uv" / name
    uv.parent.mkdir(parents = True)
    uv.write_text("#!/bin/sh\n", encoding = "utf-8")
    uv.chmod(0o755)
    assert hp._uv(str(tmp_path / "studio" / "unsloth_studio")) == str(uv)
    cmds = hp._pip_cmd("py", "install", ["w.whl"], str(tmp_path / "studio" / "unsloth_studio"))
    assert cmds[0][0] == str(uv)


@pytest.mark.parametrize(
    "env, disabled",
    [
        ({"RAG_VEC_ANN": "0"}, True),
        ({"RAG_VEC_ANN": "false"}, False),
        ({"RAG_VEC_ANN_AUTO_INSTALL": "off"}, True),
        ({}, False),
    ],
)
def test_opt_outs_read_like_the_backend(monkeypatch, env, disabled):
    for k in ("RAG_VEC_ANN", "RAG_VEC_ANN_AUTO_INSTALL"):
        monkeypatch.delenv(k, raising = False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert hp._disabled() is disabled


def test_uv_no_index_counts_as_offline(monkeypatch):
    for k in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "UNSLOTH_OFFLINE",
        "UV_OFFLINE",
        "PIP_NO_INDEX",
    ):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setenv("UV_NO_INDEX", "1")
    assert hp._offline()
