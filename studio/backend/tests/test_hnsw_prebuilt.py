# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studio/install_hnsw_prebuilt.py: hash-pinned install, probe / marker / lock, in-place debug stripping,
and one real install of the pinned pair for this interpreter."""

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


def test_the_requirements_pin_every_wheel_of_both_packages_by_hash():
    blocks = hp.requirement_blocks()
    assert set(blocks) == {n for n, _ in hp.PAIR}
    for name, version in hp.PAIR:
        block = blocks[name]
        assert block.startswith(f"{name}=={version} \\\n") and block.count("--hash=sha256:") >= 20
        assert all(len(h.split()[0]) == 64 for h in block.split("--hash=sha256:")[1:])


@pytest.mark.parametrize("uv", [True, False])
def test_installs_take_only_hash_pinned_wheels_and_ignore_user_config(monkeypatch, uv):
    monkeypatch.setattr(hp, "_uv", lambda prefix = None: "/bin/uv" if uv else None)
    cmds = hp._pip_cmd("py", "install", ["-r", "req.txt"])
    for cmd in cmds:
        assert {"--no-deps", "--require-hashes", ":all:"} <= set(cmd) and cmd[-2:] == [
            "-r",
            "req.txt",
        ]
        assert cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert ("--no-config" in cmds[0]) is uv and cmds[-1][1:3] == ["-m", "pip"]


def test_the_child_env_keeps_tls_proxy_and_index_but_drops_constraints_and_secrets(monkeypatch):
    for k, v in {"UV_CONSTRAINT": "/c.txt", "PIP_CONSTRAINT": "/c.txt", "HF_TOKEN": "hf_x",
                 "UV_INDEX_URL": "https://corp/simple", "HTTPS_PROXY": "http://p:1"}.items():  # fmt: skip
        monkeypatch.setenv(k, v)
    env = hp._child_env()
    assert "UV_CONSTRAINT" not in env and "PIP_CONSTRAINT" not in env and "HF_TOKEN" not in env
    assert env["UV_INDEX_URL"] == "https://corp/simple" and env["HTTPS_PROXY"] == "http://p:1"
    assert env["PIP_CONFIG_FILE"] == os.devnull and env["PYTHONIOENCODING"] == "utf-8"


@pytest.mark.parametrize(
    "platform, flag, owner, native",
    [("darwin", "", "", True), ("win32", "", "", True), ("linux", "", "", False),
     ("linux", "", "tauri", True), ("darwin", "0", "", False), ("linux", "1", "", True)],
)  # fmt: skip
def test_uv_uses_the_os_trust_store_where_studio_does(monkeypatch, platform, flag, owner, native):
    monkeypatch.setattr(hp.sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_TLS", flag)
    monkeypatch.setenv("UNSLOTH_STUDIO_DESKTOP_OWNER_KIND", owner)
    monkeypatch.delenv("UV_NATIVE_TLS", raising = False)
    assert (hp._child_env().get("UV_NATIVE_TLS") == "1") is native


def _info(
    prefix,
    installed = None,
    version = (3, 12),
    free_threaded = False,
):
    pkgs = {"usearch": None, "numkong": None, "numpy": "2.0", "tqdm": "4.0"}
    pkgs.update(installed or {})
    return {
        "prefix": str(prefix),
        "version": list(version),
        "free_threaded": free_threaded,
        "fp": "fp1",
        "installed": pkgs,
        "sites": {},
    }


def _names(cmd):
    if "-r" in cmd:
        text = Path(cmd[cmd.index("-r") + 1]).read_text(encoding = "utf-8")
        return [l.split("==")[0] for l in text.splitlines() if l and not l.startswith(" ")]
    return [a for a in cmd if a in ("usearch", "numkong")]


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """ensure() with the interpreter, installer, stripper and probe faked; records what ran."""
    for k in ("RAG_VEC_ANN_AUTO_INSTALL", "RAG_VEC_ANN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
              "UNSLOTH_OFFLINE", "UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX"):  # fmt: skip
        monkeypatch.delenv(k, raising = False)
    state = {
        "info": _info(tmp_path),
        "probe": (True, ""),
        "ran": [],
        "calls": [],
        "wants": [],
        "slimmed": [],
    }

    def info(python, timeout = 60.0):
        out = json.loads(json.dumps(state["info"]))
        out["sites"] = {n: "site" for n, _ in hp.PAIR if out["installed"].get(n)}
        return out

    def run_first(cmds, timeout):
        verb = cmds[0][2] if cmds[0][1] == "pip" else cmds[0][3]
        names = _names(cmds[0])
        state["ran"].append(verb)
        state["calls"].append((verb, names))
        version = dict(hp.PAIR)
        for n in names:
            state["info"]["installed"][n] = version[n] if verb == "install" else None

    monkeypatch.setattr(hp, "interpreter_info", info)
    monkeypatch.setattr(hp, "_run_first", run_first)
    monkeypatch.setattr(hp, "slim_installed", lambda site, name: state["slimmed"].append(name) or 0)

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


def test_a_fresh_install_strips_probes_and_marks_ready(flow, tmp_path):
    r = hp.ensure("py")
    assert r["status"] == "ready" and r["usearch"] == "2.26.6"
    assert flow["calls"] == [("install", ["usearch", "numkong"])]
    assert flow["slimmed"] == ["usearch", "numkong"] and flow["wants"] == [hp.PAIR_STR]
    assert hp.read_marker(str(tmp_path))["result"] == "ok"
    assert hp.ensure("py")["status"] == "already" and flow["ran"] == ["install"]


def test_a_failed_probe_uninstalls_and_marks_failed(flow, tmp_path):
    flow["probe"] = (False, "ImportError: DLL load failed")
    r = hp.ensure("py")
    assert r["status"] == "failed" and "DLL" in r["detail"]
    assert flow["calls"] == [
        ("install", ["usearch", "numkong"]),
        ("uninstall", ["usearch", "numkong"]),
    ]
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
        flow["info"]["installed"].update(dict(hp.PAIR))
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


def test_a_platform_with_no_wheel_is_unsupported_and_not_retried(flow, tmp_path, monkeypatch):
    def run_first(cmds, timeout):
        flow["ran"].append("install")
        raise RuntimeError(
            "uv: Because usearch==2.26.6 has no usable wheels and you require it ..."
        )

    monkeypatch.setattr(hp, "_run_first", run_first)
    r = hp.ensure("py")
    assert r["status"] == "unsupported" and "usable wheels" in r["detail"]
    assert hp.read_marker(str(tmp_path))["result"] == "failed"
    assert hp.ensure("py")["status"] == "failed" and flow["ran"] == ["install"]


@pytest.mark.parametrize("ok", [True, False])
def test_half_a_pair_gets_only_its_missing_half_and_keeps_the_other_on_failure(flow, tmp_path, ok):
    flow["info"] = _info(tmp_path, installed = {"numkong": "7.8.7"})
    flow["probe"] = (True, "") if ok else (False, "ImportError: MSVCP140.dll")
    r = hp.ensure("py")
    assert flow["calls"][0] == ("install", ["usearch"]) and flow["slimmed"] == ["usearch"]
    # The kept numkong is not the pinned one: probe without the pin check.
    assert flow["wants"] == [""]
    if ok:
        assert r["status"] == "ready" and len(flow["calls"]) == 1
    else:
        assert r["status"] == "failed" and flow["calls"][1] == ("uninstall", ["usearch"])
        assert flow["info"]["installed"]["numkong"] == "7.8.7"


@pytest.mark.parametrize("numkong", ["7.8.7", "8.0.0"])
def test_a_pair_from_an_older_install_is_probed_not_replaced(flow, tmp_path, numkong):
    flow["info"] = _info(tmp_path, installed = {"usearch": "2.26.6", "numkong": numkong})
    assert hp.ensure("py", check_only = True)["status"] == "already"
    assert hp.ensure("py")["status"] == "already"
    assert flow["wants"] == [""] and not flow["ran"] and not flow["slimmed"]
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
    ran = list(flow["ran"])
    flow["probe"] = (True, "")
    assert hp.ensure("py")["status"] == "failed" and flow["ran"] == ran
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
def test_opting_out_installs_nothing(flow, monkeypatch, env):
    monkeypatch.setenv(env, "0")
    assert hp.ensure("py")["status"] == "disabled" and not flow["ran"]


@pytest.mark.parametrize("env", ["HF_HUB_OFFLINE", "UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX"])
def test_offline_installs_nothing(flow, monkeypatch, env):
    monkeypatch.setenv(env, "1")
    assert hp.ensure("py")["status"] == "offline" and not flow["ran"]


def test_check_only_never_installs(flow):
    assert hp.ensure("py", check_only = True)["status"] == "missing" and not flow["ran"]


@pytest.mark.parametrize(
    "change, detail",
    [
        ({"version": (3, 10)}, "3.11"),
        ({"free_threaded": True}, "GIL"),
        ({"installed": {"tqdm": None}}, "numpy and tqdm"),
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
def test_the_pinned_pair_installs_slims_and_runs_on_this_interpreter(tmp_path):
    target = tmp_path / "site"
    hp._run_first(
        hp._pip_cmd(
            sys.executable, "install", ["--target", str(target), "-r", str(hp.REQUIREMENTS)]
        ),
        600,
    )
    for name, _ in hp.PAIR:
        hp.slim_installed(str(target), name)
    ok, reason = hp.probe(sys.executable, extra_path = str(target))
    assert ok, reason
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    libs = [f for f in target.rglob("*") if f.is_file() and f.read_bytes()[:4] == b"\x7fELF"]
    debug = [f.name for f in libs if any(n.startswith(".debug") for n in _sections(f.read_bytes()))]
    shutil.rmtree(target, ignore_errors = True)
    # Unstripped, Linux x86_64 is ~60 MB on disk; every platform's slim pair is under 15 MB.
    assert size < 15e6, f"installed pair is {size / 1e6:.1f} MB"
    assert not debug, f"debug info left in {debug}"  # fmt: skip


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


def test_slim_installed_rewrites_files_and_record_without_touching_a_hardlinked_cache(tmp_path):
    import base64
    import hashlib

    lib = _elf(True)
    site = tmp_path / "site"
    (site / "numkong").mkdir(parents = True)
    (site / "numkong-7.8.6.dist-info").mkdir()
    so = site / "numkong" / "_numkong.so"
    so.write_bytes(lib)
    (site / "numkong" / "__init__.py").write_text("x = 1\n", encoding = "utf-8")
    (site / "numkong-7.8.6.dist-info" / "RECORD").write_text(
        f"numkong/_numkong.so,sha256=old,{len(lib)}\nnumkong/__init__.py,sha256=keep,6\n"
        "numkong-7.8.6.dist-info/RECORD,,\n", encoding = "utf-8")  # fmt: skip
    cache = tmp_path / "uv-cache.so"
    os.link(so, cache)  # uv installs by hardlink from its cache
    saved = hp.slim_installed(str(site), "numkong")
    slim = so.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(slim).digest()).rstrip(b"=").decode()
    record = (site / "numkong-7.8.6.dist-info" / "RECORD").read_text(encoding = "utf-8").splitlines()
    assert saved == len(lib) - len(slim) > 0 and cache.read_bytes() == lib
    assert record == [f"numkong/_numkong.so,sha256={digest},{len(slim)}",
                      "numkong/__init__.py,sha256=keep,6", "numkong-7.8.6.dist-info/RECORD,,"]  # fmt: skip
    mac = tmp_path / "mac"
    (mac / "usearch").mkdir(parents = True)
    (mac / "usearch-2.26.6.dist-info").mkdir()
    (mac / "usearch" / "compiled.so").write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 100)
    (mac / "usearch-2.26.6.dist-info" / "RECORD").write_text(
        "usearch/compiled.so,sha256=x,104\n", encoding = "utf-8"
    )
    assert hp.slim_installed(str(mac), "usearch") == 0  # fmt: skip


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
    cmds = hp._pip_cmd(
        "py", "install", ["-r", "req.txt"], str(tmp_path / "studio" / "unsloth_studio")
    )
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
