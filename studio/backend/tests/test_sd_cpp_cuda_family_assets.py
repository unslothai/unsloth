# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-GPU-family sd.cpp CUDA bundles: the manifest-driven pick by compute capability (llama.cpp's
older / newer / portable scheme), its fallbacks, and the pre-split releases that must keep
installing the single CUDA bundle. CPU only: every host fact is passed in or monkeypatched."""

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import install_sd_cpp_prebuilt as sdmod  # noqa: E402

TAG = "master-990-abcdef0-u1234567"

# The rows of the prebuilt workflow (package_bundle.cuda_profile_info output).
_PROFILES = {
    ("cuda12", "older"): (10, ["70", "75", "80", "86", "89"], 70, 89, ["89"]),
    ("cuda12", "newer"): (20, ["86", "89", "90", "100", "103", "120"], 86, 120, ["100"]),
    ("cuda12", "portable"): (
        30,
        ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
        70,
        120,
        ["100"],
    ),
    ("cuda13", "older"): (40, ["75", "80", "86", "89"], 75, 89, ["89"]),
    ("cuda13", "newer"): (50, ["86", "89", "90", "100", "103", "120"], 86, 120, ["100"]),
    ("cuda13", "portable"): (
        60,
        ["75", "80", "86", "89", "90", "100", "103", "120"],
        75,
        120,
        ["100"],
    ),
}


def _asset_name(os_: str, line: str, klass: str) -> str:
    if klass == "portable":
        if os_ == "linux":
            return f"sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64-{line}.zip"
        return f"sd-{TAG}-bin-win-{line}-x64.zip"
    ext = "tar.gz" if os_ == "linux" else "zip"
    return f"sd-{TAG}-{os_}-x64-{line}-{klass}.{ext}"


def _manifest(skip: tuple = ()) -> dict:
    assets = []
    for os_ in ("linux", "windows"):
        for (line, klass), (rank, sms, lo, hi, ptx) in _PROFILES.items():
            name = _asset_name(os_, line, klass)
            if name in skip:
                continue
            assets.append(
                {
                    "name": name,
                    "label": "",
                    "size": 1,
                    "sha256": "0" * 64,
                    "backend": "cuda",
                    "os": os_,
                    "arch": "x64",
                    "bundle_profile": f"{line}-{klass}",
                    "runtime_line": line,
                    "coverage_class": klass,
                    "rank": rank,
                    "supported_sms": sms,
                    "min_sm": lo,
                    "max_sm": hi,
                    "ptx_sms": ptx,
                }
            )
    for plain in (
        "bin-Linux-Ubuntu-22.04-x86_64.zip",
        "bin-win-cpu-x64.zip",
        "bin-win-vulkan-x64.zip",
        "bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
        "bin-Darwin-macOS-arm64.zip",
    ):
        assets.append({"name": f"sd-{TAG}-{plain}", "label": "", "size": 1, "sha256": "0" * 64})
    return {"schema": 1, "tag": TAG, "assets": assets}


def _names(manifest: dict) -> list[str]:
    return [a["name"] for a in manifest["assets"]] + [
        sdmod.SD_MANIFEST_ASSET,
        "sd-prebuilt-sha256.json",
    ]


def _pick(
    sms,
    *,
    system = "Linux",
    major = "12",
    manifest = None,
    names = None,
):
    manifest = manifest or _manifest()
    bundles = sdmod.cuda_bundles_from_manifest(manifest, system = system, machine = "x86_64")
    attempts, _log = sdmod.select_cuda_bundles(
        bundles, names if names is not None else _names(manifest), cuda_major = major, host_sms = sms
    )
    return [b.asset_name for b in attempts]


L = lambda line, klass: _asset_name("linux", line, klass)  # noqa: E731
W = lambda line, klass: _asset_name("windows", line, klass)  # noqa: E731


@pytest.mark.parametrize(
    "sms, family",
    [
        (["70"], "older"),  # V100
        (["75"], "older"),  # T4, RTX 20
        (["80"], "older"),  # A100
        (["86"], "older"),  # RTX 30: in both families, the tighter one wins (llama.cpp's order)
        (["89"], "older"),  # L4, RTX 40
        (["90"], "newer"),  # H100
        (["100"], "newer"),  # B200
        (["103"], "newer"),  # B300 runs the sm_100 SASS
        (["120"], "newer"),  # RTX 50
        (["8.9"], "older"),  # nvidia-smi's dotted form
    ],
)
def test_each_family_gets_its_bundle_then_the_portable_one(sms, family):
    assert _pick(sms) == [L("cuda12", family), L("cuda12", "portable")]


def test_cuda13_follows_its_own_rows():
    assert _pick(["75"], major = "13") == [L("cuda13", "older"), L("cuda13", "portable")]
    assert _pick(["120"], major = "13") == [L("cuda13", "newer"), L("cuda13", "portable")]
    # CUDA 13 has no Volta build at all; nothing here, so the caller keeps the pre-split pick.
    assert _pick(["70"], major = "13") == []


def test_mixed_generations_need_the_portable_bundle():
    assert _pick(["75", "120"]) == [L("cuda12", "portable")]
    assert _pick(["89", "90"]) == [L("cuda12", "newer"), L("cuda12", "portable")]


def test_unknown_compute_capability_takes_only_the_portable_bundle():
    assert _pick([]) == [L("cuda12", "portable")]


def test_a_card_newer_than_every_list_jits_the_family_ptx_first():
    assert _pick(["130"]) == [L("cuda12", "newer"), L("cuda12", "portable")]
    assert _pick(["121"]) == [L("cuda12", "newer"), L("cuda12", "portable")]


def test_a_newer_card_beside_turing_keeps_the_newest_ptx():
    # Only the older family covers sm_75, but its sm_89 PTX would lose portable's sm_100 kernels on the new card.
    assert _pick(["75", "130"]) == [L("cuda12", "portable")]


def test_a_card_older_than_every_ptx_gets_nothing():
    # Pascal: no bundle has SASS or JIT-able PTX for it; the caller falls back as before the split.
    assert _pick(["61"]) == []


def test_windows_and_linux_pick_their_own_assets():
    assert _pick(["75"], system = "Windows") == [W("cuda12", "older"), W("cuda12", "portable")]
    assert _pick(["120"], system = "Windows", major = "13") == [
        W("cuda13", "newer"),
        W("cuda13", "portable"),
    ]
    assert _pick(["75"], system = "Linux")[0].endswith(".tar.gz")
    assert _pick(["75"], system = "Darwin") == []


def test_a_family_bundle_missing_from_the_release_falls_back_to_portable():
    m = _manifest()
    names = [n for n in _names(m) if n != L("cuda12", "older")]
    assert _pick(["75"], manifest = m, names = names) == [L("cuda12", "portable")]
    # And a manifest that never listed it (the leg failed to build) behaves the same.
    m2 = _manifest(skip = (L("cuda12", "older"),))
    assert _pick(["75"], manifest = m2) == [L("cuda12", "portable")]


def test_malformed_manifest_entries_are_ignored():
    m = _manifest()
    for a in m["assets"]:
        if a.get("coverage_class") == "older":
            a["min_sm"] = "70"  # not an int
    assert _pick(["75"], manifest = m) == [L("cuda12", "portable")]
    assert sdmod.cuda_bundles_from_manifest({"assets": "x"}, system = "Linux", machine = "x86_64") == []
    assert sdmod.cuda_bundles_from_manifest(None, system = "Linux", machine = "x86_64") == []


@pytest.mark.parametrize("system, machine", [("Linux", "x86_64"), ("Windows", "AMD64")])
@pytest.mark.parametrize(
    "accelerator, major", [("cuda", "12"), ("cuda", "13"), ("cuda", None), ("auto", None)]
)
def test_the_pre_split_resolver_never_sees_a_family_bundle(system, machine, accelerator, major):
    """A Studio from before the split picks by name; on a split release it must still land on the
    portable bundle (or the CPU one), whatever order GitHub lists the assets in."""
    names = _names(_manifest())
    for order in (names, list(reversed(names))):
        got = sdmod.resolve_release_asset(
            order, system = system, machine = machine, accelerator = accelerator, cuda_major = major
        )
        assert got is not None
        assert "-older" not in got and "-newer" not in got and not got.endswith(".tar.gz")


# ── resolution through _resolve_repo_asset (what install / --print-asset use) ──


def _release(names) -> dict:
    return {
        "tag_name": TAG,
        "assets": [
            {"name": n, "browser_download_url": f"https://example.invalid/{n}", "digest": None}
            for n in names
        ],
    }


def _route(
    monkeypatch,
    release,
    manifest,
    sms,
    *,
    system = "Linux",
    machine = "x86_64",
):
    fetched = []
    monkeypatch.setattr(sdmod, "_fetch_release", lambda *a, **k: release)
    monkeypatch.setattr(sdmod, "_fetch_manifest", lambda rel, **k: fetched.append(1) or manifest)
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (sms, sms))
    monkeypatch.setattr(sdmod.platform, "system", lambda: system)
    monkeypatch.setattr(sdmod.platform, "machine", lambda: machine)
    return fetched


def test_resolve_takes_the_family_bundle_on_a_split_release(monkeypatch):
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), m, ["75"])
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "12"
    )
    assert chosen == L("cuda12", "older")
    assert sdmod._SELECTED_CUDA_SMS[chosen] == ["70", "75", "80", "86", "89"]


def test_resolve_keeps_the_single_cuda_bundle_on_a_pre_split_release(monkeypatch):
    """The current pin: no family bundles, so no manifest fetch and the same pick as before."""
    names = [
        f"sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip",
        f"sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda13.zip",
        f"sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip",
        f"sd-{TAG}-bin-win-cuda12-x64.zip",
        sdmod.SD_MANIFEST_ASSET,
    ]
    fetched = _route(monkeypatch, _release(names), {"assets": []}, ["75"])
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "13"
    )
    assert chosen == names[1]
    assert fetched == []


def test_resolve_falls_back_to_the_portable_name_when_the_manifest_is_unusable(monkeypatch):
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), None, ["75"])
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "12"
    )
    assert chosen == L("cuda12", "portable")


def test_resolve_for_pascal_keeps_the_pre_split_pick(monkeypatch):
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), m, ["61"])
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "12"
    )
    assert chosen == L("cuda12", "portable")


def test_resolve_on_windows(monkeypatch):
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), m, ["89"], system = "Windows", machine = "AMD64")
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "13"
    )
    assert chosen == W("cuda13", "older")


def test_non_cuda_accelerators_never_read_the_manifest(monkeypatch):
    m = _manifest()
    fetched = _route(monkeypatch, _release(_names(m)), m, ["75"])
    for accel in ("auto", "cpu", "vulkan"):
        _rel, chosen = sdmod._resolve_repo_asset("unslothai/stable-diffusion.cpp", TAG, accel, None)
        assert chosen and "cuda" not in chosen
    assert fetched == []


def test_masked_gpus_are_covered_first(monkeypatch):
    """CUDA_VISIBLE_DEVICES hides an RTX 50 beside a T4: cover both, as llama.cpp does, then the
    visible-only pick."""
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), m, ["75"])
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["75", "120"], ["75"]))
    _rel, chosen = sdmod._resolve_repo_asset(
        "unslothai/stable-diffusion.cpp", TAG, "cuda", None, cuda_major = "12"
    )
    assert chosen == L("cuda12", "portable")


def test_env_override_of_compute_capability(monkeypatch):
    monkeypatch.setattr(sdmod, "_HOST_SMS_CACHE", [])
    monkeypatch.setenv("UNSLOTH_SD_CPP_CUDA_SMS", "7.5, 8.9")
    assert sdmod.host_cuda_sms() == (["75", "89"], ["75", "89"])


def test_fetch_manifest_checks_the_release_digest(monkeypatch):
    body = json.dumps(_manifest()).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sdmod.urllib.request, "urlopen", lambda req, timeout = 0: _Resp(body))
    good = {
        "assets": [
            {
                "name": sdmod.SD_MANIFEST_ASSET,
                "browser_download_url": "https://example.invalid/m",
                "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
            }
        ]
    }
    assert sdmod._fetch_manifest(good)["tag"] == TAG
    bad = {"assets": [dict(good["assets"][0], digest = "sha256:" + "0" * 64)]}
    assert sdmod._fetch_manifest(bad) is None
    assert sdmod._fetch_manifest({"assets": []}) is None


# ── install of a .tar.gz family bundle ──


def _tar_bundle(stem: str, *, extra = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
        for name, data in (
            (f"{stem}/sd-cli", b"#!/bin/sh\necho sd-cli\n"),
            (f"{stem}/sd-server", b"#!/bin/sh\necho sd-server\n"),
            (f"{stem}/libggml-cuda.so.0", b"\x7fELF fake"),
        ):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o755
            tf.addfile(ti, io.BytesIO(data))
        if extra:
            extra(tf)
    return buf.getvalue()


def _install_split(
    monkeypatch,
    tmp_path,
    tar_bytes,
    sms = ("75",),
):
    m = _manifest()
    name = L("cuda12", "older")
    rel = _release(_names(m))
    for a in rel["assets"]:
        if a["name"] == name:
            a["digest"] = "sha256:" + hashlib.sha256(tar_bytes).hexdigest()
    _route(monkeypatch, rel, m, list(sms))
    monkeypatch.setattr(sdmod, "_download", lambda url, dest, **k: dest.write_bytes(tar_bytes))
    monkeypatch.setattr(sdmod.sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", TAG)
    return name, sdmod.install(install_dir = tmp_path, accelerator = "cuda", cuda_major = "12")


def test_install_of_a_tar_gz_family_bundle(monkeypatch, tmp_path):
    stem = f"sd-{TAG}-linux-x64-cuda12-older"
    name, sd_cli = _install_split(monkeypatch, tmp_path, _tar_bundle(stem))
    assert sd_cli == tmp_path / stem / "sd-cli" and sd_cli.is_file()
    assert (tmp_path / stem / "sd-server").is_file() and (
        tmp_path / stem / "libggml-cuda.so.0"
    ).is_file()
    # Neither the tarball nor its repacked zip is left behind.
    assert not list(tmp_path.glob("*.tar.gz")) and not list(tmp_path.glob("*.zip"))
    rec = sdmod.read_install_record(tmp_path)
    assert rec["asset"] == name and rec["accelerator"] == "cuda"
    assert rec["cuda_sms"] == ["70", "75", "80", "86", "89"]
    assert sdmod.install_is_stale(tmp_path) is False


def test_a_gpu_swap_outside_the_family_makes_the_install_stale(monkeypatch, tmp_path):
    stem = f"sd-{TAG}-linux-x64-cuda12-older"
    _install_split(monkeypatch, tmp_path, _tar_bundle(stem))
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["120"], ["120"]))
    assert sdmod.install_is_stale(tmp_path) is True
    # Undetectable GPUs are "cannot tell", never a reinstall.
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: ([], []))
    assert sdmod.install_is_stale(tmp_path) is False


def test_a_visible_only_fallback_is_not_stale_for_the_masked_card(monkeypatch, tmp_path):
    """No bundle covers the hidden sm_70 and the visible sm_90 together, so the pick serves sm_90
    only; that install must stay current instead of reinstalling on every ensure."""
    (tmp_path / sdmod.INSTALL_RECORD).write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "repo": "r",
                "tag": TAG,
                "requested_tag": TAG,
                "cuda_sms": ["75", "80", "86", "89", "90", "120"],
            }
        )
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", TAG)
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["70", "90"], ["90"]))
    assert sdmod.install_is_stale(tmp_path) is False
    # Unmasked, the sm_70 card is in use and the bundle does not serve it.
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["70", "90"], ["70", "90"]))
    assert sdmod.install_is_stale(tmp_path) is True


def test_a_malformed_coverage_field_is_skipped_not_raised():
    m = _manifest()
    for a in m["assets"]:
        if a.get("coverage_class") == "older":
            a["supported_sms"] = 75
        if a.get("coverage_class") == "newer":
            a["ptx_sms"] = 100
    assert _pick(["75"], manifest = m) == [L("cuda12", "portable")]


def test_a_name_based_pick_records_no_sms_from_an_earlier_resolution(monkeypatch):
    m = _manifest()
    _route(monkeypatch, _release(_names(m)), m, ["75", "120"])
    _rel, chosen = sdmod._resolve_repo_asset("r", TAG, "cuda", None, cuda_major = "12")
    assert chosen == L("cuda12", "portable") and sdmod._SELECTED_CUDA_SMS
    _route(monkeypatch, _release(_names(m)), None, ["75", "120"])
    _rel, again = sdmod._resolve_repo_asset("r", TAG, "cuda", None, cuda_major = "12")
    assert again == chosen and sdmod._SELECTED_CUDA_SMS == {}


def test_a_pre_split_install_writes_the_same_record_as_before(tmp_path):
    sdmod._write_install_record(
        tmp_path, accelerator = "cuda", repo = "r", tag = TAG, asset = "x.zip", cuda_sms = None
    )
    assert "asset" not in sdmod.read_install_record(tmp_path)


def test_no_visible_gpu_is_not_coverage_stale(monkeypatch, tmp_path):
    """An empty or unmatched mask (a short UUID) sees no GPU: unknown, so never a reinstall loop."""
    stem = f"sd-{TAG}-linux-x64-cuda12-older"
    _install_split(monkeypatch, tmp_path, _tar_bundle(stem))
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["61", "120"], []))
    assert sdmod.install_is_stale(tmp_path) is False


def test_an_unwritable_record_keeps_the_new_coverage(monkeypatch, tmp_path):
    stem = f"sd-{TAG}-linux-x64-cuda12-older"
    _install_split(monkeypatch, tmp_path, _tar_bundle(stem))

    import builtins

    def _deny(
        path,
        mode = "r",
        *a,
        **k,
    ):
        # A locked / read-only record: the write is refused before anything is truncated.
        if "w" in mode and Path(path).name == sdmod.INSTALL_RECORD:
            raise PermissionError("read-only record")
        return builtins.open(path, mode, *a, **k)

    monkeypatch.setattr(sdmod, "open", _deny, raising = False)
    sdmod._write_install_record(
        tmp_path, accelerator = "cuda", repo = "r", tag = TAG, asset = "n", cuda_sms = ["86", "120"]
    )
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["120"], ["120"]))
    try:
        # The file still names the older bundle; this process installed one that serves sm_120.
        assert sdmod.install_is_stale(tmp_path) is False
        # Once anything rewrites the record, the file is the newer answer again.
        rec = sdmod.read_install_record(tmp_path)
        assert rec["cuda_sms"] == ["70", "75", "80", "86", "89"]
        rec["repo"] = "rewritten"
        (tmp_path / sdmod.INSTALL_RECORD).write_text(json.dumps(rec))
        assert sdmod.install_is_stale(tmp_path) is True
    finally:
        sdmod._INSTALLED_CUDA_SMS_MEMO.clear()
        sdmod._INSTALLED_PIN_MEMO.clear()
        sdmod._INSTALLED_ACCELERATOR_MEMO.clear()


def test_a_pre_split_record_is_never_coverage_stale(monkeypatch, tmp_path):
    (tmp_path / sdmod.INSTALL_RECORD).write_text(
        json.dumps({"accelerator": "cuda", "repo": "r", "tag": TAG, "requested_tag": TAG})
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", TAG)
    monkeypatch.setattr(sdmod, "host_cuda_sms", lambda: (["120"], ["120"]))
    assert sdmod.install_is_stale(tmp_path) is False


def _symlink(tf):
    ti = tarfile.TarInfo("x/evil")
    ti.type = tarfile.SYMTYPE
    ti.linkname = "/etc/passwd"
    tf.addfile(ti)


def _dotdot(tf):
    ti = tarfile.TarInfo("x/../../evil")
    ti.size = 1
    tf.addfile(ti, io.BytesIO(b"x"))


@pytest.mark.parametrize(
    "extra, match", [(_symlink, "non-regular"), (_dotdot, "unsafe member path")]
)
def test_unsafe_tar_members_are_refused_before_anything_is_written(
    monkeypatch, tmp_path, extra, match
):
    with pytest.raises(RuntimeError, match = match):
        _install_split(monkeypatch, tmp_path, _tar_bundle("x", extra = extra))
    assert not (tmp_path / "x").exists()
    assert not list(tmp_path.glob("*.tar.gz")) and not list(tmp_path.glob("*.zip"))


def test_repacked_zip_keeps_the_mode_bits(tmp_path):
    src = tmp_path / "b.tar.gz"
    src.write_bytes(_tar_bundle("s"))
    out = tmp_path / "b.zip"
    sdmod._tar_gz_to_zip(src, out)
    with zipfile.ZipFile(out) as zf:
        info = zf.getinfo("s/sd-cli")
        assert (info.external_attr >> 16) & 0o777 == 0o755
        assert zf.read("s/sd-cli").startswith(b"#!/bin/sh")
