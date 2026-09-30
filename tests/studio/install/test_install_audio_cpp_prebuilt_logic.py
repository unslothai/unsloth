# SPDX-License-Identifier: AGPL-3.0-only
# Logic tests for studio/install_audio_cpp_prebuilt.py -- the self-contained audiocpp_server installer.
# No network/GPU: asset lists are the real upstream v0.8.2 release names, archives are built in a tmp dir.

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_audio_cpp_prebuilt.py"
_STUDIO_DIR = str(MODULE_PATH.parent)
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)
SPEC = importlib.util.spec_from_file_location("studio_install_audio_cpp_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

TAG = "v0.8.2-audio8-perf-hotfix"
ASSETS = [
    "asr_validation.tar.gz",
    f"audio-{TAG}-bin-macos-arm64-metal.tar.gz",
    f"audio-{TAG}-bin-macos-x64-metal.tar.gz",
    f"audio-{TAG}-bin-ubuntu-x64-cpu-portable.tar.gz",
    f"audio-{TAG}-bin-ubuntu-x64-cpu.tar.gz",
    f"audio-{TAG}-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
    f"audio-{TAG}-bin-ubuntu-x64-vulkan-portable.tar.gz",
    f"audio-{TAG}-bin-ubuntu-x64-vulkan.tar.gz",
    f"audio-{TAG}-bin-windows-x64-cpu-portable.zip",
    f"audio-{TAG}-bin-windows-x64-cpu.zip",
    f"audio-{TAG}-bin-windows-x64-cuda12.4.zip",
    f"audio-{TAG}-bin-windows-x64-cuda13.3.zip",
    f"audio-{TAG}-bin-windows-x64-vulkan.zip",
    f"audio-{TAG}-cudart-windows-x64-cuda12.4.zip",
    f"audio-{TAG}-cudart-windows-x64-cuda13.3.zip",
    "framework.tar.gz",
    "resources.tar.gz",
]


def pick(system, machine, accel, **kw):
    return M.resolve_release_asset(ASSETS, system = system, machine = machine, accelerator = accel, **kw)


@pytest.mark.parametrize(
    "system,machine,accel,expected",
    [
        ("Windows", "AMD64", "cpu", "bin-windows-x64-cpu-portable.zip"),
        ("Windows", "AMD64", "vulkan", "bin-windows-x64-vulkan.zip"),
        ("Linux", "x86_64", "cpu", "bin-ubuntu-x64-cpu-portable.tar.gz"),
        ("Linux", "x86_64", "vulkan", "bin-ubuntu-x64-vulkan-portable.tar.gz"),
        ("Linux", "x86_64", "cuda", "bin-ubuntu-x64-cuda12.8-colab.tar.gz"),
        ("Darwin", "arm64", "metal", "bin-macos-arm64-metal.tar.gz"),
        ("Darwin", "x86_64", "metal", "bin-macos-x64-metal.tar.gz"),
    ],
)
def test_resolves_the_host_bundle(system, machine, accel, expected):
    assert pick(system, machine, accel).endswith(expected)


def test_no_bundle_for_an_unbuilt_host():
    # Upstream publishes no Linux arm64 or Windows arm64 bundles: the caller must fall back, not install x64.
    assert pick("Linux", "aarch64", "cpu") is None
    assert pick("Windows", "ARM64", "cpu") is None
    assert pick("Windows", "AMD64", "metal") is None


def test_cuda_takes_the_newest_line_the_driver_runs():
    assert pick("Windows", "AMD64", "cuda", driver_cuda = (13, 4)).endswith("cuda13.3.zip")
    # A 12.x driver cannot load a 13.3 bundle.
    assert pick("Windows", "AMD64", "cuda", driver_cuda = (12, 8)).endswith("cuda12.4.zip")
    # Nothing old enough for this driver: no CUDA bundle rather than one that fails to load.
    assert pick("Windows", "AMD64", "cuda", driver_cuda = (11, 8)) is None


def test_cuda_prefers_the_torch_line_so_its_runtime_can_be_reused():
    assert pick("Windows", "AMD64", "cuda", driver_cuda = (13, 4), prefer_cuda_major = 12).endswith(
        "cuda12.4.zip"
    )
    assert pick("Windows", "AMD64", "cuda", driver_cuda = (13, 4), prefer_cuda_major = 13).endswith(
        "cuda13.3.zip"
    )


def test_linux_cuda_prefers_the_multi_arch_build_over_colab():
    # The Unsloth fork adds a multi-arch cuda12.8 bundle next to upstream's sm_75-only Colab one.
    names = ASSETS + [f"audio-{TAG}-bin-ubuntu-x64-cuda12.8.tar.gz"]
    got = M.resolve_release_asset(names, system = "Linux", machine = "x86_64", accelerator = "cuda")
    assert got.endswith("bin-ubuntu-x64-cuda12.8.tar.gz")


def test_cudart_archive_matches_the_bundle_line():
    assert M.cudart_asset_for(ASSETS, f"audio-{TAG}-bin-windows-x64-cuda12.4.zip").endswith(
        "cudart-windows-x64-cuda12.4.zip"
    )
    assert M.cudart_asset_for(ASSETS, f"audio-{TAG}-bin-windows-x64-cuda13.3.zip").endswith(
        "cudart-windows-x64-cuda13.3.zip"
    )
    assert M.cudart_asset_for(ASSETS, f"audio-{TAG}-bin-ubuntu-x64-cuda12.8-colab.tar.gz") is None
    assert M.cudart_asset_for(ASSETS, f"audio-{TAG}-bin-windows-x64-cpu.zip") is None


@pytest.mark.parametrize(
    "version,major",
    [("2.10.0+cu130", 13), ("2.8.0+cu128", 12), ("2.5.1+cu124", 12), ("2.8.0", None)],
)
def test_torch_cuda_major(monkeypatch, version, major):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: version)
    assert M.torch_cuda_major() == major


@pytest.mark.parametrize(
    "smi,expected",
    [
        ("| NVIDIA-SMI 560.35   Driver Version: 560.35   CUDA Version: 12.6 |", (12, 6)),
        ("| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4 |", (13, 4)),
    ],
)
def test_driver_cuda_version_reads_both_spellings(monkeypatch, smi, expected):
    monkeypatch.setattr(M.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": smi})())
    assert M.nvidia_driver_cuda_version() == expected


# Install flow. Archives are real zips in tmp_path; the download copies them, the staged-server check
# is stubbed (its own tests below), and pins are a per-test dict.

FORK = M.DEFAULT_REPO
FORK_TAG = M.DEFAULT_TAG
CPU_ZIP = f"audio-{FORK_TAG}-bin-windows-x64-cpu-portable.zip"
# The real functions, before any test replaces them.
RESOLVE = M.resolve
SMOKE = M.smoke_test_staged_server


@pytest.fixture(autouse = True)
def _isolated_env(monkeypatch):
    for var in (
        "UNSLOTH_AUDIO_CPP_REPO",
        "UNSLOTH_AUDIO_CPP_TAG",
        "UNSLOTH_AUDIO_CPP_ACCELERATOR",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(M.core, "sleep_backoff", lambda *a, **k: None)


@pytest.fixture
def pins(monkeypatch):
    table = {}
    monkeypatch.setattr(M, "pinned_sha256", lambda repo, tag, asset: table.get((repo, tag, asset)))
    return table


def _sha(path):
    return M.core.sha256_file(path)


def _release(
    tmp_path,
    bundle_name,
    server_path = "bin/audiocpp_server.exe",
    tag = FORK_TAG,
    extra = (),
):
    archive = tmp_path / "src" / bundle_name
    archive.parent.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            server_path.replace("audiocpp_server.exe", M.SERVER_NAME),
            b"binary " + bundle_name.encode(),
        )
        zf.writestr("bin/README.txt", b"hi")
        for name, data in extra:
            zf.writestr(name, data)
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": bundle_name,
                "browser_download_url": str(archive),
                "digest": "sha256:" + _sha(archive),
            }
        ],
    }


def _pin_release(
    pins,
    release,
    repo = FORK,
):
    for asset in release["assets"]:
        pins[(repo, release["tag_name"], asset["name"])] = asset["digest"].partition(":")[2]


def _fake_download(url, dest):
    dest.write_bytes(Path(url).read_bytes())


def _stub_install_io(monkeypatch, version = "audio.cpp test\nbackends: cpu"):
    monkeypatch.setattr(M, "_download", _fake_download)
    monkeypatch.setattr(M, "smoke_test_staged_server", lambda server, **kw: version)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cpu")


def _install(
    monkeypatch,
    tmp_path,
    release,
    pins = None,
    repo = FORK,
    accelerator = "cpu",
    **kw,
):
    bundle = release["assets"][0]["name"]
    if pins is not None:
        _pin_release(pins, release, repo)
    monkeypatch.setattr(M, "resolve", lambda accel, token: (repo, release, bundle))
    _stub_install_io(monkeypatch)
    return M.install(install_dir = tmp_path / "audio.cpp", accelerator = accelerator, **kw)


def _tree_digest(root):
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_install_swaps_in_a_complete_tree_and_records_it(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    server = _install(monkeypatch, tmp_path, release, pins)
    root = tmp_path / "audio.cpp"
    assert server.is_file() and server.parent == root / "bin"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    assert record["backend"] == "cpu"
    assert record["server_relpath"] == f"bin/{M.SERVER_NAME}"
    assert record["published_repo"] == FORK and record["release_tag"] == FORK_TAG
    assert record["server_sha256"] == _sha(server)
    assert record["server_version"].startswith("audio.cpp test")
    assert (
        record["accelerator"],
        record["accelerator_request"],
        record["detected_accelerator"],
    ) == ("cpu", "cpu", None)
    assert (root / M.OWNERSHIP_MARKER).is_file()
    # No staging directory survives, and the install lock is prebuilt_core's.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".audio.cpp-staging-")]
    assert M.core.install_lock_path(root) == tmp_path / ".audio.cpp.install.lock"


# G2: a no-op re-run does no network work


def test_rerun_on_the_pinned_install_makes_no_release_lookup(monkeypatch, tmp_path, pins, capsys):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins, accelerator = "auto")
    before = _tree_digest(tmp_path / "audio.cpp")
    monkeypatch.setattr(
        M, "resolve", lambda *a: pytest.fail("resolve must not run on a matching install")
    )
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out
    assert _tree_digest(tmp_path / "audio.cpp") == before


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda root, rec: rec.update(release_tag = "v0.8.1-unsloth.1"), id = "older-tag"),
        pytest.param(lambda root, rec: rec.update(asset_sha256 = "0" * 64), id = "unpinned-digest"),
        pytest.param(lambda root, rec: rec.update(detected_accelerator = "cuda"), id = "host-changed"),
        pytest.param(
            lambda root, rec: (root / rec["server_relpath"]).write_bytes(b"tampered"),
            id = "server-changed",
        ),
    ],
)
def test_a_record_that_no_longer_fits_still_looks_up(monkeypatch, tmp_path, pins, mutate):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins, accelerator = "auto")
    root = tmp_path / "audio.cpp"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    mutate(root, record)
    (root / M.INSTALL_RECORD).write_text(json.dumps(record))
    asked = []
    monkeypatch.setattr(
        M, "resolve", lambda accel, token: asked.append(accel) or (FORK, release, CPU_ZIP)
    )
    M.install(install_dir = root, accelerator = "auto")
    assert asked == ["cpu"]


def test_an_explicit_request_does_not_match_an_auto_install(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins, accelerator = "auto")
    asked = []
    monkeypatch.setattr(
        M, "resolve", lambda accel, token: asked.append(accel) or (FORK, None, None)
    )
    with pytest.raises(RuntimeError, match = "No prebuilt"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "vulkan")
    assert asked == ["vulkan"]


def test_tracking_latest_always_looks_up(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins)
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", "")
    root = tmp_path / "audio.cpp"
    assert M._pinned_install_matches(root, M.read_install_record(root), "cpu", None) is None


# G2/AC3: a lookup that cannot answer keeps a complete install


def _offline(monkeypatch):
    def fetch(
        repo,
        tag,
        token,
        timeout = 30.0,
    ):
        raise OSError("network is unreachable")

    monkeypatch.setattr(M, "_fetch_release", fetch)


def _installed_then_stale(
    monkeypatch,
    tmp_path,
    pins,
    accelerator = "auto",
):
    """A complete install whose record no longer names the pinned tag, so the run must look up."""
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins, accelerator = accelerator)
    root = tmp_path / "audio.cpp"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    record["release_tag"] = "v0.8.1-unsloth.1"
    (root / M.INSTALL_RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(M, "resolve", RESOLVE)
    return root


def test_offline_rerun_keeps_a_complete_install(monkeypatch, tmp_path, pins, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path, pins)
    before = _tree_digest(root)
    _offline(monkeypatch)
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    out = capsys.readouterr().out
    assert "audio.cpp: keeping the existing complete install" in out.splitlines()
    assert _tree_digest(root) == before


def test_rate_limited_rerun_keeps_a_complete_install(monkeypatch, tmp_path, pins, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path, pins)

    def fetch(
        repo,
        tag,
        token,
        timeout = 30.0,
    ):
        raise M.GitHubRateLimited("rate limited")

    monkeypatch.setattr(M, "_fetch_release", fetch)
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    assert M.KEPT_EXISTING_LINE in capsys.readouterr().out


def test_offline_without_an_install_fails(monkeypatch, tmp_path, capsys):
    _stub_install_io(monkeypatch)
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_FAILED
    assert "no audio.cpp release lookup answered" in capsys.readouterr().err


def test_offline_keeps_nothing_for_a_tampered_server(monkeypatch, tmp_path, pins, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path, pins)
    (root / "bin" / M.SERVER_NAME).write_bytes(b"truncated")
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(root)]) == M.EXIT_FAILED
    assert M.KEPT_EXISTING_LINE not in capsys.readouterr().out


def test_offline_does_not_keep_an_install_for_another_explicit_accelerator(
    monkeypatch, tmp_path, pins
):
    root = _installed_then_stale(monkeypatch, tmp_path, pins)
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(root), "--accelerator", "vulkan"]) == M.EXIT_FAILED
    assert M.main(["--install-dir", str(root), "--force"]) == M.EXIT_FAILED


# G1: main() leaves "auto" to install(), so the CPU fallback runs from the real CLI


def _resolve_recording(release, covers = ("cpu",)):
    asked = []

    def resolve(accel, token):
        asked.append(accel)
        if accel in covers:
            return FORK, release, release["assets"][0]["name"]
        return FORK, None, None

    return asked, resolve


def test_main_auto_detected_gpu_without_a_bundle_installs_the_cpu_build(
    monkeypatch, tmp_path, pins
):
    release = _release(tmp_path, CPU_ZIP)
    _pin_release(pins, release)
    asked, resolve = _resolve_recording(release)
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cuda")
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert asked == ["cuda", "cpu"]
    record = json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())
    assert (record["backend"], record["accelerator_request"], record["detected_accelerator"]) == (
        "cpu",
        "auto",
        "cuda",
    )


def test_main_explicit_gpu_request_is_never_downgraded(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release)
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(
        M, "detect_accelerator", lambda: pytest.fail("explicit request must not detect")
    )
    assert (
        M.main(["--install-dir", str(tmp_path / "audio.cpp"), "--accelerator", "cuda"])
        == M.EXIT_FAILED
    )
    assert asked == ["cuda"]
    assert not (tmp_path / "audio.cpp").exists()


def test_print_asset_resolves_like_install(monkeypatch, tmp_path, capsys):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cuda")
    assert M.main(["--print-asset"]) == M.EXIT_OK
    assert asked == ["cuda", "cpu"]
    assert capsys.readouterr().out.strip().endswith(CPU_ZIP)


# G12: UNSLOTH_AUDIO_CPP_ACCELERATOR


def test_accelerator_env_is_an_explicit_request(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release, covers = ("cpu", "cuda"))
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    _pin_release(pins, release)
    monkeypatch.setattr(
        M, "detect_accelerator", lambda: pytest.fail("an explicit accelerator must not detect")
    )
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_ACCELERATOR", "CUDA")
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert asked == ["cuda"]
    assert (
        json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())["accelerator_request"]
        == "cuda"
    )


def test_an_unknown_accelerator_env_falls_back_to_auto(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_ACCELERATOR", "rocm")
    assert M._accelerator_from_env() == "auto"
    assert "ignoring UNSLOTH_AUDIO_CPP_ACCELERATOR" in capsys.readouterr().err


# G11: release ladder and digest pins


def test_default_ladder_is_the_pinned_fork_tag_then_its_upstream_release(monkeypatch):
    tried = []
    monkeypatch.setattr(
        M, "_fetch_release", lambda repo, tag, token: tried.append((repo, tag)) or None
    )
    monkeypatch.setattr(M, "nvidia_driver_cuda_version", lambda: None)
    assert M.resolve("cpu", None) == (FORK, None, None)
    # No "latest" rungs: a pinned install never drifts to an untested release.
    assert tried == [
        (M.DEFAULT_REPO, M.DEFAULT_TAG),
        (M.UPSTREAM_FALLBACK_REPO, M.UPSTREAM_FALLBACK_TAG),
    ]


def test_latest_is_tried_only_when_the_user_asks_for_it(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", "")
    assert M._release_ladder() == [(M.DEFAULT_REPO, None), (M.UPSTREAM_FALLBACK_REPO, None)]
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_REPO", "someone/audio.cpp")
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", "v1")
    assert M._release_ladder() == [("someone/audio.cpp", "v1")]


def test_an_unpinned_asset_is_refused_by_default(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    with pytest.raises(RuntimeError, match = "no sha256 pinned"):
        _install(monkeypatch, tmp_path, release)
    assert not (tmp_path / "audio.cpp").exists()


def test_an_unpinned_asset_is_allowed_for_a_release_the_user_picked(monkeypatch, tmp_path, pins):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", FORK_TAG)
    release = _release(tmp_path, CPU_ZIP)
    assert _install(monkeypatch, tmp_path, release).is_file()


def test_a_github_digest_that_disagrees_with_the_pin_is_refused(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    _pin_release(pins, release)
    release["assets"][0]["digest"] = "sha256:" + "1" * 64
    with pytest.raises(RuntimeError, match = "pins"):
        _install(monkeypatch, tmp_path, release)


def test_a_download_that_does_not_hash_to_the_pin_is_refused(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    pins[(FORK, FORK_TAG, CPU_ZIP)] = "0" * 64
    release["assets"][0]["digest"] = None
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        _install(monkeypatch, tmp_path, release)
    assert not (tmp_path / "audio.cpp").exists()


def test_the_pins_file_covers_both_default_releases():
    table = M.load_pins()
    for repo, tag in (
        (M.DEFAULT_REPO, M.DEFAULT_TAG),
        (M.UPSTREAM_FALLBACK_REPO, M.UPSTREAM_FALLBACK_TAG),
    ):
        names = list(table[repo][tag])
        # Stale after a tag bump until `--write-pins` regenerates it.
        assert names and all(f"audio-{tag}-" in n for n in names), (repo, tag)
        assert all(M.core.normalize_sha256_digest(table[repo][tag][n]) for n in names)
        for system, machine, accel in (
            ("Windows", "AMD64", "cpu"),
            ("Windows", "AMD64", "cuda"),
            ("Linux", "x86_64", "cpu"),
            ("Linux", "x86_64", "cuda"),
            ("Darwin", "arm64", "metal"),
        ):
            chosen = M.resolve_release_asset(
                names, system = system, machine = machine, accelerator = accel
            )
            assert chosen, (repo, tag, system, accel)
            if system == "Windows" and accel == "cuda":
                assert M.cudart_asset_for(names, chosen) in names


def test_write_pins_regenerates_from_the_release_digests(monkeypatch, tmp_path):
    releases = {
        (M.DEFAULT_REPO, M.DEFAULT_TAG): [
            {
                "name": f"audio-{M.DEFAULT_TAG}-bin-windows-x64-cpu.zip",
                "digest": "sha256:" + "b" * 64,
            },
            {
                "name": f"audio-{M.DEFAULT_TAG}-cudart-windows-x64-cuda12.4.zip",
                "digest": "sha256:" + "c" * 64,
            },
            {
                "name": f"audio-{M.DEFAULT_TAG}-bin-macos-arm64-metal.tar.gz",
                "digest": "sha256:" + "a" * 64,
            },
            {"name": "framework.tar.gz", "digest": None},
        ],
        (M.UPSTREAM_FALLBACK_REPO, M.UPSTREAM_FALLBACK_TAG): [
            {
                "name": f"audio-{M.UPSTREAM_FALLBACK_TAG}-bin-windows-x64-cpu.zip",
                "digest": "sha256:" + "d" * 64,
            },
        ],
    }
    monkeypatch.setattr(
        M,
        "_fetch_release",
        lambda repo, tag, token: {"tag_name": tag, "assets": releases[(repo, tag)]},
    )
    out = M.write_pins(tmp_path / "pins.json")
    first = out.read_bytes()
    assert M.write_pins(tmp_path / "pins.json").read_bytes() == first
    table = M.load_pins(out)
    fork = table[M.DEFAULT_REPO][M.DEFAULT_TAG]
    assert list(fork) == sorted(fork) and "framework.tar.gz" not in fork
    assert fork[f"audio-{M.DEFAULT_TAG}-cudart-windows-x64-cuda12.4.zip"] == "c" * 64
    releases[(M.DEFAULT_REPO, M.DEFAULT_TAG)][0]["digest"] = None
    with pytest.raises(RuntimeError, match = "no sha256 digest"):
        M.write_pins(tmp_path / "pins.json")


# G3(b): the staged server must start before it replaces anything


class _Ran:
    def __init__(
        self,
        returncode,
        stdout = "",
        stderr = "",
    ):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_a_staged_server_that_cannot_start_keeps_the_old_tree(monkeypatch, tmp_path, pins, capsys):
    good = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, good, pins)
    root = tmp_path / "audio.cpp"
    before = _tree_digest(root)
    newer = _release(tmp_path, f"audio-{FORK_TAG}-bin-windows-x64-cpu.zip")
    _pin_release(pins, newer)
    monkeypatch.setattr(
        M, "resolve", lambda accel, token: (FORK, newer, newer["assets"][0]["name"])
    )
    monkeypatch.setattr(M, "smoke_test_staged_server", SMOKE)
    loader_error = (
        "audiocpp_server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found"
    )
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: _Ran(127, stderr = loader_error))
    assert M.main(["--install-dir", str(root), "--accelerator", "cpu", "--force"]) == M.EXIT_FAILED
    err = capsys.readouterr().err
    assert "GLIBC_2.38" in err and "keeping the existing install" in err
    assert _tree_digest(root) == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".audio.cpp-staging-")]


def test_staged_server_check_puts_the_bundle_dir_first_and_records_the_version(
    monkeypatch, tmp_path
):
    server = tmp_path / "bin" / M.SERVER_NAME
    server.parent.mkdir()
    server.write_bytes(b"x")
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return _Ran(
            0, stdout = "audio.cpp 0.8.2\nbackends: cpu,cuda\n" if cmd[1] == "--version" else "usage"
        )

    monkeypatch.setattr(M.subprocess, "run", run)
    extra = tmp_path / "torch-lib"
    assert (
        SMOKE(server, backend = "cuda", extra_dirs = [extra]) == "audio.cpp 0.8.2\nbackends: cpu,cuda"
    )
    assert [c[0][1] for c in calls] == ["--help", "--version"]
    assert calls[0][1]["timeout"] == M.SMOKE_TIMEOUT_SECONDS
    var = (
        "PATH"
        if sys.platform == "win32"
        else "DYLD_LIBRARY_PATH"
        if sys.platform == "darwin"
        else "LD_LIBRARY_PATH"
    )
    assert calls[0][1]["env"][var].split(M.os.pathsep)[:2] == [str(server.parent), str(extra)]


def test_a_cuda_bundle_is_checked_with_the_wheel_cuda_runtime_on_the_loader_path(
    monkeypatch, tmp_path, pins
):
    # The Linux CUDA server links libcublas/libcudart/libcufft/libnccl dynamically: without torch's
    # nvidia-* wheel dirs the check would refuse every CUDA install.
    import types

    fake = types.ModuleType("install_llama_prebuilt")
    fake.python_runtime_dirs = lambda: ["/venv/nvidia/cublas/lib"]
    monkeypatch.setitem(sys.modules, "install_llama_prebuilt", fake)
    monkeypatch.setattr(M.sys, "platform", "linux")
    assert M._cuda_runtime_dirs("cuda") == ["/venv/nvidia/cublas/lib"]
    assert M._cuda_runtime_dirs("cpu") == []
    monkeypatch.setattr(M.sys, "platform", "darwin")
    assert M._cuda_runtime_dirs("cuda") == []


def test_staged_server_check_times_out(monkeypatch, tmp_path):
    def run(cmd, **kw):
        raise M.subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(M.subprocess, "run", run)
    with pytest.raises(RuntimeError, match = "did not exit within 30s"):
        SMOKE(tmp_path / M.SERVER_NAME, backend = "cpu")


def test_a_cuda_build_is_not_refused_for_a_missing_driver_it_gets_at_run_time(
    monkeypatch, tmp_path
):
    # A Docker image build sees no GPU; libcuda.so.1 comes with the driver when the container runs.
    monkeypatch.setattr(
        M.subprocess,
        "run",
        lambda *a, **k: _Ran(127, stderr = "error while loading shared libraries: libcuda.so.1"),
    )
    monkeypatch.setattr(M, "nvidia_driver_cuda_version", lambda: None)
    assert SMOKE(tmp_path / M.SERVER_NAME, backend = "cuda") is None
    with pytest.raises(RuntimeError, match = "libcuda"):
        SMOKE(tmp_path / M.SERVER_NAME, backend = "cpu")
    monkeypatch.setattr(M, "nvidia_driver_cuda_version", lambda: (12, 8))
    with pytest.raises(RuntimeError, match = "libcuda"):
        SMOKE(tmp_path / M.SERVER_NAME, backend = "cuda")


@pytest.mark.skipif(
    sys.platform == "win32", reason = "needs an executable shell script as the server"
)
def test_staged_server_check_runs_the_real_binary(tmp_path):
    server = tmp_path / M.SERVER_NAME
    server.write_text(
        '#!/bin/sh\n[ "$1" = --version ] && echo "audio.cpp fake" && exit 0\necho "loader: nope" >&2\nexit 127\n'
    )
    server.chmod(0o755)
    with pytest.raises(RuntimeError, match = "loader: nope"):
        SMOKE(server, backend = "cpu")
    server.write_text('#!/bin/sh\n[ "$1" = --version ] && echo "audio.cpp fake"\nexit 0\n')
    assert SMOKE(server, backend = "cpu") == "audio.cpp fake"


# G17: install lock and retried downloads


def test_a_held_install_lock_reports_busy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(M.core, "INSTALL_LOCK_TIMEOUT_SECONDS", 0.3)
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", lambda *a: pytest.fail("must not resolve without the lock"))
    target = tmp_path / "audio.cpp"
    with M.core.install_lock(M.core.install_lock_path(target)):
        assert M.main(["--install-dir", str(target), "--accelerator", "cpu"]) == M.EXIT_BUSY
    assert "another audio.cpp install is running" in capsys.readouterr().err


def test_downloads_are_retried(monkeypatch, tmp_path):
    import io
    import urllib.error

    attempts = []

    class Response(io.BytesIO):
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(
            self,
            request,
            timeout = None,
        ):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise urllib.error.URLError("connection reset")
            return Response(b"data")

    monkeypatch.setattr(M, "_URL_OPENER", Opener(), raising = False)
    M._download("https://github.com/o/r/releases/download/t/a.zip", tmp_path / "a.zip")
    assert (tmp_path / "a.zip").read_bytes() == b"data" and len(attempts) == 2


def test_release_lookups_are_retried(monkeypatch):
    import io
    import urllib.error

    attempts = []

    def urlopen(req, timeout = None):
        attempts.append(req.full_url)
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary failure in name resolution")
        return io.BytesIO(b'{"tag_name": "t", "assets": []}')

    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    assert M._fetch_release("o/r", "t", None)["tag_name"] == "t"
    assert len(attempts) == 3


# Carried over from the first installer


def test_refuses_to_replace_a_directory_it_does_not_own(monkeypatch, tmp_path, pins):
    foreign = tmp_path / "audio.cpp"
    foreign.mkdir()
    (foreign / "mine.txt").write_text("user data")
    release = _release(tmp_path, CPU_ZIP)
    with pytest.raises(RuntimeError, match = "not an Unsloth-managed directory"):
        _install(monkeypatch, tmp_path, release, pins)
    assert (foreign / "mine.txt").read_text() == "user data"


def test_bundle_without_a_server_leaves_the_old_tree(monkeypatch, tmp_path, pins):
    good = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, good, pins)
    bad = _release(
        tmp_path, f"audio-{FORK_TAG}-bin-windows-x64-cpu.zip", server_path = "bin/other.exe"
    )
    with pytest.raises(RuntimeError, match = "contains no"):
        _install(monkeypatch, tmp_path, bad, pins, force = True)
    assert (tmp_path / "audio.cpp" / "bin" / M.SERVER_NAME).is_file()


def test_auto_detected_gpu_without_a_bundle_installs_the_cpu_build(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    _pin_release(pins, release)
    asked, resolve = _resolve_recording(release)
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cuda")
    M.install(install_dir = tmp_path / "audio.cpp", accelerator = "auto")
    assert asked == ["cuda", "cpu"]
    assert json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())["backend"] == "cpu"


def test_an_explicit_gpu_request_is_never_downgraded(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "resolve", lambda accel, token: (FORK, None, None))
    with pytest.raises(RuntimeError, match = "No prebuilt"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cuda")


def test_intel_mac_bundle_runs_on_the_cpu_backend():
    assert M._backend_of(f"audio-{TAG}-bin-macos-x64-metal.tar.gz") == "cpu"
    assert M._backend_of(f"audio-{TAG}-bin-macos-arm64-metal.tar.gz") == "metal"


def test_record_notes_a_static_espeak_build(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP, extra = [("bin/espeak-ng-data.bin", b"data")])
    _install(monkeypatch, tmp_path, release, pins)
    assert json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())["espeak"] is True


def test_a_dir_holding_only_the_old_child_home_is_adopted(monkeypatch, tmp_path, pins):
    leftover = tmp_path / "audio.cpp" / ".child_home" / "AppData"
    leftover.mkdir(parents = True)
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins)
    assert (tmp_path / "audio.cpp" / M.OWNERSHIP_MARKER).is_file()
    assert not (tmp_path / "audio.cpp" / ".child_home").exists()


def test_retired_trees_are_swept_on_the_next_run(monkeypatch, tmp_path, pins):
    stale = tmp_path / "audio.cpp.old-1234"
    stale.mkdir()
    (stale / M.OWNERSHIP_MARKER).touch()
    foreign = tmp_path / "audio.cpp.old-user"
    foreign.mkdir()
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins)
    assert not stale.exists() and foreign.exists()


def test_a_running_server_reports_busy_before_downloading(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins)
    monkeypatch.setattr(M, "_server_in_use", lambda server: True)
    downloaded = []
    monkeypatch.setattr(M, "_download", lambda *a, **k: downloaded.append(a))
    with pytest.raises(PermissionError):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cpu", force = True)
    assert downloaded == []


def test_up_to_date_check_notices_a_lost_cuda_runtime(monkeypatch):
    record = {
        "backend": "cuda",
        "os": "windows",
        "cudart_asset": None,
        "asset": f"audio-{TAG}-bin-windows-x64-cuda13.3.zip",
    }
    monkeypatch.setattr(M, "torch_provides_cuda_runtime", lambda major: False)
    assert M._cuda_runtime_satisfied(record) is False
    monkeypatch.setattr(M, "torch_provides_cuda_runtime", lambda major: major == 13)
    assert M._cuda_runtime_satisfied(record) is True
    assert M._cuda_runtime_satisfied({**record, "cudart_asset": "x.zip"}) is True
    assert M._cuda_runtime_satisfied({"backend": "cpu"}) is True


def test_tar_extraction_without_filters_still_refuses_traversal(monkeypatch, tmp_path):
    import io
    import tarfile

    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    monkeypatch.delattr(tarfile, "data_filter", raising = False)
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match = "unsafe path"):
        M._extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_an_upstream_fallback_install_asks_for_the_fork_again(monkeypatch, tmp_path, pins):
    # The fork lookup failed on the run that installed upstream; the next run must not call that a match.
    up_zip = f"audio-{M.UPSTREAM_FALLBACK_TAG}-bin-windows-x64-cpu-portable.zip"
    release = _release(tmp_path, up_zip, tag = M.UPSTREAM_FALLBACK_TAG)
    _install(monkeypatch, tmp_path, release, pins, repo = M.UPSTREAM_FALLBACK_REPO)
    root = tmp_path / "audio.cpp"
    record = M.read_install_record(root)
    assert record["published_repo"] == M.UPSTREAM_FALLBACK_REPO
    assert M._pinned_install_matches(root, record, "cpu", None) is None


def test_the_accelerator_flag_takes_what_setup_forwards(monkeypatch, tmp_path, pins):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release, covers = ("cpu", "cuda"))
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    _pin_release(pins, release)
    root = str(tmp_path / "audio.cpp")
    assert M.main(["--install-dir", root, "--accelerator", " CUDA "]) == M.EXIT_OK
    assert asked == ["cuda"]


def test_a_killed_install_leaves_no_staging_dir_behind(monkeypatch, tmp_path, pins):
    stranded = tmp_path / ".audio.cpp-staging-killed"
    stranded.mkdir()
    (stranded / "partial.zip").write_bytes(b"x" * 16)
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, pins)
    assert not stranded.exists()


def test_the_staged_server_never_sees_secrets(monkeypatch, tmp_path):
    for name in ("GH_TOKEN", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "secret")
    env = M._loader_env(tmp_path / "bin" / M.SERVER_NAME)
    assert not {"GH_TOKEN", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert (
        str(tmp_path / "bin")
        in env[
            "PATH"
            if M.sys.platform == "win32"
            else ("DYLD_LIBRARY_PATH" if M.sys.platform == "darwin" else "LD_LIBRARY_PATH")
        ]
    )
