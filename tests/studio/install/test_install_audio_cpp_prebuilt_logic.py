# SPDX-License-Identifier: AGPL-3.0-only
# Logic tests for studio/install_audio_cpp_prebuilt.py -- the self-contained audiocpp_server installer.
# No network/GPU: asset lists are the real upstream v0.8.2 release names, archives are built in a tmp dir,
# and release lookups are served from trimmed copies of the published v0.9.1-unsloth.1 manifest and index.

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
# is stubbed (its own tests below), and every release carries its own sha256 index, as the fork's do.

FORK = M.DEFAULT_REPO
FORK_TAG = "v0.9.1-unsloth.1"
OLDER_TAG = "v0.9.0-unsloth.1"
HOTFIX_TAG = "v0.8.2-audio8-perf-hotfix-unsloth.1"
CPU_ZIP = f"audio-{FORK_TAG}-bin-windows-x64-cpu-portable.zip"
OLDER_ZIP = f"audio-{OLDER_TAG}-bin-windows-x64-cpu-portable.zip"
# The real functions, before any test replaces them.
RESOLVE = M.resolve
SMOKE = M.smoke_test_staged_server
LATEST_TAG = M._latest_tag_via_download_host


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
    monkeypatch.setattr(M, "_LATEST_TAG_MEMO", {})
    monkeypatch.setattr(M, "_RELEASE_INDEX_MEMO", {})
    # /releases/latest names FORK_TAG unless a test says otherwise.
    heads = []
    monkeypatch.setattr(
        M, "_latest_tag_via_download_host", lambda repo: heads.append(repo) or FORK_TAG
    )
    # Nothing reaches the network: a seam a test forgot to replace fails the test here.
    hits = []

    class _NoNetwork:
        def open(
            self,
            request,
            timeout = None,
        ):
            hits.append(getattr(request, "full_url", request))
            raise OSError("network disabled in tests")

    def _no_urlopen(request, timeout = None):
        hits.append(getattr(request, "full_url", request))
        raise OSError("network disabled in tests")

    monkeypatch.setattr(M.core, "_URL_OPENER", _NoNetwork())
    monkeypatch.setattr(M.urllib.request, "urlopen", _no_urlopen)
    yield heads
    assert hits == [], f"unexpected network access: {hits}"


@pytest.fixture
def heads(_isolated_env):
    """Repos the stubbed ``/releases/latest`` HEAD was asked for."""
    return _isolated_env


def _sha(path):
    return M.core.sha256_file(path)


def _release(
    tmp_path,
    bundle_name,
    server_path = "bin/audiocpp_server.exe",
    tag = FORK_TAG,
    extra = (),
    min_os = None,
):
    """What ``_release_index`` returns for a release whose one bundle is a real zip in tmp_path,
    with the index and manifest digests both naming that zip."""
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
    digest = _sha(archive)
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": bundle_name,
                "browser_download_url": str(archive),
                "manifest_sha256": digest,
                "min_os": min_os,
            }
        ],
        "checksums": {bundle_name: digest},
    }


def _names_release(tag, names):
    """A release index with these asset names and no archives behind them (lookup-only tests)."""
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": n,
                "browser_download_url": f"https://github.com/{FORK}/releases/download/{tag}/{n}",
                "manifest_sha256": None,
                "min_os": None,
            }
            for n in names
        ],
        "checksums": {n: "0" * 64 for n in names},
    }


def _fake_download(url, dest):
    dest.write_bytes(Path(url).read_bytes())


def _stub_install_io(monkeypatch, version = "audio.cpp test\nbackends: cpu"):
    monkeypatch.setattr(M, "_download", _fake_download)
    monkeypatch.setattr(M, "smoke_test_staged_server", lambda server, **kw: version)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cpu")
    monkeypatch.setattr(M, "_linux_cuda_runtime_available", lambda major: True)


def _install(
    monkeypatch,
    tmp_path,
    release,
    repo = FORK,
    accelerator = "cpu",
    **kw,
):
    bundle = release["assets"][0]["name"]
    monkeypatch.setattr(M, "resolve", lambda accel, token, tag = None: (repo, release, bundle))
    _stub_install_io(monkeypatch)
    return M.install(install_dir = tmp_path / "audio.cpp", accelerator = accelerator, **kw)


def _tree_digest(root):
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _windows_host(monkeypatch):
    # The real resolver picks by host; the fixture archives are Windows zips.
    monkeypatch.setattr(M.platform, "system", lambda: "Windows")
    monkeypatch.setattr(M.platform, "machine", lambda: "AMD64")


def _real_head(monkeypatch, answer):
    """The real memoized ``_latest_tag_via_download_host`` over a counted prebuilt_core HEAD seam."""
    calls = []

    def head(repo):
        calls.append(repo)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(M, "_latest_tag_via_download_host", LATEST_TAG)
    monkeypatch.setattr(M, "_download_host_latest_release_tag", head, raising = False)
    return calls


def _serve_index(monkeypatch, releases):
    """Replace ``_release_index`` with a ``{(repo, tag): release}`` table; returns what was asked."""
    asked = []

    def index(
        repo,
        tag,
        *,
        token = None,
        allow_unindexed = False,
    ):
        asked.append((repo, tag, allow_unindexed))
        return releases.get((repo, tag))

    monkeypatch.setattr(M, "_release_index", index)
    return asked


def _lookup_fails(monkeypatch, exc):
    """prebuilt_core's release lookup (reached through _OPS) raises ``exc``."""

    def fetch(repo, *, published_release_tag):
        raise exc

    monkeypatch.setattr(M, "fetch_release_for_install", fetch, raising = False)


def test_install_swaps_in_a_complete_tree_and_records_it(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    server = _install(monkeypatch, tmp_path, release)
    root = tmp_path / "audio.cpp"
    assert server.is_file() and server.parent == root / "bin"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    assert record["backend"] == "cpu"
    assert record["server_relpath"] == f"bin/{M.SERVER_NAME}"
    assert record["published_repo"] == FORK and record["release_tag"] == FORK_TAG
    assert record["asset_sha256"] == release["checksums"][CPU_ZIP]
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


# G2: a no-op re-run does no download, one HEAD when tracking the newest, nothing when pinned


def test_rerun_on_the_newest_install_asks_one_head_and_nothing_else(monkeypatch, tmp_path, capsys):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, accelerator = "auto")
    before = _tree_digest(tmp_path / "audio.cpp")
    calls = _real_head(monkeypatch, FORK_TAG)
    monkeypatch.setattr(
        M, "resolve", lambda *a, **k: pytest.fail("resolve must not run on a matching install")
    )
    monkeypatch.setattr(
        M, "_release_index", lambda *a, **k: pytest.fail("no release lookup expected")
    )
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out
    assert calls == [FORK]
    assert _tree_digest(tmp_path / "audio.cpp") == before


@pytest.mark.parametrize("how", ["env", "flag"])
def test_rerun_on_a_pinned_install_makes_no_network_call(monkeypatch, tmp_path, capsys, how):
    argv = ["--install-dir", str(tmp_path / "audio.cpp")]
    if how == "env":
        monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", FORK_TAG)
    else:
        argv += ["--published-release-tag", FORK_TAG]
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP), accelerator = "auto")
    monkeypatch.setattr(
        M, "_latest_tag_via_download_host", lambda repo: pytest.fail("a pin asks no HEAD")
    )
    monkeypatch.setattr(M, "resolve", lambda *a, **k: pytest.fail("no lookup expected"))
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    assert M.main(argv) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda root, rec: rec.update(release_tag = OLDER_TAG), id = "older-tag"),
        pytest.param(
            lambda root, rec: rec.update(published_repo = "0xShug0/audio.cpp"), id = "other-repo"
        ),
        pytest.param(lambda root, rec: rec.update(release_tag = "b123"), id = "unparseable-tag"),
        pytest.param(lambda root, rec: rec.update(detected_accelerator = "cuda"), id = "host-changed"),
        pytest.param(
            lambda root, rec: (root / rec["server_relpath"]).write_bytes(b"tampered"),
            id = "server-changed",
        ),
    ],
)
def test_a_record_that_no_longer_fits_still_looks_up(monkeypatch, tmp_path, mutate):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, accelerator = "auto")
    root = tmp_path / "audio.cpp"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    mutate(root, record)
    (root / M.INSTALL_RECORD).write_text(json.dumps(record))
    asked = []
    monkeypatch.setattr(
        M,
        "resolve",
        lambda accel, token, tag = None: asked.append(accel) or (FORK, release, CPU_ZIP),
    )
    M.install(install_dir = root, accelerator = "auto")
    assert asked == ["cpu"]


def test_an_explicit_request_does_not_match_an_auto_install(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, accelerator = "auto")
    asked = []
    monkeypatch.setattr(
        M, "resolve", lambda accel, token, tag = None: asked.append(accel) or (FORK, None, None)
    )
    with pytest.raises(RuntimeError, match = "No prebuilt"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "vulkan")
    assert asked == ["vulkan"]


def test_a_newer_latest_release_is_looked_up_once_and_installed(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, _release(tmp_path / "old", OLDER_ZIP, tag = OLDER_TAG))
    monkeypatch.setattr(M, "resolve", RESOLVE)
    _windows_host(monkeypatch)
    new = _release(tmp_path, CPU_ZIP)
    asked = _serve_index(monkeypatch, {(FORK, FORK_TAG): new})
    calls = _real_head(monkeypatch, FORK_TAG)
    server = M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cpu")
    # One HEAD: the re-run check asked it, and the lookup reused its answer as the tag.
    assert calls == [FORK]
    assert asked == [(FORK, FORK_TAG, False)]
    record = M.read_install_record(server.parent.parent)
    assert (record["published_repo"], record["release_tag"], record["asset"]) == (
        FORK,
        FORK_TAG,
        CPU_ZIP,
    )


# A6: an intact install is never replaced by an older release a lookup names


def test_an_older_latest_never_replaces_a_newer_install(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP), accelerator = "auto")
    root = tmp_path / "audio.cpp"
    before = _tree_digest(root)
    calls = _real_head(monkeypatch, HOTFIX_TAG)
    monkeypatch.setattr(M, "resolve", lambda *a, **k: pytest.fail("no lookup expected"))
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out
    assert calls == [FORK]
    assert _tree_digest(root) == before


def test_a_lookup_naming_an_older_release_does_not_downgrade(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP), accelerator = "auto")
    root = tmp_path / "audio.cpp"
    before = _tree_digest(root)
    # No HEAD answer, so the full lookup runs, and the API's newest is an older release.
    _real_head(monkeypatch, M.urllib.error.URLError("offline"))
    older = _release(tmp_path / "older", OLDER_ZIP, tag = OLDER_TAG)
    monkeypatch.setattr(M, "resolve", lambda accel, token, tag = None: (FORK, older, OLDER_ZIP))
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out
    assert _tree_digest(root) == before


def test_a_pinned_older_release_is_installed(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP))
    root = tmp_path / "audio.cpp"
    older = _release(tmp_path / "older", OLDER_ZIP, tag = OLDER_TAG)
    asked = []
    monkeypatch.setattr(
        M,
        "resolve",
        lambda accel, token, tag = None: asked.append(tag) or (FORK, older, OLDER_ZIP),
    )
    assert (
        M.main(
            [
                "--install-dir",
                str(root),
                "--accelerator",
                "cpu",
                "--published-release-tag",
                OLDER_TAG,
            ]
        )
        == M.EXIT_OK
    )
    assert asked == [OLDER_TAG]
    assert M.read_install_record(root)["release_tag"] == OLDER_TAG


def test_keep_newer_keeps_a_newer_install_over_the_pinned_offer(monkeypatch, tmp_path, capsys):
    # The in-app update pins the release it offered; a setup run that installed a newer one under the lock first wins.
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP))
    root = tmp_path / "audio.cpp"
    before = _tree_digest(root)
    monkeypatch.setattr(M, "resolve", lambda *a, **k: pytest.fail("no lookup expected"))
    monkeypatch.setattr(M, "_download", lambda *a: pytest.fail("no download expected"))
    capsys.readouterr()
    argv = [
        "--install-dir",
        str(root),
        "--accelerator",
        "cpu",
        "--published-release-tag",
        OLDER_TAG,
    ]
    assert M.main([*argv, "--keep-newer"]) == M.EXIT_OK
    assert f"audio.cpp: already matches {CPU_ZIP}" in capsys.readouterr().out
    assert _tree_digest(root) == before


@pytest.mark.parametrize(
    "tag, key",
    [
        ("v0.9.1-unsloth.1", (0, 9, 1, 1)),
        ("v0.9.0-unsloth.1", (0, 9, 0, 1)),
        ("v0.9.0", (0, 9, 0, 0)),
        ("v0.8.2-audio8-perf-hotfix-unsloth.1", (0, 8, 2, 1)),
        ("0.10", (0, 10, 0, 0)),
        ("b123", None),
        ("", None),
        (None, None),
        ("latest", None),
    ],
)
def test_release_version_key(tag, key):
    assert M.release_version_key(tag) == key


@pytest.mark.parametrize(
    "installed, latest, current",
    [
        (FORK_TAG, FORK_TAG, True),
        (FORK_TAG, OLDER_TAG, True),
        (FORK_TAG, HOTFIX_TAG, True),
        (OLDER_TAG, FORK_TAG, False),
        # Upstream's v0.9.0 is older than the fork's packaging of it.
        ("v0.9.0", OLDER_TAG, False),
        # Unparseable never counts as newer.
        ("b123", FORK_TAG, False),
        (FORK_TAG, "b123", False),
        (FORK_TAG, None, False),
        (None, FORK_TAG, False),
    ],
)
def test_record_is_current_never_takes_an_older_latest(installed, latest, current):
    assert M._record_is_current(installed, latest) is current


# G2/AC3: a lookup that cannot answer keeps a complete install; an integrity failure never does


def _offline(monkeypatch):
    monkeypatch.setattr(M, "_latest_tag_via_download_host", lambda repo: None)
    _lookup_fails(monkeypatch, M.urllib.error.URLError("network is unreachable"))


def _installed_then_stale(
    monkeypatch,
    tmp_path,
    accelerator = "auto",
):
    """A complete install whose record names an older release, so the run must look up."""
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release, accelerator = accelerator)
    root = tmp_path / "audio.cpp"
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    record["release_tag"] = OLDER_TAG
    (root / M.INSTALL_RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(M, "resolve", RESOLVE)
    return root


def test_offline_rerun_keeps_a_complete_install(monkeypatch, tmp_path, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path)
    before = _tree_digest(root)
    _offline(monkeypatch)
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    out = capsys.readouterr().out
    assert "audio.cpp: keeping the existing complete install" in out.splitlines()
    assert _tree_digest(root) == before


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            M.core.PrebuiltFallback(
                "release unslothai/audio.cpp@x has no audio-cpp-prebuilt-manifest.json"
            ),
            id = "prebuilt-fallback",
        ),
        pytest.param(
            M.urllib.error.URLError("temporary failure in name resolution"), id = "urlerror"
        ),
        pytest.param(OSError("connection reset"), id = "oserror"),
        pytest.param(
            RuntimeError(
                "GitHub API returned 403 for https://api.github.com/repos/unslothai/audio.cpp/releases?per_page=30"
            ),
            id = "rate-limit",
        ),
        pytest.param(M.GitHubRateLimited("rate limited"), id = "rate-limited"),
    ],
)
def test_a_lookup_that_cannot_answer_keeps_a_complete_install(monkeypatch, tmp_path, capsys, exc):
    root = _installed_then_stale(monkeypatch, tmp_path)
    before = _tree_digest(root)
    _lookup_fails(monkeypatch, exc)
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_OK
    assert M.KEPT_EXISTING_LINE in capsys.readouterr().out.splitlines()
    assert _tree_digest(root) == before


@pytest.mark.parametrize("how", ["flag", "env"])
@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(M.urllib.error.URLError("network is unreachable"), id = "offline"),
        pytest.param(M.GitHubRateLimited("rate limited"), id = "rate-limited"),
    ],
)
def test_a_pinned_release_never_keeps_the_old_tree_when_the_lookup_cannot_answer(
    monkeypatch, tmp_path, capsys, exc, how
):
    # A pin names the release to install; exit 0 would report a release that was never installed.
    root = _installed_then_stale(monkeypatch, tmp_path)
    before = _tree_digest(root)
    _lookup_fails(monkeypatch, exc)
    capsys.readouterr()
    argv = ["--install-dir", str(root)]
    if how == "flag":
        argv += ["--published-release-tag", FORK_TAG]
    else:
        monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", FORK_TAG)
    assert M.main(argv) == M.EXIT_FAILED
    assert M.KEPT_EXISTING_LINE not in capsys.readouterr().out
    assert _tree_digest(root) == before


def test_a_rate_limit_from_the_core_lookup_is_reported_as_one(monkeypatch):
    _lookup_fails(
        monkeypatch,
        RuntimeError(
            "GitHub API returned 429 for https://api.github.com/repos/unslothai/audio.cpp/releases"
        ),
    )
    with pytest.raises(M.GitHubRateLimited, match = "set GH_TOKEN or GITHUB_TOKEN"):
        M.resolve("cpu", None)
    _lookup_fails(monkeypatch, M.core.PrebuiltFallback("no manifest"))
    monkeypatch.setattr(M, "_RELEASE_INDEX_MEMO", {})
    with pytest.raises(M.ReleaseLookupUnavailable):
        M.resolve("cpu", None)


def test_an_integrity_failure_is_never_kept(monkeypatch, tmp_path, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path)
    before = _tree_digest(root)
    _lookup_fails(
        monkeypatch, M.core.ReleaseIntegrityError("manifest sha256 disagrees with the index")
    )
    capsys.readouterr()
    assert M.main(["--install-dir", str(root)]) == M.EXIT_FAILED
    captured = capsys.readouterr()
    assert M.KEPT_EXISTING_LINE not in captured.out
    assert "disagrees with the index" in captured.err
    assert _tree_digest(root) == before


def test_offline_without_an_install_fails(monkeypatch, tmp_path, capsys):
    _stub_install_io(monkeypatch)
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_FAILED
    assert "no audio.cpp release lookup answered" in capsys.readouterr().err


def test_offline_keeps_nothing_for_a_tampered_server(monkeypatch, tmp_path, capsys):
    root = _installed_then_stale(monkeypatch, tmp_path)
    (root / "bin" / M.SERVER_NAME).write_bytes(b"truncated")
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(root)]) == M.EXIT_FAILED
    assert M.KEPT_EXISTING_LINE not in capsys.readouterr().out


def test_offline_does_not_keep_an_install_for_another_explicit_accelerator(monkeypatch, tmp_path):
    root = _installed_then_stale(monkeypatch, tmp_path)
    _offline(monkeypatch)
    assert M.main(["--install-dir", str(root), "--accelerator", "vulkan"]) == M.EXIT_FAILED
    assert M.main(["--install-dir", str(root), "--force"]) == M.EXIT_FAILED


# G1: main() leaves "auto" to install(), so the CPU fallback runs from the real CLI


def _resolve_recording(release, covers = ("cpu",)):
    asked = []

    def resolve(
        accel,
        token,
        tag = None,
    ):
        asked.append(accel)
        if accel in covers:
            return FORK, release, release["assets"][0]["name"]
        return FORK, None, None

    return asked, resolve


def test_main_auto_detected_gpu_without_a_bundle_installs_the_cpu_build(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
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


def test_auto_cuda_without_a_cuda_runtime_installs_the_cpu_build_and_reruns_offline(
    monkeypatch, tmp_path
):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release, covers = ("cpu", "cuda"))
    _stub_install_io(monkeypatch)
    seen = []
    monkeypatch.setattr(
        M, "_linux_cuda_runtime_available", lambda major: seen.append(major) or False
    )
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
    monkeypatch.setattr(
        M, "resolve", lambda *a, **k: pytest.fail("a matching rerun must not look up")
    )
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    # torch's CUDA runtime arrived: the CPU fallback is no longer a match, so the GPU build is looked up.
    monkeypatch.setattr(M, "_linux_cuda_runtime_available", lambda major: True)
    looked = []
    monkeypatch.setattr(
        M,
        "resolve",
        lambda accel, token, tag = None: looked.append(accel) or (FORK, release, CPU_ZIP),
    )
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert looked == ["cuda"]


def test_explicit_cuda_is_not_downgraded_for_a_missing_cuda_runtime(monkeypatch):
    monkeypatch.setattr(M, "_linux_cuda_runtime_available", lambda major: False)
    asked = []
    monkeypatch.setattr(
        M,
        "resolve",
        lambda accel, token, tag = None: asked.append(accel)
        or (FORK, {"assets": []}, "a-bin-ubuntu-x64-cuda13.tar.gz"),
    )
    M.resolve_for_request("cuda", None, None)
    assert asked == ["cuda"]


def test_cuda_runtime_probe_needs_the_bundle_major(monkeypatch, tmp_path):
    monkeypatch.setattr(M.sys, "platform", "linux")
    monkeypatch.setattr(M, "_cuda_runtime_dirs", lambda backend: [str(tmp_path)])
    monkeypatch.setattr(M, "_ldconfig_libs", lambda: set())
    assert M._linux_cuda_runtime_available(13) is False
    for name in ("libcudart.so.13", "libcublas.so.13", "libcufft.so.12"):
        (tmp_path / name).write_bytes(b"")
    assert M._linux_cuda_runtime_available(13) is True
    assert M._linux_cuda_runtime_available(12) is False
    assert M._linux_cuda_runtime_available(None) is False
    monkeypatch.setattr(M, "_cuda_runtime_dirs", lambda backend: [])
    monkeypatch.setattr(
        M, "_ldconfig_libs", lambda: {"libcudart.so.12", "libcublas.so.12", "libcufft.so.11"}
    )
    assert M._linux_cuda_runtime_available(12) is True
    monkeypatch.setattr(M.sys, "platform", "win32")
    assert M._linux_cuda_runtime_available(None) is True


@pytest.mark.parametrize(
    "names, driver, prefer, want",
    [
        (("cuda12", "cuda13"), (13, 1), 13, "cuda13"),
        (("cuda12", "cuda13"), (12, 4), None, "cuda12"),
        (("cuda12", "cuda13"), (13, 1), 12, "cuda12"),
        (("cuda12.8", "cuda12.8-colab"), (12, 8), None, "cuda12.8"),
    ],
)
def test_cuda_lines_with_and_without_a_minor(names, driver, prefer, want):
    assets = [f"audio-T-bin-ubuntu-x64-{n}.tar.gz" for n in names]
    got = M.resolve_release_asset(
        assets,
        system = "Linux",
        machine = "x86_64",
        accelerator = "cuda",
        driver_cuda = driver,
        prefer_cuda_major = prefer,
    )
    assert got == f"audio-T-bin-ubuntu-x64-{want}.tar.gz"
    win = ["audio-T-bin-windows-x64-cuda13.zip", "audio-T-cudart-windows-x64-cuda13.zip"]
    assert M.cudart_asset_for(win, win[0]) == win[1]


def test_main_explicit_gpu_request_is_never_downgraded(monkeypatch, tmp_path):
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
    # A CUDA host with its CUDA runtime, whatever the runner has (one without resolves CPU only).
    monkeypatch.setattr(M, "_linux_cuda_runtime_available", lambda major: True)
    assert M.main(["--print-asset"]) == M.EXIT_OK
    assert asked == ["cuda", "cpu"]
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"{FORK} {FORK_TAG} {CPU_ZIP}"


def test_print_asset_takes_the_pin_flag(monkeypatch, capsys):
    # The flag overrides the env pin and asks no HEAD.
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", HOTFIX_TAG)
    monkeypatch.setattr(
        M, "_latest_tag_via_download_host", lambda repo: pytest.fail("a pin asks no HEAD")
    )
    _windows_host(monkeypatch)
    asked = _serve_index(monkeypatch, {(FORK, OLDER_TAG): _names_release(OLDER_TAG, [OLDER_ZIP])})
    argv = ["--print-asset", "--accelerator", "cpu", "--published-release-tag", OLDER_TAG]
    assert M.main(argv) == M.EXIT_OK
    assert capsys.readouterr().out.strip() == f"{FORK} {OLDER_TAG} {OLDER_ZIP}"
    # The env pin is the user's pick, so an index-less release could fall back to GitHub's digest.
    assert asked == [(FORK, OLDER_TAG, True)]


# G12: UNSLOTH_AUDIO_CPP_ACCELERATOR


def test_accelerator_env_is_an_explicit_request(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release, covers = ("cpu", "cuda"))
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
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


# G11: which release, from which repo


def test_the_default_repo_is_the_fork_with_no_fallback_repo():
    assert M.DEFAULT_REPO == "unslothai/audio.cpp" and M._repo() == M.DEFAULT_REPO
    # No upstream rung and no in-repo digest table: the fork's own release index is the trust anchor.
    assert not [name for name in vars(M) if "UPSTREAM" in name or "PINS" in name.upper()]


@pytest.mark.parametrize("value", [None, "", "latest", " LATEST "])
def test_an_unset_empty_or_latest_tag_tracks_the_newest_release(monkeypatch, heads, value):
    if value is not None:
        monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", value)
    assert M._pinned_tag() is None and M._user_picked_release() is False
    _windows_host(monkeypatch)
    asked = _serve_index(monkeypatch, {(FORK, FORK_TAG): _names_release(FORK_TAG, [CPU_ZIP])})
    assert M.resolve("cpu", None)[1:] == (_names_release(FORK_TAG, [CPU_ZIP]), CPU_ZIP)
    assert asked == [(FORK, FORK_TAG, False)] and heads == [FORK]


def test_an_exact_tag_pins_the_release(monkeypatch, heads):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", f" {OLDER_TAG} ")
    assert M._pinned_tag() == OLDER_TAG and M._user_picked_release() is True
    _windows_host(monkeypatch)
    asked = _serve_index(
        monkeypatch,
        {
            (FORK, OLDER_TAG): _names_release(OLDER_TAG, [OLDER_ZIP]),
            (FORK, FORK_TAG): _names_release(FORK_TAG, [CPU_ZIP]),
        },
    )
    assert M.resolve("cpu", None) == (FORK, _names_release(OLDER_TAG, [OLDER_ZIP]), OLDER_ZIP)
    # --published-release-tag beats the env pin; trust still follows the user's pick from the env.
    assert M.resolve("cpu", None, FORK_TAG)[2] == CPU_ZIP
    assert asked == [(FORK, OLDER_TAG, True), (FORK, FORK_TAG, True)]
    assert heads == []


def test_a_repo_override_tracks_that_repos_newest_release(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_REPO", " someone/audio.cpp ")
    monkeypatch.setattr(M, "_latest_tag_via_download_host", lambda repo: "v2")
    asked = _serve_index(monkeypatch, {})
    assert M.resolve("cpu", None) == ("someone/audio.cpp", None, None)
    assert asked == [("someone/audio.cpp", "v2", True)]


@pytest.mark.parametrize(
    "min_os, libc, refused",
    [
        ("glibc-2.35", ("glibc", "2.31"), True),
        ("glibc-2.35", ("glibc", "2.34"), True),
        ("glibc-2.35", ("glibc", "2.35"), False),
        ("glibc-2.35", ("glibc", "2.39"), False),
        ("glibc-2.35", ("", ""), False),
        ("glibc-2.35", ("glibc", "weird"), False),
        # The floor is the manifest's, not a constant.
        ("glibc-2.39", ("glibc", "2.35"), True),
        # No manifest (a user-picked release without one): no check.
        (None, ("glibc", "2.31"), False),
    ],
)
def test_old_glibc_is_refused_before_the_bundle_download(
    monkeypatch, tmp_path, min_os, libc, refused
):
    monkeypatch.setattr(M.sys, "platform", "linux")
    monkeypatch.setattr(M.platform, "libc_ver", lambda: libc)
    release = _release(tmp_path, CPU_ZIP, min_os = min_os)
    _stub_install_io(monkeypatch)
    downloads = []
    monkeypatch.setattr(
        M, "_download", lambda url, dest: downloads.append(url) or _fake_download(url, dest)
    )
    monkeypatch.setattr(M, "resolve_for_request", lambda *a: ("cpu", FORK, release, CPU_ZIP))
    target = tmp_path / "audio.cpp"
    if refused:
        floor = min_os.split("-")[1]
        with pytest.raises(RuntimeError, match = rf"need glibc {floor}\+; this host has {libc[1]}"):
            M._install_locked(target, "cpu", None, False)
        assert downloads == [] and not target.exists()
    else:
        assert M._install_locked(target, "cpu", None, False).is_file()
        assert len(downloads) == 1


# A3: every archive is checked against the release's own sha256 index


def _install_via_main(monkeypatch, tmp_path, release, *argv):
    bundle = release["assets"][0]["name"]
    monkeypatch.setattr(M, "resolve", lambda accel, token, tag = None: (FORK, release, bundle))
    _stub_install_io(monkeypatch)
    downloads = []
    monkeypatch.setattr(
        M,
        "_download",
        lambda url, dest: downloads.append(Path(url).name) or _fake_download(url, dest),
    )
    code = M.main(["--install-dir", str(tmp_path / "audio.cpp"), "--accelerator", "cpu", *argv])
    return code, downloads


def test_an_asset_the_index_does_not_cover_is_refused(monkeypatch, tmp_path, capsys):
    release = _release(tmp_path, CPU_ZIP)
    release["checksums"] = {f"audio-{FORK_TAG}-bin-windows-x64-cpu.zip": "1" * 64}
    code, downloads = _install_via_main(monkeypatch, tmp_path, release)
    assert code == M.EXIT_FAILED and downloads == []
    assert f"{CPU_ZIP} is not covered by {M.SHA256_ASSET_NAME}" in capsys.readouterr().err
    assert not (tmp_path / "audio.cpp").exists()


def test_a_manifest_digest_that_disagrees_with_the_index_is_refused(monkeypatch, tmp_path, capsys):
    release = _release(tmp_path, CPU_ZIP)
    release["assets"][0]["manifest_sha256"] = "1" * 64
    code, downloads = _install_via_main(monkeypatch, tmp_path, release)
    assert code == M.EXIT_FAILED and downloads == []
    assert "disagrees with" in capsys.readouterr().err
    assert not (tmp_path / "audio.cpp").exists()


def test_a_download_that_does_not_hash_to_the_index_is_refused(monkeypatch, tmp_path, capsys):
    release = _release(tmp_path, CPU_ZIP)
    release["checksums"][CPU_ZIP] = release["assets"][0]["manifest_sha256"] = "0" * 64
    code, downloads = _install_via_main(monkeypatch, tmp_path, release)
    assert code == M.EXIT_FAILED and downloads == [CPU_ZIP]
    assert f"sha256 mismatch for {CPU_ZIP}" in capsys.readouterr().err
    assert not (tmp_path / "audio.cpp").exists()


def _installed_at_an_older_release(monkeypatch, tmp_path):
    """What an update replaces: a Studio-managed tree of an older release."""
    old = _release(tmp_path, f"audio-{HOTFIX_TAG}-bin-windows-x64-cpu-portable.zip", tag = HOTFIX_TAG)
    _install(monkeypatch, tmp_path, old)
    return tmp_path / "audio.cpp"


def test_an_older_release_is_replaced_by_the_newest(monkeypatch, tmp_path):
    root = _installed_at_an_older_release(monkeypatch, tmp_path)
    server = _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP))
    record = json.loads((root / M.INSTALL_RECORD).read_text())
    assert (record["release_tag"], record["asset"]) == (FORK_TAG, CPU_ZIP)
    assert server.read_bytes() == b"binary " + CPU_ZIP.encode()


def test_an_update_that_fails_the_index_leaves_the_older_tree(monkeypatch, tmp_path, capsys):
    root = _installed_at_an_older_release(monkeypatch, tmp_path)
    before = _tree_digest(root)
    release = _release(tmp_path, CPU_ZIP)
    release["checksums"][CPU_ZIP] = release["assets"][0]["manifest_sha256"] = "0" * 64
    capsys.readouterr()
    code, _ = _install_via_main(monkeypatch, tmp_path, release)
    assert code == M.EXIT_FAILED
    out, err = capsys.readouterr()
    assert "sha256 mismatch" in err
    assert not any(line.startswith("audio.cpp: installed") for line in out.splitlines())
    assert _tree_digest(root) == before


WIN_CUDA_ZIP = f"audio-{FORK_TAG}-bin-windows-x64-cuda12.zip"
CUDART_ZIP = f"audio-{FORK_TAG}-cudart-windows-x64-cuda12.zip"


def _cuda_release(monkeypatch, tmp_path):
    """A Windows CUDA bundle plus its cudart archive, on a Python whose torch ships no CUDA runtime."""
    release = _release(tmp_path, WIN_CUDA_ZIP)
    runtime = tmp_path / "src" / CUDART_ZIP
    with zipfile.ZipFile(runtime, "w") as zf:
        zf.writestr("cudart64_12.dll", b"runtime")
    digest = _sha(runtime)
    release["assets"].append(
        {
            "name": CUDART_ZIP,
            "browser_download_url": str(runtime),
            "manifest_sha256": digest,
            "min_os": None,
        }
    )
    release["checksums"][CUDART_ZIP] = digest
    _stub_install_io(monkeypatch)
    downloads = []
    monkeypatch.setattr(
        M,
        "_download",
        lambda url, dest: downloads.append(Path(url).name) or _fake_download(url, dest),
    )
    monkeypatch.setattr(M, "torch_provides_cuda_runtime", lambda major: False)
    monkeypatch.setattr(M, "_cuda_runtime_dirs", lambda backend: [])
    monkeypatch.setattr(M, "resolve", lambda accel, token, tag = None: (FORK, release, WIN_CUDA_ZIP))
    return release, downloads


def test_the_cuda_runtime_archive_is_verified_against_the_index(monkeypatch, tmp_path, capsys):
    release, downloads = _cuda_release(monkeypatch, tmp_path)
    server = M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cuda")
    assert downloads == [WIN_CUDA_ZIP, CUDART_ZIP]
    assert (server.parent / "cudart64_12.dll").read_bytes() == b"runtime"
    record = M.read_install_record(tmp_path / "audio.cpp")
    assert (record["asset"], record["cudart_asset"], record["backend"]) == (
        WIN_CUDA_ZIP,
        CUDART_ZIP,
        "cuda",
    )
    err = capsys.readouterr().err
    assert f"verifying {CUDART_ZIP} against {M.SHA256_ASSET_NAME}" in err


def test_a_cuda_runtime_archive_the_index_does_not_cover_is_refused(monkeypatch, tmp_path):
    release, downloads = _cuda_release(monkeypatch, tmp_path)
    del release["checksums"][CUDART_ZIP]
    with pytest.raises(M.core.ReleaseIntegrityError, match = "not covered"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cuda")
    # Refused before either archive is fetched.
    assert downloads == [] and not (tmp_path / "audio.cpp").exists()


def test_a_cuda_runtime_download_that_does_not_hash_to_the_index_is_refused(monkeypatch, tmp_path):
    release, downloads = _cuda_release(monkeypatch, tmp_path)
    release["checksums"][CUDART_ZIP] = release["assets"][1]["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match = f"sha256 mismatch for {CUDART_ZIP}"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cuda")
    assert downloads == [WIN_CUDA_ZIP, CUDART_ZIP] and not (tmp_path / "audio.cpp").exists()


# User-picked releases without an index: GitHub's digest, and only for the user's own pick


UPSTREAM = "0xShug0/audio.cpp"
UPSTREAM_ZIP = "audio-v0.9.1-bin-windows-x64-cpu-portable.zip"


def _unindexed_upstream(
    monkeypatch,
    tmp_path,
    *,
    publishes_index = False,
):
    """The fork-style lookup fails (no manifest, no index); the API release lists GitHub digests."""
    _lookup_fails(
        monkeypatch,
        M.core.PrebuiltFallback(f"release {UPSTREAM}@v0.9.1 has no {M.MANIFEST_ASSET_NAME}"),
    )
    archive = _release(tmp_path, UPSTREAM_ZIP, tag = "v0.9.1")["assets"][0]["browser_download_url"]
    assets = [
        {
            "name": UPSTREAM_ZIP,
            "browser_download_url": archive,
            "digest": "sha256:" + _sha(Path(archive)),
        },
        {"name": "framework.tar.gz", "browser_download_url": "x", "digest": None},
    ]
    if publishes_index:
        assets.append({"name": M.SHA256_ASSET_NAME, "browser_download_url": "x", "digest": None})
    fetched = []

    def fetch(
        repo,
        tag,
        token,
        timeout = 30.0,
    ):
        fetched.append((repo, tag))
        return {"tag_name": "v0.9.1", "assets": assets}

    monkeypatch.setattr(M, "_fetch_release", fetch)
    monkeypatch.setattr(M, "_latest_tag_via_download_host", lambda repo: "v0.9.1")
    _windows_host(monkeypatch)
    return fetched


def test_a_user_picked_release_without_an_index_installs_on_githubs_digest(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_REPO", UPSTREAM)
    fetched = _unindexed_upstream(monkeypatch, tmp_path)
    _stub_install_io(monkeypatch)
    server = M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cpu")
    record = M.read_install_record(server.parent.parent)
    assert (record["published_repo"], record["release_tag"], record["asset"]) == (
        UPSTREAM,
        "v0.9.1",
        UPSTREAM_ZIP,
    )
    assert record["asset_sha256"] == _sha(tmp_path / "src" / UPSTREAM_ZIP)
    assert fetched == [(UPSTREAM, "v0.9.1")]
    assert f"publishes no {M.SHA256_ASSET_NAME}" in capsys.readouterr().err


def test_an_unindexed_release_trusts_githubs_digest_or_none():
    release = {"checksums": None}
    assert M._expected_sha256(release, {"name": "a.zip", "digest": None}) is None
    assert (
        M._expected_sha256(release, {"name": "a.zip", "digest": "sha256:" + "A" * 64}) == "a" * 64
    )


def test_the_index_is_never_skipped_by_default(monkeypatch, tmp_path):
    fetched = _unindexed_upstream(monkeypatch, tmp_path)
    with pytest.raises(M.ReleaseLookupUnavailable, match = "has no"):
        M.resolve("cpu", None)
    assert fetched == []


def test_the_pin_flag_never_skips_the_index_of_the_default_repo(monkeypatch, tmp_path):
    fetched = _unindexed_upstream(monkeypatch, tmp_path)
    with pytest.raises(M.ReleaseLookupUnavailable, match = "has no"):
        M.resolve("cpu", None, "v0.9.1")
    assert fetched == []


def test_the_in_app_pin_installs_from_a_user_picked_repo_without_an_index(monkeypatch, tmp_path):
    # The in-app update passes the tag it offered; setup accepted this repo on GitHub's digest, so the
    # update must too, or the offer could never be applied.
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_REPO", UPSTREAM)
    fetched = _unindexed_upstream(monkeypatch, tmp_path)
    repo, release, chosen = M.resolve("cpu", None, "v0.9.1")
    assert (repo, chosen, release["checksums"]) == (UPSTREAM, UPSTREAM_ZIP, None)
    assert fetched == [(UPSTREAM, "v0.9.1")]


def test_a_user_picked_release_that_publishes_an_index_never_skips_it(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_AUDIO_CPP_TAG", "v0.9.1")
    fetched = _unindexed_upstream(monkeypatch, tmp_path, publishes_index = True)
    with pytest.raises(M.ReleaseLookupUnavailable):
        M.resolve("cpu", None)
    assert fetched == [(FORK, "v0.9.1")]


# _release_index through prebuilt_core's real lookup and parsers, fed the published v0.9.1-unsloth.1
# manifest and sha256 index (trimmed to three of their artifacts; digests as published).

_V091 = "v0.9.1-unsloth.1"
_V091_COMMIT = "cf124a67cc55d8f65a9a15eec69edbff0fb212c8"
_V091_LINUX_CPU = f"audio-{_V091}-bin-ubuntu-x64-cpu-portable.tar.gz"
_V091_WIN_CUDA = f"audio-{_V091}-bin-windows-x64-cuda12.zip"
_V091_CUDART = f"audio-{_V091}-cudart-windows-x64-cuda12.zip"
_V091_SHA = {
    _V091_LINUX_CPU: "a1e61f1ed2d987b15950cb63e61872b2edd4fbf5e463602e64aa404296d901bb",
    _V091_WIN_CUDA: "e865b05fac29956110b1140457fccd9319583be3025aa6a30f1f653aa0003a75",
    _V091_CUDART: "78ff03c3215e254c5c38c8689a9054287d39d9490626a7c475babd88b326f2a6",
    f"audio.cpp-source-{_V091}.tar.gz": "d5df360aa4e2017bb733479b09b205c434561cc2f05fbd67e87e67733cc3f7a8",
    "audio-cpp-prebuilt-manifest.json": "fa6cc5965111cf4d025d2d7508877c3cbba9fe8c706f66323dd387f06d620625",
}
_V091_MANIFEST = {
    "schema_version": 1,
    "component": "audio.cpp",
    "source_repo": "unslothai/audio.cpp",
    "source_commit": _V091_COMMIT,
    "upstream_repo": "0xShug0/audio.cpp",
    "upstream_tag": "v0.9.1",
    "packaging_tag": _V091,
    "studio_protocol": {"version": 1, "probe_argv": ["--help", "--version"]},
    "artifacts": [
        {
            "asset": _V091_LINUX_CPU,
            "kind": "bundle",
            "os": "ubuntu",
            "arch": "x64",
            "accel": "cpu-portable",
            "backend": "cpu",
            "min_os": "glibc-2.35",
            "espeak": True,
            "sha256": _V091_SHA[_V091_LINUX_CPU],
        },
        {
            "asset": _V091_WIN_CUDA,
            "kind": "bundle",
            "os": "windows",
            "arch": "x64",
            "accel": "cuda12",
            "backend": "cuda",
            "runtime_line": "cuda12",
            "cuda_runtime_bundled": False,
            "min_os": "windows-10",
            "espeak": True,
            "sha256": _V091_SHA[_V091_WIN_CUDA],
        },
        {
            "asset": _V091_CUDART,
            "kind": "cuda-runtime",
            "os": "windows",
            "arch": "x64",
            "accel": "cuda12",
            "backend": "cuda",
            "runtime_line": "cuda12",
            "sha256": _V091_SHA[_V091_CUDART],
        },
    ],
}
_V091_INDEX = {
    "schema_version": 1,
    "component": "audio.cpp",
    "source_repo": "unslothai/audio.cpp",
    "source_commit": _V091_COMMIT,
    "upstream_tag": "v0.9.1",
    "release_tag": _V091,
    "artifacts": {
        name: {
            "kind": "bundle",
            "repo": "unslothai/audio.cpp",
            "sha256": digest,
            "source_commit": _V091_COMMIT,
            "upstream_tag": "v0.9.1",
        }
        for name, digest in _V091_SHA.items()
    },
}


def _cdn(name, tag = _V091):
    return f"https://github.com/{FORK}/releases/download/{tag}/{name}"


def _download_host(
    monkeypatch,
    *,
    index = None,
    manifest = None,
    latest = _V091,
):
    """Serve the fork release from the download host; every api.github.com fetch fails (offline)."""
    import copy

    files = {
        _cdn(M.SHA256_ASSET_NAME): copy.deepcopy(index or _V091_INDEX),
        _cdn(M.MANIFEST_ASSET_NAME): copy.deepcopy(manifest or _V091_MANIFEST),
    }
    requests = []

    def head(repo):
        requests.append(("HEAD", f"https://github.com/{repo}/releases/latest"))
        return latest

    def get_json(url):
        requests.append(("GET", url))
        if url not in files:
            raise M.urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return files[url]

    def fetch_json(url):
        requests.append(("API", url))
        raise M.urllib.error.URLError("api.github.com unreachable in this test")

    monkeypatch.setattr(M, "_latest_tag_via_download_host", LATEST_TAG)
    monkeypatch.setattr(M, "_download_host_latest_release_tag", head, raising = False)
    monkeypatch.setattr(M, "_download_host_json", get_json, raising = False)
    monkeypatch.setattr(M, "fetch_json", fetch_json, raising = False)
    return requests


def test_the_newest_fork_release_resolves_through_its_own_sha256_index(monkeypatch, capsys):
    requests = _download_host(monkeypatch)
    release = M._release_index(FORK, M._latest_tag_via_download_host(FORK))
    assert release["tag_name"] == _V091
    assert [a["name"] for a in release["assets"]] == [_V091_LINUX_CPU, _V091_WIN_CUDA, _V091_CUDART]
    assets = {a["name"]: a for a in release["assets"]}
    assert assets[_V091_LINUX_CPU]["browser_download_url"] == _cdn(_V091_LINUX_CPU)
    assert assets[_V091_LINUX_CPU]["min_os"] == "glibc-2.35"
    for name in (_V091_LINUX_CPU, _V091_WIN_CUDA, _V091_CUDART):
        assert M._expected_sha256(release, assets[name]) == _V091_SHA[name]
    # The index first, then the manifest, from the download host: no API call.
    assert requests == [
        ("HEAD", f"https://github.com/{FORK}/releases/latest"),
        ("GET", _cdn(M.SHA256_ASSET_NAME)),
        ("GET", _cdn(M.MANIFEST_ASSET_NAME)),
    ]
    assert f"resolved {FORK}@{_V091} via the download host" in capsys.readouterr().err
    # Memoized per (repo, tag): the CPU-fallback re-resolve fetches nothing again.
    assert M._release_index(FORK, _V091) is release
    assert len(requests) == 3


def test_the_default_install_lookup_asks_only_the_fork(monkeypatch, capsys):
    requests = _download_host(monkeypatch)
    monkeypatch.setattr(M.platform, "system", lambda: "Linux")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    assert M.main(["--print-asset", "--accelerator", "cpu"]) == M.EXIT_OK
    assert capsys.readouterr().out.strip() == f"{FORK} {_V091} {_V091_LINUX_CPU}"
    assert [kind for kind, _ in requests] == ["HEAD", "GET", "GET"]
    assert all(f"github.com/{FORK}/" in url for _, url in requests)
    assert not any("0xShug0" in url for _, url in requests)


def test_an_index_bound_to_another_release_is_rejected(monkeypatch):
    import copy

    index = copy.deepcopy(_V091_INDEX)
    index["release_tag"] = OLDER_TAG
    requests = _download_host(monkeypatch, index = index)
    with pytest.raises(M.ReleaseLookupUnavailable):
        M.resolve("cpu", None)
    # The download host refused it; the API fallback (offline here) was asked for the same tag.
    assert requests[-1] == ("API", f"https://api.github.com/repos/{FORK}/releases/tags/{_V091}")
    assert ("GET", _cdn(M.MANIFEST_ASSET_NAME)) not in requests


def test_a_manifest_digest_the_index_contradicts_is_refused_through_the_core(monkeypatch):
    import copy

    manifest = copy.deepcopy(_V091_MANIFEST)
    manifest["artifacts"][2]["sha256"] = "1" * 64
    _download_host(monkeypatch, manifest = manifest)
    release = M._release_index(FORK, _V091)
    cudart = {a["name"]: a for a in release["assets"]}[_V091_CUDART]
    with pytest.raises(M.core.ReleaseIntegrityError, match = "possibly tampered"):
        M._expected_sha256(release, cudart)


def test_latest_tag_lookup_is_memoized_and_never_raises(monkeypatch):
    calls = _real_head(monkeypatch, M.urllib.error.URLError("offline"))
    assert M._latest_tag_via_download_host(FORK) is None
    assert M._latest_tag_via_download_host(FORK) is None
    assert calls == [FORK]
    calls = _real_head(monkeypatch, FORK_TAG)
    monkeypatch.setattr(M, "_LATEST_TAG_MEMO", {})
    assert M._latest_tag_via_download_host(FORK) == FORK_TAG
    assert M._latest_tag_via_download_host(FORK) == FORK_TAG
    assert calls == [FORK]


# G3(b): the staged server must start before it replaces anything


class _Ran:
    def __init__(
        self,
        returncode,
        stdout = "",
        stderr = "",
    ):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_a_staged_server_that_cannot_start_keeps_the_old_tree(monkeypatch, tmp_path, capsys):
    good = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, good)
    root = tmp_path / "audio.cpp"
    before = _tree_digest(root)
    newer = _release(tmp_path, f"audio-{FORK_TAG}-bin-windows-x64-cpu.zip")
    monkeypatch.setattr(
        M, "resolve", lambda accel, token, tag = None: (FORK, newer, newer["assets"][0]["name"])
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
    monkeypatch, tmp_path
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


def test_refuses_to_replace_a_directory_it_does_not_own(monkeypatch, tmp_path):
    foreign = tmp_path / "audio.cpp"
    foreign.mkdir()
    (foreign / "mine.txt").write_text("user data")
    release = _release(tmp_path, CPU_ZIP)
    with pytest.raises(RuntimeError, match = "not an Unsloth-managed directory"):
        _install(monkeypatch, tmp_path, release)
    assert (foreign / "mine.txt").read_text() == "user data"


def test_bundle_without_a_server_leaves_the_old_tree(monkeypatch, tmp_path):
    good = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, good)
    bad = _release(
        tmp_path, f"audio-{FORK_TAG}-bin-windows-x64-cpu.zip", server_path = "bin/other.exe"
    )
    with pytest.raises(RuntimeError, match = "contains no"):
        _install(monkeypatch, tmp_path, bad, force = True)
    assert (tmp_path / "audio.cpp" / "bin" / M.SERVER_NAME).is_file()


def test_auto_detected_gpu_without_a_bundle_installs_the_cpu_build(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release)
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cuda")
    M.install(install_dir = tmp_path / "audio.cpp", accelerator = "auto")
    assert asked == ["cuda", "cpu"]
    assert json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())["backend"] == "cpu"


def test_an_explicit_gpu_request_is_never_downgraded(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "resolve", lambda accel, token, tag = None: (FORK, None, None))
    with pytest.raises(RuntimeError, match = "No prebuilt"):
        M.install(install_dir = tmp_path / "audio.cpp", accelerator = "cuda")


def test_intel_mac_bundle_runs_on_the_cpu_backend():
    assert M._backend_of(f"audio-{TAG}-bin-macos-x64-metal.tar.gz") == "cpu"
    assert M._backend_of(f"audio-{TAG}-bin-macos-arm64-metal.tar.gz") == "metal"


def test_record_notes_a_static_espeak_build(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP, extra = [("bin/espeak-ng-data.bin", b"data")])
    _install(monkeypatch, tmp_path, release)
    assert json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())["espeak"] is True
    target = tmp_path / "audio.cpp"
    record = json.loads((target / M.INSTALL_RECORD).read_text())
    assert M._intact_install(target, record) is not None
    # Quarantined phonemizer data makes the install incomplete, so the next update repairs it.
    next(target.rglob("espeak-ng-data.bin")).unlink()
    assert M._intact_install(target, record) is None


def test_a_dir_holding_only_the_old_child_home_is_adopted(monkeypatch, tmp_path):
    leftover = tmp_path / "audio.cpp" / ".child_home" / "AppData"
    leftover.mkdir(parents = True)
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release)
    assert (tmp_path / "audio.cpp" / M.OWNERSHIP_MARKER).is_file()
    assert not (tmp_path / "audio.cpp" / ".child_home").exists()


def test_retired_trees_are_swept_on_the_next_run(monkeypatch, tmp_path):
    stale = tmp_path / "audio.cpp.old-1234"
    stale.mkdir()
    (stale / M.OWNERSHIP_MARKER).touch()
    foreign = tmp_path / "audio.cpp.old-user"
    foreign.mkdir()
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release)
    assert not stale.exists() and foreign.exists()


def test_a_running_server_reports_busy_before_downloading(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release)
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


def test_an_upstream_record_is_replaced_by_the_forks_newest(monkeypatch, tmp_path):
    # A 0xShug0 install (the old fallback rung, or a UNSLOTH_AUDIO_CPP_REPO pick) is not this repo's release.
    up_zip = "audio-v0.9.0-bin-windows-x64-cpu-portable.zip"
    _install(
        monkeypatch,
        tmp_path,
        _release(tmp_path / "up", up_zip, tag = "v0.9.0"),
        repo = "0xShug0/audio.cpp",
    )
    root = tmp_path / "audio.cpp"
    record = M.read_install_record(root)
    assert record["published_repo"] == "0xShug0/audio.cpp"
    assert M._current_install_matches(root, record, "cpu", None, "v0.9.0") is None
    assert M._current_install_matches(root, record, "cpu", None, "v0.9.0", pinned = True) is None
    _install(monkeypatch, tmp_path, _release(tmp_path, CPU_ZIP))
    record = M.read_install_record(root)
    assert (record["published_repo"], record["release_tag"], record["asset"]) == (
        FORK,
        FORK_TAG,
        CPU_ZIP,
    )


def test_the_accelerator_flag_takes_what_setup_forwards(monkeypatch, tmp_path):
    release = _release(tmp_path, CPU_ZIP)
    asked, resolve = _resolve_recording(release, covers = ("cpu", "cuda"))
    _stub_install_io(monkeypatch)
    monkeypatch.setattr(M, "resolve", resolve)
    root = str(tmp_path / "audio.cpp")
    assert M.main(["--install-dir", root, "--accelerator", " CUDA "]) == M.EXIT_OK
    assert asked == ["cuda"]


def test_a_killed_install_leaves_no_staging_dir_behind(monkeypatch, tmp_path):
    stranded = tmp_path / ".audio.cpp-staging-killed"
    stranded.mkdir()
    (stranded / "partial.zip").write_bytes(b"x" * 16)
    release = _release(tmp_path, CPU_ZIP)
    _install(monkeypatch, tmp_path, release)
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


# The asset set unslothai/audio.cpp's prebuilt CI publishes (v0.8.2-audio8-perf-hotfix-unsloth.1).
_FORK_T = "v0.8.2-audio8-perf-hotfix-unsloth.1"
_FORK_ASSETS = [
    f"audio-{_FORK_T}-bin-{n}"
    for n in (
        "macos-arm64-metal.tar.gz",
        "macos-x64-metal.tar.gz",
        "ubuntu-arm64-cpu.tar.gz",
        "ubuntu-x64-cpu-portable.tar.gz",
        "ubuntu-x64-cpu.tar.gz",
        "ubuntu-x64-cuda12.tar.gz",
        "ubuntu-x64-cuda13.tar.gz",
        "ubuntu-x64-vulkan.tar.gz",
        "windows-x64-cpu-portable.zip",
        "windows-x64-cpu.zip",
        "windows-x64-cuda12.zip",
        "windows-x64-cuda13.zip",
        "windows-x64-vulkan.zip",
    )
] + [f"audio-{_FORK_T}-cudart-windows-x64-cuda{m}.zip" for m in (12, 13)]


@pytest.mark.parametrize(
    "system, machine, accel, driver, prefer, want, cudart",
    [
        ("Linux", "x86_64", "cpu", None, None, "ubuntu-x64-cpu-portable.tar.gz", None),
        ("Linux", "x86_64", "vulkan", None, None, "ubuntu-x64-vulkan.tar.gz", None),
        ("Linux", "x86_64", "cuda", (13, 1), 13, "ubuntu-x64-cuda13.tar.gz", None),
        ("Linux", "x86_64", "cuda", (13, 1), 12, "ubuntu-x64-cuda12.tar.gz", None),
        ("Linux", "x86_64", "cuda", (12, 4), None, "ubuntu-x64-cuda12.tar.gz", None),
        ("Linux", "x86_64", "cuda", (11, 8), None, None, None),
        ("Linux", "aarch64", "cpu", None, None, "ubuntu-arm64-cpu.tar.gz", None),
        ("Linux", "aarch64", "cuda", (13, 0), 13, None, None),
        ("Windows", "AMD64", "cpu", None, None, "windows-x64-cpu-portable.zip", None),
        ("Windows", "AMD64", "vulkan", None, None, "windows-x64-vulkan.zip", None),
        ("Windows", "AMD64", "cuda", (13, 2), None, "windows-x64-cuda13.zip", "cuda13"),
        ("Windows", "AMD64", "cuda", (12, 8), 12, "windows-x64-cuda12.zip", "cuda12"),
        ("Darwin", "arm64", "metal", None, None, "macos-arm64-metal.tar.gz", None),
        ("Darwin", "x86_64", "metal", None, None, "macos-x64-metal.tar.gz", None),
    ],
)
def test_the_fork_release_covers_every_host(system, machine, accel, driver, prefer, want, cudart):
    got = M.resolve_release_asset(
        _FORK_ASSETS,
        system = system,
        machine = machine,
        accelerator = accel,
        driver_cuda = driver,
        prefer_cuda_major = prefer,
    )
    if want is None:
        # No bundle: resolve_for_request falls back to the CPU build for an auto request.
        assert got is None
        return
    assert got == f"audio-{_FORK_T}-bin-{want}"
    rt = M.cudart_asset_for(_FORK_ASSETS, got)
    assert rt == (f"audio-{_FORK_T}-cudart-windows-x64-{cudart}.zip" if cudart else None)


# #13042: an AMD GPU auto-installs the Vulkan bundle, and the CPU one when Vulkan finds no GPU


def _fake_host(
    monkeypatch,
    tmp_path,
    *,
    vendors = (),
    smi = False,
    cuda = None,
    proc_nvidia = False,
):
    for i, vendor in enumerate(vendors):
        f = tmp_path / "drm" / f"card{i}" / "device" / "vendor"
        f.parent.mkdir(parents = True)
        f.write_text(vendor + "\n")
    monkeypatch.setattr(M.platform, "system", lambda: "Linux")
    monkeypatch.setattr(M.sys, "platform", "linux")
    monkeypatch.setattr(
        M.glob,
        "glob",
        lambda pattern: sorted(str(p) for p in (tmp_path / "drm").glob("card*/device/vendor")),
    )
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if smi else None)
    monkeypatch.setattr(M, "nvidia_driver_cuda_version", lambda: cuda)
    real_listdir = M.os.listdir
    monkeypatch.setattr(
        M.os,
        "listdir",
        lambda p: (["0000:01:00.0"] if proc_nvidia else [])
        if p == "/proc/driver/nvidia/gpus"
        else real_listdir(p),
    )


@pytest.mark.parametrize(
    "host, expected",
    [
        (dict(vendors = ("0x1002",)), "vulkan"),
        (dict(vendors = ("0x8086", "0x1002")), "vulkan"),
        (dict(vendors = ("0x8086",)), "cpu"),
        (dict(), "cpu"),
        (dict(vendors = ("0x1002",), smi = True, cuda = (12, 8)), "cuda"),
        # An NVIDIA card nvidia-smi or the driver lists without a CUDA version keeps the CPU build.
        (dict(vendors = ("0x1002",), smi = True), "cpu"),
        (dict(vendors = ("0x1002",), proc_nvidia = True), "cpu"),
    ],
)
def test_linux_amd_gpu_is_detected_as_vulkan(monkeypatch, tmp_path, host, expected):
    _fake_host(monkeypatch, tmp_path, **host)
    assert M.detect_accelerator() == expected


def test_windows_amd_gpu_is_detected_from_the_registry(monkeypatch):
    import types

    fake = types.ModuleType("install_llama_prebuilt")
    monkeypatch.setitem(sys.modules, "install_llama_prebuilt", fake)
    monkeypatch.setattr(M.platform, "system", lambda: "Windows")
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    monkeypatch.setattr(M, "nvidia_driver_cuda_version", lambda: None)
    fake.windows_amd_gpu_in_registry = lambda: True
    assert M.detect_accelerator() == "vulkan"
    fake.windows_amd_gpu_in_registry = lambda: False
    assert M.detect_accelerator() == "cpu"
    fake.windows_amd_gpu_in_registry = lambda: 1 / 0
    assert M.detect_accelerator() == "cpu"


@pytest.mark.parametrize(
    "listed, has_device",
    [
        (
            _Ran(
                0,
                stdout = 'available_devices=2\nVulkan:0 "AMD Radeon RX 9070 XT" [GPU]\nCPU:0 "x" [CPU]\n',
            ),
            True,
        ),
        (_Ran(0, stdout = 'available_devices=1\nCPU:0 "x" [CPU]\n'), False),
        (_Ran(127, stderr = "error while loading shared libraries: libvulkan.so.1"), False),
        (OSError("bad image"), False),
        (M.subprocess.TimeoutExpired("audiocpp_server", 30), False),
    ],
)
def test_the_vulkan_check_needs_a_listed_vulkan_gpu(monkeypatch, tmp_path, listed, has_device):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[1])
        if cmd[1] == "--list-devices":
            if isinstance(listed, BaseException):
                raise listed
            return listed
        return _Ran(0, stdout = "audio.cpp vk\nbackends: cpu,vulkan" if cmd[1] == "--version" else "")

    monkeypatch.setattr(M.subprocess, "run", run)
    server = tmp_path / M.SERVER_NAME
    if has_device:
        assert (
            SMOKE(server, backend = "vulkan", require_vulkan_device = True)
            == "audio.cpp vk\nbackends: cpu,vulkan"
        )
    else:
        with pytest.raises(M.NoVulkanDevice):
            SMOKE(server, backend = "vulkan", require_vulkan_device = True)
    calls.clear()
    # Not asked (an explicit request, another backend): devices are never listed.
    SMOKE(server, backend = "vulkan")
    assert "--list-devices" not in calls


VK_ZIP = f"audio-{FORK_TAG}-bin-windows-x64-vulkan.zip"


def _vulkan_host(monkeypatch, tmp_path, *, device):
    cpu = _release(tmp_path / "cpu", CPU_ZIP)
    vk = _release(tmp_path / "vk", VK_ZIP)
    asked = []

    def resolve(
        accel,
        token,
        tag = None,
    ):
        asked.append(accel)
        rel = vk if accel == "vulkan" else cpu
        return FORK, rel, rel["assets"][0]["name"]

    _stub_install_io(monkeypatch)
    checked = []

    def smoke(
        server,
        *,
        backend,
        extra_dirs = (),
        require_vulkan_device = False,
    ):
        checked.append((backend, require_vulkan_device))
        if require_vulkan_device and not device:
            raise M.NoVulkanDevice("the Vulkan build finds no Vulkan GPU on this host")
        return "audio.cpp test"

    monkeypatch.setattr(M, "smoke_test_staged_server", smoke)
    monkeypatch.setattr(M, "resolve", resolve)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "vulkan")
    return asked, checked


def _record(tmp_path):
    rec = json.loads((tmp_path / "audio.cpp" / M.INSTALL_RECORD).read_text())
    return rec["asset"], rec["backend"], rec["accelerator_request"], rec["detected_accelerator"]


def test_auto_on_an_amd_gpu_installs_the_vulkan_bundle(monkeypatch, tmp_path):
    asked, checked = _vulkan_host(monkeypatch, tmp_path, device = True)
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert asked == ["vulkan"] and checked == [("vulkan", True)]
    assert _record(tmp_path) == (VK_ZIP, "vulkan", "auto", "vulkan")


def test_auto_vulkan_without_a_vulkan_gpu_installs_the_cpu_build_once(
    monkeypatch, tmp_path, capsys
):
    asked, checked = _vulkan_host(monkeypatch, tmp_path, device = False)
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert asked == ["vulkan", "cpu"]
    assert checked == [("vulkan", True), ("cpu", False)]
    assert "finds no Vulkan GPU on this host; installing the CPU build" in capsys.readouterr().out
    assert _record(tmp_path) == (CPU_ZIP, "cpu", "auto", "vulkan")
    # The record names the detected Vulkan host, so the next setup run is a no-op.
    monkeypatch.setattr(
        M, "resolve", lambda *a, **k: pytest.fail("a matching rerun must not look up")
    )
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK


def test_an_existing_cpu_install_on_an_amd_gpu_moves_to_vulkan(monkeypatch, tmp_path):
    # What every AMD host has before #13042: auto, detected "cpu", the CPU bundle.
    asked, _ = _vulkan_host(monkeypatch, tmp_path, device = True)
    monkeypatch.setattr(M, "detect_accelerator", lambda: "cpu")
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert _record(tmp_path) == (CPU_ZIP, "cpu", "auto", "cpu")
    monkeypatch.setattr(M, "detect_accelerator", lambda: "vulkan")
    assert M.main(["--install-dir", str(tmp_path / "audio.cpp")]) == M.EXIT_OK
    assert _record(tmp_path) == (VK_ZIP, "vulkan", "auto", "vulkan")


def test_explicit_vulkan_is_not_checked_for_a_device(monkeypatch, tmp_path):
    # A Docker build sees no GPU; an explicit request installs what it names.
    asked, checked = _vulkan_host(monkeypatch, tmp_path, device = False)
    assert (
        M.main(["--install-dir", str(tmp_path / "audio.cpp"), "--accelerator", "vulkan"])
        == M.EXIT_OK
    )
    assert checked == [("vulkan", False)]
    assert _record(tmp_path)[:3] == (VK_ZIP, "vulkan", "vulkan")
