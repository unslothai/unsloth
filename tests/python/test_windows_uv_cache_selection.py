# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 must select a uv cache by the same rules install.sh does.

install.sh and install.ps1 hold two copies of one decision, and the copies drift silently
because nothing runs them against the same inputs. tests/sh/test_uv_cache_adaptive_selection.sh
covers the POSIX half; this file is the Windows half, and it asserts BEHAVIOUR by running
install.ps1's own selector, not text.

The four rules below were added to install.sh first and did not exist in install.ps1. Measured
against install.ps1 before this file: a rerun on Windows abandoned a warm Studio cache the moment
one unrelated wheel made uv's default read as warm (re-downloading the Torch and CUDA bytes it
already held), UV_NO_CACHE still probed and recorded a cache uv was not using, `archive-v0.backup`
counted as warmth, and a populated-but-unwritable cache was selected -- which uv aborts on, so
that one failed the install outright.

Runs under pwsh on any OS: the selector is filesystem logic, and the Linux and macOS CI jobs
carry pwsh, so the Windows path is exercised on every platform CI has rather than only on the
Windows runners.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell is unavailable")

# The selector and everything it calls, in source order. Sliced rather than copied, so a change to
# install.ps1 that this file does not expect fails here instead of passing against a stale paste.
_FIRST = "    function Resolve-StudioUvCachePath {"
_LAST = "    function Restore-StudioUvCacheEnvironment {"


def _selector_source() -> str:
    source = INSTALL_PS1.read_text(encoding="utf-8")
    start = source.index(_FIRST)
    end = source.index(_LAST)
    # Dedented one level: the functions live inside Install-UnslothStudio in the real file.
    return "\n".join(
        line[4:] if line.startswith("    ") else line for line in source[start:end].splitlines()
    )


def _ps_literal(value: str) -> str:
    """A PowerShell single-quoted string.

    Not json.dumps: a JSON string is double-quoted, and PowerShell does not treat `\\` in a
    double-quoted string as an escape, so `C:\\Users\\x` arrives as a path with DOUBLED
    separators. Every Windows assertion in this file then compared a doubled path against a
    real one, and the whole file only passed because POSIX paths have no backslashes to
    double. A single-quoted PowerShell string is literal; `''` is the only escape it has.
    """
    return "'" + value.replace("'", "''") + "'"


def _script(studio_root: Path, uv_stub: Path, isolated: bool) -> str:
    """Run the real selector once and print its verdict as JSON."""
    return f"""
$ErrorActionPreference = "Stop"
function step {{ param($a, $b, $c) }}
$script:StudioUvMarkerSaved = $false
$script:StudioUvMarkerExisted = $false
$script:StudioUvMarkerPrevious = $null
$script:StudioInstallCommitted = $false
{_selector_source()}
$root = {_ps_literal(str(studio_root))}
$stub = {_ps_literal(str(uv_stub))}
Set-StudioUvCacheEnvironment -StudioRoot $root -Isolated ${str(isolated).lower()} -UvExecutable $stub
$mode = $script:StudioUvCacheMode
$dir = $env:UV_CACHE_DIR
Set-StudioUvCacheForLaunch -StudioRoot $root
$markerFile = Join-Path (Join-Path $root "cache") "uv-cache-dir"
$marker = ""
if (Test-Path -LiteralPath $markerFile) {{
    $marker = (Get-Content -LiteralPath $markerFile -Raw).Trim()
}}
[pscustomobject]@{{ mode = $mode; dir = $dir; launch = $env:UV_CACHE_DIR; marker = $marker }} |
    ConvertTo-Json -Compress
"""


def _select(
    studio_root: Path,
    uv_answer: str,
    *,
    isolated: bool = False,
    env: dict | None = None,
) -> dict:
    studio_root.mkdir(parents=True, exist_ok=True)
    # A .ps1 rather than a shell script, so this runs on the Windows agents too; `exit 0` so
    # $LASTEXITCODE is set, which is what the selector reads before trusting the answer.
    stub = studio_root.parent / "uv-stub.ps1"
    stub.write_text(f"Write-Output {_ps_literal(uv_answer)}\nexit 0\n", encoding="utf-8")
    merged = {
        key: value
        for key, value in os.environ.items()
        if key not in ("UV_CACHE_DIR", "UV_NO_CACHE")
    }
    merged.update(env or {})
    # run_pwsh, not subprocess.run: a pwsh that dies at startup would read here as the selector
    # answering the wrong mode. See tests/_shared/unsloth_pwsh_runner.py.
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", _script(studio_root, stub, isolated)],
        check=True,
        capture_output=True,
        text=True,
        env=merged,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _warm(cache: Path) -> Path:
    """A cache holding package BYTES, which is the only thing that counts as warm."""
    (cache / "archive-v0" / "torch").mkdir(parents=True, exist_ok=True)
    (cache / "archive-v0" / "torch" / "libtorch.so").write_text("x")
    (cache / "CACHEDIR.TAG").write_text("x")
    return cache


def _cold(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "CACHEDIR.TAG").write_text("x")
    return cache


def _record(studio_root: Path, cache: Path) -> None:
    (studio_root / "cache").mkdir(parents=True, exist_ok=True)
    (studio_root / "cache" / "uv-cache-dir").write_text(f"{cache}\n", encoding="utf-8")


# ── the modes that already worked, so a regression in the port is visible ──────────────────


@requires_pwsh
def test_a_warm_default_is_reused_and_the_launch_repoints(tmp_path):
    default = _warm(tmp_path / "uvdefault")
    root = tmp_path / "studio"
    verdict = _select(root, str(default))
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(default), verdict
    # The install fills the shared cache; the launch must not leave the backend pointed at it.
    assert verdict["launch"] == str(root / "cache" / "uv"), verdict
    # And the marker names what the INSTALL used, which is what the update reads.
    assert verdict["marker"] == str(default), verdict


@requires_pwsh
def test_a_cold_default_falls_back_to_the_studio_cache(tmp_path):
    verdict = _select(tmp_path / "studio", str(_cold(tmp_path / "uvdefault")))
    assert verdict["mode"] == "studio", verdict
    assert verdict["dir"] == str(tmp_path / "studio" / "cache" / "uv"), verdict


@requires_pwsh
def test_metadata_only_is_not_warm(tmp_path):
    default = tmp_path / "uvdefault"
    (default / "wheels-v1").mkdir(parents=True)
    (default / "wheels-v1" / "index.msgpack").write_text("x")
    assert _select(tmp_path / "studio", str(default))["mode"] == "studio"


@requires_pwsh
def test_a_caller_override_still_outranks_everything(tmp_path):
    mine = _warm(tmp_path / "mine")
    verdict = _select(
        tmp_path / "studio", str(_warm(tmp_path / "uvdefault")), env={"UV_CACHE_DIR": str(mine)}
    )
    assert verdict["mode"] == "custom", verdict
    assert verdict["dir"] == str(mine), verdict
    # No repoint for a cache the caller chose.
    assert verdict["launch"] == str(mine), verdict


@requires_pwsh
def test_isolation_wins_over_a_warm_default(tmp_path):
    root = tmp_path / "studio"
    verdict = _select(root, str(_warm(tmp_path / "uvdefault")), isolated=True)
    assert verdict["mode"] == "isolated", verdict
    assert verdict["dir"] == str(root / "cache" / "uv"), verdict
    # Isolation records too, or a previous install's marker outlives the choice.
    assert verdict["marker"] == str(root / "cache" / "uv"), verdict


@requires_pwsh
@pytest.mark.parametrize("isolated", [False, True])
def test_uv_no_cache_records_nothing_even_when_the_directory_is_decided(tmp_path, isolated):
    """uv leaves the chosen directory completely empty under --no-cache.

    Both branches that pick a directory without probing returned before the no-cache check, so
    the marker named a cache this install never filled and the next repair could prefer it.
    """
    root = tmp_path / "studio"
    env = {"UV_NO_CACHE": "1"}
    if not isolated:
        env["UV_CACHE_DIR"] = str(tmp_path / "callercache")
    verdict = _select(root, str(_warm(tmp_path / "uvdefault")), isolated=isolated, env=env)
    assert verdict["mode"] == ("isolated" if isolated else "custom"), verdict
    assert verdict["marker"] == "", verdict


@requires_pwsh
@pytest.mark.parametrize("isolated", [False, True])
def test_the_same_two_branches_do_record_without_uv_no_cache(tmp_path, isolated):
    """So the guard above is not simply switching recording off."""
    root = tmp_path / "studio"
    env = {} if isolated else {"UV_CACHE_DIR": str(tmp_path / "callercache")}
    verdict = _select(root, str(_warm(tmp_path / "uvdefault")), isolated=isolated, env=env)
    assert verdict["marker"] != "", verdict


# ── the four rules install.sh gained and install.ps1 did not ───────────────────────────────


@requires_pwsh
def test_a_warm_recorded_cache_outranks_uvs_default(tmp_path):
    """The rerun case. One unrelated wheel in uv's default must not cost a Torch redownload."""
    root = tmp_path / "studio"
    studio_cache = _warm(root / "cache" / "uv")
    _record(root, studio_cache)
    verdict = _select(root, str(_warm(tmp_path / "uvdefault")))
    # studio, not shared: the launch repoint only has to move a cache that is not already ours.
    assert verdict["mode"] == "studio", verdict
    assert verdict["dir"] == str(studio_cache), verdict


@requires_pwsh
def test_a_recorded_cache_that_went_cold_loses_to_the_default(tmp_path):
    """...and a machine that really has moved on still reaches the shared cache."""
    root = tmp_path / "studio"
    _record(root, tmp_path / "vanished")
    default = _warm(tmp_path / "uvdefault")
    assert _select(root, str(default))["dir"] == str(default)


@requires_pwsh
def test_isolation_still_wins_over_a_warm_marker(tmp_path):
    root = tmp_path / "studio"
    _record(root, _warm(root / "cache" / "uv"))
    assert _select(root, str(_warm(tmp_path / "uvdefault")), isolated=True)["mode"] == "isolated"


@requires_pwsh
@pytest.mark.parametrize("value", ["1", "y", "Y", "t", "T", "true", "TRUE", "Yes", "on"])
def test_uv_no_cache_stands_the_selection_down(tmp_path, value):
    root = tmp_path / "studio"
    verdict = _select(root, str(_warm(tmp_path / "uvdefault")), env={"UV_NO_CACHE": value})
    assert verdict["mode"] == "studio", verdict
    # Nothing recorded: a marker written here would name a cache this install never filled.
    assert verdict["marker"] == "", verdict


@requires_pwsh
@pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
def test_a_false_uv_no_cache_does_not_stand_it_down(tmp_path, value):
    root = tmp_path / "studio"
    assert (
        _select(root, str(_warm(tmp_path / "uvdefault")), env={"UV_NO_CACHE": value})["mode"]
        == "shared"
    )


@requires_pwsh
@pytest.mark.parametrize("name", ["archive-v0.backup", "archive-backup-v0", "archive-v0.tar.gz"])
def test_a_bucket_lookalike_is_not_warmth(tmp_path, name):
    """`archive-*` matches all three, and uv can reuse the bytes in none of them."""
    default = tmp_path / "uvdefault"
    (default / name / "pkg").mkdir(parents=True)
    (default / name / "pkg" / "payload.so").write_text("x")
    (default / "CACHEDIR.TAG").write_text("x")
    assert _select(tmp_path / "studio", str(default))["mode"] == "studio"


@requires_pwsh
def test_a_studio_cache_with_an_unusable_bucket_is_not_a_fallback(tmp_path):
    """The root can be writable while a bucket uv renames into is not.

    A file where a bucket belongs is an existing path to the create uv makes, so uv refuses it.
    Selecting that cache anyway turns an install that reported success into a uv error.
    """
    root = tmp_path / "studio"
    (root / "cache" / "uv").mkdir(parents=True)
    (root / "cache" / "uv" / "archive-v0").write_text("not a directory")
    cold = tmp_path / "colddefault"
    cold.mkdir()
    verdict = _select(root, str(cold))
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(cold), verdict


@requires_pwsh
def test_the_launch_does_not_repoint_into_a_rejected_studio_cache(tmp_path):
    """Selection already preferred the warm cache here; the repoint used to hand the backend
    the Studio cache regardless, because it only asked about the root."""
    root = tmp_path / "studio"
    (root / "cache" / "uv").mkdir(parents=True)
    (root / "cache" / "uv" / "archive-v0").write_text("not a directory")
    default = _warm(tmp_path / "uvdefault")
    verdict = _select(root, str(default))
    assert verdict["mode"] == "shared", verdict
    assert verdict["launch"] == str(default), verdict


@requires_pwsh
def test_an_intact_studio_cache_is_still_the_ordinary_answer(tmp_path):
    """So the two cases above are not simply switching the Studio cache off."""
    cold = tmp_path / "colddefault"
    cold.mkdir()
    assert _select(tmp_path / "studio", str(cold))["mode"] == "studio"


@requires_pwsh
def test_a_marker_naming_a_deleted_cache_is_no_more_evidence_than_none(tmp_path):
    """A stale pointer is not a decision, so it gets the same ordering as an absent marker:
    behind uv's default when that is warm, ahead of it when it is cold."""
    # A fresh root per half: the selector RECORDS its choice, so a second call against the
    # same root is no longer reading a stale marker.
    warm_root = tmp_path / "studio-warm"
    _warm(warm_root / "cache" / "uv")
    (warm_root / "cache" / "uv-cache-dir").write_text(
        str(tmp_path / "deleted-cache"), encoding="utf-8"
    )
    chosen = _select(warm_root, str(_warm(tmp_path / "uvdefault")))
    assert chosen["mode"] == "shared", chosen

    cold_root = tmp_path / "studio-cold"
    studio_cache = _warm(cold_root / "cache" / "uv")
    (cold_root / "cache" / "uv-cache-dir").write_text(
        str(tmp_path / "deleted-cache"), encoding="utf-8"
    )
    cold = tmp_path / "colddefault"
    cold.mkdir()
    chosen = _select(cold_root, str(cold))
    assert chosen["mode"] == "studio", chosen
    assert chosen["dir"].rstrip("\\/") == str(studio_cache).rstrip("\\/"), chosen


@requires_pwsh
def test_a_cold_recorded_cache_that_still_exists_falls_through(tmp_path):
    """The other half of the same rule: a machine that really has moved on still gets shared."""
    root = tmp_path / "studio"
    _warm(root / "cache" / "uv")
    cold = tmp_path / "coldrec"
    cold.mkdir()
    (root / "cache" / "uv-cache-dir").write_text(str(cold), encoding="utf-8")
    assert _select(root, str(_warm(tmp_path / "uvdefault")))["mode"] == "shared"


@requires_pwsh
def test_a_real_bucket_beside_a_lookalike_is_still_warm(tmp_path):
    default = _warm(tmp_path / "uvdefault")
    (default / "archive-v0.backup").mkdir()
    (default / "archive-v0.tar.gz").write_text("x")
    assert _select(tmp_path / "studio", str(default))["mode"] == "shared"


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root writes through the mode bits",
)
def test_a_populated_cache_we_cannot_write_is_refused(tmp_path):
    """uv renames distributions into the buckets and aborts when it cannot, so selecting one
    we cannot write fails the install rather than saving a download."""
    default = _warm(tmp_path / "uvdefault")
    default.chmod(0o555)
    try:
        verdict = _select(tmp_path / "studio", str(default))
    finally:
        default.chmod(0o755)
    assert verdict["mode"] == "studio", verdict


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root writes through the mode bits",
)
def test_an_unwritable_bucket_is_refused_too(tmp_path):
    """The `sudo -E` leftover: the root is ours and one bucket is not."""
    default = _warm(tmp_path / "uvdefault")
    (default / "builds-v0").mkdir()
    (default / "builds-v0").chmod(0o555)
    try:
        verdict = _select(tmp_path / "studio", str(default))
    finally:
        (default / "builds-v0").chmod(0o755)
    assert verdict["mode"] == "studio", verdict


@requires_pwsh
def test_the_write_probe_leaves_nothing_behind(tmp_path):
    """It writes into a cache uv is about to fill, and into the user's own shared one."""
    default = _warm(tmp_path / "uvdefault")
    _select(tmp_path / "studio", str(default))
    assert not list(default.rglob(".unsloth-write-probe.*"))


@requires_pwsh
def test_a_lost_and_found_does_not_condemn_the_cache(tmp_path):
    """A cache-dir pointed at a mount point carries one, and uv never writes it."""
    default = _warm(tmp_path / "uvdefault")
    (default / "lost+found").mkdir()
    if os.name != "nt":
        (default / "lost+found").chmod(0o000)
    try:
        assert _select(tmp_path / "studio", str(default))["mode"] == "shared"
    finally:
        if os.name != "nt":
            (default / "lost+found").chmod(0o755)


@requires_pwsh
def test_an_unknown_bucket_kind_does_not_condemn_the_cache(tmp_path):
    """Bucket-shaped but not a kind uv creates, so uv never opens it. Refusing the whole cache
    for it redownloaded what the cache already held."""
    default = _warm(tmp_path / "uvdefault")
    (default / "unused-v999").mkdir()
    if os.name != "nt":
        (default / "unused-v999").chmod(0o000)
    try:
        assert _select(tmp_path / "studio", str(default))["mode"] == "shared"
    finally:
        if os.name != "nt":
            (default / "unused-v999").chmod(0o755)


@requires_pwsh
def test_a_read_only_managed_python_bucket_still_condemns_the_cache(tmp_path):
    """python-v0 IS uv's, since 0.8.16, and the installer installs a managed CPython. Leaving
    it out of the kind list would select a cache the install then fails to write."""
    default = _warm(tmp_path / "uvdefault")
    (default / "python-v0").mkdir()
    if os.name != "nt":
        (default / "python-v0").chmod(0o555)
    try:
        assert _select(tmp_path / "studio", str(default))["mode"] == "studio"
    finally:
        if os.name != "nt":
            (default / "python-v0").chmod(0o755)


@requires_pwsh
def test_a_differently_cased_bucket_follows_the_filesystem(tmp_path):
    """Where the directory folds, `Python-V0` IS uv's python-v0 and an unwritable one has to
    condemn the cache. Where it does not, as on a case-sensitive NTFS directory or the Linux
    filesystem this runs on, it is a directory uv never opens and must not condemn anything.
    Measured the same way the selector measures it rather than assumed from the platform."""
    default = _warm(tmp_path / "uvdefault")
    (default / "Python-V0").mkdir()
    folds = (default / "python-v0").exists()
    if os.name != "nt":
        (default / "Python-V0").chmod(0o555)
    try:
        expected = "studio" if folds else "shared"
        assert _select(tmp_path / "studio", str(default))["mode"] == expected
    finally:
        if os.name != "nt":
            (default / "Python-V0").chmod(0o755)


@requires_pwsh
def test_a_bucket_that_is_not_a_directory_is_refused(tmp_path):
    """A file where a bucket goes is an existing path to the create uv makes, which fails."""
    default = _warm(tmp_path / "uvdefault")
    (default / "builds-v0").write_text("not a directory")
    assert _select(tmp_path / "studio", str(default))["mode"] == "studio"


# ── the installed base, and the ways the candidate loop can be knocked over ────────────────


@requires_pwsh
def test_an_unmarked_warm_studio_cache_is_kept_when_the_default_is_cold(tmp_path):
    """An install from before the marker can hold gigabytes of Torch and CUDA in the Studio
    cache with nothing recording it, and abandoning that costs the downloads again.

    It is the LAST candidate though, behind uv's default: the marker arrived in b66d2a4c8 and
    the installer's early UV_CACHE_DIR block only in e12963071 the day after, so an install
    with no marker is old enough that `shared` was reachable, and there the launch repoint
    leaves backend wheels in the Studio cache while the real bytes sit in the default.
    """
    root = tmp_path / "studio"
    studio_cache = _warm(root / "cache" / "uv")
    cold = tmp_path / "colddefault"
    cold.mkdir()
    verdict = _select(root, str(cold))
    assert verdict["mode"] == "studio", verdict
    assert verdict["dir"] == str(studio_cache), verdict


@requires_pwsh
def test_a_warm_default_outranks_an_unmarked_studio_cache(tmp_path):
    """The other half of the same rule, and what the CLI updater has always done."""
    root = tmp_path / "studio"
    _warm(root / "cache" / "uv")
    default = _warm(tmp_path / "uvdefault")
    verdict = _select(root, str(default))
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(default), verdict


@requires_pwsh
def test_an_unmarked_cold_studio_cache_does_not_block_the_shared_one(tmp_path):
    root = tmp_path / "studio"
    _cold(root / "cache" / "uv")
    default = _warm(tmp_path / "uvdefault")
    verdict = _select(root, str(default))
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(default), verdict


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root reads through the mode bits",
)
def test_a_marker_we_cannot_even_stat_does_not_cost_us_uvs_default(tmp_path):
    """Test-Path throws inside an ACL-denied directory under ErrorActionPreference = Stop, and
    the selector wraps the whole candidate loop, so an unreachable marker used to take uv's
    default down with it -- a first run elevated, a second one not."""
    locked = tmp_path / "locked"
    (locked / "uv").mkdir(parents=True)
    locked.chmod(0o000)
    root = tmp_path / "studio"
    _record(root, locked / "uv")
    default = _warm(tmp_path / "uvdefault")
    try:
        verdict = _select(root, str(default))
    finally:
        locked.chmod(0o755)
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(default), verdict


@requires_pwsh
def test_a_trailing_separator_in_the_marker_is_the_same_directory(tmp_path):
    """Otherwise it compares unequal to the Studio cache and gets the launch repoint that the
    studio branch exists to avoid."""
    root = tmp_path / "studio"
    studio_cache = _warm(root / "cache" / "uv")
    (root / "cache" / "uv-cache-dir").write_text(f"{studio_cache}{os.sep}\n", encoding="utf-8")
    assert _select(root, str(_warm(tmp_path / "uvdefault")))["mode"] == "studio"


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root writes through the mode bits",
)
def test_the_launch_does_not_repoint_at_a_studio_cache_it_cannot_fill(tmp_path):
    """Repointing at a cache uv aborts on hands the autostarted backend a dead cache after an
    install that succeeded. The shared cache is the one this install actually filled."""
    root = tmp_path / "studio"
    root.mkdir()
    default = _warm(tmp_path / "uvdefault")
    root.chmod(0o555)
    try:
        verdict = _select(root, str(default))
    finally:
        root.chmod(0o755)
    assert verdict["mode"] == "shared", verdict
    assert verdict["launch"] == str(default), verdict


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root writes through the mode bits",
)
def test_a_fallback_we_cannot_write_is_not_a_fallback(tmp_path):
    """The probe refuses a whole cache for one bucket uv may never touch, so a warm cache and an
    unwritable Studio root can both be on the table. Landing on the Studio root is a certain
    failure; the refused cache is only a suspect one."""
    default = _warm(tmp_path / "uvdefault")
    (default / "builds-v0").mkdir()
    (default / "builds-v0").chmod(0o555)
    root = tmp_path / "studio"
    root.mkdir()
    root.chmod(0o555)
    try:
        verdict = _select(root, str(default))
    finally:
        root.chmod(0o755)
        (default / "builds-v0").chmod(0o755)
    assert verdict["mode"] == "shared", verdict
    assert verdict["dir"] == str(default), verdict
    assert verdict["launch"] == str(default), verdict


@requires_pwsh
@pytest.mark.skipif(
    os.name == "nt", reason="mode bits; the Windows ACL equivalent needs a second account"
)
@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root writes through the mode bits",
)
def test_a_writable_studio_cache_still_wins_over_a_refused_one(tmp_path):
    """The rule above must not turn into "a refused cache always wins"."""
    default = _warm(tmp_path / "uvdefault")
    (default / "builds-v0").mkdir()
    (default / "builds-v0").chmod(0o555)
    try:
        verdict = _select(tmp_path / "studio", str(default))
    finally:
        (default / "builds-v0").chmod(0o755)
    assert verdict["mode"] == "studio", verdict
    assert verdict["dir"] == str(tmp_path / "studio" / "cache" / "uv"), verdict


# ── the harness's own guard ────────────────────────────────────────────────────────────────


@requires_pwsh
@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\runneradmin\AppData\Local\Temp\studio\cache\uv",
        r"\\server\share\uv",
        "/tmp/plain/posix/uv",
        "with a space/uv",
        "with'a'quote/uv",
    ],
)
def test_the_harness_hands_powershell_the_path_it_was_given(value):
    """A path handed to pwsh must arrive byte-identical.

    json.dumps looked right and was not: a JSON string is double-quoted, and PowerShell does
    not read `\\` in a double-quoted string as an escape, so every Windows path arrived with
    DOUBLED separators and every assertion in this file compared a doubled path against a real
    one. It passed anyway on Linux and macOS, where paths have no backslashes to double, so
    only a Windows runner could see it. This case is the Windows runner, in one line.
    """
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", f"Write-Output {_ps_literal(value)}"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip("\r\n") == value
