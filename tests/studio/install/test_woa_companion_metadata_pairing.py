# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stable win_arm64 torchvision pairs with the torch its own METADATA admits, not by a version offset.

The out-of-tree index ships torch 2.15.0+cu135 beside torchvision 0.29.2+cu135, whose METADATA says
``Requires-Dist: torch>=2.14.0``. The offset rule (torchvision 0.(M+15) for torch 2.M) rejected that pair
and the installer walked back to torch 2.14.0+cu134. The listings under fixtures/woa_torch_index are
recorded pages of that index; every wheel read is served from a zip built here, so nothing reaches
the network."""

from __future__ import annotations

import hashlib
import pathlib
import zipfile

import pytest

from woa_ps_harness import (
    INSTALL_SRC,
    JOIN_URL_PATH,
    NV_GA,
    NV_NIGHTLY,
    SETUP_SRC,
    _function_source,
    _ps_kv,
    _ps_last,
    _script,
    functions,
    native_probe_script,
    requires_pwsh,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "woa_torch_index"
METADATA = FIXTURES / "metadata"

# What the native route lifts out of install.ps1 to run the real probe against served pages.
PROBE_LIFTS = (
    "Test-WoaWheelTags",
    "Test-WoaWheelTagsUsable",
    "Test-WoaWheelPairsWithTorch",
    "Get-WoaMetadataTorchRequirement",
    "Get-WoaWheelTorchRequirement",
    "Get-WoaCudaWheelVersion",
    "Test-WoaCudaWheel",
)


def _ps_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _pages(indexes: dict) -> str:
    """Invoke-RestMethod serving recorded project pages, keyed by the project URL the probe builds."""
    rows = []
    for index_url, folder in indexes.items():
        for project in ("torch", "torchvision", "torchaudio"):
            rows.append(
                f"{_ps_str(f'{index_url}/{project}/')} = {_ps_str(str(FIXTURES / folder / f'{project}.html'))}"
            )
    return _script(
        "$script:Pages = @{ " + "; ".join(rows) + " }",
        "function Invoke-RestMethod { param($Uri, [switch]$UseBasicParsing, $TimeoutSec)",
        "  $p = $script:Pages[[string]$Uri]",
        "  if (-not $p) { throw ('not served: ' + $Uri) }",
        "  return [System.IO.File]::ReadAllText($p) }",
    )


def _http(wheels: dict, *, fail: bool = False) -> str:
    """Invoke-WoaHttpBytes serving byte ranges of local files, by wheel (or .metadata) file name."""
    rows = "; ".join(f"{_ps_str(name)} = {_ps_str(str(path))}" for name, path in wheels.items())
    return _script(
        "$script:HttpCalls = @()",
        f"$script:HttpFails = ${str(fail).lower()}",
        "$script:Wheels = @{ " + rows + " }",
        "function Invoke-WoaHttpBytes {",
        "  param([string]$Uri, [long]$Offset = -1, [long]$Length = 0, [long]$Suffix = 0, [long]$MaxBytes = 1048576)",
        "  $script:HttpCalls += $Uri",
        "  if ($script:HttpFails) { return $null }",
        "  $path = $script:Wheels[[System.Uri]::UnescapeDataString(($Uri -split '/')[-1])]",
        "  if (-not $path) { return $null }",
        "  $all = [System.IO.File]::ReadAllBytes($path); $n = [long]$all.Length",
        "  if ($Suffix -gt 0) { $s = [Math]::Max(0, $n - $Suffix); $e = $n - 1 }",
        "  elseif ($Offset -ge 0) { $s = $Offset; $e = [Math]::Min($n - 1, $Offset + $Length - 1) }",
        "  else { $s = 0; $e = $n - 1 }",
        "  return @{ Bytes = [byte[]]$all[$s..$e]; Total = $n } }",
    )


def _filler(size: int) -> bytes:
    """Deterministic incompressible bytes, so the archive's directory sits where a real wheel's does."""
    out = bytearray()
    seed = b"woa"
    while len(out) < size:
        seed = hashlib.sha256(seed).digest()
        out += seed
    return bytes(out[:size])


def _wheel(
    directory: pathlib.Path,
    name: str,
    metadata: str,
    *,
    entries: int = 8,
) -> pathlib.Path:
    """A wheel laid out like the real one: dist-info METADATA early, payload after, directory last."""
    version = name.split("-")[1]
    path = directory / name
    with zipfile.ZipFile(path, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("torchvision/__init__.py", "")
        zf.writestr(f"torchvision-{version}.dist-info/METADATA", metadata)
        for i in range(entries):
            zf.writestr(f"torchvision/_payload/{'x' * 80}_{i:05d}.pyd", _filler(256))
        zf.writestr("torchvision/_C.pyd", _filler(96 * 1024), compress_type = zipfile.ZIP_STORED)
    return path


def _recorded(name: str) -> str:
    return (METADATA / name.replace(".whl", ".METADATA")).read_text(encoding = "utf-8")


STABLE_VISION = (
    "torchvision-0.29.2+cu135-py3-none-win_arm64.whl",
    "torchvision-0.29.2+cu134-py3-none-win_arm64.whl",
    "torchvision-0.29.0+cu134-cp313-cp313-win_arm64.whl",
)


def _stable_wheels(tmp_path, overrides: dict | None = None) -> dict:
    wheels = {}
    for name in STABLE_VISION:
        text = (overrides or {}).get(name, _recorded(name))
        wheels[name] = _wheel(tmp_path, name, text)
    return wheels


def _native(
    indexes: dict,
    wheels: dict,
    *,
    driver: str = "@(13, 5)",
    fail: bool = False,
) -> dict:
    script = native_probe_script(
        driver = driver,
        stubs = (
            "$script:WoaNvidiaTorchIndexUrls = @(" + ", ".join(_ps_str(u) for u in indexes) + ")",
            JOIN_URL_PATH,
            "function Remove-IndexUrlCredentials { param($Url) $Url }",
            _pages(indexes),
            _http(wheels, fail = fail),
        ),
        lifts = PROBE_LIFTS,
        outputs = (
            "Write-Output ('INDEX=' + $script:WoaTorchIndexUrl)",
            "Write-Output ('TORCH=' + $script:WoaTorchWheelVersion)",
            "Write-Output ('VISION=' + $script:WoaVisionWheelVersion)",
            "Write-Output ('HTTP=' + ($script:HttpCalls -join ' '))",
        ),
    )
    return _ps_kv(script)


@requires_pwsh
class TestTheRecordedIndexPairsTheNewestTorch:
    def test_stable_listing_takes_torch_2_15_cu135(self, tmp_path):
        out = _native({NV_GA: "stable"}, _stable_wheels(tmp_path))
        assert out["NATIVE"] == "True", out["MSG"]
        assert out["INDEX"] == NV_GA
        assert (out["TORCH"], out["VISION"]) == ("2.15.0+cu135", "0.29.2+cu135"), out["MSG"]
        assert "trying torch" not in out["MSG"]
        # Only the one companion that shares torch's build tag is read, never a torch wheel.
        assert out["HTTP"] and "torch-2." not in out["HTTP"]
        assert all(
            "torchvision-0.29.2%2Bcu135" in u or "0.29.2+cu135" in u for u in out["HTTP"].split()
        )

    def test_a_13_4_driver_takes_the_cu134_build_of_the_same_release(self, tmp_path):
        out = _native({NV_GA: "stable"}, _stable_wheels(tmp_path), driver = "@(13, 4)")
        assert (out["TORCH"], out["VISION"]) == ("2.15.0+cu134", "0.29.2+cu134"), out["MSG"]

    def test_nightly_listing_pairs_by_stamp_without_reading_any_wheel(self, tmp_path):
        out = _native({NV_NIGHTLY: "nightly"}, {})
        assert out["NATIVE"] == "True", out["MSG"]
        assert (out["TORCH"], out["VISION"]) == (
            "2.16.0.dev20261127+cu135",
            "0.30.0.dev20261127+cu135",
        ), out["MSG"]
        assert out["HTTP"] == ""

    def test_stable_index_still_outranks_nightly(self, tmp_path):
        out = _native({NV_GA: "stable", NV_NIGHTLY: "nightly"}, _stable_wheels(tmp_path))
        assert out["INDEX"] == NV_GA
        assert (out["TORCH"], out["VISION"]) == ("2.15.0+cu135", "0.29.2+cu135")

    def test_a_companion_that_excludes_the_newest_torch_walks_back(self, tmp_path):
        capped = {}
        for name in STABLE_VISION[:2]:
            capped[name] = _recorded(name).replace(
                "Requires-Dist: torch>=2.14.0", "Requires-Dist: torch<2.15,>=2.14.0"
            )
        out = _native({NV_GA: "stable"}, _stable_wheels(tmp_path, capped))
        assert out["NATIVE"] == "True", out["MSG"]
        assert (out["TORCH"], out["VISION"]) == ("2.14.0+cu134", "0.29.2+cu134"), out["MSG"]
        assert (
            "publishes torch 2.15.0+cu135 but no torchvision paired with it; trying torch 2.14.0"
            in out["MSG"]
        )

    def test_unreadable_metadata_falls_back_to_the_offset_rule(self, tmp_path):
        out = _native({NV_GA: "stable"}, _stable_wheels(tmp_path), fail = True)
        assert out["HTTP"], "the metadata read was never attempted"
        # Offset: torch 2.15 wants torchvision 0.30, absent, so the walk ends at 2.14 / 0.29.
        assert (out["TORCH"], out["VISION"]) == ("2.14.0+cu134", "0.29.2+cu134"), out["MSG"]


@requires_pwsh
class TestTheMetadataRead:
    def _requirement(
        self,
        wheels: dict,
        anchor_attrs: str = "",
    ) -> tuple:
        name = STABLE_VISION[0]
        href = name.replace("+", "%2B")
        script = _script(
            _http(wheels),
            functions(
                INSTALL_SRC, "Get-WoaMetadataTorchRequirement", "Get-WoaWheelTorchRequirement"
            ),
            f"$r = Get-WoaWheelTorchRequirement -WheelUrl 'https://i.test/torchvision/{href}'"
            f" -HasCoreMetadata ${'true' if anchor_attrs else 'false'}",
            "Write-Output ('REQ=' + $r)",
            "Write-Output ('HTTP=' + $script:HttpCalls.Count)",
        )
        out = _ps_kv(script)
        return out["REQ"], int(out["HTTP"])

    def test_the_central_directory_inside_the_tail(self, tmp_path):
        name = STABLE_VISION[0]
        req, calls = self._requirement({name: _wheel(tmp_path, name, _recorded(name))})
        assert req == ">=2.14.0"
        assert calls == 3, "tail, local header, METADATA: never the whole wheel"

    def test_a_directory_larger_than_the_tail_is_fetched_by_range(self, tmp_path):
        name = STABLE_VISION[0]
        wheel = _wheel(tmp_path, name, _recorded(name), entries = 1200)
        with zipfile.ZipFile(wheel) as zf:
            assert sum(46 + len(i.filename) for i in zf.infolist()) > 65558
        req, calls = self._requirement({name: wheel})
        assert req == ">=2.14.0"
        assert calls == 4

    def test_pep_658_metadata_is_used_when_the_index_serves_it(self, tmp_path):
        name = STABLE_VISION[0]
        meta = tmp_path / (name + ".metadata")
        meta.write_text(_recorded(name).replace(">=2.14.0", "==2.15.0"), encoding = "utf-8")
        req, calls = self._requirement({name + ".metadata": meta}, anchor_attrs = "core")
        assert (req, calls) == ("==2.15.0", 1)

    def test_a_missing_wheel_reads_as_unknown(self, tmp_path):
        assert self._requirement({}) == ("", 1)


@requires_pwsh
@pytest.mark.parametrize(
    "metadata, expected",
    [
        (_recorded(STABLE_VISION[0]), ">=2.14.0"),
        (_recorded(STABLE_VISION[2]), ""),
        ("Name: x\nRequires-Dist: torch (>=2.14,<2.16)\n", ">=2.14,<2.16"),
        ("Name: x\nRequires-Dist: torchaudio>=2.0\nRequires-Dist: torch==2.15.*\n", "==2.15.*"),
        ('Name: x\nRequires-Dist: torch>=9; extra == "full"\n', ""),
        ("Name: x\n\nRequires-Dist: torch>=9\n", ""),
    ],
    ids = [
        "recorded-0.29.2",
        "recorded-0.29.0",
        "parenthesised",
        "among-others",
        "marker-only",
        "body-not-header",
    ],
)
def test_the_torch_requirement_is_read_from_the_header(metadata, expected):
    script = _script(
        _function_source(INSTALL_SRC, "Get-WoaMetadataTorchRequirement"),
        f"Write-Output ('[' + (Get-WoaMetadataTorchRequirement -Metadata {_ps_str(metadata)}) + ']')",
    )
    assert _ps_last(script)[1:-1] == expected


@requires_pwsh
@pytest.mark.parametrize(
    "torch_v, other_v, requirement, pairs",
    [
        ("2.15.0+cu135", "0.29.2+cu135", ">=2.14.0", True),
        ("2.15.0+cu135", "0.29.2+cu134", ">=2.14.0", False),
        ("2.15.0+cu135", "0.29.2+cu135", "", False),
        ("2.14.0+cu134", "0.29.0+cu134", "", True),
        ("2.15.0+cu135", "0.29.2+cu135", "<2.15,>=2.14.0", False),
        ("2.15.0+cu135", "0.29.2+cu135", "==2.15.*", True),
        ("2.16.0+cu135", "0.30.0+cu135", "~=2.15.0", False),
        ("2.15.1+cu135", "0.30.0+cu135", "~=2.15.0", True),
        ("2.15.0+cu135", "0.30.0+cu135", "!=2.15.0", False),
        # A clause this cannot read falls back to the offset rule rather than guessing.
        ("2.14.0+cu134", "0.29.0+cu134", ">=2.14.0rc1", True),
        ("2.15.0+cu135", "0.29.2+cu135", ">=2.14.0rc1", False),
        # Nightlies keep the stamp rule whatever the metadata says.
        ("2.16.0.dev20261127+cu135", "0.30.0.dev20261127+cu135", "==9.9", True),
        ("2.16.0.dev20261127+cu135", "0.30.0.dev20261126+cu135", ">=2.0", False),
    ],
)
def test_what_a_declared_requirement_pairs(torch_v, other_v, requirement, pairs):
    for source, fn in (
        (INSTALL_SRC, "Test-WoaWheelPairsWithTorch"),
        (SETUP_SRC, "Test-WoaPairsWithTorchParity"),
    ):
        script = _script(
            _function_source(source, fn),
            f"Write-Output ({fn} -TorchVersion '{torch_v}' -OtherVersion '{other_v}'"
            f" -TorchRequirement {_ps_str(requirement)})",
        )
        assert (_ps_last(script) == "True") is pairs, (fn, torch_v, other_v, requirement)


def _body(text: str) -> list:
    lines = text.split("\n")[1:]
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


@pytest.mark.parametrize(
    "install_fn, setup_fn",
    [
        ("Invoke-WoaHttpBytes", "Invoke-WoaHttpBytesParity"),
        ("Get-WoaMetadataTorchRequirement", "Get-WoaMetadataTorchRequirementParity"),
        ("Test-WoaWheelPairsWithTorch", "Test-WoaPairsWithTorchParity"),
    ],
)
def test_the_new_parity_copies_have_not_drifted(install_fn, setup_fn):
    assert _body(_function_source(INSTALL_SRC, install_fn)) == _body(
        _function_source(SETUP_SRC, setup_fn)
    )


def test_the_usable_tag_parity_copy_differs_only_in_its_callee():
    install = _body(_function_source(INSTALL_SRC, "Test-WoaWheelTagsUsable"))
    setup = [
        l.replace("Test-WoaWheelTagsParity", "Test-WoaWheelTags")
        for l in _body(_function_source(SETUP_SRC, "Test-WoaWheelTagsUsableParity"))
    ]
    assert install == setup


def test_the_wheel_reader_parity_copy_differs_only_in_its_callees():
    install = _body(_function_source(INSTALL_SRC, "Get-WoaWheelTorchRequirement"))
    setup = [
        l.replace("Invoke-WoaHttpBytesParity", "Invoke-WoaHttpBytes")
        .replace("Get-WoaMetadataTorchRequirementParity", "Get-WoaMetadataTorchRequirement")
        .replace("WoaWheelTorchRequirementCacheParity", "WoaWheelTorchRequirementCache")
        for l in _body(_function_source(SETUP_SRC, "Get-WoaWheelTorchRequirementParity"))
    ]
    assert install == setup


@requires_pwsh
@pytest.mark.parametrize(
    "folder, expected", [("stable", "0.29.2+cu135"), ("nightly", "0.30.0.dev20261127+cu135")]
)
def test_the_repair_probe_in_setup_pairs_the_same_way(tmp_path, folder, expected):
    index = NV_GA if folder == "stable" else NV_NIGHTLY
    torch_v = "2.15.0+cu135" if folder == "stable" else "2.16.0.dev20261127+cu135"
    script = _script(
        _pages({index: folder}),
        _http(_stable_wheels(tmp_path)),
        functions(
            SETUP_SRC,
            "Test-WoaWheelTagsParity",
            "Test-WoaWheelTagsUsableParity",
            "Test-WoaPairsWithTorchParity",
            "Get-WoaMetadataTorchRequirementParity",
            "Get-WoaWheelTorchRequirementParity",
            "Get-WoaCudaWheelVersionParity",
        ),
        "function Invoke-WoaHttpBytesParity { param([string]$Uri, [long]$Offset = -1, [long]$Length = 0,"
        " [long]$Suffix = 0, [long]$MaxBytes = 1048576) Invoke-WoaHttpBytes -Uri $Uri -Offset $Offset"
        " -Length $Length -Suffix $Suffix -MaxBytes $MaxBytes }",
        f"$t = Get-WoaCudaWheelVersionParity -IndexUrl '{index}' -PyTag 'cp313' -AbiTag 'cp313' -MaxCuda '13.5'",
        f"$v = Get-WoaCudaWheelVersionParity -IndexUrl '{index}' -PyTag 'cp313' -AbiTag 'cp313'"
        " -Project 'torchvision' -PairWith $t",
        "Write-Output ('TORCH=' + $t)",
        "Write-Output ('VISION=' + $v)",
    )
    out = _ps_kv(script)
    assert (out["TORCH"], out["VISION"]) == (torch_v, expected)


def test_the_repair_keeps_an_installed_pair_its_own_metadata_admits():
    """setup.ps1 keeps the installed trio when the index pairs nothing; that check reads the installed
    torchvision's torch requirement too, or torch 2.15.0 + torchvision 0.29.2 is never kept."""
    start = SETUP_SRC.index("$_woaInstalledPair = (& (Join-Path $VenvDir")
    line = SETUP_SRC[start : SETUP_SRC.index("\n", start)]
    code = line.split('-c "', 1)[1].rsplit('" 2>$null', 1)[0]
    assert code.startswith("import importlib.metadata as m, re; ")
    keep = SETUP_SRC[start : SETUP_SRC.index("substep", start)]
    assert "-TorchRequirement $_woaKeptRequirement" in keep

    class Meta:
        @staticmethod
        def version(name):
            return {"torch": "2.15.0+cu135", "torchvision": "0.29.2+cu135"}[name]

        @staticmethod
        def requires(name):
            return ["numpy", "torch >= 2.14.0", "torchaudio>=2", 'gdown>=4.7.3; extra == "gdown"']

    import contextlib
    import io
    import re

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code.split("; ", 1)[1], {"m": Meta, "re": re})
    assert out.getvalue().strip() == "2.15.0+cu135|0.29.2+cu135|>=2.14.0"
