"""Host-macOS-version-aware llama.cpp prebuilt selection; Mach-O samples synthesized in-process, all I/O monkeypatched."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_macos", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

HostInfo = ILP.HostInfo
PrebuiltFallback = ILP.PrebuiltFallback

_CPU_TYPE_ARM64 = 0x0100000C
_CPU_TYPE_X86_64 = 0x01000007


def make_macos_host(macos_version, *, arm64 = True):
    return HostInfo(
        system = "Darwin",
        machine = "arm64" if arm64 else "x86_64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = not arm64,
        is_arm64 = arm64,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        macos_version = macos_version,
    )


def thin_macho(
    minos = (14, 0),
    *,
    cputype = _CPU_TYPE_ARM64,
    build_version = True,
):
    """Synthesize a minimal little-endian 64-bit Mach-O carrying a macOS
    minimum-version load command."""
    encoded = (minos[0] << 16) | (minos[1] << 8)
    if build_version:
        # LC_BUILD_VERSION: cmd, cmdsize, platform(=1 macOS), minos, sdk, ntools
        load_command = struct.pack("<6I", 0x32, 24, 1, encoded, encoded, 0)
    else:
        # LC_VERSION_MIN_MACOSX: cmd, cmdsize, version, sdk
        load_command = struct.pack("<4I", 0x24, 16, encoded, encoded)
    header = struct.pack("<8I", 0xFEEDFACF, cputype, 0, 0x2, 1, len(load_command), 0, 0)
    return header + load_command


def fat_macho(slices):
    """Synthesize a big-endian universal binary from (cputype, thin_bytes)."""
    header = struct.pack(">2I", 0xCAFEBABE, len(slices))
    data_offset = 8 + 20 * len(slices)
    arch_entries = b""
    body = b""
    for cputype, thin in slices:
        offset = data_offset + len(body)
        arch_entries += struct.pack(">5I", cputype, 0, offset, len(thin), 0)
        body += thin
    return header + arch_entries + body


class TestParseMacosVersion:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("14.7.1", (14, 7)),
            ("15.5", (15, 5)),
            ("26.0", (26, 0)),
            ("26", (26, 0)),
            ("13", (13, 0)),
            ("", None),
            (None, None),
            ("not-a-version", None),
        ],
    )
    def test_parse(self, value, expected):
        assert ILP.parse_macos_version(value) == expected


class TestHostSupportsMacosMinos:
    def test_older_host_rejects_newer_prebuilt(self):
        assert not ILP.host_supports_macos_minos(make_macos_host((14, 0)), (26, 0))

    def test_same_version_supported(self):
        assert ILP.host_supports_macos_minos(make_macos_host((26, 0)), (26, 0))

    def test_newer_host_supports_older_prebuilt(self):
        assert ILP.host_supports_macos_minos(make_macos_host((15, 5)), (14, 0))

    def test_unknown_host_defers_to_runtime(self):
        assert ILP.host_supports_macos_minos(make_macos_host(None), (26, 0))

    def test_unknown_minos_defers_to_runtime(self):
        assert ILP.host_supports_macos_minos(make_macos_host((14, 0)), None)


class TestMachoMinimumMacos:
    def test_build_version_thin(self, tmp_path):
        path = tmp_path / "lib.dylib"
        path.write_bytes(thin_macho((26, 0)))
        assert ILP.macho_minimum_macos(path) == (26, 0)

    def test_legacy_version_min_thin(self, tmp_path):
        path = tmp_path / "lib.dylib"
        path.write_bytes(thin_macho((14, 0), build_version = False))
        assert ILP.macho_minimum_macos(path) == (14, 0)

    def test_universal_prefers_host_arch_slice(self, tmp_path):
        # arm64 slice needs macOS 14, x86_64 slice needs macOS 26.
        path = tmp_path / "fat"
        path.write_bytes(
            fat_macho(
                [
                    (_CPU_TYPE_ARM64, thin_macho((14, 0), cputype = _CPU_TYPE_ARM64)),
                    (_CPU_TYPE_X86_64, thin_macho((26, 0), cputype = _CPU_TYPE_X86_64)),
                ]
            )
        )
        assert ILP.macho_minimum_macos(path, make_macos_host((14, 0))) == (14, 0)
        assert ILP.macho_minimum_macos(path, make_macos_host((26, 0), arm64 = False)) == (26, 0)

    def test_non_macho_returns_none(self, tmp_path):
        path = tmp_path / "script.sh"
        path.write_bytes(b'#!/bin/sh\nexec real "$@"\n')
        assert ILP.macho_minimum_macos(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert ILP.macho_minimum_macos(tmp_path / "nope") is None


class TestLooksLikeMacosIncompatibility:
    def test_built_for_newer_os(self):
        assert ILP.looks_like_macos_incompatibility(
            "dyld: ... (built for macOS 26.0 which is newer than running OS)"
        )

    def test_metal_residency_symbol(self):
        assert ILP.looks_like_macos_incompatibility(
            "Symbol not found: _OBJC_CLASS_$_MTLResidencySetDescriptor"
        )

    def test_benign_error(self):
        assert not ILP.looks_like_macos_incompatibility("some unrelated failure")

    def test_empty(self):
        assert not ILP.looks_like_macos_incompatibility("")


class TestPreflightMacosInstalledBinaries:
    def _install_dir(self, tmp_path, dylib_minos):
        bin_dir = tmp_path / "build" / "bin"
        bin_dir.mkdir(parents = True)
        (bin_dir / "libggml-metal.dylib").write_bytes(thin_macho(dylib_minos))
        server = tmp_path / "llama-server"
        server.write_bytes(thin_macho(dylib_minos))
        quantize = tmp_path / "llama-quantize"
        quantize.write_bytes(thin_macho(dylib_minos))
        return tmp_path, (server, quantize)

    def test_rejects_too_new_dylib(self, tmp_path):
        install_dir, binaries = self._install_dir(tmp_path, (26, 0))
        with pytest.raises(PrebuiltFallback, match = "newer macOS"):
            ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((14, 0)))

    def test_accepts_compatible_prebuilt(self, tmp_path):
        install_dir, binaries = self._install_dir(tmp_path, (14, 0))
        # Must not raise on a macOS 15 host.
        ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))

    def test_skips_when_host_version_unknown(self, tmp_path):
        install_dir, binaries = self._install_dir(tmp_path, (26, 0))
        # Unknown host version -> defer to runtime validation, do not raise.
        ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host(None))

    def test_noop_on_non_macos_host(self, tmp_path):
        install_dir, binaries = self._install_dir(tmp_path, (26, 0))
        linux_host = HostInfo(
            system = "Linux",
            machine = "x86_64",
            is_windows = False,
            is_linux = True,
            is_macos = False,
            is_x86_64 = True,
            is_arm64 = False,
            nvidia_smi = None,
            driver_cuda_version = None,
            compute_caps = [],
            visible_cuda_devices = None,
            has_physical_nvidia = False,
            has_usable_nvidia = False,
        )
        ILP.preflight_macos_installed_binaries(binaries, install_dir, linux_host)


def _fake_macos_releases(tags):
    return [
        {
            "tag_name": tag,
            "assets": [
                {
                    "name": f"llama-{tag}-bin-macos-arm64.tar.gz",
                    "browser_download_url": f"https://example.com/{tag}.tar.gz",
                }
            ],
        }
        for tag in tags
    ]


class TestMacosReleasePin:
    """Pre-26 upstream macOS pins the last loadable ggml-org release."""

    TAGS = [f"b{n}" for n in range(9442, 9400, -1)]  # newest-first, includes b9415

    def _patch_releases(self, monkeypatch):
        def fake_iter(repo, published_release_tag, requested_tag):
            # Real iterator yields only the requested tag when one is pinned.
            if requested_tag and requested_tag != "latest":
                return _fake_macos_releases([requested_tag])
            return _fake_macos_releases(self.TAGS)

        monkeypatch.setattr(ILP, "iter_release_payloads_by_time", fake_iter)

    def test_pre26_host_pins_b9415(self, monkeypatch):
        self._patch_releases(monkeypatch)
        tag, plans = ILP.resolve_simple_install_release_plans(
            "latest",
            make_macos_host((14, 0)),
            "ggml-org/llama.cpp",
            "",
        )
        assert tag == ILP._PINNED_MACOS_FALLBACK_TAG == "b9415"
        assert len(plans) == 1
        assert plans[0].release_tag == "b9415"

    def test_tahoe_host_takes_latest(self, monkeypatch):
        self._patch_releases(monkeypatch)
        tag, plans = ILP.resolve_simple_install_release_plans(
            "latest",
            make_macos_host((26, 0)),
            "ggml-org/llama.cpp",
            "",
        )
        assert tag == "latest"
        assert plans[0].release_tag == self.TAGS[0]  # newest release
        assert len(plans) == ILP.DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS

    def test_unknown_macos_host_uses_default(self, monkeypatch):
        self._patch_releases(monkeypatch)
        _tag, plans = ILP.resolve_simple_install_release_plans(
            "latest",
            make_macos_host(None),
            "ggml-org/llama.cpp",
            "",
        )
        assert len(plans) == ILP.DEFAULT_MAX_PREBUILT_RELEASE_FALLBACKS


class TestForwardsBackwardsCompat:
    """The gate is host >= prebuilt minos with no hardcoded version; each host takes the newest release it can load across a multi-tier release set."""

    # Newest first: future 27 builds, current 26 builds, an old 14 tier, a 13.
    RELEASES = [
        ("b9600", (27, 0)),
        ("b9450", (26, 0)),
        ("b9415", (14, 0)),
        ("b8300", (13, 0)),
    ]

    def _select(self, tmp_path, host_version):
        for tag, minos in self.RELEASES:
            bin_dir = tmp_path / tag / "build" / "bin"
            bin_dir.mkdir(parents = True)
            (bin_dir / "libggml-metal.dylib").write_bytes(thin_macho(minos))
            try:
                ILP.preflight_macos_installed_binaries(
                    (), tmp_path / tag, make_macos_host(host_version)
                )
                return tag
            except PrebuiltFallback:
                continue
        return None

    @pytest.mark.parametrize(
        "host_version, expected",
        [
            ((13, 0), "b8300"),  # older host takes the older prebuilt
            ((14, 7), "b9415"),  # backwards: skip 26/27, take newest that loads
            ((15, 5), "b9415"),
            ((26, 0), "b9450"),  # unchanged: newest <= host
            ((27, 1), "b9600"),  # forwards: future host takes the future build
        ],
    )
    def test_selects_newest_loadable(self, tmp_path, host_version, expected):
        assert self._select(tmp_path, host_version) == expected

    def test_host_below_prebuilt_floor_falls_through(self, tmp_path):
        # macOS 12 is below every prebuilt -> nothing matches -> source build.
        assert self._select(tmp_path, (12, 0)) is None
