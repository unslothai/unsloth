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

    def test_skips_the_minos_comparison_when_host_version_unknown(self, tmp_path):
        install_dir, binaries = self._install_dir(tmp_path, (26, 0))
        # No host version to compare against, so the static check cannot run.
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


class TestMacosDyldLoadProbe:
    """The minos scan reads a header; it cannot see an install name pointing at a
    library that exists on the builder and nowhere else. That shipped: a bundle
    whose libggml-rpc.0.dylib wanted /usr/lib/librdma.dylib passed preflight, was
    logged "prebuilt installed and validated", and then died on first launch."""

    def _bundle(
        self,
        tmp_path,
        *,
        exit_code,
        message = "",
    ):
        bin_dir = tmp_path / "build" / "bin"
        bin_dir.mkdir(parents = True)
        # A real spawnable file, not a Mach-O sample: the point is to reach dyld.
        # macho_minimum_macos returns None for a non-Mach-O, so the minos gate ahead of the probe stays quiet and the
        # probe is what decides.
        server = bin_dir / "llama-server"
        server.write_text(f'#!/bin/sh\necho "{message}" >&2\nexit {exit_code}\n')
        server.chmod(0o755)
        return tmp_path, (server,)

    def test_rejects_a_binary_dyld_will_not_load(self, tmp_path):
        install_dir, binaries = self._bundle(
            tmp_path,
            exit_code = 1,
            message = "dyld: Library not loaded: /usr/lib/librdma.dylib",
        )
        with pytest.raises(PrebuiltFallback, match = "does not load on this host"):
            ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))

    def test_the_message_names_the_library(self, tmp_path):
        install_dir, binaries = self._bundle(
            tmp_path,
            exit_code = 1,
            message = "dyld: Library not loaded: /usr/lib/librdma.dylib",
        )
        with pytest.raises(PrebuiltFallback) as caught:
            ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))
        # Without the name the operator gets an exit code and no lead to follow.
        assert "librdma" in str(caught.value)

    def test_accepts_a_binary_that_loads(self, tmp_path):
        install_dir, binaries = self._bundle(tmp_path, exit_code = 0, message = "")
        ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))

    def test_a_binary_that_cannot_be_spawned_is_not_a_link_failure(self, tmp_path):
        """A refusal to exec says nothing about the link graph, and treating it as
        a verdict would reject healthy bundles on a loaded or locked-down host."""
        install_dir, binaries = self._bundle(tmp_path, exit_code = 0)
        binaries[0].chmod(0o644)
        ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))

    def test_the_probe_still_runs_when_the_host_version_is_unknown(self, tmp_path):
        """Only the static comparison needs the version; dyld does not.

        Skipping both left the checksummed path with no check at all, since the
        runtime validation it deferred to is disabled by default (#5854).
        """
        install_dir, binaries = self._bundle(
            tmp_path,
            exit_code = 1,
            message = "dyld[1]: Library not loaded: /usr/lib/librdma.dylib",
        )
        with pytest.raises(PrebuiltFallback, match = "does not load on this host"):
            ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host(None))

    def test_a_nonzero_exit_with_program_output_is_not_a_link_failure(self, tmp_path):
        """llama-quantize answers --version by printing its quantization table and
        exiting non-zero. Reading the exit code as the verdict rejected every
        published prebuilt, walked the release history to its end, and fell back to
        a source build on every macOS runner.
        """
        install_dir, binaries = self._bundle(
            tmp_path,
            exit_code = 1,
            message = (
                "llama-quantize: 7 or Q8_0 : 7.96G, +0.0026 ppl @ Llama-3-8B | "
                "1 or F16 : 14.00G, +0.0020 ppl @ Mistral-7B"
            ),
        )
        ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))

    def test_the_referencing_dylib_survives_into_the_message(self, tmp_path):
        """dyld names the library AND the dylib that asked for it. The second half
        is what says libggml-rpc rather than llama-server is at fault, so keep
        enough of the tail to carry it."""
        real = (
            "dyld[41694]: Library not loaded: /usr/lib/librdma.dylib\\n"
            "  Referenced from: /Users/r/.unsloth/llama.cpp/build/bin/libggml-rpc.0.dylib\\n"
            "  Reason: tried: '/usr/lib/librdma.dylib' (no such file)"
        )
        install_dir, binaries = self._bundle(tmp_path, exit_code = 1, message = real)
        with pytest.raises(PrebuiltFallback) as caught:
            ILP.preflight_macos_installed_binaries(binaries, install_dir, make_macos_host((15, 5)))
        message = str(caught.value)
        assert "librdma" in message
        assert "libggml-rpc" in message


class TestLooksLikeMacosLoaderFailure:
    """Narrow by design: a false positive rejects a working prebuilt."""

    @pytest.mark.parametrize(
        "text",
        [
            "dyld[41694]: Library not loaded: /usr/lib/librdma.dylib",
            "dyld: Library not loaded: /usr/lib/libfoo.dylib",
            "Symbol not found: _OBJC_CLASS_$_MTLResidencySetDescriptor",
            "dyld[1]: no suitable image found.",
        ],
    )
    def test_loader_failures_are_recognised(self, text):
        assert ILP.looks_like_macos_loader_failure(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "llama-quantize: 7 or Q8_0 : 7.96G, +0.0026 ppl @ Llama-3-8B",
            "usage: llama-quantize [--help] model-f32.gguf",
            "error: failed to load model 'foo.gguf'",
            "main: build = 10639 (f6f92fe)",
            # DYLD_PRINT_LIBRARIES narrates a perfectly healthy load under the
            # same prefix a failure uses, so the prefix alone cannot be the test.
            "dyld[4711]: /usr/lib/libSystem.B.dylib\ndyld[4711]: /usr/lib/libc++.1.dylib",
        ],
    )
    def test_ordinary_program_output_is_not_a_loader_failure(self, text):
        assert not ILP.looks_like_macos_loader_failure(text)


class TestTheProbeEnvironment:
    def test_dyld_diagnostics_are_kept_out_of_the_probe(self, tmp_path, monkeypatch):
        """A user's DYLD_PRINT_* would otherwise be echoed into the output the
        probe reads, and llama-quantize's nonzero --version would turn that
        narration into a rejection."""
        monkeypatch.setenv("DYLD_PRINT_LIBRARIES", "1")
        monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/evil.dylib")
        seen = {}

        bin_dir = tmp_path / "build" / "bin"
        bin_dir.mkdir(parents = True)
        server = bin_dir / "llama-server"
        server.write_text("#!/bin/sh\nexit 0\n")
        server.chmod(0o755)

        def capture(command, **kwargs):
            seen.update(kwargs.get("env") or {})

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr(ILP, "run_capture", capture)
        ILP.macos_dyld_load_issues([server], tmp_path, make_macos_host((15, 5)))
        assert "DYLD_PRINT_LIBRARIES" not in seen
        assert "DYLD_INSERT_LIBRARIES" not in seen


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
