# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention kernel's compiled-image cache: one compile per architecture per process, kept on disk for the
next process. CPU only: NVRTC and the driver are stand-ins, except where a test says it needs NVRTC."""

from __future__ import annotations

import contextlib
import multiprocessing
import os

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as K

SOURCE = "// a kernel\n"
DEFINES = {"HD": 128, "BN": 128}
IDENTITY = {"nvrtc": [13, 0], "nvrtc_file": ["libnvrtc.so.13", 1, 2], "driver": [13, 0]}


class _Toolchain:
    """Counts what would have run NVRTC and the module loads."""

    def __init__(self):
        self.ptx = []
        self.cubin = []
        self.loaded = []

    def compile_ptx(
        self,
        source,
        arch,
        defines = None,
    ):
        self.ptx.append((arch, dict(defines or {})))
        return f"PTX {arch} {sorted((defines or {}).items())} {source}".encode() + b"\0"

    def compile_cubin(
        self,
        source,
        arch,
        defines = None,
        max_registers = None,
    ):
        self.cubin.append((arch, dict(defines or {}), max_registers))
        return f"\x7fELF {arch} {sorted((defines or {}).items())} {max_registers} {source}".encode()


@pytest.fixture
def chain(monkeypatch, tmp_path):
    tc = _Toolchain()
    monkeypatch.setenv(K.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(K, "_BINARIES", {})
    monkeypatch.setattr(K, "_FUNCTIONS", {})
    monkeypatch.setattr(K, "_IDENTITY", {})
    monkeypatch.setattr(K, "_TRUSTED", {})
    monkeypatch.setattr(K, "_LOGGED", set())
    monkeypatch.setattr(K, "CACHE_STATS", {key: 0 for key in K.CACHE_STATS})
    monkeypatch.setattr(K, "toolchain_identity", lambda: IDENTITY)
    monkeypatch.setattr(K, "compile_ptx", tc.compile_ptx)
    monkeypatch.setattr(K, "compile_cubin", tc.compile_cubin)
    return tc


def _version_dir(tmp_path):
    return tmp_path / "cache" / K.version_dir_name()


def test_a_miss_compiles_once_and_the_next_process_reads_the_disk(chain, tmp_path):
    kind, image = K.binary((8, 9), SOURCE, DEFINES)
    assert (
        kind == "cubin"
        and image.startswith(b"\x7fELF")
        and chain.cubin == [("sm_89", DEFINES, None)]
    )
    assert not chain.ptx  # one NVRTC run; the driver has nothing to compile at load
    files = list(_version_dir(tmp_path).iterdir())
    assert [f.suffix for f in files] == [".cubin"] and K.CACHE_STATS["disk_write"] == 1
    assert K.binary((8, 9), SOURCE, DEFINES) == (kind, image) and K.CACHE_STATS["memory_hit"] == 1
    K._BINARIES.clear()  # a new process
    assert K.binary((8, 9), SOURCE, DEFINES) == (kind, image)
    assert K.CACHE_STATS["disk_hit"] == 1 and len(chain.cubin) == 1


def test_the_register_cap_goes_to_the_machine_code_compile(chain):
    K.binary((8, 9), SOURCE, DEFINES, 168)
    assert chain.cubin == [("sm_89", DEFINES, 168)]


def test_an_nvrtc_without_machine_code_keeps_the_ptx(chain, tmp_path, monkeypatch):
    """No nvrtcGetCUBIN or no --ptxas-options: the capped PTX, as before the cache (the driver compiles it)."""

    def fails(*a, **k):
        raise RuntimeError("this NVRTC has no nvrtcGetCUBIN")

    monkeypatch.setattr(K, "compile_cubin", fails)
    kind, image = K.binary((8, 9), SOURCE, DEFINES, 168)
    # NVRTC's PTX for the virtual architecture: the same PTX as for sm_89, without NVRTC's own machine-code pass
    assert kind == "ptx" and chain.ptx[0][0] == "compute_89"
    assert image == K.cap_registers(chain.compile_ptx(SOURCE, "compute_89", DEFINES), 168)
    assert [p.suffix for p in _version_dir(tmp_path).iterdir()] == [".ptx"]
    K._BINARIES.clear()
    assert (
        K.binary((8, 9), SOURCE, DEFINES, 168) == (kind, image) and K.CACHE_STATS["disk_hit"] == 1
    )


@pytest.mark.parametrize(
    "driver, kind",
    [([13, 0], "cubin"), ([12, 8], "cubin"), (None, "cubin"), ([13, 1], "ptx"), ([14, 0], "ptx")],
    ids = ["same", "older_driver", "no_driver", "newer_minor", "newer_major"],
)
def test_a_driver_newer_than_nvrtc_keeps_compiling_the_ptx(chain, monkeypatch, driver, kind):
    """NVRTC's machine code only where it is what a PTX load would run: NVRTC at least as new as the driver."""
    monkeypatch.setattr(K, "toolchain_identity", lambda: {**IDENTITY, "driver": driver})
    assert K.binary((8, 9), SOURCE, DEFINES, 168)[0] == kind
    assert len(chain.cubin) == (kind == "cubin") and len(chain.ptx) == (kind == "ptx")


def test_the_key_names_everything_that_went_into_the_image(chain, monkeypatch):
    base = K.cache_key("cubin", SOURCE, "sm_89", DEFINES, None)
    assert base == K.cache_key(
        "cubin", SOURCE, "sm_89", dict(reversed(list(DEFINES.items()))), None
    )
    assert base.endswith(".cubin") and all(
        c in "0123456789abcdef." for c in base.replace("cubin", "")
    )
    others = {
        K.cache_key("cubin", SOURCE + " ", "sm_89", DEFINES, None),
        K.cache_key("cubin", SOURCE, "sm_86", DEFINES, None),
        K.cache_key("cubin", SOURCE, "sm_89", {**DEFINES, "BN": 64}, None),
        K.cache_key("cubin", SOURCE, "sm_89", DEFINES, 168),
        K.cache_key("ptx", SOURCE, "sm_89", DEFINES, None),
    }
    for change in (
        {"nvrtc": [12, 8]},
        {"nvrtc_file": ["libnvrtc.so.12", 1, 2]},
        {"driver": [12, 9]},
    ):
        monkeypatch.setattr(K, "toolchain_identity", lambda c = change: {**IDENTITY, **c})
        others.add(K.cache_key("cubin", SOURCE, "sm_89", DEFINES, None))
    assert base not in others and len(others) == 8
    # a new kernel source or cache format is a new version directory
    name = K.version_dir_name()
    monkeypatch.setattr(K, "CACHE_FORMAT", K.CACHE_FORMAT + 1)
    assert K.version_dir_name() != name and K._VERSION_DIR.match(K.version_dir_name())


@pytest.mark.parametrize(
    "damage",
    [
        lambda b: b[:-3],  # torn write
        lambda b: b[:20] + bytes([b[20] ^ 1]) + b[21:],  # a flipped bit in the checksum
        lambda b: b[:-1] + bytes([b[-1] ^ 1]),  # a flipped bit in the image
        lambda b: b"",  # empty
        lambda b: b"junk",  # not ours
    ],
    ids = ["torn", "checksum", "image", "empty", "junk"],
)
def test_a_corrupt_entry_is_rebuilt(chain, tmp_path, damage):
    good = K.binary((8, 0), SOURCE, DEFINES)
    (path,) = list(_version_dir(tmp_path).iterdir())
    path.write_bytes(damage(path.read_bytes()))
    K._BINARIES.clear()
    assert K.binary((8, 0), SOURCE, DEFINES) == good
    assert K.CACHE_STATS["disk_bad"] == 1 and len(chain.cubin) == 2
    K._BINARIES.clear()
    assert K.binary((8, 0), SOURCE, DEFINES) == good and K.CACHE_STATS["disk_hit"] == 1


def test_an_unreadable_or_unwritable_cache_only_costs_a_compile(chain, tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv(
        K.CACHE_ENV, str(blocker / "below")
    )  # a directory cannot be made under a file
    assert K.binary((8, 9), SOURCE, DEFINES)[0] == "cubin" and K.CACHE_STATS["disk_write"] == 0
    K._BINARIES.clear()
    assert K.binary((8, 9), SOURCE, DEFINES)[0] == "cubin" and len(chain.cubin) == 2


def test_a_rename_refused_by_another_process_leaves_no_temporary_file(chain, tmp_path, monkeypatch):
    """Windows refuses to replace a file another process holds open: that process's copy is the same image."""

    def refuse(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(K.os, "replace", refuse)
    K.binary((8, 9), SOURCE, DEFINES)
    assert list(_version_dir(tmp_path).iterdir()) == []


def test_paths_with_spaces_and_non_ascii_names(chain, tmp_path, monkeypatch):
    root = tmp_path / "Program Files" / "Ünsloth Studio"
    monkeypatch.setenv(K.CACHE_ENV, str(root))
    image = K.binary((7, 5), SOURCE, DEFINES)
    K._BINARIES.clear()
    assert K.binary((7, 5), SOURCE, DEFINES) == image and K.CACHE_STATS["disk_hit"] == 1
    assert (root / K.version_dir_name()).is_dir()


def test_old_versions_and_extra_images_are_pruned(chain, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    for i in range(6):
        old = cache / f"v{i}-{i:016x}"
        old.mkdir(parents = True)
        (old / "a.cubin").write_bytes(b"x")
        os.utime(old, (1000 + i, 1000 + i))
    (cache / "not-ours").mkdir()
    (cache / "v1-notahexdigest").mkdir()
    monkeypatch.setattr(K, "CACHE_MAX_FILES", 3)
    for bn in (16, 32, 64, 128, 256):
        K.binary((8, 9), SOURCE, {**DEFINES, "BN": bn})
    left = sorted(p.name for p in cache.iterdir())
    kept = [f"v4-{4:016x}", f"v5-{5:016x}"]  # the newest others, CACHE_KEEP - 1 of them
    assert K.CACHE_KEEP == 3
    assert left == sorted(kept + [K.version_dir_name(), "not-ours", "v1-notahexdigest"])
    assert len(list(_version_dir(tmp_path).iterdir())) == 3


def test_the_cache_dir_follows_the_studio_setting(monkeypatch, tmp_path):
    from utils.paths import storage_roots

    monkeypatch.delenv(K.CACHE_ENV, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio home"))
    expected = os.path.join(str(storage_roots.cache_root()), K.CACHE_DIR_NAME)
    assert K.cache_root() == expected
    assert os.path.commonpath([expected, str(tmp_path)]) == str(tmp_path)
    monkeypatch.setenv(K.CACHE_ENV, str(tmp_path / "elsewhere"))
    assert K.cache_root() == str(tmp_path / "elsewhere")
    for off in ("0", "off", " OFF "):
        monkeypatch.setenv(K.CACHE_ENV, off)
        assert K.cache_root() is None


def test_no_disk_cache_still_compiles_once_per_process(chain, tmp_path, monkeypatch):
    monkeypatch.setenv(K.CACHE_ENV, "0")
    K.binary((8, 9), SOURCE, DEFINES)
    K.binary((8, 9), SOURCE, DEFINES)
    assert len(chain.cubin) == 1 and not (tmp_path / "cache").exists()


class _Driver:
    def __init__(self):
        self.functions = []

    def cuModuleGetFunction(self, handle, module, name):
        self.functions.append(name)
        return 0


@pytest.fixture
def cards(chain, monkeypatch):
    """Two (or more) fake cards: module loads are recorded per device."""
    drv = _Driver()
    current = []

    @contextlib.contextmanager
    def device(index):
        current.append(index)
        yield

    def load_image(image):
        chain.loaded.append((current[-1], bytes(image)))
        return K.ctypes.c_void_p(len(chain.loaded))

    monkeypatch.setattr(K, "nvrtc", lambda: object())
    monkeypatch.setattr(K, "driver", lambda: drv)
    monkeypatch.setattr(K, "_load_image", load_image)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: None)
    return chain


def test_two_cards_of_one_architecture_compile_once(cards):
    names = (K.KERNEL_NAME,)
    first = K.load_module(0, (8, 9), SOURCE, names, DEFINES, 168)
    second = K.load_module(1, (8, 9), SOURCE, names, DEFINES, 168)
    assert first is not second and first[K.KERNEL_NAME].handle is not second[K.KERNEL_NAME].handle
    assert len(cards.cubin) == 1 and not cards.ptx
    assert [index for index, _ in cards.loaded] == [0, 1] and cards.loaded[0][1] == cards.loaded[1][
        1
    ]
    assert (
        K.load_module(1, (8, 9), SOURCE, names, DEFINES, 168) is second and len(cards.loaded) == 2
    )
    K.load_module(2, (8, 6), SOURCE, names, DEFINES, 168)  # another architecture is another image
    assert [arch for arch, _, _ in cards.cubin] == ["sm_89", "sm_86"]


def test_machine_code_the_driver_will_not_load_falls_back_to_the_ptx(cards, tmp_path, monkeypatch):
    loaded = []

    def load_image(image):
        loaded.append(bytes(image))
        if image.startswith(b"\x7fELF"):
            raise K.CudaError("cuModuleLoadData", 209)
        return K.ctypes.c_void_p(1)

    monkeypatch.setattr(K, "_load_image", load_image)
    K.load_module(0, (8, 0), SOURCE, (K.KERNEL_NAME,), DEFINES)
    assert K.CACHE_STATS["load_fallback"] == 1 and loaded[-1].startswith(b"PTX compute_80")
    # the cubin dropped from disk for the PTX and its marker, and the other cards of the process load the PTX at once
    assert sorted(p.suffix for p in _version_dir(tmp_path).iterdir()) == [".ptx", ".ptx-only"]
    K.load_module(1, (8, 0), SOURCE, (K.KERNEL_NAME,), DEFINES)
    assert (
        K.CACHE_STATS["load_fallback"] == 1
        and len(cards.ptx) == 1
        and loaded[-1].startswith(b"PTX")
    )

    def broken(image):
        raise K.CudaError("cuModuleLoadData", 218)

    monkeypatch.setattr(K, "_load_image", broken)
    with pytest.raises(RuntimeError, match = "218"):
        K.load_module(1, (8, 0), SOURCE, (K.KERNEL_NAME,), {**DEFINES, "BN": 64})


def _writer(root: str, rounds: int, out) -> None:
    """One process of the concurrency test: compiles (a slow stand-in) and reads the image many times."""
    import time

    from core.inference import diffusion_int8_attention_cuda as cuda

    os.environ[cuda.CACHE_ENV] = root
    cuda.toolchain_identity = lambda: IDENTITY

    def slow_cubin(
        source,
        arch,
        defines = None,
        max_registers = None,
    ):
        time.sleep(0.01)
        # big enough that a torn read would show
        return b"\x7fELF" + f"{arch} {sorted((defines or {}).items())} {source}".encode() * 2000

    cuda.compile_cubin = slow_cubin
    seen = set()
    for i in range(rounds):
        cuda._BINARIES.clear()
        seen.add(cuda.binary((8, 9), SOURCE, {"HD": 128, "R": (i % 3) * 10 ** (i % 3)})[1])
    out.put((sorted(len(s) for s in seen), dict(cuda.CACHE_STATS)))


def test_two_processes_share_the_cache_without_torn_reads(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    root = str(tmp_path / "shared")
    procs = [ctx.Process(target = _writer, args = (root, 40, out)) for _ in range(2)]
    for p in procs:
        p.start()
    results = [out.get(timeout = 300) for _ in procs]
    for p in procs:
        p.join(timeout = 60)
    assert all(p.exitcode == 0 for p in procs)
    sizes = {size for found, _ in results for size in found}
    assert len(sizes) == 3  # one image per define set, never a partial one
    assert all(stats["disk_bad"] == 0 for _, stats in results)
    assert sum(stats["disk_hit"] for _, stats in results) > 0
    folder = os.path.join(root, os.listdir(root)[0])
    assert sorted(os.listdir(folder)) == sorted(
        n for n in os.listdir(folder) if not n.startswith(".tmp-")
    )
    assert len(os.listdir(folder)) == 3


def _nvrtc_ok() -> bool:
    try:
        K.nvrtc()
        return True
    except Exception:  # noqa: BLE001
        return False


def _supported(arch: int) -> bool:
    import ctypes

    lib = K.nvrtc()
    count = ctypes.c_int()
    if lib.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0:
        return False
    archs = (ctypes.c_int * count.value)()
    return lib.nvrtcGetSupportedArchs(archs) == 0 and arch in set(archs)


@pytest.mark.skipif(not _nvrtc_ok(), reason = "NVRTC ships with the CUDA builds of torch")
@pytest.mark.parametrize("capability", sorted(I.KERNEL_CONFIGS), ids = lambda c: f"sm{c[0]}{c[1]}")
def test_nvrtc_makes_the_images_from_the_same_ptx(capability):
    """The fallback PTX is compiled for compute_XY: byte for byte the PTX NVRTC gives for sm_XY (what the loader
    used before). The machine code is NVRTC's for sm_XY, with the register cap given to its PTX compiler alone: the
    PTX of that compile is the uncapped PTX too, and the cap changes the machine code."""
    arch = 10 * capability[0] + capability[1]
    if not _supported(arch):
        pytest.skip(f"NVRTC {K.nvrtc_version()} cannot target sm_{arch}")
    for head_dim, entry in I.KERNEL_CONFIGS[capability].items():
        config = I.select_config(entry, 0)
        defines = I.kernel_defines(head_dim, config)
        ptx = K.compile_ptx(K.KERNEL_SOURCE, f"sm_{arch}", defines)
        assert K.compile_ptx(K.KERNEL_SOURCE, f"compute_{arch}", defines) == ptx
        cap = config.get("max_registers")
        if cap:
            capped_ptx, capped = K._nvrtc_outputs(
                K.KERNEL_SOURCE,
                f"sm_{arch}",
                defines,
                (f"--ptxas-options=--maxrregcount={cap}",),
                ("PTX", "CUBIN"),
            )
            assert capped_ptx == ptx
            assert capped == K.compile_cubin(K.KERNEL_SOURCE, f"sm_{arch}", defines, cap)
            assert capped != K.compile_cubin(K.KERNEL_SOURCE, f"sm_{arch}", defines)
        quant = {
            "HD": head_dim,
            "BN": entry["bn"],
            "RGX": min(I.ROT_GROUP, head_dim).bit_length() - 5,
        }
        assert K.compile_ptx(K.QUANT_SOURCE, f"compute_{arch}", quant) == K.compile_ptx(
            K.QUANT_SOURCE, f"sm_{arch}", quant
        )
    assert K.compile_cubin(K.QUANT_SOURCE, f"sm_{arch}", quant).startswith(b"\x7fELF")


def _failing_cubin_loads(monkeypatch, rc):
    loaded = []

    def load_image(image):
        loaded.append(bytes(image))
        if image.startswith(b"\x7fELF"):
            raise K.CudaError("cuModuleLoadData", rc)
        return K.ctypes.c_void_p(1)

    monkeypatch.setattr(K, "_load_image", load_image)
    return loaded


def _new_process():
    K._BINARIES.clear()
    K._FUNCTIONS.clear()
    K._TRUSTED.clear()


@pytest.mark.parametrize(
    "rc", [2, 3, 201], ids = ["out_of_memory", "not_initialized", "invalid_context"]
)
def test_a_transient_load_failure_is_raised_and_keeps_the_cubin(cards, tmp_path, monkeypatch, rc):
    """Nothing about the image: no PTX fallback, nothing removed, the next try loads the same cubin."""
    loaded = _failing_cubin_loads(monkeypatch, rc)
    with pytest.raises(K.CudaError) as raised:
        K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES)
    assert raised.value.rc == rc and K.CACHE_STATS["load_fallback"] == 0 and not cards.ptx
    assert [p.suffix for p in _version_dir(tmp_path).iterdir()] == [".cubin"]
    monkeypatch.setattr(
        K, "_load_image", lambda image: loaded.append(bytes(image)) or K.ctypes.c_void_p(1)
    )
    K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES)
    assert loaded[-1].startswith(b"\x7fELF") and len(cards.cubin) == 1
    _new_process()
    K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES)
    assert K.CACHE_STATS["disk_hit"] == 1 and len(cards.cubin) == 1


@pytest.mark.parametrize("rc", sorted(K._IMAGE_UNUSABLE))
def test_a_cubin_this_host_cannot_load_is_remembered_as_ptx_only(cards, tmp_path, monkeypatch, rc):
    loaded = _failing_cubin_loads(monkeypatch, rc)
    K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES, 168)
    assert K.CACHE_STATS["load_fallback"] == 1 and len(cards.cubin) == 1 and len(cards.ptx) == 1
    assert sorted(p.suffix for p in _version_dir(tmp_path).iterdir()) == [".ptx", ".ptx-only"]
    ptx = loaded[-1]
    assert ptx == K.cap_registers(_Toolchain().compile_ptx(SOURCE, "compute_89", DEFINES), 168)
    _new_process()
    del loaded[:]
    K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES, 168)
    # straight to the PTX on disk: no cubin compile, read or load
    assert loaded == [ptx] and len(cards.cubin) == 1 and len(cards.ptx) == 1
    assert K.CACHE_STATS["disk_hit"] == 1 and K.CACHE_STATS["load_fallback"] == 1
    # the PTX pruned but the marker kept: the PTX is compiled again, never the cubin
    next(_version_dir(tmp_path).glob("*.ptx")).unlink()
    _new_process()
    K.load_module(0, (8, 9), SOURCE, (K.KERNEL_NAME,), DEFINES, 168)
    assert loaded[-1] == ptx and len(cards.cubin) == 1 and len(cards.ptx) == 2


posix_only = pytest.mark.skipif(os.name == "nt", reason = "POSIX ownership and mode bits")


@posix_only
@pytest.mark.parametrize("where", ["root", "version"])
@pytest.mark.parametrize("how", ["world_writable", "group_writable", "foreign_owner"])
def test_a_directory_others_can_write_is_not_read(chain, tmp_path, monkeypatch, where, how):
    good = K.binary((8, 9), SOURCE, DEFINES)
    folder = tmp_path / "cache" if where == "root" else _version_dir(tmp_path)
    (image,) = list(_version_dir(tmp_path).iterdir())
    planted = image.read_bytes()
    if how == "foreign_owner":
        uid = os.getuid()
        monkeypatch.setattr(K.os, "getuid", lambda: uid + 1)
    else:
        folder.chmod(0o777 if how == "world_writable" else 0o770)
    _new_process()
    assert K.binary((8, 9), SOURCE, DEFINES) == good  # compiled in memory
    assert len(chain.cubin) == 2 and K.CACHE_STATS["disk_hit"] == 0
    assert K.CACHE_STATS["disk_refused"] == 1
    # nothing written there either
    assert image.read_bytes() == planted and K.CACHE_STATS["disk_write"] == 1
    K._BINARIES.clear()  # checked once per process per directory
    K.binary((8, 9), SOURCE, {**DEFINES, "BN": 64})
    assert K.CACHE_STATS["disk_refused"] == 1 and len(chain.cubin) == 3


@posix_only
@pytest.mark.parametrize("mode", [0o666, 0o620])
def test_a_file_others_can_write_is_not_read(chain, tmp_path, mode):
    good = K.binary((8, 9), SOURCE, DEFINES)
    (image,) = list(_version_dir(tmp_path).iterdir())
    image.chmod(mode)
    _new_process()
    assert K.binary((8, 9), SOURCE, DEFINES) == good
    assert len(chain.cubin) == 2 and K.CACHE_STATS["disk_hit"] == 0
    assert K.CACHE_STATS["disk_refused"] == 1
    assert image.stat().st_mode & 0o777 == 0o600  # replaced by this process's own private file


@posix_only
def test_the_cache_is_made_private(chain, tmp_path, monkeypatch):
    root = tmp_path / "a" / "cache"
    monkeypatch.setenv(K.CACHE_ENV, str(root))
    old = os.umask(0o002)  # a umask that leaves new directories group-writable
    try:
        K.binary((8, 9), SOURCE, DEFINES)
    finally:
        os.umask(old)
    folder = root / K.version_dir_name()
    (image,) = list(folder.iterdir())
    modes = [p.stat().st_mode & 0o777 for p in (root, folder, image)]
    assert modes == [0o700, 0o700, 0o600]
    _new_process()
    K.binary((8, 9), SOURCE, DEFINES)
    assert K.CACHE_STATS["disk_hit"] == 1 and K.CACHE_STATS["disk_refused"] == 0
