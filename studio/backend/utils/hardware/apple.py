# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple Silicon GPU temperature and power -- no sudo required.

Mirrors macmon's approach (https://github.com/vladkens/macmon):
  * Temperature: average of the AppleSMC "Tg*" float keys (available since
    macOS 14; on older systems the keys are absent and this returns None).
  * Power: IOReport "Energy Model" group, "GPU Energy" channel. Each poll
    diffs the energy counter against the previous poll's sample, so the
    result is the average wattage over the polling window. The first poll
    only sets the baseline and returns None.

Public API (never raises; returns None when sensors are unavailable):
    read_gpu_temperature_c()
    read_gpu_power_w()
"""

import ctypes
import struct
import time
from typing import Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_IOKIT_PATH = "/System/Library/Frameworks/IOKit.framework/IOKit"
_CF_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_IOREPORT_PATH = "/usr/lib/libIOReport.dylib"

# AppleSMC user-client protocol (same constants as macmon / SMCKit).
_SMC_SELECTOR_HANDLE_EVENT = 2
_SMC_CMD_READ_BYTES = 5
_SMC_CMD_KEY_AT_INDEX = 8
_SMC_CMD_KEY_INFO = 9

_MAX_VALID_TEMP_C = 150.0
_CF_STRING_ENCODING_UTF8 = 0x08000100
_ENERGY_UNIT_DIVISORS = {"mJ": 1e3, "uJ": 1e6, "nJ": 1e9}


# ========== Pure helpers ==========


def _fourcc(key: str) -> int:
    """Encode a 4-char SMC key/type name as a big-endian integer."""
    return int.from_bytes(key.encode("ascii"), "big")


def _fourcc_str(value: int) -> str:
    return value.to_bytes(4, "big").decode("ascii", errors = "replace")


def _watts(energy: int, unit: str, elapsed_s: float) -> Optional[float]:
    """Convert an IOReport energy counter delta into average watts."""
    divisor = _ENERGY_UNIT_DIVISORS.get(unit.strip())
    if divisor is None or elapsed_s <= 0:
        return None
    return energy / divisor / elapsed_s


def _average_valid_temps(values: Iterable[float]) -> Optional[float]:
    valid = [v for v in values if 0.0 < v <= _MAX_VALID_TEMP_C]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 1)


def _is_gpu_energy_channel(name: str) -> bool:
    # Exact "GPU Energy" plus "DIE_N_GPU Energy" on Ultra chips; the separate
    # "GPU SRAM*" channels are not GPU core power.
    return name.endswith("GPU Energy") and "SRAM" not in name


# ========== AppleSMC structs (layout must match the kernel exactly) ==========


class _SMCKeyDataVers(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint8),
        ("minor", ctypes.c_uint8),
        ("build", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("release", ctypes.c_uint16),
    ]


class _SMCPLimitData(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("length", ctypes.c_uint16),
        ("cpu_p_limit", ctypes.c_uint32),
        ("gpu_p_limit", ctypes.c_uint32),
        ("mem_p_limit", ctypes.c_uint32),
    ]


class _SMCKeyInfo(ctypes.Structure):
    _fields_ = [
        ("data_size", ctypes.c_uint32),
        ("data_type", ctypes.c_uint32),
        ("data_attributes", ctypes.c_uint8),
    ]


class _SMCKeyData(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint32),
        ("vers", _SMCKeyDataVers),
        ("p_limit_data", _SMCPLimitData),
        ("key_info", _SMCKeyInfo),
        ("result", ctypes.c_uint8),
        ("status", ctypes.c_uint8),
        ("data8", ctypes.c_uint8),
        ("data32", ctypes.c_uint32),
        ("bytes", ctypes.c_uint8 * 32),
    ]


# ========== Library loaders ==========


def _load_iokit() -> ctypes.CDLL:
    iokit = ctypes.CDLL(_IOKIT_PATH)
    iokit.IOServiceMatching.restype = ctypes.c_void_p
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingServices.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    iokit.IOIteratorNext.restype = ctypes.c_uint32
    iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
    iokit.IORegistryEntryGetName.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    iokit.IOServiceOpen.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
    iokit.IOConnectCallStructMethod.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    return iokit


def _load_cf() -> ctypes.CDLL:
    cf = ctypes.CDLL(_CF_PATH)
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    return cf


def _load_ioreport() -> ctypes.CDLL:
    ior = ctypes.CDLL(_IOREPORT_PATH)
    ior.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
    ior.IOReportCopyChannelsInGroup.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    ior.IOReportCreateSubscription.restype = ctypes.c_void_p
    ior.IOReportCreateSubscription.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    ior.IOReportCreateSamples.restype = ctypes.c_void_p
    ior.IOReportCreateSamples.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ior.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
    ior.IOReportCreateSamplesDelta.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ior.IOReportChannelGetChannelName.restype = ctypes.c_void_p
    ior.IOReportChannelGetChannelName.argtypes = [ctypes.c_void_p]
    ior.IOReportChannelGetUnitLabel.restype = ctypes.c_void_p
    ior.IOReportChannelGetUnitLabel.argtypes = [ctypes.c_void_p]
    ior.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
    ior.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    return ior


def _cfstr(cf: ctypes.CDLL, text: str) -> int:
    return cf.CFStringCreateWithCString(
        None, text.encode("utf-8"), _CF_STRING_ENCODING_UTF8
    )


def _from_cfstr(cf: ctypes.CDLL, ref: Optional[int]) -> str:
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(128)
    if not cf.CFStringGetCString(ref, buf, len(buf), _CF_STRING_ENCODING_UTF8):
        return ""
    return buf.value.decode("utf-8", errors = "replace").strip()


# ========== SMC connection (GPU temperature) ==========


class _SMCConnection:
    """Connection to AppleSMCKeysEndpoint; discovers "Tg*" GPU temp keys once."""

    def __init__(self):
        self._iokit = _load_iokit()
        self._conn = self._open()
        self._key_info_cache: dict[int, _SMCKeyInfo] = {}
        self.gpu_keys = [
            key
            for key in self._all_keys()
            if key.startswith("Tg") and self.read_float(key) is not None
        ]

    def _open(self) -> int:
        iterator = ctypes.c_uint32(0)
        matching = self._iokit.IOServiceMatching(b"AppleSMC")
        if (
            self._iokit.IOServiceGetMatchingServices(
                0, matching, ctypes.byref(iterator)
            )
            != 0
        ):
            raise OSError("AppleSMC service not found")
        try:
            conn = self._open_keys_endpoint(iterator.value)
        finally:
            self._iokit.IOObjectRelease(iterator.value)
        if conn is None:
            raise OSError("AppleSMCKeysEndpoint not found")
        return conn

    def _open_keys_endpoint(self, iterator: int) -> Optional[int]:
        task = ctypes.CDLL(None).mach_task_self()
        while device := self._iokit.IOIteratorNext(iterator):
            name = ctypes.create_string_buffer(128)
            self._iokit.IORegistryEntryGetName(device, name)
            if name.value != b"AppleSMCKeysEndpoint":
                self._iokit.IOObjectRelease(device)
                continue
            conn = ctypes.c_uint32(0)
            status = self._iokit.IOServiceOpen(device, task, 0, ctypes.byref(conn))
            self._iokit.IOObjectRelease(device)
            if status != 0:
                raise OSError(f"IOServiceOpen(AppleSMCKeysEndpoint) failed: {status}")
            return conn.value
        return None

    def _call(self, ival: _SMCKeyData) -> _SMCKeyData:
        oval = _SMCKeyData()
        olen = ctypes.c_size_t(ctypes.sizeof(_SMCKeyData))
        status = self._iokit.IOConnectCallStructMethod(
            self._conn,
            _SMC_SELECTOR_HANDLE_EVENT,
            ctypes.byref(ival),
            ctypes.sizeof(_SMCKeyData),
            ctypes.byref(oval),
            ctypes.byref(olen),
        )
        if status != 0:
            raise OSError(f"IOConnectCallStructMethod failed: {status}")
        if oval.result != 0:
            raise OSError(f"SMC result code: {oval.result}")
        return oval

    def _read_key_info(self, key_id: int) -> _SMCKeyInfo:
        cached = self._key_info_cache.get(key_id)
        if cached is not None:
            return cached
        oval = self._call(_SMCKeyData(key = key_id, data8 = _SMC_CMD_KEY_INFO))
        self._key_info_cache[key_id] = oval.key_info
        return oval.key_info

    def _read_bytes(self, key: str) -> Optional[bytes]:
        try:
            key_id = _fourcc(key)
            info = self._read_key_info(key_id)
            oval = self._call(
                _SMCKeyData(key = key_id, data8 = _SMC_CMD_READ_BYTES, key_info = info)
            )
            return bytes(oval.bytes[: info.data_size])
        except OSError:
            return None

    def read_float(self, key: str) -> Optional[float]:
        try:
            info = self._read_key_info(_fourcc(key))
        except OSError:
            return None
        if info.data_size != 4 or info.data_type != _fourcc("flt "):
            return None
        data = self._read_bytes(key)
        if data is None or len(data) != 4:
            return None
        return struct.unpack("<f", data)[0]

    def _key_name_at(self, index: int) -> Optional[str]:
        try:
            oval = self._call(_SMCKeyData(data8 = _SMC_CMD_KEY_AT_INDEX, data32 = index))
            return oval.key.to_bytes(4, "big").decode("ascii")
        except (OSError, UnicodeDecodeError):
            return None

    def _all_keys(self) -> list[str]:
        count_bytes = self._read_bytes("#KEY")
        if count_bytes is None or len(count_bytes) != 4:
            return []
        count = int.from_bytes(count_bytes, "big")
        names = (self._key_name_at(i) for i in range(count))
        return [name for name in names if name is not None]

    def gpu_temperature_c(self) -> Optional[float]:
        readings = (self.read_float(key) for key in self.gpu_keys)
        return _average_valid_temps(value for value in readings if value is not None)


# ========== IOReport subscription (GPU power) ==========


class _IOReportEnergy:
    """Persistent subscription to the "Energy Model" group for GPU wattage."""

    def __init__(self):
        self._cf = _load_cf()
        self._ior = _load_ioreport()
        self._channels = self._ior.IOReportCopyChannelsInGroup(
            _cfstr(self._cf, "Energy Model"), None, 0, 0, 0
        )
        if not self._channels:
            raise OSError("IOReport 'Energy Model' channel group unavailable")
        subscribed = ctypes.c_void_p()
        self._sub = self._ior.IOReportCreateSubscription(
            None, self._channels, ctypes.byref(subscribed), 0, None
        )
        if not self._sub:
            raise OSError("IOReportCreateSubscription failed")
        # Sample with the channels IOReport subscribes us to, not the requested
        # group (matches macmon); fall back if the OS leaves it unset.
        self._sample_channels = subscribed if subscribed else self._channels
        self._channels_key = _cfstr(self._cf, "IOReportChannels")
        self._prev: Optional[tuple[int, float]] = None  # (sample ref, monotonic s)

    def gpu_power_w(self) -> Optional[float]:
        sample = self._ior.IOReportCreateSamples(self._sub, self._sample_channels, None)
        if not sample:
            return None
        now = time.monotonic()
        prev, self._prev = self._prev, (sample, now)
        if prev is None:
            return None
        prev_sample, prev_time = prev
        delta = self._ior.IOReportCreateSamplesDelta(prev_sample, sample, None)
        self._cf.CFRelease(prev_sample)
        if not delta:
            return None
        try:
            return self._gpu_watts_from_delta(delta, now - prev_time)
        finally:
            self._cf.CFRelease(delta)

    def _gpu_watts_from_delta(self, delta: int, elapsed_s: float) -> Optional[float]:
        items = self._cf.CFDictionaryGetValue(delta, self._channels_key)
        if not items:
            return None
        total: Optional[float] = None
        for i in range(self._cf.CFArrayGetCount(items)):
            item = self._cf.CFArrayGetValueAtIndex(items, i)
            name = _from_cfstr(self._cf, self._ior.IOReportChannelGetChannelName(item))
            if not _is_gpu_energy_channel(name):
                continue
            unit = _from_cfstr(self._cf, self._ior.IOReportChannelGetUnitLabel(item))
            energy = self._ior.IOReportSimpleGetIntegerValue(item, 0)
            watts = _watts(energy, unit, elapsed_s)
            if watts is not None:
                total = (total or 0.0) + watts
        if (
            total is None or total < 0
        ):  # negative = counter reset; show -- not a bogus draw
            return None
        return round(total, 1)


# ========== Public API (module singletons, failure-latched) ==========

_smc: Optional[_SMCConnection] = None
_smc_failed = False
_energy: Optional[_IOReportEnergy] = None
_energy_failed = False


def read_gpu_temperature_c() -> Optional[float]:
    """Average Apple GPU die temperature in degrees C, or None if unavailable."""
    global _smc, _smc_failed
    if _smc_failed:
        return None
    try:
        if _smc is None:
            _smc = _SMCConnection()
        return _smc.gpu_temperature_c()
    except Exception as e:
        _smc_failed = True
        logger.warning("Apple SMC GPU temperature unavailable: %s", e)
        return None


def read_gpu_power_w() -> Optional[float]:
    """Average GPU power in watts since the previous call, or None.

    The first call establishes the baseline sample and returns None.
    """
    global _energy, _energy_failed
    if _energy_failed:
        return None
    try:
        if _energy is None:
            _energy = _IOReportEnergy()
        return _energy.gpu_power_w()
    except Exception as e:
        _energy_failed = True
        logger.warning("Apple IOReport GPU power unavailable: %s", e)
        return None
