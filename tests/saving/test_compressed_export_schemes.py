"""CPU-only, deterministic checks for the compressed-tensors export registry and the
`save_method` normalization logic.

No GPU, no model load, no torch math - just the pure routing logic - so a registry or
alias regression is caught fast on CPU-only CI.
"""

from __future__ import annotations

import pytest

from unsloth.save import COMPRESSED_EXPORT_SCHEMES, _normalize_compressed_method


def test_registry_entries_are_well_formed():
    assert COMPRESSED_EXPORT_SCHEMES, "compressed export registry must not be empty"
    for alias, value in COMPRESSED_EXPORT_SCHEMES.items():
        assert (
            isinstance(alias, str) and alias == alias.lower()
        ), f"alias must be a lowercase str: {alias!r}"
        assert (
            isinstance(value, tuple) and len(value) == 3
        ), f"{alias!r} must map to a (scheme, needs_calib, suffix) tuple"
        scheme, needs_calib, suffix = value
        assert isinstance(scheme, str) and scheme, f"{alias!r}: scheme must be a non-empty str"
        assert isinstance(needs_calib, bool), f"{alias!r}: needs_calibration must be a bool"
        assert isinstance(suffix, str) and suffix, f"{alias!r}: suffix must be a non-empty str"
        # The suffix builds the sibling output dir "<save_dir>-<suffix>"; keep it path-safe.
        assert not (
            set(suffix) & set("/\\ ")
        ), f"{alias!r}: suffix {suffix!r} must be filesystem-safe"


def test_every_alias_round_trips_case_and_separator_insensitive():
    for alias, value in COMPRESSED_EXPORT_SCHEMES.items():
        assert _normalize_compressed_method(alias) == value
        assert _normalize_compressed_method(alias.upper()) == value
        # users may pass dashes / surrounding whitespace
        assert _normalize_compressed_method(f"  {alias.replace('_', '-')}  ") == value


@pytest.mark.parametrize(
    "method", ["merged_16bit", "16bit", "merged_4bit", "lora", "", None, 123, ["fp8"]]
)
def test_standard_save_methods_are_not_treated_as_compressed(method):
    assert _normalize_compressed_method(method) is None


@pytest.mark.parametrize(
    "method", ["fp8_turbo", "nvfp4_xl", "w4a99", "mxfp3", "int8_banana", "fp4_max"]
)
def test_near_miss_compressed_names_raise(method):
    # Names that clearly intend a compressed scheme but are unsupported must fail loudly,
    # not fall through to the generic "unknown save_method" path.
    with pytest.raises(RuntimeError):
        _normalize_compressed_method(method)


def test_calibration_flags_match_known_schemes():
    # Only static FP8 and NVFP4 require calibration data; everything else is data-free.
    assert _normalize_compressed_method("fp8")[1] is False
    assert _normalize_compressed_method("fp8_static")[1] is True
    assert _normalize_compressed_method("nvfp4")[1] is True
    assert _normalize_compressed_method("mxfp4")[1] is False


def test_core_aliases_present():
    for alias in ("fp8", "fp8_dynamic", "fp8_static", "mxfp4", "nvfp4", "int8", "w4a16", "w8a8"):
        assert alias in COMPRESSED_EXPORT_SCHEMES, f"expected core alias {alias!r} in registry"
