# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Lowercased mapper keys must resolve to correctly-cased repo ids.

``__get_model_name`` only ever looks up ``model_name.lower()``, and returns the
mapper's value verbatim as the repo id to download. If the value is lowercased
too, every resolution returns an id that only huggingface.co can resolve, since
it matches repo ids case-insensitively. An ``HF_ENDPOINT`` mirror, an offline
snapshot, or a case-sensitive filesystem does not (unslothai/unsloth#2506).

``mapper.py`` is loaded by path: importing ``unsloth.models.mapper`` pulls in
``unsloth/__init__.py``, which requires an accelerator, and this invariant is
pure dict construction. Same reason ``test_mapper_no_duplicate_keys.py``
inspects the source instead of importing the package.
"""

import importlib.util
import os

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")


def _load_mapper():
    spec = importlib.util.spec_from_file_location("_unsloth_mapper", MAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_mismatches(mapping):
    """Lowered keys whose value differs from the exact-case key's value."""
    exact_by_lower = {}
    for key in mapping:
        if key != key.lower():
            exact_by_lower.setdefault(key.lower(), key)

    mismatches = []
    for lowered, exact in exact_by_lower.items():
        if lowered in mapping and mapping[lowered] != mapping[exact]:
            mismatches.append((lowered, mapping[lowered], mapping[exact]))
    return mismatches


def test_int_to_float_mapper_preserves_case():
    mapper = _load_mapper()
    mismatches = _case_mismatches(mapper.INT_TO_FLOAT_MAPPER)
    assert not mismatches, (
        "A lowercased key must resolve to the same repo id as its exact-case "
        "counterpart; a lowercased value only resolves on huggingface.co. "
        f"{len(mismatches)} mismatch(es), first 5: {mismatches[:5]}"
    )


def test_float_to_int_mapper_preserves_case():
    mapper = _load_mapper()
    mismatches = _case_mismatches(mapper.FLOAT_TO_INT_MAPPER)
    assert not mismatches, (
        "A lowercased key must resolve to the same repo id as its exact-case "
        f"counterpart. {len(mismatches)} mismatch(es), first 5: {mismatches[:5]}"
    )


def test_lowercase_lookup_returns_upstream_casing():
    # gemma-4-E2B is the clearest case: the repo carries capitals the lowered
    # lookup used to destroy, so the resolved id 404s off huggingface.co.
    mapper = _load_mapper()
    resolved = mapper.INT_TO_FLOAT_MAPPER.get("unsloth/gemma-4-e2b-unsloth-bnb-4bit")
    assert resolved == "unsloth/gemma-4-E2B", resolved


def test_case_insensitive_lookup_still_works():
    # The fix must not remove the lowered keys: __get_model_name only looks up
    # model_name.lower(), so dropping them would break every resolution.
    mapper = _load_mapper()
    for mapping in (mapper.INT_TO_FLOAT_MAPPER, mapper.FLOAT_TO_INT_MAPPER):
        lowered = [key for key in mapping if key == key.lower()]
        assert len(lowered) > 100, len(lowered)
    assert "unsloth/gemma-4-e2b-unsloth-bnb-4bit" in mapper.INT_TO_FLOAT_MAPPER


def _declared_repo_ids(mapper):
    """Every repo id written out in the ``__INT_TO_FLOAT_MAPPER`` registry.

    Name mangling only applies inside class bodies, so the module-level private
    registry is reachable by ``getattr``.
    """
    registry = getattr(mapper, "__INT_TO_FLOAT_MAPPER")
    declared = set()
    for key, values in registry.items():
        declared.add(key)
        groups = values.values() if isinstance(values, dict) else (values,)
        for group in groups:
            declared.update(value for value in group if value)
    return declared


def test_every_resolved_value_is_a_declared_repo_id():
    # Guards the construction convention itself. A `.lower()` applied to a value
    # produces a string that appears nowhere in the registry, so this fails for
    # any future edit that reintroduces one.
    mapper = _load_mapper()
    declared = _declared_repo_ids(mapper)
    for name in ("INT_TO_FLOAT_MAPPER", "FLOAT_TO_INT_MAPPER"):
        undeclared = sorted(
            value for value in set(getattr(mapper, name).values()) if value not in declared
        )
        assert not undeclared, (
            f"{name} resolves to {len(undeclared)} id(s) that are not written "
            f"anywhere in the registry, so they were transformed in transit: "
            f"{undeclared[:5]}"
        )
