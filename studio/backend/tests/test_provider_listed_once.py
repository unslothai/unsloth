# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every provider is offered to the Connection dropdown exactly once.

DeepInfra was briefly registered twice -- once in ``PROVIDER_REGISTRY`` and
once in the frontend's ``CUSTOM_PROVIDER_PRESETS`` -- which cost more than a
duplicated row. ``toExternalBackendProviderType`` maps a preset type through
``CUSTOM_BACKEND_PROVIDER_TYPE`` ("openai"), so a connection created from the
preset saved ``provider_type = "openai"`` and every backend gate keyed on the
provider type missed it: ``service_tier`` never reached the wire, and the
reasoning table, the models.dev namespace and the vision sets all keyed on a
type the row no longer had.

``test_deepinfra_provider.py`` could not catch that: it builds
``ChatCompletionRequest(provider_type = "deepinfra", ...)`` directly, so it
never walks the path that rewrites the type. These tests pin the rule the
dialog depends on -- fixed-endpoint vendors live in the registry, user-supplied
endpoints live in the presets, and the two lists never overlap.

The frontend half reads ``external-providers.ts`` as source text rather than
importing it. The presets and the two catalog sets are plain literals, so a
source assertion catches a preset being reintroduced -- the regression that
matters -- without importing a ``.ts`` module from a Python process.
"""

import re
from pathlib import Path

from core.inference.providers import PROVIDER_REGISTRY, list_available_providers

FRONTEND_PROVIDERS_TS = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "chat"
    / "external-providers.ts"
)


def _frontend_source() -> str:
    assert FRONTEND_PROVIDERS_TS.is_file(), FRONTEND_PROVIDERS_TS
    return FRONTEND_PROVIDERS_TS.read_text(encoding = "utf-8")


def _preset_types(source: str) -> list[str]:
    """The ``providerType`` of every entry in ``CUSTOM_PROVIDER_PRESETS``."""
    start = source.index("export const CUSTOM_PROVIDER_PRESETS")
    end = source.index("] as const;", start)
    return re.findall(r'providerType:\s*"([^"]+)"', source[start:end])


def _set_members(source: str, const_name: str) -> set[str]:
    """The string members of ``const <name> = new Set([...])``."""
    marker = f"const {const_name} = new Set(["
    assert marker in source, f"{const_name} not found in external-providers.ts"
    start = source.index(marker)
    end = source.index("]);", start)
    return set(re.findall(r'"([^"]+)"', source[start:end]))


# ── The dropdown offers each provider once ───────────────────────────


def test_every_visible_provider_type_is_unique():
    offered = [entry["provider_type"] for entry in list_available_providers()]
    duplicates = sorted({name for name in offered if offered.count(name) > 1})
    assert duplicates == [], duplicates


def test_no_preset_shadows_a_visible_registry_provider():
    """A preset renders in its own dropdown group, so a preset whose type is also a
    non-hidden registry entry appears twice."""
    source = _frontend_source()
    visible = {
        entry["provider_type"]
        for entry in list_available_providers()
        if not entry["hidden"]
    }
    shadowing = sorted(set(_preset_types(source)) & visible)
    assert shadowing == [], shadowing


def test_presets_are_only_the_self_hosted_endpoints():
    # The three the user supplies a base URL for. A cloud vendor here would be a
    # second row, and -- via toExternalBackendProviderType -- would save as "custom".
    assert _preset_types(_frontend_source()) == ["llama_cpp", "vllm", "ollama"]


def test_every_preset_type_is_hidden_in_the_registry():
    """A preset is the visible face of its entry; the registry row must stay hidden
    or the dropdown lists it twice."""
    source = _frontend_source()
    for provider_type in _preset_types(source):
        assert PROVIDER_REGISTRY[provider_type].get("hidden") is True, provider_type


def test_presets_are_local_or_user_supplied_endpoints():
    """The entries that defer to the frontend never carry a public vendor URL."""
    for preset_type in _preset_types(_frontend_source()):
        base_url = PROVIDER_REGISTRY[preset_type]["base_url"]
        assert base_url == "" or base_url.startswith(
            ("http://localhost", "http://127.0.0.1")
        ), (preset_type, base_url)


# ── DeepInfra specifically ───────────────────────────────────────────


def test_deepinfra_is_not_a_preset():
    assert "deepinfra" not in _preset_types(_frontend_source())


def test_deepinfra_does_not_require_a_user_supplied_base_url():
    """Membership in the remote-catalog set is what makes the form demand a Base URL.
    DeepInfra's endpoint is fixed in the registry, so the field must not appear."""
    assert "deepinfra" not in _set_members(
        _frontend_source(), "REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES"
    )


def test_deepinfra_keeps_the_additional_model_ids_box():
    """Dropping the remote-catalog set must not also drop typed-in ids: a private
    ``deploy_id:`` deployment is not listed by /v1/models."""
    assert "deepinfra" in _set_members(
        _frontend_source(), "MANUAL_MODEL_IDS_WITH_CATALOG_PROVIDER_TYPES"
    )


def test_deepinfra_is_a_first_class_connection():
    entry = next(
        item
        for item in list_available_providers()
        if item["provider_type"] == "deepinfra"
    )
    # Not hidden, so the registry itself renders it -- the preset was the duplicate.
    assert entry["hidden"] is False
    assert entry["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert entry["supports_service_tier"] is True
    assert entry["service_tier_options"] == ["default", "priority", "flex"]
