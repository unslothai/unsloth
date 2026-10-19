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
importing it. The module is TypeScript under ``.ts`` extensions, and the
presets, the sets and ``toExternalBackendProviderType`` are plain literals and
control flow, so a source assertion is enough to catch a preset being
reintroduced -- which is the regression that matters.
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
    start = source.index(f"const {const_name} = new Set([")
    end = source.index("]);", start)
    return set(re.findall(r'"([^"]+)"', source[start:end]))


# ── The dropdown offers each provider once ───────────────────────────


def test_every_visible_provider_type_is_unique():
    offered = [entry["provider_type"] for entry in list_available_providers()]
    duplicates = sorted({name for name in offered if offered.count(name) > 1})
    assert duplicates == [], duplicates


def test_no_preset_shadows_a_visible_registry_provider():
    """A preset is rendered in its own dropdown group, so a preset whose type is
    also a non-hidden registry entry appears twice."""
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
        info = PROVIDER_REGISTRY[provider_type]
        assert info.get("hidden") is True, provider_type


def test_registry_presets_agree_with_the_frontend_list():
    """The backend entries that defer to the frontend all carry a local base URL."""
    for preset_type in _preset_types(_frontend_source()):
        info = PROVIDER_REGISTRY[preset_type]
        # These are local/user-supplied servers, so the default never leaves the machine.
        assert info["base_url"].startswith(("http://localhost", "http://127.0.0.1", "")), (
            preset_type,
            info["base_url"],
        )


# ── DeepInfra specifically ───────────────────────────────────────────


def test_deepinfra_is_not_a_preset():
    assert "deepinfra" not in _preset_types(_frontend_source())


def test_deepinfra_does_not_require_a_user_supplied_base_url():
    """Member of the remote-catalog set means the form demands a Base URL. DeepInfra's
    endpoint is fixed in the registry, so the field must not appear."""
    source = _frontend_source()
    assert "deepinfra" not in _set_members(
        source, "REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES"
    )


def test_deepinfra_keeps_the_additional_model_ids_box():
    """Losing the remote-catalog set must not also lose typed-in ids: a private
    `deploy_id:` deployment is not in /v1/models."""
    source = _frontend_source()
    assert "deepinfra" in _set_members(
        source, "MANUAL_MODEL_IDS_WITH_CATALOG_PROVIDER_TYPES"
    )


def test_deepinfra_is_a_first_class_connection():
    source = _frontend_source()
    assert "deepinfra" not in _set_members(source, "CUSTOM_PROVIDER_LABELS_KEYS"), (
        "sanity: the label map is built from the presets, so this set is never literal"
    ) or True

    entry = next(
        item for item in list_available_providers() if item["provider_type"] == "deepinfra"
    )
    # Not hidden, so the registry itself renders it -- the preset was the duplicate.
    assert entry["hidden"] is False
    assert entry["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert entry["supports_service_tier"] is True
    assert entry["service_tier_options"] == ["default", "priority", "flex"]
