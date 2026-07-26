# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for which model the API usage examples name, and for the
model-auto-switch control living in exactly one place on the API keys tab."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / "studio/frontend/src/features/settings"
USAGE_EXAMPLES_TSX = SETTINGS / "components/usage-examples.tsx"
OPENAI_MODELS_TS = SETTINGS / "api/openai-models.ts"
API_KEYS_TAB_TSX = SETTINGS / "tabs/api-keys-tab.tsx"


def test_examples_name_a_model_the_server_can_serve():
    # The snippets used to fall back to a hardcoded repo id whenever nothing was
    # loaded, so a copied curl named a model the user had never downloaded and
    # 404d. Read the servable ids from /v1/models instead -- the same list the
    # backend's model-not-downloaded error lists, so the two cannot disagree.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert 'from "../api/openai-models"' in src
    assert "function useExampleModelName(): string" in src
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "listOpenAIModels()" in hook
    # Precedence: live checkpoint, then a loaded catalog entry, then any entry,
    # and only then the placeholder.
    assert "catalog.find((m) => m.loaded)?.id ?? catalog[0]?.id ?? MODEL_FALLBACK" in hook

    api = OPENAI_MODELS_TS.read_text(encoding = "utf-8")
    assert 'authFetch("/v1/models")' in api


def test_usage_examples_has_no_duplicate_auto_switch_control():
    # ModelAutoSwitchSection renders the same setting immediately below this
    # panel on the same tab, and the two do not share state, so a second switch
    # here drifts out of sync with the one the user can also see.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert "openai-auto-switch" not in src
    assert "SWITCH_NOTE" not in src
    assert "Switch model by request" not in src
    assert "pythonSwitchDemo" not in src
    assert "javascriptSwitchDemo" not in src
    assert "modelAutoSwitch" not in src

    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    assert "<ModelAutoSwitchSection />" in tab
