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
    assert "catalog.find((m) => m.loaded) ?? catalog[0]" in hook
    # The snippet pins the quant so the request names the file on disk.
    assert "`${pick.id}:${pick.quant}`" in hook

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


API_MONITOR_TSX = SETTINGS / "components/api-monitor-console.tsx"


def test_api_monitor_pages_five_at_a_time():
    # 50 terminal entries are retained backend-side; the console used to dump all
    # of them into one scroller.
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "const PAGE_SIZE = 5;" in src
    assert "ordered.slice(" in src
    # Paging back must freeze the id order, or live traffic reorders history
    # under the cursor between polls.
    assert "frozenIds" in src
    assert "setFrozenIds((prev) => prev ?? entries.map((entry) => entry.id))" in src


def test_api_monitor_renders_lifecycle_rows():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "function LifecycleEntry(" in src
    assert 'entry.kind === "lifecycle"' in src
    for label in ("Loading model", "Model loaded", "Model unloaded"):
        assert label in src
    # Lifecycle rows have no prompt/reply to fetch.
    assert "isLifecycle(entry) || !expandedIds.has(entry.id)" in src


def test_auto_switch_section_sits_above_the_monitor():
    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    assert tab.index("<ModelAutoSwitchSection />") < tab.index("<ApiMonitorConsole />")
    assert tab.index("<ApiMonitorConsole />") < tab.index("<UsageExamples")
