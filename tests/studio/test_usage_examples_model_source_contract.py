# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for which model the API usage examples name, and for the
model-auto-switch control living in exactly one place on the API keys tab."""

import re
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
    # and only then no model at all.
    assert "catalog?.find((m) => m.loaded) ?? catalog?.[0]" in hook
    # The snippet pins the quant so the request names the file on disk.
    assert "`${pick.id}:${pick.quant}`" in hook

    api = OPENAI_MODELS_TS.read_text(encoding = "utf-8")
    assert 'authFetch("/v1/models")' in api


def test_examples_never_print_a_hardcoded_model_id():
    # The bug this contract exists for. The catalog started as `[]`, so the very
    # first render, and every render after a slow or failed /v1/models fetch,
    # printed a copyable snippet naming a repo id the server cannot serve. The
    # catalog is tri-state now (null until /v1 answers) and the panel says to
    # load or download a model rather than naming one that does not exist.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert "MODEL_FALLBACK" not in src
    # No repo-shaped literal anywhere: a snippet may only name what /v1 returns.
    assert re.search(r'"unsloth/[^"]+"', src) is None
    assert "function useExampleModelName(): string | null" in src
    assert "useState<OpenAIModel[] | null>(null)" in src
    # Nothing servable means nothing is built, so there is nothing to copy.
    assert "(model ? buildSnippets(base, key, model, os) : null)" in src
    assert "if (!snippets) return;" in src
    assert "{snippets ? (" in src
    assert 't("settings.apiKeys.usageNoModel")' in src

    en = EN_TS.read_text(encoding = "utf-8")
    assert "usageNoModel:" in en


def test_catalog_refresh_follows_the_loaded_model():
    # `[needsCatalog]` alone never re-ran: it stays true the whole time there is
    # no local checkpoint, so a model finishing its load left the snippet naming
    # whatever the first fetch happened to see.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "}, [needsCatalog, checkpoint, ggufVariant]);" in hook
    # A finishing download moves no store state at all, so the fetch also
    # retries itself until /v1 has something, on a timer the effect clears.
    assert "window.setTimeout(update, CATALOG_RETRY_MS)" in hook
    assert "window.clearTimeout(timeoutId)" in hook
    assert "const CATALOG_RETRY_MS = 15000;" in src


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


AUTO_SWITCH_TSX = SETTINGS / "components/model-auto-switch-section.tsx"
EN_TS = REPO / "studio/frontend/src/i18n/locales/en.ts"


def test_api_monitor_renders_download_rows():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert 'entry.event === "download"' in src
    for label in ("Downloading model", "Model downloaded", "Model download failed"):
        assert label in src


def test_monitor_can_unload_the_loaded_model():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "unloadActiveModel" in src
    # Always rendered, so the manual release path stays discoverable; disabled
    # when idle or mid-unload rather than hidden.
    assert "disabled={unloading || !data?.active_model}" in src
    assert "{data?.active_model ? (" not in src
    # /unload matches on the internal id, which this response deliberately omits
    # (it would be a host path), so it must be read from status.
    assert "resolveInferenceCheckpointId(status)" in src
    assert "unloadModel({ model_path: checkpoint })" in src


def test_auto_download_toggle_is_gated_on_auto_switch():
    # Downloading a model auto-switch is not allowed to load would fetch
    # gigabytes nothing can then serve, so the row follows the enabled flag.
    src = AUTO_SWITCH_TSX.read_text(encoding = "utf-8")
    assert "modelAutoSwitch.autoDownload" in src
    assert "settings?.autoDownloadModel ?? false" in src
    row = src[src.find("modelAutoSwitch.autoDownload") :]
    assert "disabled={!settings?.enabled || isSaving}" in row[: row.find("</SettingsRow>")]


def test_auto_download_copy_warns_about_api_key_holders():
    en = EN_TS.read_text(encoding = "utf-8")
    start = en.find("autoDownloadDescription:")
    assert start != -1
    description = en[start : en.find("\n", en.find('",', start))]
    assert "API key" in description
