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
    # A hardcoded repo id made copied curls 404; read the servable ids from /v1/models.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert 'from "../api/openai-models"' in src
    assert "function useExampleModelName(): string" in src
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "listOpenAIModels()" in hook
    # Precedence: live checkpoint, then a loaded entry, then any entry if switching is on.
    assert "catalog?.find((m) => m.loaded) ?? (autoSwitch ? catalog?.[0] : undefined)" in hook
    # The snippet pins the quant so the request names the file on disk.
    assert "`${pick.id}:${pick.quant}`" in hook

    api = OPENAI_MODELS_TS.read_text(encoding = "utf-8")
    assert 'authFetch("/v1/models")' in api


def test_examples_never_print_a_hardcoded_model_id():
    # The bug this exists for: a `[]` catalog printed a snippet before /v1/models answered.
    # It is tri-state now, and the panel asks for a model instead.
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
    # A dep list missing these never re-ran, so a finished load left the first fetch's
    # name. Nor may it be gated on having no checkpoint: the store keeps one across an
    # idle unload, which changes nothing React can see.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "}, [checkpoint, ggufVariant]);" in hook
    assert "needsCatalog" not in hook
    # A finishing download moves no store state, so the fetch retries on a timer too,
    # and residency only slows that timer rather than stopping it.
    assert "CATALOG_RETRY_MS" in hook and "CATALOG_IDLE_MS" in hook
    assert "window.clearTimeout(timeoutId)" in hook
    assert "const CATALOG_RETRY_MS = 15000;" in src
    assert "const CATALOG_IDLE_MS = 60000;" in src


def test_a_stored_checkpoint_needs_catalog_evidence():
    # The store keeps a checkpoint across an idle unload and across a deletion, so
    # preferring it on the switch setting alone named a model /v1/models had proved
    # absent, and the snippets 404d instead of falling back.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert 'const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));' in hook
    # Resident, or downloaded with something able to reload it. Never the setting alone.
    assert "(!!entry && (entry.loaded || autoSwitch || idleReload))" in hook
    assert "autoSwitch ||\n" not in hook


def test_standalone_idle_unload_still_names_the_stored_checkpoint():
    # UNSLOTH_MODEL_IDLE_TTL without auto-switch reloads exactly what it freed, so the
    # stored checkpoint stays runnable and the panel must keep showing it. The stash
    # restores only that model, so it can never pick catalog[0].
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "const [idleReload, setIdleReload] = useState(false);" in hook
    assert "setIdleReload(settings[1])" in hook
    assert "s.idleUnloadActive" in hook
    # fromCatalog stays gated on auto-switch alone.
    assert "?? (autoSwitch ? catalog?.[0] : undefined)" in hook
    assert "idleReload ? catalog" not in hook


def test_a_failed_refresh_does_not_erase_what_the_server_holds():
    # Catching into [] and false made a transient error authoritative: the panel dropped
    # a still-servable model and printed "No model". The catalog is deliberately
    # tri-state, and a failure must stay the unknown state.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "listOpenAIModels().catch(() => null)" in hook
    assert ".catch(() => null)," in hook
    assert "if (models !== null) setCatalog(models);" in hook
    assert "if (settings !== null) {" in hook
    # The old negatives must be gone entirely.
    assert "catch(() => [] as OpenAIModel[])" not in hook
    assert "catch(() => [false, false] as const)" not in hook
    assert "catch(() => false)" not in hook


def test_the_pinned_quant_comes_from_the_catalog():
    # Catalog membership proves the repo, not the saved quant: the stored one can name
    # a file deleted while another quant remains, so pinning it 404d on a missing quant
    # with a runnable one listed.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find("// Backend PATH detection")]
    assert "const quant = catalog === null ? ggufVariant : entry?.quant;" in hook
    assert "`${checkpoint}:${ggufVariant}`" not in hook


def test_usage_examples_has_no_duplicate_auto_switch_control():
    # ModelAutoSwitchSection renders this setting just below and shares no state with it.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    # Reading the setting is fine; writing it here is what would be a second control.
    assert "updateOpenAIAutoSwitchSettings" not in src
    assert "SWITCH_NOTE" not in src
    assert "Switch model by request" not in src
    assert "pythonSwitchDemo" not in src
    assert "javascriptSwitchDemo" not in src
    assert "modelAutoSwitch" not in src

    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    assert "<ModelAutoSwitchSection />" in tab


# The monitor moved onto its own page; Settings keeps configuration and links across.
API_MONITOR_TSX = REPO / "studio/frontend/src/features/api-monitor/api-monitor-page.tsx"
# Their own module: the overlay mounts from __root.tsx, so importing from the page
# pulled it into the eager bundle.
API_MONITOR_LIFECYCLE_TS = REPO / "studio/frontend/src/features/api-monitor/lifecycle.ts"
MONITOR_LINK_TSX = SETTINGS / "components/monitor-link.tsx"


def test_api_monitor_history_does_not_reorder_under_the_reader():
    # The backend moves an entry to the front as it finishes, so the page pauses the poll
    # to hold the whole list still while a payload is read.
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "paused" in src
    assert "setPaused" in src
    # Filters and search are what keep 50 rows usable without paging.
    assert "filterEntries(" in src
    assert "STATUS_FILTERS" in src


def test_api_monitor_renders_lifecycle_rows():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    labels = API_MONITOR_LIFECYCLE_TS.read_text(encoding = "utf-8")
    assert "export function isLifecycleEntry(" in labels
    assert 'entry.kind === "lifecycle"' in labels
    for label in ("Loading model", "Model loaded", "Model unloaded"):
        assert label in labels
    # A lifecycle row has no prompt or reply, so it is not selectable for detail.
    assert "if (isLifecycleEntry(entry)) {" in src
    assert 'from "./lifecycle"' in src


def test_auto_switch_section_sits_above_the_usage_examples():
    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    # Configuration still comes ahead of the examples that depend on it.
    assert tab.index("<MonitorLink />") < tab.index("<ModelAutoSwitchSection />")
    assert tab.index("<ModelAutoSwitchSection />") < tab.index("<UsageExamples")


AUTO_SWITCH_TSX = SETTINGS / "components/model-auto-switch-section.tsx"
EN_TS = REPO / "studio/frontend/src/i18n/locales/en.ts"


def test_api_monitor_renders_download_rows():
    src = API_MONITOR_LIFECYCLE_TS.read_text(encoding = "utf-8")
    assert 'entry.event === "download"' in src
    for label in ("Downloading model", "Model downloaded", "Model download failed"):
        assert label in src


def test_monitor_can_unload_the_loaded_model():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "unloadActiveModel" in src
    # Always rendered so the manual release stays discoverable; disabled, not hidden.
    assert "disabled={unloading || !data?.active_model}" in src
    assert "{data?.active_model ? (" not in src
    # /unload matches on the internal id, omitted here (a host path), so read it from status.
    assert "resolveInferenceCheckpointId(status)" in src
    assert "unloadModel({ model_path: checkpoint })" in src


def test_settings_still_reaches_the_monitor():
    # The console is gone, so Settings must still have a way through to it.
    link = MONITOR_LINK_TSX.read_text(encoding = "utf-8")
    assert 'to: "/api-monitor"' in link


def test_auto_download_toggle_is_gated_on_auto_switch():
    # Downloading what auto-switch cannot load fetches gigabytes nothing can serve.
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
