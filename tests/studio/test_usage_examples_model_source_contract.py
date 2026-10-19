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
KEYLESS_SECTION_TSX = SETTINGS / "components/keyless-api-access-section.tsx"
KEYLESS_ELIGIBILITY_TS = SETTINGS / "components/keyless-example-eligibility.ts"

# Ends the slice on the declaration below, not a comment: prose can move alone.
AFTER_HOOK = "function canUseLocalAgentDetection(base: string): boolean {"


def test_examples_name_a_model_the_server_can_serve():
    # A hardcoded repo id made copied curls 404; read the servable ids from /v1/models.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert 'from "../api/openai-models"' in src
    assert "function useExampleModelName(keylessOnly: boolean): string" in src
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert "listOpenAIModels()" in hook
    # Precedence: live checkpoint, then a loaded entry, then any entry if switching is on.
    assert "catalog?.find((m) => m.loaded) ??" in hook
    assert "(!keylessOnly && autoSwitch ? catalog?.[0] : undefined)" in hook
    assert "`${pick.id}:${pick.quant}`" in hook

    api = OPENAI_MODELS_TS.read_text(encoding = "utf-8")
    assert 'authFetch("/v1/models")' in api


def test_examples_never_print_a_hardcoded_model_id():
    # The catalog is tri-state: a `[]` printed a snippet before /v1/models answered.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert "MODEL_FALLBACK" not in src
    assert re.search(r'"unsloth/[^"]+"', src) is None
    assert "function useExampleModelName(keylessOnly: boolean): string | null" in src
    assert "useState<OpenAIModel[] | null>(null)" in src
    assert "(model ? buildSnippets(base, key, toolsKey, model, os) : null)" in src
    assert "if (!snippets) return;" in src
    assert "{snippets ? (" in src
    assert 't("settings.apiKeys.usageNoModel")' in src

    en = EN_TS.read_text(encoding = "utf-8")
    assert "usageNoModel:" in en


def test_catalog_refresh_follows_the_loaded_model():
    # Without these deps a finished load kept the first fetch's name; the store keeps a
    # checkpoint across idle unloads, so it cannot gate on having none.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert "}, [checkpoint, ggufVariant]);" in hook
    assert "needsCatalog" not in hook
    # A finishing download moves no store state, so the fetch also retries on a timer.
    assert "CATALOG_RETRY_MS" in hook and "CATALOG_IDLE_MS" in hook
    assert "window.clearTimeout(timeoutId)" in hook
    assert "const CATALOG_RETRY_MS = 15000;" in src
    assert "const CATALOG_IDLE_MS = 60000;" in src


def test_a_stored_checkpoint_needs_catalog_evidence():
    # The store keeps a checkpoint across idle unloads and deletions, so it must be in the catalog.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert 'const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));' in hook
    assert "entry.loaded || (!keylessOnly && autoSwitch)" in hook
    assert "autoSwitch ||\n" not in hook


def test_idle_unload_does_not_guess_the_stashed_checkpoint():
    # the idle stash is process-wide, but the browser checkpoint is not.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert "idleReload" not in hook
    assert "idleUnloadActive" not in hook


def test_a_failed_refresh_does_not_erase_what_the_server_holds():
    # A failure must stay the unknown state, not [] or false, or a transient error drops the model.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert "listOpenAIModels().catch(() => null)" in hook
    assert ".catch(() => null)," in hook
    assert "if (models !== null) setCatalog(models);" in hook
    assert "if (settings !== null) {" in hook
    assert "catch(() => [] as OpenAIModel[])" not in hook
    assert "catch(() => [false, false] as const)" not in hook
    assert "catch(() => false)" not in hook


def test_the_pinned_quant_comes_from_the_catalog():
    # Catalog membership proves the repo, not the saved quant, which may have been deleted.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = src[src.find("function useExampleModelName") : src.find(AFTER_HOOK)]
    assert "const quant = catalog === null ? ggufVariant : entry?.quant;" in hook
    assert "`${checkpoint}:${ggufVariant}`" not in hook


def test_usage_examples_has_no_duplicate_auto_switch_control():
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert "updateOpenAIAutoSwitchSettings" not in src
    assert "SWITCH_NOTE" not in src
    assert "Switch model by request" not in src
    assert "pythonSwitchDemo" not in src
    assert "javascriptSwitchDemo" not in src
    assert "modelAutoSwitch" not in src

    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    assert "<ModelAutoSwitchSection />" in tab


API_MONITOR_TSX = REPO / "studio/frontend/src/features/api-monitor/api-monitor-page.tsx"
# Own module: the overlay mounts from __root.tsx, so importing the page made it eager.
API_MONITOR_LIFECYCLE_TS = REPO / "studio/frontend/src/features/api-monitor/lifecycle.ts"
MONITOR_LINK_TSX = SETTINGS / "components/monitor-link.tsx"


def test_api_monitor_history_does_not_reorder_under_the_reader():
    # The backend reorders entries as they finish, so the page pauses polling while reading.
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    assert "paused" in src
    assert "setPaused" in src
    assert "filterEntries(" in src
    assert "STATUS_FILTERS" in src


def test_api_monitor_renders_lifecycle_rows():
    src = API_MONITOR_TSX.read_text(encoding = "utf-8")
    labels = API_MONITOR_LIFECYCLE_TS.read_text(encoding = "utf-8")
    assert "export function isLifecycleEntry(" in labels
    assert 'entry.kind === "lifecycle"' in labels
    for label in ("Loading model", "Model loaded", "Model unloaded"):
        assert label in labels
    assert "if (isLifecycleEntry(entry)) {" in src
    assert 'from "./lifecycle"' in src


def test_auto_switch_section_sits_above_the_usage_examples():
    tab = API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
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
    # Always rendered so the manual release stays discoverable; disabled, not hidden. Read the
    # button's own disabled= as a set of || terms: #11223 added `modelLoading` (no unload while a
    # model loads), and the guard is that these two stay among them, not the exact spelling.
    click = "onClick={() => void unloadActiveModel()}"
    button = src[src.index(click) + len(click) :]
    disabled = re.match(r"\s*disabled=\{([^}]*)\}", button)
    assert disabled, "the unload button no longer sits next to its disabled= prop"
    terms = {term.strip() for term in disabled.group(1).split("||")}
    assert {"unloading", "!data?.active_model"} <= terms, terms
    assert "{data?.active_model ? (" not in src
    # /unload matches on the internal id, omitted here (a host path), so read it from status.
    assert "resolveInferenceCheckpointId(status)" in src
    assert "unloadModel({ model_path: checkpoint })" in src


def test_settings_still_reaches_the_monitor():
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


def test_keyless_examples_match_transport_tool_and_full_scope_policy():
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    builder = src[src.find("function buildSnippets") : src.find("const KEY_PLACEHOLDER")]
    variants = ("curlTools", "pythonTools", "javascriptTools", "curlAdvanced")
    assert all(
        "toolsKey"
        in next(row for row in builder.splitlines() if row.strip().startswith(f"{variant}:"))
        for variant in variants
    )
    assert "keylessBase && keylessTools" in src
    assert "apiKey || (keylessBase ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER)" in src
    assert 'const KEYLESS_KEY_PLACEHOLDER = "not-needed"' in src
    assert "keylessBaseEligible(base, keylessScope, keylessExposure)" in src
    eligibility = KEYLESS_ELIGIBILITY_TS.read_text(encoding = "utf-8")
    assert 'exposure === "colab" || exposure === "public_url"' in eligibility
    assert "if (isLoopbackHost(host)) return true;" in eligibility
    assert 'return scope === "inference";' in eligibility
    assert "!(useTunnel && cloudflareUrl)" in src
    assert 'keylessBase && !apiKey && keylessScope === "inference"' in src
    section = KEYLESS_SECTION_TSX.read_text(encoding = "utf-8")
    assert "[cloudflareUrl, onSettingsChange]" in section
    assert "delete" in section[section.find("  full: {") : section.find("  tools: {")]
    assert "disabled on localhost" in section
    assert "read the files and settings in Unsloth" not in section
