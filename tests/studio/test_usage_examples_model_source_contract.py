# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for which model the API usage examples name, and for the
model-auto-switch control living in exactly one place on the API keys tab."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / "studio/frontend/src/features/settings"
USAGE_EXAMPLES_TSX = SETTINGS / "components/usage-examples.tsx"
# The model the examples name is resolved here; the panel only renders the answer.
EXAMPLE_MODEL_TS = SETTINGS / "lib/example-model.ts"
EXAMPLE_MODEL_ID_TS = SETTINGS / "lib/example-model-id.ts"
OPENAI_MODELS_TS = SETTINGS / "api/openai-models.ts"
API_KEYS_TAB_TSX = SETTINGS / "tabs/api-keys-tab.tsx"
KEYLESS_SECTION_TSX = SETTINGS / "components/keyless-api-access-section.tsx"
KEYLESS_ELIGIBILITY_TS = SETTINGS / "components/keyless-example-eligibility.ts"

# Ends the hook slice on the declaration below it, not a comment: prose can move alone.
AFTER_HOOK = "function canUseLocalAgentDetection(base: string): boolean {"


def _hook(src: str) -> str:
    """The fetch/refresh hook, which stayed in the panel."""
    return src[src.find("function useExampleModel") : src.find(AFTER_HOOK)]


def _resolver() -> str:
    """The model choice itself, which moved into its own module."""
    return EXAMPLE_MODEL_TS.read_text(encoding = "utf-8")


def test_examples_name_a_model_the_server_can_serve():
    # A hardcoded repo id made copied curls 404; read the servable ids from /v1/models.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert 'from "../api/openai-models"' in src
    assert "function useExampleModel(" in src
    assert "listOpenAIModels()" in _hook(src)
    resolver = _resolver()
    # Precedence: live checkpoint, then a loaded entry, then any entry if switching is on.
    assert "catalog?.find((m) => m.loaded) ??" in resolver
    assert "(!keylessOnly && autoSwitch ? catalog?.[0] : undefined)" in resolver
    # The snippet pins the quant so the request names the file on disk.
    assert "return pinQuant(pick.id, vouchedQuant(pick));" in resolver
    assert 'return quant && !id.includes(":") ? `${id}:${quant}` : id;' in resolver

    api = OPENAI_MODELS_TS.read_text(encoding = "utf-8")
    assert 'authFetch("/v1/models")' in api


def test_examples_never_print_a_hardcoded_model_id():
    # The bug this exists for: a `[]` catalog printed a snippet before /v1/models answered.
    # It is tri-state now, and the panel asks for a model instead.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    assert "MODEL_FALLBACK" not in src
    # No repo-shaped literal in the panel or the resolver: a snippet may only name what
    # /v1 returns, or the one shipped example, which lives in its own module so the
    # Agents tab and this panel cannot drift apart.
    assert re.search(r'"unsloth/[^"]+"', src) is None
    assert re.search(r'"unsloth/[^"]+"', _resolver()) is None
    ids = EXAMPLE_MODEL_ID_TS.read_text(encoding = "utf-8")
    assert 'export const EXAMPLE_MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF";' in ids
    assert 'export const EXAMPLE_MODEL_VARIANT = "UD-Q4_K_XL";' in ids
    agents = (SETTINGS / "tabs/agents-tab.tsx").read_text(encoding = "utf-8")
    assert 'from "../lib/example-model-id"' in agents
    assert "useState<OpenAIModel[] | null>(null)" in src
    # Nothing servable means nothing is built, so there is nothing to copy.
    assert "(model ? buildSnippets(base, key, toolsKey, model, os) : null)" in src
    assert "if (!snippets) return;" in src
    assert "{snippets ? (" in src
    assert 't("settings.apiKeys.usageNoModel")' in src

    en = EN_TS.read_text(encoding = "utf-8")
    assert "usageNoModel:" in en


def test_catalog_refresh_follows_the_loaded_model():
    # A dep list missing these never re-ran, so a finished load left the first fetch's name. Nor may it be gated on
    # having no checkpoint: the store keeps one across an idle unload, which changes nothing React can see.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = _hook(src)
    assert "}, [checkpoint, ggufVariant]);" in hook
    assert "needsCatalog" not in hook
    # A finishing download moves no store state, so the fetch retries on a timer too, and residency only slows that
    # timer rather than stopping it.
    assert "CATALOG_RETRY_MS" in hook and "CATALOG_IDLE_MS" in hook
    assert "window.clearTimeout(timeoutId)" in hook
    assert "const CATALOG_RETRY_MS = 15000;" in src
    assert "const CATALOG_IDLE_MS = 60000;" in src


def test_a_stored_checkpoint_needs_catalog_evidence():
    # The store keeps a checkpoint across an idle unload and across a deletion, so
    # preferring it on the switch setting alone named a model /v1/models had proved
    # absent, and the snippets 404d instead of falling back.
    resolver = _resolver()
    assert (
        'const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));' in resolver
    )
    # resident, or downloaded with switching able to load this exact catalog entry.
    assert "entry.loaded || (!keylessOnly && autoSwitch)" in resolver
    assert "autoSwitch ||\n" not in resolver


def test_idle_unload_does_not_guess_the_stashed_checkpoint():
    # the idle stash is process-wide, but the browser checkpoint is not.
    hook = _hook(USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8"))
    assert hook, "the fetch hook must stay in the panel"
    for source in (hook, _resolver()):
        assert "idleReload" not in source
        assert "idleUnloadActive" not in source


def test_a_failed_refresh_does_not_erase_what_the_server_holds():
    # Catching into [] and false made a transient error authoritative: the panel dropped a still-servable model and
    # printed "No model". The catalog is deliberately tri-state, and a failure must stay the unknown state.
    src = USAGE_EXAMPLES_TSX.read_text(encoding = "utf-8")
    hook = _hook(src)
    assert "listOpenAIModels().catch(() => null)" in hook
    assert ".catch(() => null)," in hook
    assert "if (models !== null) setCatalog(models);" in hook
    assert "if (settings !== null) {" in hook
    # The old negatives must be gone entirely.
    assert "catch(() => [] as OpenAIModel[])" not in hook
    assert "catch(() => [false, false] as const)" not in hook
    assert "catch(() => false)" not in hook


def test_the_pinned_quant_comes_from_the_catalog():
    # Catalog membership proves the repo, not the saved quant: the stored one can name a file deleted while another
    # quant remains, so pinning it 404d on a missing quant with a runnable one listed.
    resolver = _resolver()
    assert "catalog === null ? (ggufVariant ?? undefined) : vouchedQuant(entry)" in resolver
    assert "`${checkpoint}:${ggufVariant}`" not in resolver
    # An explicit empty `quants` is the server declining to vouch for any pin on that
    # id, so neither path may fall back to the singular field there.
    assert "model.quants?.length === 0" in resolver


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
    # Mounted here and nowhere else; it may report its value down, but it owns it.
    assert "<ModelAutoSwitchSection" in tab


# The monitor moved onto its own page; Settings keeps configuration and links across.
API_MONITOR_TSX = REPO / "studio/frontend/src/features/api-monitor/api-monitor-page.tsx"
# Their own module: the overlay mounts from __root.tsx, so importing from the page pulled it into the eager bundle.
API_MONITOR_LIFECYCLE_TS = REPO / "studio/frontend/src/features/api-monitor/lifecycle.ts"
MONITOR_LINK_TSX = SETTINGS / "components/monitor-link.tsx"


def test_api_monitor_history_does_not_reorder_under_the_reader():
    # The backend moves an entry to the front as it finishes, so the page pauses the poll to hold the whole list still
    # while a payload is read.
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
    assert tab.index("<MonitorLink />") < tab.index("<ModelAutoSwitchSection")
    assert tab.index("<ModelAutoSwitchSection") < tab.index("<UsageExamples")


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
    assert "useExampleModel(" in src
    assert "keylessBase && !apiKey," in src
    # The section that owns the switch reports it down, so the verdict is not left to
    # a catalog poll that idles at a minute once a model is resident.
    assert "autoSwitchOverride ?? autoSwitch" in src
    assert "onEnabledChange={setAutoSwitchEnabled}" in (
        API_KEYS_TAB_TSX.read_text(encoding = "utf-8")
    )
    section = KEYLESS_SECTION_TSX.read_text(encoding = "utf-8")
    assert "[cloudflareUrl, onSettingsChange]" in section
    assert "delete" in section[section.find("  full: {") : section.find("  tools: {")]
    assert "including on localhost" in section
    assert "read the files and settings in Unsloth" not in section
