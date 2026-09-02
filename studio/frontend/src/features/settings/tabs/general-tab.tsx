// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { PermissionModeDropdown, useChatRuntimeStore } from "@/features/chat";
// From the keys module, not the barrel or the store: both are in an import cycle with this file,
// so the key was still in its temporal dead zone when the module-scope list below read it, killing
// the module graph. The keys module imports nothing, so it is always evaluated first.
import { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "@/features/chat/stores/sidebar-organization-keys";
import {
  LOADED_MODELS_PREFERENCE_KEYS,
  setShowLoadedModels,
  useShowLoadedModels,
} from "@/features/loaded-models";

import { useHfTokenStore } from "@/features/hub";
import {
  emitTrainingRunsChanged,
  TRAINING_UI_PREFERENCE_KEYS,
} from "@/features/training";
import {
  setShowLlamaUpdateBanner,
  useShowLlamaUpdateBanner,
} from "@/hooks/use-llama-update-pref";
import { useHfTokenValidation } from "@/hooks";
import { LOCALE_STORAGE_KEY, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Check, Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  type HelperPrecacheSettings,
  loadHelperPrecacheSettings,
  updateHelperPrecacheSettings,
} from "../api/helper-precache";
import {
  type PreviewSharingSettings,
  loadPreviewSharing,
  rotatePreviewLinks,
  updatePreviewSharing,
} from "../api/preview-sharing";
import {
  DEFAULT_UPLOAD_LIMIT_MB,
  type UploadLimitSettings,
  loadUploadLimitSettings,
  updateUploadLimitSettings,
} from "../api/upload-limit";
import { loadCloseToTray, updateCloseToTray } from "../api/close-to-tray";
import { loadLaunchAtLogin, updateLaunchAtLogin } from "../api/launch-at-login";
import { ChangePasswordDialog } from "../components/change-password-dialog";
import { DesktopRepairControl } from "../components/desktop-repair-control";
import {
  DesktopUpdateControl,
  DesktopUpdateNote,
} from "../components/desktop-update-control";
import { DocumentsRagSection } from "../components/documents-rag-section";
import { LanguageSelect } from "../components/language-select";
import { TRANSPORT_MODE_STORAGE_KEY } from "@/features/hub";
import { DownloadTransportRow } from "../components/download-transport-row";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { StudioVersionSection } from "../components/studio-version-section";
import { useDesktopBooleanSetting } from "../hooks/use-desktop-boolean-setting";
import { KEYBOARD_SHORTCUTS_STORAGE_KEY } from "../stores/keyboard-shortcuts-store";
import { SETTINGS_PANEL_PREFS_STORAGE_KEY } from "../stores/settings-panel-prefs-store";
import { CHAT_PROJECT_ATTACHMENT_TARGET_KEY } from "@/features/chat/utils/project-attachment-target";

// Keys cleared by "Reset all local preferences". NEVER include auth/session keys here -- that
// would log the user out (unsloth_auth_token, unsloth_auth_refresh_token, and
// unsloth_auth_must_change_password are excluded).
const PREFS_KEYS: string[] = [
  // Appearance
  "theme",
  "palette",
  "unsloth_appearance_customization",
  LOCALE_STORAGE_KEY,
  // UI state
  "sidebar_pinned",
  "sidebar_width",
  "chat_settings_width",
  "unsloth_sidebar_navigate_open",
  // Grouping, sort and the manual row order.
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  "unsloth_settings_active_tab",
  SETTINGS_PANEL_PREFS_STORAGE_KEY,
  // Rebound chords. Without this a reset leaves the user on shortcuts they
  // asked to throw away, and a chord bound to something unusable has no
  // escape hatch from this button.
  KEYBOARD_SHORTCUTS_STORAGE_KEY,
  // Outranks the install-wide setting, so a reset that left it behind would keep ignoring
  // transport changes made elsewhere.
  TRANSPORT_MODE_STORAGE_KEY,
  // Chat runtime prefs
  CHAT_PROJECT_ATTACHMENT_TARGET_KEY,
  "unsloth_chat_auto_title",
  "unsloth_chat_permission_mode",
  // Legacy confirm key: loadPermissionMode falls back to it, so clear both or a reset restores it.
  "unsloth_chat_confirm_tool_calls",
  "unsloth_hf_token",
  "unsloth_auto_heal_tool_calls",
  "unsloth_nudge_tool_calls",
  "unsloth_max_tool_calls_per_message",
  "unsloth_tool_call_timeout",
  "unsloth_chat_inference_params",
  "unsloth_chat_collapsible_state",
  "unsloth_chat_preferences",
  "unsloth_model_configs",
  "unsloth_model_configs_migrated",
  "unsloth_load_settings",
  "unsloth_model_advanced_settings",
  "unsloth_chat_load_on_selection",
  // Model selector settings ("Select model settings" group)
  "unsloth_chat_expand_quantizations",
  "unsloth_chat_show_all_quantizations",
  // The memory bar's opt-in. Reset All advertises restoring defaults and this
  // feature's default is off, so leaving the key out left it switched on across
  // a reset that said it had turned everything back.
  //
  // Spelled out rather than imported as CHAT_SHOW_MEMORY_BAR_KEY, for the same
  // reason the note above gives: it lives in chat-runtime-store, which is in an
  // import cycle with this file, so the constant would still be in its temporal
  // dead zone when this module-scope list is built. A test pins this literal
  // against the store's constant so the two cannot drift apart silently.
  "unsloth_chat_show_memory_bar",
  "unsloth_models_fit_on_device_only",
  // Chat presets
  "unsloth_chat_custom_presets",
  "unsloth_chat_active_preset",
  "unsloth_chat_system_prompts",
  "unsloth_chat_system_prompts_migrated",
  // Training UI prefs
  "unsloth_training_config_v1",
  "unsloth_prev_max_steps",
  "unsloth_prev_save_steps",
  ...TRAINING_UI_PREFERENCE_KEYS,
  // Profile personalization
  "unsloth_user_profile",
  // Guided tour flags
  "tour:studio:v1",
  // Update notifications
  "unsloth_show_llama_update_banner",
  "unsloth_monitor_overlay",
  LOADED_MODELS_PREFERENCE_KEYS.show,
  LOADED_MODELS_PREFERENCE_KEYS.collapsed,
  LOADED_MODELS_PREFERENCE_KEYS.position,
  LOADED_MODELS_PREFERENCE_KEYS.dismissed,
  // Voice settings
  "unsloth_voice_settings",
  // Retired keys. The onboarding wizard is gone, but installs that ran it still
  // carry its flag, so a reset has to clear it or the orphan outlives the app.
  "unsloth_onboarding_done",
];

// Set by resetAllPrefs so the unmount-commit effect skips writing back the in-memory draft.
let resetInProgress = false;

function resetAllPrefs() {
  resetInProgress = true;
  for (const key of PREFS_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch {
      // ignore
    }
  }
  window.location.reload();
}

export function GeneralTab() {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const setHfToken = useChatRuntimeStore((s) => s.setHfToken);

  const hfTokenPersistenceError = useHfTokenStore(
    (s) => s.persistenceError,
  );
  const showLlamaUpdates = useShowLlamaUpdateBanner();
  const showLoadedModels = useShowLoadedModels();

  const [draftToken, setDraftToken] = useState(hfToken ?? "");
  const [showToken, setShowToken] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [uploadLimit, setUploadLimit] = useState<UploadLimitSettings | null>(
    null,
  );
  const [draftUploadLimit, setDraftUploadLimit] = useState(
    String(DEFAULT_UPLOAD_LIMIT_MB),
  );
  const [uploadLimitError, setUploadLimitError] = useState<string | null>(null);
  const [isSavingUploadLimit, setIsSavingUploadLimit] = useState(false);
  const [helperPrecache, setHelperPrecache] =
    useState<HelperPrecacheSettings | null>(null);
  const [helperPrecacheError, setHelperPrecacheError] = useState<string | null>(
    null,
  );
  const [isSavingHelperPrecache, setIsSavingHelperPrecache] = useState(false);
  const [previewSharing, setPreviewSharing] =
    useState<PreviewSharingSettings | null>(null);
  const [previewSharingError, setPreviewSharingError] = useState<string | null>(
    null,
  );
  const [isSavingPreviewSharing, setIsSavingPreviewSharing] = useState(false);
  const [revokePreviewOpen, setRevokePreviewOpen] = useState(false);
  const [isRevokingPreview, setIsRevokingPreview] = useState(false);
  const launchAtLoginSetting = useDesktopBooleanSetting({
    enabled: isTauri,
    load: loadLaunchAtLogin,
    save: updateLaunchAtLogin,
    loadError: t("settings.general.startup.loadError"),
    saveError: t("settings.general.startup.saveError"),
  });
  const closeToTraySetting = useDesktopBooleanSetting({
    enabled: isTauri,
    load: loadCloseToTray,
    save: updateCloseToTray,
    loadError: t("settings.general.startup.loadError"),
    saveError: t("settings.general.startup.closeToTraySaveError"),
  });

  const draftRef = useRef(draftToken);
  useEffect(() => {
    draftRef.current = draftToken;
  }, [draftToken]);

  // Commit on unmount (dialog close / tab switch), skipped during the reset-prefs flow.
  useEffect(() => {
    return () => {
      if (resetInProgress) return;
      const trimmed = draftRef.current.trim();
      const current = useChatRuntimeStore.getState().hfToken;
      if (trimmed !== current) {
        useChatRuntimeStore.getState().setHfToken(trimmed);
      }
    };
  }, []);

  const commitToken = () => {
    const trimmed = draftToken.trim();
    if (trimmed !== draftToken) setDraftToken(trimmed);
    if (trimmed !== hfToken) setHfToken(trimmed);
  };

  const clearHfToken = () => {
    draftRef.current = "";
    setDraftToken("");
    setHfToken("");
  };

  // Only show the success tick after the authenticated validation endpoint confirms this token:
  // a saved token alone may still be malformed, expired, or revoked.
  const tokenIsCurrent =
    draftToken.trim().length > 0 && draftToken.trim() === (hfToken ?? "");
  const tokenValidation = useHfTokenValidation(hfToken ?? "");
  const tokenValidated = tokenIsCurrent && tokenValidation.isValid === true;

  useEffect(() => {
    let cancelled = false;
    void loadUploadLimitSettings()
      .then((settings) => {
        if (cancelled) return;
        setUploadLimit(settings);
        setDraftUploadLimit(String(settings.maxUploadSizeMb));
      })
      .catch((error) => {
        if (cancelled) return;
        setUploadLimitError(
          error instanceof Error
            ? error.message
            : "Failed to load upload limit.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    void loadHelperPrecacheSettings()
      .then((settings) => {
        if (cancelled) return;
        setHelperPrecache(settings);
        setHelperPrecacheError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setHelperPrecacheError(
          error instanceof Error
            ? error.message
            : t("settings.general.helperLlm.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  useEffect(() => {
    let cancelled = false;
    void loadPreviewSharing()
      .then((settings) => {
        if (cancelled) return;
        setPreviewSharing(settings);
        setPreviewSharingError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setPreviewSharingError(
          error instanceof Error
            ? error.message
            : t("settings.general.previewSharing.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);


  const saveHelperPrecache = async (enabled: boolean) => {
    setIsSavingHelperPrecache(true);
    setHelperPrecacheError(null);
    try {
      const settings = await updateHelperPrecacheSettings(enabled);
      setHelperPrecache(settings);
    } catch (error) {
      setHelperPrecacheError(
        error instanceof Error
          ? error.message
          : t("settings.general.helperLlm.saveError"),
      );
    } finally {
      setIsSavingHelperPrecache(false);
    }
  };

  const savePreviewSharing = async (enabled: boolean) => {
    setIsSavingPreviewSharing(true);
    setPreviewSharingError(null);
    try {
      const settings = await updatePreviewSharing(enabled);
      setPreviewSharing(settings);
      // Toggling sharing changes whether /api/train/runs returns preview_sig, so refresh the grid.
      emitTrainingRunsChanged();
    } catch (error) {
      setPreviewSharingError(
        error instanceof Error
          ? error.message
          : t("settings.general.previewSharing.saveError"),
      );
    } finally {
      setIsSavingPreviewSharing(false);
    }
  };

  const revokePreviewLinks = async () => {
    setIsRevokingPreview(true);
    try {
      await rotatePreviewLinks();
      // The secret rotated, so any preview_sig the history grid holds is stale.
      emitTrainingRunsChanged();
      setRevokePreviewOpen(false);
      toast.success(t("settings.general.previewSharing.revoked"));
    } catch (error) {
      toast.error(t("settings.general.previewSharing.revokeError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setIsRevokingPreview(false);
    }
  };


  const saveUploadLimit = async () => {
    const parsed = Number(draftUploadLimit);
    if (!Number.isInteger(parsed)) {
      setUploadLimitError("Enter a whole number of MB.");
      return;
    }
    const min = uploadLimit?.minUploadSizeMb ?? 1;
    const max = uploadLimit?.maxAllowedUploadSizeMb ?? 8192;
    if (parsed < min || parsed > max) {
      setUploadLimitError(`Enter a value from ${min} to ${max} MB.`);
      return;
    }
    setIsSavingUploadLimit(true);
    setUploadLimitError(null);
    try {
      const settings = await updateUploadLimitSettings(parsed);
      setUploadLimit(settings);
      setDraftUploadLimit(String(settings.maxUploadSizeMb));
    } catch (error) {
      setUploadLimitError(
        error instanceof Error ? error.message : "Failed to save upload limit.",
      );
    } finally {
      setIsSavingUploadLimit(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.general.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.general.description")}
        </p>
      </header>

      {/* Desktop-only, and self-gating: outside the desktop app both render
          nothing and the section keeps just the version rows. */}
      <StudioVersionSection>
        {isTauri ? (
          <div data-settings-label={t("settings.about.updates")}>
            <DesktopUpdateControl />
            <DesktopUpdateNote />
          </div>
        ) : null}
      </StudioVersionSection>

      <SettingsSection title={t("settings.general.account")}>
        <SettingsRow
          label={t("settings.general.huggingFaceToken")}
          description={t("settings.general.huggingFaceTokenDescription")}
        >
          <div className="flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-2">
              <div className="relative w-[260px]">
                <Input
                  type={showToken ? "text" : "password"}
                  name="hf-token"
                  autoComplete="new-password"
                  spellCheck={false}
                  placeholder="hf_…"
                  value={draftToken}
                  onChange={(e) => setDraftToken(e.target.value)}
                  onBlur={commitToken}
                  className={cn(
                    "h-8 w-full font-mono text-xs",
                    tokenValidated ? "pr-14" : "pr-8",
                  )}
                />
                {tokenValidated ? (
                  // Decorative: pointer-events-none lets clicks reach the input underneath.
                  <span
                    className="pointer-events-none absolute right-7 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center text-emerald-600 duration-150 animate-in fade-in zoom-in dark:text-emerald-500"
                    role="img"
                    aria-label={t("settings.general.tokenValidated")}
                  >
                    <Check className="size-4" strokeWidth={2.5} />
                  </span>
                ) : null}
                <button
                  type="button"
                  onClick={() => setShowToken((s) => !s)}
                  className="absolute right-1.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
                  aria-label={
                    showToken
                      ? t("settings.general.hideToken")
                      : t("settings.general.showToken")
                  }
                  tabIndex={-1}
                >
                  {showToken ? (
                    <EyeOff className="size-3.5" />
                  ) : (
                    <Eye className="size-3.5" />
                  )}
                </button>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                disabled={!draftToken && !hfToken}
                onClick={clearHfToken}
              >
                {t("settings.general.clearToken")}
              </Button>
            </div>
            {hfTokenPersistenceError ? (
              <p className="max-w-[330px] text-right text-xs text-destructive">
                {hfTokenPersistenceError}
              </p>
            ) : tokenValidation.isChecking ? (
              <p className="text-xs text-muted-foreground">
                {t("settings.general.checkingToken")}
              </p>
            ) : tokenValidation.error ? (
              <p className="max-w-[330px] text-right text-xs text-destructive">
                {tokenValidation.error}
              </p>
            ) : null}
          </div>
        </SettingsRow>
        {/* The desktop app authenticates via desktop auto-auth with a generated
            secret, so this password only governs remote browsers and is managed
            in Remote access instead. Web only. */}
        {isTauri ? null : (
          <SettingsRow
            label={t("settings.general.password")}
            description={t("settings.general.passwordDescription")}
          >
            <ChangePasswordDialog />
          </SettingsRow>
        )}
      </SettingsSection>

      <SettingsSection title={t("settings.appearance.language.title")}>
        <SettingsRow
          label={t("settings.appearance.language.label")}
          description={t("settings.appearance.language.description")}
        >
          <LanguageSelect />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.general.permissions.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.permissions.bypassLabel")}
          description={t("settings.general.permissions.bypassDescription")}
        >
          <PermissionModeDropdown />
        </SettingsRow>
      </SettingsSection>

      {isTauri ? (
        <SettingsSection title={t("settings.general.startup.sectionTitle")}>
          <SettingsRow
            label={t("settings.general.startup.launchAtLogin")}
            description={t("settings.general.startup.launchAtLoginDescription")}
          >
            <div className="flex flex-col items-end gap-1">
              <Switch
                checked={launchAtLoginSetting.value ?? false}
                disabled={
                  launchAtLoginSetting.value === null || launchAtLoginSetting.saving
                }
                onCheckedChange={(enabled) => void launchAtLoginSetting.update(enabled)}
              />
              {launchAtLoginSetting.error ? (
                <span className="max-w-[260px] text-right text-xs text-destructive">
                  {launchAtLoginSetting.error}
                </span>
              ) : null}
            </div>
          </SettingsRow>

          {closeToTraySetting.supported ? (
            <SettingsRow
              label={t("settings.general.startup.closeToTray")}
              description={t("settings.general.startup.closeToTrayDescription")}
            >
              <div className="flex flex-col items-end gap-1">
                <Switch
                  checked={closeToTraySetting.value ?? false}
                  disabled={
                    closeToTraySetting.value === null || closeToTraySetting.saving
                  }
                  onCheckedChange={(enabled) => void closeToTraySetting.update(enabled)}
                />
                {closeToTraySetting.error ? (
                  <span className="max-w-[260px] text-right text-xs text-destructive">
                    {closeToTraySetting.error}
                  </span>
                ) : null}
              </div>
            </SettingsRow>
          ) : null}
        </SettingsSection>
      ) : null}

      <SettingsSection title={t("settings.general.notifications.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.notifications.showLoadedModels")}
          description={t(
            "settings.general.notifications.showLoadedModelsDescription",
          )}
        >
          <Switch
            checked={showLoadedModels}
            onCheckedChange={setShowLoadedModels}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.general.notifications.showLlamaUpdates")}
          description={t(
            "settings.general.notifications.showLlamaUpdatesDescription",
          )}
        >
          <Switch
            checked={showLlamaUpdates}
            onCheckedChange={setShowLlamaUpdateBanner}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.general.previewSharing.sectionTitle")}
      >
        <SettingsRow
          label={t("settings.general.previewSharing.enableLabel")}
          description={t("settings.general.previewSharing.enableDescription")}
        >
          <div className="flex flex-col items-end gap-1">
            <Switch
              checked={previewSharing?.enabled ?? false}
              disabled={!previewSharing || isSavingPreviewSharing}
              onCheckedChange={(enabled) => void savePreviewSharing(enabled)}
            />
            {previewSharingError ? (
              <span className="max-w-[260px] text-right text-xs text-destructive">
                {previewSharingError}
              </span>
            ) : null}
          </div>
        </SettingsRow>
        <SettingsRow
          destructive={true}
          label={t("settings.general.previewSharing.revokeLabel")}
          description={t("settings.general.previewSharing.revokeDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setRevokePreviewOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            {t("settings.general.previewSharing.revokeAction")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <DocumentsRagSection />

      <SettingsSection title={t("settings.general.downloads.sectionTitle")}>
        <DownloadTransportRow />
      </SettingsSection>

      <SettingsSection title={t("settings.general.uploads.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.uploads.maxUploadSize")}
          description={t("settings.general.uploads.maxUploadSizeDescription", {
            defaultSize: String(
              uploadLimit?.defaultUploadSizeMb ?? DEFAULT_UPLOAD_LIMIT_MB,
            ),
          })}
        >
          <div className="flex flex-col items-end gap-1">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <Input
                  type="number"
                  min={uploadLimit?.minUploadSizeMb ?? 1}
                  max={uploadLimit?.maxAllowedUploadSizeMb ?? 8192}
                  step={1}
                  value={draftUploadLimit}
                  aria-label={t("settings.general.uploads.maxUploadSize")}
                  onChange={(event) => setDraftUploadLimit(event.target.value)}
                  className="h-8 w-24"
                />
                <span className="text-xs font-medium text-muted-foreground">
                  MB
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={isSavingUploadLimit}
                onClick={() => void saveUploadLimit()}
              >
                {isSavingUploadLimit ? t("common.saving") : t("common.save")}
              </Button>
            </div>
            {uploadLimitError ? (
              <span className="max-w-[260px] text-right text-xs text-destructive">
                {uploadLimitError}
              </span>
            ) : null}
          </div>
        </SettingsRow>
      </SettingsSection>


      <SettingsSection title={t("settings.general.helperLlm.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.helperLlm.preloadOnStartup")}
          description={t(
            "settings.general.helperLlm.preloadOnStartupDescription",
          )}
        >
          <div className="flex flex-col items-end gap-1">
            <Switch
              checked={helperPrecache?.enabled ?? false}
              disabled={
                !helperPrecache ||
                isSavingHelperPrecache ||
                helperPrecache.disabledByEnv
              }
              onCheckedChange={(enabled) => void saveHelperPrecache(enabled)}
            />
            {helperPrecache?.disabledByEnv ? (
              <span className="max-w-[260px] text-right text-xs text-muted-foreground">
                {t("settings.general.helperLlm.disabledByEnv")}
              </span>
            ) : helperPrecacheError ? (
              <span className="max-w-[260px] text-right text-xs text-destructive">
                {helperPrecacheError}
              </span>
            ) : null}
          </div>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.general.resetPreferences.sectionTitle")}
      >
        <SettingsRow
          destructive={true}
          label={t("settings.general.resetPreferences.label")}
          description={t("settings.general.resetPreferences.description")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            {t("settings.general.resetPreferences.action")}
          </Button>
        </SettingsRow>
        {/* Same section as the reset row: both rewrite state the user cannot easily put
            back, and the desktop-only repair renders nothing on the web build, which
            would leave a section header with no rows under it if it had its own. */}
        <DesktopRepairControl />
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.general.resetPreferences.confirmTitle")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.general.resetPreferences.confirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={resetAllPrefs}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {t("settings.general.resetPreferences.confirmAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={revokePreviewOpen} onOpenChange={setRevokePreviewOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.general.previewSharing.revokeConfirmTitle")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.general.previewSharing.revokeConfirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRevokePreviewOpen(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              onClick={() => void revokePreviewLinks()}
              disabled={isRevokingPreview}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {isRevokingPreview
                ? t("settings.general.previewSharing.revoking")
                : t("settings.general.previewSharing.revokeConfirmAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
