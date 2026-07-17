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
import { usePlatformStore } from "@/config/env";
import { resetOnboardingDone } from "@/features/auth";
import { PermissionModeDropdown, useChatRuntimeStore } from "@/features/chat";
import { openModelsDir } from "@/features/native-intents";
import { emitTrainingRunsChanged } from "@/features/training";
import {
  setShowLlamaUpdateBanner,
  useShowLlamaUpdateBanner,
} from "@/hooks/use-llama-update-pref";
import { LOCALE_STORAGE_KEY, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { Check, Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  EmbeddingModelBlockedError,
  type EmbeddingModelSettings,
  EmbeddingModelVerificationError,
  loadEmbeddingModelSettings,
  resetEmbeddingModelSettings,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import {
  type HelperPrecacheSettings,
  loadHelperPrecacheSettings,
  updateHelperPrecacheSettings,
} from "../api/helper-precache";
import { type ModelsFolder, loadModelsFolder } from "../api/models-folder";
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
import { ChangePasswordDialog } from "../components/change-password-dialog";
import { EmbeddingModelCombobox } from "../components/embedding-model-combobox";
import { LanguageSelect } from "../components/language-select";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { StudioVersionSection } from "../components/studio-version-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

// Keys cleared by "Reset all local preferences".
// NEVER include auth/session keys here — clearing them would log the user out
// or force re-onboarding. Explicitly excluded: unsloth_auth_token,
// unsloth_auth_refresh_token, unsloth_auth_must_change_password,
// unsloth_onboarding_done.
const PREFS_KEYS: string[] = [
  // Appearance
  "theme",
  "palette",
  "unsloth_appearance_customization",
  LOCALE_STORAGE_KEY,
  // UI state
  "sidebar_pinned",
  "unsloth_sidebar_navigate_open",
  "unsloth_settings_active_tab",
  // Chat runtime prefs
  "unsloth_chat_auto_title",
  "unsloth_chat_permission_mode",
  // Legacy confirm key: loadPermissionMode falls back to it, so clear both or
  // a reset would restore the old level instead of the fresh default.
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
  "unsloth_chat_load_on_selection",
  // Model selector settings ("Select model settings" group)
  "unsloth_chat_expand_quantizations",
  "unsloth_chat_show_all_quantizations",
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
  // Profile personalization
  "unsloth_user_profile",
  // Guided tour flags
  "tour:studio:v1",
  // Update notifications
  "unsloth_show_llama_update_banner",
  "unsloth_monitor_overlay",
  // Voice settings
  "unsloth_voice_settings",
];

// Set by resetAllPrefs so the unmount-commit effect skips writing back the
// in-memory draft, else cleanup would re-persist the just-cleared HF token.
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
  const navigate = useNavigate();
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const { pathname, search } = useRouterState({
    select: (s) => ({
      pathname: s.location.pathname,
      search:
        "searchStr" in s.location
          ? ((s.location as { searchStr?: string }).searchStr ?? "")
          : typeof window !== "undefined"
            ? window.location.search
            : "",
    }),
  });
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const setHfToken = useChatRuntimeStore((s) => s.setHfToken);
  const chatOnly = usePlatformStore((s) => s.chatOnly);
  const showLlamaUpdates = useShowLlamaUpdateBanner();
  const redirectTo = `${pathname}${search}`;

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
  const [modelsFolder, setModelsFolder] = useState<ModelsFolder | null>(null);
  const [embeddingModel, setEmbeddingModel] =
    useState<EmbeddingModelSettings | null>(null);
  const [draftEmbeddingModel, setDraftEmbeddingModel] = useState("");
  const [embeddingModelError, setEmbeddingModelError] = useState<string | null>(
    null,
  );
  // Set after a 409 (unverifiable model); offers "Save anyway".
  const [embeddingModelNeedsForce, setEmbeddingModelNeedsForce] =
    useState(false);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);

  const draftRef = useRef(draftToken);
  useEffect(() => {
    draftRef.current = draftToken;
  }, [draftToken]);

  // Commit on unmount (dialog close / tab switch). Skip during reset-prefs
  // flow so we don't re-persist the draft after localStorage was cleared.
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

  // Show an "accepted" tick once a non-empty token has been committed to the
  // store and the field still matches it (i.e. not mid-edit). Gives the user
  // feedback that a pasted token was saved.
  const tokenSaved =
    draftToken.trim().length > 0 && draftToken.trim() === (hfToken ?? "");

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

  useEffect(() => {
    let cancelled = false;
    void loadEmbeddingModelSettings()
      .then((settings) => {
        if (cancelled) return;
        setEmbeddingModel(settings);
        setDraftEmbeddingModel(settings.embeddingModel);
      })
      .catch((error) => {
        if (cancelled) return;
        setEmbeddingModelError(
          error instanceof Error
            ? error.message
            : t("settings.general.rag.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  useEffect(() => {
    let cancelled = false;
    void loadModelsFolder()
      .then((folder) => {
        if (cancelled) return;
        setModelsFolder(folder);
      })
      .catch(() => {
        // Non-critical: leave the row hidden if the path can't be resolved.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Desktop opens the folder in the OS file manager; the browser can't, so it
  // falls back to copying the path (which is the info users actually want).
  const handleModelsFolder = async () => {
    const folder = modelsFolder;
    if (!folder) return;
    if (isTauri) {
      try {
        await openModelsDir(folder.path);
      } catch (error) {
        toast.error(t("settings.general.storage.openError"), {
          description: error instanceof Error ? error.message : undefined,
        });
      }
      return;
    }
    if (await copyToClipboard(folder.path)) {
      toast.success(t("settings.general.storage.copied"));
    } else {
      toast.error(t("settings.general.storage.copyError"));
    }
  };

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
      // Toggling sharing changes whether /api/train/runs returns preview_sig, so
      // refresh the history grid (hide/show the Copy preview link buttons).
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
      // The secret rotated, so any preview_sig the history grid still holds is
      // now stale. Refresh so copied links use freshly minted signatures.
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

  const saveEmbeddingModel = async (force: boolean) => {
    const trimmed = draftEmbeddingModel.trim();
    if (!trimmed) {
      setEmbeddingModelError(t("settings.general.rag.emptyError"));
      return;
    }
    setIsSavingEmbeddingModel(true);
    setEmbeddingModelError(null);
    try {
      const settings = await updateEmbeddingModelSettings(trimmed, {
        hfToken: hfToken || undefined,
        force,
      });
      setEmbeddingModel(settings);
      setDraftEmbeddingModel(settings.embeddingModel);
      setEmbeddingModelNeedsForce(false);
      toast.success(t("settings.general.rag.saved"), {
        description: t("settings.general.rag.reindexWarning"),
      });
    } catch (error) {
      // A hard security block cannot be forced; keep the "save anyway" action hidden.
      if (error instanceof EmbeddingModelBlockedError) {
        setEmbeddingModelNeedsForce(false);
      } else if (error instanceof EmbeddingModelVerificationError) {
        setEmbeddingModelNeedsForce(true);
      }
      setEmbeddingModelError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  const resetEmbeddingModel = async () => {
    setIsSavingEmbeddingModel(true);
    setEmbeddingModelError(null);
    setEmbeddingModelNeedsForce(false);
    try {
      const settings = await resetEmbeddingModelSettings();
      setEmbeddingModel(settings);
      setDraftEmbeddingModel(settings.embeddingModel);
    } catch (error) {
      setEmbeddingModelError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
    } finally {
      setIsSavingEmbeddingModel(false);
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

      <StudioVersionSection />

      <SettingsSection title={t("settings.general.account")}>
        <SettingsRow
          label={t("settings.general.huggingFaceToken")}
          description={t("settings.general.huggingFaceTokenDescription")}
        >
          <div className="relative w-[260px]">
            <Input
              type={showToken ? "text" : "password"}
              placeholder="hf_…"
              value={draftToken}
              onChange={(e) => setDraftToken(e.target.value)}
              onBlur={commitToken}
              className={cn(
                "h-8 w-full font-mono text-xs",
                tokenSaved ? "pr-14" : "pr-8",
              )}
            />
            {tokenSaved ? (
              // Decorative: pointer-events-none lets clicks reach the input
              // underneath so the field still focuses anywhere.
              <span
                className="pointer-events-none absolute right-7 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center text-emerald-600 duration-150 animate-in fade-in zoom-in dark:text-emerald-500"
                role="img"
                aria-label={t("settings.general.tokenSaved")}
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
        </SettingsRow>
        {/* The desktop app authenticates via desktop auto-auth with a generated
            secret, so there is no user-entered password to change here (and
            changing it would clear the desktop secret). Web only. */}
        {isTauri ? null : (
          <SettingsRow
            label={t("settings.general.password")}
            description={t("settings.general.passwordDescription")}
          >
            <ChangePasswordDialog />
          </SettingsRow>
        )}
      </SettingsSection>

      {modelsFolder ? (
        <SettingsSection title={t("settings.general.storage.sectionTitle")}>
          <SettingsRow
            label={t("settings.general.storage.modelsFolder")}
            description={t("settings.general.storage.modelsFolderDescription")}
          >
            <div className="flex items-center gap-2">
              <span
                title={modelsFolder.path}
                className="max-w-[280px] truncate font-mono text-xs text-muted-foreground"
              >
                {modelsFolder.path}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void handleModelsFolder()}
              >
                {isTauri
                  ? t("settings.general.storage.openAction")
                  : t("settings.general.storage.copyAction")}
              </Button>
            </div>
          </SettingsRow>
        </SettingsSection>
      ) : null}

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

      <SettingsSection title={t("settings.general.notifications.sectionTitle")}>
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

      <SettingsSection title={t("settings.general.rag.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.rag.embeddingModel")}
          description={t("settings.general.rag.embeddingModelDescription", {
            defaultModel: embeddingModel?.defaultEmbeddingModel ?? "",
          })}
          className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"
        >
          <div className="flex flex-col items-end gap-1 max-[360px]:w-full">
            <div className="flex items-center gap-2 max-[360px]:w-full">
              <EmbeddingModelCombobox
                value={draftEmbeddingModel}
                onChange={(next) => {
                  setDraftEmbeddingModel(next);
                  setEmbeddingModelNeedsForce(false);
                  setEmbeddingModelError(null);
                }}
                accessToken={hfToken || undefined}
                disabled={!embeddingModel}
                placeholder={embeddingModel?.defaultEmbeddingModel ?? ""}
                ariaLabel={t("settings.general.rag.embeddingModel")}
                className="w-[220px] max-[360px]:min-w-0 max-[360px]:flex-1"
              />
              <Button
                variant="outline"
                size="sm"
                disabled={
                  !embeddingModel ||
                  isSavingEmbeddingModel ||
                  draftEmbeddingModel.trim() === embeddingModel.embeddingModel
                }
                onClick={() => void saveEmbeddingModel(false)}
              >
                {isSavingEmbeddingModel ? t("common.saving") : t("common.save")}
              </Button>
            </div>
            {embeddingModelError ? (
              <span className="max-w-[300px] text-right text-xs text-destructive">
                {embeddingModelError}
              </span>
            ) : null}
            <div className="flex items-center gap-2">
              {embeddingModelNeedsForce ? (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={isSavingEmbeddingModel}
                  onClick={() => void saveEmbeddingModel(true)}
                >
                  {t("settings.general.rag.saveAnyway")}
                </Button>
              ) : null}
              {embeddingModel?.isCustom ? (
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={isSavingEmbeddingModel}
                  onClick={() => void resetEmbeddingModel()}
                >
                  {t("settings.general.rag.resetAction")}
                </Button>
              ) : null}
            </div>
            <span className="max-w-[300px] text-right text-xs text-muted-foreground">
              {t("settings.general.rag.reindexWarning")}
            </span>
          </div>
        </SettingsRow>
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
                  aria-label="Training dataset upload cap in MB"
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

      {!chatOnly && (
        <SettingsSection title={t("settings.general.gettingStarted")}>
          <SettingsRow
            label={t("settings.general.startOnboarding")}
            description={t("settings.general.startOnboardingDescription")}
          >
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                resetOnboardingDone();
                closeDialog();
                navigate({ to: "/onboarding", search: { redirectTo } });
              }}
            >
              {t("settings.general.startOnboardingAction")}
            </Button>
          </SettingsRow>
        </SettingsSection>
      )}

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
