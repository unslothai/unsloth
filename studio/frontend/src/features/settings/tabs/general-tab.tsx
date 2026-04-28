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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useI18n } from "@/features/i18n";
import { useSettingsDialogStore } from "@/features/settings";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

// Keys cleared by "Reset all local preferences".
//
// NEVER include auth / session keys here — resetting them would log the user
// out, which is not what users expect from a "reset preferences" button.
//
// Explicitly EXCLUDED:
//   - "unsloth_auth_token"                 (auth: access token)
//   - "unsloth_auth_refresh_token"         (auth: refresh token)
//   - "unsloth_auth_must_change_password"  (auth: forced password change flag)
//   - "unsloth_onboarding_done"            (session: would force re-onboarding)
const PREFS_KEYS: string[] = [
  // Appearance
  "theme",
  // UI state
  "sidebar_pinned",
  "unsloth_sidebar_navigate_open",
  "unsloth_settings_active_tab",
  // Chat runtime prefs
  "unsloth_chat_auto_title",
  "unsloth_hf_token",
  "unsloth_auto_heal_tool_calls",
  "unsloth_max_tool_calls_per_message",
  "unsloth_tool_call_timeout",
  "unsloth_chat_inference_params",
  "unsloth_chat_collapsible_state",
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
];

// Set to true from resetAllPrefs so the unmount-commit effect skips writing
// back the in-memory draft — otherwise the cleanup would re-persist the old
// HF token into localStorage after it was just cleared, and the subsequent
// reload would read the re-written value.
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
  const { t } = useI18n();
  const navigate = useNavigate();
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const { pathname, search } = useRouterState({
    select: (s) => ({
      pathname: s.location.pathname,
      search:
        "searchStr" in s.location
          ? (s.location as { searchStr?: string }).searchStr ?? ""
          : typeof window !== "undefined"
            ? window.location.search
            : "",
    }),
  });
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const setHfToken = useChatRuntimeStore((s) => s.setHfToken);
  const autoTitle = useChatRuntimeStore((s) => s.autoTitle);
  const setAutoTitle = useChatRuntimeStore((s) => s.setAutoTitle);
  const chatOnly = usePlatformStore((s) => s.chatOnly);
  const redirectTo = `${pathname}${search}`;

  const [draftToken, setDraftToken] = useState(hfToken ?? "");
  const [showToken, setShowToken] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

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

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">{t("settings.general.title")}</h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.general.subtitle")}
        </p>
      </header>

      <SettingsSection title={t("settings.general.account")}>
        <SettingsRow
          label={t("settings.general.hfToken.label")}
          description={t("settings.general.hfToken.description")}
        >
          <div className="relative w-[260px]">
            <Input
              type={showToken ? "text" : "password"}
              placeholder={t("settings.general.hfToken.placeholder")}
              value={draftToken}
              onChange={(e) => setDraftToken(e.target.value)}
              onBlur={commitToken}
              className="h-8 w-full pr-8 font-mono text-xs"
            />
            <button
              type="button"
              onClick={() => setShowToken((s) => !s)}
              className="absolute right-1.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
              aria-label={showToken ? t("settings.general.hfToken.hide") : t("settings.general.hfToken.show")}
              tabIndex={-1}
            >
              {showToken ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
            </button>
          </div>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.general.chatDefaults")}>
        <SettingsRow
          label={t("settings.general.autoTitle.label")}
          description={t("settings.general.autoTitle.description")}
        >
          <Switch checked={autoTitle} onCheckedChange={setAutoTitle} />
        </SettingsRow>
      </SettingsSection>

      {!chatOnly && (
        <SettingsSection title={t("settings.general.gettingStarted")}>
          <SettingsRow
            label={t("settings.general.startOnboarding.label")}
            description={t("settings.general.startOnboarding.description")}
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
              {t("settings.general.startOnboarding.cta")}
            </Button>
          </SettingsRow>
        </SettingsSection>
      )}

      <SettingsSection title={t("settings.general.dangerZone")}>
        <SettingsRow
          destructive
          label={t("settings.general.resetPrefs.label")}
          description={t("settings.general.resetPrefs.description")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            {t("settings.general.resetPrefs.cta")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t("settings.general.resetConfirm.title")}</DialogTitle>
            <DialogDescription>
              {t("settings.general.resetConfirm.description")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>{t("shutdown.cancel")}</Button>
            <Button
              onClick={resetAllPrefs}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {t("settings.general.resetConfirm.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
