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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useState } from "react";
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
  // Guided tour flags
  "tour:studio:v1",
];

function resetAllPrefs() {
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
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const setHfToken = useChatRuntimeStore((s) => s.setHfToken);
  const autoTitle = useChatRuntimeStore((s) => s.autoTitle);
  const setAutoTitle = useChatRuntimeStore((s) => s.setAutoTitle);

  const [draftToken, setDraftToken] = useState(hfToken ?? "");
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">General</h1>
        <p className="text-xs text-muted-foreground">
          Global preferences for Unsloth Studio.
        </p>
      </header>

      <SettingsSection title="Account">
        <SettingsRow
          label="Hugging Face token"
          description="Used to load gated models and push artifacts."
        >
          <Input
            type="password"
            placeholder="hf_…"
            value={draftToken}
            onChange={(e) => setDraftToken(e.target.value)}
            onBlur={() => {
              if (draftToken !== hfToken) setHfToken(draftToken);
            }}
            className="h-8 w-[260px] font-mono text-xs"
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Chat defaults">
        <SettingsRow
          label="Auto-title new chats"
          description="Generate a short title from the first message."
        >
          <Switch checked={autoTitle} onCheckedChange={setAutoTitle} />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Danger zone">
        <SettingsRow
          destructive
          label="Reset all local preferences"
          description="Clears theme, tokens, sidebar state, and presets. Chats and API keys are not affected."
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            Reset preferences
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Reset all local preferences?</DialogTitle>
            <DialogDescription>
              This clears your theme, tokens, and stored settings, then reloads
              Studio. Chats and API keys are not affected.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={resetAllPrefs}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              Reset and reload
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
