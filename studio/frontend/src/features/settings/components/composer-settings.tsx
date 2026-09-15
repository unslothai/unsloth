// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  useChatPreferencesStore,
  composerShortcutLabels,
} from "@/features/chat";
import { useT } from "@/i18n";
import { isMacPlatform } from "../lib/keyboard-shortcuts";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ComposerSettings() {
  const t = useT();
  const prefs = useChatPreferencesStore();
  const ref = useRef<HTMLDivElement>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const labels = composerShortcutLabels(prefs.sendShortcut, isMacPlatform());
  useEffect(() => {
    if (scrollTarget !== "chat-composer") return;
    const frame = requestAnimationFrame(() => {
      ref.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      useSettingsDialogStore.getState().consumeScrollTarget("chat-composer");
    });
    return () => cancelAnimationFrame(frame);
  }, [scrollTarget]);

  return (
    <div ref={ref}>
      <SettingsSection title={t("composerSettings.title")}>
        <div className="mt-2 divide-y divide-border/60 rounded-2xl border border-border/60 bg-muted/20 px-4">
          <SettingsRow
            label={t("composerSettings.plainText")}
            description={t("composerSettings.plainTextDescription")}
          >
            <Switch
              aria-label={t("composerSettings.plainText")}
              checked={prefs.plainTextComposer}
              onCheckedChange={prefs.setPlainTextComposer}
            />
          </SettingsRow>
          <SettingsRow label={t("composerSettings.showContext")}>
            <Switch
              aria-label={t("composerSettings.showContext")}
              checked={prefs.showContextWindowUsage}
              onCheckedChange={prefs.setShowContextWindowUsage}
            />
          </SettingsRow>
          <SettingsRow
            label={t("composerSettings.sendShortcut")}
            description={t("composerSettings.sendDescription")}
          >
            <Select
              value={prefs.sendShortcut}
              onValueChange={(value) =>
                prefs.setSendShortcut(
                  value === "mod-enter" ? "mod-enter" : "enter",
                )
              }
            >
              <SelectTrigger
                className="w-36"
                aria-label={t("composerSettings.sendShortcut")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="enter">Enter</SelectItem>
                <SelectItem value="mod-enter">
                  {composerShortcutLabels("mod-enter", isMacPlatform()).send}
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>
          <SettingsRow
            label={t("composerSettings.followUp")}
            description={t("composerSettings.followUpDescription", {
              shortcut: labels.opposite,
            })}
            hint={t("composerSettings.steerDescription")}
          >
            <div
              className="inline-flex gap-1 rounded-full bg-muted p-1"
              role="group"
              aria-label={t("composerSettings.followUp")}
            >
              {(["queue", "steer"] as const).map((behavior) => (
                <button
                  key={behavior}
                  type="button"
                  aria-pressed={prefs.followUpBehavior === behavior}
                  onClick={() => prefs.setFollowUpBehavior(behavior)}
                  className="cursor-pointer rounded-full px-3 py-1.5 text-sm text-muted-foreground transition-colors aria-pressed:bg-background aria-pressed:text-foreground aria-pressed:shadow-sm"
                >
                  {t(
                    behavior === "queue"
                      ? "composerSettings.queue"
                      : "composerSettings.steer",
                  )}
                </button>
              ))}
            </div>
          </SettingsRow>
        </div>
      </SettingsSection>
    </div>
  );
}
