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
import { cn } from "@/lib/utils";
import {
  useChatPreferencesStore,
  composerShortcutLabels,
} from "@/features/chat";
import { useT } from "@/i18n";
import { isMacPlatform } from "../lib/keyboard-shortcuts";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ComposerSettings({ embedded = false }: { embedded?: boolean }) {
  const t = useT();
  const prefs = useChatPreferencesStore();
  const ref = useRef<HTMLDivElement>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const labels = composerShortcutLabels(prefs.sendShortcut, isMacPlatform());
  // The multiline mode's chord changes once the prompt has a line break.
  const multiline = composerShortcutLabels(prefs.sendShortcut, isMacPlatform(), "\n");
  const oppositeShortcut =
    labels.opposite === multiline.opposite
      ? labels.opposite
      : t("composerSettings.followUpMultilineShortcut", {
          shortcut: labels.opposite,
          multiline: multiline.opposite,
        });
  const mod = isMacPlatform() ? "\u2318" : "Ctrl";
  const sendDescriptionKey = {
    enter: "composerSettings.sendEnterDescription",
    "mod-enter-multiline": "composerSettings.sendMultilineDescription",
    "mod-enter": "composerSettings.sendAlwaysDescription",
  } as const;
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
      <SettingsSection
        title={t("composerSettings.title")}
        hideHeading={embedded}
      >
        <div
          className={
            embedded
              ? ""
              : "mt-2 divide-y divide-border/60 rounded-2xl border border-border/60 bg-muted/20 px-4"
          }
        >
          <SettingsRow
            label={t("composerSettings.sendShortcut")}
            description={t(sendDescriptionKey[prefs.sendShortcut], { mod })}
          >
            <Select
              value={prefs.sendShortcut}
              onValueChange={(value) =>
                prefs.setSendShortcut(
                  value === "mod-enter" || value === "mod-enter-multiline"
                    ? value
                    : "enter",
                )
              }
            >
              <SelectTrigger
                className="w-auto max-w-72"
                aria-label={t("composerSettings.sendShortcut")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectItem value="enter">Enter</SelectItem>
                <SelectItem value="mod-enter-multiline">
                  {t("composerSettings.sendMultiline", { mod })}
                </SelectItem>
                <SelectItem value="mod-enter">
                  {t("composerSettings.sendAlways", { mod })}
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>
          <SettingsRow
            label={t("composerSettings.followUp")}
            description={t("composerSettings.followUpDescription", {
              shortcut: oppositeShortcut,
            })}
            hint={t("composerSettings.steerDescription")}
          >
            {/* Shared track + pill selector, as on the API key expiry row. */}
            <div
              className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
              role="group"
              aria-label={t("composerSettings.followUp")}
            >
              {(["queue", "steer"] as const).map((behavior) => (
                <button
                  key={behavior}
                  type="button"
                  aria-pressed={prefs.followUpBehavior === behavior}
                  onClick={() => prefs.setFollowUpBehavior(behavior)}
                  className={cn(
                    "inline-flex h-8 cursor-pointer items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                    prefs.followUpBehavior === behavior
                      ? "hub-tab-toggle-pill text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
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
        </div>
      </SettingsSection>
    </div>
  );
}
