// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { forgetAgentsMd, refreshContextUsage } from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useEffect, useId, useState } from "react";
import {
  type ChatMemorySettings,
  loadChatMemory,
  updateChatMemory,
} from "../api/chat-memory";
import { chatMemoryBytes, normaliseChatMemory } from "../lib/chat-memory-size";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ChatMemorySection() {
  const t = useT();
  const editorId = useId();
  const [settings, setSettings] = useState<ChatMemorySettings | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadChatMemory(t("settings.chat.memory.loadError"))
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setDraft(loaded.text);
        setError(null);
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.chat.memory.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const save = async (patch: { enabled?: boolean; text?: string }) => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateChatMemory(
        patch,
        t("settings.chat.memory.saveError"),
      );
      setSettings(saved);
      const submitted = patch.text;
      if (submitted !== undefined) {
        // Keep anything typed while the save was in flight.
        setDraft((current) =>
          normaliseChatMemory(current) === normaliseChatMemory(submitted)
            ? saved.text
            : current,
        );
        toast.success(t("settings.chat.memory.saved"));
      }
      forgetAgentsMd();
      void refreshContextUsage({ invalidate: true });
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.chat.memory.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const bytes = chatMemoryBytes(draft);
  const maxBytes = settings?.maxBytes ?? 32 * 1024;
  const tooLarge = bytes > maxBytes;
  const dirty =
    settings !== null &&
    normaliseChatMemory(draft) !== normaliseChatMemory(settings.text);

  return (
    <SettingsSection
      title={t("settings.chat.memory.title")}
      description={t("settings.chat.memory.description")}
    >
      <SettingsRow
        label={t("settings.chat.memory.enable")}
        description={
          settings?.lockedByEnvironment
            ? t("settings.chat.memory.locked")
            : t("settings.chat.memory.enableDescription")
        }
      >
        <Switch
          aria-label={t("settings.chat.memory.enable")}
          checked={
            (settings?.enabled ?? false) && !settings?.lockedByEnvironment
          }
          disabled={!settings || saving || settings.lockedByEnvironment}
          onCheckedChange={(enabled) => void save({ enabled })}
        />
      </SettingsRow>
      <div className="flex flex-col gap-2 pb-3">
        <label htmlFor={editorId} className="sr-only">
          {t("settings.chat.memory.editor")}
        </label>
        <Textarea
          id={editorId}
          fieldSizing="fixed"
          rows={6}
          className="max-h-[calc(320px*var(--ui-space-scale,1))] resize-y font-mono text-xs md:text-xs"
          placeholder={t("settings.chat.memory.placeholder")}
          value={draft}
          disabled={!settings}
          aria-invalid={tooLarge || undefined}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <span className="min-w-0 text-xs text-muted-foreground">
            {settings
              ? t("settings.chat.memory.hint", { path: settings.path })
              : null}
          </span>
          <div className="flex items-center gap-3">
            <span
              className={cn(
                "text-xs tabular-nums",
                tooLarge ? "text-destructive" : "text-muted-foreground",
              )}
            >
              {t("settings.chat.memory.size", {
                used: (bytes / 1024).toLocaleString(undefined, {
                  maximumFractionDigits: 1,
                }),
                max: Math.round(maxBytes / 1024).toLocaleString(),
              })}
            </span>
            <Button
              variant="outline"
              disabled={!dirty || tooLarge || saving}
              onClick={() => void save({ text: draft })}
            >
              {t("settings.chat.memory.save")}
            </Button>
          </div>
        </div>
        {error ? (
          <span role="alert" className="text-right text-xs text-destructive">
            {error}
          </span>
        ) : null}
      </div>
    </SettingsSection>
  );
}
