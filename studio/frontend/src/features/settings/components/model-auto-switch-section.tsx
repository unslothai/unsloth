// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type OpenAIAutoSwitchSettings,
  loadOpenAIAutoSwitchSettings,
  updateOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ModelAutoSwitchSection() {
  const t = useT();
  const [settings, setSettings] = useState<OpenAIAutoSwitchSettings | null>(
    null,
  );
  const [draftIdleSeconds, setDraftIdleSeconds] = useState("0");
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadOpenAIAutoSwitchSettings()
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setDraftIdleSeconds(String(loaded.autoUnloadIdleSeconds));
        setError(null);
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.general.modelAutoSwitch.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  // Parse the idle-seconds draft to a non-negative integer; empty/invalid -> null.
  const parseIdleSeconds = (): number | null => {
    if (!draftIdleSeconds.trim()) {
      return null;
    }
    const parsed = Number(draftIdleSeconds);
    return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
  };

  const persist = async (
    enabled: boolean,
    idleSeconds: number | undefined,
    syncDraft = true,
    keepKv?: boolean,
  ) => {
    setIsSaving(true);
    setError(null);
    try {
      const saved = await updateOpenAIAutoSwitchSettings(
        enabled,
        idleSeconds,
        keepKv,
      );
      setSettings(saved);
      if (syncDraft) {
        setDraftIdleSeconds(String(saved.autoUnloadIdleSeconds));
      }
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.modelAutoSwitch.saveError"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Idle-unload is tied to auto-switch (the freed model reloads via the swap).
  // Toggling off preserves the saved seconds rather than zeroing them — the
  // backend gates unloading on the enabled flag, so it never unloads while off.
  // Enabling commits the drafted value, falling back to the last saved one so
  // it can never get stuck.
  const handleToggle = (enabled: boolean) => {
    const savedIdleSeconds = settings?.autoUnloadIdleSeconds ?? 0;
    if (!enabled) {
      void persist(false, savedIdleSeconds, false);
      return;
    }
    void persist(true, parseIdleSeconds() ?? savedIdleSeconds);
  };

  const handleSaveIdle = () => {
    const idleSeconds = parseIdleSeconds();
    if (idleSeconds === null) {
      setError(t("settings.general.modelAutoSwitch.idleError"));
      return;
    }
    void persist(true, idleSeconds);
  };

  const handleKeepKvToggle = (keepKv: boolean) => {
    if (!settings) return;
    void persist(settings.enabled, undefined, false, keepKv);
  };

  return (
    <SettingsSection title={t("settings.general.modelAutoSwitch.sectionTitle")}>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.enable")}
        description={t("settings.general.modelAutoSwitch.enableDescription")}
      >
        <Switch
          checked={settings?.enabled ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={handleToggle}
        />
      </SettingsRow>
      <SettingsRow
        label={t("settings.general.modelAutoSwitch.idleUnload")}
        description={t(
          "settings.general.modelAutoSwitch.idleUnloadDescription",
        )}
      >
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <Input
                type="number"
                min={0}
                step={1}
                value={draftIdleSeconds}
                aria-label="Idle auto-unload seconds"
                disabled={!settings?.enabled || isSaving}
                onChange={(event) => setDraftIdleSeconds(event.target.value)}
                className="h-8 w-24"
              />
              <span className="text-xs font-medium text-muted-foreground">
                s
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={!settings?.enabled || isSaving}
              onClick={handleSaveIdle}
            >
              {isSaving ? t("common.saving") : t("common.save")}
            </Button>
          </div>
          {error ? (
            <span className="max-w-[260px] text-right text-xs text-destructive">
              {error}
            </span>
          ) : settings && !settings.enabled && settings.idleUnloadActive ? (
            <span className="max-w-[260px] text-right text-xs text-muted-foreground">
              {t("settings.general.modelAutoSwitch.idleActiveViaEnv")}
            </span>
          ) : settings && !settings.enabled ? (
            <span className="max-w-[260px] text-right text-xs text-muted-foreground">
              {t("settings.general.modelAutoSwitch.idleNeedsEnable")}
            </span>
          ) : null}
        </div>
      </SettingsRow>
      {settings?.idleUnloadActive ? (
        <SettingsRow
          label={t("settings.general.modelAutoSwitch.keepKv")}
          description={t("settings.general.modelAutoSwitch.keepKvDescription")}
        >
          <Switch
            checked={settings.autoUnloadKeepKv}
            disabled={isSaving}
            onCheckedChange={handleKeepKvToggle}
          />
        </SettingsRow>
      ) : null}
    </SettingsSection>
  );
}
