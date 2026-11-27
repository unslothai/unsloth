// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type FirstTokenTimeoutSettings,
  firstTokenTimeoutSettings,
} from "../api/first-token-timeout";
import { SettingsRow } from "./settings-row";

type Mode = "default" | "custom" | "unlimited";
const modeFor = (minutes: number, defaultMinutes: number): Mode =>
  minutes === 0
    ? "unlimited"
    : minutes === defaultMinutes
      ? "default"
      : "custom";

export function FirstTokenTimeoutRow() {
  const t = useT();
  const [settings, setSettings] = useState<FirstTokenTimeoutSettings | null>(
    null,
  );
  const [mode, setMode] = useState<Mode>("default");
  const [customMinutes, setCustomMinutes] = useState("60");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void firstTokenTimeoutSettings(t("settings.chat.firstTokenLoadError"))
      .then((value) => {
        if (cancelled) return;
        setSettings(value);
        setMode(modeFor(value.minutes, value.default_minutes));
        if (value.minutes > 0) setCustomMinutes(String(value.minutes));
      })
      .catch((reason) => {
        if (!cancelled)
          setError(String(reason instanceof Error ? reason.message : reason));
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const minutes =
    mode === "unlimited"
      ? 0
      : mode === "default"
        ? (settings?.default_minutes ?? 20)
        : Number(customMinutes);
  const valid =
    mode !== "custom" ||
    (customMinutes.trim() !== "" &&
      Number.isInteger(minutes) &&
      minutes >= 1 &&
      minutes <= (settings?.max_minutes ?? 10080));
  const save = async () => {
    if (!valid || !settings) return;
    setSaving(true);
    setError(null);
    try {
      setSettings(
        await firstTokenTimeoutSettings(
          t("settings.chat.firstTokenSaveError"),
          minutes,
        ),
      );
    } catch (reason) {
      setError(String(reason instanceof Error ? reason.message : reason));
    } finally {
      setSaving(false);
    }
  };

  return (
    <SettingsRow
      label={t("settings.chat.firstTokenLabel")}
      description={t("settings.chat.firstTokenDescription")}
    >
      <div className="flex flex-col items-end gap-2">
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Select
            value={mode}
            onValueChange={(value) => setMode(value as Mode)}
            disabled={!settings || saving}
          >
            <SelectTrigger
              aria-label={t("settings.chat.firstTokenLabel")}
              className="w-40 data-[size=default]:h-control"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">
                {t("settings.chat.firstTokenDefault")}
              </SelectItem>
              <SelectItem value="custom">
                {t("settings.chat.firstTokenCustom")}
              </SelectItem>
              <SelectItem value="unlimited">
                {t("settings.chat.firstTokenUnlimited")}
              </SelectItem>
            </SelectContent>
          </Select>
          {mode === "custom" ? (
            <Input
              type="number"
              min={1}
              max={settings?.max_minutes ?? 10080}
              step={1}
              value={customMinutes}
              onChange={(event) => setCustomMinutes(event.target.value)}
              aria-label={t("settings.chat.firstTokenMinutes")}
              aria-invalid={!valid}
              disabled={saving}
              className="w-24"
            />
          ) : null}
          <Button
            variant="outline"
            size="sm"
            onClick={() => void save()}
            disabled={
              !settings || saving || !valid || minutes === settings.minutes
            }
          >
            {saving
              ? t("settings.chat.firstTokenSaving")
              : t("settings.chat.firstTokenSave")}
          </Button>
        </div>
        {mode === "custom" ? (
          <p className="text-xs text-muted-foreground">
            {t("settings.chat.firstTokenRange")}
          </p>
        ) : null}
        {error ? (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    </SettingsRow>
  );
}
