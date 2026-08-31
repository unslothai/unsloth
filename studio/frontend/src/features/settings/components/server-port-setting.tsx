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
  loadServerPort,
  parseCustomServerPort,
  updateServerPort,
} from "../api/server-port";

type PortMode = "automatic" | "custom";

export function ServerPortSetting() {
  const t = useT();
  const [savedPort, setSavedPort] = useState<number | null>();
  const [mode, setMode] = useState<PortMode>("automatic");
  const [draft, setDraft] = useState("8888");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void loadServerPort()
      .then((port) => {
        if (!active) return;
        setSavedPort(port);
        setMode(port === null ? "automatic" : "custom");
        if (port !== null) setDraft(String(port));
      })
      .catch(() => {
        if (!active) return;
        setSavedPort(null);
        setError(t("settings.general.startup.loadError"));
      });
    return () => {
      active = false;
    };
  }, [t]);

  const customPort = parseCustomServerPort(draft);
  const nextPort = mode === "automatic" ? null : customPort;
  const invalid = mode === "custom" && customPort === null;
  const dirty = savedPort !== undefined && !invalid && savedPort !== nextPort;

  async function save() {
    if (!dirty) return;
    setSaving(true);
    setError(null);
    try {
      const saved = await updateServerPort(nextPort);
      setSavedPort(saved);
    } catch {
      setError(t("settings.general.startup.serverPortSaveError"));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex max-w-[320px] flex-col items-end gap-1.5">
      <div className="flex flex-wrap items-center justify-end gap-2">
        <Select
          value={mode}
          disabled={savedPort === undefined || saving}
          onValueChange={(value) => {
            setMode(value as PortMode);
            setError(null);
          }}
        >
          <SelectTrigger
            size="sm"
            className="w-32"
            aria-label={t("settings.general.startup.serverPort")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="automatic">
              {t("settings.general.startup.serverPortAutomatic")}
            </SelectItem>
            <SelectItem value="custom">
              {t("settings.general.startup.serverPortCustom")}
            </SelectItem>
          </SelectContent>
        </Select>
        {mode === "custom" ? (
          <Input
            type="number"
            min={1}
            max={65535}
            inputMode="numeric"
            value={draft}
            disabled={saving}
            aria-invalid={invalid}
            aria-label={t("settings.general.startup.serverPortCustom")}
            className="h-8 w-24"
            onChange={(event) => {
              setDraft(event.target.value);
              setError(null);
            }}
          />
        ) : null}
        <Button size="sm" disabled={!dirty || saving} onClick={() => void save()}>
          {saving ? t("common.saving") : t("common.save")}
        </Button>
      </div>
      {invalid ? (
        <span className="text-right text-xs text-destructive">
          {t("settings.general.startup.serverPortInvalid")}
        </span>
      ) : error ? (
        <span className="text-right text-xs text-destructive">{error}</span>
      ) : null}
    </div>
  );
}
