// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  loadQ4nxExportSettings,
  updateQ4nxExportSettings,
  useQ4nxExportStore,
} from "../api/q4nx-export";
import { SettingsRow } from "./settings-row";

/** Owner-only opt-in for Q4NX export on a machine without an AMD NPU. */
export function Q4nxExportRow() {
  const t = useT();
  const settings = useQ4nxExportStore((s) => s.settings);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void loadQ4nxExportSettings().catch((err) =>
      setError(err instanceof Error ? err.message : String(err)),
    );
  }, []);

  const onToggle = async (next: boolean) => {
    setSaving(true);
    setError(null);
    try {
      await updateQ4nxExportSettings(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const note =
    error ??
    (settings?.npuSupported
      ? t("settings.general.developerMode.q4nxExportNpuPresent")
      : null);

  return (
    <SettingsRow
      label={t("settings.general.developerMode.q4nxExport")}
      description={
        <>
          {t("settings.general.developerMode.q4nxExportDescription")}
          {note ? (
            <span
              className={
                error
                  ? "mt-1.5 block text-destructive"
                  : "mt-1.5 block text-muted-foreground/80"
              }
            >
              {note}
            </span>
          ) : null}
        </>
      }
    >
      <Switch
        data-testid="developer-mode-q4nx-export"
        checked={settings?.npuSupported || settings?.enabled || false}
        disabled={!settings || saving || settings.npuSupported}
        onCheckedChange={(next) => void onToggle(next)}
      />
    </SettingsRow>
  );
}
