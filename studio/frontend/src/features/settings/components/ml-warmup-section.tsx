// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type MlWarmupFeature,
  type MlWarmupSettings,
  hardwareWarmupRequired,
  loadMlWarmupSettings,
  updateMlWarmupSettings,
  updateWarmupSelection,
} from "../api/ml-warmup";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const FEATURE_ROWS = [
  ["hardware", "hardwareLabel", "hardwareDescription"],
  ["text", "textLabel", "textDescription"],
  ["datasets", "datasetsLabel", "datasetsDescription"],
  ["media", "mediaLabel", "mediaDescription"],
  ["quantization", "quantizationLabel", "quantizationDescription"],
] as const;

/** Installation-wide choices; mounted only for the owner and applied on restart. */
export function MlWarmupSection() {
  const t = useT();
  const [settings, setSettings] = useState<MlWarmupSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadMlWarmupSettings()
      .then((next) => {
        if (cancelled) return;
        setSettings(next);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof Error
            ? err.message
            : t("settings.general.mlWarmup.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const save = async (feature: MlWarmupFeature, enabled: boolean) => {
    if (!settings || saving) return;
    setSaving(true);
    setError(null);
    try {
      const selected = updateWarmupSelection(settings.features, feature, enabled);
      setSettings(await updateMlWarmupSettings(selected));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("settings.general.mlWarmup.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const hardwareRequired = hardwareWarmupRequired(settings?.features ?? []);
  const note =
    error ??
    (settings?.disabledByEnv
      ? t("settings.general.mlWarmup.disabledByEnv")
      : null);

  return (
    <SettingsSection
      title={t("settings.general.mlWarmup.sectionTitle")}
      description={
        t("settings.general.mlWarmup.description") +
        " " +
        t("settings.general.mlWarmup.restartRequired")
      }
    >
      {FEATURE_ROWS.map(([feature, label, description]) => {
        const required = feature === "hardware" && hardwareRequired;
        const row = (
          <SettingsRow
            label={t(`settings.general.mlWarmup.${label}`)}
            description={
              t(`settings.general.mlWarmup.${description}`) +
              (required
                ? " " + t("settings.general.mlWarmup.requiredBySelections")
                : "")
            }
          >
            <Switch
              aria-label={t(`settings.general.mlWarmup.${label}`)}
              checked={settings?.effectiveFeatures.includes(feature) ?? false}
              disabled={!settings || saving || settings.disabledByEnv || required}
              onCheckedChange={(enabled) => void save(feature, enabled)}
            />
          </SettingsRow>
        );
        return feature === "quantization" ? (
          <div key={feature} className="mt-1 border-t border-border/60 pt-3">
            <p className="text-xs font-medium text-muted-foreground">
              {t("settings.general.mlWarmup.advanced")}
            </p>
            {row}
          </div>
        ) : (
          <div key={feature}>{row}</div>
        );
      })}
      {note ? (
        <p
          role={error ? "alert" : undefined}
          className={
            error
              ? "mt-1 text-xs text-destructive"
              : "mt-1 text-xs text-muted-foreground"
          }
        >
          {note}
        </p>
      ) : null}
    </SettingsSection>
  );
}
