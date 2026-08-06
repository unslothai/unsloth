// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  updateModelMemorySettings,
} from "../api/model-memory";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ModelMemorySection() {
  const t = useT();
  const [settings, setSettings] = useState<ModelMemorySettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadModelMemorySettings()
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setError(null);
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.resources.modelMemory.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const persist = async (
    patch: Partial<Pick<ModelMemorySettings, "keepResident" | "noRamReserve">>,
  ) => {
    setIsSaving(true);
    setError(null);
    try {
      setSettings(await updateModelMemorySettings(patch));
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.resources.modelMemory.saveError"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Both on suppresses --mlock. Say so, rather than looking like a no-op.
  const mlockVetoed =
    settings?.keepResident === true && settings.mlockActive === false;

  return (
    <SettingsSection title={t("settings.resources.modelMemory.title")}>
      <SettingsRow
        label={t("settings.resources.modelMemory.keepResident")}
        description={t(
          "settings.resources.modelMemory.keepResidentDescription",
        )}
        hint={t("settings.resources.modelMemory.keepResidentHint")}
      >
        <Switch
          aria-label={t("settings.resources.modelMemory.keepResident")}
          checked={settings?.keepResident ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={(keepResident) => void persist({ keepResident })}
        />
      </SettingsRow>
      <SettingsRow
        label={t("settings.resources.modelMemory.noRamReserve")}
        description={t(
          "settings.resources.modelMemory.noRamReserveDescription",
        )}
        hint={t("settings.resources.modelMemory.noRamReserveHint")}
      >
        <Switch
          aria-label={t("settings.resources.modelMemory.noRamReserve")}
          checked={settings?.noRamReserve ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={(noRamReserve) => void persist({ noRamReserve })}
        />
      </SettingsRow>
      {error ? (
        <p className="pb-3 text-xs text-destructive">{error}</p>
      ) : (
        <>
          {mlockVetoed ? (
            <p className="pb-1 text-xs text-muted-foreground">
              {t("settings.resources.modelMemory.mlockVetoed")}
            </p>
          ) : null}
          {settings?.reloadRequired ? (
            <p className="pb-3 text-xs text-muted-foreground">
              {t("settings.resources.modelMemory.reloadRequired")}
            </p>
          ) : null}
        </>
      )}
    </SettingsSection>
  );
}
