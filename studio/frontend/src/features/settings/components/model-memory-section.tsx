// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { formatBytes } from "@/features/hub/lib/format";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  subscribeModelMemorySettings,
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
    // Force: mlockApplicable and reloadRequired describe the RUNNING launch, so
    // a value read before the user loaded a model would keep the panel quiet
    // about a lock that is being skipped.
    void loadModelMemorySettings({ force: true })
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
    const unsubscribe = subscribeModelMemorySettings((next) => {
      if (!cancelled) setSettings(next);
    });
    return () => {
      cancelled = true;
      unsubscribe();
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
  // Keyed on the toggles, not mlockActive: that now also reads false when the
  // running model simply had nothing in host RAM to lock, which is a different
  // reason than the one this line gives.
  const mlockVetoed =
    settings?.keepResident === true && settings.noRamReserve === true;
  // A finite locked-memory cap means llama.cpp logs "failed to mlock" and
  // carries on, so residency would look enabled but do nothing.
  const memlockCap =
    settings?.mlockActive === true && settings.memlockLimitBytes !== null
      ? settings.memlockLimitBytes
      : null;
  // The loaded model is fully on a discrete GPU, so there is no host copy to
  // pin and the lock is skipped deliberately. Without this the panel is silent:
  // the toggle reads on, mlockActive reads false, reloadRequired reads false,
  // and nothing explains why the setting changed nothing (issue #9549). Not
  // shown when no-reserve is the only enabled setting or an explicit lock is
  // active, since either case would make this explanation false.
  const mlockNotApplicable =
    settings?.mlockApplicable === false &&
    settings.keepResident === true &&
    settings.noRamReserve === false &&
    settings.mlockActive === false;

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
          {mlockNotApplicable ? (
            <p className="pb-1 text-xs text-muted-foreground">
              {t("settings.resources.modelMemory.mlockNotApplicable")}
            </p>
          ) : null}
          {memlockCap !== null ? (
            <p className="pb-1 text-xs text-amber-600 dark:text-amber-400">
              {t("settings.resources.modelMemory.memlockCapped", {
                limit: formatBytes(memlockCap),
              })}
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
