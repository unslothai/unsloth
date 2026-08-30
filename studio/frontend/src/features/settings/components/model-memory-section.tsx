// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { formatBytes } from "@/features/hub/lib/format";
import { useT } from "@/i18n";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { useEffect, useState } from "react";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  subscribeModelMemorySettings,
  updateModelMemorySettings,
} from "../api/model-memory";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const MODEL_MEMORY_POLL_MS = 5000;

function coalesceRefreshes(refreshOnce: () => Promise<void>): () => void {
  let refreshInFlight = false;
  let refreshQueued = false;
  const finishRefresh = () => {
    refreshInFlight = false;
    if (refreshQueued) {
      refreshQueued = false;
      refresh();
    }
  };
  const refresh = () => {
    if (refreshInFlight) {
      refreshQueued = true;
      return;
    }
    refreshInFlight = true;
    refreshOnce().then(finishRefresh, finishRefresh);
  };
  return refresh;
}

async function refreshModelMemoryState(
  isCurrent: () => boolean,
  setSettings: (settings: ModelMemorySettings) => void,
  setError: (error: string | null) => void,
  fallbackError: string,
): Promise<void> {
  try {
    const loaded = await loadModelMemorySettings({ force: true });
    if (!isCurrent()) {
      return;
    }
    setSettings(loaded);
    setError(null);
  } catch (loadError) {
    if (!isCurrent()) {
      return;
    }
    setError(loadError instanceof Error ? loadError.message : fallbackError);
  }
}

function isRefreshCurrent(
  cancelled: boolean,
  currentGeneration: number,
  expectedGeneration: number,
): boolean {
  return !cancelled && currentGeneration === expectedGeneration;
}

export function ModelMemorySection() {
  const t = useT();
  const [settings, setSettings] = useState<ModelMemorySettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let refreshGeneration = 0;
    const refresh = coalesceRefreshes(async () => {
      const generation = ++refreshGeneration;
      await refreshModelMemoryState(
        () => isRefreshCurrent(cancelled, refreshGeneration, generation),
        setSettings,
        setLoadError,
        t("settings.resources.modelMemory.loadError"),
      );
    });

    refresh();
    const unsubscribeSettings = subscribeModelMemorySettings((next) => {
      if (cancelled) return;
      refreshGeneration += 1;
      setSettings(next);
      setLoadError(null);
    });
    const unsubscribeLifecycle = subscribeModelLifecycle(refresh);
    const timer = window.setInterval(() => {
      if (!document.hidden) refresh();
    }, MODEL_MEMORY_POLL_MS);
    const onWake = () => {
      if (!document.hidden) refresh();
    };
    window.addEventListener("focus", onWake);
    document.addEventListener("visibilitychange", onWake);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.removeEventListener("focus", onWake);
      document.removeEventListener("visibilitychange", onWake);
      unsubscribeLifecycle();
      unsubscribeSettings();
    };
  }, [t]);

  const persist = async (
    patch: Partial<Pick<ModelMemorySettings, "keepResident" | "noRamReserve">>,
  ) => {
    setIsSaving(true);
    setSaveError(null);
    try {
      setSettings(await updateModelMemorySettings(patch));
    } catch (saveFailure) {
      setSaveError(
        saveFailure instanceof Error
          ? saveFailure.message
          : t("settings.resources.modelMemory.saveError"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // A loaded child may stay locked until reload, so show the veto only after it took effect.
  const mlockVetoed =
    settings?.keepResident === true &&
    settings.noRamReserve === true &&
    settings.mlockActive === false;
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
    settings?.mlockSkipReason === "full_gpu_offload" &&
    settings.keepResident === true &&
    settings.noRamReserve === false &&
    settings.mlockActive === false;
  const mlockUngoverned =
    settings?.mlockSkipReason === "ungoverned" &&
    settings.keepResident === true &&
    settings.noRamReserve === false &&
    settings.mlockActive === false;
  const error = saveError ?? loadError;

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
          {mlockUngoverned ? (
            <p className="pb-1 text-xs text-muted-foreground">
              {t("settings.resources.modelMemory.mlockUngoverned")}
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
