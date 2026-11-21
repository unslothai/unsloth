// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  resyncInferenceStatusAfterKeptModelsUnload,
  useChatRuntimeStore,
} from "@/features/chat";
import { formatBytes } from "@/features/hub/lib/format";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useT } from "@/i18n";
import { ALLOW_OVERSIZED_LOAD_STORAGE_KEY } from "@/lib/unified-memory-load-refusal";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { type Ref, useEffect, useRef, useState } from "react";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  updateModelMemorySettings,
} from "../api/model-memory";
import {
  type MultiModelMode,
  isMultiModelMode,
  keepsModelsLoaded,
  loadMultiModelMode,
  updateMultiModelMode,
} from "../api/multi-model";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { MediaIdleUnloadSetting } from "./model-auto-switch-section";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

function MlockNotApplicableNote({
  settings,
}: { settings: ModelMemorySettings | null }) {
  const t = useT();
  if (
    !settings?.keepResident ||
    settings.noRamReserve ||
    settings.mlockActive ||
    settings.mlockApplicable
  ) {
    return null;
  }
  return (
    <p className="pb-1 text-xs text-muted-foreground">
      {t("settings.resources.modelMemory.mlockNotApplicable")}
    </p>
  );
}

// Forced: the running model may have changed since the last read.
async function refreshModelMemory(
  isCancelled: () => boolean,
  setSettings: (settings: ModelMemorySettings) => void,
  setError: (error: string | null) => void,
  fallbackError: string,
): Promise<void> {
  try {
    const loaded = await loadModelMemorySettings({ force: true });
    if (!isCancelled()) {
      setSettings(loaded);
      setError(null);
    }
  } catch (loadError) {
    if (!isCancelled()) {
      setError(loadError instanceof Error ? loadError.message : fallbackError);
    }
  }
}

const MULTI_MODEL_DESCRIPTION = {
  off: "settings.resources.modelMemory.multiModelDescription",
  fit: "settings.resources.modelMemory.multiModelDescriptionFit",
  always: "settings.resources.modelMemory.multiModelDescriptionAlways",
} as const;

// `api`: the API tab's copy, only the rows that decide which models a server can answer for.
export function ModelMemorySection({
  ref,
  variant = "full",
}: { ref?: Ref<HTMLElement>; variant?: "full" | "api" }) {
  const full = variant === "full";
  const t = useT();
  const [settings, setSettings] = useState<ModelMemorySettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  // A read resolving after a refresh or save no longer describes the panel.
  const latestRef = useRef(0);
  // Seeded from the chat store so the description does not flash before the read.
  const [mode, setMode] = useState<MultiModelMode>(() =>
    useChatRuntimeStore.getState().keepModelsLoaded ? "fit" : "off",
  );
  const [multiModelRead, setMultiModelRead] = useState(false);
  const [allowOversizedLoad, setAllowOversizedLoad] = usePersistedToggle(
    ALLOW_OVERSIZED_LOAD_STORAGE_KEY,
  );
  const [isSavingMultiModel, setIsSavingMultiModel] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadMultiModelMode(t("settings.resources.modelMemory.loadError")).then(
      (loaded) => {
        if (!cancelled) {
          setMode(loaded);
          useChatRuntimeStore
            .getState()
            .setKeepModelsLoaded(keepsModelsLoaded(loaded));
          setMultiModelRead(true);
        }
      },
      (loadError) => {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("settings.resources.modelMemory.loadError"),
          );
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [t]);

  const persistMultiModel = async (next: MultiModelMode) => {
    setIsSavingMultiModel(true);
    setError(null);
    try {
      const saved = await updateMultiModelMode(
        next,
        t("settings.resources.modelMemory.saveError"),
      );
      const wasKeeping = keepsModelsLoaded(mode);
      setMode(saved);
      useChatRuntimeStore
        .getState()
        .setKeepModelsLoaded(keepsModelsLoaded(saved));
      if (wasKeeping && !keepsModelsLoaded(saved)) {
        void resyncInferenceStatusAfterKeptModelsUnload();
      }
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.resources.modelMemory.saveError"),
      );
    } finally {
      setIsSavingMultiModel(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      const id = ++latestRef.current;
      refreshModelMemory(
        () => cancelled || id !== latestRef.current,
        setSettings,
        setError,
        t("settings.resources.modelMemory.loadError"),
      );
    };
    refresh();
    const unsubscribe = subscribeModelLifecycle(({ loading }) => {
      if (!loading) {
        refresh();
      }
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
    latestRef.current += 1;
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

  // Keyed on the toggles, not mlockActive, which is also false when nothing was in host RAM.
  const mlockVetoed =
    settings?.keepResident === true && settings.noRamReserve === true;
  // A finite memlock cap makes llama.cpp log "failed to mlock" and carry on.
  const memlockCap =
    settings?.mlockActive === true && settings.memlockLimitBytes !== null
      ? settings.memlockLimitBytes
      : null;

  return (
    <SettingsSection
      title={t("settings.resources.modelMemory.title")}
      ref={ref}
      action={
        full ? undefined : (
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            onClick={() =>
              useSettingsDialogStore
                .getState()
                .openDialog("resources", { scrollTarget: "resources-model-memory" })
            }
          >
            {t("settings.resources.modelMemory.openSystemSettings")}
            <HugeiconsIcon icon={ChevronRightStandardIcon} className="size-3" />
          </button>
        )
      }
    >
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
      {full ? (
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
      ) : null}
      <SettingsRow
        label={t("settings.resources.modelMemory.multiModel")}
        description={t(MULTI_MODEL_DESCRIPTION[mode])}
        hint={t("settings.resources.modelMemory.multiModelHint")}
      >
        <Select
          value={mode}
          disabled={!multiModelRead || isSavingMultiModel}
          onValueChange={(next) => {
            if (isMultiModelMode(next)) {
              void persistMultiModel(next);
            }
          }}
        >
          <SelectTrigger
            aria-label={t("settings.resources.modelMemory.multiModel")}
            className="w-48"
            size="sm"
            data-testid="multi-model-mode-select"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="off">
              {t("settings.resources.modelMemory.multiModelOff")}
            </SelectItem>
            <SelectItem value="fit">
              {t("settings.resources.modelMemory.multiModelFit")}
            </SelectItem>
            <SelectItem value="always">
              {t("settings.resources.modelMemory.multiModelAlways")}
            </SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>
      {full ? (
        <SettingsRow
          label={t("settings.resources.modelMemory.allowOversizedLoad")}
          description={t(
            "settings.resources.modelMemory.allowOversizedLoadDescription",
          )}
          hint={t("settings.resources.modelMemory.allowOversizedLoadHint")}
        >
          <Switch
            aria-label={t("settings.resources.modelMemory.allowOversizedLoad")}
            checked={allowOversizedLoad}
            onCheckedChange={setAllowOversizedLoad}
            data-testid="allow-oversized-load-switch"
          />
        </SettingsRow>
      ) : null}
      {error ? (
        <p className="pb-3 text-xs text-destructive">{error}</p>
      ) : (
        <>
          {mlockVetoed ? (
            <p className="pb-1 text-xs text-muted-foreground">
              {t("settings.resources.modelMemory.mlockVetoed")}
            </p>
          ) : null}
          <MlockNotApplicableNote settings={settings} />
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
      {full ? (
        <MediaIdleUnloadSetting residencyKey={settings?.keepResident} />
      ) : null}
    </SettingsSection>
  );
}
