// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  deleteMediaGenerationPreset,
  getMediaGenerationPresetSettings,
  saveMediaGenerationPresetSettings,
  upsertMediaGenerationPreset,
} from "./api";
import {
  DEFAULT_PRESET_NAME,
  configKey,
  getBuiltinVariantName,
  mergeUntouchedParams,
} from "./preset-policy";
import type {
  MediaGenerationKind,
  MediaGenerationPreset,
  MediaGenerationPresetSettings,
  MediaGenerationPresetState,
} from "./types";

const presetMutationQueues = new Map<MediaGenerationKind, Promise<unknown>>();
const presetWriter =
  globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
const writeClockKey = "unsloth_media_generation_preset_write_clock";
let lastWriteTimestamp = 0;

function nextWrite() {
  let sharedTimestamp = 0;
  try {
    const storedTimestamp = Number(
      globalThis.localStorage?.getItem(writeClockKey),
    );
    sharedTimestamp =
      Number.isSafeInteger(storedTimestamp) &&
      storedTimestamp >= 0 &&
      storedTimestamp < Number.MAX_SAFE_INTEGER
        ? storedTimestamp
        : 0;
  } catch {
    // Storage can be unavailable in a restricted webview. The page-local clock still orders writes.
  }
  lastWriteTimestamp = Math.max(
    Date.now(),
    lastWriteTimestamp + 1,
    sharedTimestamp + 1,
  );
  try {
    globalThis.localStorage?.setItem(writeClockKey, String(lastWriteTimestamp));
  } catch {
    // Best effort only. The timestamp and writer still form a total order without storage.
  }
  return {
    timestamp: lastWriteTimestamp,
    writer: presetWriter,
  };
}

function saveState<Params, LoadConfig>(
  kind: MediaGenerationKind,
  settings: MediaGenerationPresetState<Params, LoadConfig>,
  keepalive = false,
) {
  return saveMediaGenerationPresetSettings(kind, settings, {
    ...nextWrite(),
    keepalive,
  });
}

function enqueuePresetMutation<Result>(
  kind: MediaGenerationKind,
  operation: (write: ReturnType<typeof nextWrite>) => Promise<Result>,
) {
  const write = nextWrite();
  const previous = presetMutationQueues.get(kind) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(() => operation(write));
  presetMutationQueues.set(kind, next);
  return next;
}

export interface MediaGenerationPresetsOptions<
  Params extends object,
  LoadConfig,
> {
  kind: MediaGenerationKind;
  defaultParams: Params;
  currentParams: Params;
  currentLoadConfig?: LoadConfig;
  applyParams: (params: Params) => Params;
  applyLoadConfig: (config?: LoadConfig) => void;
  modelLoaded: boolean;
  normalizeParams?: (params: Params) => unknown;
}

export function useMediaGenerationPresets<Params extends object, LoadConfig>({
  kind,
  defaultParams,
  currentParams,
  currentLoadConfig,
  applyParams,
  applyLoadConfig,
  modelLoaded,
  normalizeParams,
}: MediaGenerationPresetsOptions<Params, LoadConfig>) {
  const paramsKey = useCallback(
    (params: Params) =>
      configKey(normalizeParams ? normalizeParams(params) : params),
    [normalizeParams],
  );
  const [customPresets, setCustomPresets] = useState<
    MediaGenerationPreset<Params, LoadConfig>[]
  >([]);
  const [effectiveDefaultParams, setEffectiveDefaultParams] =
    useState(defaultParams);
  const [activePreset, setActivePreset] = useState(DEFAULT_PRESET_NAME);
  const [hydrationSource, setHydrationSource] = useState<
    "pending" | "fresh" | "saved"
  >("pending");
  const hydrated = hydrationSource !== "pending";
  const hasPersistedSettings = hydrationSource === "saved";
  const initialParamsKey = useRef(configKey(currentParams));
  const initialLoadConfigKey = useRef(configKey(currentLoadConfig));
  const currentParamsRef = useRef(currentParams);
  const currentLoadConfigRef = useRef(currentLoadConfig);
  const latestSettingsRef = useRef<MediaGenerationPresetState<
    Params,
    LoadConfig
  > | null>(null);
  const baselineParamsRef = useRef(defaultParams);
  const baselineLoadConfigRef = useRef<LoadConfig | undefined>(undefined);
  const activePresetRef = useRef(activePreset);
  const isDefaultUnmodifiedRef = useRef(true);
  const hydratedRef = useRef(hydrated);
  const defaultParamsRef = useRef(defaultParams);
  const applyParamsRef = useRef(applyParams);
  const applyLoadConfigRef = useRef(applyLoadConfig);
  const paramsKeyRef = useRef(paramsKey);
  useLayoutEffect(() => {
    currentParamsRef.current = currentParams;
    currentLoadConfigRef.current = currentLoadConfig;
    hydratedRef.current = hydrated;
    activePresetRef.current = activePreset;
    isDefaultUnmodifiedRef.current =
      activePreset === DEFAULT_PRESET_NAME &&
      (!hydrated ||
        (paramsKey(currentParams) === paramsKey(baselineParamsRef.current) &&
          configKey(currentLoadConfig) ===
            configKey(baselineLoadConfigRef.current)));
    applyParamsRef.current = applyParams;
    applyLoadConfigRef.current = applyLoadConfig;
    paramsKeyRef.current = paramsKey;
  }, [
    activePreset,
    applyLoadConfig,
    applyParams,
    currentLoadConfig,
    currentParams,
    hydrated,
    paramsKey,
  ]);

  const presets = useMemo(
    () => [
      { name: DEFAULT_PRESET_NAME, params: effectiveDefaultParams },
      ...customPresets,
    ],
    [customPresets, effectiveDefaultParams],
  );

  const hydrateFreshSettings = useCallback(() => {
    const paramsUntouched =
      configKey(currentParamsRef.current) === initialParamsKey.current;
    baselineParamsRef.current = paramsUntouched
      ? currentParamsRef.current
      : defaultParamsRef.current;
    baselineLoadConfigRef.current = undefined;
    setCustomPresets([]);
    setActivePreset(DEFAULT_PRESET_NAME);
    setHydrationSource("fresh");
  }, []);

  const hydrateSavedSettings = useCallback(
    (settings: MediaGenerationPresetSettings<Params, LoadConfig>) => {
      const custom = settings.customPresets ?? [];
      setCustomPresets(custom);
      const available = new Set([
        DEFAULT_PRESET_NAME,
        ...custom.map((preset) => preset.name),
      ]);
      const selected = available.has(settings.activePreset)
        ? settings.activePreset
        : DEFAULT_PRESET_NAME;
      // The baseline owns model defaults applied before hydration; the resident prop can be stale.
      const currentDefaultParams = baselineParamsRef.current;
      const definition = custom.find((preset) => preset.name === selected) ?? {
        name: DEFAULT_PRESET_NAME,
        params: currentDefaultParams,
      };
      baselineParamsRef.current = definition.params;
      baselineLoadConfigRef.current = definition.loadConfig;
      setActivePreset(definition.name);
      const paramsUntouched =
        configKey(currentParamsRef.current) === initialParamsKey.current;
      const loadConfigUntouched =
        configKey(currentLoadConfigRef.current) ===
        initialLoadConfigKey.current;
      if (paramsUntouched) {
        applyParamsRef.current(settings.currentParams);
      }
      if (loadConfigUntouched) {
        applyLoadConfigRef.current(settings.currentLoadConfig ?? undefined);
      }
      setHydrationSource("saved");
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    getMediaGenerationPresetSettings<Params, LoadConfig>(kind)
      .then((settings) => {
        if (cancelled) {
          return;
        }
        if (!settings.saved) {
          hydrateFreshSettings();
          return;
        }
        hydrateSavedSettings(settings);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        toast.error(`Could not load ${kind} presets`);
      });
    return () => {
      cancelled = true;
    };
  }, [hydrateFreshSettings, hydrateSavedSettings, kind]);

  const settings = useMemo<MediaGenerationPresetState<Params, LoadConfig>>(
    () => ({
      currentParams,
      currentLoadConfig: currentLoadConfig ?? null,
      activePreset,
    }),
    [activePreset, currentLoadConfig, currentParams],
  );
  useLayoutEffect(() => {
    latestSettingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const timer = window.setTimeout(() => {
      saveState(kind, settings).catch(() => {
        toast.error(`Could not save ${kind} presets`);
      });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [hydrated, kind, settings]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const flush = () => {
      const latest = latestSettingsRef.current;
      if (latest) {
        saveState(kind, latest, true).catch(() => undefined);
      }
    };
    window.addEventListener("beforeunload", flush);
    return () => {
      window.removeEventListener("beforeunload", flush);
      flush();
    };
  }, [hydrated, kind]);

  const selectPreset = useCallback(
    (name: string) => {
      if (!hydrated) {
        return;
      }
      const preset =
        name === DEFAULT_PRESET_NAME
          ? { name, params: defaultParamsRef.current }
          : customPresets.find((candidate) => candidate.name === name);
      if (!preset) {
        return;
      }
      const applied = applyParamsRef.current(preset.params);
      const loadConfigChanged =
        configKey(currentLoadConfig) !== configKey(preset.loadConfig);
      applyLoadConfigRef.current(preset.loadConfig);
      baselineParamsRef.current = applied;
      baselineLoadConfigRef.current = preset.loadConfig;
      setActivePreset(preset.name);
      if (loadConfigChanged && modelLoaded) {
        toast.info(
          "Reapply the loaded model to use the selected load settings.",
        );
      }
    },
    [currentLoadConfig, customPresets, hydrated, modelLoaded],
  );

  const savePreset = useCallback(
    async (rawName: string): Promise<string | null> => {
      if (!hydrated) {
        return null;
      }
      const trimmed = rawName.trim();
      if (!trimmed) {
        toast.error("Enter a preset name");
        return null;
      }
      const usedNames = new Set([
        DEFAULT_PRESET_NAME,
        ...customPresets.map((preset) => preset.name),
      ]);
      const name =
        trimmed === DEFAULT_PRESET_NAME
          ? getBuiltinVariantName(usedNames)
          : trimmed;
      const overwriting = customPresets.some((preset) => preset.name === name);
      if (!overwriting && customPresets.length >= 100) {
        toast.error("Delete a preset before saving another one");
        return null;
      }
      const preset: MediaGenerationPreset<Params, LoadConfig> = {
        name,
        params: currentParamsRef.current,
        ...(currentLoadConfig ? { loadConfig: currentLoadConfig } : {}),
      };
      try {
        await enqueuePresetMutation(kind, (write) =>
          upsertMediaGenerationPreset(kind, preset, write),
        );
      } catch {
        toast.error(`Could not save ${kind} preset`);
        return null;
      }
      setCustomPresets((current) => [
        ...current.filter((candidate) => candidate.name !== name),
        preset,
      ]);
      baselineParamsRef.current = preset.params;
      baselineLoadConfigRef.current = preset.loadConfig;
      setActivePreset(name);
      return name;
    },
    [currentLoadConfig, customPresets, hydrated, kind],
  );

  const deletePreset = useCallback(async (): Promise<boolean> => {
    if (!hydrated || activePreset === DEFAULT_PRESET_NAME) {
      return false;
    }
    if (!customPresets.some((preset) => preset.name === activePreset)) {
      return false;
    }
    const deletedName = activePreset;
    const paramsBeforeDelete = currentParamsRef.current;
    const loadConfigBeforeDelete = currentLoadConfigRef.current;
    try {
      await enqueuePresetMutation(kind, (write) =>
        deleteMediaGenerationPreset(kind, deletedName, write),
      );
    } catch {
      toast.error(`Could not delete ${kind} preset`);
      return false;
    }
    setCustomPresets((current) =>
      current.filter((preset) => preset.name !== deletedName),
    );
    const formUnchanged =
      paramsKeyRef.current(currentParamsRef.current) ===
        paramsKeyRef.current(paramsBeforeDelete) &&
      configKey(currentLoadConfigRef.current) ===
        configKey(loadConfigBeforeDelete);
    const applied = formUnchanged
      ? applyParamsRef.current(defaultParamsRef.current)
      : defaultParamsRef.current;
    if (formUnchanged) {
      applyLoadConfigRef.current(undefined);
    }
    baselineParamsRef.current = applied;
    baselineLoadConfigRef.current = undefined;
    setActivePreset(DEFAULT_PRESET_NAME);
    if (formUnchanged && loadConfigBeforeDelete !== undefined && modelLoaded) {
      toast.info(
        "Reapply the loaded model to use the automatic load settings.",
      );
    }
    return true;
  }, [activePreset, customPresets, hydrated, kind, modelLoaded]);

  const applyDynamicDefault = useCallback((patch: Partial<Params>) => {
    const previousDefault = defaultParamsRef.current;
    const nextDefault = { ...previousDefault, ...patch };
    const previousBaseline = baselineParamsRef.current;
    const ownsDefault = activePresetRef.current === DEFAULT_PRESET_NAME;
    const appliesToForm = isDefaultUnmodifiedRef.current;
    const pending = !hydratedRef.current;
    defaultParamsRef.current = nextDefault;
    setEffectiveDefaultParams(nextDefault);
    const paramsUntouched =
      configKey(currentParamsRef.current) === configKey(previousBaseline);
    if (appliesToForm) {
      const applied = applyParamsRef.current(
        pending
          ? mergeUntouchedParams(
              previousBaseline,
              currentParamsRef.current,
              nextDefault,
            )
          : nextDefault,
      );
      if (pending && paramsUntouched) {
        initialParamsKey.current = configKey(applied);
      }
    }
    if (ownsDefault) {
      baselineParamsRef.current = nextDefault;
    }
    return () => {
      // A newer model default owns the definition. A preset selection owns only the form baseline,
      // so a failed load still restores Default without overwriting that newer user choice.
      if (defaultParamsRef.current !== nextDefault) {
        return false;
      }
      defaultParamsRef.current = previousDefault;
      setEffectiveDefaultParams(previousDefault);
      if (baselineParamsRef.current === nextDefault) {
        const current = currentParamsRef.current;
        const optimisticUntouched =
          configKey(current) === configKey(nextDefault);
        const applied = applyParamsRef.current(
          mergeUntouchedParams(nextDefault, current, previousBaseline),
        );
        if (pending && optimisticUntouched) {
          initialParamsKey.current = configKey(applied);
        }
        baselineParamsRef.current = previousBaseline;
      }
      return true;
    };
  }, []);

  const activeDefinition =
    activePreset === DEFAULT_PRESET_NAME
      ? { name: DEFAULT_PRESET_NAME, params: effectiveDefaultParams }
      : customPresets.find((preset) => preset.name === activePreset);
  const hasUnsavedChanges = activeDefinition
    ? paramsKey(currentParams) !== paramsKey(activeDefinition.params) ||
      configKey(currentLoadConfig) !== configKey(activeDefinition.loadConfig)
    : false;

  return {
    activePreset,
    isDefaultUnmodifiedRef,
    presets,
    hydrated,
    hasPersistedSettings,
    hasUnsavedChanges,
    selectPreset,
    savePreset,
    deletePreset,
    applyDynamicDefault,
  };
}
