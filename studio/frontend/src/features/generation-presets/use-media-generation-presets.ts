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
  normalizeCustomPresets,
  presetSource,
} from "./preset-policy";
import type {
  MediaGenerationKind,
  MediaGenerationPreset,
  MediaGenerationPresetSettings,
  MediaPresetSource,
} from "./types";

const saveQueues = new Map<MediaGenerationKind, Promise<unknown>>();
const presetWriter =
  globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
let writeSequence = 0;

function nextWrite() {
  writeSequence += 1;
  return { writer: presetWriter, sequence: writeSequence };
}

function enqueueSave<Params, LoadConfig>(
  kind: MediaGenerationKind,
  settings: MediaGenerationPresetSettings<Params, LoadConfig>,
  keepalive = false,
) {
  const write = nextWrite();
  const operation = () =>
    saveMediaGenerationPresetSettings(kind, settings, { ...write, keepalive });
  if (keepalive) {
    return operation();
  }
  const previous = saveQueues.get(kind) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(operation);
  saveQueues.set(kind, next);
  return next;
}

export interface MediaGenerationPresetsOptions<Params extends object, LoadConfig> {
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
  const [activePresetSource, setActivePresetSource] =
    useState<MediaPresetSource>("builtin-default");
  const [hydrationSource, setHydrationSource] = useState<
    "pending" | "fresh" | "saved"
  >("pending");
  const hydrated = hydrationSource !== "pending";
  const hasPersistedSettings = hydrationSource === "saved";
  const initialParamsKey = useRef(configKey(currentParams));
  const initialLoadConfigKey = useRef(configKey(currentLoadConfig));
  const currentParamsRef = useRef(currentParams);
  const currentLoadConfigRef = useRef(currentLoadConfig);
  const latestSettingsRef = useRef<MediaGenerationPresetSettings<
    Params,
    LoadConfig
  > | null>(null);
  const baselineParamsRef = useRef(defaultParams);
  const baselineLoadConfigRef = useRef<LoadConfig | undefined>(undefined);
  const activePresetSourceRef = useRef(activePresetSource);
  const hydratedRef = useRef(hydrated);
  const defaultParamsRef = useRef(defaultParams);
  const applyParamsRef = useRef(applyParams);
  const applyLoadConfigRef = useRef(applyLoadConfig);
  const paramsKeyRef = useRef(paramsKey);
  useLayoutEffect(() => {
    currentParamsRef.current = currentParams;
    currentLoadConfigRef.current = currentLoadConfig;
    hydratedRef.current = hydrated;
    activePresetSourceRef.current =
      hydrated &&
      (paramsKey(currentParams) !== paramsKey(baselineParamsRef.current) ||
        configKey(currentLoadConfig) !==
          configKey(baselineLoadConfigRef.current))
        ? "modified"
        : activePresetSource;
    applyParamsRef.current = applyParams;
    applyLoadConfigRef.current = applyLoadConfig;
    paramsKeyRef.current = paramsKey;
  }, [
    activePresetSource,
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
    const loadConfigUntouched =
      configKey(currentLoadConfigRef.current) === initialLoadConfigKey.current;
    baselineParamsRef.current = paramsUntouched
      ? currentParamsRef.current
      : defaultParamsRef.current;
    baselineLoadConfigRef.current = undefined;
    setCustomPresets([]);
    setActivePreset(DEFAULT_PRESET_NAME);
    setActivePresetSource(
      paramsUntouched && loadConfigUntouched
        ? "builtin-default"
        : "modified",
    );
    setHydrationSource("fresh");
  }, []);

  const hydrateSavedSettings = useCallback(
    (settings: MediaGenerationPresetSettings<Params, LoadConfig>) => {
      const normalized = normalizeCustomPresets(settings.customPresets ?? []);
      setCustomPresets(normalized);
      const available = new Set([
        DEFAULT_PRESET_NAME,
        ...normalized.map((preset) => preset.name),
      ]);
      const selected = available.has(settings.activePreset)
        ? settings.activePreset
        : DEFAULT_PRESET_NAME;
      // The baseline owns model defaults applied before hydration; the resident prop can be stale.
      const currentDefaultParams = baselineParamsRef.current;
      const definition = normalized.find((preset) => preset.name === selected) ?? {
        name: DEFAULT_PRESET_NAME,
        params: currentDefaultParams,
      };
      baselineParamsRef.current = definition.params;
      baselineLoadConfigRef.current = definition.loadConfig;
      setActivePreset(definition.name);
      const paramsUntouched =
        configKey(currentParamsRef.current) === initialParamsKey.current;
      const loadConfigUntouched =
        configKey(currentLoadConfigRef.current) === initialLoadConfigKey.current;
      if (paramsUntouched) {
        const paramsToApply =
          settings.activePresetSource === "modified"
            ? settings.currentParams
            : definition.params;
        applyParamsRef.current(paramsToApply);
      }
      if (loadConfigUntouched) {
        applyLoadConfigRef.current(
          settings.activePresetSource === "modified"
            ? (settings.currentLoadConfig ?? undefined)
            : definition.loadConfig,
        );
      }
      setActivePresetSource(
        paramsUntouched && loadConfigUntouched
          ? settings.activePresetSource === "modified"
            ? "modified"
            : presetSource(definition.name)
          : "modified",
      );
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

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const key = paramsKey(currentParams);
    const nextSource =
      key === paramsKey(baselineParamsRef.current) &&
      configKey(currentLoadConfig) === configKey(baselineLoadConfigRef.current)
        ? presetSource(activePreset)
        : "modified";
    setActivePresetSource((current) =>
      current === nextSource ? current : nextSource,
    );
  }, [activePreset, currentLoadConfig, currentParams, hydrated, paramsKey]);

  const settings = useMemo<MediaGenerationPresetSettings<Params, LoadConfig>>(
    () => ({
      currentParams,
      currentLoadConfig: currentLoadConfig ?? null,
      customPresets,
      activePreset,
      activePresetSource,
    }),
    [activePreset, activePresetSource, currentLoadConfig, currentParams, customPresets],
  );
  useLayoutEffect(() => {
    latestSettingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const timer = window.setTimeout(() => {
      enqueueSave(kind, settings).catch(() => {
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
        enqueueSave(kind, latest, true).catch(() => undefined);
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
      setActivePresetSource(presetSource(preset.name));
      if (loadConfigChanged && modelLoaded) {
        toast.info(
          "Reapply the loaded model to use the selected load settings.",
        );
      }
    },
    [currentLoadConfig, customPresets, hydrated, modelLoaded],
  );

  const savePreset = useCallback(
    (rawName: string): string | null => {
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
      upsertMediaGenerationPreset(kind, preset, nextWrite()).catch(() =>
        toast.error(`Could not save ${kind} preset`),
      );
      setCustomPresets((current) => [
        ...current.filter((candidate) => candidate.name !== name),
        preset,
      ]);
      baselineParamsRef.current = preset.params;
      baselineLoadConfigRef.current = preset.loadConfig;
      setActivePreset(name);
      setActivePresetSource("custom");
      return name;
    },
    [currentLoadConfig, customPresets, hydrated, kind],
  );

  const deletePreset = useCallback(() => {
    if (!hydrated || activePreset === DEFAULT_PRESET_NAME) {
      return;
    }
    if (!customPresets.some((preset) => preset.name === activePreset)) {
      return;
    }
    deleteMediaGenerationPreset(kind, activePreset, nextWrite()).catch(() =>
      toast.error(`Could not delete ${kind} preset`),
    );
    setCustomPresets((current) =>
      current.filter((preset) => preset.name !== activePreset),
    );
    const applied = applyParamsRef.current(defaultParamsRef.current);
    const loadConfigChanged = currentLoadConfig !== undefined;
    applyLoadConfigRef.current(undefined);
    baselineParamsRef.current = applied;
    baselineLoadConfigRef.current = undefined;
    setActivePreset(DEFAULT_PRESET_NAME);
    setActivePresetSource("builtin-default");
    if (loadConfigChanged && modelLoaded) {
      toast.info("Reapply the loaded model to use the automatic load settings.");
    }
  }, [activePreset, currentLoadConfig, customPresets, hydrated, kind, modelLoaded]);

  const applyDynamicDefault = useCallback((patch: Partial<Params>) => {
    const previousDefault = defaultParamsRef.current;
    const nextDefault = { ...previousDefault, ...patch };
    defaultParamsRef.current = nextDefault;
    setEffectiveDefaultParams(nextDefault);
    const appliesToForm = activePresetSourceRef.current === "builtin-default";
    const previousBaseline = baselineParamsRef.current;
    const pending = !hydratedRef.current;
    const paramsUntouched =
      configKey(currentParamsRef.current) ===
      configKey(previousBaseline);
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
      baselineParamsRef.current = nextDefault;
    }
    return (restore: (current: Params) => Params) => {
      // A newer model default owns the definition. A preset selection owns only the form baseline,
      // so a failed load still restores Default without overwriting that newer user choice.
      if (defaultParamsRef.current !== nextDefault) {
        return false;
      }
      defaultParamsRef.current = previousDefault;
      setEffectiveDefaultParams(previousDefault);
      if (baselineParamsRef.current === nextDefault) {
        applyParamsRef.current(restore(currentParamsRef.current));
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
    activePresetSource,
    activePresetSourceRef,
    customPresets,
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
