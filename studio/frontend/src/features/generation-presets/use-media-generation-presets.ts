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
  PresetWriteRefused,
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

function refusalMessage(error: unknown, fallback: string) {
  return error instanceof PresetWriteRefused ? error.message : fallback;
}

function claimForm(claim: { current: number }) {
  claim.current += 1;
  return claim.current;
}

function saveState<Params>(
  kind: MediaGenerationKind,
  settings: MediaGenerationPresetState<Params>,
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

export interface MediaGenerationPresetsOptions<Params extends object> {
  kind: MediaGenerationKind;
  defaultParams: Params;
  currentParams: Params;
  applyParams: (params: Params) => Params;
  normalizeParams?: (params: Params) => unknown;
}

export function useMediaGenerationPresets<Params extends object>({
  kind,
  defaultParams,
  currentParams,
  applyParams,
  normalizeParams,
}: MediaGenerationPresetsOptions<Params>) {
  const paramsKey = useCallback(
    (params: Params) =>
      configKey(normalizeParams ? normalizeParams(params) : params),
    [normalizeParams],
  );
  const [customPresets, setCustomPresets] = useState<
    MediaGenerationPreset<Params>[]
  >([]);
  const [effectiveDefaultParams, setEffectiveDefaultParams] =
    useState(defaultParams);
  const [activePreset, setActivePreset] = useState(DEFAULT_PRESET_NAME);
  const [hydrationSource, setHydrationSource] = useState<
    "pending" | "fresh" | "saved" | "unreadable"
  >("pending");
  // Settled, not readable: an unreadable store still answers "stored settings do not own the form",
  // which is what the load-state controls wait for. Only the preset UI itself needs a readable store.
  const hydrated = hydrationSource !== "pending";
  const presetsReady =
    hydrationSource === "fresh" || hydrationSource === "saved";
  const initialParamsKey = useRef(configKey(currentParams));
  const currentParamsRef = useRef(currentParams);
  const latestSettingsRef = useRef<MediaGenerationPresetState<Params> | null>(
    null,
  );
  const baselineParamsRef = useRef(defaultParams);
  const activePresetRef = useRef(activePreset);
  // Bumped by every action that takes over the form, so a write that resolves late can still
  // update the list without moving a selection the user made while it was in flight.
  const formClaim = useRef(0);
  const isDefaultUnmodifiedRef = useRef(true);
  const hydratedRef = useRef(hydrated);
  const defaultParamsRef = useRef(defaultParams);
  const applyParamsRef = useRef(applyParams);
  const paramsKeyRef = useRef(paramsKey);
  useLayoutEffect(() => {
    currentParamsRef.current = currentParams;
    hydratedRef.current = hydrated;
    activePresetRef.current = activePreset;
    isDefaultUnmodifiedRef.current =
      activePreset === DEFAULT_PRESET_NAME &&
      (!hydrated ||
        paramsKey(currentParams) === paramsKey(baselineParamsRef.current));
    applyParamsRef.current = applyParams;
    paramsKeyRef.current = paramsKey;
  }, [activePreset, applyParams, currentParams, hydrated, paramsKey]);

  const presets = useMemo(
    () => [
      { name: DEFAULT_PRESET_NAME, params: effectiveDefaultParams },
      ...customPresets,
    ],
    [customPresets, effectiveDefaultParams],
  );

  const hydrateLocalSettings = useCallback((source: "fresh" | "unreadable") => {
    const paramsUntouched =
      configKey(currentParamsRef.current) === initialParamsKey.current;
    baselineParamsRef.current = paramsUntouched
      ? currentParamsRef.current
      : defaultParamsRef.current;
    setCustomPresets([]);
    setActivePreset(DEFAULT_PRESET_NAME);
    setHydrationSource(source);
  }, []);

  const hydrateSavedSettings = useCallback(
    (settings: MediaGenerationPresetSettings<Params>) => {
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
      setActivePreset(definition.name);
      if (configKey(currentParamsRef.current) === initialParamsKey.current) {
        applyParamsRef.current(settings.currentParams);
      }
      setHydrationSource("saved");
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    getMediaGenerationPresetSettings<Params>(kind)
      .then((settings) => {
        if (cancelled) {
          return;
        }
        if (!settings.saved) {
          hydrateLocalSettings("fresh");
          return;
        }
        hydrateSavedSettings(settings);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        // The read is the only thing that failed. Settle the page on local defaults so the load
        // controls that wait on hydration still work, but never write back over a store this
        // session could not read.
        hydrateLocalSettings("unreadable");
        toast.error(`Could not load ${kind} presets`);
      });
    return () => {
      cancelled = true;
    };
  }, [hydrateLocalSettings, hydrateSavedSettings, kind]);

  const settings = useMemo<MediaGenerationPresetState<Params>>(
    () => ({
      currentParams,
      activePreset,
    }),
    [activePreset, currentParams],
  );
  useLayoutEffect(() => {
    latestSettingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    if (!presetsReady) {
      return;
    }
    const timer = window.setTimeout(() => {
      saveState(kind, settings).catch(() => {
        toast.error(`Could not save ${kind} presets`);
      });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [kind, presetsReady, settings]);

  useEffect(() => {
    if (!presetsReady) {
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
  }, [kind, presetsReady]);

  const selectPreset = useCallback(
    (name: string) => {
      if (!presetsReady) {
        return;
      }
      const preset =
        name === DEFAULT_PRESET_NAME
          ? { name, params: defaultParamsRef.current }
          : customPresets.find((candidate) => candidate.name === name);
      if (!preset) {
        return;
      }
      claimForm(formClaim);
      baselineParamsRef.current = applyParamsRef.current(preset.params);
      setActivePreset(preset.name);
    },
    [customPresets, presetsReady],
  );

  const savePreset = useCallback(
    async (rawName: string): Promise<string | null> => {
      if (!presetsReady) {
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
      const preset: MediaGenerationPreset<Params> = {
        name,
        params: currentParamsRef.current,
      };
      const claim = claimForm(formClaim);
      try {
        await enqueuePresetMutation(kind, (write) =>
          upsertMediaGenerationPreset(kind, preset, write),
        );
      } catch (error) {
        toast.error(refusalMessage(error, `Could not save ${kind} preset`));
        return null;
      }
      setCustomPresets((current) => [
        ...current.filter((candidate) => candidate.name !== name),
        preset,
      ]);
      if (formClaim.current !== claim) {
        return name;
      }
      baselineParamsRef.current = preset.params;
      setActivePreset(name);
      return name;
    },
    [customPresets, kind, presetsReady],
  );

  // A delete leaves Default selected. The form itself only follows when the user did not edit it
  // while the request was in flight; their newer input owns it in that case.
  const restoreDefaultAfterDelete = useCallback(
    (paramsBeforeDelete: Params) => {
      const formUnchanged =
        paramsKeyRef.current(currentParamsRef.current) ===
        paramsKeyRef.current(paramsBeforeDelete);
      baselineParamsRef.current = formUnchanged
        ? applyParamsRef.current(defaultParamsRef.current)
        : defaultParamsRef.current;
      setActivePreset(DEFAULT_PRESET_NAME);
    },
    [],
  );

  const deletePreset = useCallback(async (): Promise<boolean> => {
    if (!presetsReady || activePreset === DEFAULT_PRESET_NAME) {
      return false;
    }
    if (!customPresets.some((preset) => preset.name === activePreset)) {
      return false;
    }
    const deletedName = activePreset;
    const paramsBeforeDelete = currentParamsRef.current;
    const claim = claimForm(formClaim);
    try {
      await enqueuePresetMutation(kind, (write) =>
        deleteMediaGenerationPreset(kind, deletedName, write),
      );
    } catch (error) {
      toast.error(refusalMessage(error, `Could not delete ${kind} preset`));
      return false;
    }
    setCustomPresets((current) =>
      current.filter((preset) => preset.name !== deletedName),
    );
    if (formClaim.current === claim) {
      restoreDefaultAfterDelete(paramsBeforeDelete);
    }
    return true;
  }, [
    activePreset,
    customPresets,
    kind,
    presetsReady,
    restoreDefaultAfterDelete,
  ]);

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
      // A newly picked model's recipe takes over the form exactly as a preset selection does, so a
      // save still in flight must not put its older snapshot back over it once it answers.
      claimForm(formClaim);
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
    ? paramsKey(currentParams) !== paramsKey(activeDefinition.params)
    : false;

  return {
    activePreset,
    isDefaultUnmodifiedRef,
    presets,
    hydrated,
    presetsReady,
    hasUnsavedChanges,
    selectPreset,
    savePreset,
    deletePreset,
    applyDynamicDefault,
  };
}
