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
} from "./preset-policy";
import type {
  MediaGenerationKind,
  MediaGenerationPreset,
  MediaGenerationPresetSettings,
  MediaGenerationPresetState,
} from "./types";

function refusalMessage(error: unknown, fallback: string) {
  return error instanceof PresetWriteRefused ? error.message : fallback;
}

function claimForm(claim: { current: number }) {
  claim.current += 1;
  return claim.current;
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
  const [activePreset, setActivePreset] = useState(DEFAULT_PRESET_NAME);
  const [hydrationSource, setHydrationSource] = useState<
    "pending" | "fresh" | "saved" | "unreadable"
  >("pending");
  // Settled, not readable: an unreadable store still answers "stored settings do not own the form",
  // which is what the load-state controls wait for. Only the preset UI itself needs a readable store.
  const hydrated = hydrationSource !== "pending";
  // Whether the store supplied the recipe. Model defaults must not seed over it. This is the
  // opposite of the rule for load options, which the resident build owns (features/resident-load).
  const storedRecipe = hydrationSource === "saved";
  const presetsReady =
    hydrationSource === "fresh" || hydrationSource === "saved";
  const currentParamsRef = useRef(currentParams);
  const latestSettingsRef = useRef<MediaGenerationPresetState<Params> | null>(
    null,
  );
  const baselineParamsRef = useRef(defaultParams);
  const activePresetRef = useRef(activePreset);
  // Bumped by every action that takes over the form, so a write that resolves late can still
  // update the list without moving a selection the user made while it was in flight.
  const formClaim = useRef(0);
  const defaultParamsRef = useRef(defaultParams);
  const applyParamsRef = useRef(applyParams);
  const paramsKeyRef = useRef(paramsKey);
  useLayoutEffect(() => {
    currentParamsRef.current = currentParams;
    activePresetRef.current = activePreset;
    applyParamsRef.current = applyParams;
    defaultParamsRef.current = defaultParams;
    paramsKeyRef.current = paramsKey;
  }, [
    activePreset,
    applyParams,
    currentParams,
    defaultParams,
    hydrated,
    paramsKey,
  ]);

  const presets = useMemo(
    () => [
      { name: DEFAULT_PRESET_NAME, params: defaultParams },
      ...customPresets,
    ],
    [customPresets, defaultParams],
  );

  const hydrateLocalSettings = useCallback((source: "fresh" | "unreadable") => {
    baselineParamsRef.current = defaultParamsRef.current;
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
      // Unconditional. Comparing against the mount-time value cannot tell a user edit from a
      // model-driven one, and guessing wrong let a resident model's defaults be autosaved over
      // the stored recipe. Ordering settles it instead: the store answers, then model defaults
      // only fill a form the store did not.
      applyParamsRef.current(settings.currentParams);
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
      saveMediaGenerationPresetSettings(kind, settings).catch(() => {
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
        saveMediaGenerationPresetSettings(kind, latest, true).catch(
          () => undefined,
        );
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
        await upsertMediaGenerationPreset(kind, preset);
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
      await deleteMediaGenerationPreset(kind, deletedName);
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

  const activeDefinition =
    activePreset === DEFAULT_PRESET_NAME
      ? { name: DEFAULT_PRESET_NAME, params: defaultParams }
      : customPresets.find((preset) => preset.name === activePreset);
  const hasUnsavedChanges = activeDefinition
    ? paramsKey(currentParams) !== paramsKey(activeDefinition.params)
    : false;

  return {
    activePreset,
    presets,
    hydrated,
    storedRecipe,
    presetsReady,
    hasUnsavedChanges,
    selectPreset,
    savePreset,
    deletePreset,
  };
}
