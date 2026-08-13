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
    "pending" | "fresh" | "saved" | "claiming" | "claimed" | "unreadable"
  >("pending");
  // Settled, not readable: an unreadable store still answers "stored settings do not own the form",
  // which is what the load-state controls wait for. Only the preset UI itself needs a readable store.
  const hydrated = hydrationSource !== "pending";
  // Whether the store supplied the recipe. Model defaults must not seed over it. This is the
  // opposite of the rule for load options, which the resident build owns (features/resident-load).
  const storedRecipe = hydrationSource === "saved";
  const presetsReady =
    hydrationSource === "fresh" ||
    hydrationSource === "saved" ||
    hydrationSource === "claimed";
  const currentParamsRef = useRef(currentParams);
  const latestSettingsRef = useRef<MediaGenerationPresetState<Params> | null>(
    null,
  );
  const deferredSavedSettingsRef =
    useRef<MediaGenerationPresetSettings<Params> | null>(null);
  const deferredFreshSettingsRef = useRef(false);
  const inflightWriteRef = useRef<Promise<unknown>>(Promise.resolve());
  const baselineParamsRef = useRef(defaultParams);
  const activePresetRef = useRef(activePreset);
  // Bumped by every action that takes over the form, so a write that resolves late can still
  // update the list without moving a selection the user made while it was in flight.
  const formClaim = useRef(0);
  const committedRecipeClaim = useRef(0);
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

  // `custom` for a store that holds named presets but no recipe of its own: the library is still
  // the user's, and only the recipe falls back to the model's defaults.
  const hydrateLocalSettings = useCallback(
    (
      source: "fresh" | "unreadable",
      custom: MediaGenerationPreset<Params>[] = [],
    ) => {
      deferredSavedSettingsRef.current = null;
      deferredFreshSettingsRef.current = false;
      baselineParamsRef.current = defaultParamsRef.current;
      setCustomPresets(custom);
      setActivePreset(DEFAULT_PRESET_NAME);
      if (source === "fresh" && formClaim.current !== 0) {
        const committed = committedRecipeClaim.current === formClaim.current;
        deferredFreshSettingsRef.current = !committed;
        setHydrationSource(committed ? "claimed" : "claiming");
        return;
      }
      setHydrationSource(source);
    },
    [],
  );

  const hydrateSavedSettings = useCallback(
    (settings: MediaGenerationPresetSettings<Params>) => {
      const custom = settings.customPresets ?? [];
      deferredFreshSettingsRef.current = false;
      setCustomPresets(custom);
      // A model pick made while the request was in flight is newer than storage. Keep its form
      // values, but still hydrate the named presets so the user's library remains available.
      if (formClaim.current !== 0) {
        const committed = committedRecipeClaim.current === formClaim.current;
        deferredSavedSettingsRef.current = committed ? null : settings;
        setHydrationSource(committed ? "claimed" : "claiming");
        return;
      }
      deferredSavedSettingsRef.current = null;
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
      // Unconditional: comparing against the mount-time value cannot tell a user edit from a
      // model-driven one, and guessing wrong autosaved a resident model's defaults over the
      // stored recipe. Ordering settles it -- the store answers, then defaults fill the rest.
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
          hydrateLocalSettings("fresh", settings.customPresets ?? []);
          return;
        }
        hydrateSavedSettings(settings);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        // Only the read failed. Settle on local defaults so the controls waiting on hydration
        // still work, but never write back over a store this session could not read.
        hydrateLocalSettings("unreadable");
        toast.error(`Could not load ${kind} presets`);
      });
    return () => {
      cancelled = true;
    };
  }, [hydrateLocalSettings, hydrateSavedSettings, kind]);

  // Explicit model picks own the form over a settings request that was already in flight. A load
  // that fails gives that ownership back, provided no newer form action superseded the pick.
  const claimRecipe = useCallback(() => {
    const previousClaim = formClaim.current;
    const previousCommittedClaim = committedRecipeClaim.current;
    const claim = claimForm(formClaim);
    let settled = false;
    const ownsClaim = () => !settled && formClaim.current === claim;
    return {
      commit: () => {
        if (!ownsClaim()) {
          return;
        }
        settled = true;
        committedRecipeClaim.current = claim;
        deferredSavedSettingsRef.current = null;
        deferredFreshSettingsRef.current = false;
        setHydrationSource((source) =>
          source === "claiming" ? "claimed" : source,
        );
      },
      release: () => {
        if (!ownsClaim()) {
          return;
        }
        settled = true;
        committedRecipeClaim.current = previousCommittedClaim;
        formClaim.current = previousClaim;
        if (previousClaim !== 0 && previousCommittedClaim === previousClaim) {
          deferredSavedSettingsRef.current = null;
          deferredFreshSettingsRef.current = false;
          setHydrationSource((source) =>
            source === "claiming" ? "claimed" : source,
          );
          return;
        }
        if (previousClaim !== 0 || !deferredSavedSettingsRef.current) {
          if (previousClaim === 0 && deferredFreshSettingsRef.current) {
            deferredFreshSettingsRef.current = false;
            setHydrationSource("fresh");
          }
          return;
        }
        const deferred = deferredSavedSettingsRef.current;
        deferredSavedSettingsRef.current = null;
        hydrateSavedSettings(deferred);
      },
    };
  }, [hydrateSavedSettings]);

  // How many actions have taken the form so far. A pick records it and compares later: a different
  // value means something newer owns the form, so the pick's defaults must not land on top of it
  // and its rollback is not the one to restore. Read per pick, so each baselines on itself.
  const formClaimId = useCallback(() => formClaim.current, []);

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

  // One state write at a time. Two in flight and the store keeps whichever the backend saw LAST,
  // so an older snapshot could win and the newest recipe come back changed on the next read.
  // Same chain the chat settings path keeps, for the same reason.
  const queueWrite = useCallback((write: () => Promise<unknown>) => {
    inflightWriteRef.current = inflightWriteRef.current
      .catch(() => undefined)
      .then(write);
  }, []);

  useEffect(() => {
    if (!presetsReady) {
      return;
    }
    const timer = window.setTimeout(() => {
      queueWrite(() =>
        saveMediaGenerationPresetSettings(kind, settings).catch(() => {
          toast.error(`Could not save ${kind} presets`);
        }),
      );
    }, 400);
    return () => window.clearTimeout(timer);
  }, [kind, presetsReady, queueWrite, settings]);

  useEffect(() => {
    if (!presetsReady) {
      return;
    }
    const flush = () => {
      const latest = latestSettingsRef.current;
      if (latest) {
        queueWrite(() =>
          saveMediaGenerationPresetSettings(kind, latest, true).catch(
            () => undefined,
          ),
        );
      }
    };
    window.addEventListener("beforeunload", flush);
    return () => {
      window.removeEventListener("beforeunload", flush);
      flush();
    };
  }, [kind, presetsReady, queueWrite]);

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
      const previousClaim = formClaim.current;
      const claim = claimForm(formClaim);
      try {
        await upsertMediaGenerationPreset(kind, preset);
      } catch (error) {
        // A refused write took nothing over, so it must not keep the claim it made: a pending
        // model pick would go on reading the form as claimed by something newer than itself.
        if (formClaim.current === claim) formClaim.current = previousClaim;
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

  // A delete always leaves Default selected: the preset is gone whoever owns the form, and naming
  // it would point the control at a definition that no longer exists. The form VALUES follow only
  // when this delete still owns the form; an edit made while the request was in flight owns it.
  const restoreDefaultAfterDelete = useCallback(
    (paramsBeforeDelete: Params, ownsForm: boolean) => {
      const formUnchanged =
        ownsForm &&
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
    const previousClaim = formClaim.current;
    const claim = claimForm(formClaim);
    try {
      await deleteMediaGenerationPreset(kind, deletedName);
    } catch (error) {
      // Same as a refused save: a delete that did not happen has not taken the form.
      if (formClaim.current === claim) formClaim.current = previousClaim;
      toast.error(refusalMessage(error, `Could not delete ${kind} preset`));
      return false;
    }
    setCustomPresets((current) =>
      current.filter((preset) => preset.name !== deletedName),
    );
    restoreDefaultAfterDelete(paramsBeforeDelete, formClaim.current === claim);
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
    claimRecipe,
    formClaimId,
    hasUnsavedChanges,
    selectPreset,
    savePreset,
    deletePreset,
  };
}
