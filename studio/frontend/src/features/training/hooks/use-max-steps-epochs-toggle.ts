// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

const PREV_MAX_STEPS_KEY = "unsloth_prev_max_steps";
const PREV_SAVE_STEPS_KEY = "unsloth_prev_save_steps";
const DEFAULT_MAX_STEPS = 60;
const DEFAULT_EPOCHS = 3;

function readStoredNumber(key: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  try {
    const value = window.localStorage.getItem(key);
    if (value === null) return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function writeStoredNumber(key: string, value: number): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    // Best effort only; ignore storage errors in restricted environments.
  }
}

function normalizePrevMaxSteps(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : DEFAULT_MAX_STEPS;
}

function normalizePrevSaveSteps(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

type UseMaxStepsEpochsToggleParams = {
  maxSteps: number;
  epochs: number;
  saveSteps: number;
  setMaxSteps: (value: number) => void;
  setEpochs: (value: number) => void;
  setSaveSteps: (value: number) => void;
  defaultEpochs?: number;
};

type UseMaxStepsEpochsToggleResult = {
  useEpochs: boolean;
  toggleUseEpochs: () => void;
};

export function useMaxStepsEpochsToggle({
  maxSteps,
  epochs,
  saveSteps,
  setMaxSteps,
  setEpochs,
  setSaveSteps,
  defaultEpochs = DEFAULT_EPOCHS,
}: UseMaxStepsEpochsToggleParams): UseMaxStepsEpochsToggleResult {
  const useEpochs = maxSteps === 0;
  const [prevMaxSteps, setPrevMaxSteps] = useState(() =>
    normalizePrevMaxSteps(readStoredNumber(PREV_MAX_STEPS_KEY, DEFAULT_MAX_STEPS)),
  );
  const [prevSaveSteps, setPrevSaveSteps] = useState(() => {
    if (maxSteps === 0 && saveSteps > 0) {
      return normalizePrevSaveSteps(saveSteps);
    }
    return normalizePrevSaveSteps(readStoredNumber(PREV_SAVE_STEPS_KEY, 0));
  });

  useEffect(() => {
    if (maxSteps > 0) {
      const normalized = normalizePrevMaxSteps(maxSteps);
      setPrevMaxSteps(normalized);
      writeStoredNumber(PREV_MAX_STEPS_KEY, normalized);
    }
  }, [maxSteps]);

  useEffect(() => {
    if (!useEpochs) {
      const normalized = normalizePrevSaveSteps(saveSteps);
      setPrevSaveSteps(normalized);
      writeStoredNumber(PREV_SAVE_STEPS_KEY, normalized);
    }
  }, [saveSteps, useEpochs]);

  const toggleUseEpochs = useCallback(() => {
    if (useEpochs) {
      setMaxSteps(normalizePrevMaxSteps(prevMaxSteps));
      setSaveSteps(normalizePrevSaveSteps(prevSaveSteps));
      return;
    }

    setMaxSteps(0);
    setEpochs(epochs || defaultEpochs);
  }, [
    defaultEpochs,
    epochs,
    prevMaxSteps,
    prevSaveSteps,
    setEpochs,
    setMaxSteps,
    setSaveSteps,
    useEpochs,
  ]);

  return { useEpochs, toggleUseEpochs };
}
