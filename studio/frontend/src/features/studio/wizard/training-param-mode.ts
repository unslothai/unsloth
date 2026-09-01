// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_PARAM_MODE_STORAGE_KEY,
} from "@/features/training";
import { useCallback, useState } from "react";

export type ParamMode = "simple" | "advanced";

function isParamMode(value: string | null): value is ParamMode {
  return value === "simple" || value === "advanced";
}

function readParamMode(): ParamMode {
  if (typeof window === "undefined") {
    return "simple";
  }
  try {
    const stored = window.localStorage.getItem(TRAINING_PARAM_MODE_STORAGE_KEY);
    if (isParamMode(stored)) {
      return stored;
    }
    const legacy = window.localStorage.getItem(
      LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
    );
    if (isParamMode(legacy)) {
      window.localStorage.setItem(TRAINING_PARAM_MODE_STORAGE_KEY, legacy);
      window.localStorage.removeItem(LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY);
      return legacy;
    }
  } catch {
    return "simple";
  }
  return "simple";
}

export function useParamMode(): [ParamMode, (next: ParamMode) => void] {
  const [mode, setMode] = useState<ParamMode>(readParamMode);
  const update = useCallback((next: ParamMode) => {
    setMode(next);
    try {
      window.localStorage.setItem(TRAINING_PARAM_MODE_STORAGE_KEY, next);
      window.localStorage.removeItem(LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY);
    } catch {
      return;
    }
  }, []);
  return [mode, update];
}
