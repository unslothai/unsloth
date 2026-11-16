// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { getNpuCompat } from "./api";
import type { NpuCompat } from "./compat";

let cached: NpuCompat | null = null;
let pending: Promise<NpuCompat | null> | null = null;

function fetchOnce(): Promise<NpuCompat | null> {
  pending ??= getNpuCompat()
    .then((compat) => {
      // An unknown half (NPU not enabled yet, offline) is asked again next time.
      if (compat.fastflowlm_known && compat.converter_known) cached = compat;
      return compat;
    })
    .catch(() => null)
    .finally(() => {
      pending = null;
    });
  return pending;
}

/** What the installed FastFlowLM runs and the Q4NX converter converts; null until read. */
export function useNpuCompat(active = true): NpuCompat | null {
  const [compat, setCompat] = useState<NpuCompat | null>(cached);
  useEffect(() => {
    if (!active || cached) return;
    let live = true;
    void fetchOnce().then((next) => {
      if (live && next) setCompat(next);
    });
    return () => {
      live = false;
    };
  }, [active]);
  return compat;
}
