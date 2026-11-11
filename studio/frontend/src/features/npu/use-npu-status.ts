// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type NpuStatus, getNpuStatus, npuProvisionalRetryMs } from "./api";

let cached: NpuStatus | null = null;
let pending: Promise<NpuStatus | null> | null = null;

function fetchOnce(): Promise<NpuStatus | null> {
  pending ??= getNpuStatus()
    .then((status) => {
      cached = status;
      return status;
    })
    .catch(() => null)
    .finally(() => {
      pending = null;
    });
  return pending;
}

export function useNpuStatus(
  active = true,
): [NpuStatus | null, (next: NpuStatus) => void] {
  const [status, setStatus] = useState<NpuStatus | null>(cached);
  useEffect(() => {
    if (!active) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const read = (attempt: number) =>
      void fetchOnce().then((next) => {
        if (!live || !next) return;
        setStatus(next);
        const wait = npuProvisionalRetryMs(next, attempt);
        if (wait != null) timer = setTimeout(() => read(attempt + 1), wait);
      });
    read(0);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [active]);
  const update = (next: NpuStatus) => {
    cached = next;
    setStatus(next);
  };
  return [status, update];
}
