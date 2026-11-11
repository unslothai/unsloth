// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type NpuStatus, getNpuStatus, npuStatusIsProvisional } from "./api";

// The backend's GPU scan takes a second or two on a fresh start; bounded so a host that never answers is not polled.
const PROVISIONAL_RETRIES = 3;
const PROVISIONAL_RETRY_MS = 2000;

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
    const read = (retries: number) =>
      void fetchOnce().then((next) => {
        if (!live || !next) return;
        setStatus(next);
        if (retries > 0 && npuStatusIsProvisional(next)) {
          timer = setTimeout(() => read(retries - 1), PROVISIONAL_RETRY_MS);
        }
      });
    read(PROVISIONAL_RETRIES);
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
