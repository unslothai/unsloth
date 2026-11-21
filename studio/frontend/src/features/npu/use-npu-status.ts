// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type NpuStatus, getNpuStatus } from "./api";

let cached: NpuStatus | null = null;
let pending: Promise<NpuStatus | null> | null = null;
// Every mounted consumer, so enabling the NPU in one view updates the others (the live monitor).
const listeners = new Set<(status: NpuStatus) => void>();

function publish(status: NpuStatus): void {
  cached = status;
  for (const listener of listeners) listener(status);
}

function fetchOnce(): Promise<NpuStatus | null> {
  pending ??= getNpuStatus()
    .then((status) => {
      publish(status);
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
    listeners.add(setStatus);
    return () => {
      listeners.delete(setStatus);
    };
  }, []);
  useEffect(() => {
    if (active) void fetchOnce();
  }, [active]);
  return [status, publish];
}
