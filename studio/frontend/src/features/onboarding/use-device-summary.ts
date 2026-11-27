// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { authFetch } from "@/features/auth";
import {
  refreshHardwareInfo,
  useHardwareInfo,
} from "@/hooks/use-hardware-info";
import { useEffect, useState } from "react";
import { videoAvailable } from "./lib/pages";

export type DeviceSummary =
  | { status: "checking" }
  | { status: "error" }
  | {
      status: "ready";
      gpuName: string | null;
      gpuCount: number;
      vramGb: number | null;
      unifiedMemory: boolean;
      appleSilicon: boolean;
      ramGb: number | null;
      train: boolean;
      image: boolean;
      video: boolean;
    };

async function readRamGb(): Promise<number | null> {
  try {
    const res = await authFetch("/api/system");
    if (!res.ok) return null;
    const data = (await res.json()) as { memory?: Record<string, unknown> };
    const total = data.memory?.total_gb;
    return typeof total === "number" ? total : null;
  } catch {
    return null;
  }
}

const DETECT_POLL_MS = 1500;
const DETECT_TIMEOUT_MS = 60_000;

// Reactive, not a snapshot: on a first run the verdict is still a guess when the dialog mounts.
export function useDeviceSummary(): DeviceSummary {
  const hardware = useHardwareInfo();
  const unknown = usePlatformStore((s) => s.capabilitiesUnknown());
  const appleSilicon = usePlatformStore((s) => s.appleSilicon);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const trainUnavailable = usePlatformStore((s) => s.isTrainUnavailable());
  const [ramGb, setRamGb] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const deadline = Date.now() + DETECT_TIMEOUT_MS;
    const probe = async () => {
      await fetchDeviceType({ detectHardware: true }).catch(() => undefined);
      if (cancelled) return;
      if (!usePlatformStore.getState().capabilitiesUnknown()) {
        // GPU fields cached before detection settled would be stale.
        void refreshHardwareInfo();
        return;
      }
      if (Date.now() > deadline) {
        setFailed(true);
        return;
      }
      timer = window.setTimeout(probe, DETECT_POLL_MS);
    };
    void probe();
    void readRamGb().then((gb) => {
      if (!cancelled) setRamGb(gb);
    });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  if (failed) return { status: "error" };
  if (unknown || !hardware.loaded) return { status: "checking" };
  const primary = hardware.gpus[0];
  return {
    status: "ready",
    gpuName: hardware.gpuName ?? primary?.name ?? null,
    gpuCount: hardware.gpus.length,
    vramGb: hardware.vramTotalGb ?? primary?.vramTotalGb ?? null,
    unifiedMemory: appleSilicon || primary?.unifiedMemory === true,
    appleSilicon,
    ramGb,
    train: !trainUnavailable,
    image: !chatOnly,
    video: videoAvailable(chatOnly, hardware.videoSupported),
  };
}
