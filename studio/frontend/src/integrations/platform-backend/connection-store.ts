import { create } from "zustand";

import { isPlatformApiError } from "./errors";
import { getSystemHealth, getSystemPing, getSystemVersion } from "./system-api";
import type { PlatformSystemHealth } from "./types";

export type PlatformConnectionStatus =
  | "idle"
  | "checking"
  | "connected"
  | "degraded"
  | "disconnected"
  | "unauthorized";

export interface PlatformConnectionError {
  kind: "permission" | "timeout" | "network" | "api";
  message: string;
}

interface PlatformConnectionState {
  status: PlatformConnectionStatus;
  ping: string | null;
  version: string | null;
  health: PlatformSystemHealth | null;
  error: PlatformConnectionError | null;
  lastCheckedAt: string | null;
  checkConnection: (signal?: AbortSignal) => Promise<void>;
  reset: () => void;
}

let checkGeneration = 0;

const initialState = {
  status: "idle" as const,
  ping: null,
  version: null,
  health: null,
  error: null,
  lastCheckedAt: null,
};

function normalizeConnectionError(error: unknown): {
  status: PlatformConnectionStatus;
  error: PlatformConnectionError;
} {
  if (isPlatformApiError(error)) {
    if (error.isPermissionError) {
      return {
        status: "unauthorized",
        error: { kind: "permission", message: error.message },
      };
    }
    if (error.isTimeout) {
      return {
        status: "disconnected",
        error: { kind: "timeout", message: error.message },
      };
    }
    return {
      status: "disconnected",
      error: {
        kind: error.code === "NETWORK_ERROR" ? "network" : "api",
        message: error.message,
      },
    };
  }
  return {
    status: "disconnected",
    error: {
      kind: "network",
      message: "Rag Platform bağlantısı doğrulanamadı.",
    },
  };
}

export const usePlatformConnectionStore = create<PlatformConnectionState>(
  (set) => ({
    ...initialState,
    checkConnection: async (signal) => {
      const generation = ++checkGeneration;
      set({ status: "checking", error: null });

      const [ping, version, health] = await Promise.allSettled([
        getSystemPing(signal),
        getSystemVersion(signal),
        getSystemHealth(signal),
      ]);

      if (generation !== checkGeneration) return;
      if (signal?.aborted) {
        set({ status: "idle", error: null });
        return;
      }

      const requiredFailure =
        ping.status === "rejected"
          ? ping
          : version.status === "rejected"
            ? version
            : null;
      if (requiredFailure) {
        const normalized = normalizeConnectionError(requiredFailure.reason);
        set({
          ...normalized,
          ping: ping.status === "fulfilled" ? ping.value : null,
          version: version.status === "fulfilled" ? version.value : null,
          health: health.status === "fulfilled" ? health.value : null,
          lastCheckedAt: new Date().toISOString(),
        });
        return;
      }

      if (ping.status !== "fulfilled" || version.status !== "fulfilled") return;

      const healthy =
        health.status === "fulfilled" &&
        health.value.status.toLowerCase() === "ok";
      set({
        status: healthy ? "connected" : "degraded",
        ping: ping.value,
        version: version.value,
        health: health.status === "fulfilled" ? health.value : null,
        error:
          health.status === "rejected"
            ? normalizeConnectionError(health.reason).error
            : null,
        lastCheckedAt: new Date().toISOString(),
      });
    },
    reset: () => {
      checkGeneration += 1;
      set(initialState);
    },
  }),
);
