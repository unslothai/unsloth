import { platformRequest } from "./client";
import type { PlatformSystemHealth } from "./types";

export function getSystemPing(signal?: AbortSignal): Promise<string> {
  return platformRequest<string>("/system/ping", {
    responseType: "text",
    signal,
  });
}

export function getSystemVersion(signal?: AbortSignal): Promise<string> {
  return platformRequest<string>("/system/version", { signal });
}

export function getSystemHealth(
  signal?: AbortSignal,
): Promise<PlatformSystemHealth> {
  return platformRequest<PlatformSystemHealth>("/system/healthz", {
    responseType: "json",
    signal,
  });
}
