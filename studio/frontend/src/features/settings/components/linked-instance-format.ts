// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LinkedInstanceInfo } from "../api/linked-instances";

export type ConnectionKind = "tunnel" | "loopback" | "lan" | "public";

export function connectionKind(baseUrl: string): ConnectionKind {
  let host: string;
  try {
    host = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return "public";
  }
  if (host.endsWith(".trycloudflare.com")) return "tunnel";
  if (host === "localhost" || host === "::1" || host.startsWith("127.")) {
    return "loopback";
  }
  if (
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host) ||
    host.endsWith(".local")
  ) {
    return "lan";
  }
  return "public";
}

export function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).host;
  } catch {
    return baseUrl;
  }
}

export function formatGb(value: number | null | undefined): string | null {
  if (value == null) return null;
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} GB`;
}

type PoolGpu = {
  name: string;
  vram_total_gb?: number | null;
  vram_used_gb?: number | null;
};

const sum = (values: (number | null | undefined)[]) =>
  values.every((v) => v != null)
    ? values.reduce<number>((a, v) => a + (v as number), 0)
    : null;

/** Every GPU as one pool: "2× NVIDIA L4" or "RX 6500 XT + RX 5700 XT", with summed VRAM. */
export function gpuPool(gpus: PoolGpu[]) {
  const counts = new Map<string, number>();
  for (const gpu of gpus) counts.set(gpu.name, (counts.get(gpu.name) ?? 0) + 1);
  const label = [...counts]
    .map(([name, n]) => (n > 1 ? `${n}× ${name}` : name))
    .join(" + ");
  return {
    label,
    total: sum(gpus.map((g) => g.vram_total_gb)),
    used: sum(gpus.map((g) => g.vram_used_gb)),
  };
}

export function formatUptime(
  seconds: number | null | undefined,
): string | null {
  if (seconds == null) return null;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${Math.max(minutes, 1)} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ${minutes % 60} min`;
  return `${Math.floor(hours / 24)} d ${hours % 24} h`;
}

/** "Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-..." reads as "Linux (WSL2)". */
export function platformLabel(platform: string | null): string | null {
  if (!platform) return null;
  const lower = platform.toLowerCase();
  if (lower.includes("wsl")) return "Linux (WSL2)";
  if (lower.startsWith("linux")) return "Linux";
  if (lower.startsWith("windows")) return "Windows";
  if (lower.startsWith("darwin") || lower.startsWith("macos")) return "macOS";
  return platform.split("-")[0] || platform;
}

/** "CUDA 12.8", "ROCm 7.2", "Apple MLX" or null when nothing was reported. */
export function acceleratorLabel(info: LinkedInstanceInfo): string | null {
  if (info.cuda) return `CUDA ${info.cuda}`;
  if (info.rocm) return `ROCm ${info.rocm.split(".").slice(0, 2).join(".")}`;
  const backend = info.device_backend?.toLowerCase();
  if (backend === "mlx" || backend === "mps") return "Apple MLX";
  if (backend === "xpu") return "Intel XPU";
  return null;
}

export function installLabel(source: string | null): string | null {
  return source ? source.replaceAll("_", " ") : null;
}
