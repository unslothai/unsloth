// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    Math.max(Math.floor(Math.log(bytes) / Math.log(1024)), 0),
    units.length - 1,
  );
  const value = bytes / 1024 ** i;
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatRate(bytesPerSec: number): string {
  if (!Number.isFinite(bytesPerSec) || bytesPerSec <= 0) return "";
  return `${formatBytes(bytesPerSec)}/s`;
}

export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s left`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return rem ? `${m}m ${rem}s left` : `${m}m left`;
  }
  if (s < 86400) {
    const h = Math.floor(s / 3600);
    const rem = Math.floor((s % 3600) / 60);
    return rem ? `${h}h ${rem}m left` : `${h}h left`;
  }
  const d = Math.floor(s / 86400);
  const rem = Math.floor((s % 86400) / 3600);
  return rem ? `${d}d ${rem}h left` : `${d}d left`;
}

export function extractParamLabel(id: string): string | null {
  const name = id.split("/").pop() ?? id;
  const match = name.match(/(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/);
  return match ? `${match[1]}B` : null;
}

export function ownerOf(id: string): string {
  return id.includes("/") ? id.split("/")[0] : "";
}

/** Last path segment of a model id or path (e.g. "owner/name" -> "name"). */
export function modelShortName(id: string): string {
  return id.split("/").pop() || id;
}

export function datasetShortName(id: string): string {
  return id.split("/").pop() || id;
}

export function repoOf(id: string): string {
  return id.includes("/") ? id.split("/").slice(1).join("/") : id;
}

export function formatRelativeShort(iso?: string): string {
  if (!iso) return "N/A";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "N/A";
  const diffMs = Date.now() - then;
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  if (days < 365) {
    const months = Math.floor(days / 30);
    return `${months}mo ago`;
  }
  const years = Math.floor(days / 365);
  return `${years}y ago`;
}
