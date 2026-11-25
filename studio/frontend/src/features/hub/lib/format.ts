// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const HOURS_PER_DAY = 24;
const SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const MAX_DISPLAYABLE_ETA_SECONDS = HOURS_PER_DAY * SECONDS_PER_HOUR;

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

/** Decimal unit (index into B, KB, MB, GB, TB), scaled value and decimals for a byte count. */
export function scaleBytes(bytes: number): { value: number; unit: number; decimals: number } {
  let unit = Math.min(
    Math.max(Math.floor(Math.log(bytes) / Math.log(1000)), 0),
    BYTE_UNITS.length - 1,
  );
  let value = bytes / 1000 ** unit;
  let decimals = unit > 0 && Number(value.toFixed(1)) < 10 ? 1 : 0;
  while (unit < BYTE_UNITS.length - 1 && Number(value.toFixed(decimals)) >= 1000) {
    unit += 1;
    value = bytes / 1000 ** unit;
    decimals = Number(value.toFixed(1)) < 10 ? 1 : 0;
  }
  return { value: Number(value.toFixed(decimals)), unit, decimals };
}

export function formatBytes(bytes: number, unitSeparator = " "): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "N/A";
  if (bytes === 0) return "0 B";
  const { value, unit, decimals } = scaleBytes(bytes);
  return `${value.toFixed(decimals)}${unitSeparator}${BYTE_UNITS[unit]}`;
}

export function formatRate(bytesPerSec: number): string {
  if (!Number.isFinite(bytesPerSec) || bytesPerSec <= 0) return "";
  return `${formatBytes(bytesPerSec)}/s`;
}

// A day or more collapses to "> 24h left".
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const s = Math.round(seconds);
  if (s >= MAX_DISPLAYABLE_ETA_SECONDS) return `> ${HOURS_PER_DAY}h left`;
  if (s < SECONDS_PER_MINUTE) return `${s}s left`;
  if (s < SECONDS_PER_HOUR) {
    const m = Math.floor(s / SECONDS_PER_MINUTE);
    const rem = s % SECONDS_PER_MINUTE;
    return rem ? `${m}m ${rem}s left` : `${m}m left`;
  }
  const h = Math.floor(s / SECONDS_PER_HOUR);
  const rem = Math.floor((s % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  return rem ? `${h}h ${rem}m left` : `${h}h left`;
}

export function ownerOf(id: string): string {
  return id.includes("/") ? id.split("/")[0] : "";
}

export function repoOf(id: string): string {
  return id.includes("/") ? id.split("/").slice(1).join("/") : id;
}

export function formatShortDate(iso?: string): string {
  if (!iso) return "N/A";
  const time = new Date(iso).getTime();
  if (Number.isNaN(time)) return "N/A";
  return new Intl.DateTimeFormat("en", {
    month: "short",
    year: "numeric",
  }).format(time);
}

export function formatRelativeShort(iso?: string): string {
  if (!iso) return "N/A";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "N/A";
  const diffMs = Math.max(0, Date.now() - then);
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

export function formatRelativeLong(iso?: string): string {
  if (!iso) return "N/A";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "N/A";
  const diffMs = Math.max(0, Date.now() - then);
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "just now";
  const unit = (value: number, name: string) =>
    `${value} ${name}${value === 1 ? "" : "s"} ago`;
  if (minutes < 60) return unit(minutes, "minute");
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return unit(hours, "hour");
  const days = Math.floor(hours / 24);
  if (days < 30) return unit(days, "day");
  if (days < 365) return unit(Math.floor(days / 30), "month");
  return unit(Math.floor(days / 365), "year");
}
