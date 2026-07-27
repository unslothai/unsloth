// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const TRANSPORT = {
  HTTP: "http",
  XET: "xet",
} as const;

export const TRANSPORT_MODES = [TRANSPORT.HTTP, TRANSPORT.XET] as const;
export type TransportMode = (typeof TRANSPORT_MODES)[number];
// Xet by default; effectiveTransportMode() downgrades to HTTP if hf_xet is missing.
export const DEFAULT_TRANSPORT_MODE: TransportMode = TRANSPORT.XET;

export function isTransportMode(value: unknown): value is TransportMode {
  return (
    typeof value === "string" &&
    (TRANSPORT_MODES as readonly string[]).includes(value)
  );
}

export const DOWNLOAD_KIND = {
  MODEL: "model",
  DATASET: "dataset",
} as const;

export const DOWNLOAD_KINDS = [
  DOWNLOAD_KIND.MODEL,
  DOWNLOAD_KIND.DATASET,
] as const;
export type DownloadKind = (typeof DOWNLOAD_KINDS)[number];

export function isDownloadKind(value: unknown): value is DownloadKind {
  return (
    typeof value === "string" &&
    (DOWNLOAD_KINDS as readonly string[]).includes(value)
  );
}
