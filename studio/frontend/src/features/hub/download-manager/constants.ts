// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const TRANSPORT = {
  HTTP: "http",
  XET: "xet",
  AUTO: "auto",
} as const;

// The two transports a download can run on. "auto" is a preference, resolved to one of these
// before a download starts; only these reach a `.transport` marker on disk, which records the
// writer so a resume picks the right strategy.
export const RESOLVED_TRANSPORTS = [TRANSPORT.HTTP, TRANSPORT.XET] as const;
export type ResolvedTransport = (typeof RESOLVED_TRANSPORTS)[number];

export const TRANSPORT_MODES = [
  TRANSPORT.AUTO,
  TRANSPORT.HTTP,
  TRANSPORT.XET,
] as const;
export type TransportMode = (typeof TRANSPORT_MODES)[number];
// Auto by default: the backend picks per machine (RAM, hf_xet build, recent Xet failures), and
// effectiveTransportMode() resolves that to a concrete transport before any download starts.
export const DEFAULT_TRANSPORT_MODE: TransportMode = TRANSPORT.AUTO;

export function isTransportMode(value: unknown): value is TransportMode {
  return (
    typeof value === "string" &&
    (TRANSPORT_MODES as readonly string[]).includes(value)
  );
}

/** The transport a started job is really running on.
 *
 * An accepted start can mean the backend attached this client to a job another
 * one had already begun, which keeps the transport it started on. Trusting the
 * locally requested value there offers Pause for a Xet run, or Cancel for a
 * resumable HTTP one. */
export function transportAfterStart(
  requested: ResolvedTransport,
  reported: unknown,
): ResolvedTransport {
  return isResolvedTransport(reported) ? reported : requested;
}

/** Whether a probe response describes the run a job is currently on.
 *
 * A cancel and restart between the request and its reply makes the answer
 * about a different job, possibly on the other transport. A job with no
 * generation recorded yet has nothing better to go on, so any generation the
 * probe reports is taken (along with the generation itself). */
export function probeDescribesCurrentRun(
  known: unknown,
  reported: unknown,
): boolean {
  return Number.isSafeInteger(known)
    ? reported === known
    : Number.isSafeInteger(reported);
}

export function isResolvedTransport(
  value: unknown,
): value is ResolvedTransport {
  return (
    typeof value === "string" &&
    (RESOLVED_TRANSPORTS as readonly string[]).includes(value)
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
