// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const TRANSPORT = {
  HTTP: "http",
  XET: "xet",
  AUTO: "auto",
} as const;

// Only resolved transports are written to the on-disk `.transport` marker.
export const RESOLVED_TRANSPORTS = [TRANSPORT.HTTP, TRANSPORT.XET] as const;
export type ResolvedTransport = (typeof RESOLVED_TRANSPORTS)[number];

export const TRANSPORT_MODES = [
  TRANSPORT.AUTO,
  TRANSPORT.HTTP,
  TRANSPORT.XET,
] as const;
export type TransportMode = (typeof TRANSPORT_MODES)[number];
export const DEFAULT_TRANSPORT_MODE: TransportMode = TRANSPORT.AUTO;

export function pickTransportMode(
  stored: unknown,
  installed: unknown,
): TransportMode {
  if (isTransportMode(stored)) {
    return stored;
  }
  if (isTransportMode(installed)) {
    return installed;
  }
  return DEFAULT_TRANSPORT_MODE;
}

export function isTransportMode(value: unknown): value is TransportMode {
  return (
    typeof value === "string" &&
    (TRANSPORT_MODES as readonly string[]).includes(value)
  );
}

/** May attach to another client's job, which keeps its own transport. */
export function transportAfterStart(
  requested: ResolvedTransport,
  reported: unknown,
): ResolvedTransport {
  return isResolvedTransport(reported) ? reported : requested;
}

/** A cancel/restart between request and reply makes the answer about a different run. */
export function probeDescribesCurrentRun(
  known: unknown,
  reported: unknown,
): boolean {
  return Number.isSafeInteger(known)
    ? reported === known
    : Number.isSafeInteger(reported);
}

export type AdoptedTransports = {
  transport?: ResolvedTransport;
  cancelTransport?: ResolvedTransport;
};

/** `cancelTransport: null` = no marker; undefined = source cannot report one. */
export type ReportedTransports = {
  transport?: ResolvedTransport;
  cancelTransport?: ResolvedTransport | null;
};

/** Merge both fields so a probe carrying only one (e.g. /download-status) cannot erase the other. */
export function adoptedTransports(
  reported: ReportedTransports,
  existing: AdoptedTransports | undefined,
): AdoptedTransports {
  return {
    transport: reported.transport ?? existing?.transport,
    cancelTransport:
      reported.cancelTransport === undefined
        ? existing?.cancelTransport
        : (reported.cancelTransport ?? undefined),
  };
}

export function isResolvedTransport(
  value: unknown,
): value is ResolvedTransport {
  return (
    typeof value === "string" &&
    (RESOLVED_TRANSPORTS as readonly string[]).includes(value)
  );
}

export type MismatchStartAction = ResolvedTransport | "conflict";

/** Xet cannot byte-resume; only explicit Xet vs a resumable HTTP partial asks the user. */
export function mismatchStartAction(
  preferred: TransportMode,
  resolved: ResolvedTransport,
  lastTransport: ResolvedTransport,
  resumable = true,
): MismatchStartAction {
  if (lastTransport === resolved) return resolved;
  if (lastTransport === TRANSPORT.HTTP) {
    if (preferred === TRANSPORT.XET && resolved === TRANSPORT.XET) {
      return resumable ? "conflict" : resolved;
    }
    return TRANSPORT.HTTP;
  }
  return TRANSPORT.HTTP;
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
