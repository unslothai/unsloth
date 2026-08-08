// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ScopedRefreshGate {
  scopeKey: string;
  version: number;
  inFlight: {
    scopeKey: string;
    version: number;
    promise: Promise<unknown>;
  } | null;
}

export function createScopedRefreshGate(scopeKey: string): ScopedRefreshGate {
  return { scopeKey, version: 0, inFlight: null };
}

export function setScopedRefreshScope(
  gate: ScopedRefreshGate,
  scopeKey: string,
): void {
  if (gate.scopeKey === scopeKey) {
    return;
  }
  gate.scopeKey = scopeKey;
  gate.version += 1;
}

export function runScopedRefresh<T>(
  gate: ScopedRefreshGate,
  scopeKey: string,
  run: (isCurrent: () => boolean) => Promise<T>,
): Promise<T | undefined> {
  const version = gate.version;
  if (gate.scopeKey !== scopeKey) {
    return Promise.resolve(undefined);
  }
  if (
    gate.inFlight?.scopeKey === scopeKey &&
    gate.inFlight.version === version
  ) {
    return gate.inFlight.promise as Promise<T>;
  }

  const isCurrent = () =>
    gate.scopeKey === scopeKey && gate.version === version;
  const promise = run(isCurrent).finally(() => {
    if (gate.inFlight?.promise === promise) {
      gate.inFlight = null;
    }
  });
  gate.inFlight = { scopeKey, version, promise };
  return promise;
}
