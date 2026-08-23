// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Listener = () => void;

export type HiddenModelMatchersSnapshot = {
  readonly revision: number;
  readonly ready: boolean;
  readonly needles: readonly string[];
  readonly exactIds: readonly string[];
  readonly exactPaths: readonly string[];
};

type HiddenModelMatchersReplacement = Omit<
  HiddenModelMatchersSnapshot,
  "revision" | "ready"
>;

let snapshot: HiddenModelMatchersSnapshot = {
  revision: 0,
  ready: false,
  needles: [],
  exactIds: [],
  exactPaths: [],
};
const listeners = new Set<Listener>();

function stringsEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

export function getHiddenModelMatchersSnapshot(): HiddenModelMatchersSnapshot {
  return snapshot;
}

export function subscribeHiddenModelMatchers(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function replaceHiddenModelMatchers(
  next: HiddenModelMatchersReplacement,
): void {
  if (
    snapshot.ready &&
    stringsEqual(snapshot.needles, next.needles) &&
    stringsEqual(snapshot.exactIds, next.exactIds) &&
    stringsEqual(snapshot.exactPaths, next.exactPaths)
  ) {
    return;
  }
  snapshot = {
    ...next,
    ready: true,
    revision: snapshot.revision + 1,
  };
  for (const listener of [...listeners]) {
    listener();
  }
}
