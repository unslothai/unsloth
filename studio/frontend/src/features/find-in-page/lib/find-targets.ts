// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Something besides the chat that the find bar can search, such as the browser panel's page: it
// does its own matching (a page lives in another document) and the bar shows its count, walks its
// matches and offers a button to switch to it while it has something to search.

/** A target's answer for the current query. `count` is null when it can only step, not count. */
export type FindTargetResult = { count: number | null; active: number };

export type FindTarget = {
  id: string;
  /** Whether it has anything to search right now. */
  available: () => boolean;
  /** Whether `node` is inside it, so the chord opens the bar searching it from there. */
  contains: (node: Node) => boolean;
  /** Searches for `query`; an empty one clears its highlights. Results arrive through `notify`. */
  search: (query: string) => void;
  step: (delta: -1 | 1) => void;
  /** The latest result, the same object until it changes. */
  result: () => FindTargetResult;
};

export const EMPTY_FIND_RESULT: FindTargetResult = { count: 0, active: -1 };

const targets = new Map<string, FindTarget>();
const listeners = new Set<() => void>();
let version = 0;

/** Tells the bar a target's availability or result changed. */
export function notifyFindTargets(): void {
  version += 1;
  for (const listener of listeners) listener();
}

export function registerFindTarget(target: FindTarget): () => void {
  targets.set(target.id, target);
  notifyFindTargets();
  return () => {
    if (targets.get(target.id) === target) targets.delete(target.id);
    notifyFindTargets();
  };
}

export function subscribeFindTargets(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function findTargetsVersion(): number {
  return version;
}

export function findTarget(id: string): FindTarget | undefined {
  return targets.get(id);
}

/** The targets with something to search. */
export function availableFindTargets(): FindTarget[] {
  return [...targets.values()].filter((target) => target.available());
}

/** The available target holding `node`, if any. */
export function findTargetHolding(node: Node | null): FindTarget | undefined {
  if (!node) return undefined;
  return availableFindTargets().find((target) => target.contains(node));
}

const requests = new Set<(targetId: string | null) => void>();

/** Opens the find bar searching `targetId` (null: the chat), as the chord would. */
export function requestFind(targetId: string | null): void {
  for (const listener of requests) listener(targetId);
}

export function onFindRequest(listener: (targetId: string | null) => void): () => void {
  requests.add(listener);
  return () => {
    requests.delete(listener);
  };
}
