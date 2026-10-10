// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FindTargetResult = { count: number | null; active: number };

export type FindTarget = {
  id: string;
  available: () => boolean;
  contains: (node: Node) => boolean;
  search: (query: string) => void;
  step: (delta: -1 | 1) => void;
  result: () => FindTargetResult;
};

export const EMPTY_FIND_RESULT: FindTargetResult = { count: 0, active: -1 };

const targets = new Map<string, FindTarget>();
const listeners = new Set<() => void>();
let version = 0;

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

export function availableFindTargets(): FindTarget[] {
  return [...targets.values()].filter((target) => target.available());
}

export function findTargetHolding(node: Node | null): FindTarget | undefined {
  if (!node) return undefined;
  return availableFindTargets().find((target) => target.contains(node));
}

const requests = new Set<(targetId: string | null) => void>();

export function requestFind(targetId: string | null): void {
  for (const listener of requests) listener(targetId);
}

export function onFindRequest(listener: (targetId: string | null) => void): () => void {
  requests.add(listener);
  return () => {
    requests.delete(listener);
  };
}
