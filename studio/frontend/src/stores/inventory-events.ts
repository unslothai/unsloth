// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

type Listener = () => void;

let _version = 0;
const listeners = new Set<Listener>();

export function getInventoryVersion(): number {
  return _version;
}

export function bumpInventoryVersion(): void {
  _version += 1;
  for (const fn of [...listeners]) fn();
}

export function subscribeInventory(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useInventoryVersion(): number {
  return useSyncExternalStore(
    subscribeInventory,
    getInventoryVersion,
    getInventoryVersion,
  );
}
