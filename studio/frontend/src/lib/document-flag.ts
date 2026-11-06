// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLayoutEffect } from "react";

const holders = new Map<string, number>();

export function acquireDocumentFlag(name: string): () => void {
  const root = document.documentElement;
  holders.set(name, (holders.get(name) ?? 0) + 1);
  root.setAttribute(name, "");
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const left = (holders.get(name) ?? 1) - 1;
    if (left > 0) {
      holders.set(name, left);
      return;
    }
    holders.delete(name);
    root.removeAttribute(name);
  };
}

export function useDocumentFlag(name: string, active: boolean): void {
  useLayoutEffect(() => {
    if (!active) return;
    return acquireDocumentFlag(name);
  }, [name, active]);
}
