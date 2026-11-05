// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ParamSpec } from "../types";

export function paramVisible(
  param: ParamSpec,
  params: Record<string, unknown>,
  all: ParamSpec[],
): boolean {
  if (!param.show_when) return true;
  return Object.entries(param.show_when).every(([id, values]) => {
    const value = params[id] ?? all.find((p) => p.id === id)?.default;
    return values.includes(String(value ?? ""));
  });
}
