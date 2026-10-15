// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Leading boundary skips version digits and MoE active params, so "Qwen3-30B-A3B" reads as 30B.
const PARAM_COUNT_RE = /(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/;

function matchParamCount(id: string): RegExpMatchArray | null {
  const name = id.split("/").pop() ?? id;
  return name.match(PARAM_COUNT_RE);
}

export function extractParamLabel(id: string): string | null {
  const m = matchParamCount(id);
  return m ? `${m[1]}B` : null;
}

export function parseParamCountB(id: string): number | null {
  const m = matchParamCount(id);
  if (!m) return null;
  const v = Number.parseFloat(m[1]);
  return Number.isFinite(v) ? v : null;
}
