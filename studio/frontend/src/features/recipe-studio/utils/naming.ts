// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { NodeConfig } from "../types";

export function nextName(existing: NodeConfig[], prefix: string): string {
  const counts = existing
    .map((item) => item.name)
    .filter((name) => name.startsWith(prefix))
    .map((name) => {
      const suffix = name.slice(prefix.length);
      const num = Number.parseInt(suffix.replace("_", ""), 10);
      return Number.isNaN(num) ? 0 : num;
    });
  const next = counts.length > 0 ? Math.max(...counts) + 1 : 1;
  return `${prefix}_${next}`;
}
