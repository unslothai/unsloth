// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FlowCapabilitySet } from "./api";

export interface CapabilityItem {
  key: string;
  label: string;
}

/** One readable line per thing the owner is asked to approve. */
export function pendingItems(
  pending: FlowCapabilitySet | null | undefined,
): CapabilityItem[] {
  if (!pending) return [];
  const items: CapabilityItem[] = [];
  if (pending.python.length > 0)
    items.push({
      key: "python",
      label:
        pending.python.length === 1
          ? "Runs Python code (1 block)"
          : `Runs Python code (${pending.python.length} blocks)`,
    });
  for (const host of pending.hosts)
    items.push({ key: `host:${host}`, label: `Calls ${host}` });
  if (pending.mcp)
    items.push({ key: "mcp", label: "Uses your MCP servers' tools" });
  for (const provider of pending.providers)
    items.push({
      key: `provider:${provider}`,
      label: `Uses your ${provider} connection`,
    });
  if (pending.triggers)
    items.push({
      key: "triggers",
      label: "Starts on its own (schedule or webhook) while Studio is open",
    });
  return items;
}

export function webhookUrl(origin: string, path: string): string {
  return `${origin.replace(/\/$/, "")}${path}`;
}
