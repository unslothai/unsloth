// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One paste into Cursor or another agent. Asks for the biggest reclaimable
 * files, split by how safe they are to remove, and tells the agent to wait.
 * Shown only when the installer sets `diskFull` on `install-failed`.
 */
export function diskCleanupAgentPrompt(error: string): string {
  return [
    "The Unsloth installer failed because the disk is full. This was the error:",
    "",
    error.trim(),
    "",
    "Show me the biggest files and low-hanging fruit I can clear off this computer to free several GB. Split the list into:",
    "1. Safe to delete, because it rebuilds itself (browser caches, package-manager download caches, updater leftovers).",
    "2. Rebuildable project junk (node_modules, virtualenvs, build folders) I can restore later.",
    "3. Large apps and personal files I should decide on myself.",
    "",
    "Give sizes. Do not delete anything until I tell you which group to clear.",
  ].join("\n");
}
