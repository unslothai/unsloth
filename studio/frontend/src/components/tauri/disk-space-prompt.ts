// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Phrases install.rs keeps when a download dies because the disk is full. */
export function isDiskFullInstallError(error: string | null): boolean {
  if (!error) return false;
  const text = error.toLowerCase();
  return (
    text.includes("no space left on device")
    || text.includes("not enough space on the disk")
    || text.includes("os error 28")
    || text.includes("os error 112")
    || text.includes("enospc")
    || text.includes("disk quota exceeded")
  );
}

/**
 * One paste into Cursor or another agent. Asks for the biggest reclaimable
 * files, split by how safe they are to remove, and tells the agent to wait.
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
