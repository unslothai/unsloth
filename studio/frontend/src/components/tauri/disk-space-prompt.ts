// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Same set as `is_disk_full_text` in studio/src-tauri/src/install.rs.
 * uv prints Rust `std::io::Error` Display, `{strerror} (os error {code})`:
 * https://github.com/rust-lang/rust/blob/master/library/core/src/io/error.rs
 * Windows 112 and 39 are the two codes Rust maps to StorageFull:
 * https://github.com/rust-lang/rust/blob/master/library/std/src/sys/io/error/windows.rs
 */
export function isDiskFullInstallError(error: string | null): boolean {
  if (!error) return false;
  const text = error.toLowerCase();
  return (
    text.includes("no space left on device")
    || text.includes("not enough space on the disk")
    || text.includes("(os error 28)")
    || text.includes("(os error 112)")
    || text.includes("(os error 39)")
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
