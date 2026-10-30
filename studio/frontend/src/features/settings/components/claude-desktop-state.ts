// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ClaudeDesktopStatus } from "../api/claude-desktop";

export type ClaudeDesktopView =
  | { kind: "loading" }
  | { kind: "remote" }
  | { kind: "unsupported" }
  | { kind: "not_installed"; downloadUrl: string }
  | { kind: "held"; pid: number | null }
  | { kind: "connected"; model: string | null; pendingRestart: boolean }
  | { kind: "idle"; model: string | null };

// The fields every Desktop apps card's status has (Claude Desktop, Codex Desktop).
export type DesktopAppStatus = Pick<
  ClaudeDesktopStatus,
  | "supported"
  | "installed"
  | "download_url"
  | "connected"
  | "held_by"
  | "held_by_pid"
  | "model"
  | "local"
>;

/** What a Desktop apps card shows, in priority order: what blocks Connect comes first. */
export function claudeDesktopView(
  status: DesktopAppStatus | null,
  pendingRestart = false,
): ClaudeDesktopView {
  if (status === null) {
    return { kind: "loading" };
  }
  if (status.connected) {
    return { kind: "connected", model: status.model, pendingRestart };
  }
  if (status.held_by) {
    return { kind: "held", pid: status.held_by_pid };
  }
  if (!status.supported) {
    return { kind: "unsupported" };
  }
  if (!status.installed) {
    return { kind: "not_installed", downloadUrl: status.download_url };
  }
  if (!status.local) {
    return { kind: "remote" };
  }
  return { kind: "idle", model: status.model };
}
