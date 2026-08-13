// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

import { sandboxRoutePrefix } from "./sandbox-files";

/** Route that opens a session's sandbox in the OS file manager. */
export function sandboxRevealPath(sessionId: string): string {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  return `${prefix}/reveal${query}`;
}

/**
 * Whether this session's sandbox holds anything.
 *
 * Used to tell a candidate folder apart from one that was never written to.
 * A sandbox that does not exist walks to nothing, so it reports no files
 * rather than an error, which is what makes this usable as a probe.
 */
export async function sandboxHasFiles(sessionId: string): Promise<boolean> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const response = await authFetch(`${prefix}${query}`);
  if (!response.ok) return false;
  const body: unknown = await response.json();
  const files = (body as { files?: unknown } | null)?.files;
  return Array.isArray(files) && files.length > 0;
}

/**
 * Open a chat's sandbox folder in the OS file manager.
 *
 * The backend does the opening, so the folder lands on the user's own desktop
 * only when the backend runs there. Callers gate this on the desktop app.
 */
export async function revealSandbox(sessionId: string): Promise<void> {
  const response = await authFetch(sandboxRevealPath(sessionId), {
    method: "POST",
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        const value = (body as { detail?: unknown }).detail;
        if (typeof value === "string") detail = value;
      }
    } catch {
      // A non-JSON error body leaves the status as the only thing to report.
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
}
