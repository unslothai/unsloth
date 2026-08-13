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
 * A sandbox that does not exist walks to nothing, so "no sandbox" already
 * arrives as a successful listing with an empty array. That is what makes a
 * non-OK response mean something else entirely, and why it is thrown rather
 * than read as "no files": swallowing it would send the caller on to its
 * fallback and open a different workspace, reporting nothing.
 */
export async function sandboxHasFiles(sessionId: string): Promise<boolean> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const response = await authFetch(`${prefix}${query}`);
  if (!response.ok) {
    throw new Error(`Could not read the chat's folder (${response.status})`);
  }
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
