// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { sandboxRoutePrefix } from "./sandbox-files";

export function sandboxRevealPath(sessionId: string): string {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  return `${prefix}/reveal${query}`;
}

/** A missing sandbox lists as []; a non-OK is thrown so callers never open a different workspace. */
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

/** The backend does the opening, so callers gate this on the desktop app. */
export async function revealSandbox(sessionId: string): Promise<void> {
  const response = await authFetch(sandboxRevealPath(sessionId), {
    method: "POST",
  });
  if (!response.ok) {
    // Status lets callers tell "no folder yet" (404) from a failure.
    throw Object.assign(new Error(await readFastApiError(response)), {
      status: response.status,
    });
  }
}
