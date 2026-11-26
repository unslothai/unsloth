// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

import { sandboxRoutePrefix } from "./sandbox-files";

/** Mirrors SANDBOX_UPLOAD_MAX_BYTES in backend/utils/upload_limits.py. */
export const MAX_SANDBOX_UPLOAD_BYTES = 512 * 1024 * 1024;

/** Told the session id after each upload, so open views can re-list the folder. */
export const sandboxUploadListeners = new Set<(sessionId: string) => void>();

export type SandboxUploadResult = {
  name: string;
  path: string;
  replaced: boolean;
};

/**
 * Put one file into a chat's sandbox folder, at its root. A taken name is refused with a 409
 * unless `overwrite`, so a second upload never silently replaces what a tool wrote.
 */
export async function uploadToSandbox(
  sessionId: string,
  file: File,
  overwrite = false,
): Promise<SandboxUploadResult> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("overwrite", overwrite ? "true" : "false");
  const response = await authFetch(`${prefix}/upload${query}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body: unknown = await response.json();
      const value = (body as { detail?: unknown } | null)?.detail;
      if (typeof value === "string") detail = value;
    } catch {
      // A non-JSON error body leaves the status as the only thing to report.
    }
    throw Object.assign(
      new Error(detail || `Request failed (${response.status})`),
      {
        status: response.status,
      },
    );
  }
  const result = (await response.json()) as SandboxUploadResult;
  for (const listener of sandboxUploadListeners) {
    listener(sessionId);
  }
  return result;
}
