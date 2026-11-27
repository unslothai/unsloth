// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

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
 * Uploads to sandbox root; a taken name gets 409 unless overwrite, protecting tool-written files.
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
    throw Object.assign(new Error(await readFastApiError(response)), {
      status: response.status,
    });
  }
  const result = (await response.json()) as SandboxUploadResult;
  for (const listener of sandboxUploadListeners) {
    listener(sessionId);
  }
  return result;
}
