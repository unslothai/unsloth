// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { downloadShortLivedUrl } from "@/lib/native-files";

import { sandboxFilePath, sandboxRoutePrefix } from "./sandbox-files";

// Stream directly to disk: artifacts can be gigabytes.
export async function saveSandboxFile(
  sessionId: string,
  name: string,
  saveAs = name,
): Promise<void> {
  const path = sandboxFilePath(sessionId, name);
  await downloadShortLivedUrl(saveAs, async () => {
    const query = new URLSearchParams({ session_id: sessionId, filename: name });
    const response = await authFetch(apiUrl(`/api/inference/sandbox-download-link?${query}`), {
      method: "POST",
    });
    if (!response.ok) throw new Error(`Download refused (${response.status})`);
    const { token } = (await response.json()) as { token: string };
    const separator = path.includes("?") ? "&" : "?";
    return apiUrl(`${path}${separator}download_token=${encodeURIComponent(token)}`);
  });
}

/** Relative names in this chat's folder; [] when nothing has been written yet. */
export async function listSandboxFiles(sessionId: string): Promise<string[]> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const response = await authFetch(apiUrl(`${prefix}${query}`));
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const { files } = (await response.json()) as { files?: unknown };
  if (!Array.isArray(files)) return [];
  return files
    .map((entry) => (entry as { name?: unknown } | null)?.name)
    .filter(
      (name): name is string => typeof name === "string" && name.length > 0,
    );
}
