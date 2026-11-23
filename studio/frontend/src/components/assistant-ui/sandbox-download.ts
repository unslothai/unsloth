// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { downloadUrlStreaming } from "@/lib/native-files";

import { sandboxFilePath } from "./sandbox-files";

// Streamed to disk rather than buffered: artifacts can be gigabytes. The bearer goes in the query.
export async function saveSandboxFile(
  sessionId: string,
  name: string,
): Promise<void> {
  const path = sandboxFilePath(sessionId, name);
  // The URL bearer is never refreshed, so authFetch refreshes first; the HEAD confirms the file.
  const probe = await authFetch(apiUrl(path), { method: "HEAD" });
  if (!probe.ok) throw new Error(`Download refused (${probe.status})`);
  const token = getAuthToken();
  const separator = path.includes("?") ? "&" : "?";
  // Absolute: the native command rejects a relative URL.
  const url = apiUrl(
    token ? `${path}${separator}token=${encodeURIComponent(token)}` : path,
  );
  await downloadUrlStreaming(url, name);
}
