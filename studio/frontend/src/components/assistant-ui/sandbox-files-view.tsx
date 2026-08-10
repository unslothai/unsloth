// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Download01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { downloadUrlStreaming, isDownloadCancelled } from "@/lib/native-files";

import { sandboxFilePath, type SandboxFile } from "./sandbox-files";

function formatSize(size: number | null): string {
  if (size === null || size === undefined || Number.isNaN(size)) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function SandboxFileRow({
  sessionId,
  file,
}: {
  sessionId: string;
  file: SandboxFile;
}) {
  const [busy, setBusy] = useState(false);

  // Streamed to the chosen path rather than buffered: a tool can write a
  // multi-gigabyte artifact, and a Blob plus the IPC copy of it would be two
  // more of it in the renderer. The route takes the bearer as a query
  // parameter, since nothing here sends headers.
  const save = useCallback(async () => {
    setBusy(true);
    try {
      const path = sandboxFilePath(sessionId, file.name);
      // The bearer rides in the URL, so nothing refreshes it: an access token
      // that expired during the session would otherwise save a 401 body under
      // the file's name. authFetch refreshes and retries, and the token is read
      // after it; the HEAD also settles whether the file is still there.
      const probe = await authFetch(apiUrl(path), { method: "HEAD" });
      if (!probe.ok) throw new Error(`Download refused (${probe.status})`);
      const token = getAuthToken();
      const separator = path.includes("?") ? "&" : "?";
      // Absolute: the native command parses this and rejects a relative URL,
      // so a bare /api path failed before the request was made.
      const url = apiUrl(
        token ? `${path}${separator}token=${encodeURIComponent(token)}` : path,
      );
      await downloadUrlStreaming(url, file.name);
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(`Could not save ${file.name}.`);
      }
    } finally {
      setBusy(false);
    }
  }, [file.name, sessionId]);

  return (
    <button
      type="button"
      onClick={save}
      disabled={busy}
      title={`Save ${file.name}`}
      className="flex items-center gap-2 rounded border border-border px-2 py-1 text-xs text-foreground hover:bg-muted disabled:opacity-60"
    >
      <HugeiconsIcon icon={File02Icon} className="size-3.5 shrink-0" />
      <span className="truncate font-mono">{file.name}</span>
      {file.size !== null && (
        <span className="text-muted-foreground">{formatSize(file.size)}</span>
      )}
      <HugeiconsIcon icon={Download01Icon} className="size-3.5 shrink-0" />
    </button>
  );
}

/**
 * "Files created" row under a tool card. Without it the only trace of a written
 * file was the model mentioning it in prose.
 */
export function SandboxFiles({
  sessionId,
  files,
}: {
  sessionId: string;
  files: SandboxFile[];
}) {
  if (!sessionId || files.length === 0) return null;
  return (
    <div className="mt-2 border-t border-dashed pt-2">
      <span className="text-xs font-medium text-muted-foreground">
        {files.length === 1 ? "file created" : "files created"}
      </span>
      <div className="mt-1 flex flex-wrap gap-1.5">
        {files.map((file) => (
          <SandboxFileRow key={file.name} sessionId={sessionId} file={file} />
        ))}
      </div>
    </div>
  );
}
