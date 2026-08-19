// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Download01Icon,
  File02Icon,
  FolderOpenIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl, isTauri } from "@/lib/api-base";
import { downloadUrlStreaming, isDownloadCancelled } from "@/lib/native-files";

import { sandboxFilePath, type SandboxFile } from "./sandbox-files";
import { revealSandbox } from "./sandbox-reveal";

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
  // multi-gigabyte artifact, and a Blob plus its IPC copy would be two more of
  // it in the renderer. The bearer goes in the query: no headers are sent.
  const save = useCallback(async () => {
    setBusy(true);
    try {
      const path = sandboxFilePath(sessionId, file.name);
      // The bearer rides in the URL, so nothing refreshes it: an expired
      // access token would save a 401 body under the file's name. authFetch
      // refreshes and retries, and the HEAD settles that the file is there.
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
 * The row's heading, doubling as the way into the folder itself on desktop.
 * The backend opens the file manager, so in a browser it stays plain text and
 * says why.
 */
function SandboxFolderLabel({
  sessionId,
  label,
}: {
  sessionId: string;
  label: string;
}) {
  const open = useCallback(() => {
    revealSandbox(sessionId).catch(() => {
      toast.error("Could not open the chat folder.");
    });
  }, [sessionId]);

  if (!isTauri) {
    return (
      <span
        className="text-xs font-medium text-muted-foreground"
        title="Opening the folder needs the desktop app. In a browser, save a file with the button below."
      >
        {label}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={open}
      title="Open the folder these files were written to"
      className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
    >
      <HugeiconsIcon icon={FolderOpenIcon} className="size-3.5 shrink-0" />
      {label}
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
      <SandboxFolderLabel
        sessionId={sessionId}
        label={files.length === 1 ? "file created" : "files created"}
      />
      <div className="mt-1 flex flex-wrap gap-1.5">
        {files.map((file) => (
          <SandboxFileRow key={file.name} sessionId={sessionId} file={file} />
        ))}
      </div>
    </div>
  );
}
