// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Download01Icon,
  FileEmpty02Icon,
  FolderOpenIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl, isTauri } from "@/lib/api-base";
import { downloadUrlStreaming, isDownloadCancelled } from "@/lib/native-files";
import { cn } from "@/lib/utils";

import { FileContextMenu, loadSandboxFile } from "./link-context-menu";
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

  // Streamed to disk rather than buffered: artifacts can be gigabytes. The bearer goes in the query.
  const save = useCallback(async () => {
    setBusy(true);
    try {
      const path = sandboxFilePath(sessionId, file.name);
      // The URL bearer is never refreshed, so authFetch refreshes first; the HEAD confirms the file.
      const probe = await authFetch(apiUrl(path), { method: "HEAD" });
      if (!probe.ok) throw new Error(`Download refused (${probe.status})`);
      const token = getAuthToken();
      const separator = path.includes("?") ? "&" : "?";
      // Absolute: the native command rejects a relative URL.
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
    <FileContextMenu
      file={{
        name: file.name.slice(file.name.lastIndexOf("/") + 1),
        load: () => loadSandboxFile(sessionId, file.name),
        open: () => void save(),
        sandbox: { sessionId, file: file.name },
      }}
    >
      <button
        type="button"
        onClick={save}
        disabled={busy}
        title={`Save ${file.name}`}
        className="flex items-center gap-2 rounded border border-border px-2 py-1 text-xs text-foreground hover:bg-muted disabled:opacity-60"
      >
        <HugeiconsIcon icon={FileEmpty02Icon} className="size-3.5 shrink-0" />
        <span className="truncate font-mono">{file.name}</span>
        {file.size !== null && (
          <span className="text-muted-foreground">{formatSize(file.size)}</span>
        )}
        <HugeiconsIcon icon={Download01Icon} className="size-3.5 shrink-0" />
      </button>
    </FileContextMenu>
  );
}

/** Opens the folder on desktop only: the backend opens the file manager. */
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

export function SandboxFiles({
  sessionId,
  files,
  className,
}: {
  sessionId: string;
  files: SandboxFile[];
  className?: string;
}) {
  if (!sessionId || files.length === 0) return null;
  return (
    <div className={cn("mt-2 border-t border-dashed pt-2", className)}>
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
