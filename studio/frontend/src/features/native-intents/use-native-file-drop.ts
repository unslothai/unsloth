// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import type React from "react";
import { useCallback, useRef, useState } from "react";
import { registerNativeAttachmentPath } from "./api";
import { nativeAttachmentIntentToFile } from "./native-attachment-file";
import type { NativeIntent } from "./types";
import { useNativeDropTarget } from "./use-native-drop-target";

const PATH_SEPARATOR_RE = /[\\/]/;

/** File name at the end of an OS path, either separator. */
function nativeFileName(path: string): string {
  const segments = path.split(PATH_SEPARATOR_RE);
  return segments[segments.length - 1] || path;
}

/** Extensions from an `accept` list (".pdf,.md"), lowercased with their dot. */
function acceptedExts(accept: string | undefined): string[] {
  if (!accept) return [];
  return accept
    .split(",")
    .map((entry) => entry.trim().toLowerCase())
    .filter((entry) => entry.startsWith("."));
}

function hasAcceptedExt(name: string, exts: string[]): boolean {
  if (exts.length === 0) {
    return true;
  }
  const lower = name.toLowerCase();
  return exts.some((ext) => lower.endsWith(ext));
}

/** Nothing in the payload was droppable here. */
function toastNothingAccepted(
  names: string[],
  accept: string | undefined,
): void {
  // A folder is one extension-less name, and the native side takes files only.
  const looksLikeFolder = names.some((name) => !name.includes("."));
  if (looksLikeFolder) {
    toast.error("Folders can't be dropped here", {
      description: "Drop the files inside it, or use the picker button.",
    });
    return;
  }
  toast.error(
    names.length === 1
      ? "That file type can't be dropped here"
      : "Those file types can't be dropped here",
    accept ? { description: `Accepts ${accept}.` } : undefined,
  );
}

/** Some of the payload was droppable and the rest was not. */
function toastPartiallySkipped(count: number): void {
  if (count <= 0) {
    return;
  }
  toast.error(
    count === 1
      ? "Skipped a file this zone doesn't accept"
      : `Skipped ${count} files this zone doesn't accept`,
  );
}

function reasonText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

/** Register each path, then read it back if the caller wants Files. Per path,
 * so one bad file does not discard the siblings that registered cleanly. */
async function registerDroppedPaths(
  paths: string[],
  register: (path: string) => Promise<NativeIntent>,
  asIntents: boolean,
): Promise<{
  ready: Array<NativeIntent | File>;
  failed: number;
  reason?: unknown;
}> {
  const settled = await Promise.allSettled(
    paths.map(async (path) => {
      const intent = await register(path);
      return asIntents ? intent : await nativeAttachmentIntentToFile(intent);
    }),
  );
  const ready = settled.flatMap((result) =>
    result.status === "fulfilled" ? [result.value] : [],
  );
  const rejection = settled.find((result) => result.status === "rejected");
  return {
    ready,
    failed: settled.length - ready.length,
    reason: rejection?.status === "rejected" ? rejection.reason : undefined,
  };
}

/** Registered or read back, and did not survive it. */
function toastReadFailures(count: number, reason: unknown): void {
  toast.error(
    count === 1
      ? "Couldn't read a dropped file"
      : `Couldn't read ${count} dropped files`,
    { description: reason === undefined ? undefined : reasonText(reason) },
  );
}

export interface NativeFileDropOptions {
  /** Receives real Files, whether the OS or the DOM delivered them. */
  onFiles: (files: File[]) => void | Promise<void>;
  /** Take registered paths instead of Files, for zones that upload by lease.
   * The native reader only serves media inline, so documents need this. */
  onNativeIntents?: (intents: NativeIntent[]) => void | Promise<void>;
  /** Extensions this zone takes, as an `accept` list (".pdf,.md"). Omitted takes anything. */
  accept?: string;
  /** Refuse drops, with `disabledReason` said out loud rather than swallowed. */
  disabled?: boolean;
  disabledReason?: string;
  /** Single-slot pickers take only the first file of a batch. */
  multiple?: boolean;
  /** Register under a policy other than the attachment one (datasets, models). */
  register?: (path: string) => Promise<NativeIntent>;
}

export interface NativeFileDrop {
  /** Attach to the element that owns the drop; claims native drops landing on it. */
  ref: (element: HTMLElement | null) => void;
  dragging: boolean;
  dragHandlers: {
    onDragEnter: (event: React.DragEvent) => void;
    onDragOver: (event: React.DragEvent) => void;
    onDragLeave: (event: React.DragEvent) => void;
    onDrop: (event: React.DragEvent) => void;
  };
}

/** One drop zone that works on web and on desktop.
 *
 * Tauri delivers OS drops window-wide and suppresses the webview's own drop
 * events, so a zone wired only to `onDrop` is dead in the desktop app (#9036).
 * This claims the native drop for the element and hands back the same Files. */
export function useNativeFileDrop(
  options: NativeFileDropOptions,
): NativeFileDrop {
  const [dragging, setDragging] = useState(false);
  // Read through a ref so a fresh caller closure does not re-register the target.
  const latest = useRef(options);
  latest.current = options;
  // dragenter/dragleave fire per child, so a raw boolean flickers on inner moves.
  const dragDepth = useRef(0);

  const deliver = useCallback((files: File[]) => {
    const current = latest.current;
    if (files.length === 0) return;
    void current.onFiles(
      current.multiple === false ? files.slice(0, 1) : files,
    );
  }, []);

  const handleNativePaths = useCallback(
    async (paths: string[]) => {
      const current = latest.current;
      if (current.disabled) {
        toast.error(
          current.disabledReason ?? "This drop zone is busy right now",
        );
        return;
      }
      const exts = acceptedExts(current.accept);
      const supported = paths.filter((path) =>
        hasAcceptedExt(nativeFileName(path), exts),
      );
      if (supported.length === 0) {
        toastNothingAccepted(paths.map(nativeFileName), current.accept);
        return;
      }
      const takeIntents = current.onNativeIntents;
      const { ready, failed, reason } = await registerDroppedPaths(
        current.multiple === false ? supported.slice(0, 1) : supported,
        current.register ?? registerNativeAttachmentPath,
        Boolean(takeIntents),
      );
      if (ready.length > 0) {
        if (takeIntents) {
          void takeIntents(ready as NativeIntent[]);
        } else {
          deliver(ready as File[]);
        }
      }
      if (failed > 0) {
        toastReadFailures(failed, reason);
        return;
      }
      toastPartiallySkipped(paths.length - supported.length);
    },
    [deliver],
  );

  const ref = useNativeDropTarget({
    onDrop: (paths) => void handleNativePaths(paths),
    onDragOver: (over) => setDragging(over && !latest.current.disabled),
  });

  const endDrag = useCallback(() => {
    dragDepth.current = 0;
    setDragging(false);
  }, []);

  // Files only: preventDefault on a text drag kills editing in wrapped inputs.
  const isFileDrag = (event: React.DragEvent): boolean =>
    Array.from(event.dataTransfer?.types ?? []).includes("Files");

  // preventDefault runs even under Tauri and while disabled, or the webview
  // navigates to the dropped file.
  const dragHandlers = {
    onDragEnter: (event: React.DragEvent) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      if (isTauri || latest.current.disabled) return;
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (event: React.DragEvent) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      if (isTauri || latest.current.disabled) return;
      event.dataTransfer.dropEffect = "copy";
    },
    onDragLeave: (event: React.DragEvent) => {
      if (isTauri || !isFileDrag(event)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (event: React.DragEvent) => {
      if (!isFileDrag(event)) {
        return;
      }
      event.preventDefault();
      // The native target owns this on desktop; both would attach it twice.
      if (isTauri) {
        return;
      }
      endDrag();
      const current = latest.current;
      if (current.disabled) {
        toast.error(
          current.disabledReason ?? "This drop zone is busy right now",
        );
        return;
      }
      const exts = acceptedExts(current.accept);
      const dropped = Array.from(event.dataTransfer.files ?? []);
      const supported = dropped.filter((file) =>
        hasAcceptedExt(file.name, exts),
      );
      if (dropped.length > 0 && supported.length === 0) {
        toastNothingAccepted(
          dropped.map((file) => file.name),
          current.accept,
        );
        return;
      }
      deliver(supported);
      toastPartiallySkipped(dropped.length - supported.length);
    },
  };

  return { ref, dragging, dragHandlers };
}
