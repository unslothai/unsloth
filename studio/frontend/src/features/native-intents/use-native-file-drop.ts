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

function nativeFileName(path: string): string {
  const segments = path.split(PATH_SEPARATOR_RE);
  return segments[segments.length - 1] || path;
}

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

function toastNothingAccepted(
  names: string[],
  accept: string | undefined,
): void {
  // A folder is an extension-less name, and the native side takes files only.
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

/** Per path, so one bad file does not discard siblings. */
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

function toastReadFailures(count: number, reason: unknown): void {
  toast.error(
    count === 1
      ? "Couldn't read a dropped file"
      : `Couldn't read ${count} dropped files`,
    { description: reason === undefined ? undefined : reasonText(reason) },
  );
}

export interface NativeFileDropOptions {
  onFiles: (files: File[]) => void | Promise<void>;
  /** The native reader only serves media inline, so documents need registered paths. */
  onNativeIntents?: (intents: NativeIntent[]) => void | Promise<void>;
  accept?: string;
  disabled?: boolean;
  disabledReason?: string;
  multiple?: boolean;
  register?: (path: string) => Promise<NativeIntent>;
}

export interface NativeFileDrop {
  ref: (element: HTMLElement | null) => void;
  dragging: boolean;
  dragHandlers: {
    onDragEnter: (event: React.DragEvent) => void;
    onDragOver: (event: React.DragEvent) => void;
    onDragLeave: (event: React.DragEvent) => void;
    onDrop: (event: React.DragEvent) => void;
  };
}

/** Drop zone for web and desktop: Tauri suppresses webview drop events, so `onDrop` alone is
  dead on desktop. */
export function useNativeFileDrop(
  options: NativeFileDropOptions,
): NativeFileDrop {
  const [dragging, setDragging] = useState(false);
  // Ref so a fresh caller closure does not re-register the target.
  const latest = useRef(options);
  latest.current = options;
  // dragenter/dragleave fire per child, so a boolean would flicker.
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

  // preventDefault always, or the webview navigates to the dropped file.
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
