// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  nativeFileName,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { MAX_FOLDER_FILES, filesFromDrop } from "@/lib/dropped-folders";
import { toast } from "@/lib/toast";
import {
  type DragEvent as ReactDragEvent,
  useCallback,
  useRef,
  useState,
} from "react";
import {
  SUPPORTED_SOURCES_HINT,
  isSupportedSourceName,
  partitionSupported,
} from "./source-drop-policy";
import { type RagUploadItem, uploadItemFromIntent } from "./use-rag-documents";

export interface SourceDropOptions {
  /** Receives what the drop yielded; never called with an empty list. */
  onItems: (items: RagUploadItem[]) => void;
  /** Set while the surface cannot take files, such as during an upload the
   * uploader would not keep separate from a second one. The drop is refused
   * with this message rather than started. */
  disabledReason?: string;
}

export interface SourceDropProps {
  onDragEnter: (event: ReactDragEvent) => void;
  onDragOver: (event: ReactDragEvent) => void;
  onDragLeave: (event: ReactDragEvent) => void;
  onDrop: (event: ReactDragEvent) => void;
}

export interface SourceDrop {
  dragging: boolean;
  dropProps: SourceDropProps;
  nativeDropTarget: (element: HTMLElement | null) => void;
}

function isFileDrag(event: ReactDragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

/** Drop files onto a RAG surface, from the browser or from the desktop, where a
 * drop arrives as a path rather than bytes. */
export function useSourceDrop({
  onItems,
  disabledReason,
}: SourceDropOptions): SourceDrop {
  const disabled = disabledReason !== undefined;
  // A native drop registers its paths before upload() flips `uploading`, so
  // that window needs its own flag or a second drop starts a second upload.
  const registering = useRef(false);
  // Count enter/leave pairs: children fire dragleave on the parent.
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  const reportUnsupported = useCallback((names: string[]) => {
    if (names.length === 0) return;
    toast.info(
      names.length === 1
        ? `Can't add ${names[0]}`
        : `Can't add ${names.length} files`,
      { description: SUPPORTED_SOURCES_HINT },
    );
  }, []);

  const addNativePaths = useCallback(
    async (paths: string[]) => {
      const { supported, unsupported } = partitionSupported(
        paths,
        nativeFileName,
      );
      reportUnsupported(unsupported);
      if (supported.length === 0) return;
      registering.current = true;
      // Per path, so one rejected file does not discard the rest of the drop.
      const settled = await Promise.allSettled(
        supported.map(registerNativeAttachmentPath),
      ).finally(() => {
        registering.current = false;
      });
      const items = settled.flatMap((result) =>
        result.status === "fulfilled"
          ? [uploadItemFromIntent(result.value)]
          : [],
      );
      const failed = settled.length - items.length;
      if (failed > 0) {
        toast.error(
          failed === 1
            ? "Couldn't add a dropped file"
            : `Couldn't add ${failed} dropped files`,
        );
      }
      if (items.length > 0) onItems(items);
    },
    [onItems, reportUnsupported],
  );

  /** Why a drop cannot be taken right now, or undefined when it can. */
  const refusal = useCallback(
    () =>
      disabledReason ??
      (registering.current
        ? "Still adding the last drop. Try again in a moment."
        : undefined),
    [disabledReason],
  );

  const nativeDropTarget = useNativeDropTarget({
    onDrop: (paths) => {
      const reason = refusal();
      if (reason !== undefined) {
        toast.info(reason);
        return;
      }
      void addNativePaths(paths);
    },
    onDragOver: (over) => setDragging(over && !disabled),
  });

  const endDrag = useCallback(() => {
    dragDepth.current = 0;
    setDragging(false);
  }, []);

  // Every drag is cancelled, not just a file one: nothing above these surfaces
  // cancels a link drop, so the browser would navigate away from Unsloth.
  const dropProps: SourceDropProps = {
    onDragEnter: (event) => {
      event.preventDefault();
      if (disabled || !isFileDrag(event)) return;
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (event) => {
      event.preventDefault();
      if (disabled || !isFileDrag(event)) return;
      event.dataTransfer.dropEffect = "copy";
    },
    onDragLeave: (event) => {
      if (disabled || !isFileDrag(event)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (event) => {
      event.preventDefault();
      endDrag();
      if (!isFileDrag(event)) return;
      const reason = refusal();
      if (reason !== undefined) {
        toast.info(reason);
        return;
      }
      // Read synchronously: the browser clears the drop's items once this handler returns.
      void filesFromDrop(event.dataTransfer, isSupportedSourceName).then(
        ({ files, truncated, hadFolder }) => {
          const { supported, unsupported } = partitionSupported(
            files,
            (file) => file.name,
          );
          // Inside a folder, files of other types are expected and not worth a toast each.
          if (!hadFolder) reportUnsupported(unsupported);
          if (hadFolder && supported.length === 0) {
            toast.info("No supported files in that folder", {
              description: SUPPORTED_SOURCES_HINT,
            });
          }
          if (truncated > 0) {
            toast.info(`Added the first ${MAX_FOLDER_FILES} files`, {
              description: `${truncated} more were left out. Drop a smaller folder for the rest.`,
            });
          }
          if (supported.length > 0) {
            onItems(supported.map((file) => ({ kind: "file" as const, file })));
          }
        },
        () => toast.error("Couldn't read the dropped folder"),
      );
    },
  };

  return { dragging, dropProps, nativeDropTarget };
}
