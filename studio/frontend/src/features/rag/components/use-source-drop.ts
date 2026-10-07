// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  nativeFileName,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";
import {
  type DragEvent as ReactDragEvent,
  useCallback,
  useRef,
  useState,
} from "react";
import { partitionSupported } from "./source-drop-policy";
import { type RagUploadItem, uploadItemFromIntent } from "./use-rag-documents";

export interface SourceDropOptions {
  onItems: (items: RagUploadItem[]) => void;
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

export function useSourceDrop({
  onItems,
  disabledReason,
}: SourceDropOptions): SourceDrop {
  const disabled = disabledReason !== undefined;
  // A native drop registers before upload() flips `uploading`, so it needs its own flag.
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
      { description: "Supported types: documents and source code files" },
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

  // Cancel every drag, not just files, or a link drop navigates away from Unsloth.
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
      const { supported, unsupported } = partitionSupported(
        Array.from(event.dataTransfer.files ?? []),
        (file) => file.name,
      );
      reportUnsupported(unsupported);
      if (supported.length > 0) {
        onItems(supported.map((file) => ({ kind: "file" as const, file })));
      }
    },
  };

  return { dragging, dropProps, nativeDropTarget };
}
