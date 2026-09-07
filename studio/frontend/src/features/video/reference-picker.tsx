// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { Delete02Icon, FlimSlateIcon, MusicNote01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useNativeFileDrop } from "@/features/native-intents";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  REFERENCE_DROP_ACCEPT,
  REFERENCE_PICKER_ACCEPT,
  createReferenceSelectionGate,
  readReferenceFile,
} from "./reference-budget";
import { classifiedAttachmentFile } from "@/lib/video-utils";

/** One staged reference file: the data URL the request carries, plus its name for the chip. */
export interface ReferenceMedia {
  name: string;
  dataUrl: string;
  durationSeconds?: number;
}

/** How long to wait for a browser to report a clip's duration before giving up on it. */
export const REFERENCE_DURATION_TIMEOUT_MS = 15_000;

/** Read a clip's duration, resolving undefined when the browser cannot report one. An element
 *  firing neither loadedmetadata nor error would leave this pending forever, and its picker slot
 *  with it, so the wait is bounded; callers already treat an unknown duration as "no auto trim".
 *  The source stays a data URL because WebKit reports no metadata for an object URL. */
function readVideoDuration(dataUrl: string): Promise<number | undefined> {
  return new Promise((resolve) => {
    const media = document.createElement("video");
    let settled = false;
    const finish = (duration?: number) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      media.onloadedmetadata = null;
      media.onerror = null;
      media.removeAttribute("src");
      // Without a load() the element can keep the resource it was decoding.
      media.load();
      resolve(duration);
    };
    const timer = setTimeout(() => finish(), REFERENCE_DURATION_TIMEOUT_MS);
    media.preload = "metadata";
    media.onloadedmetadata = () =>
      finish(Number.isFinite(media.duration) && media.duration > 0 ? media.duration : undefined);
    media.onerror = () => finish();
    media.src = dataUrl;
  });
}

/** Reference video or audio picker that displays the selected filename. */
export function ReferenceMediaPicker({
  kind,
  value,
  onChange,
  label,
  compact = false,
}: {
  kind: "video" | "audio";
  value: ReferenceMedia | null;
  onChange: (media: ReferenceMedia | null) => void;
  /** Prompt shown in the empty state. */
  label: string;
  /** Half-height row, for the soundtrack slot that sits under its video. */
  compact?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Only the newest read from this mounted picker may update the list.
  const [gate] = useState(createReferenceSelectionGate);
  useEffect(() => gate.mount(), [gate]);
  // Index-keyed slots can receive a sibling when an earlier reference is removed.
  const seen = useRef(value);
  useEffect(() => {
    if (seen.current === value) return;
    seen.current = value;
    gate.invalidate();
  }, [value, gate]);

  const readFile = useCallback(
    async (picked: File | undefined | null) => {
      if (!picked) return;
      const claim = gate.begin();
      // A .3gp is a recording or a clip and its name says neither, so read the
      // container's tracks before the kind check, as chat and compare do.
      const file = await classifiedAttachmentFile(picked);
      if (!claim.isCurrent()) return;
      readReferenceFile(kind, file, {
        onLoaded: (dataUrl) => {
          if (!claim.isCurrent()) return;
          if (dataUrl === null) {
            onChange(null);
            return;
          }
          if (kind !== "video") {
            onChange({ name: file.name, dataUrl });
            return;
          }
          void readVideoDuration(dataUrl).then((durationSeconds) => {
            if (!claim.isCurrent()) return;
            onChange({ name: file.name, dataUrl, durationSeconds });
          });
        },
        onError: (message) => toast.error(message),
      });
    },
    [gate, kind, onChange],
  );

  // Tauri suppresses the webview's own drop events, so the handlers below never fire on the desktop
  // app; this claims the OS drop for the button (#9036).
  const { ref: dropRef, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: (files) => void readFile(files[0]),
    accept: REFERENCE_DROP_ACCEPT[kind],
    multiple: false,
  });

  const icon = kind === "video" ? FlimSlateIcon : MusicNote01Icon;

  if (value) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 rounded-[10px] border border-border bg-muted/30 px-2.5",
          compact ? "h-8" : "h-11",
        )}
      >
        <HugeiconsIcon icon={icon} className="size-3.5 shrink-0 text-muted-foreground" />
        {/* The file input is early-returned past in this branch, so the
            type=file redaction never sees it: mark the name itself. */}
        <span
          data-reload-snapshot-sensitive
          className="min-w-0 flex-1 truncate text-ui-11 text-foreground"
        >
          {value.name}
        </span>
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={`Remove ${label}`}
              className="size-6 shrink-0"
              onClick={() => {
                onChange(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Remove</TooltipContent>
        </Tooltip>
      </div>
    );
  }

  return (
    <button
      type="button"
      ref={dropRef}
      onClick={() => inputRef.current?.click()}
      {...dragHandlers}
      className={cn(
        "flex w-full items-center justify-center gap-1.5 rounded-[10px] border border-dashed text-ui-11 transition-colors",
        compact ? "h-8" : "h-11",
        dragging
          ? "border-primary/60 bg-primary/5 text-foreground"
          : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
      )}
    >
      <HugeiconsIcon icon={icon} className="size-3.5" />
      <span>{label}</span>
      <input
        ref={inputRef}
        type="file"
        accept={REFERENCE_PICKER_ACCEPT[kind]}
        className="hidden"
        onChange={(e) => void readFile(e.target.files?.[0])}
      />
    </button>
  );
}
