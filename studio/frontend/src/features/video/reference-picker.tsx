// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";
import { Delete02Icon, FlimSlateIcon, MusicNote01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  CHAT_AUDIO_DROP_ACCEPT,
  CHAT_VIDEO_DROP_ACCEPT,
} from "@/features/native-intents/drop-paths";
import { useNativeFileDrop } from "@/features/native-intents";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import { readReferenceFile } from "./reference-budget";

/** One staged reference file: the data URL the request carries, plus its name for the chip. */
export interface ReferenceMedia {
  name: string;
  dataUrl: string;
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

  const readFile = useCallback(
    (file: File | undefined | null) => {
      readReferenceFile(kind, file, {
        onLoaded: (dataUrl) =>
          onChange(dataUrl === null || !file ? null : { name: file.name, dataUrl }),
        onError: (message) => toast.error(message),
      });
    },
    [kind, onChange],
  );

  // Tauri suppresses the webview's own drop events, so the handlers below never
  // fire on the desktop app; this claims the OS drop for the button (#9036).
  const { ref: dropRef, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: (files) => readFile(files[0]),
    accept: kind === "video" ? CHAT_VIDEO_DROP_ACCEPT : CHAT_AUDIO_DROP_ACCEPT,
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
        <span className="min-w-0 flex-1 truncate text-ui-11 text-foreground">{value.name}</span>
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
        accept={`${kind}/*`}
        className="hidden"
        onChange={(e) => readFile(e.target.files?.[0])}
      />
    </button>
  );
}
