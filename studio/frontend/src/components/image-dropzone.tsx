// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { Delete02Icon, ImageAdd02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  readNativeAttachmentFile,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

// All registerNativeAttachmentPath takes. The picker itself accepts image/*,
// so name the droppable formats instead of surfacing the backend's refusal.
const NATIVE_IMAGE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"];

/** Shared image picker that returns a data URL. */
export function ImageDropzone({
  value,
  onChange,
  label = "Click or drop an image",
  removeLabel = "Remove source image",
  className,
}: {
  value: string | null;
  onChange: (dataUrl: string | null) => void;
  /** Prompt shown in the empty state. */
  label?: string;
  /** Accessible remove-button name. */
  removeLabel?: string;
  className?: string;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  // File reads can finish after another picker action. Only the newest
  // selection may update the shared field.
  const selection = useRef(0);
  // The sequence is per instance, so it cannot see a read outliving this
  // picker: `onChange` is shared, and a late write lands on whoever holds it.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  // The reference slots are keyed by index, so removing one shifts a different
  // image into a picker that stays mounted. A `value` this picker did not set
  // means a read in flight is for the slot as it used to be.
  const seen = useRef(value);
  useEffect(() => {
    if (seen.current === value) return;
    seen.current = value;
    selection.current += 1;
  }, [value]);

  const readFile = useCallback(
    (file: File | undefined | null) => {
      if (!file || !file.type.startsWith("image/")) {
        if (file) toast.error("Please choose an image file");
        return;
      }
      selection.current += 1;
      const claimed = selection.current;
      const reader = new FileReader();
      reader.onload = () => {
        if (claimed !== selection.current) return;
        onChange(typeof reader.result === "string" ? reader.result : null);
      };
      reader.onerror = () => toast.error("Could not read the image");
      reader.readAsDataURL(file);
    },
    [onChange],
  );

  // Tauri suppresses the webview drop event, so desktop drops arrive as paths
  // that the native side registers and reads for this picker.
  const readNativePath = useCallback(
    async (path: string | undefined) => {
      if (!path) return;
      if (!NATIVE_IMAGE_EXTS.includes(path.split(".").pop()?.toLowerCase() ?? "")) {
        toast.error("Drop a JPEG, PNG, WebP or GIF image", {
          description: "Other image formats can still be chosen with the picker.",
        });
        return;
      }
      selection.current += 1;
      const claimed = selection.current;
      try {
        const intent = await registerNativeAttachmentPath(path);
        const file = await readNativeAttachmentFile(intent.path.token);
        if (!mounted.current || claimed !== selection.current) return;
        onChange(`data:${file.mimeType};base64,${file.base64}`);
      } catch (error) {
        toast.error("Could not read the image", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [onChange],
  );

  const nativeDropRef = useNativeDropTarget({
    onDrop: (paths) => void readNativePath(paths[0]),
    onDragOver: setDragging,
  });

  if (value) {
    return (
      <div className={cn("relative overflow-hidden rounded-[10px] border border-border", className)}>
        <img src={value} alt="Source" className="max-h-44 w-full object-contain bg-muted/30" />
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              aria-label={removeLabel}
              className="absolute right-1.5 top-1.5 size-7"
              onClick={() => {
                selection.current += 1;
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
      ref={nativeDropRef}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        readFile(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "flex h-28 w-full flex-col items-center justify-center gap-1 rounded-xl border border-dashed text-xs transition-colors",
        dragging
          ? "border-primary/60 bg-primary/5 text-foreground"
          : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
        className,
      )}
    >
      <HugeiconsIcon icon={ImageAdd02Icon} className="size-5" />
      <span>{label}</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => readFile(e.target.files?.[0])}
      />
    </button>
  );
}
