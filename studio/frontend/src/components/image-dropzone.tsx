// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef, useState } from "react";
import { Delete02Icon, ImageAdd02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

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

  const readFile = useCallback(
    (file: File | undefined | null) => {
      if (!file || !file.type.startsWith("image/")) {
        if (file) toast.error("Please choose an image file");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => onChange(typeof reader.result === "string" ? reader.result : null);
      reader.onerror = () => toast.error("Could not read the image");
      reader.readAsDataURL(file);
    },
    [onChange],
  );

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
