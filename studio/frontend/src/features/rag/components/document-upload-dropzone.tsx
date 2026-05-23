// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Upload04Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";

const ACCEPTED = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";

export function DocumentUploadDropzone({
  onFiles,
  disabled,
  className,
}: {
  onFiles: (files: File[]) => void | Promise<void>;
  disabled?: boolean;
  className?: string;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0 || disabled) return;
    void onFiles(Array.from(files));
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-4 py-6 transition-colors",
        isDragging
          ? "border-primary bg-primary/5"
          : "border-border/60 bg-muted/30",
        disabled && "opacity-60",
        className,
      )}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <HugeiconsIcon
        icon={Upload04Icon}
        size={24}
        className="text-muted-foreground"
      />
      <div className="text-sm text-muted-foreground">
        Drop files here or
        <Button
          variant="link"
          size="sm"
          className="px-1"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          browse
        </Button>
      </div>
      <div className="text-xs text-muted-foreground">
        PDF, TXT, MD, DOCX, HTML
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        multiple
        hidden
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
