// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AlertCircleIcon, FileXIcon } from "lucide-react";
import type { FC } from "react";

interface PreviewUnavailableProps {
  /** Filename if known; "Document" otherwise. */
  filename?: string;
  /** One-line reason — from the backend's error body if available,
   *  else a generic copy. */
  reason: string;
  /** "missing" → deleted/404; "error" → other failures. Icon + tone
   *  change so a stale citation reads as "no longer available" rather
   *  than a transient blip. */
  variant?: "missing" | "error";
}

export const PreviewUnavailable: FC<PreviewUnavailableProps> = ({
  filename,
  reason,
  variant = "error",
}) => {
  const Icon = variant === "missing" ? FileXIcon : AlertCircleIcon;
  const headline =
    variant === "missing" ? "Document unavailable" : "Couldn't load preview";

  return (
    <output className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
      <Icon
        className="size-10 text-muted-foreground"
        strokeWidth={1.5}
        aria-hidden={true}
      />
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium">{headline}</p>
        {filename ? (
          <p className="text-xs text-muted-foreground" title={filename}>
            {filename}
          </p>
        ) : null}
        <p className="mt-1 max-w-xs text-xs text-muted-foreground">{reason}</p>
      </div>
    </output>
  );
};
