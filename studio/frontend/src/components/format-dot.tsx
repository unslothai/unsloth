// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";

export type FormatDotTone = "gguf" | "checkpoint" | "adapter";

const FORMAT_DOT_LABEL: Record<FormatDotTone, string> = {
  gguf: "GGUF",
  checkpoint: "Safetensors",
  adapter: "Adapter",
};

const FORMAT_DOT_CLASS: Record<FormatDotTone, string> = {
  gguf: "bg-format-gguf",
  checkpoint: "bg-format-checkpoint",
  adapter: "bg-format-adapter",
};

export function FormatDot({
  tone,
  label,
  className,
}: {
  tone: FormatDotTone;
  label?: string;
  className?: string;
}) {
  return (
    <span
      role="img"
      aria-label={label ?? FORMAT_DOT_LABEL[tone]}
      className={cn(
        "inline-block size-[5px] shrink-0 rounded-full",
        FORMAT_DOT_CLASS[tone],
        className,
      )}
    />
  );
}
