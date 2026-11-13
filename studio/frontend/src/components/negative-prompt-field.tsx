// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { InfoHint } from "@/components/ui/info-hint";
import { Textarea } from "@/components/ui/textarea";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";

// Below the unload request budget; restored legacy recipes may be longer.
export const NEGATIVE_PROMPT_MAX_LENGTH = 20_000;

// Prompt boxes: scrollbar inset from the corners.
export const PROMPT_BOX_CLASS = "image-prompt-box rounded-xl";

export function NegativePromptField({
  value,
  onChange,
  open,
  onOpenChange,
  hint,
  textareaClassName = PROMPT_BOX_CLASS,
}: {
  value: string;
  onChange: (value: string) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  hint: string;
  textareaClassName?: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          aria-expanded={open}
          className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Negative prompt
        </button>
        <button
          type="button"
          // The labelled button above is the accessible toggle; this is decoration.
          aria-hidden={true}
          tabIndex={-1}
          onClick={() => onOpenChange(!open)}
          className="inline-flex"
        >
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={cn(
              "size-3 text-muted-foreground transition-transform",
              open && "rotate-180",
            )}
          />
        </button>
        <InfoHint>{hint}</InfoHint>
      </div>
      {open && (
        <Textarea
          rows={2}
          className={textareaClassName}
          maxLength={NEGATIVE_PROMPT_MAX_LENGTH}
          placeholder="What to avoid (optional)"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}
