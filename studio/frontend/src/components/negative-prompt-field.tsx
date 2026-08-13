// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown } from "lucide-react";

import { InfoHint } from "@/components/ui/info-hint";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

/** Collapsible negative prompt, shared by the Images and Video Create panels. */
export function NegativePromptField({
  value,
  onChange,
  open,
  onOpenChange,
  hint,
}: {
  value: string;
  onChange: (value: string) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  hint: string;
}) {
  // Same shape as Field, so it keeps the panel's spacing.
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
        {/* Chevron before the hint, so it stays next to the label it expands. */}
        <button
          type="button"
          // The labelled button above is the accessible toggle; this is decoration.
          aria-hidden={true}
          tabIndex={-1}
          onClick={() => onOpenChange(!open)}
          className="inline-flex"
        >
          <ChevronDown
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
          placeholder="What to avoid (optional)"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}
