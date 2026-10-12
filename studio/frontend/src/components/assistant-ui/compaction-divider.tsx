// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveIcon } from "lucide-react";
import type { FC, ReactNode } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type ContextTruncation,
  promptWasShortened,
} from "@/features/chat/utils/context-truncation";
import { cn } from "@/lib/utils";

const DetailRow: FC<{ label: string; value: ReactNode }> = ({
  label,
  value,
}) => (
  <div className="flex items-center justify-between gap-4">
    <span className="text-muted-foreground">{label}</span>
    <span className="font-mono tabular-nums">{value}</span>
  </div>
);

/**
 * Marks a compaction in the transcript: a short labelled rule, with the detail behind the label.
 *
 * Deliberately NOT a message: it renders from `metadata.custom.contextTruncation`, so it is
 * never sent to the model, editable, or exported as content, yet unlike a toast it survives a
 * reload.
 *
 * `between` sits in the row slot above the first row the model still saw after the compaction,
 * in the same column as the messages. `inside` opens the reply that compacted, for a compaction
 * with no gap between rows to mark: a tool loop evicting its own exchanges, or a turn saved
 * before its row was recorded.
 *
 * Rendered once per COMPACTION, gated by the caller, not once per compacted turn: a thread past
 * its window refits on every request, so per-turn would mean a divider on every reply forever.
 */
export const CompactionDivider: FC<{
  truncation: ContextTruncation;
  placement: "between" | "inside";
}> = ({ truncation, placement }) => {
  if (!promptWasShortened(truncation)) return null;

  const summarized = truncation.summarized === true;
  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;
  const before = truncation.prompt_tokens_before;
  const after = truncation.prompt_tokens_after;

  return (
    <div
      data-testid="compaction-notice"
      data-placement={placement}
      data-dropped={dropped}
      data-archived={archived}
      data-recalled={recalled}
      className={cn(
        "flex w-full items-center gap-3 text-muted-foreground text-sm",
        placement === "between"
          ? // Same column as the messages it sits between: it is their sibling, not their child.
            "mx-auto mb-6 max-w-(--thread-content-max-width)"
          : "mb-3",
      )}
    >
      <span aria-hidden={true} className="h-px flex-1 bg-border" />
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            data-slot="compaction-notice-label"
            className="inline-flex shrink-0 cursor-default items-center gap-1.5 whitespace-nowrap rounded-sm leading-none transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2"
          >
            <ArchiveIcon className="size-3.5" aria-hidden={true} />
            Context compacted
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" sideOffset={8} variant="rich">
          <div className="grid max-w-72 gap-1.5 text-xs">
            <p className="leading-snug">
              {summarized
                ? "The provider summarized older messages to make room."
                : "Older messages were dropped from the model's context to make room."}{" "}
              {archived
                ? "They are saved and searchable, and the parts relevant to each question are brought back automatically."
                : "The full conversation is still visible and saved here."}
            </p>
            <DetailRow
              label={summarized ? "Messages summarized" : "Messages dropped"}
              value={dropped}
            />
            {recalled > 0 && (
              <DetailRow label="Passages recalled" value={recalled} />
            )}
            {before !== undefined && after !== undefined && (
              <DetailRow
                label="Prompt tokens"
                value={`${before.toLocaleString()} → ${after.toLocaleString()}`}
              />
            )}
          </div>
        </TooltipContent>
      </Tooltip>
      <span aria-hidden={true} className="h-px flex-1 bg-border" />
    </div>
  );
};
