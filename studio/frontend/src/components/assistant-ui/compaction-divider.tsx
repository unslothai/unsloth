// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveIcon } from "lucide-react";
import type { FC, ReactNode } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { type ContextTruncation, promptWasShortened } from "@/features/chat";

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
 * reload. Drawn in the row slot above the prompt whose request compacted (see
 * `compactionDividerRows`), in the same column as the messages it sits between.
 *
 * Rendered once per COMPACTION, gated by the caller, not once per compacted turn: a thread past
 * its window refits on every request, so per-turn would mean a divider on every reply forever.
 */
export const CompactionDivider: FC<{ truncation: ContextTruncation }> = ({
  truncation,
}) => {
  if (!promptWasShortened(truncation)) return null;

  const summarized = truncation.summarized === true;
  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;
  const before = truncation.prompt_tokens_before;
  const after = truncation.prompt_tokens_after;

  return (
    <div
      data-testid="compaction-divider"
      className="mx-auto mb-6 flex w-full max-w-(--thread-content-max-width) items-center gap-3 text-muted-foreground text-sm"
    >
      <span aria-hidden={true} className="h-px flex-1 bg-border" />
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            data-slot="compaction-divider-label"
            // Padding cancelled by negative margin: a finger-sized target without moving the rule.
            // Positioned, or the reply above (itself positioned) takes the top of the target.
            className="relative -mx-2 -my-2.5 inline-flex shrink-0 cursor-default items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-2.5 leading-none transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
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
