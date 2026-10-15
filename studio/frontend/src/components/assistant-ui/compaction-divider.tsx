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

/** renders metadata once per compaction where it cut and survives reloads without entering model context, editable content, or exports. */
export const CompactionDivider: FC<
  (
    | { truncation: ContextTruncation; pending?: false }
    | { truncation?: undefined; pending: true }
  ) & { className?: string }
> = ({ truncation, pending, className }) => {
  if (pending) {
    return (
      <DividerRule className={className}>
        <p className="max-w-72 text-xs leading-snug">
          Your next message starts a fresh context. Older messages stay here,
          and the parts relevant to each question are brought back
          automatically.
        </p>
      </DividerRule>
    );
  }
  if (!promptWasShortened(truncation)) return null;

  const summarized = truncation.summarized === true;
  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;
  const before = truncation.prompt_tokens_before;
  const after = truncation.prompt_tokens_after;

  return (
    <DividerRule className={className}>
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
    </DividerRule>
  );
};

const DividerRule: FC<{ children: ReactNode; className?: string }> = ({
  children,
  className = "mb-6",
}) => (
  <div
    data-testid="compaction-divider"
    className={cn(
      "mx-auto flex w-full max-w-(--thread-content-max-width) items-center gap-3 text-muted-foreground text-sm",
      className,
    )}
  >
    <span aria-hidden={true} className="h-px flex-1 bg-border" />
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="compaction-divider-label"
          // negative margins keep a large target from moving the rule.
          // positioning stops the reply above from covering the target.
          className="relative -mx-2 -my-2.5 inline-flex shrink-0 cursor-default items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-2.5 leading-none transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          <ArchiveIcon className="size-3.5" aria-hidden={true} />
          Context compacted
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8} variant="rich">
        {children}
      </TooltipContent>
    </Tooltip>
    <span aria-hidden={true} className="h-px flex-1 bg-border" />
  </div>
);
