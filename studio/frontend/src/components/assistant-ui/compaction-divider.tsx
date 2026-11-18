// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { ArchiveIcon, Clock01Icon } from "@hugeicons/core-free-icons";
import type { FC, ReactNode } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  COMPACTION_NOTICE_LABEL,
  COMPACTION_NOTICE_TEXT,
  type CompactionNoticeKind,
  type ContextTruncation,
  compactionNoticeKind,
  promptWasShortened,
} from "@/features/chat";
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
    | { truncation: ContextTruncation; pending?: false; onCancel?: undefined }
    | { truncation?: undefined; pending: true; onCancel?: () => void }
  ) & { className?: string; /** A cloud connection summarizes instead of starting a checkpoint. */ cloud?: boolean }
> = ({ truncation, pending, onCancel, className, cloud }) => {
  // Queued, not done: plain text and a dashed rule, so it never reads as a finished compaction.
  if (pending) {
    return (
      <DividerRule
        className={className}
        pending={true}
        label={cloud ? "Summarizes on next message" : "Compacts on next message"}
        action={
          onCancel ? (
            <button
              type="button"
              data-slot="compaction-divider-cancel"
              onClick={onCancel}
              aria-label={cloud ? "Cancel summary" : "Cancel compaction"}
              className="relative shrink-0 rounded-full px-1.5 py-1 leading-none underline-offset-2 transition-colors hover:text-foreground hover:underline focus-visible:outline-2 focus-visible:outline-ring"
            >
              Cancel
            </button>
          ) : null
        }
      />
    );
  }
  if (!promptWasShortened(truncation)) return null;

  const kind = compactionNoticeKind(truncation);
  const summarized = truncation.summarized === true;
  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;
  const before = truncation.prompt_tokens_before;
  const after = truncation.prompt_tokens_after;

  return (
    <DividerRule
      className={className}
      label={COMPACTION_NOTICE_LABEL[kind]}
      kind={kind}
    >
      <div className="grid max-w-72 gap-1.5 text-xs">
        <p className="leading-snug">
          {COMPACTION_NOTICE_TEXT[kind]}{" "}
          {archived
            ? kind === "summarized"
              ? "Their exact wording is saved and searchable."
              : "They are saved and searchable, and the parts relevant to each question are brought back automatically."
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

/** A finished compaction keeps its details on hover and focus; a queued one is plain text. */
const DividerRule: FC<{
  children?: ReactNode;
  className?: string;
  label?: string;
  kind?: CompactionNoticeKind;
  pending?: boolean;
  action?: ReactNode;
}> = ({
  children,
  className = "mb-6",
  label = "Context compacted",
  kind,
  pending = false,
  action,
}) => {
  const rule = pending
    ? "h-0 flex-1 border-border border-t border-dashed"
    : "h-px flex-1 bg-border";
  return (
    <div
      data-testid="compaction-divider"
      data-compaction-kind={kind}
      data-compaction-pending={pending ? "true" : undefined}
      className={cn(
        "mx-auto flex w-full max-w-(--thread-content-max-width) items-center gap-3 text-muted-foreground text-sm",
        className,
      )}
    >
      <span aria-hidden={true} className={rule} />
      <span className="inline-flex shrink-0 items-center gap-1">
        {pending ? (
          <span
            data-slot="compaction-divider-label"
            className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap leading-none"
          >
            <HugeiconsIcon icon={Clock01Icon} className="size-3.5" aria-hidden={true} />
            {label}
          </span>
        ) : (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                data-slot="compaction-divider-label"
                // Negative margins keep the rule put; relative stops the reply covering the target.
                className="relative -mx-2 -my-2.5 inline-flex shrink-0 cursor-default items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-2.5 leading-none transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
              >
                <HugeiconsIcon icon={ArchiveIcon} className="size-3.5" aria-hidden={true} />
                {label}
              </button>
            </TooltipTrigger>
            <TooltipContent side="top" sideOffset={8} variant="rich">
              {children}
            </TooltipContent>
          </Tooltip>
        )}
        {action ? (
          <>
            <span aria-hidden={true} className="text-muted-foreground/60">
              ·
            </span>
            {action}
          </>
        ) : null}
      </span>
      <span aria-hidden={true} className={rule} />
    </div>
  );
};
