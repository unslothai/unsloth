// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveIcon } from "lucide-react";
import type { FC } from "react";

import {
  type ContextTruncation,
  promptWasShortened,
} from "@/features/chat/utils/context-truncation";

/**
 * Compaction notice, rendered from message metadata so it is never sent to the model but survives
 * reload. Once per compaction, gated by the caller, not per compacted turn.
 */
export const CompactionNotice: FC<{ truncation: ContextTruncation }> = ({
  truncation,
}) => {
  if (!promptWasShortened(truncation)) return null;

  const dropped = truncation.dropped_messages;
  const archived = truncation.archived_messages ?? 0;
  const recalled = truncation.recalled_chunks ?? 0;

  const detail = archived
    ? "They are saved and searchable, and the parts relevant to each question are brought back automatically."
    : "The full conversation is still visible and saved here.";

  return (
    <div
      className="aui-compaction-notice mb-3 flex items-start gap-2 rounded-lg border border-border/60 bg-muted/40 px-3 py-2 text-ui-13 text-muted-foreground"
      data-testid="compaction-notice"
      data-dropped={dropped}
      data-archived={archived}
      data-recalled={recalled}
    >
      <ArchiveIcon className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      <div className="min-w-0">
        <span className="font-medium text-foreground/80">
          This conversation got long, so it was compacted.
        </span>{" "}
        <span>
          {truncation.summarized
            ? "The provider summarized older messages to make room."
            : "Older messages were dropped from the model's context to make room."}{" "}
          {detail}
        </span>
        <span>
          {" "}
          ({dropped} {dropped === 1 ? "message" : "messages"}{" "}
          {truncation.summarized ? "summarized" : "dropped"} here
          {recalled > 0
            ? `, ${recalled} earlier ${recalled === 1 ? "passage" : "passages"} recalled`
            : ""}
          .)
        </span>
      </div>
    </div>
  );
};
