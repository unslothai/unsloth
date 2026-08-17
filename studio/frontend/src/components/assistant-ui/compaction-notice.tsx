// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArchiveIcon } from "lucide-react";
import type { FC } from "react";

import type { ContextTruncation } from "@/features/chat/utils/context-truncation";

/**
 * Tells the user their conversation was compacted, on the turn where it STARTED.
 *
 * Deliberately NOT a message. It renders inside the assistant message's own container from
 * `metadata.custom.contextTruncation`, so it is never part of the conversation sent to the
 * model, never editable, and never exported as content -- but unlike the toast that used to
 * be the only signal, it survives a reload and stays attached to the turn it describes. A
 * toast that vanishes after eight seconds leaves the user with no way to find out why the
 * model suddenly seemed to forget the start of a long chat.
 *
 * Rendered once per COMPACTION, gated by the caller, not once per compacted turn. A
 * thread that has outgrown its window runs the fit on every request from then on, so a
 * notice per compacted turn is a notice on every reply forever. The caller shows this
 * only when the eviction boundary actually moved, which with the server's compaction
 * headroom happens in occasional steps rather than on every turn.
 */
export const CompactionNotice: FC<{ truncation: ContextTruncation }> = ({
  truncation,
}) => {
  if (!truncation?.fits || !truncation.dropped_messages) return null;

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
          Older messages were dropped from the model&apos;s context to make room. {detail}
        </span>
        <span>
          {" "}
          ({dropped} {dropped === 1 ? "message" : "messages"} dropped here
          {recalled > 0
            ? `, ${recalled} earlier ${recalled === 1 ? "passage" : "passages"} recalled`
            : ""}
          .)
        </span>
      </div>
    </div>
  );
};
