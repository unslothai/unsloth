// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { type Ref, useMemo } from "react";
import { EDIT_DIFF_MAX_WORDS, countChanges, diffSegments } from "../edit-diff";
import { EDIT_COPY } from "../edit-policy";

export function TranscriptDiffEditor({
  value,
  onChange,
  transcript,
  disabled = false,
  onReset,
  id,
  textareaRef,
  dataTour,
}: {
  value: string;
  onChange: (value: string) => void;
  transcript: string;
  disabled?: boolean;
  onReset: () => void;
  id: string;
  textareaRef?: Ref<HTMLTextAreaElement>;
  dataTour?: string;
}) {
  const segments = useMemo(
    () => diffSegments(transcript, value),
    [transcript, value],
  );
  const changes = useMemo(
    () => countChanges(transcript, value),
    [transcript, value],
  );
  const hintId = `${id}-hint`;
  const diffId = `${id}-diff`;

  return (
    <div className="grid gap-1.5" data-tour={dataTour}>
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <label
          htmlFor={id}
          className="whitespace-nowrap text-ui-13 font-medium text-foreground"
        >
          {EDIT_COPY.changesLabel}
        </label>
        <div className="flex shrink-0 items-center gap-1.5">
          {changes === null ? (
            <Badge variant="outline">Too long</Badge>
          ) : changes > 0 ? (
            <Badge variant="outline">
              {changes === 1 ? "1 change" : `${changes} changes`}
            </Badge>
          ) : null}
          <Button
            type="button"
            variant="ghost"
            size="xs"
            disabled={disabled || value === transcript}
            onClick={onReset}
          >
            {EDIT_COPY.resetLabel}
          </Button>
        </div>
      </div>
      <Textarea
        id={id}
        ref={textareaRef}
        value={value}
        disabled={disabled}
        aria-describedby={`${hintId} ${diffId}`}
        onChange={(event) => onChange(event.target.value)}
        className="min-h-16"
      />
      {segments && changes ? (
        <p
          id={diffId}
          aria-label="Changes from the transcript"
          className="rounded-xl bg-muted/50 px-3 py-2 text-ui-13 leading-relaxed text-foreground [overflow-wrap:anywhere]"
        >
          {segments.map((segment, index) => {
            return (
              // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional and rebuilt on every change.
              <span key={index}>
                {index > 0 ? " " : ""}
                {segment.kind === "insert" ? (
                  <ins className="rounded-sm bg-secondary px-0.5 text-secondary-foreground underline decoration-2 underline-offset-2">
                    <span className="sr-only">{segment.srLabel}</span>
                    {segment.text}
                  </ins>
                ) : segment.kind === "delete" ? (
                  <del className="text-muted-foreground line-through">
                    <span className="sr-only">{segment.srLabel}</span>
                    {segment.text}
                  </del>
                ) : (
                  segment.text
                )}
              </span>
            );
          })}
        </p>
      ) : (
        <span id={diffId} hidden={true} />
      )}
      <p
        id={hintId}
        className="min-w-0 text-ui-11p5 leading-snug text-muted-foreground"
      >
        {changes === null
          ? `Diff shown for up to ${EDIT_DIFF_MAX_WORDS} words.`
          : changes === 0
            ? EDIT_COPY.changesHint
            : "Underlined words are new, struck-through words are removed."}
      </p>
    </div>
  );
}
