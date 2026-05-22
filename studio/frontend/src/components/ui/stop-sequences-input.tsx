// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import { type KeyboardEvent, useState } from "react";

/**
 * Chips editor for the `stop` / `stop_sequences` array.
 *
 * Commit a chip with Enter or comma; press Backspace on an empty input
 * to delete the most recent chip. Caps at `maxEntries` -- OpenAI Chat
 * Completions documents a hard cap of 4 stop sequences; Anthropic
 * Messages accepts arbitrarily many. Pass `Infinity` to disable the cap.
 */
export interface StopSequencesInputProps {
  value: string[];
  onChange: (next: string[]) => void;
  maxEntries?: number;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  "aria-label"?: string;
}

export function StopSequencesInput({
  value,
  onChange,
  maxEntries = 4,
  disabled,
  placeholder = "Add stop sequence",
  className,
  "aria-label": ariaLabel,
}: StopSequencesInputProps) {
  const [draft, setDraft] = useState("");
  const atCap = value.length >= maxEntries;

  function commitDraft() {
    const trimmed = draft.trim();
    if (!trimmed) return;
    if (atCap) return;
    if (value.includes(trimmed)) {
      setDraft("");
      return;
    }
    onChange([...value, trimmed]);
    setDraft("");
  }

  function removeChip(index: number) {
    if (disabled) return;
    onChange(value.filter((_, i) => i !== index));
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (disabled) return;
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      commitDraft();
      return;
    }
    if (event.key === "Backspace" && !draft && value.length > 0) {
      event.preventDefault();
      onChange(value.slice(0, -1));
    }
  }

  return (
    <div
      data-slot="stop-sequences-input"
      className={cn(
        "flex flex-wrap items-center gap-1.5 rounded-md border border-input bg-transparent px-2 py-1.5 text-sm",
        "focus-within:border-ring focus-within:ring-[1px] focus-within:ring-ring/40",
        disabled && "cursor-not-allowed opacity-60",
        className,
      )}
      aria-label={ariaLabel}
    >
      {value.map((entry, index) => (
        <Badge
          // Stop sequences are not guaranteed unique across edits (the
          // user could enter "END" twice if the previous one was just
          // deleted), so combine value + index for a stable key.
          key={`${entry}-${index}`}
          variant="secondary"
          className="gap-1 pl-2 pr-1"
        >
          <span className="font-mono">{entry}</span>
          {!disabled ? (
            <button
              type="button"
              onClick={() => removeChip(index)}
              className="ml-0.5 rounded-full p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label={`Remove stop sequence ${entry}`}
            >
              <XIcon className="size-3" />
            </button>
          ) : null}
        </Badge>
      ))}
      <Input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={commitDraft}
        placeholder={atCap ? `Max ${maxEntries} stops` : placeholder}
        disabled={disabled || atCap}
        aria-label={ariaLabel || placeholder}
        className={cn(
          "h-6 min-w-[8ch] flex-1 border-0 bg-transparent p-0 text-sm shadow-none",
          "focus-visible:ring-0 focus-visible:border-0",
        )}
      />
    </div>
  );
}
