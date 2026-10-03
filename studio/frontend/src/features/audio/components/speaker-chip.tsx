// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useId, useState } from "react";
import {
  SPEAKER_NAME_MAX_LENGTH,
  type TranscriptSpeaker,
  sanitizeSpeakerName,
  speakerIndex,
  speakerLabel,
} from "../transcript-model";

const SPEAKER_COLORS = 5;

// The name always accompanies the colour, never the colour alone.
export function SpeakerChip({
  id,
  speakers,
  names,
  onRename,
}: {
  id: string;
  speakers: readonly TranscriptSpeaker[];
  names: Readonly<Record<string, string>>;
  onRename: (id: string, name: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const inputId = useId();
  const label = speakerLabel(id, speakers, names);
  const color = ((speakerIndex(id, speakers) - 1) % SPEAKER_COLORS) + 1;
  const save = () => {
    onRename(id, sanitizeSpeakerName(draft));
    setOpen(false);
  };
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (next) setDraft(names[id] ?? "");
        setOpen(next);
      }}
    >
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={`Rename ${label}`}
          className="inline-flex max-w-full shrink-0 items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-ui-11p5 font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span
            aria-hidden="true"
            className="size-[calc(8px*var(--ui-space-scale,1))] shrink-0 rounded-full"
            style={{ backgroundColor: `var(--chart-${color})` }}
          />
          <span className="min-w-0 truncate">{label}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 gap-2 p-3">
        <label htmlFor={inputId} className="text-ui-13 font-medium">
          Name
        </label>
        <Input
          id={inputId}
          value={draft}
          placeholder={
            speakers.find((speaker) => speaker.id === id)?.label ?? id
          }
          maxLength={SPEAKER_NAME_MAX_LENGTH}
          autoFocus={true}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              save();
            }
          }}
        />
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          Renames every line by this speaker.
        </p>
        <div className="flex justify-end gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setOpen(false)}
          >
            Cancel
          </Button>
          <Button type="button" variant="secondary" size="sm" onClick={save}>
            Save
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
