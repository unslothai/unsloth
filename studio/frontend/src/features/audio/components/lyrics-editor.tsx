// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useRef } from "react";
import { insertSectionTag, sectionTags } from "../music/music-policy";
import type { MusicModeRule } from "../music/music-types";

export function LyricsEditor({
  id,
  value,
  onChange,
  sectionCase,
  disabled,
  placeholder,
  describedBy,
}: {
  id: string;
  value: string;
  onChange: (next: string) => void;
  sectionCase: MusicModeRule["section_case"];
  disabled: boolean;
  placeholder?: string;
  describedBy?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const tags = sectionTags(sectionCase);
  const insert = (tag: string) => {
    const textarea = textareaRef.current;
    const cursor = textarea ? textarea.selectionStart : value.length;
    const next = insertSectionTag(value, cursor, tag);
    onChange(next.text);
    // Move the caret once React has written the value.
    requestAnimationFrame(() => {
      const current = textareaRef.current;
      if (!current) return;
      current.focus();
      current.setSelectionRange(next.cursor, next.cursor);
    });
  };
  return (
    <div className="grid gap-1.5">
      <div
        role="toolbar"
        aria-label="Insert a section"
        className="flex flex-wrap gap-1"
      >
        {tags.map((section) => (
          <Button
            key={section.tag}
            type="button"
            variant="muted"
            size="sm"
            disabled={disabled}
            className="h-[calc(24px*var(--ui-space-scale,1))] px-2.5 text-ui-11"
            onClick={() => insert(section.tag)}
            aria-label={`Insert ${section.name} section`}
          >
            {section.tag}
          </Button>
        ))}
      </div>
      <Textarea
        ref={textareaRef}
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        aria-describedby={describedBy}
        className="min-h-36 font-normal"
      />
    </div>
  );
}
