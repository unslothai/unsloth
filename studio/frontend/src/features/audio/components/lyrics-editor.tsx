// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Textarea } from "@/components/ui/textarea";
import { Add01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef } from "react";
import { insertSectionTag, sectionTags } from "../music/music-policy";
import type { MusicModeRule } from "../music/music-types";

export function LyricsEditor({
  id,
  label,
  value,
  onChange,
  sectionCase,
  disabled,
  placeholder,
  describedBy,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (next: string) => void;
  sectionCase: MusicModeRule["section_case"];
  disabled: boolean;
  placeholder?: string;
  describedBy?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const inserted = useRef(false);
  const tags = sectionTags(sectionCase);
  const insert = (tag: string) => {
    const textarea = textareaRef.current;
    const cursor = textarea ? textarea.selectionStart : value.length;
    const next = insertSectionTag(value, cursor, tag);
    inserted.current = true;
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
      <div className="flex min-w-0 items-center justify-between gap-2">
        <label htmlFor={id} className="text-ui-13 font-medium text-foreground">
          {label}
        </label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={disabled}
              className="-me-2 h-auto px-2 py-1 text-ui-11p5"
            >
              <HugeiconsIcon icon={Add01Icon} className="size-3" />
              Add section
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            className="w-auto min-w-40"
            onCloseAutoFocus={(event) => {
              // After a pick the caret goes back into the lyrics, not to this button.
              if (inserted.current) event.preventDefault();
              inserted.current = false;
            }}
          >
            {tags.map((section) => (
              <DropdownMenuItem
                key={section.tag}
                onSelect={() => insert(section.tag)}
              >
                {section.name}
                <span className="ms-auto text-muted-foreground">
                  {section.tag}
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
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
