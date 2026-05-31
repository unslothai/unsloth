// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { MarkdownNoteConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type MarkdownNoteDialogProps = {
  config: MarkdownNoteConfig;
  onUpdate: (patch: Partial<MarkdownNoteConfig>) => void;
};

export function MarkdownNoteDialog({
  config,
  onUpdate,
}: MarkdownNoteDialogProps): ReactElement {
  const markdownId = `${config.id}-markdown`;
  const colorId = `${config.id}-note-color`;
  const opacity =
    Number.parseInt(config.note_opacity ?? "35", 10) > 0
      ? Math.max(0, Math.min(100, Number.parseInt(config.note_opacity ?? "35", 10)))
      : 35;

  return (
    <div className="space-y-4">
      <NameField value={config.name} onChange={(value) => onUpdate({ name: value })} />
      <div className="grid gap-3">
        <FieldLabel
          label="Note style"
          htmlFor={colorId}
          hint="Pick a color and opacity for this note block."
        />
        <div className="flex items-center gap-3">
          <input
            id={colorId}
            type="color"
            className="nodrag h-9 w-14 cursor-pointer rounded-md border border-border/60 bg-transparent p-1"
            value={config.note_color ?? "#FDE68A"}
            onChange={(event) => onUpdate({ note_color: event.target.value })}
          />
          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Opacity</span>
              <span className="text-xs tabular-nums text-muted-foreground">{opacity}%</span>
            </div>
            <Slider
              min={5}
              max={100}
              step={1}
              value={[opacity]}
              onValueChange={([value]) =>
                onUpdate({ note_opacity: String(Math.round(value)) })
              }
            />
          </div>
        </div>
      </div>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Markdown"
          htmlFor={markdownId}
          hint="UI-only note. Not sent to backend payload recipe."
        />
        <Textarea
          id={markdownId}
          className="corner-squircle nodrag min-h-[180px]"
          placeholder="## Note"
          value={config.markdown}
          onChange={(event) => onUpdate({ markdown: event.target.value })}
        />
      </div>
    </div>
  );
}
