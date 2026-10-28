// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  BaseNode,
  BaseNodeContent,
} from "@/features/recipe-studio/components/rf-ui/base-node";
import { type NodeProps, NodeResizer } from "@xyflow/react";
import { Suspense, lazy, memo, useState } from "react";
import { groupGenerate } from "../recipe/recipe-columns";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { RfNode } from "./derive-rf";

const MarkdownPreview = lazy(() =>
  import("@/components/markdown/markdown-preview").then((m) => ({
    default: m.MarkdownPreview,
  })),
);

// Recipe Studio's resizer: invisible corners, the card border is the frame.
const RESIZER = {
  color: "var(--primary)",
  lineClassName: "!border-transparent !shadow-none",
  lineStyle: { opacity: 0 },
  handleClassName: "!h-3 !w-3 !border-transparent !bg-transparent",
  handleStyle: { opacity: 0 },
} as const;

/** Mixes a hex into the frame surface, so tints read on both themes. */
function tint(color: string, percent: number): string {
  return `color-mix(in oklab, ${color} ${percent}%, transparent)`;
}

function GroupFrameImpl({ data, selected }: NodeProps<RfNode>) {
  const groupId = (data as { groupId: string }).groupId;
  const group = useDocStore((s) => s.graph.groups[groupId]);
  const upsertGroup = useDocStore((s) => s.upsertGroup);
  const resizeGroup = useDocStore((s) => s.resizeGroup);
  const [editing, setEditing] = useState(false);
  if (!group) return null;
  const color = group.color ?? "#8b5cf6";
  return (
    <div
      className="corner-squircle h-full w-full rounded-2xl border shadow-sm"
      style={{
        borderColor: tint(color, selected ? 70 : 40),
        background: tint(color, 6),
      }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={200}
        minHeight={120}
        {...RESIZER}
        onResizeEnd={(_, p) =>
          resizeGroup({
            ...group,
            rect: [p.x, p.y, Math.round(p.width), Math.round(p.height)],
          })
        }
      />
      <div
        className="flow-group-drag flex cursor-grab items-center gap-2 border-b px-3 py-2 active:cursor-grabbing"
        style={{ borderColor: tint(color, 25) }}
        onDoubleClick={() => setEditing(true)}
      >
        <span
          className="size-2.5 shrink-0 rounded-full"
          style={{ background: color }}
        />
        {editing ? (
          <Input
            // biome-ignore lint/a11y/noAutofocus: rename starts on purpose
            autoFocus={true}
            defaultValue={group.label}
            className="nodrag h-7 font-semibold"
            onBlur={(e) => {
              setEditing(false);
              if (e.target.value !== group.label)
                upsertGroup({ ...group, label: e.target.value || "Group" });
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === "Escape")
                (e.target as HTMLInputElement).blur();
            }}
          />
        ) : (
          <span className="truncate text-sm font-semibold text-foreground">
            {group.label}
          </span>
        )}
        {group.kind === "recipe" && (
          <button
            type="button"
            className="nodrag ml-auto shrink-0 rounded px-1.5 py-0.5 text-[11px] hover:bg-white/10"
            style={{ color, background: `${color}22` }}
            title="Models, seed, rows and columns of this recipe"
            onClick={() => {
              const step = groupGenerate(useDocStore.getState().graph, group.id);
              if (!step) return;
              useUiStore.getState().setSelection([step.id], []);
              useUiStore.getState().openInspector(step.id);
            }}
          >
            Data Recipe
          </button>
        )}
      </div>
    </div>
  );
}

/** Recipe Studio's markdown note: rendered when idle, double-click to edit. */
function NoteFrameImpl({ data, selected }: NodeProps<RfNode>) {
  const noteId = (data as { noteId: string }).noteId;
  const note = useDocStore((s) => s.graph.notes[noteId]);
  const upsertNote = useDocStore((s) => s.upsertNote);
  const [editing, setEditing] = useState(false);
  if (!note) return null;
  const color = note.color || "#FDE68A";
  const showEditor = editing || !note.text;
  return (
    <BaseNode
      className="corner-squircle h-full w-full min-w-0 overflow-hidden rounded-xl border-border/60 shadow-sm"
      style={{ background: tint(color, 35), borderColor: tint(color, 55) }}
      onDoubleClick={(e) => {
        e.stopPropagation();
        setEditing(true);
      }}
      title={showEditor ? undefined : "Double-click to edit"}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={140}
        minHeight={60}
        {...RESIZER}
        onResizeEnd={(_, p) =>
          upsertNote({
            ...note,
            rect: [p.x, p.y, Math.round(p.width), Math.round(p.height)],
          })
        }
      />
      {showEditor ? (
        <Textarea
          fieldSizing="fixed"
          // biome-ignore lint/a11y/noAutofocus: editing starts on double-click
          autoFocus={editing}
          className="nodrag nowheel h-full min-h-0 w-full resize-none rounded-xl border-0 bg-transparent text-xs leading-snug shadow-none focus-visible:ring-0 dark:bg-transparent"
          placeholder="Write a note…"
          value={note.text}
          onBlur={() => setEditing(false)}
          onChange={(e) =>
            upsertNote({ ...note, text: e.target.value }, `note:${note.id}`)
          }
        />
      ) : (
        <BaseNodeContent className="nowheel h-full overflow-y-auto px-3 py-2">
          <Suspense
            fallback={
              <p className="whitespace-pre-wrap text-xs">{note.text}</p>
            }
          >
            <MarkdownPreview markdown={note.text} plain={true} />
          </Suspense>
        </BaseNodeContent>
      )}
    </BaseNode>
  );
}

export const GroupFrame = memo(GroupFrameImpl);
export const NoteFrame = memo(NoteFrameImpl);
