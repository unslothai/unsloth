// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  BaseNode,
  BaseNodeContent,
} from "@/features/recipe-studio/components/rf-ui/base-node";
import { HugeiconsIcon } from "@hugeicons/react";
import { type NodeProps, NodeResizer, useReactFlow } from "@xyflow/react";
import { Suspense, lazy, memo, useState } from "react";
import { useDocStore } from "../stores/doc-store";
import type { RfNode } from "./derive-rf";
import { Delete02Icon } from "./icon-data";
import { portInk, studioInk } from "./theme-colors";

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

/** Mixes a colour into the frame surface, so tints read on both themes. */
function tint(color: string, percent: number): string {
  return `color-mix(in oklab, ${studioInk(color)} ${percent}%, transparent)`;
}

// Studio surfaces: groups and default notes are neutral; the group colour is its dot.
// Studio Card's edge (components/ui/card.tsx), no fill: cards inside read against the canvas.
const GROUP_EDGE =
  "color-mix(in oklab, var(--foreground) calc(10% * var(--contrast-edge-gain, 1)), transparent)";
const NOTE_STYLE = { background: "var(--muted)", borderColor: "var(--border)" };
// Recipe Studio's and the examples' default note yellow reads as "no colour picked".
const DEFAULT_NOTE = "#fde68a";

/** Top-right trash, shown on hover or selection; goes through the canvas delete, so it is one undo step. */
function FrameDelete({
  id,
  label,
  className,
}: {
  id: string;
  label: string;
  className?: string;
}) {
  const { deleteElements } = useReactFlow();
  return (
    <button
      type="button"
      title={`${label} (Delete)`}
      aria-label={label}
      data-testid="frame-delete"
      className={`nodrag flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-destructive focus-visible:opacity-100 group-hover/frame:opacity-100 group-data-[selected]/frame:opacity-100 ${className ?? ""}`}
      onClick={(e) => {
        e.stopPropagation();
        void deleteElements({ nodes: [{ id }] });
      }}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <HugeiconsIcon icon={Delete02Icon} size={14} />
    </button>
  );
}

function GroupFrameImpl({ id, data, selected }: NodeProps<RfNode>) {
  const groupId = (data as { groupId: string }).groupId;
  const group = useDocStore((s) => s.graph.groups[groupId]);
  const upsertGroup = useDocStore((s) => s.upsertGroup);
  const resizeGroup = useDocStore((s) => s.resizeGroup);
  const [editing, setEditing] = useState(false);
  if (!group) return null;
  const color = group.color ?? "#8b5cf6";
  return (
    <div
      className="group/frame corner-squircle h-full w-full rounded-2xl border"
      data-selected={selected || undefined}
      style={{
        borderColor: selected ? "var(--ring-select)" : GROUP_EDGE,
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
        style={{ borderColor: GROUP_EDGE }}
        onDoubleClick={() => setEditing(true)}
      >
        <span
          className="size-2.5 shrink-0 rounded-full"
          style={{ background: portInk(color) }}
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
        {/* Its blocks stay on the canvas; only the frame goes. */}
        <FrameDelete
          id={id}
          label="Delete group (its blocks stay)"
          className="-my-1 ml-auto"
        />
      </div>
    </div>
  );
}

/** Recipe Studio's markdown note: rendered when idle, double-click to edit. */
function NoteFrameImpl({ id, data, selected }: NodeProps<RfNode>) {
  const noteId = (data as { noteId: string }).noteId;
  const note = useDocStore((s) => s.graph.notes[noteId]);
  const upsertNote = useDocStore((s) => s.upsertNote);
  const [editing, setEditing] = useState(false);
  if (!note) return null;
  const color =
    note.color?.toLowerCase() === DEFAULT_NOTE ? undefined : note.color;
  // Editing only after a double-click: an empty note shows its placeholder and still drags.
  const showEditor = editing;
  return (
    <BaseNode
      className="group/frame corner-squircle h-full w-full min-w-0 overflow-hidden rounded-xl border-border/60 shadow-sm"
      data-selected={selected || undefined}
      style={
        color
          ? { background: tint(color, 35), borderColor: tint(color, 55) }
          : NOTE_STYLE
      }
      onDoubleClick={(e) => {
        e.stopPropagation();
        setEditing(true);
      }}
      title={showEditor ? undefined : "Double-click to edit"}
    >
      {/* Always mounted (handles stay invisible): resizing needs no click to select first. */}
      <NodeResizer
        isVisible={true}
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
      {!showEditor && (
        <FrameDelete
          id={id}
          label="Delete note"
          className="absolute top-1 right-1 z-10"
        />
      )}
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
      ) : !note.text ? (
        <p className="px-3 py-2 text-muted-foreground text-xs">
          Double-click to write a note…
        </p>
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
