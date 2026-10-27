// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type NodeProps, NodeResizer } from "@xyflow/react";
import { Suspense, lazy, memo, useState } from "react";
import { groupGenerate } from "../recipe/recipe-columns";
import { noteRecipeGroup, withRecipeNotes } from "../recipe/recipe-notes";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { RfNode } from "./derive-rf";

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
      className="h-full w-full rounded-2xl border-2"
      style={{ borderColor: `${color}88`, background: `${color}12` }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={200}
        minHeight={120}
        lineClassName="!border-white/30"
        handleClassName="!h-2.5 !w-2.5 !rounded-sm !border-white/60 !bg-zinc-800"
        onResizeEnd={(_, p) =>
          resizeGroup({
            ...group,
            rect: [p.x, p.y, Math.round(p.width), Math.round(p.height)],
          })
        }
      />
      <div
        className="flow-group-drag flex cursor-grab items-center gap-2 rounded-t-2xl px-3 py-1.5 active:cursor-grabbing"
        style={{ background: `${color}33` }}
        onDoubleClick={() => setEditing(true)}
      >
        {editing ? (
          <input
            // biome-ignore lint/a11y/noAutofocus: rename starts on purpose
            autoFocus={true}
            defaultValue={group.label}
            className="nodrag w-full bg-transparent text-sm font-semibold outline-none"
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
          <span className="truncate text-sm font-semibold" style={{ color }}>
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

const MarkdownPreview = lazy(() =>
  import("@/components/markdown/markdown-preview").then((m) => ({
    default: m.MarkdownPreview,
  })),
);

function NoteFrameImpl({ data, selected }: NodeProps<RfNode>) {
  const noteId = (data as { noteId: string }).noteId;
  const note = useDocStore((s) => s.graph.notes[noteId]);
  const upsertNote = useDocStore((s) => s.upsertNote);
  // A note from a Data Recipe reads as markdown, like Recipe Studio, and edits go back into the recipe.
  const fromRecipe = useDocStore((s) => !!noteRecipeGroup(s.graph, noteId));
  const [editing, setEditing] = useState(false);
  if (!note) return null;
  const write = (text: string) => {
    const next = { ...note, text };
    const { graph, patchGraph } = useDocStore.getState();
    const group = fromRecipe ? noteRecipeGroup(graph, note.id) : null;
    if (!group) return upsertNote(next, `note:${note.id}`);
    const notes = graph.noteOrder.map((id) =>
      id === note.id ? next : graph.notes[id],
    );
    patchGraph(
      { notes: [next], groups: [withRecipeNotes(group, notes)] },
      `note:${note.id}`,
    );
  };
  if (fromRecipe && !editing) {
    return (
      <div
        className="h-full w-full overflow-hidden rounded-lg border border-amber-300/30 bg-amber-200/10 shadow-md"
        style={
          note.color
            ? { borderColor: `${note.color}55`, background: `${note.color}14` }
            : undefined
        }
        onDoubleClick={(e) => {
          e.stopPropagation();
          setEditing(true);
        }}
        title="Double-click to edit"
      >
        <NodeResizer
          isVisible={selected}
          minWidth={140}
          minHeight={60}
          lineClassName="!border-amber-200/30"
          handleClassName="!h-2 !w-2 !rounded-sm !border-amber-200/60 !bg-zinc-800"
          onResizeEnd={(_, p) =>
            upsertNote({
              ...note,
              rect: [p.x, p.y, Math.round(p.width), Math.round(p.height)],
            })
          }
        />
        <Suspense
          fallback={
            <p className="whitespace-pre-wrap p-2 text-[12px] text-zinc-300">
              {note.text}
            </p>
          }
        >
          <MarkdownPreview
            markdown={note.text}
            plain={true}
            className="text-zinc-200 [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-[13px]"
          />
        </Suspense>
      </div>
    );
  }
  return (
    <div className="h-full w-full rounded-lg border border-amber-300/30 bg-amber-200/10 p-2 shadow-md">
      <NodeResizer
        isVisible={selected}
        minWidth={140}
        minHeight={60}
        lineClassName="!border-amber-200/30"
        handleClassName="!h-2 !w-2 !rounded-sm !border-amber-200/60 !bg-zinc-800"
        onResizeEnd={(_, p) =>
          upsertNote({
            ...note,
            rect: [p.x, p.y, Math.round(p.width), Math.round(p.height)],
          })
        }
      />
      <textarea
        className="nodrag nowheel h-full w-full resize-none bg-transparent text-[12px] leading-snug text-amber-100 outline-none placeholder:text-amber-100/40"
        placeholder="Write a note…"
        value={note.text}
        // biome-ignore lint/a11y/noAutofocus: editing starts on double-click
        autoFocus={editing}
        onBlur={() => setEditing(false)}
        onChange={(e) => write(e.target.value)}
      />
    </div>
  );
}

export const GroupFrame = memo(GroupFrameImpl);
export const NoteFrame = memo(NoteFrameImpl);
