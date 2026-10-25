// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import {
  DownloadIcon,
  Move3dIcon,
  Redo2Icon,
  Rotate3dIcon,
  Scale3dIcon,
  Undo2Icon,
} from "lucide-react";
import { useRef } from "react";

import { cn } from "@/lib/utils";

import {
  type EditorState,
  type GizmoMode,
  PROP_KINDS,
  PROP_LABELS,
  type WorldEditor,
} from "./props3d";

const MODES: [GizmoMode, string, typeof Move3dIcon, string][] = [
  ["translate", "Move", Move3dIcon, "G"],
  ["rotate", "Rotate", Rotate3dIcon, "R"],
  ["scale", "Scale", Scale3dIcon, "S"],
];

export interface WorldEditorPanelProps {
  editor: WorldEditor;
  state: EditorState;
  status: string;
  error: string | null;
  canSave: boolean;
  canImport: boolean;
  dirty: boolean;
  onImport(file: File): void;
  onExport(): void;
  onSave(): void;
  onDone(): void;
}

const btn =
  "flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-white/80 hover:bg-white/15 hover:text-white disabled:opacity-35 disabled:hover:bg-transparent";

export default function WorldEditorPanel({
  editor,
  state,
  status,
  error,
  canSave,
  canImport,
  dirty,
  onImport,
  onExport,
  onSave,
  onDone,
}: WorldEditorPanelProps) {
  const file = useRef<HTMLInputElement>(null);
  const sel = state.props.find((p) => p.id === state.selected) ?? null;
  return (
    <div
      className="absolute top-16 right-4 bottom-4 z-20 flex w-72 flex-col gap-3 overflow-hidden rounded-2xl bg-zinc-950/85 p-3 text-white shadow-2xl ring-1 ring-white/10 backdrop-blur-md"
      data-testid="world-editor"
    >
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">Edit scene</h3>
        <span
          className="text-[11px] text-white/50"
          data-testid="world-editor-status"
        >
          {status}
        </span>
      </div>

      <div>
        <p className="mb-1.5 text-[11px] text-white/45 uppercase tracking-wide">
          Add
        </p>
        <div className="grid grid-cols-3 gap-1">
          {PROP_KINDS.map((k) => (
            <button
              key={k}
              type="button"
              className={cn(btn, "justify-center bg-white/5")}
              data-testid={`world-editor-add-${k}`}
              onClick={() => editor.add(k as Exclude<typeof k, "model">)}
            >
              {PROP_LABELS[k]}
            </button>
          ))}
        </div>
        <input
          ref={file}
          type="file"
          accept=".glb,model/gltf-binary"
          className="hidden"
          data-testid="world-editor-file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) onImport(f);
          }}
        />
        <button
          type="button"
          disabled={!canImport}
          className={cn(btn, "mt-1 w-full justify-center bg-white/5")}
          title={
            canImport
              ? "A .glb model, up to 20 MB"
              : "This player cannot store models"
          }
          onClick={() => file.current?.click()}
        >
          Import .glb model
        </button>
      </div>

      <div className="flex items-center gap-1">
        {MODES.map(([m, label, Icon, key]) => (
          <button
            key={m}
            type="button"
            title={`${label} (${key})`}
            aria-pressed={state.mode === m}
            className={cn(btn, state.mode === m && "bg-white/20 text-white")}
            onClick={() => editor.setMode(m)}
          >
            <Icon className="size-3.5" />
            {label}
          </button>
        ))}
        <div className="ml-auto flex">
          <button
            type="button"
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
            disabled={!state.canUndo}
            className={btn}
            onClick={() => editor.undo()}
          >
            <Undo2Icon className="size-3.5" />
          </button>
          <button
            type="button"
            title="Redo (Ctrl+Shift+Z)"
            aria-label="Redo"
            disabled={!state.canRedo}
            className={btn}
            onClick={() => editor.redo()}
          >
            <Redo2Icon className="size-3.5" />
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto rounded-lg bg-white/5 p-1">
        {state.props.length === 0 ? (
          <p className="p-2 text-white/45 text-xs">
            Nothing placed yet. Add a prop, then drag the arrows to move it.
          </p>
        ) : (
          state.props.map((p) => (
            <button
              key={p.id}
              type="button"
              data-testid="world-editor-item"
              onClick={() => editor.select(p.id)}
              className={cn(
                "flex w-full items-center justify-between rounded-md px-2 py-1 text-left text-xs hover:bg-white/10",
                p.id === state.selected && "bg-white/20",
              )}
            >
              <span className="truncate">{p.name || PROP_LABELS[p.kind]}</span>
              <span className="text-white/40">
                {Math.floor(p.x)},{Math.floor(p.y)}
              </span>
            </button>
          ))
        )}
      </div>

      {sel ? (
        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1.5 text-white/75">
            <input
              type="checkbox"
              checked={sel.solid !== false}
              onChange={(e) =>
                editor.update(sel.id, { solid: e.target.checked })
              }
            />
            Blocks walking
          </label>
          <button
            type="button"
            className={cn(btn, "ml-auto text-red-300")}
            title="Delete (X)"
            onClick={() => editor.remove(sel.id)}
          >
            Delete
          </button>
        </div>
      ) : null}

      {error ? (
        <p
          className="rounded-md bg-red-500/15 px-2 py-1.5 text-red-200 text-xs"
          role="alert"
        >
          {error}
        </p>
      ) : null}

      <div className="flex items-center gap-1">
        <button
          type="button"
          className={btn}
          onClick={onExport}
          title="Download the scene as a .glb for Blender"
        >
          <DownloadIcon className="size-3.5" /> Export .glb
        </button>
        <div className="ml-auto flex gap-1">
          {canSave ? (
            <button
              type="button"
              disabled={!dirty}
              className={cn(btn, "bg-white/10")}
              data-testid="world-editor-save"
              onClick={onSave}
            >
              Save
            </button>
          ) : null}
          <button
            type="button"
            className={cn(
              btn,
              "bg-violet-500/70 text-white hover:bg-violet-500",
            )}
            data-testid="world-editor-done"
            onClick={onDone}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
