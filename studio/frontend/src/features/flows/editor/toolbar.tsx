// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { useReactFlow } from "@xyflow/react";
import { memo } from "react";
import { toast } from "sonner";
import { newId, useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import { FIT_MIN_ZOOM } from "./canvas-rules";
import {
  GridViewIcon,
  HierarchySquare02Icon,
  MagnetIcon,
  Redo02Icon,
  StickyNote01Icon,
  Undo02Icon,
} from "./icon-data";
import { autoLayout } from "./layout";
import { groupSelection } from "./use-editor-shortcuts";

const SAVE_LABEL: Record<string, string> = {
  saved: "Saved",
  dirty: "Unsaved",
  saving: "Saving…",
  error: "Save failed",
  conflict: "Conflict",
};

function IconButton({
  icon,
  label,
  onClick,
  disabled,
  active,
}: {
  icon: typeof MagnetIcon;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className={
        active
          ? "bg-background text-foreground shadow-sm dark:bg-input/30"
          : "text-muted-foreground"
      }
    >
      <HugeiconsIcon icon={icon} size={16} strokeWidth={1.8} />
    </Button>
  );
}

function ToolbarImpl({ readOnly = false }: { readOnly?: boolean }) {
  const name = useDocStore((s) => s.meta.name);
  const kind = useDocStore((s) => s.meta.kind);
  const canUndo = useDocStore((s) => s.past.length > 0);
  const canRedo = useDocStore((s) => s.future.length > 0);
  const saveState = useUiStore((s) => s.saveState);
  const snap = useUiStore((s) => s.snapToGrid);
  const wireStyle = useUiStore((s) => s.wireStyle);
  const { getInternalNode, getViewport, fitView } = useReactFlow();

  return (
    <div className="grid h-12 shrink-0 grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 border-border border-b bg-background px-2 text-foreground">
      <div className="flex min-w-0 items-center gap-2">
        {/* Sized to the name: the hidden copy sets the width, the input fills it. */}
        <div className="inline-grid min-w-0 max-w-[min(22rem,50vw)] font-semibold text-sm">
          <span
            aria-hidden={true}
            className="invisible col-start-1 row-start-1 truncate whitespace-pre border border-transparent px-2"
          >
            {name || " "}
          </span>
          <input
            value={name}
            readOnly={readOnly}
            onChange={(e) =>
              useDocStore
                .getState()
                .setMeta({ name: e.target.value }, "meta:name")
            }
            onBlur={(e) => {
              if (!e.target.value.trim())
                useDocStore.getState().setMeta({ name: "Untitled flow" });
            }}
            className="col-start-1 row-start-1 h-7 w-full min-w-0 truncate rounded-md border border-transparent bg-transparent px-2 text-foreground outline-none transition hover:border-border focus-visible:border-ring"
            aria-label="Flow name"
            data-testid="flows-name"
          />
        </div>
        <Badge
          variant="secondary"
          className="h-6 shrink-0 text-ui-10 capitalize"
        >
          {kind}
        </Badge>
        <span
          className={`shrink-0 text-xs ${saveState === "error" || saveState === "conflict" ? "text-destructive" : "text-muted-foreground"}`}
          data-testid="flows-save-state"
        >
          {SAVE_LABEL[saveState]}
        </span>
      </div>

      <div className="flex h-9 items-center gap-0.5 rounded-full bg-muted p-[3px]">
        <IconButton
          icon={Undo02Icon}
          label="Undo (Ctrl+Z)"
          onClick={() => useDocStore.getState().undo()}
          disabled={readOnly || !canUndo}
        />
        <IconButton
          icon={Redo02Icon}
          label="Redo (Ctrl+Shift+Z)"
          onClick={() => useDocStore.getState().redo()}
          disabled={readOnly || !canRedo}
        />
        <span className="mx-1 h-4 w-px bg-border" />
        <IconButton
          icon={MagnetIcon}
          label="Snap to grid"
          onClick={() => useUiStore.getState().toggleSnap()}
          active={snap}
        />
        <fieldset
          className="flex items-center"
          title="Wire style: Bus routes wires around nodes in lanes; Curved draws direct curves"
          data-testid="flows-wire-style"
        >
          <legend className="sr-only">Wire style</legend>
          {(["bus", "curved"] as const).map((style) => (
            <button
              key={style}
              type="button"
              aria-pressed={wireStyle === style}
              onClick={() => useUiStore.getState().setWireStyle(style)}
              className={`h-8 rounded-full px-2.5 text-xs capitalize transition ${wireStyle === style ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              data-testid={`flows-wire-style-${style}`}
            >
              {style}
            </button>
          ))}
        </fieldset>
        <span className="mx-1 h-4 w-px bg-border" />
        <IconButton
          icon={HierarchySquare02Icon}
          label="Tidy up layout"
          disabled={readOnly}
          onClick={() => {
            void autoLayout(getInternalNode, kind === "story" ? "LR" : "LR")
              .then(() =>
                setTimeout(
                  () =>
                    void fitView({
                      padding: 0.2,
                      duration: 300,
                      minZoom: FIT_MIN_ZOOM,
                    }),
                  50,
                ),
              )
              .catch(() => toast.error("Layout failed"));
          }}
        />
        <IconButton
          icon={GridViewIcon}
          label="Group selection (Shift+G)"
          disabled={readOnly}
          onClick={() => {
            const vp = getViewport();
            groupSelection([
              -vp.x / vp.zoom + 80,
              -vp.y / vp.zoom + 80,
              520,
              340,
            ]);
          }}
        />
        <IconButton
          icon={StickyNote01Icon}
          label="Add note"
          disabled={readOnly}
          onClick={() => {
            const vp = getViewport();
            useDocStore.getState().upsertNote({
              id: newId("note"),
              text: "",
              rect: [
                Math.round(-vp.x / vp.zoom + 120),
                Math.round(-vp.y / vp.zoom + 120),
                240,
                140,
              ],
            });
          }}
        />
      </div>

    </div>
  );
}

export const Toolbar = memo(ToolbarImpl);
