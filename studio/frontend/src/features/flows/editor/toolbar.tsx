// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { HugeiconsIcon } from "@hugeicons/react";
import { useReactFlow } from "@xyflow/react";
import { memo } from "react";
import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import { type TidyDirection, useUiStore } from "../stores/ui-store";
import { FIT_MIN_ZOOM } from "./canvas-rules";
import {
  ArrowDown01Icon,
  GroupItemsIcon,
  HierarchySquare02Icon,
  MagnetIcon,
  Redo02Icon,
  Route01Icon,
  StickyNote01Icon,
  Undo02Icon,
} from "./icon-data";
import { autoLayout } from "./layout";
import {
  PLACE_GROUP,
  PLACE_NOTE,
  togglePlacement,
  usePlacement,
} from "./placement";
import { REROUTE_TYPE } from "./reroutes";
import { groupSelection } from "./use-editor-shortcuts";

const SAVE_LABEL: Record<string, string> = {
  saved: "Saved",
  dirty: "Unsaved",
  saving: "Saving…",
  error: "Save failed",
  conflict: "Conflict",
};

const TIDY_LABEL = { LR: "left to right", TB: "top to bottom" } as const;

/** Tidy up, in the last direction picked; the chevron picks the other one. */
function TidyButton({ disabled }: { disabled: boolean }) {
  const { getInternalNode, fitView } = useReactFlow();
  const direction = useUiStore((s) => s.tidyDirection);
  const tidy = (next: TidyDirection) => {
    useUiStore.getState().setTidyDirection(next);
    void autoLayout(getInternalNode, next)
      .then(() =>
        setTimeout(
          () =>
            void fitView({ padding: 0.2, duration: 300, minZoom: FIT_MIN_ZOOM }),
          50,
        ),
      )
      .catch(() => toast.error("Layout failed"));
  };
  const icon = (dir: TidyDirection) => (
    <HugeiconsIcon
      icon={HierarchySquare02Icon}
      size={16}
      strokeWidth={1.8}
      className={dir === "LR" ? "-rotate-90" : undefined}
    />
  );
  return (
    <span className="flex items-center" data-testid="flows-tidy">
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label={`Tidy up layout, ${TIDY_LABEL[direction]}`}
        title={`Tidy up layout, ${TIDY_LABEL[direction]}`}
        disabled={disabled}
        onClick={() => tidy(direction)}
        className="text-muted-foreground"
      >
        {icon(direction)}
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Tidy up direction"
            title="Tidy up direction"
            disabled={disabled}
            className="-ml-1.5 w-4 text-muted-foreground"
          >
            <HugeiconsIcon icon={ArrowDown01Icon} size={12} strokeWidth={2} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="center">
          {(["LR", "TB"] as const).map((dir) => (
            <DropdownMenuItem
              key={dir}
              onSelect={() => tidy(dir)}
              data-testid={`flows-tidy-${dir}`}
              className={dir === direction ? "font-medium" : undefined}
            >
              {icon(dir)}
              Tidy {TIDY_LABEL[dir]}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  );
}

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
          ? "bg-sidebar-accent text-foreground shadow-sm"
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
  const armed = usePlacement((s) => s.armed);
  const snap = useUiStore((s) => s.snapToGrid);
  const wireStyle = useUiStore((s) => s.wireStyle);

  return (
    <div className="grid h-12 shrink-0 grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 bg-sidebar px-2 text-foreground">
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
          className="h-6 shrink-0 bg-background text-ui-10 capitalize"
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

      {/* Inverted against the sidebar-coloured bar: the pill takes the canvas colour, a pressed item the bar's. */}
      <div className="flex h-9 items-center gap-0.5 rounded-full bg-background p-[3px]">
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
              className={`h-8 rounded-full px-2.5 text-xs capitalize transition ${wireStyle === style ? "bg-sidebar-accent text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              data-testid={`flows-wire-style-${style}`}
            >
              {style}
            </button>
          ))}
        </fieldset>
        <span className="mx-1 h-4 w-px bg-border" />
        <TidyButton disabled={readOnly} />
        <IconButton
          icon={GroupItemsIcon}
          label="Add group (Shift+G groups the selection)"
          disabled={readOnly}
          active={armed === PLACE_GROUP}
          onClick={() => {
            if (useUiStore.getState().selectedNodes.length) groupSelection();
            else togglePlacement(PLACE_GROUP);
          }}
        />
        <IconButton
          icon={StickyNote01Icon}
          label="Add note"
          disabled={readOnly}
          active={armed === PLACE_NOTE}
          onClick={() => togglePlacement(PLACE_NOTE)}
        />
        <IconButton
          icon={Route01Icon}
          label="Add reroute (drop it on a wire to split it)"
          disabled={readOnly}
          active={armed === REROUTE_TYPE}
          onClick={() => togglePlacement(REROUTE_TYPE)}
        />
      </div>
    </div>
  );
}

export const Toolbar = memo(ToolbarImpl);
