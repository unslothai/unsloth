// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { useReactFlow } from "@xyflow/react";
import { memo, useState } from "react";
import { toast } from "sonner";
import { newId, useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import {
  DownloadRunDialog,
  PlanDialog,
  RunDialog,
  ShareDialog,
  collectFlowParams,
} from "./dialogs";
import {
  ArrowLeft01Icon,
  GameController01Icon,
  GridViewIcon,
  HierarchySquare02Icon,
  InformationCircleIcon,
  MagnetIcon,
  PlayIcon,
  Redo02Icon,
  Share08Icon,
  SidebarLeftIcon,
  StickyNote01Icon,
  StopIcon,
  Undo02Icon,
} from "./icon-data";
import { autoLayout } from "./layout";
import { runFlow, stopFlow } from "./run-controller";
import { flushSave } from "./use-autosave";
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
  icon: typeof PlayIcon;
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
      className={active ? "bg-muted text-foreground" : "text-muted-foreground"}
    >
      <HugeiconsIcon icon={icon} size={16} strokeWidth={1.8} />
    </Button>
  );
}

function ToolbarImpl({
  onBack,
  onPlay,
  onApp,
}: { onBack: () => void; onPlay: () => void; onApp?: () => void }) {
  const name = useDocStore((s) => s.meta.name);
  const kind = useDocStore((s) => s.meta.kind);
  const playsChat = useDocStore((s) =>
    s.graph.nodeOrder.some((id) => {
      const n = s.graph.nodes[id];
      return n?.type === "chat.output" && !n.disabled;
    }),
  );
  const canUndo = useDocStore((s) => s.past.length > 0);
  const canRedo = useDocStore((s) => s.future.length > 0);
  const saveState = useUiStore((s) => s.saveState);
  const snap = useUiStore((s) => s.snapToGrid);
  const paletteOpen = useUiStore((s) => s.paletteOpen);
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const running =
    runStatus === "running" ||
    runStatus === "queued" ||
    runStatus === "cancelling";
  const { getInternalNode, getViewport, fitView } = useReactFlow();
  const [dialog, setDialog] = useState<"run" | "plan" | "share" | null>(null);

  return (
    <div className="flex h-12 shrink-0 items-center gap-1 border-border border-b bg-background px-2 text-foreground">
      <IconButton icon={ArrowLeft01Icon} label="All flows" onClick={onBack} />
      <IconButton
        icon={SidebarLeftIcon}
        label="Toggle node palette"
        onClick={() => useUiStore.getState().togglePalette()}
        active={paletteOpen}
      />
      <input
        value={name}
        onChange={(e) =>
          useDocStore.getState().setMeta({ name: e.target.value }, "meta:name")
        }
        onBlur={(e) => {
          if (!e.target.value.trim())
            useDocStore.getState().setMeta({ name: "Untitled flow" });
        }}
        className="ml-1 w-[min(320px,30vw)] rounded-full bg-transparent px-2.5 py-1 font-semibold text-[14px] outline-none hover:bg-muted focus:bg-muted"
        aria-label="Flow name"
        data-testid="flows-name"
      />
      <Badge variant="secondary" className="capitalize">
        {kind}
      </Badge>
      <span
        className={`ml-1 text-[11px] ${saveState === "error" || saveState === "conflict" ? "text-destructive" : "text-muted-foreground"}`}
        data-testid="flows-save-state"
      >
        {SAVE_LABEL[saveState]}
      </span>

      <div className="mx-2 h-5 w-px bg-border" />
      <IconButton
        icon={Undo02Icon}
        label="Undo (Ctrl+Z)"
        onClick={() => useDocStore.getState().undo()}
        disabled={!canUndo}
      />
      <IconButton
        icon={Redo02Icon}
        label="Redo (Ctrl+Shift+Z)"
        onClick={() => useDocStore.getState().redo()}
        disabled={!canRedo}
      />
      <IconButton
        icon={MagnetIcon}
        label="Snap to grid"
        onClick={() => useUiStore.getState().toggleSnap()}
        active={snap}
      />
      <IconButton
        icon={HierarchySquare02Icon}
        label="Tidy up layout"
        onClick={() => {
          void autoLayout(getInternalNode, kind === "story" ? "LR" : "LR")
            .then(() =>
              setTimeout(
                () => void fitView({ padding: 0.2, duration: 300 }),
                50,
              ),
            )
            .catch(() => toast.error("Layout failed"));
        }}
      />
      <IconButton
        icon={GridViewIcon}
        label="Group selection (Shift+G)"
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

      <div className="ml-auto flex items-center gap-1.5">
        <IconButton
          icon={InformationCircleIcon}
          label="What this flow needs"
          onClick={() => setDialog("plan")}
        />
        <IconButton
          icon={Share08Icon}
          label="Share"
          onClick={() =>
            void flushSave().then((saved) => saved && setDialog("share"))
          }
        />
        {onApp && kind === "pipeline" && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onApp}
            title="Open this flow as a simple form with its results"
            data-testid="flows-open-app"
          >
            <HugeiconsIcon icon={GridViewIcon} size={15} /> Open as app
          </Button>
        )}
        {(kind === "story" || kind === "world" || playsChat) && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onPlay}
            data-testid="flows-play"
          >
            <HugeiconsIcon icon={GameController01Icon} size={15} /> Play
          </Button>
        )}
        {running ? (
          <Button
            type="button"
            variant="destructive"
            size="sm"
            onClick={() => void stopFlow()}
            disabled={runStatus === "cancelling"}
            data-testid="flows-stop"
          >
            <HugeiconsIcon icon={StopIcon} size={14} />{" "}
            {runStatus === "cancelling" ? "Stopping…" : "Stop"}
          </Button>
        ) : (
          <Button
            type="button"
            variant="dark"
            size="sm"
            onClick={() =>
              collectFlowParams().length ? setDialog("run") : void runFlow()
            }
            title="Run (Ctrl+Enter)"
            data-testid="flows-run"
          >
            <HugeiconsIcon icon={PlayIcon} size={14} /> Run
          </Button>
        )}
      </div>
      <RunDialog
        open={dialog === "run"}
        onOpenChange={(open) => setDialog(open ? "run" : null)}
      />
      <PlanDialog
        open={dialog === "plan"}
        onOpenChange={(open) => setDialog(open ? "plan" : null)}
      />
      <ShareDialog
        open={dialog === "share"}
        onOpenChange={(open) => setDialog(open ? "share" : null)}
      />
      <DownloadRunDialog />
    </div>
  );
}

export const Toolbar = memo(ToolbarImpl);
