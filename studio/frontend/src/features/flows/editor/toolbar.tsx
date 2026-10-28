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
  outline,
}: {
  icon: typeof PlayIcon;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  outline?: boolean;
}) {
  return (
    <Button
      type="button"
      variant={outline ? "outline" : "ghost"}
      size="icon-sm"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className={
        active
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground"
      }
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
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const running =
    runStatus === "running" ||
    runStatus === "queued" ||
    runStatus === "cancelling";
  const { getInternalNode, getViewport, fitView } = useReactFlow();
  const [dialog, setDialog] = useState<"run" | "plan" | "share" | null>(null);

  return (
    <div className="grid h-12 shrink-0 grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 border-border border-b bg-background px-2 text-foreground">
      <div className="flex min-w-0 items-center gap-2">
        <IconButton icon={ArrowLeft01Icon} label="All flows" onClick={onBack} />
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
          disabled={!canUndo}
        />
        <IconButton
          icon={Redo02Icon}
          label="Redo (Ctrl+Shift+Z)"
          onClick={() => useDocStore.getState().redo()}
          disabled={!canRedo}
        />
        <span className="mx-1 h-4 w-px bg-border" />
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
      </div>

      <div className="flex items-center justify-self-end gap-2">
        <IconButton
          icon={InformationCircleIcon}
          label="What this flow needs"
          onClick={() => setDialog("plan")}
          outline={true}
        />
        <IconButton
          icon={Share08Icon}
          label="Share"
          outline={true}
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
