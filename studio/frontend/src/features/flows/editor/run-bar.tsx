// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useState } from "react";
import { useDocStore } from "../stores/doc-store";
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
  GameController01Icon,
  GridViewIcon,
  InformationCircleIcon,
  MoreHorizontalIcon,
  PlayIcon,
  Share08Icon,
  StopIcon,
} from "./icon-data";
import { runFlow, stopFlow } from "./run-controller";
import { flushSave } from "./use-autosave";

// Clears the run drawer (h-65) under the canvas column.
const DRAWER_CLEARANCE = "calc(16.25rem + 0.75rem)";

/** Run and the flow's other actions, floating bottom centre the way Recipe Studio's
 *  RunValidateFloatingControls do, so the window corners stay clear. */
function RunBarImpl({
  onPlay,
  onApp,
}: { onPlay: () => void; onApp?: () => void }) {
  const kind = useDocStore((s) => s.meta.kind);
  const playsChat = useDocStore((s) =>
    s.graph.nodeOrder.some((id) => {
      const n = s.graph.nodes[id];
      return n?.type === "chat.output" && !n.disabled;
    }),
  );
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const drawerOpen = useUiStore((s) => s.drawerOpen);
  const running =
    runStatus === "running" ||
    runStatus === "queued" ||
    runStatus === "cancelling";
  const [dialog, setDialog] = useState<"run" | "plan" | "share" | null>(null);

  return (
    <div
      className="pointer-events-none absolute inset-x-0 z-20 flex justify-center"
      style={{ bottom: drawerOpen ? DRAWER_CLEARANCE : "0.75rem" }}
    >
      <div
        className="pointer-events-auto flex items-center gap-2"
        data-testid="flows-run-bar"
      >
        {running ? (
          <Button
            type="button"
            variant="destructive"
            className="corner-squircle"
            onClick={() => void stopFlow()}
            disabled={runStatus === "cancelling"}
            data-testid="flows-stop"
          >
            <HugeiconsIcon icon={StopIcon} className="size-4" />
            {runStatus === "cancelling" ? "Stopping…" : "Stop"}
          </Button>
        ) : (
          <Button
            type="button"
            variant="dark"
            className="corner-squircle"
            onClick={() =>
              collectFlowParams().length ? setDialog("run") : void runFlow()
            }
            title="Run (Ctrl+Enter)"
            data-testid="flows-run"
          >
            <HugeiconsIcon icon={PlayIcon} className="size-4" />
            Run
          </Button>
        )}
        {(kind === "story" || kind === "world" || playsChat) && (
          <Button
            type="button"
            variant="outline"
            className="corner-squircle"
            onClick={onPlay}
            data-testid="flows-play"
          >
            <HugeiconsIcon icon={GameController01Icon} className="size-4" />
            Play
          </Button>
        )}
        {onApp && kind === "pipeline" && (
          <Button
            type="button"
            variant="outline"
            className="corner-squircle"
            onClick={onApp}
            title="Open this flow as a simple form with its results"
            data-testid="flows-open-app"
          >
            <HugeiconsIcon icon={GridViewIcon} className="size-4" />
            Open as app
          </Button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="corner-squircle"
              aria-label="More actions"
              title="More actions"
              data-testid="flows-more"
            >
              <HugeiconsIcon icon={MoreHorizontalIcon} className="size-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="center">
            <DropdownMenuItem
              onSelect={() => setDialog("plan")}
              data-testid="flows-plan"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-4" />
              What this flow needs
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() =>
                void flushSave().then((saved) => saved && setDialog("share"))
              }
              data-testid="flows-share"
            >
              <HugeiconsIcon icon={Share08Icon} className="size-4" />
              Share
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        {!drawerOpen && runStatus && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => useUiStore.getState().setDrawer(true, "run")}
            className="corner-squircle capitalize"
          >
            Run: {runStatus}
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

export const RunBar = memo(RunBarImpl);
