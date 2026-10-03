// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Edit's model tools: how the loaded model applies the changes, and the exact runtime requests
// they become, from the same builder the run uses.

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef } from "react";
import { type EditAdapter, buildEditRun } from "../edit-adapters";
import { countChanges } from "../edit-diff";
import { EDIT_COPY } from "../edit-policy";
import { EDIT_PANEL_LOGIC, type EditPanelLogic } from "./edit-panel-logic";
import type { AudioToolPanel, CoreInputs } from "./types";

export const EDIT_PREVIEW_OPEN_KEY = "unsloth_audio_edit_preview_open";

function EditHowPanel({
  adapter,
  core,
}: {
  adapter: EditAdapter;
  core?: CoreInputs;
}) {
  const [open, setOpen] = usePersistedToggle(EDIT_PREVIEW_OPEN_KEY);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const edit = core?.edit;
  const { calls, changes } = useMemo(() => {
    if (!edit) return { calls: [], changes: 0 };
    const run = buildEditRun(adapter, {
      transcript: edit.transcript,
      edited: edit.edited,
      mode: edit.mode,
      delivery: { speed: edit.speed, pitchSteps: edit.pitchSteps },
      advanced: edit.advanced,
    });
    const calls = adapter.preview(run);
    const changes =
      run.edit?.mode === "delivery"
        ? calls.length
        : (countChanges(edit.transcript, edit.edited) ?? 0);
    return { calls, changes };
  }, [adapter, edit]);

  return (
    <section className="grid gap-3" aria-label={EDIT_COPY.howItEditsTitle}>
      <div className="grid gap-0.5">
        <h3 className="text-ui-13 font-medium text-foreground">
          {EDIT_COPY.howItEditsTitle}
        </h3>
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {adapter.howItEdits(changes)}
        </p>
      </div>
      <Collapsible
        ref={previewRef}
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          // The rail's footer covers its bottom edge, so bring the opened preview to the middle.
          if (next)
            window.requestAnimationFrame(() =>
              previewRef.current?.scrollIntoView({ block: "center" }),
            );
        }}
      >
        <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-full px-3 py-1.5 text-left text-ui-12p5 font-medium text-foreground transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none">
          <span className="min-w-0 flex-1">{EDIT_COPY.previewTitle}</span>
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            className={cn(
              "size-3.5 shrink-0 text-muted-foreground transition-transform duration-150 motion-reduce:transition-none",
              open && "rotate-180",
            )}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="motion-reduce:animate-none">
          <div className="grid gap-2 pt-2">
            {calls.length === 0 ? (
              <p className="px-3 text-ui-11p5 leading-snug text-muted-foreground">
                Make a change in ② to see the request.
              </p>
            ) : (
              calls.map((call, index) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: calls are positional; call N edits call N-1's output.
                <div key={index} className="grid gap-1">
                  <p className="px-1 text-ui-11p5 text-muted-foreground">
                    Call {index + 1} of {calls.length} ·{" "}
                    <span className="font-mono">POST {call.path}</span>
                  </p>
                  <pre className="max-h-[calc(240px*var(--ui-space-scale,1))] overflow-auto whitespace-pre-wrap rounded-xl bg-muted/50 px-3 py-2 font-mono text-ui-11 leading-snug text-foreground [overflow-wrap:anywhere]">
                    {JSON.stringify(call.body, null, 2)}
                  </pre>
                </div>
              ))
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </section>
  );
}

function editPanel(logic: EditPanelLogic): AudioToolPanel<null> {
  const { adapter } = logic;
  return {
    ...logic,
    Component: ({ core }) => <EditHowPanel adapter={adapter} core={core} />,
  };
}

/** Edit's panels, one per edit family. Each model matches at most one. */
export const EDIT_TOOL_PANELS: readonly AudioToolPanel<null>[] =
  EDIT_PANEL_LOGIC.map(editPanel);
