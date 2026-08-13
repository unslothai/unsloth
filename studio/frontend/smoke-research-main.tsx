// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for the #8483 deep research freeze, driven by tests/studio/playwright_research_freeze.py.
// Real activity panel and report renderer against the real store, so the measured main-thread cost
// of a streaming run is the app's, not a mock's.
// Same shape as smoke-ansi.html/smoke-ansi-main.tsx: a vite entry, no backend, no auth.

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { ResearchActivityPanel } from "@/features/chat";
/* eslint-disable no-restricted-imports -- a measurement entry point, not app code: it drives the
   research store directly, which the chat barrel does not export. */
import {
  ingestResearchUpdate,
  useResearchRunStore,
} from "@/features/chat/stores/research-run-store";
import type {
  ResearchEvent,
  ResearchEventType,
  ResearchRun,
} from "@/features/chat/types/research";
/* eslint-enable no-restricted-imports */
import { type ReactElement, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const RUN_ID = "smoke-run";
const THREAD_ID = "smoke-thread";

function baseRun(): ResearchRun {
  return {
    id: RUN_ID,
    threadId: THREAD_ID,
    userMessageId: "smoke-user-message",
    status: "running",
    plan: null,
    planRevision: 1,
    planHash: null,
    steps: [],
    sources: [],
    lastEventSeq: 0,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

let currentRun = baseRun();
let seq = 0;

function push(
  event: ResearchEventType,
  data: Record<string, unknown>,
  runPatch?: Partial<ResearchRun>,
): void {
  seq += 1;
  // Always a fresh run object, as research-api.ts did for deltas before #8483. Held constant on
  // purpose: this harness measures the panel and report renderer, not run identity, which
  // tests/research-run-identity.test.ts covers against the real followResearchRun.
  const run: ResearchRun = {
    ...currentRun,
    ...runPatch,
    lastEventSeq: seq,
    updatedAt: Date.now(),
  };
  currentRun = run;
  const payload: ResearchEvent = {
    id: seq,
    event,
    createdAt: Date.now(),
    data: { ...data, run } as ResearchEvent["data"],
    run,
  };
  ingestResearchUpdate(run, payload);
}

function Harness(): ReactElement {
  // Mounted by seed(), not at start: the panel's scroll hook is a useLayoutEffect keyed on runId,
  // so rendering with no session takes the loading branch, observes no viewport, and never re-runs.
  // The app only opens the panel for a run it already holds; the harness must do the same.
  const [panelMounted, setPanelMounted] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [clicks, setClicks] = useState(0);

  useEffect(() => {
    const api = {
      /** Seed a run the panel can render. */
      seed(): void {
        currentRun = baseRun();
        seq = 0;
        push("run.created", {}, { status: "running" });
        push("run.started", {}, { status: "running" });
        setPanelMounted(true);
      },
      /** One reasoning delta, the event the synthesis phase emits ~12x a second. */
      delta(text: string, phase = "synthesis"): void {
        push("reasoning.updated", {
          callId: "call-1",
          attempt: 0,
          phase,
          reasoningDelta: text,
        });
      },
      /** One report-progress delta. */
      reportDelta(length: number): void {
        push("report.updated", { attempt: 0, length, delta: 32 });
      },
      /** A search step plus its sources: what grows the activity list row by row. */
      step(position: number): void {
        push("step.started", {
          attempt: 0,
          stepPosition: position,
          action: "search",
          title: `Searching the web (${position})`,
          input: `query ${position}`,
        });
        for (let index = 0; index < 4; index += 1) {
          push("source.added", {
            attempt: 0,
            stepPosition: position,
            url: `https://example.invalid/${position}/${index}`,
            title: `Source ${position}.${index}`,
            snippet: "A snippet long enough to wrap onto a second line in the panel.",
          });
        }
        push("step.completed", {
          attempt: 0,
          stepPosition: position,
          sourceCount: 4,
        });
      },
      /** A plan awaiting approval, which mounts PlanReview's modal Dialog. */
      awaitApproval(): void {
        push(
          "plan.ready",
          {},
          {
            status: "awaiting_approval",
            plan: {
              title: "Smoke plan",
              steps: [
                { title: "Step one", query: "first query" },
                { title: "Step two", query: "second query" },
              ],
            },
          },
        );
      },
      /** Approve it: the status change unmounts PlanReview while it is open. */
      approve(): void {
        push("run.approved", {}, { status: "queued" });
      },
      /** Unmount the whole panel, as closing the research pane does. */
      closePanel(): void {
        setPanelMounted(false);
      },
      openPanel(): void {
        setPanelMounted(true);
      },
      /** Publish a finished report through the real renderer. */
      publishReport(markdown: string): void {
        setReport(markdown);
      },
      clearReport(): void {
        setReport(null);
      },
      /** What the store holds, for assertions about activity count and status. */
      state(): { activities: number; status: string | undefined } {
        const session = useResearchRunStore.getState().sessions[RUN_ID];
        return {
          activities: session?.activities.length ?? 0,
          status: session?.run.status,
        };
      },
      clicks(): number {
        return clicksRef.current;
      },
    };
    (window as unknown as { __research: typeof api }).__research = api;
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", gap: "8px" }}>
      <div style={{ width: "420px", height: "100%", position: "relative" }}>
        {panelMounted ? (
          <ResearchActivityPanel
            runId={RUN_ID}
            onClose={() => setPanelMounted(false)}
          />
        ) : null}
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: "12px" }}>
        {/* The click probe: a stranded body pointer-events:none stops this counting up. */}
        <button
          type="button"
          data-smoke="click-probe"
          onClick={() => {
            clicksRef.current += 1;
            setClicks((value) => value + 1);
          }}
        >
          clicked {clicks}
        </button>
        <section data-smoke="report">
          {report === null ? null : <MarkdownPreview markdown={report} />}
        </section>
      </div>
    </div>
  );
}

const clicksRef = { current: 0 };

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}
createRoot(root).render(<Harness />);
