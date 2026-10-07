// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_research_freeze.py; real panel and store, no backend.

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
  // Fresh run object each time on purpose; run identity is covered by research-run-identity.test.ts.
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
  // Mounted by seed(): the scroll hook keys on runId and never re-runs from the loading branch.
  const [panelMounted, setPanelMounted] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [clicks, setClicks] = useState(0);

  useEffect(() => {
    const api = {
      seed(): void {
        currentRun = baseRun();
        seq = 0;
        push("run.created", {}, { status: "running" });
        push("run.started", {}, { status: "running" });
        setPanelMounted(true);
      },
      delta(text: string, phase = "synthesis"): void {
        push("reasoning.updated", {
          callId: "call-1",
          attempt: 0,
          phase,
          reasoningDelta: text,
        });
      },
      reportDelta(length: number): void {
        push("report.updated", { attempt: 0, length, delta: 32 });
      },
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
      approve(): void {
        push("run.approved", {}, { status: "queued" });
      },
      closePanel(): void {
        setPanelMounted(false);
      },
      openPanel(): void {
        setPanelMounted(true);
      },
      publishReport(markdown: string): void {
        setReport(markdown);
      },
      clearReport(): void {
        setReport(null);
      },
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
        {/* A stranded body pointer-events:none stops this counting up. */}
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
