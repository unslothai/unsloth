// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mount the real view against an in-memory queue; no model or backend is needed.
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "motion/react";
import { PromptQueueList } from "@/components/assistant-ui/lazy-prompt-queue-list";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  type PromptQueueUIItem,
  reorderPromptQueueItems,
  useChatPreferencesStore,
} from "@/features/chat";
import "./src/index.css";

const queueSize = Math.max(
  3,
  Math.min(500, Number(new URLSearchParams(location.search).get("size")) || 3),
);
const initialItems: PromptQueueUIItem[] = Array.from(
  { length: queueSize },
  (_, index) => ({
    id: `q${index}`,
    runId: "smoke",
    prompt:
      ["First prompt", "Second prompt", "Third prompt"][index] ??
      `Prompt ${index + 1}`,
    position: index + 1,
    total: queueSize,
    status: "queued",
    threadIds: ["smoke"],
    canEdit: true,
    canRemove: true,
  }),
);

function App() {
  const followUpBehavior = useChatPreferencesStore((s) => s.followUpBehavior);
  const [items, setItems] = useState(initialItems);
  const [paused, setPaused] = useState(false);
  const [rejectMoves, setRejectMoves] = useState(false);
  const [steered, setSteered] = useState("");
  const [moveAttempts, setMoveAttempts] = useState(0);
  const [draft, setDraft] = useState("Unsent composer draft");
  const [submissions, setSubmissions] = useState(0);
  return (
    <TooltipProvider>
      <div style={{ padding: 16 }}>
        <button
          type="button"
          onClick={() => {
            setItems(initialItems);
            setPaused(false);
            setRejectMoves(false);
            setSteered("");
            setMoveAttempts(0);
            setDraft("Unsent composer draft");
            setSubmissions(0);
            useChatPreferencesStore.getState().setFollowUpBehavior("queue");
          }}
        >
          Reset fixture
        </button>
        <button type="button" onClick={() => setRejectMoves(true)}>
          Simulate dispatch race
        </button>
        <button type="button" onClick={() => setPaused(true)}>
          Simulate paused queue
        </button>
        <button type="button" onClick={() => setItems((old) => old.slice(1))}>
          Dispatch first
        </button>
        <button
          type="button"
          onClick={() => {
            setMoveAttempts(0);
            setItems(
              Array.from({ length: 12 }, (_, index) => ({
                ...initialItems[0],
                id: `long-${index}`,
                prompt: `Queued prompt ${index + 1}`,
                position: index + 1,
                total: 12,
              })),
            );
          }}
        >
          Long queue
        </button>
        <button
          type="button"
          onClick={() =>
            setItems((old) =>
              old.map((item) => ({
                ...item,
                canEdit: false,
                canRemove: false,
              })),
            )
          }
        >
          Lock prompts
        </button>
        <output aria-label="Steered prompt">{steered}</output>
        <output aria-label="Follow-up behavior">{followUpBehavior}</output>
        <output aria-label="Move attempts">{moveAttempts}</output>
      </div>
      <form
        style={{ maxWidth: 850, margin: "180px auto 0" }}
        onSubmit={(event) => {
          event.preventDefault();
          setSubmissions((count) => count + 1);
          setDraft("");
        }}
      >
        <PromptQueueList
          entry={{
            runId: "smoke",
            current: 1,
            total: items.length,
            local: true,
            temporary: false,
            dispatched: false,
            paused,
          }}
          items={items}
          onEdit={(id, prompt) => {
            setItems((old) =>
              old.map((item) =>
                item.id === id ? { ...item, prompt: prompt.trim() } : item,
              ),
            );
            return true;
          }}
          onRemove={(id) => {
            setItems((old) => old.filter((item) => item.id !== id));
            return true;
          }}
          onMove={(id, targetId) => {
            setMoveAttempts((count) => count + 1);
            if (rejectMoves) return false;
            const reordered = reorderPromptQueueItems(
              items,
              items.findIndex((item) => item.id === id),
              items.findIndex((item) => item.id === targetId),
            );
            if (!reordered) return false;
            setItems(reordered);
            return true;
          }}
          onSteer={(id) => {
            if (rejectMoves) return false;
            const item = items.find((item) => item.id === id);
            if (!item) return false;
            setSteered(item.prompt);
            setItems((old) => old.filter((item) => item.id !== id));
            setPaused(false);
            return true;
          }}
          onResume={() => setPaused(false)}
        />
        <textarea
          aria-label="Composer draft"
          value={draft}
          onChange={(event) => setDraft(event.currentTarget.value)}
        />
        <output aria-label="Composer submissions">{submissions}</output>
      </form>
    </TooltipProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </StrictMode>,
);
