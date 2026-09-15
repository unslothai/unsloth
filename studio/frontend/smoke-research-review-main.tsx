// SPDX-License-Identifier: AGPL-3.0-only
// Real message and plan dialog; no backend or model required.
import {
  AssistantRuntimeProvider,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
} from "@assistant-ui/react";
import { createRoot } from "react-dom/client";
/* eslint-disable no-restricted-imports -- Regression harness drives the real research UI/store. */
import { ResearchActivityPanel } from "@/features/chat";
import { ResearchMessage } from "@/features/chat/components/research-message";
import {
  ingestResearchUpdate,
  useResearchRunStore,
} from "@/features/chat/stores/research-run-store";
import type { ResearchRun } from "@/features/chat/types/research";
/* eslint-enable no-restricted-imports */
import "./src/index.css";

const run: ResearchRun = {
  id: "review-run",
  threadId: "review-thread",
  userMessageId: "question",
  assistantMessageId: "answer",
  status: "awaiting_approval",
  plan: {
    title: "Research plan",
    steps: [{ title: "Search", query: "topic" }],
  },
  planRevision: 1,
  planHash: "plan-hash",
  steps: [],
  sources: [],
  lastEventSeq: 0,
  createdAt: 1,
  updatedAt: 1,
};
ingestResearchUpdate(run);
// Model the existing follower without starting a backend connection.
useResearchRunStore.getState().setFollowing(run.id, true, "connected");
useResearchRunStore.getState().openPanel(run.id);

const adapter = {
  async run() {
    throw new Error("No model calls expected");
  },
};

function Message() {
  return (
    <MessagePrimitive.Root data-testid="research-message">
      <ResearchMessage />
    </MessagePrimitive.Root>
  );
}

function Harness() {
  const runtime = useLocalRuntime(adapter, {
    initialMessages: [
      {
        id: "answer",
        role: "assistant",
        content: [{ type: "text", text: "Research" }],
        metadata: { custom: { researchRunId: run.id, researchRun: run } },
      },
    ],
  });
  const openRunId = useResearchRunStore((state) => state.openRunId);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div style={{ display: "flex", height: "100vh", gap: 24 }}>
        <ThreadPrimitive.Root>
          <ThreadPrimitive.Messages
            components={{ Message }}
          />
        </ThreadPrimitive.Root>
        <div style={{ width: 420 }}>
          {openRunId && (
            <ResearchActivityPanel
              runId={openRunId}
              onClose={() => useResearchRunStore.getState().closePanel()}
            />
          )}
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}

Object.assign(window, {
  __review: {
    state: () => useResearchRunStore.getState(),
    closePanel: () => useResearchRunStore.getState().closePanel(),
    setError: () =>
      useResearchRunStore
        .getState()
        .setConnectionError(run.id, "Existing connection error"),
    setDraft: () => {
      useResearchRunStore.getState().setPlanReviewEditing(run.id, true);
      useResearchRunStore.getState().setPlanReviewDraft(run.id, {
        title: "Unsaved edits",
        steps: [{ title: "Edited step", query: "edited" }],
      });
    },
    complete: () =>
      ingestResearchUpdate({
        ...run,
        status: "completed",
        report: "Finished report",
        updatedAt: 2,
      }),
  },
});
createRoot(document.getElementById("root")!).render(<Harness />);
