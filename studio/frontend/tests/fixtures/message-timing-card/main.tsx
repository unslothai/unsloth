// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ActionBarPrimitive,
  AssistantRuntimeProvider,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { createRoot } from "react-dom/client";
import { MessageTiming } from "@/components/assistant-ui/message-timing";
import { useActionBarFocusReveal } from "@/components/assistant-ui/use-action-bar-focus-reveal";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./styles.css";

const reply = (id: string, promptMs: number): ThreadMessageLike => ({
  id,
  role: "assistant",
  content: `Reply ${id}`,
  status: { type: "complete", reason: "stop" },
  metadata: {
    timing: { totalStreamTime: 1234, firstTokenTime: 120 },
    custom: {
      serverTimings: {
        prompt_n: 457,
        prompt_ms: promptMs,
        prompt_per_second: 1000,
        predicted_n: 50,
        predicted_ms: 1000,
        predicted_per_second: 50,
      },
      stopReason: "stop",
    },
  },
});

const MESSAGES: ThreadMessageLike[] = [
  { id: "u1", role: "user", content: "hi" },
  reply("older", 457),
  { id: "u2", role: "user", content: "again" },
  reply("newest", 458),
];

function Assistant() {
  const reveal = useActionBarFocusReveal();
  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root block"
      tabIndex={0}
      {...reveal}
    >
      <div className="h-28">
        <MessagePrimitive.Parts />
      </div>
      {/* thread.tsx's autohide: only the newest reply's bar stays mounted off hover. */}
      <ActionBarPrimitive.Root autohide="not-last" className="flex h-12 items-center px-12">
        <MessageTiming side="top" className="h-8 px-2" />
      </ActionBarPrimitive.Root>
    </MessagePrimitive.Root>
  );
}

function App() {
  const runtime = useExternalStoreRuntime({
    messages: MESSAGES,
    convertMessage: (m: ThreadMessageLike) => m,
    onNew: async () => {},
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <div className="h-80" />
        <ThreadPrimitive.Messages
          components={{ UserMessage: () => null, AssistantMessage: Assistant }}
        />
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
