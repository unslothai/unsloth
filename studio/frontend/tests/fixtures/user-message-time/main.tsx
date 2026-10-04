// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Real runtime, focus handling, footer, timestamp and stylesheet. Only message
// data and the unrelated action handlers are fixtures; no backend is needed.
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AssistantRuntimeProvider,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type ThreadMessage,
} from "@assistant-ui/react";
import { useState, type CSSProperties } from "react";
import { createRoot } from "react-dom/client";
import {
  UserMessageActionBar,
  UserMessageFooter,
} from "@/components/assistant-ui/user-message-actions";
import { useActionBarFocusReveal } from "@/components/assistant-ui/use-action-bar-focus-reveal";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { setLocale, type Locale } from "@/i18n";
import "./styles.css";

const params = new URLSearchParams(location.search);
const now = new Date();
const createdAt = params.has("invalid")
  ? new Date(NaN)
  : params.has("today")
    ? now
    : new Date(now.getFullYear() - 1, 8, 11, 23, 22);
const initial: ThreadMessage[] = Array.from(
  { length: Number(params.get("count") ?? 1) },
  (_, i) => ({
    id: `user-${i}`,
    role: "user",
    createdAt,
    content: [{ type: "text", text: "Hello" }],
    attachments: [],
    metadata: {
      custom: params.has("estimated") ? { createdAtEstimated: true } : {},
    },
  }),
);
initial.push({
  id: "assistant",
  role: "assistant",
  createdAt: now,
  content: [{ type: "text", text: "Reply" }],
  status: { type: "running" },
  metadata: {
    custom: {},
    unstable_state: null,
    steps: [],
    unstable_annotations: [],
    unstable_data: [],
  },
});

function User() {
  const reveal = useActionBarFocusReveal();
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root mx-auto flex w-full min-w-0 flex-col items-end py-4"
      tabIndex={0}
      {...reveal}
    >
      <div className="rounded-2xl bg-muted px-4 py-2">Hello</div>
      <UserMessageFooter>
        <UserMessageActionBar>
          {["Copy", "Edit", "Fork", "Delete"].map((label) => (
            <TooltipIconButton key={label} tooltip={label}>
              <span aria-hidden="true">{label[0]}</span>
            </TooltipIconButton>
          ))}
        </UserMessageActionBar>
        {params.has("branches") && (
          <div className="aui-user-branch-picker ml-0.5 inline-flex shrink-0 items-center text-ui-13">
            <button className="aui-branch-chevron-btn" aria-label="Previous">
              ‹
            </button>
            <span className="font-mono tabular-nums">1/2</span>
            <button className="aui-branch-chevron-btn" aria-label="Next">
              ›
            </button>
          </div>
        )}
      </UserMessageFooter>
      <span
        className="aui-user-reveal-sentinel"
        tabIndex={0}
        aria-label="Message actions"
      />
    </MessagePrimitive.Root>
  );
}
function Assistant() {
  return (
    <MessagePrimitive.Root>
      <MessagePrimitive.Parts />
    </MessagePrimitive.Root>
  );
}
function PopupAssistant() {
  const reveal = useActionBarFocusReveal();
  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root"
      tabIndex={0}
      {...reveal}
    >
      <MessagePrimitive.Parts />
      <ActionBarPrimitive.Root
        autohide="always"
        className="aui-assistant-action-bar-root"
      >
        <ActionBarMorePrimitive.Root modal={false}>
          <ActionBarMorePrimitive.Trigger>More</ActionBarMorePrimitive.Trigger>
          <ActionBarMorePrimitive.Content
            onCloseAutoFocus={(event) => event.preventDefault()}
          >
            <ActionBarMorePrimitive.Item>
              Menu action
            </ActionBarMorePrimitive.Item>
          </ActionBarMorePrimitive.Content>
        </ActionBarMorePrimitive.Root>
      </ActionBarPrimitive.Root>
    </MessagePrimitive.Root>
  );
}
declare global {
  interface Window {
    messageTimeFixture: {
      updateReply: () => void;
      setLocale: typeof setLocale;
    };
  }
}
function App() {
  const [messages, setMessages] = useState(initial);
  const runtime = useExternalStoreRuntime({
    messages,
    isRunning: true,
    onNew: async () => {},
  });
  window.messageTimeFixture = {
    setLocale,
    updateReply: () =>
      setMessages((previous) =>
        previous.map((message) =>
          message.role === "assistant"
            ? {
                ...message,
                content: [{ type: "text", text: `Streaming ${Math.random()}` }],
              }
            : message,
        ),
      ),
  };
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <button id="before">Before chat</button>
        <ThreadPrimitive.Root
          className="aui-root flex flex-col overflow-hidden"
          style={
            {
              height: 500,
              "--thread-content-max-width": "46rem",
            } as CSSProperties
          }
        >
          <ThreadPrimitive.Viewport
            autoScroll={false}
            className="aui-thread-viewport min-w-0 overflow-auto px-5"
          >
            <ThreadPrimitive.Messages
              components={{
                UserMessage: User,
                AssistantMessage: params.has("popup")
                  ? PopupAssistant
                  : Assistant,
              }}
            />
          </ThreadPrimitive.Viewport>
        </ThreadPrimitive.Root>
        <button id="after">After chat</button>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
await setLocale((params.get("locale") ?? "en") as Locale);
if (params.has("scale"))
  document.documentElement.style.setProperty(
    "--ui-font-size-scale",
    params.get("scale")!,
  );
createRoot(document.getElementById("root")!).render(<App />);
