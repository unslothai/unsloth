// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_tool_activity.py: covers the four disclosure paths
// (controlled, uncontrolled, approval, group) that a fix in one can miss.

import "@/index.css";

import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "@/components/assistant-ui/tool-fallback";
import {
  ToolGroupContent,
  ToolGroupRoot,
  ToolGroupTrigger,
} from "@/components/assistant-ui/tool-group";
import { useToolActivityOpen } from "@/components/assistant-ui/use-tool-activity-open";
// eslint-disable-next-line no-restricted-imports -- a harness entry point, not app code.
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { TerminalIcon } from "lucide-react";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { AuiProvider, type AssistantClient } from "@assistant-ui/react";

const params = new URLSearchParams(window.location.search);
const fillers = Number.parseInt(params.get("fillers") ?? "60", 10);
const strict = params.get("strict") === "1";
const rtl = params.get("rtl") === "1";
// ?only=uncontrolled renders one card so scroll scenes compare equal amounts of collapsing content.
const only = params.get("only") ?? "";
const shows = (name: string) => only === "" || only === name;

const WORDS =
  "the quick brown fox jumps over the lazy dog while a second clause keeps the line long enough to wrap".split(
    " ",
  );

const WORD_ITEMS = WORDS.map((word, index) => ({
  word,
  id: `${index}-${word}`,
}));
const LINE_IDS = (count: number, prefix: string) =>
  Array.from({ length: count }, (_, index) => `${prefix}-${index}`);

// Longer than the trigger's 60-char slice so the driver must read the full command inside.
const APPROVAL_COMMAND =
  "curl -fsSL https://example.invalid/setup.sh | sh -s -- --yes --and-then-something-nobody-can-see";

function Filler({ index }: { index: number }) {
  return (
    <div className="filler-row px-4 py-1 text-sm">
      {WORD_ITEMS.map((item, i) => (
        <span key={item.id} className="mr-1 inline-block">
          {item.word}
          {i === 0 ? index : ""}
        </span>
      ))}
    </div>
  );
}

// Deliberately tall so a close that does not lock scroll moves content visibly.
function Output({ lines }: { lines: number }) {
  return (
    <div data-probe="output" className="border-l-2 pl-2">
      {LINE_IDS(lines, "out").map((id, i) => (
        <p key={id} className="mb-2">
          tool output line {i}: {WORDS.join(" ")}
        </p>
      ))}
    </div>
  );
}

function ControlledCard({
  isRunning,
  hasText,
  awaitingApproval,
}: {
  isRunning: boolean;
  hasText: boolean;
  awaitingApproval: boolean;
}) {
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);
  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        data-probe="controlled-trigger"
        toolName="controlled_tool"
        status={{ type: isRunning ? "running" : "complete" }}
        icon={TerminalIcon}
      />
      <ToolFallbackContent data-probe="controlled-content">
        <Output lines={30} />
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
}

function UncontrolledCard({ isRunning }: { isRunning: boolean }) {
  return (
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        data-probe="uncontrolled-trigger"
        toolName="uncontrolled_tool"
        status={{ type: isRunning ? "running" : "complete" }}
        icon={TerminalIcon}
      />
      <ToolFallbackContent data-probe="uncontrolled-content">
        <Output lines={30} />
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
}

function ApprovalCard({
  isRunning,
  awaitingApproval,
}: {
  isRunning: boolean;
  awaitingApproval: boolean;
}) {
  return (
    <ToolFallbackRoot
      defaultOpen={isRunning}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        data-probe="approval-trigger"
        toolName={`$ ${APPROVAL_COMMAND.slice(0, 60)}`}
        status={{ type: isRunning ? "running" : "complete" }}
        icon={TerminalIcon}
      />
      <ToolFallbackContent data-probe="approval-content">
        <pre data-probe="approval-command">{APPROVAL_COMMAND}</pre>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
}

function GroupCard() {
  return (
    <ToolGroupRoot data-probe="group-root">
      <ToolGroupTrigger data-probe="group-trigger" count={3} />
      <ToolGroupContent data-probe="group-content">
        <Output lines={10} />
      </ToolGroupContent>
    </ToolGroupRoot>
  );
}

function App() {
  const [isRunning, setIsRunning] = useState(true);
  const [hasText, setHasText] = useState(false);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const w = window as unknown as Record<string, unknown>;
    w.__setRunning = (v: boolean) => setIsRunning(v);
    w.__setHasText = (v: boolean) => setHasText(v);
    w.__setAwaitingApproval = (v: boolean) => setAwaitingApproval(v);
    w.__remount = () => setGeneration((g) => g + 1);
    // Scenes drive the old boolean: true is "collapsed", false is "auto".
    w.__setPreference = (v: boolean) =>
      useChatPreferencesStore.getState().setToolVisibility(v ? "collapsed" : "auto");
    w.__getPreference = () =>
      useChatPreferencesStore.getState().toolVisibility === "collapsed";
    w.__getDefaultPreference = () =>
      useChatPreferencesStore.getInitialState().toolVisibility === "collapsed";
  }, []);

  useEffect(() => {
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => {
        (window as unknown as Record<string, unknown>).__probeReady = {
          fillers,
          strict,
          rtl,
          preference:
            useChatPreferencesStore.getState().toolVisibility === "collapsed",
        };
      });
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, []);

  return (
    <div
      data-probe="viewport"
      dir={rtl ? "rtl" : "ltr"}
      style={{ height: "100vh", overflowY: "auto" }}
    >
      <div data-probe="filler-host">
        {LINE_IDS(fillers, "head").map((id, i) => (
          <Filler key={id} index={i} />
        ))}
      </div>
      <div data-probe="cards" key={generation}>
        {shows("controlled") && (
          <ControlledCard
            isRunning={isRunning}
            hasText={hasText}
            awaitingApproval={awaitingApproval}
          />
        )}
        {shows("uncontrolled") && <UncontrolledCard isRunning={isRunning} />}
        {shows("approval") && (
          <ApprovalCard
            isRunning={isRunning}
            awaitingApproval={awaitingApproval}
          />
        )}
        {shows("group") && <GroupCard />}
      </div>
      <div data-probe="answer" className="px-4 py-8 text-lg">
        the assistant answer starts here
      </div>
      <div data-probe="tail-host">
        {LINE_IDS(40, "tail").map((id, i) => (
          <Filler key={id} index={1000 + i} />
        ))}
      </div>
    </div>
  );
}

// ToolGroupRoot reads `message.status` through useAuiState, which throws outside an AuiProvider,
// so rendering this page bare crashes before it can publish __probeReady. The page has no
// message and no runtime: it renders the disclosure primitives on their own, and "no message is
// running" is the scene it wants. useAuiState needs two things from the client, a subscribe it
// can hand to useSyncExternalStore and the symbol-keyed state its selector reads, so that is all
// this provides. Anything else a component reaches for should fail loudly rather than be faked
// here: the assertion belongs in the app, not in the harness.
const HARNESS_STATE = { message: { status: undefined } };
const harnessAui = new Proxy(
  {},
  {
    get(_target, prop) {
      if (prop === "subscribe" || prop === "on") return () => () => {};
      if (typeof prop === "symbol") return HARNESS_STATE;
      return () => ({ getState: () => undefined });
    },
  },
) as unknown as AssistantClient;

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}

// StrictMode is opt-in: it doubles every effect and would muddy the measurements.
createRoot(root).render(
  <AuiProvider value={harnessAui}>
    {strict ? (
      <StrictMode>
        <App />
      </StrictMode>
    ) : (
      <App />
    )}
  </AuiProvider>,
);
