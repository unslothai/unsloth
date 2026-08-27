// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_tool_activity.py.
//
// The node suite can reach the two open-state reducers and the preferences
// store, but not a rendered Radix Collapsible: there is no jsdom and no JSX
// loader, so `aria-expanded`, whether closed content is in the DOM at all, and
// what a close does to scroll position are only answerable here.
//
// It mounts the four disclosure paths the preference has to reach, because they
// arrive at the same primitive by different routes and a fix that lands in one
// can miss the others:
//
//   controlled    useToolActivityOpen(isRunning, hasText) -- web search,
//                 knowledge base, code execution. Collapses from an effect.
//   uncontrolled  <ToolFallbackRoot defaultOpen={isRunning}> -- terminal and
//                 every generic/MCP tool. Collapses during render.
//   approval      the same card with awaitingApproval set. Its payload is the
//                 command being approved, and Allow/Deny render outside the
//                 card, so this one must stay open whatever the preference says.
//   group         <ToolGroupRoot> -- the multi-call wrapper, which owns its
//                 open state separately from the cards inside it.
//
// Tall filler sits above the cards so a collapse can be measured for viewport
// movement, and an explicit `overflow-y: auto` ancestor exists because that is
// what useCollapseScrollLock walks up to find; against a plain document body
// the hook finds nothing and silently does nothing.

import "@/index.css";

// Load-bearing import order. `tool-group.tsx` reaches the `@/features/chat`
// barrel, which re-exports chat-runtime-store, which sits in a cycle with it;
// entering that graph from the barrel or from `assistant-ui/thread` dies with
// "Cannot access 'CHAT_GPU_MEMORY_MODE_KEY' before initialization". Entering
// from the store module orders it correctly. Observed, not theorised, and a
// property of the app's module graph rather than of anything under test.
/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import "@/features/chat/stores/chat-runtime-store";

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

const params = new URLSearchParams(window.location.search);
const fillers = Number.parseInt(params.get("fillers") ?? "60", 10);
const strict = params.get("strict") === "1";
const rtl = params.get("rtl") === "1";
// `?only=uncontrolled` renders a single card. The scroll scene needs it: the
// preference closes every card at once and a chevron closes one, so measuring
// them against each other on the full page compares different amounts of
// content collapsing, not different code paths.
const only = params.get("only") ?? "";
const shows = (name: string) => only === "" || only === name;

const WORDS =
  "the quick brown fox jumps over the lazy dog while a second clause keeps the line long enough to wrap".split(
    " ",
  );

// Stable ids, assigned once. The list never reorders, but keying off the array
// index would still trip biome's noArrayIndexKey, and a smoke harness that
// lands new lint findings is a smoke harness nobody wants to keep.
const WORD_ITEMS = WORDS.map((word, index) => ({
  word,
  id: `${index}-${word}`,
}));
const LINE_IDS = (count: number, prefix: string) =>
  Array.from({ length: count }, (_, index) => `${prefix}-${index}`);

// Long enough that the trigger's 60-character slice cannot show all of it. The
// driver looks for the tail, so "the user can read the command" cannot be
// satisfied by the truncated trigger label alone.
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

// Deliberately tall: a close that does not lock scroll is only visible when the
// thing collapsing is big enough to move everything below it.
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

/** Mirrors tool-ui-web-search.tsx: state from the hook, setter straight back. */
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

/** Mirrors tool-ui-terminal.tsx, including `defaultOpen` being a live prop. */
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

/**
 * A terminal card parked on an allow/deny decision. The trigger carries the
 * same truncated label the real one does, and the full command lives inside the
 * collapsible, so the scene can ask the only question that matters: with the
 * preference on, can the user read what they are approving?
 */
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

/** The multi-call wrapper, uncontrolled exactly as ToolGroupImpl leaves it. */
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
  // Remount key, so a scene can ask "what does a card mounting NOW do"
  // separately from "what does an already-mounted card do".
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const w = window as unknown as Record<string, unknown>;
    w.__setRunning = (v: boolean) => setIsRunning(v);
    w.__setHasText = (v: boolean) => setHasText(v);
    w.__setAwaitingApproval = (v: boolean) => setAwaitingApproval(v);
    w.__remount = () => setGeneration((g) => g + 1);
    w.__setPreference = (v: boolean) =>
      useChatPreferencesStore.getState().setCollapseToolActivityByDefault(v);
    w.__getPreference = () =>
      useChatPreferencesStore.getState().collapseToolActivityByDefault;
    // The declared default, so a scene can check "landed on the default"
    // without hard-coding which default that currently is.
    w.__getDefaultPreference = () =>
      useChatPreferencesStore.getInitialState().collapseToolActivityByDefault;
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
            useChatPreferencesStore.getState().collapseToolActivityByDefault,
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

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}

// StrictMode is a scene, not the default: it double-renders, which is exactly
// what the render-phase setState in ToolFallbackRoot/ToolGroupRoot has to
// survive, but it also doubles every effect and would muddy the measurements.
createRoot(root).render(
  strict ? (
    <StrictMode>
      <App />
    </StrictMode>
  ) : (
    <App />
  ),
);
