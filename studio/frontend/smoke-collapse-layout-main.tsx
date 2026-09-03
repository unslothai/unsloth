// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_collapse_layout.py.
//
// It answers one question: when a collapsible toggles, does the browser lay out the WHOLE document,
// and does that cost scale with how big the document is?
//
// The design point that makes the answer readable is that the COLLAPSIBLE'S OWN CONTENT IS
// IDENTICAL IN EVERY RUN. Only the filler around it changes size. So if a toggle gets more
// expensive as `fillers` grows, the extra cost cannot be the pane's own content -- it is the rest
// of the document being laid out because of a toggle that has nothing to do with it.
//
// Four arms, selected with `?arm=`:
//
//   radix-height     Radix `CollapsibleContent` + the `animate-collapsible-*` height keyframes.
//                    Today's mechanism.
//   radix-grid       Radix `CollapsibleContent` + a `grid-template-rows: 0fr -> 1fr` transition,
//                    with nothing anywhere reading `--radix-collapsible-content-height`. This is
//                    the CSS-ONLY version of the fix, and it exists to be disproved: Radix's
//                    layout effect measures whether or not the variable is consumed, so this arm
//                    should still force a full-document layout. If it does, a keyframe swap alone
//                    is not the fix.
//   unmeasured-grid  The local `UnmeasuredCollapsible` + the same grid transition. No measurement
//                    at all.
//   reasoning        The real `ReasoningRoot` / `ReasoningTrigger` / `ReasoningContent` /
//                    `ReasoningText`, which follow GRID_COLLAPSE_REASONING_ENABLED. Run the page
//                    once per flag value to get the real before/after rather than a model of it.
//
// No backend, no runtime: the reasoning primitives are plain components, and giving them a
// synthetic runtime would only add nodes that are not part of what is being measured.

import "@/index.css";

// FIRST, and load-bearing. `reasoning.tsx` sits in an import cycle with `markdown-text.tsx` and the
// `@/features/chat` barrel, and `thread.tsx` reads `MarkdownText` at module scope. Entering that
// cycle from `reasoning.tsx` evaluates `thread.tsx` while `markdown-text.tsx` is still
// initialising, and the page dies with "Cannot access 'MarkdownText' before initialization".
// Entering from `thread.tsx`, which is what the app and smoke-heavy-thread.html both do, orders it
// correctly. This is a property of the app's module graph, not of anything under test here.
import "@/components/assistant-ui/thread";

import {
  ReasoningContent,
  ReasoningRoot,
  ReasoningText,
  ReasoningTrigger,
} from "@/components/assistant-ui/reasoning";
import { Collapsible as CollapsiblePrimitive } from "radix-ui";
import {
  UnmeasuredCollapsible,
  UnmeasuredCollapsibleContent,
  UnmeasuredCollapsibleTrigger,
} from "@/components/ui/unmeasured-collapsible";
import type { JSX } from "react";
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

const params = new URLSearchParams(window.location.search);
const arm = params.get("arm") ?? "radix-height";
const fillers = Number.parseInt(params.get("fillers") ?? "0", 10);
const paneParagraphs = Number.parseInt(params.get("paneParagraphs") ?? "40", 10);

const WORDS =
  "the quick brown fox jumps over the lazy dog while a second clause keeps the line long enough to wrap".split(
    " ",
  );

// One filler row is a block with a dozen inline boxes, which is roughly the shape of a line of
// rendered prose. Layout objects, not characters, are the unit that Blink's layout cost is
// proportional to, so the filler is built out of boxes rather than out of one long string.
function Filler({ index }: { index: number }) {
  return (
    <div className="filler-row px-4 py-1 text-sm">
      {WORDS.map((word, i) => (
        <span key={`${word}-${i}`} className="mr-1 inline-block">
          {word}
          {i === 0 ? index : ""}
        </span>
      ))}
    </div>
  );
}

function PaneBody({ extra }: { extra: number }) {
  return (
    <>
      {Array.from({ length: paneParagraphs + extra }, (_, i) => (
        <p key={i} className="mb-2">
          Reasoning paragraph {i}: {WORDS.join(" ")} {WORDS.join(" ")}
        </p>
      ))}
    </>
  );
}

// The two Radix arms use the RAW primitive rather than `@/components/ui/collapsible`, so that the
// class list under test is exactly the one written here. The project wrapper prepends
// `animate-collapsible-down` / `animate-collapsible-up`, and tailwind-merge does not know those as
// members of its `animate` group, so a wrapper-based grid arm would silently keep the height
// keyframes and the comparison would be between two things that both animate height.
//
// `heightContentClass` is `reasoning.tsx`'s own list with the duration inlined.
//
// The two grid arms do NOT share a class string, and the first attempt at this got it wrong in a
// way worth recording. Radix has no "presence separate from state", so its arm has to drive the
// row size off `data-state`. `UnmeasuredCollapsibleContent` drives it off its own staged
// `expanded`, which lags `data-state` by two frames precisely so the `0fr` start value exists.
// Giving the unmeasured arm the `data-[state=open]:grid-rows-[1fr]` variant as well let the
// attribute selector -- higher specificity than the plain class -- win at mount, and the pane
// snapped open with no transition while still reporting a clean layout count. A cheap-looking
// number for an animation that was not running.
const heightContentClass =
  "overflow-hidden ease-out data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down data-[state=closed]:fill-mode-forwards data-[state=open]:duration-200 data-[state=closed]:duration-200";

const radixGridContentClass =
  "grid transition-[grid-template-rows] duration-200 ease-out data-[state=open]:grid-rows-[1fr] data-[state=closed]:grid-rows-[0fr]";

const unmeasuredGridContentClass = "transition-[grid-template-rows] duration-200 ease-out";

function RadixHeightArm({ extra }: ArmProps) {
  return (
    <CollapsiblePrimitive.Root data-probe="collapsible" className="border p-2">
      <CollapsiblePrimitive.Trigger data-probe="trigger">
        toggle
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content data-probe="content" className={heightContentClass}>
        <PaneBody extra={extra} />
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  );
}

function RadixGridArm({ extra }: ArmProps) {
  return (
    <CollapsiblePrimitive.Root data-probe="collapsible" className="border p-2">
      <CollapsiblePrimitive.Trigger data-probe="trigger">
        toggle
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content data-probe="content" className={radixGridContentClass}>
        <div className="min-h-0 overflow-hidden">
          <PaneBody extra={extra} />
        </div>
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  );
}

function UnmeasuredGridArm({ extra }: ArmProps) {
  return (
    <UnmeasuredCollapsible data-probe="collapsible" className="border p-2">
      <UnmeasuredCollapsibleTrigger data-probe="trigger">
        toggle
      </UnmeasuredCollapsibleTrigger>
      <UnmeasuredCollapsibleContent data-probe="content" className={unmeasuredGridContentClass}>
        <PaneBody extra={extra} />
      </UnmeasuredCollapsibleContent>
    </UnmeasuredCollapsible>
  );
}

function ReasoningArm({ extra }: ArmProps) {
  const [open, setOpen] = useState(false);
  return (
    <ReasoningRoot data-probe="collapsible" open={open} onOpenChange={setOpen}>
      <ReasoningTrigger data-probe="trigger" duration={3} />
      <ReasoningContent data-probe="content">
        <ReasoningText>
          <PaneBody extra={extra} />
        </ReasoningText>
      </ReasoningContent>
    </ReasoningRoot>
  );
}

type ArmProps = { extra: number };

const ARMS: Record<string, (props: ArmProps) => JSX.Element> = {
  "radix-height": RadixHeightArm,
  "radix-grid": RadixGridArm,
  "unmeasured-grid": UnmeasuredGridArm,
  reasoning: ReasoningArm,
};

function App() {
  const Arm = ARMS[arm];
  if (!Arm) {
    throw new Error(`unknown arm: ${arm}`);
  }
  // Content streaming into an OPEN pane is the case a height-based collapse gets wrong: the height
  // it animated to was measured once, at toggle time, so content arriving afterwards either
  // overflows the clip or needs a fresh measurement. `1fr` re-resolves every frame, so the driver
  // grows the pane mid-flight and checks the rendered height followed it.
  const [extra, setExtra] = useState(0);
  useEffect(() => {
    (window as unknown as Record<string, unknown>).__probeGrow = (n: number) =>
      setExtra((current) => current + n);
    (window as unknown as Record<string, unknown>).__probeReset = () => setExtra(0);
  }, []);

  // Readiness is published from HERE, after this component and its filler tree have committed,
  // and not from module scope after two frames of the initial render. `createRoot` renders
  // concurrently, so a large `fillers` cell can still be mid-mount two frames in, and a count
  // taken then reports the document as far smaller than it ends up. That number is the x axis of
  // the whole experiment, so getting it from a pre-commit DOM would silently flatten the curve.
  // The frames are still waited out, so the count is taken after layout rather than during it.
  useEffect(() => {
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => {
        (window as unknown as Record<string, unknown>).__probeReady = {
          arm,
          fillers,
          paneParagraphs,
          elements: document.getElementsByTagName("*").length,
        };
      });
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, []);
  return (
    <div>
      {/* The collapsible sits FIRST so that the filler is all after it in document order. A
          full-document layout has to walk it either way; a correctly scoped subtree layout does
          not. */}
      <div data-probe="pane-host">
        <Arm extra={extra} />
      </div>
      <div data-probe="filler-host">
        {Array.from({ length: fillers }, (_, i) => (
          <Filler key={i} index={i} />
        ))}
      </div>
    </div>
  );
}

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}

createRoot(root).render(<App />);

// `__probeReady` is published from an effect inside `App` (see above), not from here. At module
// scope the only thing that can be waited on is frames, and frames do not imply that a concurrent
// initial mount has committed.
