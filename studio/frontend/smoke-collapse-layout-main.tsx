// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_collapse_layout.py: does a collapsible toggle lay out the
// whole document? Pane content is fixed; only filler size varies. Arms selected with `?arm=`.

import "@/index.css";

// Import first: entering the reasoning.tsx import cycle elsewhere hits MarkdownText in TDZ.
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

// Raw Radix primitive: the project wrapper adds height keyframes that tailwind-merge cannot drop.
// The two grid arms must not share classes: data-state would snap the unmeasured arm open.
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
  // Streaming into an open pane: `1fr` must re-resolve every frame, unlike a measured height.
  const [extra, setExtra] = useState(0);
  useEffect(() => {
    (window as unknown as Record<string, unknown>).__probeGrow = (n: number) =>
      setExtra((current) => current + n);
    (window as unknown as Record<string, unknown>).__probeReset = () => setExtra(0);
  }, []);

  // Publish readiness after commit: concurrent createRoot may still be mounting large fillers.
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

