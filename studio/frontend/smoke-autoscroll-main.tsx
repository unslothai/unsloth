// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_chat_autoscroll.py; vite entry, no backend.

import {
  IntentAwareScrollProvider,
  useIntentAwareAutoScroll,
} from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const PARAGRAPH =
  "British cultural exports carry several distinct threads, and the reception of each " +
  "varies by audience, decade and medium. This paragraph exists to give the viewport real " +
  "prose to lay out and to wrap over several lines at this width.";

function Harness(): ReactElement {
  const { ref, context } = useIntentAwareAutoScroll();
  const [blocks, setBlocks] = useState<string[]>([]);
  const [tail, setTail] = useState("");
  const viewportRef = useRef<HTMLElement | null>(null);
  const spacerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const viewport = () => viewportRef.current;
    const api = {
      seed(count = 40): void {
        setBlocks(Array.from({ length: count }, (_, index) => `${index}`));
        setTail("");
      },
      token(text: string): void {
        setTail((current) => current + text);
      },
      block(): void {
        setBlocks((current) => [...current, `${current.length}`]);
        setTail("");
      },
      /** Grow content with no mutation or resize, like a late image or webfont. */
      growSilently(px: number): void {
        const spacer = spacerRef.current;
        if (spacer) spacer.style.height = `${px}px`;
      },
      resetGrowth(): void {
        const spacer = spacerRef.current;
        if (spacer) spacer.style.height = "0px";
      },
      distanceFromBottom(): number {
        const element = viewport();
        if (!element) return -1;
        return Math.max(
          0,
          element.scrollHeight - element.scrollTop - element.clientHeight,
        );
      },
      isAtBottom(): boolean {
        return context.getIsAtBottom();
      },
      scrollUpBy(px: number): void {
        const element = viewport();
        if (element) element.scrollTop = Math.max(0, element.scrollTop - px);
      },
      scrollDownBy(px: number): void {
        const element = viewport();
        if (element) element.scrollTop += px;
      },
      detach(): void {
        context.detachFromBottom();
      },
      scrollToBottom(): void {
        context.scrollToBottom("instant");
      },
      metrics(): { scrollHeight: number; scrollTop: number; clientHeight: number } {
        const element = viewport();
        if (!element) return { scrollHeight: -1, scrollTop: -1, clientHeight: -1 };
        return {
          scrollHeight: element.scrollHeight,
          scrollTop: element.scrollTop,
          clientHeight: element.clientHeight,
        };
      },
    };
    (window as unknown as { __autoscroll: typeof api }).__autoscroll = api;
  }, [context]);

  return (
    <IntentAwareScrollProvider value={context}>
      <div
        data-smoke="viewport"
        ref={(element) => {
          viewportRef.current = element;
          ref(element);
        }}
        style={{
          height: "100vh",
          width: "760px",
          overflowY: "auto",
          padding: "16px",
          paddingBottom: "var(--aui-scroll-stabilizer, 0px)",
        }}
      >
        {blocks.map((id) => (
          <p key={id} style={{ margin: "0 0 16px" }}>
            Message {id}. {PARAGRAPH}
          </p>
        ))}
        <p data-smoke="tail" style={{ margin: "0 0 16px" }}>
          {tail}
        </p>
        <div ref={spacerRef} data-smoke="spacer" style={{ height: "0px" }} />
      </div>
    </IntentAwareScrollProvider>
  );
}

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}
createRoot(root).render(<Harness />);
