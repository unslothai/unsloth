// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReasoningReadingAnchor } from "./reasoning-transcript-index.ts";

const passages = "p, pre, li, h1, h2, h3, h4, h5, h6, td, th";

function textNodes(root: Element): Text[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode())
    if (node.textContent && node.parentElement?.closest(passages))
      nodes.push(node as Text);
  return nodes;
}

/** Locate a text range without changing the user's native selection. */
export function reasoningTextRange(
  root: Element,
  text: string,
  occurrence = 0,
): Range | null {
  const nodes = textNodes(root);
  const source = nodes.map((node) => node.data).join("");
  let start = -1;
  for (let i = 0; i <= occurrence; i += 1) {
    start = source.indexOf(text, start + 1);
    if (start < 0) return null;
  }
  const range = document.createRange();
  let offset = 0;
  let started = false;
  for (const node of nodes) {
    if (!started && start < offset + node.length) {
      range.setStart(node, start - offset);
      started = true;
    }
    if (started && start + text.length <= offset + node.length) {
      range.setEnd(node, start + text.length - offset);
      return range;
    }
    offset += node.length;
  }
  return null;
}

/** Capture the first visible character, including when its block starts above the viewport. */
export function captureReasoningAnchor(
  root: Element,
  viewport: Element,
): ReasoningReadingAnchor | undefined {
  const bounds = viewport.getBoundingClientRect();
  const nodes = textNodes(root);
  let preceding = "";
  for (const node of nodes) {
    const range = document.createRange();
    range.selectNodeContents(node);
    const rect = range.getBoundingClientRect();
    if (
      rect.bottom <= bounds.top + 2 ||
      rect.top >= bounds.bottom ||
      !rect.width
    ) {
      preceding += node.data;
      continue;
    }
    let low = 0;
    let high = node.length - 1;
    while (low < high) {
      const middle = (low + high) >>> 1;
      range.setStart(node, middle);
      range.setEnd(node, middle + 1);
      if (range.getBoundingClientRect().bottom <= bounds.top + 2)
        low = middle + 1;
      else high = middle;
    }
    // Stay inside this text node so formatting and fragment boundaries cannot
    // make the anchor needle depend on synthesized layout whitespace.
    const text = node.data.slice(low, Math.min(node.length, low + 32));
    range.setStart(node, low);
    range.setEnd(node, low + text.length);
    const prefix = preceding + node.data.slice(0, low);
    let occurrence = 0;
    for (
      let at = prefix.indexOf(text);
      at >= 0;
      at = prefix.indexOf(text, at + 1)
    )
      occurrence += 1;
    return { text, top: range.getBoundingClientRect().top, occurrence };
  }
  return undefined;
}
