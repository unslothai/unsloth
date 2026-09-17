// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `shiki` with one change: every highlighter it hands out counts the source it
// is asked to tokenize. The explicit `createHighlighter` below shadows the one
// the star export would provide.
export * from "shiki";
import { createHighlighter as createRealHighlighter } from "shiki";

export const tokenized = { characters: 0, calls: 0 };

export const createHighlighter = async (
  ...args: Parameters<typeof createRealHighlighter>
) => {
  const highlighter = await createRealHighlighter(...args);
  const codeToTokens = highlighter.codeToTokens.bind(highlighter);
  highlighter.codeToTokens = (code, options) => {
    tokenized.characters += code.length;
    tokenized.calls += 1;
    return codeToTokens(code, options);
  };
  return highlighter;
};
