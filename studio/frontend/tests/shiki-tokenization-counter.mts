// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `shiki/core` with one change: every highlighter a bundled factory hands out
// counts the source it is asked to tokenize. The explicit
// `createBundledHighlighter` below shadows the one the star export would provide.
export * from "shiki/core";
import { createBundledHighlighter as createRealBundledHighlighter } from "shiki/core";

export const tokenized = { characters: 0, calls: 0 };

export const createBundledHighlighter: typeof createRealBundledHighlighter = (
  bundle,
) => {
  const createRealHighlighter = createRealBundledHighlighter(bundle);
  return async (...args) => {
    const highlighter = await createRealHighlighter(...args);
    const codeToTokens = highlighter.codeToTokens.bind(highlighter);
    highlighter.codeToTokens = (code, options) => {
      tokenized.characters += code.length;
      tokenized.calls += 1;
      return codeToTokens(code, options);
    };
    return highlighter;
  };
};
