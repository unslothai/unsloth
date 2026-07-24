// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MemoryScopeRequest } from "../types/api";

const MAX_CAPTURE_SOURCE_CHARS = 4_000;
const DURABLE_SIGNAL_RE =
  /\b(?:i|my|we|our|for me|actually|no longer|instead|prefer|always|never|this (?:project|repo|app)|(?:project|repo|app) (?:uses|is|has)|goal|constraint|we use)\b/i;
const CODE_LINE_RE =
  /^\s*(?:import\s|export\s|const\s|let\s|var\s|def\s|class\s|function\s|\{|\}|\[|\]|#include\b)/;
const EXPLICIT_COMMAND_RE =
  /^(?:please\s+)?(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?(?:(?:remember|forget)\b|(?:remove|delete)\b[^\n]*\bmemor(?:y|ies)\b)/i;
const LINE_BREAK_RE = /\r?\n/;
const FENCED_CODE_RE = /```[\s\S]*?```/g;
const INTERROGATIVE_RE =
  /^(?:can|could|would|should|do|does|did|is|are|will|what|when|where|why|how)\b/i;
const QUESTION_END_RE = /\?\s*$/;
const SENTENCE_BREAK_RE = /(?<=[.!?])\s+|\r?\n/;

// Admit durable declarations cheaply; the server validates before commit.
export function shouldCaptureMemoryCandidate(text: string): boolean {
  const source = text.trim();
  if (!source || source.length > MAX_CAPTURE_SOURCE_CHARS) {
    return false;
  }
  if (EXPLICIT_COMMAND_RE.test(source)) {
    return false;
  }

  const lines = source.split(LINE_BREAK_RE);
  const codeLines = lines.filter((line) => CODE_LINE_RE.test(line)).length;
  const fencedCodeChars = (source.match(FENCED_CODE_RE) ?? []).reduce(
    (total, block) => total + block.length,
    0,
  );
  if (codeLines > lines.length / 2 || fencedCodeChars > source.length / 2) {
    return false;
  }

  // Skip standalone questions while retaining durable declarations.
  if (INTERROGATIVE_RE.test(source) && QUESTION_END_RE.test(source)) {
    return false;
  }

  if (!DURABLE_SIGNAL_RE.test(source)) {
    return false;
  }
  if (QUESTION_END_RE.test(source)) {
    const hasDeclarativeSignal = source
      .split(SENTENCE_BREAK_RE)
      .some(
        (sentence) =>
          !QUESTION_END_RE.test(sentence) && DURABLE_SIGNAL_RE.test(sentence),
      );
    if (!hasDeclarativeSignal) {
      return false;
    }
  }

  return true;
}

export function buildMemoryScope({
  threadId,
  sourceMessageId,
  incognito,
  referenceMemories,
  autoSaveMemories,
}: {
  threadId?: string;
  sourceMessageId?: string;
  incognito: boolean;
  referenceMemories: boolean;
  autoSaveMemories: boolean;
}): MemoryScopeRequest | undefined {
  if (incognito || !threadId || !sourceMessageId) {
    return undefined;
  }
  return {
    thread_id: threadId,
    source_message_id: sourceMessageId,
    recall: referenceMemories,
    auto_capture: autoSaveMemories,
    allow_explicit_commands: true,
  };
}
