// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The speech model usually omits the final mark, and sometimes picks a full
// stop for a question ("How to switch Git branch."). Fix the mark at the end of
// the last sentence without rewriting the sentence itself.

const TERMINAL_MARK = /([.!?…])(["'”’)\]]*)$/;

const LAST_SENTENCE = /[^.!?…]+[.!?…]?/g;

function lastSentence(text: string): string {
  // The closing quote/bracket after the final mark is not a new sentence.
  const core = text.replace(/["'”’)\]]+$/, "").trimEnd();
  const sentences = (core || text).match(LAST_SENTENCE);
  return (sentences?.at(-1) ?? core ?? text).trim();
}

function isQuestionSentence(sentence: string): boolean {
  const clean = sentence.replace(/^[\s"'“”‘’([{]+/, "").trim();
  if (!clean) return false;
  // A quoted question at the end ("She asked, \"How...\"") is still the
  // sentence being punctuated.
  const quote = Math.max(clean.lastIndexOf('"'), clean.lastIndexOf("“"));
  const candidate = quote >= 0 ? clean.slice(quote + 1).trim() : clean;
  const text = candidate || clean;

  // Exclamations that open with a wh-word.
  if (/^(what\s+a|what\s+an)\b/i.test(text)) return false;
  if (
    /^how\s+(beautiful|nice|lovely|great|wonderful|amazing|kind|sad|strange|interesting|cool|terrible|awful|funny)\b/i.test(
      text,
    )
  ) {
    return false;
  }

  // Imperatives that look like auxiliary-first questions.
  if (
    /^do\s+(it|this|that|these|those|not|the|a|an|your|my|our|their|his|her|some|any|anything|everything|nothing|one|me|us|him|them)\b/i.test(
      text,
    )
  ) {
    return false;
  }
  if (/^have\s+(a|an|some|any|it|this|that|these|those)\b/i.test(text)) {
    return false;
  }

  // Question openings that are not a bare wh-word.
  if (/^(how\s+to|what\s+about|why\s+not|what\s+if|how\s+come)\b/i.test(text)) {
    return true;
  }

  return /^(what|why|how|when|where|who|whom|whose|which|is|are|was|were|do|does|did|can|could|will|would|should|may|might|have|has|had)\b/i.test(
    text,
  );
}

/**
 * Add the missing final mark, and turn a full stop into a question mark when
 * the last sentence is a question. Short non-question fragments are left alone:
 * a pause is not a sentence.
 */
export function ensureTerminalPunctuation(text: string): string {
  const trimmed = text.trimEnd();
  if (!trimmed) return text;

  const match = trimmed.match(TERMINAL_MARK);
  const question = isQuestionSentence(lastSentence(trimmed));

  if (match) {
    if (match[1] !== "." || !question) return text;
    const mark = match.index ?? trimmed.length;
    return `${trimmed.slice(0, mark)}?${match[2]}`;
  }

  if (question) return `${trimmed}?`;
  if (trimmed.split(/\s+/).length < 4 && !trimmed.includes("=")) return text;
  return `${trimmed}.`;
}
