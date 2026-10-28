// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { streamOneShotCompletion } from "@/features/chat";
import { cleanTranslation } from "./translate-prompt";

/** Room for a long passage. */
const TRANSLATE_MAX_TOKENS = 4096;

export class TranslateLengthError extends Error {
  constructor() {
    super("The translation was cut off.");
  }
}

/** Streams one translation on `checkpoint`; `onText` gets the cleaned text so far. */
export async function streamTranslation({
  checkpoint,
  systemPrompt,
  text,
  signal,
  onText,
}: {
  checkpoint: string;
  systemPrompt: string;
  text: string;
  signal: AbortSignal;
  onText: (text: string) => void;
}): Promise<string> {
  const result = await streamOneShotCompletion({
    checkpoint,
    systemPrompt,
    prompt: text,
    maxTokens: TRANSLATE_MAX_TOKENS,
    signal,
    onText: (raw) => onText(cleanTranslation(raw)),
  });
  if (!result) throw new Error("This model's connection is unavailable.");
  if (result.finishReason === "length") throw new TranslateLengthError();
  return cleanTranslation(result.text);
}
