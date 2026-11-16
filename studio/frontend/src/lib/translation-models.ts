// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Own module so Chat and node tests can import it without the Translate page.

/** Models the Translate page downloads on first use. They answer every prompt with a translation. */
export const TRANSLATION_ONLY_REPOS = [
  "unsloth/Hy-MT2-1.8B-GGUF",
  "unsloth/Hy-MT2-7B-GGUF",
] as const;

const normalize = (id: string | null | undefined) =>
  (id ?? "").trim().toLowerCase();

export function isTranslationOnlyRepo(id: string | null | undefined): boolean {
  const wanted = normalize(id);
  return (
    wanted !== "" &&
    TRANSLATION_ONLY_REPOS.some((repo) => repo.toLowerCase() === wanted)
  );
}

/**
 * Translate loads its model into the chat slot when models are not kept side by side. Chat reads
 * that slot as empty, unless the user picked the model in Chat themselves.
 */
export function isTranslationHeldSlot(
  activeModel: string | null | undefined,
  chatSelection: string | null | undefined,
): boolean {
  return (
    isTranslationOnlyRepo(activeModel) &&
    normalize(activeModel) !== normalize(chatSelection)
  );
}
