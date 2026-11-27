// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Own module so Chat and node tests can import it without the Translate page.

// Hy-MT2 stays listed: a copy still on disk is a translation model, not a chat model.
export const TRANSLATION_ONLY_REPOS = [
  "unslothai/Hy-MT3-1.8B-GGUF",
  "unslothai/Hy-MT3-7B-GGUF",
  "unslothai/Hy-MT3-30B-A3B-GGUF",
  "unslothai/Hy-MT3-1.8B-FP8",
  "unslothai/Hy-MT3-7B-FP8",
  "unslothai/Hy-MT3-30B-A3B-FP8",
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

/** Chat reads the slot Translate loaded into as empty, unless the user picked that model in Chat. */
export function isTranslationHeldSlot(
  activeModel: string | null | undefined,
  chatSelection: string | null | undefined,
): boolean {
  return (
    isTranslationOnlyRepo(activeModel) &&
    normalize(activeModel) !== normalize(chatSelection)
  );
}
