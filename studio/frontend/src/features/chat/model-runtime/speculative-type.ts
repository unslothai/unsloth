// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function normalizeSpeculativeType(
  value: string | null | undefined,
): string | null {
  if (value == null) return null;
  const s = String(value).trim().toLowerCase();
  if (!s) return null;
  if (s === "auto" || s === "default") return "auto";
  if (s === "off") return "off";
  if (s === "ngram-simple") return "ngram-simple";
  if (s === "mtp" || s === "draft-mtp") return "mtp";
  if (s === "ngram" || s === "ngram-mod") return "ngram";
  if (s === "mtp+ngram") return "mtp+ngram";
  const parts = s.split(",").map((p) => p.trim()).filter(Boolean);
  const hasMtp = parts.some((p) => p === "mtp" || p === "draft-mtp");
  const hasNgram = parts.some((p) => p === "ngram" || p === "ngram-mod");
  if (hasMtp && hasNgram) return "mtp+ngram";
  if (hasMtp) return "mtp";
  if (hasNgram) return "ngram";
  return "auto";
}
