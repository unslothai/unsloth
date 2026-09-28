// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { nonnegativeDecimal, type PublishedPricing } from "./model-pricing.ts";

export interface FastPair {
  standard: string;
  fast: string;
  source?: "catalog" | "user";
  verificationSource?: string;
  evidence?: string;
}
export interface FastCandidate {
  id: string;
  name?: string | null;
  description?: string | null;
}

export function normalizeFastPairs(value: unknown): FastPair[] {
  if (!Array.isArray(value)) return [];
  const used = new Set<string>();
  return value.flatMap((pair) => {
    if (
      !pair ||
      typeof pair.standard !== "string" ||
      typeof pair.fast !== "string"
    )
      return [];
    const standard = pair.standard.trim(),
      fast = pair.fast.trim();
    if (
      !standard ||
      !fast ||
      standard === fast ||
      used.has(standard) ||
      used.has(fast)
    )
      return [];
    used.add(standard);
    used.add(fast);
    return [{ standard, fast, source: "user" as const }];
  });
}

/** Description-based discovery is deliberately conservative: explicit speed edition AND a
 * same-checkpoint claim naming exactly one catalog model. A suffix alone is never evidence. */
export function discoverFastPairs(
  models: readonly FastCandidate[],
): FastPair[] {
  const proposed: FastPair[] = [];
  for (const fast of models) {
    const description = fast.description ?? "";
    if (
      !/\b(?:fast(?:er)?(?:\s+speed)?|high[- ]speed|ultra[- ]?speed)\s+(?:edition|variant|version)\b/i.test(
        description,
      )
    )
      continue;
    const claim = description.match(
      /\b(?:same|identical)\b[^!?]{0,180}\b(?:checkpoint|weights)\b[^!?]{0,100}/i,
    )?.[0];
    if (
      !claim ||
      /\b(?:not|different|unlike)\b/i.test(
        description.slice(
          Math.max(0, description.indexOf(claim) - 48),
          description.indexOf(claim) + claim.length,
        ),
      )
    )
      continue;
    const candidates = models.filter((base) => {
      if (
        base.id === fast.id ||
        base.id.split("/")[0] !== fast.id.split("/")[0]
      )
        return false;
      const slug = base.id.split("/").at(-1)!;
      // Exact published identifier/name boundaries avoid matching another model's version or suffix.
      const escaped = slug.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`(?:^|[^\\w.-])${escaped}(?![\\w.-])`, "i").test(claim);
    });
    if (candidates.length === 1)
      proposed.push({
        standard: candidates[0].id,
        fast: fast.id,
        source: "catalog",
        verificationSource: `https://openrouter.ai/${fast.id.split("/").map(encodeURIComponent).join("/")}`,
        evidence: claim,
      });
  }
  // No arbitrary winner when two faster editions name the same base.
  return proposed.filter(
    (pair) =>
      proposed.filter(
        (other) =>
          other.standard === pair.standard ||
          other.fast === pair.standard ||
          other.standard === pair.fast,
      ).length === 1,
  );
}

export function resolveFastPairs(
  models: readonly FastCandidate[],
  configured: unknown,
  autoDetect = true,
): FastPair[] {
  const manual = normalizeFastPairs(configured);
  const used = new Set(manual.flatMap((pair) => [pair.standard, pair.fast]));
  return [
    ...manual,
    ...(autoDetect
      ? discoverFastPairs(models).filter(
          (pair) => !used.has(pair.standard) && !used.has(pair.fast),
        )
      : []),
  ];
}

export function verifiedFastVariant(
  provider: string | undefined,
  model: string | undefined,
  enabled: readonly string[],
  available?: readonly string[],
  pairs: readonly FastPair[] = [],
) {
  if (provider !== "openrouter") return null;
  const pair = pairs.find((p) => p.standard === model || p.fast === model);
  if (!pair) return null;
  const isFast = model === pair.fast;
  const destination = isFast ? pair.standard : pair.fast;
  const enabledCompanion = enabled.includes(destination);
  const availableCompanion =
    !available?.length || available.includes(destination);
  return {
    pair,
    isFast,
    destination,
    enabledCompanion,
    availableCompanion,
    reason: !availableCompanion
      ? "Companion model is no longer available in this connection. Refresh connection settings."
      : null,
  };
}

export function priceDifference(
  from: PublishedPricing | null,
  to: PublishedPricing | null,
): string | null {
  if (!from || !to) return null;
  if (from.overrides?.length || to.overrides?.length) return "Variable rates";
  const ratios = ["prompt", "completion"].map((key) => {
    const base = nonnegativeDecimal(from.rates[key]);
    const target = nonnegativeDecimal(to.rates[key]);
    return base != null && base > 0 && target != null ? target / base : null;
  });
  const [input, output] = ratios;
  if (input == null || output == null) return null;
  const format = (value: number) =>
    new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
  return Math.abs(input - output) < 1e-9
    ? `${format(input)}× published rates`
    : `Input ${format(input)}× · Output ${format(output)}×`;
}
