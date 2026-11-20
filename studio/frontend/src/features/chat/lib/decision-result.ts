// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Turns the composer's decision questions into a /v1/systemone body, and the
 * answer back into the markdown the assistant bubble shows.
 */

// eslint-disable-next-line no-restricted-imports -- Leaf module: request types and pure builders, no React.
import {
  type DecisionAnswer,
  criterionText,
} from "@/features/settings/lib/decision-request";
import type { LoadedDecisionResponse } from "../api/decision-api";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function answerBlock(
  name: string,
  answer: DecisionAnswer,
  showName: boolean,
): string {
  const heading = (label: string, detail?: string) =>
    `${showName ? `**${name}** — ` : ""}**${label}**${detail ? ` · ${detail}` : ""}`;

  if (answer.type === "noul") {
    const yes = answer.noul;
    return [
      heading(yes >= 0.5 ? "Yes" : "No", percent(Math.max(yes, 1 - yes))),
      `- Yes — ${percent(yes)}`,
      `- No — ${percent(1 - yes)}`,
    ].join("\n");
  }

  if (answer.type === "choice") {
    const rows = Object.entries(answer.probabilities).sort(
      (a, b) => b[1] - a[1],
    );
    return [
      heading(answer.choice, percent(answer.confidence)),
      ...rows.map(([option, value]) => `- ${option} — ${percent(value)}`),
    ].join("\n");
  }

  const keys = Object.keys(answer.probabilities).sort(
    (a, b) => Number(a) - Number(b),
  );
  const top = keys.reduce(
    (best, key) =>
      (answer.probabilities[key] ?? 0) > (answer.probabilities[best] ?? -1)
        ? key
        : best,
    keys[0] ?? "",
  );
  return [
    heading(
      criterionText(answer.legend?.[top]) ?? top,
      `score ${answer.score.toFixed(2)} · ${percent(answer.confidence)}`,
    ),
    ...keys.map((key) => {
      const label = criterionText(answer.legend?.[key]) ?? key;
      return `- ${key}. ${label} — ${percent(answer.probabilities[key] ?? 0)}`;
    }),
  ].join("\n");
}

/** The raw body a decision model returned, as text. */
export function decisionNativeText(native: unknown): string {
  if (native === undefined) return "";
  if (typeof native === "string") return native;
  try {
    return JSON.stringify(native, null, 2) ?? "";
  } catch {
    return String(native);
  }
}

/** The assistant message text for a decision answer. */
export function formatDecisionResult(
  response: LoadedDecisionResponse,
): string {
  const answers = Object.entries(response.answers ?? {});
  if (answers.length === 0) return "The model returned no answers.";

  const blocks = answers.map(([name, answer]) =>
    answerBlock(name, answer, answers.length > 1),
  );
  const meta = [
    typeof response.latency_ms === "number"
      ? `${Math.round(response.latency_ms)} ms`
      : null,
    typeof response.usage?.input_tokens === "number"
      ? `${response.usage.input_tokens} input tokens`
      : null,
  ].filter(Boolean);

  return meta.length > 0
    ? `${blocks.join("\n\n")}\n\n_${meta.join(" · ")}_`
    : blocks.join("\n\n");
}
