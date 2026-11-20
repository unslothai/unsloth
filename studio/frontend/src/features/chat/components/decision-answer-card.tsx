// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A decision answer rendered as a card in the thread instead of prose.
 *
 * The card names the answer once, next to Copy JSON, and shows only the
 * distribution below it. Timing lives in the message toolbar like every other
 * reply, and Copy JSON hands back exactly what the model returned.
 */

import { AnswerDistribution } from "@/components/decision/decision-parts";
import { Button } from "@/components/ui/button";
// eslint-disable-next-line no-restricted-imports -- Leaf module: request types, no React.
import {
  type DecisionAnswer,
  type DecisionType,
  criterionText,
  scoreLevels,
} from "@/features/settings/lib/decision-request";
import { useLocale, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { type ReactElement, useMemo, useState } from "react";
import { decisionNativeText } from "../lib/decision-result";

export type DecisionMessagePayload = {
  type: DecisionType;
  answers: Record<string, DecisionAnswer>;
  /** Choice options by wire key ("1", "2"), so a row reads "2 Mango". */
  labels?: Record<string, string>;
  usage?: { input_tokens?: number; output_tokens?: number };
  latency_ms?: number;
  model?: string | null;
  /** The exact body llama-server returned, for the copy button. */
  native: unknown;
};

export function DecisionAnswerCard({
  payload,
}: {
  payload: DecisionMessagePayload;
}): ReactElement {
  const t = useT();
  const locale = useLocale();
  const [copied, setCopied] = useState(false);

  const percent = useMemo(() => {
    const format = new Intl.NumberFormat(locale, {
      style: "percent",
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
    return (value: number) => format.format(value);
  }, [locale]);

  const answers = Object.entries(payload.answers ?? {});
  const first = answers[0]?.[1];

  // The answer, named once: Yes / No, the chosen option, or the top level.
  let summary: string | null = null;
  if (first?.type === "noul") {
    summary = first.noul >= 0.5 ? t("decisions.yes") : t("decisions.no");
  } else if (first?.type === "choice") {
    const name = payload.labels?.[first.choice] ?? "";
    summary = name ? `${first.choice} ${name}` : first.choice;
  } else if (first?.type === "score") {
    const { top, max } = scoreLevels(first);
    const label = criterionText(first.legend?.[top]) ?? top;
    summary = `${label} · ${first.score.toFixed(2)} / ${max}`;
  }

  return (
    <section className="flex flex-col gap-4 rounded-2xl border border-border/70 bg-card/60 p-4">
      <header className="flex items-center justify-between gap-3">
        <span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">
          {summary}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 shrink-0 px-2 text-ui-12 text-muted-foreground"
          onClick={() => {
            void copyToClipboard(decisionNativeText(payload.native)).then((ok) => {
              if (!ok) return;
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            });
          }}
        >
          {copied ? t("decisions.copied") : t("decisions.copyJson")}
        </Button>
      </header>

      <div className="flex flex-col gap-4">
        {answers.map(([name, answer]) => (
          <AnswerDistribution
            key={name}
            answer={answer}
            question={undefined}
            percent={percent}
            labels={payload.labels}
          />
        ))}
      </div>
    </section>
  );
}
