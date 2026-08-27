// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, type Dispatch, type SetStateAction } from "react";
import {
  fetchAdmissions,
  fetchContradictions,
  fetchRollouts,
} from "./api/memory-api";

const HYGIENE_LABEL: Record<
  "compact" | "contradictions" | "admissions" | "rollouts",
  TranslationKey
> = {
  compact: "unforgettable.hygiene.compact",
  contradictions: "unforgettable.hygiene.contradictions",
  admissions: "unforgettable.hygiene.admissions",
  rollouts: "unforgettable.hygiene.rollouts",
};

export type HygieneReport = {
  compact?: string;
  contradictions?: string;
  admissions?: string;
  rollouts?: string;
};

export function HygienePanel({
  busy,
  hygiene,
  onCompact,
  onReport,
}: {
  busy: boolean;
  hygiene: HygieneReport;
  onCompact: (apply: boolean) => void;
  onReport: Dispatch<SetStateAction<HygieneReport>>;
}) {
  const t = useT();
  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      fetchContradictions(),
      fetchAdmissions(),
      fetchRollouts(),
    ])
      .then(([contradictions, admissions, rollouts]) => {
        if (cancelled) return;
        onReport((prev) => ({
          ...prev,
          contradictions: JSON.stringify(contradictions, null, 2),
          admissions: JSON.stringify(admissions, null, 2),
          rollouts: JSON.stringify(rollouts, null, 2),
        }));
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        toast.error(
          error instanceof Error
            ? error.message
            : t("unforgettable.errors.load"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [onReport, t]);
  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => onCompact(false)}
        >
          {t("unforgettable.hygiene.compact")}
        </Button>
        <Button
          type="button"
          size="sm"
          disabled={busy}
          onClick={() => onCompact(true)}
        >
          {t("unforgettable.hygiene.compactApply")}
        </Button>
      </div>
      {(["compact", "contradictions", "admissions", "rollouts"] as const).map(
        (key) =>
          hygiene[key] ? (
            <section key={key}>
              <h2 className="mb-1 text-sm font-medium">
                {t(HYGIENE_LABEL[key])}
              </h2>
              <pre className="max-h-56 overflow-auto rounded-lg bg-muted/50 p-3 text-xs">
                {hygiene[key]}
              </pre>
            </section>
          ) : null,
      )}
    </div>
  );
}
