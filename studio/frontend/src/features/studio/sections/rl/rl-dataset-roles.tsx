// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getHfToken } from "@/features/hub";
import {
  checkDatasetFormat,
  useTrainingConfigStore,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import type { TrainingObjective } from "@/types/training";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";

type RlRole = "prompt" | "answer" | "chosen" | "rejected" | "system";

// Mirrors RL_ROLES / _AUTO_ROLE_NAMES in studio/backend/core/training/rl.py.
const ROLES: Record<Exclude<TrainingObjective, "sft">, readonly RlRole[]> = {
  dpo: ["prompt", "chosen", "rejected", "system"],
  orpo: ["prompt", "chosen", "rejected", "system"],
  grpo: ["prompt", "answer", "system"],
};
const REQUIRED: Record<Exclude<TrainingObjective, "sft">, readonly RlRole[]> = {
  dpo: ["prompt", "chosen", "rejected"],
  orpo: ["prompt", "chosen", "rejected"],
  grpo: ["prompt"],
};
const AUTO_NAMES: Record<RlRole, readonly string[]> = {
  prompt: ["prompt", "question", "instruction", "problem", "query", "input"],
  answer: ["answer", "solution", "final_answer", "target", "label"],
  chosen: ["chosen", "accepted", "preferred"],
  rejected: ["rejected", "dispreferred"],
  system: ["system", "system_prompt"],
};
const ROLE_LABEL: Record<RlRole, TranslationKey> = {
  prompt: "rl.dataset.role.prompt",
  answer: "rl.dataset.role.answer",
  chosen: "rl.dataset.role.chosen",
  rejected: "rl.dataset.role.rejected",
  system: "rl.dataset.role.system",
};
const IGNORE = "__ignore";

function autoMapping(
  columns: string[],
  roles: readonly RlRole[],
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const role of roles) {
    const column = columns.find(
      (c) => AUTO_NAMES[role].includes(c.toLowerCase()) && !(c in out),
    );
    if (column) {
      out[column] = role;
    }
  }
  return out;
}

export function RlDatasetRoles({
  objective,
}: {
  objective: Exclude<TrainingObjective, "sft">;
}): ReactElement {
  const t = useT();
  const config = useTrainingConfigStore(
    useShallow((s) => ({
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      uploadedFile: s.uploadedFile,
      datasetSubset: s.datasetSubset,
      datasetSplit: s.datasetSplit,
      mapping: s.rlRoleMapping,
      setMapping: s.setRlRoleMapping,
    })),
  );
  const [columns, setColumns] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const roles = ROLES[objective];
  const datasetName =
    config.datasetSource === "huggingface"
      ? config.dataset
      : config.datasetSource === "upload"
        ? config.uploadedFile
        : null;

  const loadColumns = async () => {
    if (!datasetName) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await checkDatasetFormat({
        datasetName,
        hfToken: getHfToken() || null,
        subset: config.datasetSubset,
        split: config.datasetSplit,
        isVlm: false,
      });
      setColumns(res.columns);
      if (Object.keys(config.mapping).length === 0) {
        config.setMapping(autoMapping(res.columns, roles));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const setRole = (column: string, role: string) => {
    const next = { ...config.mapping };
    delete next[column];
    if (role !== IGNORE) {
      for (const [col, r] of Object.entries(next)) {
        if (r === role) {
          delete next[col];
        }
      }
      next[column] = role;
    }
    config.setMapping(next);
  };

  const mapped = new Set(Object.values(config.mapping));
  // Unmapped required roles still auto-detect on the backend, so only flag what has no match at all.
  const missing = REQUIRED[objective].filter(
    (role) =>
      !(
        mapped.has(role) ||
        (columns ?? []).some((c) => AUTO_NAMES[role].includes(c.toLowerCase()))
      ),
  );

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border/70 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-xs font-medium text-foreground">
            {t("rl.dataset.title")}
            <NewBadge />
          </p>
          <p className="text-ui-11p5 text-muted-foreground/85">
            {t("rl.dataset.description")}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={!datasetName || loading}
          onClick={loadColumns}
        >
          {loading ? t("rl.dataset.loading") : t("rl.dataset.loadColumns")}
        </Button>
      </div>
      {!datasetName && (
        <p className="text-ui-11p5 text-muted-foreground">
          {t("rl.dataset.noDataset")}
        </p>
      )}
      {error && (
        <p className="text-ui-11p5 text-destructive">
          {t("rl.dataset.error", { error })}
        </p>
      )}
      {columns?.map((column) => (
        <div
          key={column}
          className="grid grid-cols-[minmax(0,1fr)_calc(170px*var(--ui-space-scale,1))] items-center gap-2"
        >
          <span className="truncate font-mono text-xs text-foreground">
            {column}
          </span>
          <Select
            value={config.mapping[column] ?? IGNORE}
            onValueChange={(role) => setRole(column, role)}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={IGNORE}>
                {t("rl.dataset.role.ignore")}
              </SelectItem>
              {roles.map((role) => (
                <SelectItem key={role} value={role}>
                  {t(ROLE_LABEL[role])}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      ))}
      {columns && missing.length > 0 && (
        <p className="text-ui-11p5 text-destructive">
          {t("rl.dataset.missing", {
            roles: missing.map((role) => t(ROLE_LABEL[role])).join(", "),
          })}
        </p>
      )}
      {objective === "grpo" && (
        <p className="text-ui-11p5 text-muted-foreground/85">
          {t("rl.dataset.answerNote")}
        </p>
      )}
    </div>
  );
}
