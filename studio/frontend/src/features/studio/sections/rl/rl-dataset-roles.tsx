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
  previewCell,
  useRlWorkspaceStore,
  useTrainingConfigStore,
  RL_REQUIRED_ROLES,
  RL_ROLES,
  type RlObjective,
  type RlRole,
  missingRlRoles,
  resolveRlMapping,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  AiChat02Icon,
  Cancel01Icon,
  Message01Icon,
  RefreshIcon,
  Target02Icon,
  Tick02Icon,
  ViewOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";

const ROLE_LABEL: Record<RlRole, TranslationKey> = {
  prompt: "rl.dataset.role.prompt",
  answer: "rl.dataset.role.answer",
  chosen: "rl.dataset.role.chosen",
  rejected: "rl.dataset.role.rejected",
  system: "rl.dataset.role.system",
};
const ROLE_ICON: Record<RlRole, IconSvgElement> = {
  prompt: Message01Icon,
  answer: Target02Icon,
  chosen: Tick02Icon,
  rejected: Cancel01Icon,
  system: AiChat02Icon,
};
const ROLE_TONE: Record<RlRole, string> = {
  prompt: "text-sky-700 dark:text-sky-300",
  answer: "text-amber-700 dark:text-amber-300",
  chosen: "text-emerald-700 dark:text-emerald-300",
  rejected: "text-rose-700 dark:text-rose-300",
  system: "text-violet-700 dark:text-violet-300",
};
const IGNORE = "__ignore";

function RoleChip({
  role,
  mapped,
  required,
}: {
  role: RlRole;
  mapped: boolean;
  required: boolean;
}): ReactElement {
  const t = useT();
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-ui-10 font-medium",
        mapped
          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
          : required
            ? "bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300"
            : "bg-muted text-muted-foreground",
      )}
    >
      <HugeiconsIcon
        icon={mapped ? Tick02Icon : ROLE_ICON[role]}
        className="size-3"
      />
      {t(ROLE_LABEL[role])}
    </span>
  );
}

export function RlDatasetRoles({
  objective,
}: {
  objective: RlObjective;
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
  const { previewRow, setPreviewRow } = useRlWorkspaceStore(
    useShallow((s) => ({
      previewRow: s.previewRow,
      setPreviewRow: s.setPreviewRow,
    })),
  );
  const [columns, setColumns] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latestRequest = useRef(0);
  const roles = RL_ROLES[objective];
  const datasetName =
    config.datasetSource === "huggingface"
      ? config.dataset
      : config.datasetSource === "upload"
        ? config.uploadedFile
        : null;

  const { datasetSubset, datasetSplit, mapping, setMapping } = config;
  const loadColumns = useCallback(async () => {
    if (!datasetName) {
      return;
    }
    // Picking a dataset changes the split a moment later; a reply for the old split
    // ("Bad split: train") must not land on top of the newer one.
    const request = ++latestRequest.current;
    setLoading(true);
    setError(null);
    try {
      const res = await checkDatasetFormat({
        datasetName,
        hfToken: getHfToken() || null,
        subset: datasetSubset,
        split: datasetSplit,
        isVlm: false,
      });
      if (request === latestRequest.current) {
        setColumns(res.columns);
        setPreviewRow(res.preview_samples?.[0] ?? null);
      }
    } catch (err) {
      if (request === latestRequest.current) {
        setColumns(null);
        setPreviewRow(null);
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (request === latestRequest.current) {
        setLoading(false);
      }
    }
  }, [datasetName, datasetSubset, datasetSplit, setPreviewRow]);

  // Reload on remount and on dataset changes, so roles survive leaving the page.
  useEffect(() => {
    // Deferred so loadColumns' state resets run in a callback, not the effect body.
    void Promise.resolve().then(loadColumns);
  }, [loadColumns]);

  // Drop roles from another objective and fill the rest from column names.
  useEffect(() => {
    if (!columns) {
      return;
    }
    const next = resolveRlMapping(objective, columns, mapping);
    if (JSON.stringify(next) !== JSON.stringify(mapping)) {
      setMapping(next);
    }
  }, [columns, objective, mapping, setMapping]);

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

  const missing = columns
    ? missingRlRoles(objective, columns, config.mapping)
    : [];
  const mappedRoles = new Set(Object.values(config.mapping));
  const required = RL_REQUIRED_ROLES[objective];
  const chipRoles = roles.filter(
    (role) => required.includes(role) || role === "answer",
  );

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border/70 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-xs font-medium text-foreground">
            {t("rl.dataset.title")}
            <NewBadge />
          </p>
          <p className="text-ui-11p5 text-muted-foreground/85">
            {t("rl.dataset.description")}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {columns &&
            chipRoles.map((role) => (
              <RoleChip
                key={role}
                role={role}
                mapped={mappedRoles.has(role)}
                required={required.includes(role)}
              />
            ))}
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            aria-label={t("rl.dataset.loadColumns")}
            title={t("rl.dataset.loadColumns")}
            disabled={!datasetName}
            onClick={loadColumns}
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={cn("size-3.5", loading && "animate-spin")}
            />
          </Button>
        </div>
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
      {loading && !columns && (
        <p className="text-ui-11p5 text-muted-foreground">
          {t("rl.dataset.loading")}
        </p>
      )}
      {columns && columns.length > 0 && (
        <div className="flex flex-col">
          <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_calc(220px*var(--ui-space-scale,1))] gap-3 pb-1.5 text-ui-10 uppercase tracking-[0.05em] text-muted-foreground/70">
            <span>{t("rl.dataset.columnHeader")}</span>
            <span>{t("rl.dataset.sampleHeader")}</span>
            <span>{t("rl.dataset.roleHeader")}</span>
          </div>
          {columns.map((column) => {
            const role = config.mapping[column] as RlRole | undefined;
            const sample = previewCell(previewRow?.[column]);
            return (
              <div
                key={column}
                className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_calc(220px*var(--ui-space-scale,1))] items-center gap-3 border-t border-border/50 py-2"
              >
                <span className="truncate font-mono text-xs text-foreground">
                  {column}
                </span>
                <span
                  className="truncate text-ui-11p5 text-muted-foreground"
                  title={sample}
                >
                  {sample || "—"}
                </span>
                <Select
                  value={role ?? IGNORE}
                  onValueChange={(next) => setRole(column, next)}
                >
                  <SelectTrigger
                    size="sm"
                    className={cn("w-full", role && ROLE_TONE[role])}
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={IGNORE}>
                      {t("rl.dataset.role.ignore")}
                    </SelectItem>
                    {roles.map((r: RlRole) => (
                      <SelectItem key={r} value={r}>
                        <span className="flex items-center gap-1.5">
                          <HugeiconsIcon
                            icon={ROLE_ICON[r]}
                            className={cn("size-3.5", ROLE_TONE[r])}
                          />
                          {t(ROLE_LABEL[r])}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            );
          })}
        </div>
      )}
      {columns && missing.length > 0 && (
        <p className="text-ui-11p5 text-amber-700 dark:text-amber-300">
          {t("rl.dataset.missing", {
            roles: missing.map((role) => t(ROLE_LABEL[role])).join(", "),
          })}
        </p>
      )}
      {objective === "grpo" && (
        <p className="flex items-center gap-1.5 text-ui-11p5 text-muted-foreground/85">
          <HugeiconsIcon icon={ViewOffIcon} className="size-3.5 shrink-0" />
          {t("rl.dataset.answerNote")}
        </p>
      )}
    </div>
  );
}
