// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { toastError, toastSuccess } from "@/shared/toast";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import {
  type LegacyImportItemResult,
  type LegacyImportResult,
  type UserAssetsBootstrap,
  bootstrapUserAssets,
  importLegacyUserAssets,
} from "./api";

const SOURCE = "recipe-indexeddb-v1";
const LEGACY_PAGE_SIZE = 100;
const MAX_LEGACY_BATCH_BYTES = 8 * 1024 * 1024;
const MAX_VISIBLE_DETAILS = 100;
const utf8Encoder = new TextEncoder();

type LegacyPage<T> = {
  items: T[];
  nextCursor: string | null;
};

type LegacyPageReader<T> = (
  cursor?: string | null,
  limit?: number,
) => Promise<LegacyPage<T>>;

type LegacyRecipe = { id: string; revision?: number } & object;
type LegacyExecution = { id: string } & object;

type PendingImport = {
  bootstrap: UserAssetsBootstrap;
  recipeCount: number;
  executionCount: number;
};

type LegacyImportCoordinatorProps = {
  onImported: () => void;
  readRecipes: LegacyPageReader<LegacyRecipe>;
  readExecutions: LegacyPageReader<LegacyExecution>;
};

function withoutRevision(recipe: LegacyRecipe) {
  return { ...recipe, revision: undefined };
}

function splitByLegacyBatchBytes<T extends object>(
  items: T[],
  kind: "recipes" | "executions",
): T[][] {
  const batches: T[][] = [];
  let current: T[] = [];
  const byteLength = (values: T[]): number =>
    utf8Encoder.encode(
      JSON.stringify({
        recipes: kind === "recipes" ? values : [],
        executions: kind === "executions" ? values : [],
      }),
    ).byteLength;

  for (const item of items) {
    const candidate = [...current, item];
    if (current.length > 0 && byteLength(candidate) > MAX_LEGACY_BATCH_BYTES) {
      batches.push(current);
      current = [item];
    } else {
      current = candidate;
    }
  }
  if (current.length > 0) {
    batches.push(current);
  }
  return batches;
}

async function scanUncoveredCount<T extends { id: string }>(
  readPage: LegacyPageReader<T>,
  importedIds: ReadonlySet<string>,
): Promise<number> {
  let cursor: string | null = null;
  let count = 0;
  do {
    const page = await readPage(cursor, LEGACY_PAGE_SIZE);
    count += page.items.reduce(
      (total, item) => total + (importedIds.has(item.id) ? 0 : 1),
      0,
    );
    if (cursor !== null && page.nextCursor === cursor) {
      throw new Error("Legacy import reader returned a repeated cursor.");
    }
    cursor = page.nextCursor;
  } while (cursor);
  return count;
}

function createAggregate(): LegacyImportResult {
  return { recipes: [], executions: [], summary: {} };
}

function addEffectiveResults(
  aggregate: LegacyImportResult,
  kind: "recipes" | "executions",
  results: LegacyImportItemResult[],
): void {
  for (const item of results) {
    aggregate.summary[item.outcome] =
      (aggregate.summary[item.outcome] ?? 0) + 1;
  }
  const visible = results.filter((item) => item.outcome !== "imported");
  aggregate[kind].push(
    ...visible.slice(
      0,
      Math.max(0, MAX_VISIBLE_DETAILS - aggregate[kind].length),
    ),
  );
}

function effectiveRetryResults(
  initial: LegacyImportItemResult[],
  retried: LegacyImportItemResult[],
): LegacyImportItemResult[] {
  const retries = new Map(
    retried.filter((item) => item.id).map((item) => [item.id, item]),
  );
  return initial.map((item) =>
    item.outcome === "missing_parent" && item.id
      ? (retries.get(item.id) ?? item)
      : item,
  );
}

export function LegacyImportCoordinator({
  onImported,
  readRecipes,
  readExecutions,
}: LegacyImportCoordinatorProps): ReactElement | null {
  const t = useT();
  const [pending, setPending] = useState<PendingImport | null>(null);
  const [result, setResult] = useState<LegacyImportResult | null>(null);
  const [open, setOpen] = useState(false);
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    let active = true;
    bootstrapUserAssets()
      .then(async (bootstrap) => {
        const importedRecipes = new Set(bootstrap.importLedger.recipes);
        const importedExecutions = new Set(bootstrap.importLedger.executions);
        const [recipeCount, executionCount] = await Promise.all([
          scanUncoveredCount(readRecipes, importedRecipes),
          scanUncoveredCount(readExecutions, importedExecutions),
        ]);
        if (!active || (recipeCount === 0 && executionCount === 0)) return;
        setPending({ bootstrap, recipeCount, executionCount });
        setOpen(true);
      })
      .catch((error) => {
        // The authoritative recipe fetch owns the visible backend error state.
        console.error("Legacy recipe scan failed:", error);
      });
    return () => {
      active = false;
    };
  }, [readExecutions, readRecipes]);

  const outcomeLabels = useMemo<Record<string, string>>(
    () => ({
      imported: t("dataRecipes.import.imported"),
      redacted: t("dataRecipes.import.redacted"),
      already_imported: t("dataRecipes.import.alreadyImported"),
      id_retired: t("dataRecipes.import.retired"),
      rejected: t("dataRecipes.import.rejected"),
      missing_parent: t("dataRecipes.import.missingParent"),
    }),
    [t],
  );

  const reasonLabels = useMemo<Record<string, string>>(
    () => ({
      already_exists: t("dataRecipes.import.reasons.alreadyExists"),
      id_retired: t("dataRecipes.import.reasons.idRetired"),
      missing_parent: t("dataRecipes.import.reasons.missingParent"),
      invalid_identifier: t("dataRecipes.import.reasons.invalidIdentifier"),
      invalid_id: t("dataRecipes.import.reasons.invalidIdentifier"),
      invalid_json: t("dataRecipes.import.reasons.invalidData"),
      invalid_json_object: t("dataRecipes.import.reasons.invalidData"),
      invalid_timestamp: t("dataRecipes.import.reasons.invalidData"),
      invalid_name: t("dataRecipes.import.reasons.invalidName"),
      invalid_execution_metadata: t(
        "dataRecipes.import.reasons.invalidExecution",
      ),
      field_limit_exceeded: t("dataRecipes.import.reasons.fieldLimit"),
      size_limit_exceeded: t("dataRecipes.import.reasons.fieldLimit"),
      secret_field_forbidden: t("dataRecipes.import.reasons.secretField"),
      secret_fields_present: t("dataRecipes.import.reasons.secretField"),
      invalid_legacy_batch: t("dataRecipes.import.reasons.batchLimit"),
      legacy_batch_limit_exceeded: t("dataRecipes.import.reasons.batchLimit"),
    }),
    [t],
  );

  const summary = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.summary).map(([outcome, count]) => ({
      outcome,
      count,
      label: outcomeLabels[outcome] ?? t("dataRecipes.import.unknownOutcome"),
    }));
  }, [outcomeLabels, result, t]);

  async function importRecipePages(
    aggregate: LegacyImportResult,
    bootstrap: UserAssetsBootstrap,
  ): Promise<void> {
    const importedIds = new Set(bootstrap.importLedger.recipes);
    let cursor: string | null = null;
    do {
      const page = await readRecipes(cursor, LEGACY_PAGE_SIZE);
      const recipes = page.items
        .filter((item) => !importedIds.has(item.id))
        .map(withoutRevision);
      for (const recipeBatch of splitByLegacyBatchBytes(recipes, "recipes")) {
        const batch = await importLegacyUserAssets({
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: recipeBatch,
          executions: [],
        });
        addEffectiveResults(aggregate, "recipes", batch.recipes);
      }
      if (cursor !== null && page.nextCursor === cursor)
        throw new Error("Legacy recipe cursor did not advance.");
      cursor = page.nextCursor;
    } while (cursor);
  }

  async function importExecutionPages(
    aggregate: LegacyImportResult,
    bootstrap: UserAssetsBootstrap,
  ): Promise<void> {
    const importedIds = new Set(bootstrap.importLedger.executions);
    let cursor: string | null = null;
    do {
      const page = await readExecutions(cursor, LEGACY_PAGE_SIZE);
      const executions = page.items.filter((item) => !importedIds.has(item.id));
      for (const executionBatch of splitByLegacyBatchBytes(
        executions,
        "executions",
      )) {
        const batch = await importLegacyUserAssets({
          source: SOURCE,
          confirmSubject: bootstrap.subject,
          recipes: [],
          executions: executionBatch,
        });
        const retryIds = new Set(
          batch.executions
            .filter((item) => item.outcome === "missing_parent" && item.id)
            .map((item) => item.id as string),
        );
        let effective = batch.executions;
        if (retryIds.size > 0) {
          const retry = await importLegacyUserAssets({
            source: SOURCE,
            confirmSubject: bootstrap.subject,
            recipes: [],
            executions: executionBatch.filter((item) => retryIds.has(item.id)),
          });
          effective = effectiveRetryResults(batch.executions, retry.executions);
        }
        addEffectiveResults(aggregate, "executions", effective);
      }
      if (cursor !== null && page.nextCursor === cursor)
        throw new Error("Legacy execution cursor did not advance.");
      cursor = page.nextCursor;
    } while (cursor);
  }

  async function handleImport(): Promise<void> {
    if (!pending || importing) return;
    setImporting(true);
    try {
      const nextResult = createAggregate();
      await importRecipePages(nextResult, pending.bootstrap);
      await importExecutionPages(nextResult, pending.bootstrap);
      setResult(nextResult);
      onImported();
      toastSuccess(t("dataRecipes.import.complete"));
    } catch (error) {
      const detail =
        typeof error === "object" && error !== null && "detail" in error
          ? (error.detail as { code?: unknown })
          : null;
      const code = typeof detail?.code === "string" ? detail.code : undefined;
      toastError(
        t("dataRecipes.import.failed"),
        (code ? reasonLabels[code] : undefined) ??
          t("dataRecipes.import.failedDescription"),
      );
    } finally {
      setImporting(false);
    }
  }

  if (!pending) return null;

  const details = result ? [...result.recipes, ...result.executions] : [];
  const recipeCountKey =
    pending.recipeCount === 1
      ? "dataRecipes.import.recipeCountOne"
      : "dataRecipes.import.recipeCountOther";
  const executionCountKey =
    pending.executionCount === 1
      ? "dataRecipes.import.executionCountOne"
      : "dataRecipes.import.executionCountOther";

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => !importing && setOpen(nextOpen)}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("dataRecipes.import.title")}</DialogTitle>
          <DialogDescription>
            {t("dataRecipes.import.description", {
              account: pending.bootstrap.subject,
            })}
          </DialogDescription>
        </DialogHeader>
        {result ? (
          <div className="space-y-3 rounded-lg border bg-muted/20 p-3 text-sm">
            <div className="space-y-2">
              {summary.map((item) => (
                <div key={item.outcome} className="flex justify-between gap-4">
                  <span>{item.label}</span>
                  <span className="font-medium tabular-nums">{item.count}</span>
                </div>
              ))}
            </div>
            {details.length > 0 ? (
              <ul className="max-h-36 space-y-1 overflow-y-auto border-t pt-2 text-xs text-muted-foreground">
                {details.map((item, index) => (
                  <li key={`${item.id ?? "unknown"}-${index}`}>
                    <span className="font-mono">
                      {item.id ?? t("dataRecipes.import.unknownId")}
                    </span>
                    {` — ${outcomeLabels[item.outcome] ?? t("dataRecipes.import.unknownOutcome")}`}
                    {item.reason
                      ? ` (${reasonLabels[item.reason] ?? t("dataRecipes.import.reasons.unknown")})`
                      : ""}
                    {item.redactedPaths?.length
                      ? `: ${t("dataRecipes.import.redactedFields", {
                          fields: item.redactedPaths.join(", "),
                        })}`
                      : ""}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {t(recipeCountKey, { count: pending.recipeCount })}
            {" · "}
            {t(executionCountKey, { count: pending.executionCount })}
          </p>
        )}
        <DialogFooter>
          {result ? (
            <Button type="button" onClick={() => setOpen(false)}>
              {t("dataRecipes.import.close")}
            </Button>
          ) : (
            <>
              <Button
                type="button"
                variant="outline"
                onClick={() => setOpen(false)}
              >
                {t("dataRecipes.import.cancel")}
              </Button>
              <Button
                type="button"
                disabled={importing}
                onClick={() => void handleImport()}
              >
                {importing
                  ? t("dataRecipes.import.importing")
                  : t("dataRecipes.import.confirm")}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
