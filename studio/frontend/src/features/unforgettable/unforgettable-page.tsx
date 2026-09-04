// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useSettingsDialogStore } from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  admitRecord,
  compileRecord,
  deprecateRecord,
  fetchAdapters,
  fetchCompiled,
  fetchRecords,
  fetchSummary,
  patchProposedRecord,
  promoteAdapter,
  rejectRecord,
  rollbackAdapter,
  runCompact,
  runMine,
  runReview,
  uncompileRecord,
} from "./api/memory-api";
import { ageLabel, shortId } from "./format";
import { HygienePanel, type HygieneReport } from "./hygiene-panel";
import { Inspector } from "./inspector";
import { SidecarPanel } from "./sidecar-panel";
import type {
  AdapterRow,
  MemoryRecord,
  MemorySummary,
  OperatorItem,
  WorkspaceTab,
} from "./types";

const WORKSPACES: WorkspaceTab[] = [
  "inbox",
  "notebook",
  "standing",
  "archive",
  "sidecar",
  "hygiene",
];

const WORKSPACE_LABEL: Record<WorkspaceTab, TranslationKey> = {
  inbox: "unforgettable.workspace.inbox",
  notebook: "unforgettable.workspace.notebook",
  standing: "unforgettable.workspace.standing",
  archive: "unforgettable.workspace.archive",
  sidecar: "unforgettable.workspace.sidecar",
  hygiene: "unforgettable.workspace.hygiene",
};

function statusForTab(tab: WorkspaceTab): string | undefined {
  if (tab === "inbox") return "proposed";
  if (tab === "notebook") return "active";
  if (tab === "archive") return "deprecated,superseded,rejected";
  return undefined;
}

export function UnforgettablePage() {
  const t = useT();
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const [summary, setSummary] = useState<MemorySummary | null>(null);
  const [tab, setTab] = useState<WorkspaceTab>("inbox");
  const [query, setQuery] = useState("");
  const [committedQuery, setCommittedQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<string | null>(null);
  const [trustFilter, setTrustFilter] = useState<string | null>(null);
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [draftBody, setDraftBody] = useState("");
  const [busy, setBusy] = useState(false);
  const [force, setForce] = useState(false);
  const [adapters, setAdapters] = useState<AdapterRow[]>([]);
  const [hygiene, setHygiene] = useState<HygieneReport>({});
  const [operatorReport, setOperatorReport] = useState<OperatorItem[] | null>(
    null,
  );

  const selected = useMemo(
    () => records.find((row) => row.id === selectedId) ?? null,
    [records, selectedId],
  );

  const reloadSummary = useCallback(async () => {
    const next = await fetchSummary();
    setSummary(next);
  }, []);

  const reloadList = useCallback(async () => {
    if (tab === "sidecar") {
      const data = await fetchAdapters();
      setAdapters(data.adapters);
      setRecords([]);
      return;
    }
    if (tab === "standing") {
      const data = await fetchCompiled();
      setRecords(data.records);
      return;
    }
    if (tab === "hygiene") {
      setRecords([]);
      return;
    }
    const data = await fetchRecords({
      status: statusForTab(tab),
      kind: kindFilter ?? undefined,
      provenance: trustFilter ?? undefined,
      q: committedQuery.trim() || undefined,
      limit: 60,
    });
    const rows =
      tab === "inbox"
        ? data.records.filter((row) => row.kind !== "episode")
        : data.records;
    setRecords(rows);
    setSelectedId((current) =>
      current && rows.some((row) => row.id === current)
        ? current
        : (rows[0]?.id ?? null),
    );
  }, [committedQuery, kindFilter, tab, trustFilter]);

  const refresh = useCallback(async () => {
    try {
      await Promise.all([reloadSummary(), reloadList()]);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t("unforgettable.errors.load"),
      );
    }
  }, [reloadList, reloadSummary, t]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selected) {
      setDraftTitle("");
      setDraftBody("");
      return;
    }
    setDraftTitle(selected.title);
    setDraftBody(selected.body);
  }, [selected]);

  async function run(action: () => Promise<unknown>, ok?: string) {
    setBusy(true);
    try {
      await action();
      if (ok) toast.success(ok);
      await refresh();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t("unforgettable.errors.action"),
      );
    } finally {
      setBusy(false);
    }
  }

  const byStatus = summary?.records.by_status ?? {};
  const byKind = summary?.records.by_kind ?? {};
  const byTrust = summary?.records.by_provenance ?? {};
  const inject = summary?.last_inject;
  const standingChars = inject?.standing_chars ?? 0;
  const retrieveChars = inject?.retrieve_chars ?? 0;
  const trajChars = inject?.trajectory_chars ?? 0;
  const injectTotal = Math.max(
    inject?.total_chars ?? standingChars + retrieveChars + trajChars,
    1,
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 p-6 lg:px-8">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            {t("unforgettable.page.title")}
          </h1>
          <p className="truncate text-xs text-muted-foreground" title={summary?.db_path}>
            {summary?.db_path || t("unforgettable.page.loading")}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") setCommittedQuery(query);
            }}
            onBlur={() => setCommittedQuery(query)}
            placeholder={t("unforgettable.page.searchPlaceholder")}
            className="w-56"
          />
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => openSettings("unforgettable")}
          >
            {t("unforgettable.page.settings")}
          </Button>
        </div>
      </header>

      <section className="rounded-xl border border-border/70 bg-card/40 p-4">
        <p className="mb-2 text-xs font-medium text-muted-foreground">
          {t("unforgettable.inject.label")}
        </p>
        {inject ? (
          <>
            <div className="flex h-2 overflow-hidden rounded-full bg-muted">
              <span
                className="bg-control-accent"
                style={{ width: `${(standingChars / injectTotal) * 100}%` }}
              />
              <span
                className="bg-sky-500/80"
                style={{ width: `${(retrieveChars / injectTotal) * 100}%` }}
              />
              <span
                className="bg-amber-500/80"
                style={{ width: `${(trajChars / injectTotal) * 100}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {t("unforgettable.inject.standing")} {standingChars} ·{" "}
              {t("unforgettable.inject.retrieve")} {retrieveChars} ·{" "}
              {t("unforgettable.inject.traj")} {trajChars}
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">
            {t("unforgettable.inject.none")}
          </p>
        )}
      </section>

      <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
        {(
          [
            ["inbox", byStatus.proposed ?? 0, t("unforgettable.tiles.proposed")],
            ["notebook", byStatus.active ?? 0, t("unforgettable.tiles.active")],
            [
              "standing",
              summary?.compiled_count ?? 0,
              t("unforgettable.tiles.compiled"),
            ],
            [
              "archive",
              summary?.archive_count ?? 0,
              t("unforgettable.tiles.archived"),
            ],
            [
              "sidecar",
              summary?.adapters.promoted ?? 0,
              summary?.adapters.promoted_id
                ? shortId(summary.adapters.promoted_id)
                : t("unforgettable.tiles.noneLive"),
            ],
          ] as const
        ).map(([id, count, caption]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={cn(
              "rounded-xl border px-3 py-3 text-left transition-colors",
              tab === id
                ? "border-control-accent bg-control-accent/10"
                : "border-border/70 hover:bg-accent/40",
            )}
          >
            <div className="text-xs font-medium text-muted-foreground">
              {t(WORKSPACE_LABEL[id])}
            </div>
            <div className="font-heading text-2xl font-semibold">{count}</div>
            <div className="text-xs text-muted-foreground">{caption}</div>
          </button>
        ))}
      </section>

      <div className="flex flex-wrap gap-1.5">
        <span className="text-xs text-muted-foreground">
          {t("unforgettable.trust.label")}
        </span>
        {Object.entries(byTrust).map(([name, count]) => (
          <button
            key={name}
            type="button"
            onClick={() =>
              setTrustFilter((current) => (current === name ? null : name))
            }
            className={cn(
              "rounded-full px-2 py-0.5 text-xs",
              trustFilter === name
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50",
            )}
          >
            {name} {count}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-1.5">
        <span className="text-xs text-muted-foreground">
          {t("unforgettable.kinds.label")}
        </span>
        {Object.entries(byKind)
          .filter(([, count]) => count > 0)
          .map(([name, count]) => (
            <button
              key={name}
              type="button"
              onClick={() =>
                setKindFilter((current) => (current === name ? null : name))
              }
              className={cn(
                "rounded-full px-2 py-0.5 text-xs",
                kindFilter === name
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50",
              )}
            >
              {name} {count}
            </button>
          ))}
      </div>

      <nav className="flex flex-wrap gap-1">
        {WORKSPACES.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={cn(
              "rounded-full px-3 py-1.5 text-sm font-medium",
              tab === id
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50",
            )}
          >
            {t(WORKSPACE_LABEL[id])}
          </button>
        ))}
      </nav>

      {tab === "inbox" ? (
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() =>
              void run(async () => {
                const report = await runReview(false);
                setOperatorReport(report.items);
              })
            }
          >
            {t("unforgettable.queue.askVoter")}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() =>
              void run(async () => {
                const report = await runMine(false);
                setOperatorReport(report.items);
              })
            }
          >
            {t("unforgettable.queue.mine")}
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={busy}
            onClick={() =>
              void run(async () => {
                const report = await runReview(true);
                setOperatorReport(report.items);
              }, t("unforgettable.queue.applied"))
            }
          >
            {t("unforgettable.queue.applyReview")}
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={busy}
            onClick={() =>
              void run(async () => {
                const report = await runMine(true);
                setOperatorReport(report.items);
              }, t("unforgettable.queue.applied"))
            }
          >
            {t("unforgettable.queue.applyMine")}
          </Button>
        </div>
      ) : null}

      {operatorReport ? (
        <pre className="max-h-40 overflow-auto rounded-lg bg-muted/50 p-3 text-xs">
          {JSON.stringify(operatorReport, null, 2)}
        </pre>
      ) : null}

      {tab === "hygiene" ? (
        <HygienePanel
          busy={busy}
          hygiene={hygiene}
          onCompact={(apply) =>
            void run(async () => {
              const report = await runCompact(apply);
              setHygiene((prev) => ({
                ...prev,
                compact: JSON.stringify(report, null, 2),
              }));
            })
          }
          onReport={setHygiene}
        />
      ) : tab === "sidecar" ? (
        <SidecarPanel
          adapters={adapters}
          busy={busy}
          onPromote={(id) =>
            void run(() => promoteAdapter(id), t("unforgettable.sidecar.promoted"))
          }
          onRollback={() =>
            void run(() => rollbackAdapter(), t("unforgettable.sidecar.rolledBack"))
          }
        />
      ) : (
        <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(20rem,28rem)]">
          <ul className="min-h-0 overflow-auto rounded-xl border border-border/70">
            {records.length === 0 ? (
              <li className="p-6 text-sm text-muted-foreground">
                {t("unforgettable.queue.empty")}
              </li>
            ) : (
              records.map((row) => (
                <li key={row.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(row.id)}
                    className={cn(
                      "flex w-full flex-col gap-0.5 border-b border-border/50 px-3 py-2.5 text-left",
                      selectedId === row.id
                        ? "bg-accent/60"
                        : "hover:bg-accent/30",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium">
                        {row.title}
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {ageLabel(row.updated_at)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {row.kind} · {row.provenance} · {row.status}
                    </div>
                  </button>
                </li>
              ))
            )}
          </ul>
          <Inspector
            record={selected}
            tab={tab}
            draftTitle={draftTitle}
            draftBody={draftBody}
            force={force}
            busy={busy}
            onTitle={setDraftTitle}
            onBody={setDraftBody}
            onForce={setForce}
            onAdmit={() =>
              selected &&
              void run(
                () => admitRecord(selected.id, force),
                t("unforgettable.inspector.admitted"),
              )
            }
            onReject={() =>
              selected &&
              void run(
                () => rejectRecord(selected.id),
                t("unforgettable.inspector.rejected"),
              )
            }
            onSave={() =>
              selected &&
              void run(
                () =>
                  patchProposedRecord(selected.id, {
                    title: draftTitle,
                    body: draftBody,
                  }),
                t("unforgettable.inspector.saved"),
              )
            }
            onCompile={() =>
              selected &&
              void run(
                () => compileRecord(selected.id),
                t("unforgettable.inspector.compiled"),
              )
            }
            onUncompile={() =>
              selected &&
              void run(
                () => uncompileRecord(selected.id),
                t("unforgettable.inspector.uncompiled"),
              )
            }
            onDeprecate={() =>
              selected &&
              void run(
                () => deprecateRecord(selected.id),
                t("unforgettable.inspector.deprecated"),
              )
            }
          />
        </div>
      )}
    </div>
  );
}
