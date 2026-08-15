// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsDialogStore } from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  admitRecord,
  compileRecord,
  deprecateRecord,
  fetchAdapters,
  fetchCompiled,
  fetchContradictions,
  fetchAdmissions,
  fetchRecords,
  fetchRollouts,
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

const HYGIENE_LABEL: Record<
  "compact" | "contradictions" | "admissions" | "rollouts",
  TranslationKey
> = {
  compact: "unforgettable.hygiene.compact",
  contradictions: "unforgettable.hygiene.contradictions",
  admissions: "unforgettable.hygiene.admissions",
  rollouts: "unforgettable.hygiene.rollouts",
};

function statusForTab(tab: WorkspaceTab): string | undefined {
  if (tab === "inbox") return "proposed";
  if (tab === "notebook") return "active";
  if (tab === "archive") return "deprecated,superseded,rejected";
  return undefined;
}

function shortId(id: string) {
  return id.slice(0, 8);
}

function ageLabel(iso?: string) {
  if (!iso) return "";
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return "";
  const delta = Date.now() - then;
  const minutes = Math.round(delta / 60000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
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
  const [hygiene, setHygiene] = useState<{
    compact?: string;
    contradictions?: string;
    admissions?: string;
    rollouts?: string;
  }>({});
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

function Inspector({
  record,
  tab,
  draftTitle,
  draftBody,
  force,
  busy,
  onTitle,
  onBody,
  onForce,
  onAdmit,
  onReject,
  onSave,
  onCompile,
  onUncompile,
  onDeprecate,
}: {
  record: MemoryRecord | null;
  tab: WorkspaceTab;
  draftTitle: string;
  draftBody: string;
  force: boolean;
  busy: boolean;
  onTitle: (value: string) => void;
  onBody: (value: string) => void;
  onForce: (value: boolean) => void;
  onAdmit: () => void;
  onReject: () => void;
  onSave: () => void;
  onCompile: () => void;
  onUncompile: () => void;
  onDeprecate: () => void;
}) {
  const t = useT();
  if (!record) {
    return (
      <div className="rounded-xl border border-dashed border-border/70 p-6 text-sm text-muted-foreground">
        {t("unforgettable.inspector.noSelection")}
      </div>
    );
  }
  const proposed = record.status === "proposed";
  const canAdmit =
    record.status === "proposed" ||
    record.status === "deprecated" ||
    force;
  const canReject = record.status === "proposed";
  const canCompile = record.kind === "procedure" && record.status === "active";
  return (
    <div className="flex min-h-0 flex-col gap-3 rounded-xl border border-border/70 p-4">
      <div className="text-xs text-muted-foreground">
        {shortId(record.id)} · {record.kind} · {record.status} ·{" "}
        {record.provenance}
        {record.source_episode_id
          ? ` · ep ${shortId(record.source_episode_id)}`
          : ""}
      </div>
      <Input
        value={draftTitle}
        onChange={(event) => onTitle(event.target.value)}
        disabled={!proposed || busy}
      />
      <Textarea
        value={draftBody}
        onChange={(event) => onBody(event.target.value)}
        disabled={!proposed || busy}
        className="min-h-40"
      />
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={force}
          onChange={(event) => onForce(event.target.checked)}
        />
        {t("unforgettable.inspector.force")}
      </label>
      <div className="flex flex-wrap gap-2">
        {proposed ? (
          <Button type="button" size="sm" disabled={busy} onClick={onSave}>
            {t("unforgettable.inspector.save")}
          </Button>
        ) : null}
        {canAdmit ? (
          <Button type="button" size="sm" disabled={busy} onClick={onAdmit}>
            {t("unforgettable.inspector.admit")}
          </Button>
        ) : null}
        {canReject ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onReject}
          >
            {t("unforgettable.inspector.reject")}
          </Button>
        ) : null}
        {canCompile && tab !== "standing" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onCompile}
          >
            {t("unforgettable.inspector.compile")}
          </Button>
        ) : null}
        {tab === "standing" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onUncompile}
          >
            {t("unforgettable.inspector.uncompile")}
          </Button>
        ) : null}
        {record.status === "active" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={onDeprecate}
          >
            {t("unforgettable.inspector.deprecate")}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

function HygienePanel({
  busy,
  hygiene,
  onCompact,
  onReport,
}: {
  busy: boolean;
  hygiene: {
    compact?: string;
    contradictions?: string;
    admissions?: string;
    rollouts?: string;
  };
  onCompact: (apply: boolean) => void;
  onReport: Dispatch<
    SetStateAction<{
      compact?: string;
      contradictions?: string;
      admissions?: string;
      rollouts?: string;
    }>
  >;
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
          error instanceof Error ? error.message : t("unforgettable.errors.load"),
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

function SidecarPanel({
  adapters,
  busy,
  onPromote,
  onRollback,
}: {
  adapters: AdapterRow[];
  busy: boolean;
  onPromote: (id: string) => void;
  onRollback: () => void;
}) {
  const t = useT();
  if (adapters.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/70 p-6 text-sm text-muted-foreground">
        {t("unforgettable.sidecar.empty")}
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={onRollback}
        >
          {t("unforgettable.sidecar.rollback")}
        </Button>
      </div>
      <ul className="divide-y divide-border/60 rounded-xl border border-border/70">
        {adapters.map((adapter) => (
          <li
            key={adapter.id}
            className="flex items-center justify-between gap-3 px-3 py-2.5"
          >
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">
                {shortId(adapter.id)} · {adapter.status}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {adapter.recipe} · {adapter.backend}
              </div>
            </div>
            {adapter.status !== "promoted" ? (
              <Button
                type="button"
                size="sm"
                disabled={busy}
                onClick={() => onPromote(adapter.id)}
              >
                {t("unforgettable.sidecar.promote")}
              </Button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
