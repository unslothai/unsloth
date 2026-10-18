// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Voice > Voice mode: the addressed gate, the decision log toggle, one card
// per voice model from /voice/laya/stats, the JSONL export and the clear button.

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { listStoredChatThreads } from "@/features/chat/utils/chat-history-storage";
import {
  clearLayaLog,
  exportLayaLog,
  fetchLayaStats,
  type LayaModel,
  type LayaModelStats,
  type LayaStats,
} from "@/features/chat/voice/laya-log-api";
import { useT } from "@/i18n";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import { useVoiceSettingsStore } from "../stores/voice-settings-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const MODELS: LayaModel[] = ["judge", "interject", "filler", "addressed"];
// The backend's voice_log.ERROR_KINDS.
const ERROR_KINDS = [
  "finished_but_continued",
  "held_but_flushed",
  "ignored_but_repeated",
  "queued_but_reasked",
  "echoed",
  "too_late",
  "missed_address",
] as const;
type ErrorKind = (typeof ERROR_KINDS)[number];
const isErrorKind = (k: string): k is ErrorKind => (ERROR_KINDS as readonly string[]).includes(k);
const CARDS = "grid grid-cols-[repeat(auto-fit,minmax(calc(11rem*var(--ui-space-scale,1)),1fr))] gap-2 py-3";
const ALL_CHATS = "__all__";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function ModelCard({
  model,
  stats,
}: {
  model: LayaModel;
  stats: LayaModelStats | undefined;
}) {
  const t = useT();
  const verdicts = stats?.verdicts ?? 0;
  const errors = Object.entries(stats?.errors ?? {});
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-lg border border-border/70 px-3 py-2.5">
      <span className="text-xs font-medium text-muted-foreground">
        {t(`settings.voice.voiceMode.models.${model}`)}
      </span>
      {verdicts === 0 ? (
        <span className="text-sm text-muted-foreground">
          {t("settings.voice.voiceMode.noRows")}
        </span>
      ) : (
        <>
          <span className="text-lg font-semibold text-foreground">
            {stats?.accuracy === null || stats?.accuracy === undefined
              ? t("settings.voice.voiceMode.noOutcomes")
              : `${Math.round(stats.accuracy * 100)}%`}
          </span>
          <span className="text-xs text-muted-foreground">
            {t("settings.voice.voiceMode.verdictsAndLatency", {
              n: String(verdicts),
              ms: stats?.latency_p50_ms === null || stats?.latency_p50_ms === undefined
                ? "–"
                : String(Math.round(stats.latency_p50_ms)),
            })}
          </span>
          {errors.map(([kind, count]) => (
            <span key={kind} className="text-xs text-muted-foreground">
              {isErrorKind(kind) ? t(`settings.voice.voiceMode.errors.${kind}`) : kind}: {count}
            </span>
          ))}
        </>
      )}
    </div>
  );
}

export function VoiceDecisionLogSection() {
  const t = useT();
  const enabled = useVoiceSettingsStore((s) => s.voiceLogEnabled);
  const setEnabled = useVoiceSettingsStore((s) => s.setVoiceLogEnabled);
  const addressedGate = useVoiceSettingsStore((s) => s.voiceAddressedGate);
  const setAddressedGate = useVoiceSettingsStore((s) => s.setVoiceAddressedGate);
  const exportedAt = useVoiceSettingsStore((s) => s.voiceLogExportedAt);
  const setExportedAt = useVoiceSettingsStore((s) => s.setVoiceLogExportedAt);
  const [stats, setStats] = useState<LayaStats | null>(null);
  const [threads, setThreads] = useState<{ id: string; title: string }[]>([]);
  const [model, setModel] = useState<LayaModel | "all">("all");
  const [only, setOnly] = useState<"disagreements" | "all">("disagreements");
  const [chat, setChat] = useState(ALL_CHATS);
  const [sinceLast, setSinceLast] = useState(false);
  const [busy, setBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const refresh = useCallback(() => {
    void fetchLayaStats().then((s) => setStats(s));
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh, enabled]);
  useEffect(() => {
    void listStoredChatThreads({ includeArchived: true })
      .then((list) => setThreads(list.map((th) => ({ id: th.id, title: th.title }))))
      .catch(() => setThreads([]));
  }, []);

  const rows = stats?.rows ?? 0;
  const doExport = async () => {
    setBusy(true);
    try {
      const n = await exportLayaLog({
        model,
        only,
        threadId: chat === ALL_CHATS ? null : chat,
        since: sinceLast && exportedAt ? new Date(exportedAt).toISOString() : null,
      });
      setExportedAt(Date.now());
      toast.success(t("settings.voice.voiceMode.exported", { n: String(n) }));
    } catch (error) {
      if (!isDownloadCancelled(error)) toast.error(t("settings.voice.voiceMode.exportFailed"));
    } finally {
      setBusy(false);
    }
  };
  const doClear = async () => {
    setConfirmClear(false);
    setBusy(true);
    try {
      await clearLayaLog();
      refresh();
    } catch {
      toast.error(t("settings.voice.voiceMode.clearFailed"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <SettingsSection title={t("settings.voice.voiceMode.sectionTitle")}>
      <SettingsRow
        label={t("settings.voice.voiceMode.addressedLabel")}
        description={t("settings.voice.voiceMode.addressedDescription")}
      >
        <Switch checked={addressedGate} onCheckedChange={setAddressedGate} />
      </SettingsRow>
      <SettingsRow
        label={t("settings.voice.voiceMode.logLabel")}
        description={t("settings.voice.voiceMode.logDescription")}
      >
        <Switch checked={enabled} onCheckedChange={setEnabled} />
      </SettingsRow>
      <div className={CARDS}>
        {MODELS.map((m) => (
          <ModelCard key={m} model={m} stats={stats?.models[m]} />
        ))}
      </div>
      <SettingsRow
        label={t("settings.voice.voiceMode.exportLabel")}
        description={
          rows === 0
            ? t("settings.voice.voiceMode.noRows")
            : t("settings.voice.voiceMode.exportDescription", {
                rows: String(rows),
                size: formatBytes(stats?.log_bytes ?? 0),
              })
        }
        className="items-start"
      >
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Select value={model} onValueChange={(v) => setModel(v as LayaModel | "all")}>
            <SelectTrigger size="sm" className="w-36" aria-label={t("settings.voice.voiceMode.modelLabel")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("settings.voice.voiceMode.models.all")}</SelectItem>
              {MODELS.map((m) => (
                <SelectItem key={m} value={m}>
                  {t(`settings.voice.voiceMode.models.${m}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={only} onValueChange={(v) => setOnly(v === "all" ? "all" : "disagreements")}>
            <SelectTrigger size="sm" className="w-44" aria-label={t("settings.voice.voiceMode.onlyLabel")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="disagreements">{t("settings.voice.voiceMode.onlyDisagreements")}</SelectItem>
              <SelectItem value="all">{t("settings.voice.voiceMode.onlyAll")}</SelectItem>
            </SelectContent>
          </Select>
          <Select value={chat} onValueChange={setChat}>
            <SelectTrigger size="sm" className="w-44" aria-label={t("settings.voice.voiceMode.chatLabel")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_CHATS}>{t("settings.voice.voiceMode.allChats")}</SelectItem>
              {threads.map((th) => (
                <SelectItem key={th.id} value={th.id}>
                  <span className="truncate">{th.title}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={sinceLast ? "since" : "all"}
            onValueChange={(v) => setSinceLast(v === "since")}
          >
            <SelectTrigger size="sm" className="w-40" aria-label={t("settings.voice.voiceMode.sinceLabel")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("settings.voice.voiceMode.sinceAll")}</SelectItem>
              <SelectItem value="since" disabled={!exportedAt}>
                {t("settings.voice.voiceMode.sinceLast")}
              </SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" disabled={busy || rows === 0} onClick={() => void doExport()}>
            {t("settings.voice.voiceMode.exportButton")}
          </Button>
        </div>
      </SettingsRow>
      <SettingsRow
        label={t("settings.voice.voiceMode.clearLabel")}
        description={t("settings.voice.voiceMode.clearDescription")}
      >
        <Button variant="outline" size="sm" disabled={busy || rows === 0} onClick={() => setConfirmClear(true)}>
          {t("settings.voice.voiceMode.clearButton")}
        </Button>
      </SettingsRow>
      <AlertDialog open={confirmClear} onOpenChange={(open) => !open && setConfirmClear(false)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("settings.voice.voiceMode.clearConfirmTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.voice.voiceMode.clearConfirmDescription", { rows: String(rows) })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={() => void doClear()}>
              {t("settings.voice.voiceMode.clearButton")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SettingsSection>
  );
}
