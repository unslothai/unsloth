// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { type TranslationKey, useT } from "@/i18n";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type HostPrepJob,
  type RuntimeInstallJob,
  type SandboxSettingsUpdate,
  type SandboxStatus,
  type SandboxToolStatus,
  loadHostPreparation,
  loadRuntimeInstall,
  loadSandboxStatus,
  startHostPreparation,
  startRuntimeInstall,
  updateSandboxSettings,
} from "../api/sandbox-isolation";
import { isSettingsRouteAbsent } from "../api/settings-route-absent";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  HOST_PREP_POLL_MS,
  type HostPrepStatus,
  isOlderJob,
  jobOutputLines,
  jobResult,
  shouldPollJob,
  toolRowView,
  windowsView,
} from "./sandbox-tab-state";

const PREP_STATUS_KEYS: Record<HostPrepStatus, TranslationKey | null> = {
  runtimeMissing: null,
  off: null,
  prepared: "settings.sandbox.prepPrepared",
  needsPreparing: "settings.sandbox.prepNeeds",
  needsPreparingAgain: "settings.sandbox.prepNeedsAgain",
  unknown: "settings.sandbox.prepUnknown",
};

const NOTE_CLASS =
  "max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs";

function ToolRow({
  label,
  tool,
  shell,
}: {
  label: string;
  tool: SandboxToolStatus;
  shell?: SandboxStatus["terminalShell"];
}) {
  const t = useT();
  const view = toolRowView(tool, shell ?? null);
  const description = view.isolated
    ? view.runsInCmd
      ? t("settings.sandbox.runsInCmd")
      : undefined
    : view.reason || t("settings.sandbox.noReason");
  return (
    <SettingsRow label={label} description={description}>
      <Badge variant={view.isolated ? "secondary" : "outline"}>
        {view.isolated
          ? t("settings.sandbox.osIsolation", { backend: view.backendLabel })
          : t("settings.sandbox.softwareSafeguards")}
      </Badge>
    </SettingsRow>
  );
}

export function SandboxTab() {
  const t = useT();
  const [status, setStatus] = useState<SandboxStatus | null>(null);
  const [job, setJob] = useState<HostPrepJob | null>(null);
  const [runtimeJob, setRuntimeJob] = useState<RuntimeInstallJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Save and prepare failures sit with the Windows controls, e.g. the remote-browser refusal.
  const [actionError, setActionError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [restored, setRestored] = useState<number | null>(null);
  const [absent, setAbsent] = useState(false);
  const mounted = useRef(true);
  // Bumped by every read and save: an older status read must not overwrite a newer answer.
  const statusGeneration = useRef(0);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Callers set `loading`: a synchronous setState from the mount effect would cascade a render.
  const refresh = useCallback(
    (force: boolean) => {
      const generation = ++statusGeneration.current;
      const current = () =>
        mounted.current && generation === statusGeneration.current;
      return loadSandboxStatus(force, t("settings.sandbox.loadError"))
        .then((loaded) => {
          if (!current()) return;
          setStatus(loaded);
          setError(null);
        })
        .catch((loadError) => {
          if (!current()) return;
          if (isSettingsRouteAbsent(loadError)) {
            setAbsent(true);
            return;
          }
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("settings.sandbox.loadError"),
          );
        })
        .finally(() => {
          if (current()) setLoading(false);
        });
    },
    [t],
  );

  useEffect(() => {
    void refresh(false);
    // A job started from another window, or before this tab was reopened, keeps reporting here.
    void loadHostPreparation(t("settings.sandbox.prepareError"))
      .then((current) => {
        if (!mounted.current || current.state === "idle") return;
        setJob((shown) => (isOlderJob(current, shown) ? shown : current));
      })
      .catch(() => undefined);
    void loadRuntimeInstall(t("settings.sandbox.installRuntimeError"))
      .then((current) => {
        if (!mounted.current || current.state === "idle") return;
        setRuntimeJob((shown) =>
          isOlderJob(current, shown) ? shown : current,
        );
      })
      .catch(() => undefined);
  }, [refresh, t]);

  useEffect(() => {
    if (!shouldPollJob(runtimeJob)) return;
    const timer = window.setTimeout(() => {
      void loadRuntimeInstall(t("settings.sandbox.installRuntimeError"))
        .then((next) => {
          if (!mounted.current) return;
          setRuntimeJob(next);
          if (!shouldPollJob(next)) {
            setLoading(true);
            void refresh(true);
          }
        })
        .catch((pollError) => {
          if (!mounted.current) return;
          setActionError(
            pollError instanceof Error
              ? pollError.message
              : t("settings.sandbox.installRuntimeError"),
          );
          setRuntimeJob(null);
        });
    }, HOST_PREP_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [runtimeJob, refresh, t]);

  // Poll while the elevated helper runs; the status is re-read once it finishes.
  useEffect(() => {
    if (!shouldPollJob(job)) return;
    const timer = window.setTimeout(() => {
      void loadHostPreparation(t("settings.sandbox.prepareError"))
        .then((next) => {
          if (!mounted.current) return;
          setJob(next);
          if (!shouldPollJob(next)) {
            setLoading(true);
            void refresh(true);
          }
        })
        .catch((pollError) => {
          if (!mounted.current) return;
          setActionError(
            pollError instanceof Error
              ? pollError.message
              : t("settings.sandbox.prepareError"),
          );
          setJob(null);
        });
    }, HOST_PREP_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [job, refresh, t]);

  const save = async (update: SandboxSettingsUpdate) => {
    const generation = ++statusGeneration.current;
    setSaving(true);
    setActionError(null);
    setRestored(null);
    try {
      const next = await updateSandboxSettings(
        update,
        t("settings.sandbox.saveError"),
      );
      if (!mounted.current) return;
      if (generation === statusGeneration.current) setStatus(next);
      setLoading(false);
      if (next.restored) setRestored(next.restored);
    } catch (saveError) {
      if (!mounted.current) return;
      setActionError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.sandbox.saveError"),
      );
    } finally {
      if (mounted.current) setSaving(false);
    }
  };

  const prepare = async () => {
    setActionError(null);
    try {
      const started = await startHostPreparation(
        t("settings.sandbox.prepareError"),
      );
      if (!mounted.current) return;
      setJob(started);
      if (!shouldPollJob(started)) {
        setLoading(true);
        void refresh(true);
      }
    } catch (prepareError) {
      if (!mounted.current) return;
      setActionError(
        prepareError instanceof Error
          ? prepareError.message
          : t("settings.sandbox.prepareError"),
      );
    }
  };

  const installRuntime = async () => {
    setActionError(null);
    try {
      const started = await startRuntimeInstall(
        t("settings.sandbox.installRuntimeError"),
      );
      if (!mounted.current) return;
      setRuntimeJob(started);
      if (!shouldPollJob(started)) {
        setLoading(true);
        void refresh(true);
      }
    } catch (installError) {
      if (!mounted.current) return;
      setActionError(
        installError instanceof Error
          ? installError.message
          : t("settings.sandbox.installRuntimeError"),
      );
    }
  };

  const windows = status?.windows ?? null;
  const view = windows ? windowsView(windows, job, saving) : null;
  const runtimeRunning = runtimeJob?.state === "running";
  const runtimeFailed = jobResult(runtimeJob) === "failed";
  const runtimeOutput = jobOutputLines(runtimeJob);
  const result = jobResult(job);
  const outputLines = jobOutputLines(job);
  const prepKey = view ? PREP_STATUS_KEYS[view.prep] : null;

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.sandbox.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.sandbox.title")}
        </h1>
        <p
          data-settings-label={t("settings.sandbox.description")}
          className="text-xs text-muted-foreground"
        >
          {t("settings.sandbox.description")}
        </p>
      </header>

      {absent ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.sandbox.unsupported")}
        </p>
      ) : (
        <>
          <SettingsSection title={t("settings.sandbox.toolsSection")}>
            {status ? (
              <>
                <ToolRow
                  label={t("settings.sandbox.python")}
                  tool={status.python}
                />
                <ToolRow
                  label={t("settings.sandbox.terminal")}
                  tool={status.terminal}
                  shell={status.terminalShell}
                />
              </>
            ) : null}
            <div className="flex items-center justify-end gap-2 py-2">
              {error ? (
                <span className={`${NOTE_CLASS} text-destructive`}>
                  {error}
                </span>
              ) : null}
              <Button
                size="sm"
                variant="outline"
                // A read started mid-save can see the old value and would drop the save's answer.
                disabled={loading || saving}
                onClick={() => {
                  setLoading(true);
                  void refresh(true);
                }}
              >
                {loading ? (
                  <Spinner />
                ) : (
                  <HugeiconsIcon strokeWidth={1.75} icon={RefreshIcon} />
                )}
                {t("settings.sandbox.refresh")}
              </Button>
            </div>
          </SettingsSection>

          {windows && view ? (
            <SettingsSection
              title={t("settings.sandbox.windowsSection")}
              description={t("settings.sandbox.windowsDescription")}
            >
              {view.unsupported ? (
                <p className="py-3 text-sm text-muted-foreground">
                  {view.unsupported === "arch"
                    ? t("settings.sandbox.unsupportedArch")
                    : t("settings.sandbox.unsupportedBuild")}
                </p>
              ) : view.runtimeMissing ? (
                <div className="flex flex-col gap-2 py-3">
                  <SettingsRow
                    label={t("settings.sandbox.runtimeLabel")}
                    description={t("settings.sandbox.runtimeMissing")}
                  >
                    {view.showInstallRuntime ? (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={runtimeRunning}
                        onClick={() => void installRuntime()}
                      >
                        {runtimeRunning ? <Spinner /> : null}
                        {runtimeRunning
                          ? t("settings.sandbox.installingRuntime")
                          : t("settings.sandbox.installRuntime")}
                      </Button>
                    ) : null}
                  </SettingsRow>
                  {runtimeFailed ? (
                    <p className="text-xs text-destructive">
                      {runtimeJob?.note ||
                        t("settings.sandbox.installRuntimeFailed")}
                    </p>
                  ) : null}
                  {runtimeOutput.length > 0 ? (
                    <pre className="whitespace-pre-wrap break-words font-mono text-ui-11 text-muted-foreground">
                      {runtimeOutput.join("\n")}
                    </pre>
                  ) : null}
                  {actionError ? (
                    <p className="text-xs text-destructive">{actionError}</p>
                  ) : null}
                </div>
              ) : (
                <>
                  <SettingsRow
                    label={t("settings.sandbox.optInLabel")}
                    description={t("settings.sandbox.optInDescription")}
                  >
                    <div className="flex flex-col items-end gap-1">
                      <Switch
                        aria-label={t("settings.sandbox.optInLabel")}
                        checked={view.optInChecked}
                        disabled={view.optInDisabled}
                        onCheckedChange={(allowDaclFallback) =>
                          void save({ allowDaclFallback })
                        }
                      />
                      {view.optInLocked ? (
                        <span className={`${NOTE_CLASS} text-muted-foreground`}>
                          {t("settings.sandbox.lockedDacl")}
                        </span>
                      ) : null}
                    </div>
                  </SettingsRow>
                  <p className="pb-2 text-xs text-muted-foreground leading-relaxed">
                    {t("settings.sandbox.disclosure")}
                  </p>
                  {restored ? (
                    <p className="pb-2 text-xs text-muted-foreground">
                      {t("settings.sandbox.restored", { count: restored })}
                    </p>
                  ) : null}
                  {view.showGrantsRow ? (
                    <SettingsRow
                      label={t("settings.sandbox.grantsLabel")}
                      description={t("settings.sandbox.grantsDescription")}
                    >
                      <div className="flex flex-col items-end gap-1">
                        <Switch
                          aria-label={t("settings.sandbox.grantsLabel")}
                          checked={view.grantsChecked}
                          disabled={view.grantsDisabled}
                          onCheckedChange={(persistentReadGrants) =>
                            void save({ persistentReadGrants })
                          }
                        />
                        {view.grantsLocked ? (
                          <span
                            className={`${NOTE_CLASS} text-muted-foreground`}
                          >
                            {t("settings.sandbox.lockedGrants")}
                          </span>
                        ) : null}
                      </div>
                    </SettingsRow>
                  ) : null}
                  {prepKey ? (
                    <SettingsRow
                      label={t("settings.sandbox.hostPrepLabel")}
                      description={t(prepKey)}
                    >
                      <div className="flex flex-col items-end gap-1">
                        {view.showPrepareButton ? (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={view.prepareDisabled}
                            onClick={() => void prepare()}
                          >
                            {job?.state === "running" ? <Spinner /> : null}
                            {t("settings.sandbox.prepareButton")}
                          </Button>
                        ) : null}
                        {job?.state === "running" ? (
                          <span
                            className={`${NOTE_CLASS} text-muted-foreground`}
                          >
                            {t("settings.sandbox.preparing")}
                          </span>
                        ) : null}
                        {result === "succeeded" ? (
                          <span
                            className={`${NOTE_CLASS} text-muted-foreground`}
                          >
                            {t("settings.sandbox.prepSucceeded")}
                          </span>
                        ) : null}
                        {result === "declined" ? (
                          <span className={`${NOTE_CLASS} text-destructive`}>
                            {t("settings.sandbox.prepDeclined")}
                          </span>
                        ) : null}
                        {result === "failed" ? (
                          <span className={`${NOTE_CLASS} text-destructive`}>
                            {t("settings.sandbox.prepFailed")}
                          </span>
                        ) : null}
                        {outputLines.length > 0 ? (
                          <pre className="max-w-[calc(360px*var(--ui-space-scale,1))] whitespace-pre-wrap break-words text-right font-mono text-ui-11 text-muted-foreground">
                            {outputLines.join("\n")}
                          </pre>
                        ) : null}
                      </div>
                    </SettingsRow>
                  ) : null}
                  {actionError ? (
                    <p className="pb-2 text-xs text-destructive">
                      {actionError}
                    </p>
                  ) : null}
                </>
              )}
            </SettingsSection>
          ) : null}
        </>
      )}
    </div>
  );
}
