// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import {
  PermissionModeDropdown,
  SandboxSetupDialog,
  type SandboxSetupJob,
  type SandboxSetupOperation,
  forgetSandboxCapability,
  loadSandboxSetup,
  pickSandboxLevel,
  sandboxSwitchState,
  startSandboxSetup,
  useActivePermissionMode,
  useChatRuntimeStore,
  useSandboxCapability,
} from "@/features/chat";
import { type TranslationKey, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { ShieldIcon } from "@/lib/shield-cog-icon";
import { cn } from "@/lib/utils";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type HostPrepJob,
  type SandboxSettingsUpdate,
  type SandboxStatus,
  type SandboxToolStatus,
  loadHostPreparation,
  loadSandboxStatus,
  startHostPreparation,
  updateSandboxSettings,
} from "../api/sandbox-isolation";
import { isSettingsRouteAbsent } from "../api/settings-route-absent";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import {
  HOST_PREP_POLL_MS,
  type HostPrepStatus,
  isOlderJob,
  jobOutputLines,
  jobResult,
  setupRowView,
  shouldPollJob,
  toolRowView,
  toolRowsQuiet,
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
  quiet = false,
  withRemediation = true,
}: {
  label: string;
  tool: SandboxToolStatus;
  shell?: SandboxStatus["terminalShell"];
  quiet?: boolean;
  /** Off on Windows: the generic MXC remediation repeats the Windows section. */
  withRemediation?: boolean;
}) {
  const t = useT();
  const view = toolRowView(tool, shell ?? null);
  const reason = view.reason || t("settings.sandbox.noReason");
  const description = quiet && !view.isolated ? undefined : view.isolated ? (
    view.runsInCmd ? (
      t("settings.sandbox.runsInCmd")
    ) : undefined
  ) : withRemediation && view.remediation ? (
    <>
      <span className="block">{reason}</span>
      <span className="mt-1 block">{view.remediation}</span>
    </>
  ) : (
    reason
  );
  return (
    <SettingsRow label={label} description={description}>
      <Badge
        variant="outline"
        // Grey and sized like the dropdown pills.
        className="h-8 gap-1.5 border-transparent bg-muted px-3 text-sm text-foreground [&>svg]:size-3.5!"
      >
        {view.isolated ? <HugeiconsIcon icon={ShieldIcon} strokeWidth={1.75} /> : null}
        {view.isolated
          ? t("settings.sandbox.osIsolation", { backend: view.backendLabel })
          : t("settings.sandbox.softwareSafeguards")}
      </Badge>
    </SettingsRow>
  );
}

const SANDBOX_LEVEL_TEXT = {
  off: { name: "settings.sandbox.levelOff", detail: "settings.sandbox.levelFullAccessNote" },
  low: { name: "settings.sandbox.levelLow", detail: "settings.sandbox.levelLowDetail" },
  high: { name: "settings.sandbox.levelHigh", detail: "settings.sandbox.levelHighDetail" },
} as const satisfies Record<string, { name: TranslationKey; detail: TranslationKey }>;

/** "Name: detail", so the text clearly describes the selected option. */
function SelectedOptionDescription({ name, detail }: { name: string; detail: string }) {
  return (
    <>
      <span className="font-medium text-foreground">{name}:</span> {detail}
    </>
  );
}

/** Per account, so every account sees it; the OS sandbox sections below are the owner's. */
function PermissionsSection() {
  const t = useT();
  const permissionsRef = useRef<HTMLElement | null>(null);
  const activePermission = useActivePermissionMode();
  const sandboxLevel = useChatRuntimeStore((s) => s.sandboxLevel);
  const setSandboxLevel = useChatRuntimeStore((s) => s.setSandboxLevel);
  const capability = useSandboxCapability(sandboxLevel === "high");
  const { checked, disabled } = sandboxSwitchState(
    sandboxLevel,
    activePermission.value,
    capability,
  );
  const [setupOpen, setSetupOpen] = useState(false);
  const levelDescriptionId = useId();
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore((s) => s.consumeScrollTarget);

  useEffect(() => {
    if (scrollTarget !== "sandbox-permissions") return;
    const frame = window.requestAnimationFrame(() => {
      permissionsRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      consumeScrollTarget("sandbox-permissions");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);

  // Full access turns the sandbox off: show Disabled and lock Low and High.
  const activeLevel = disabled ? "off" : checked ? "high" : "low";
  const levels = disabled ? (["off", "low", "high"] as const) : (["low", "high"] as const);

  return (
    <SettingsSection
      ref={permissionsRef}
      title={t("settings.general.permissions.sectionTitle")}
      description={t("settings.sandbox.permissionsIntro")}
    >
      <SettingsRow
        label={t("settings.sandbox.permissionLabel")}
        description={
          <SelectedOptionDescription
            name={t(`settings.general.permissions.names.${activePermission.value}`)}
            detail={t(`settings.general.permissions.details.${activePermission.value}`)}
          />
        }
      >
        <PermissionModeDropdown sandboxControls={false} />
      </SettingsRow>
      <SettingsRow
        label={t("settings.sandbox.levelLabel")}
        description={
          <span id={levelDescriptionId}>
            <SelectedOptionDescription
              name={t(SANDBOX_LEVEL_TEXT[activeLevel].name)}
              detail={t(SANDBOX_LEVEL_TEXT[activeLevel].detail)}
            />
          </span>
        }
      >
        {/* Same toggle as Follow-up behavior. */}
        <div
          className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
          role="group"
          aria-label={t("settings.sandbox.levelLabel")}
          aria-describedby={levelDescriptionId}
        >
          {levels.map((level) => {
            const selected = level === activeLevel;
            return (
              <button
                key={level}
                type="button"
                aria-pressed={selected}
                disabled={disabled && !selected}
                onClick={() => {
                  if (selected || level === "off") return;
                  void pickSandboxLevel(level, setSandboxLevel, () => setSetupOpen(true));
                }}
                className={cn(
                  "inline-flex h-8 cursor-pointer items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
                  selected
                    ? "hub-tab-toggle-pill cursor-default text-foreground"
                    : "text-muted-foreground hover:text-foreground disabled:hover:text-muted-foreground",
                )}
              >
                {t(SANDBOX_LEVEL_TEXT[level].name)}
              </button>
            );
          })}
        </div>
      </SettingsRow>
      {/* Its own instance: the chat-page root dialog is not mounted on every page. */}
      <SandboxSetupDialog
        open={setupOpen}
        onOpenChange={setSetupOpen}
        onLearnMore={() =>
          permissionsRef.current?.scrollIntoView({ block: "start", behavior: "smooth" })
        }
      />
    </SettingsSection>
  );
}

export function SandboxTab() {
  const t = useT();
  const isOwner = useIsAccountOwner();
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
      <PermissionsSection />
      {/* Not mounted for managed accounts, so none of the owner-only sandbox routes are called. */}
      {isOwner ? (
        <OsSandboxSections />
      ) : (
        <p className="text-xs text-muted-foreground">{t("settings.sandbox.managedNote")}</p>
      )}
    </div>
  );
}

function OsSandboxSections() {
  const t = useT();
  const [status, setStatus] = useState<SandboxStatus | null>(null);
  const [job, setJob] = useState<HostPrepJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [restored, setRestored] = useState<number | null>(null);
  const [absent, setAbsent] = useState(false);
  // The Windows prepare step keeps `job`.
  const [setupJob, setSetupJob] = useState<SandboxSetupJob | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);
  const mounted = useRef(true);
  // Bumped by every read and save so an older read cannot overwrite a newer answer.
  const statusGeneration = useRef(0);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Callers set `loading`: a synchronous setState in the mount effect would cascade a render.
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
    // A job started elsewhere keeps reporting here.
    void loadHostPreparation(t("settings.sandbox.prepareError"))
      .then((current) => {
        if (!mounted.current || current.state === "idle") return;
        setJob((shown) => (isOlderJob(current, shown) ? shown : current));
      })
      .catch(() => undefined);
    void loadSandboxSetup(t("sandboxSetup.startError"))
      .then((current) => {
        if (!mounted.current || current.state === "idle") return;
        setSetupJob((shown) => (isOlderJob(current, shown) ? shown : current));
      })
      .catch(() => undefined);
  }, [refresh, t]);

  useEffect(() => {
    if (!shouldPollJob(setupJob)) return;
    const timer = window.setTimeout(() => {
      void loadSandboxSetup(t("sandboxSetup.startError"))
        .then((next) => {
          if (!mounted.current) return;
          setSetupJob(next);
          if (!shouldPollJob(next)) {
            forgetSandboxCapability();
            setLoading(true);
            void refresh(true);
          }
        })
        .catch((pollError) => {
          if (!mounted.current) return;
          setSetupError(
            pollError instanceof Error
              ? pollError.message
              : t("sandboxSetup.startError"),
          );
          setSetupJob(null);
        });
    }, HOST_PREP_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [setupJob, refresh, t]);

  useEffect(() => {
    if (!shouldPollJob(job)) return;
    const timer = window.setTimeout(() => {
      void loadHostPreparation(t("settings.sandbox.prepareError"))
        .then((next) => {
          if (!shouldPollJob(next)) forgetSandboxCapability();
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
      // The chat picker's cached answer predates this change (the opt-in decides MXC).
      forgetSandboxCapability();
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

  const runSetup = async (
    operation: SandboxSetupOperation,
    consentDaclFallback: boolean,
  ) => {
    setSetupError(null);
    try {
      const started = await startSandboxSetup(
        operation,
        { consentDaclFallback },
        t("sandboxSetup.startError"),
      );
      if (!mounted.current) return;
      setSetupJob(started);
      if (!shouldPollJob(started)) {
        forgetSandboxCapability();
        setLoading(true);
        void refresh(true);
      }
    } catch (startError) {
      if (!mounted.current) return;
      setSetupError(
        startError instanceof Error
          ? startError.message
          : t("sandboxSetup.startError"),
      );
    }
  };

  const copyCommand = async (command: string) => {
    if (await copyToClipboard(command)) {
      toast.success(t("sandboxSetup.copied"));
    } else {
      toast.error(t("sandboxSetup.copyFailed"));
    }
  };

  const windows = status?.windows ?? null;
  const view = windows ? windowsView(windows, job, saving) : null;
  const runtimeJob = setupJob?.operation === "windows-runtime" ? setupJob : null;
  const runtimeRunning = runtimeJob?.state === "running";
  const runtimeFailed = jobResult(runtimeJob) === "failed";
  const runtimeOutput = jobOutputLines(runtimeJob);
  const result = jobResult(job);
  const outputLines = jobOutputLines(job);
  const prepKey = view ? PREP_STATUS_KEYS[view.prep] : null;
  const setupRow = status
    ? setupRowView(status, setupJob, setupJob?.manualCommand ?? "")
    : null;
  const quietTools = toolRowsQuiet(setupRow?.show ?? false, view);
  const setupResult = jobResult(setupJob);
  const setupOutput = jobOutputLines(setupJob);
  const setupNote =
    setupResult === "declined" || setupResult === "failed"
      ? (setupJob?.note ?? "")
      : "";
  const setupRunning = setupJob?.state === "running";

  return (
    <>
      {absent ? (
        <p className="text-sm text-muted-foreground">
          {t("settings.sandbox.unsupported")}
        </p>
      ) : (
        <>
          <SettingsSection
            title={t("settings.sandbox.toolsSection")}
            action={
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground"
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
                  <HugeiconsIcon strokeWidth={1.75} icon={Refresh01Icon} />
                )}
                {t("settings.sandbox.refresh")}
              </Button>
            }
          >
            {status ? (
              <>
                <ToolRow
                  label={t("settings.sandbox.python")}
                  tool={status.python}
                  quiet={quietTools}
                  withRemediation={!windows}
                />
                <ToolRow
                  label={t("settings.sandbox.terminal")}
                  tool={status.terminal}
                  shell={status.terminalShell}
                  quiet={quietTools}
                  withRemediation={!windows}
                />
                {setupRow?.show ? (
                  <SettingsRow
                    label={t("settings.sandbox.setupLabel")}
                    description={
                      setupRow.builtIn ? (
                        <>
                          <span className="block">
                            {t("settings.sandbox.macosBuiltIn")}
                          </span>
                          {setupRow.reason ? (
                            <span className="mt-1 block">
                              {setupRow.reason}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        setupRow.reason || status.python.reason || undefined
                      )
                    }
                  >
                    <div className="flex flex-col items-end gap-1">
                      {setupRow.showInstall ? (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={setupRow.installDisabled}
                          onClick={() => void runSetup("linux-install", false)}
                        >
                          {setupRunning ? <Spinner /> : null}
                          {t("sandboxSetup.install")}
                        </Button>
                      ) : null}
                      {setupRunning ? (
                        <span className={`${NOTE_CLASS} text-muted-foreground`}>
                          {t("sandboxSetup.running")}
                        </span>
                      ) : null}
                      {setupResult === "succeeded" ? (
                        <span className={`${NOTE_CLASS} text-muted-foreground`}>
                          {t("settings.sandbox.setupSucceeded")}
                        </span>
                      ) : null}
                      {setupResult === "declined" ? (
                        <span className={`${NOTE_CLASS} text-destructive`}>
                          {t("sandboxSetup.declined")}
                        </span>
                      ) : null}
                      {setupResult === "failed" ? (
                        <span className={`${NOTE_CLASS} text-destructive`}>
                          {t("sandboxSetup.failed")}
                        </span>
                      ) : null}
                      {setupNote ? (
                        <span className={`${NOTE_CLASS} text-destructive`}>
                          {setupNote}
                        </span>
                      ) : null}
                      {setupOutput.length > 0 ? (
                        <pre className="max-w-[calc(360px*var(--ui-space-scale,1))] whitespace-pre-wrap break-words text-right font-mono text-ui-11 text-muted-foreground">
                          {setupOutput.join("\n")}
                        </pre>
                      ) : null}
                      {setupRow.command ? (
                        <>
                          <span
                            className={`${NOTE_CLASS} text-muted-foreground`}
                          >
                            {t("settings.sandbox.setupCommandHint")}
                          </span>
                          <pre className="max-w-[calc(360px*var(--ui-space-scale,1))] whitespace-pre-wrap break-all rounded-md bg-muted px-2 py-1.5 text-left font-mono text-ui-11">
                            {setupRow.command}
                          </pre>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => void copyCommand(setupRow.command)}
                          >
                            {t("sandboxSetup.copyCommand")}
                          </Button>
                        </>
                      ) : null}
                      {setupError ? (
                        <span className={`${NOTE_CLASS} text-destructive`}>
                          {setupError}
                        </span>
                      ) : null}
                    </div>
                  </SettingsRow>
                ) : null}
              </>
            ) : null}
            {error ? (
              <p className="pb-2 text-xs text-destructive">{error}</p>
            ) : null}
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
                    {view.showInstallRuntime || runtimeRunning ? (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={setupRunning}
                        onClick={() => void runSetup("windows-runtime", false)}
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
                  {setupError ? (
                    <p className="text-xs text-destructive">{setupError}</p>
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
    </>
  );
}
