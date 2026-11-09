// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  type DeveloperModeSettings,
  loadDeveloperModeSettings,
  updateDeveloperModeSettings,
  useCopyButton,
} from "@/features/settings";
import { useT } from "@/i18n";
import { Copy01Icon } from "@/lib/copy-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowUpRight01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  type ClaudeAgentStatus,
  getClaudeAgentStatus,
  installClaudeAgentSdk,
} from "./api/providers-api";
import {
  CLAUDE_CODE_SETUP_URL,
  CLAUDE_CODE_SIGN_IN,
  type ClaudeAgentChecklistItem,
  type ClaudeAgentModelOption,
  claudeAgentChecklist,
  claudeAgentInitialSelection,
  claudeAgentInstallCommand,
  claudeAgentNpmInstall,
  claudeAgentModelGroups,
  claudeAgentModelOptions,
  claudeAgentModelsToSave,
  claudeAgentRechecksOnReturn,
  claudeAgentSdkCommand,
  claudeAgentSetupStep,
  claudeAgentStatusLine,
  rememberClaudeAgentModels,
} from "./claude-agent";

const RECHECK_DELAY_MS = 400;

export function ClaudeExperimentalBadge() {
  const t = useT();
  return (
    <span className="shrink-0 rounded-full border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-ui-10 leading-none text-amber-700 dark:text-amber-400">
      {t("claudeAgent.experimental")}
    </span>
  );
}

function CommandLine({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);
  const label = copied ? t("claudeAgent.copied") : t("claudeAgent.copy");
  return (
    <div className="flex min-w-0 items-center gap-1 rounded-md border border-border/70 bg-background/70 py-1 pr-1 pl-2.5">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap font-mono text-xs">
        {command}
      </code>
      <Button
        type="button"
        size="icon-sm"
        variant="ghost"
        className="size-7 shrink-0"
        onClick={() => void copy()}
        aria-label={label}
        title={label}
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          className="size-3.5"
        />
      </Button>
    </div>
  );
}

function ChecklistStep({
  index,
  item,
  title,
  help,
  children,
}: {
  index: number;
  item: ClaudeAgentChecklistItem;
  title: string;
  help: string;
  children?: ReactNode;
}) {
  const done = item.state === "done";
  return (
    <li
      data-step={item.id}
      data-state={item.state}
      className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2.5"
    >
      <span
        className={cn(
          "flex size-6 items-center justify-center rounded-full border text-xs font-medium tabular-nums",
          done
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
            : item.state === "current"
              ? "border-ring-strong text-foreground"
              : "border-border text-muted-foreground",
        )}
      >
        {done ? (
          <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
        ) : (
          index
        )}
      </span>
      <div className="flex min-w-0 flex-col gap-1.5 pt-0.5">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span
            className={cn(
              "text-sm font-medium",
              done && "text-muted-foreground line-through",
            )}
          >
            {title}
          </span>
          {done ? null : (
            <span className="text-xs text-muted-foreground">{help}</span>
          )}
        </div>
        {done ? null : children}
      </div>
    </li>
  );
}

function Disclosure({
  open,
  onToggle,
  label,
  hint,
}: {
  open: boolean;
  onToggle: () => void;
  label: string;
  hint?: string;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      className="flex min-h-9 w-full items-center gap-2 px-4 py-1.5 text-left text-xs font-medium text-muted-foreground hover:bg-muted/35 hover:text-foreground"
    >
      <HugeiconsIcon
        icon={ArrowDown01Icon}
        className={cn("size-3.5 transition-transform", !open && "-rotate-90")}
      />
      {label}
      {hint ? <span className="font-normal tabular-nums">{hint}</span> : null}
    </button>
  );
}

export function ClaudeAgentSetup({
  savedModels,
  savedMaxOutputTokens,
  off,
  saving,
  onTurnedOn,
  onSave,
  onCancel,
}: {
  savedModels: readonly string[] | null;
  savedMaxOutputTokens: string;
  off: boolean;
  saving: boolean;
  onTurnedOn: () => Promise<void>;
  onSave: (models: string[], maxOutputTokens: string) => void;
  onCancel: () => void;
}) {
  const t = useT();
  const [status, setStatus] = useState<ClaudeAgentStatus | null>(null);
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [turningOn, setTurningOn] = useState(false);
  const [developerMode, setDeveloperMode] =
    useState<DeveloperModeSettings | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [extraModels, setExtraModels] = useState("");
  const [maxTokens, setMaxTokens] = useState(savedMaxOutputTokens);
  const [moreOpen, setMoreOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const requestRef = useRef(0);
  const busyRef = useRef(false);
  const selectionReadyRef = useRef(false);

  const forcedOff = developerMode?.claudeAgentEnvOverride === false;
  const step = claudeAgentSetupStep({
    off,
    forcedOff,
    checking,
    status,
    error,
  });
  const options = useMemo(
    () => (status?.ready ? claudeAgentModelOptions(status.models) : []),
    [status],
  );
  const groups = claudeAgentModelGroups(options);
  const checklist = claudeAgentChecklist(step);
  const npmInstall = claudeAgentNpmInstall(status?.host_os);
  const modelsToSave = claudeAgentModelsToSave(options, selected, extraModels);

  function applyStatus(next: ClaudeAgentStatus) {
    setStatus(next);
    rememberClaudeAgentModels(next.models);
    if (!next.ready || selectionReadyRef.current) return;
    selectionReadyRef.current = true;
    const nextOptions = claudeAgentModelOptions(next.models);
    const initial = claudeAgentInitialSelection(nextOptions, savedModels);
    setSelected(initial.selected);
    setExtraModels(initial.extra.join("\n"));
    if (initial.extra.length > 0 || savedMaxOutputTokens) setAdvancedOpen(true);
    const more = new Set(
      claudeAgentModelGroups(nextOptions).more.map((option) => option.id),
    );
    if (initial.selected.some((id) => more.has(id))) setMoreOpen(true);
  }

  async function run(request: () => Promise<ClaudeAgentStatus>) {
    const ticket = ++requestRef.current;
    busyRef.current = true;
    setError("");
    try {
      const next = await request();
      if (ticket === requestRef.current) applyStatus(next);
    } catch (cause) {
      if (ticket === requestRef.current) {
        setError(
          cause instanceof Error ? cause.message : t("claudeAgent.checkFailed"),
        );
      }
    } finally {
      if (ticket === requestRef.current) busyRef.current = false;
    }
  }

  async function check() {
    setChecking(true);
    await run(getClaudeAgentStatus);
    setChecking(false);
  }

  async function install() {
    setInstalling(true);
    await run(installClaudeAgentSdk);
    setInstalling(false);
  }

  async function turnOn() {
    setTurningOn(true);
    setError("");
    try {
      setDeveloperMode(
        await updateDeveloperModeSettings({ claudeAgent: true }),
      );
      await onTurnedOn();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setTurningOn(false);
    }
  }

  const checkRef = useRef(check);
  useEffect(() => {
    checkRef.current = check;
  });

  useEffect(() => {
    if (!off) {
      void checkRef.current();
      return;
    }
    let live = true;
    void loadDeveloperModeSettings()
      .then((settings) => {
        if (live) setDeveloperMode(settings);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [off]);

  // Back from the terminal: check again so the steps turn green by themselves.
  const rechecks = claudeAgentRechecksOnReturn(step);
  useEffect(() => {
    if (!rechecks || typeof window === "undefined") return;
    let timer: number | undefined;
    const onReturn = () => {
      if (document.hidden) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        if (!busyRef.current) void checkRef.current();
      }, RECHECK_DELAY_MS);
    };
    window.addEventListener("focus", onReturn);
    document.addEventListener("visibilitychange", onReturn);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("focus", onReturn);
      document.removeEventListener("visibilitychange", onReturn);
    };
  }, [rechecks]);

  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id)
        ? current.filter((value) => value !== id)
        : [...current, id],
    );

  const checkAgainButton = (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="w-fit"
      disabled={checking || installing}
      onClick={() => void check()}
    >
      {checking ? <Spinner className="mr-2 size-3.5" /> : null}
      {t("claudeAgent.checkAgain")}
    </Button>
  );

  const modelRow = (option: ClaudeAgentModelOption) => (
    <li key={option.id}>
      <label
        htmlFor={`claude-agent-model-${option.id}`}
        className="flex min-h-9 cursor-pointer items-center gap-2.5 px-4 py-1.5 hover:bg-muted/35"
      >
        <Checkbox
          id={`claude-agent-model-${option.id}`}
          checked={selected.includes(option.id)}
          onCheckedChange={() => toggle(option.id)}
        />
        <span className="min-w-0 truncate text-sm">{option.name}</span>
        <span className="ml-auto shrink-0 font-mono text-ui-11 text-muted-foreground/70">
          {option.id}
        </span>
      </label>
    </li>
  );

  const isOff = step === "off" || step === "forced_off";

  return (
    <div className="flex flex-col gap-3" data-testid="claude-agent-setup">
      <section className="flex flex-col gap-3 rounded-xl border border-border/70 bg-muted/[0.12] px-4 py-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">
              {t("claudeAgent.title")}
            </span>
            <ClaudeExperimentalBadge />
          </div>
          <p className="text-xs text-muted-foreground">
            {isOff ? t("claudeAgent.offBody") : t("claudeAgent.intro")}
          </p>
        </div>

        {isOff ? (
          <div className="flex flex-col items-start gap-2">
            {step === "forced_off" ? (
              <p className="text-xs text-amber-700 dark:text-amber-400">
                {t("claudeAgent.forcedOff")}
              </p>
            ) : null}
            {error ? (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            ) : null}
            <Button
              type="button"
              size="sm"
              variant="dark"
              data-testid="claude-agent-turn-on"
              disabled={step === "forced_off" || turningOn}
              onClick={() => void turnOn()}
            >
              {turningOn ? <Spinner className="mr-2 size-3.5" /> : null}
              {turningOn ? t("claudeAgent.turningOn") : t("claudeAgent.turnOn")}
            </Button>
          </div>
        ) : step === "checking" ? (
          <p
            className="flex items-center gap-2 text-xs text-muted-foreground"
            aria-live="polite"
          >
            <Spinner className="size-3.5" />
            {t("claudeAgent.checking")}
          </p>
        ) : step === "ready" && status ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p
              className="flex min-w-0 items-center gap-2 text-xs"
              aria-live="polite"
            >
              <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
              <span className="font-medium">{t("claudeAgent.ready")}</span>
              <span className="truncate text-muted-foreground">
                {claudeAgentStatusLine(status)}
              </span>
            </p>
            {checkAgainButton}
          </div>
        ) : checklist ? (
          <div className="flex flex-col gap-3">
            <p className="flex items-center gap-2 text-xs" aria-live="polite">
              <span className="size-2 shrink-0 rounded-full bg-amber-500" />
              {t("claudeAgent.notReady")}
            </p>
            <ol
              className="flex flex-col gap-3"
              data-testid="claude-agent-checklist"
            >
              <ChecklistStep
                index={1}
                item={checklist[0]}
                title={t("claudeAgent.stepInstall")}
                help={t("claudeAgent.stepInstallHelp")}
              >
                <CommandLine
                  command={claudeAgentInstallCommand(status?.host_os)}
                />
                {npmInstall ? (
                  <>
                    <span className="text-xs text-muted-foreground">
                      {t("claudeAgent.stepInstallNpm")}
                    </span>
                    <CommandLine command={npmInstall} />
                  </>
                ) : null}
                <a
                  href={CLAUDE_CODE_SETUP_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex w-fit items-center gap-1 text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                >
                  {t("claudeAgent.stepInstallOther")}
                  <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
                </a>
              </ChecklistStep>
              <ChecklistStep
                index={2}
                item={checklist[1]}
                title={t("claudeAgent.stepSignIn")}
                help={t("claudeAgent.stepSignInHelp")}
              >
                <CommandLine command={CLAUDE_CODE_SIGN_IN} />
              </ChecklistStep>
              <ChecklistStep
                index={3}
                item={checklist[2]}
                title={t("claudeAgent.stepCheck")}
                help={t("claudeAgent.stepCheckHelp")}
              >
                {checkAgainButton}
              </ChecklistStep>
            </ol>
          </div>
        ) : step === "sdk_missing" ? (
          <div className="flex flex-col gap-2">
            <p className="text-xs">
              {installing
                ? t("claudeAgent.installing")
                : t("claudeAgent.sdkMissing")}
            </p>
            {error ? (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            ) : null}
            {status?.message && !installing ? (
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">
                  {t("claudeAgent.sdkManual")}
                </span>
                <CommandLine command={claudeAgentSdkCommand(status.message)} />
              </div>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                size="sm"
                disabled={checking || installing}
                onClick={() => void install()}
              >
                {installing ? <Spinner className="mr-2 size-3.5" /> : null}
                {t("claudeAgent.install")}
              </Button>
              {checkAgainButton}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-2">
            <p role="alert" className="text-xs text-destructive">
              {status?.message || error || t("claudeAgent.checkFailed")}
            </p>
            {checkAgainButton}
          </div>
        )}
      </section>

      {step === "ready" ? (
        <section className="overflow-hidden rounded-xl border border-border/70 bg-muted/[0.12]">
          <div className="flex items-center justify-between gap-3 border-border/60 border-b px-4 py-3">
            <Label className="text-sm font-medium">
              {t("claudeAgent.models")}
            </Label>
            <span className="text-xs tabular-nums text-muted-foreground">
              {t("claudeAgent.modelsSelected", { count: modelsToSave.length })}
            </span>
          </div>
          <ul
            className="divide-y divide-border/60"
            data-testid="claude-agent-models"
          >
            {groups.main.map(modelRow)}
          </ul>
          {groups.more.length ? (
            <div className="border-border/60 border-t">
              <Disclosure
                open={moreOpen}
                onToggle={() => setMoreOpen((open) => !open)}
                label={t("claudeAgent.moreModels")}
                hint={String(groups.more.length)}
              />
              {moreOpen ? (
                <ul className="divide-y divide-border/60 border-border/60 border-t">
                  {groups.more.map(modelRow)}
                </ul>
              ) : null}
            </div>
          ) : null}
          <div className="border-border/60 border-t">
            <Disclosure
              open={advancedOpen}
              onToggle={() => setAdvancedOpen((open) => !open)}
              label={t("claudeAgent.advanced")}
            />
            {advancedOpen ? (
              <div className="flex flex-col gap-3 border-border/60 border-t px-4 py-3">
                <div className="flex flex-col gap-1.5">
                  <Label
                    htmlFor="claude-agent-extra-models"
                    className="text-sm font-medium"
                  >
                    {t("claudeAgent.extraModels")}
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {t("claudeAgent.extraModelsHelp")}
                  </p>
                  <Textarea
                    id="claude-agent-extra-models"
                    value={extraModels}
                    onChange={(event) => setExtraModels(event.target.value)}
                    placeholder="claude-opus-4-8"
                    rows={2}
                    className="min-h-14 font-mono text-sm"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label
                    htmlFor="claude-agent-max-tokens"
                    className="text-sm font-medium"
                  >
                    {t("claudeAgent.maxTokens")}
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {t("claudeAgent.maxTokensHelp")}
                  </p>
                  <Input
                    id="claude-agent-max-tokens"
                    type="text"
                    inputMode="numeric"
                    autoComplete="off"
                    spellCheck={false}
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(event.target.value)}
                    placeholder="32768"
                    className="h-9 max-w-48 text-sm"
                  />
                </div>
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      <div className="-mb-3 flex flex-wrap items-center justify-end gap-2">
        {step === "ready" && modelsToSave.length === 0 ? (
          <span className="text-xs text-muted-foreground">
            {t("claudeAgent.pickModel")}
          </span>
        ) : null}
        {step === "ready" ? (
          <Button
            type="button"
            size="sm"
            variant="dark"
            data-testid="claude-agent-connect"
            disabled={saving || checking || modelsToSave.length === 0}
            onClick={() => onSave(modelsToSave, maxTokens)}
          >
            {saving ? <Spinner className="mr-2 size-3.5" /> : null}
            {savedModels ? t("claudeAgent.save") : t("claudeAgent.connect")}
          </Button>
        ) : null}
        <Button type="button" size="sm" variant="outline" onClick={onCancel}>
          {t("claudeAgent.cancel")}
        </Button>
      </div>
    </div>
  );
}
