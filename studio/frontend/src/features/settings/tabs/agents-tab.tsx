// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getClientPlatform } from "@/components/tauri/window-titlebar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { getApiBase, isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  Book03Icon,
  Copy01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import { ApiProviderLogo } from "../../chat/api-provider-logo";
import { type CodingAgentsInfo, loadCodingAgents } from "../api/coding-agents";
import {
  buildAgentCommand,
  isLoopbackHost,
  normalizeHost,
} from "../components/agent-command";
import { SettingsSection } from "../components/settings-section";

const DOCS_URL = "https://unsloth.ai/docs/integrations/unsloth-start";

function isLoopbackBase(base: string): boolean {
  try {
    return isLoopbackHost(normalizeHost(new URL(base).hostname));
  } catch {
    return false;
  }
}

// Desktop-only: a browser loopback URL may be an SSH/port forward to another host.
function canUseLocalAgentDetection(base: string): boolean {
  return isTauri && isLoopbackBase(base);
}

// One timeout, reset on re-click and cleared on unmount, so the tick never leaks.
function useCopyButton(text: string) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    },
    [],
  );

  const copy = async () => {
    if (!(await copyToClipboard(text))) return;
    setCopied(true);
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = window.setTimeout(() => {
      setCopied(false);
      timeoutRef.current = null;
    }, 1600);
  };

  return { copied, copy };
}

// Ids match the backend detection list; agents without an official `logo` asset get a monogram.
// Names are untranslated, so `settings.agents.intro` lists them all to keep them searchable.
const SUPPORTED_AGENTS: {
  id: string;
  name: string;
  logo?: string;
  color?: string;
  mark?: string;
}[] = [
  { id: "claude", name: "Claude Code", logo: "anthropic" },
  { id: "codex", name: "OpenAI Codex", logo: "openai" },
  { id: "hermes", name: "Hermes", color: "#8B5CF6", mark: "He" },
  { id: "openclaw", name: "OpenClaw", color: "#F59E0B", mark: "Ol" },
  { id: "opencode", name: "OpenCode", color: "#3B82F6", mark: "Oc" },
  { id: "pi", name: "Pi", color: "#EC4899", mark: "Pi" },
];

/** Official brand logo when available, else a brand-colored monogram tile. */
function AgentIcon({
  logo,
  color,
  mark,
}: {
  logo?: string;
  color?: string;
  mark?: string;
}) {
  if (logo) {
    return (
      <span className="flex size-7 shrink-0 items-center justify-center overflow-hidden rounded-md">
        <ApiProviderLogo providerType={logo} className="size-7 rounded-md" />
      </span>
    );
  }
  return (
    <span
      aria-hidden={true}
      style={{ backgroundColor: color }}
      className="flex size-7 shrink-0 items-center justify-center rounded-md font-heading text-ui-11 font-semibold text-white"
    >
      {mark}
    </span>
  );
}

function InlineCommand({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);

  return (
    <>
      <button
        type="button"
        onClick={copy}
        title={copied ? t("settings.agents.copied") : t("settings.agents.copy")}
        aria-label={`${
          copied ? t("settings.agents.copied") : t("settings.agents.copy")
        }: ${command}`}
        className="inline-flex min-w-0 max-w-full items-center gap-2 rounded-md border border-border bg-muted/40 py-1.5 pl-2.5 pr-2 font-mono text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-white/[0.04]"
      >
        {/* Truncate: a remote base makes the command long enough to push the icon out. */}
        <span className="truncate whitespace-nowrap">{command}</span>
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          strokeWidth={2}
          className={cn(
            "size-3.5 shrink-0",
            copied ? "text-control-accent" : "text-muted-foreground",
          )}
        />
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? t("settings.agents.copied") : ""}
      </span>
    </>
  );
}

// Flag tokens are literal; only the descriptions are localized.
const OPTION_ROWS: { flag: string; descKey: TranslationKey }[] = [
  { flag: "--model, -m", descKey: "settings.agents.options.model" },
  {
    flag: "--context-length",
    descKey: "settings.agents.options.contextLength",
  },
  { flag: "--gguf-variant", descKey: "settings.agents.options.ggufVariant" },
  {
    flag: "--load-in-4bit / --no-load-in-4bit",
    descKey: "settings.agents.options.loadIn4bit",
  },
  {
    flag: "--tensor-parallel / --no-tensor-parallel",
    descKey: "settings.agents.options.tensorParallel",
  },
  { flag: "--serve / --no-serve", descKey: "settings.agents.options.serve" },
  {
    flag: "--launch / --no-launch",
    descKey: "settings.agents.options.launch",
  },
  {
    flag: "--persist / --no-persist",
    descKey: "settings.agents.options.persist",
  },
  { flag: "--api-key", descKey: "settings.agents.options.apiKey" },
  { flag: "--yolo", descKey: "settings.agents.options.yolo" },
];

const QUICKSTART_AGENT = "claude";

// Flags only: agentCommand supplies the prefix so every example targets the Studio
// this tab shows. Kept single line so the copy pastes as-is.
const MODEL_SUFFIX_FLAGS =
  "--model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL --context-length 32768";

const MODEL_VARIANT_FLAGS =
  "--model unsloth/gemma-4-E2B-it-GGUF --gguf-variant UD-Q4_K_XL --context-length 32768";

const REMOTE_CMD_UNIX = `export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude`;

// PowerShell uses $env: assignments; export is POSIX-only.
const REMOTE_CMD_WINDOWS = `$env:UNSLOTH_STUDIO_URL = "https://studio.example.com"
$env:UNSLOTH_API_KEY = "sk-unsloth-..."
unsloth start claude`;

// Independent alternatives, each with its own copy button (not one script).
const PASSTHROUGH_EXAMPLES = [
  { agent: "claude", flags: "--continue" },
  { agent: "codex", flags: "--persist resume --last" },
];

const DRY_RUN_FLAGS = "--no-launch";

function CommandBlock({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);

  return (
    <div className="group relative">
      <pre className="hover-scrollbar overflow-x-auto rounded-lg border border-border bg-muted/40 py-3 pl-3.5 pr-11 text-xs leading-relaxed text-foreground dark:bg-white/[0.04]">
        <code className="font-mono whitespace-pre">{command}</code>
      </pre>
      <button
        type="button"
        onClick={copy}
        aria-label={
          copied ? t("settings.agents.copied") : t("settings.agents.copy")
        }
        className="absolute top-2 right-2 flex size-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          className={cn("size-3.5", copied && "text-control-accent")}
          strokeWidth={2}
        />
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? t("settings.agents.copied") : ""}
      </span>
    </div>
  );
}

export function AgentsTab() {
  const t = useT();
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const deviceType = usePlatformStore((s) => s.deviceType);
  const [info, setInfo] = useState<CodingAgentsInfo | null>(null);

  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const localDetection = canUseLocalAgentDetection(serverUrl ?? origin);

  // The remote snippet runs on the client, so use the client platform, not deviceType.
  // Anchor the match: a bare includes("win") would also match "darwin".
  const [isWindowsClient] = useState(() => {
    const p = getClientPlatform();
    return p.startsWith("win") || p.includes("windows");
  });

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // A remote backend's PATH says nothing about the machine running the copied command.
  useEffect(() => {
    if (!localDetection) return;
    let cancelled = false;
    loadCodingAgents()
      .then((next) => {
        if (!cancelled) setInfo(next);
      })
      .catch(() => {
        // Best-effort; the tab still works without PATH detection.
      });
    return () => {
      cancelled = true;
    };
  }, [localDetection]);

  // Derive visibility from localDetection instead of clearing info in the effect.
  const visibleInfo = localDetection ? info : null;
  const detected = new Set(visibleInfo?.detected ?? []);
  const remoteCommand = isWindowsClient ? REMOTE_CMD_WINDOWS : REMOTE_CMD_UNIX;

  // `codex` needs a GGUF model (unsloth_cli's _require_gguf_for_codex exits otherwise), so flag
  // its row instead of offering a failing command. Same three signals the API usage panel uses.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const isGguf =
    activeGgufVariant != null ||
    activeNativePathToken != null ||
    ggufContextLength != null;

  // Build from the reachable base: a bare `unsloth start` only probes 127.0.0.1:8888, but the
  // desktop falls back across 8888-8908 and Studio may be remote. The browser must use its own
  // origin, since /api/health reports the backend's localhost (the user's, behind a tunnel);
  // the desktop has no window origin and falls back to getApiBase() while serverUrl loads.
  // No --api-key: the CLI caches an explicit key per base, so a placeholder would overwrite a
  // working saved one. Omitting it replays the saved key; the remote section covers first setup.
  const commandBase = isTauri ? (serverUrl ?? getApiBase()) : origin;
  // The command runs wherever the CLI is. For a loopback base that is this Studio's
  // own host, so use deviceType, which reports wsl where the browser would claim
  // Windows and emit $env: syntax bash rejects. A remote base is reached from the
  // viewer's machine instead, so only the client platform describes that shell.
  const commandOs =
    (isLoopbackBase(commandBase) ? deviceType === "windows" : isWindowsClient)
      ? "windows"
      : "unix";
  const agentCommand = (agentId: string) =>
    buildAgentCommand(commandBase, null, commandOs, agentId);
  const example = (agentId: string, flags: string) =>
    `${agentCommand(agentId)} ${flags}`;

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      {/* data-settings-label lets indexed settings search scroll to these. */}
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.agents.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.agents.title")}
        </h1>
        <p
          data-settings-label={t("settings.agents.description")}
          className="text-xs text-muted-foreground leading-relaxed"
        >
          {t("settings.agents.description")}
        </p>
      </header>

      <p
        data-settings-label={t("settings.agents.intro")}
        className="text-sm text-muted-foreground leading-relaxed"
      >
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground dark:bg-white/[0.08]">
          unsloth start
        </code>{" "}
        {t("settings.agents.intro")}
      </p>

      <a
        href={DOCS_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex w-fit items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <HugeiconsIcon icon={Book03Icon} className="size-3.5" />
        {t("settings.agents.readDocs")}
        <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
      </a>

      <SettingsSection
        title={t("settings.agents.quickstart.title")}
        description={t("settings.agents.quickstart.description")}
      >
        <div className="pt-2">
          <CommandBlock command={agentCommand(QUICKSTART_AGENT)} />
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.supportedAgents.title")}
        description={t("settings.agents.supportedAgents.description")}
      >
        <div className="mt-1 flex flex-col divide-y divide-border/60">
          {SUPPORTED_AGENTS.map((agent) => (
            <div
              key={agent.id}
              className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 py-2.5"
            >
              <div className="flex min-w-0 items-center gap-3">
                <AgentIcon
                  logo={agent.logo}
                  color={agent.color}
                  mark={agent.mark}
                />
                <span className="truncate text-sm font-medium text-foreground">
                  {agent.name}
                </span>
                {detected.has(agent.id) ? (
                  <span className="shrink-0 rounded-full bg-control-accent/10 px-2 py-1 text-ui-10 leading-none font-semibold text-control-accent">
                    {t("settings.agents.quickstart.installed")}
                  </span>
                ) : null}
                {agent.id === "codex" && !isGguf ? (
                  <span className="shrink-0 rounded-full bg-muted px-2 py-1 text-ui-10 leading-none font-semibold text-muted-foreground">
                    {t("settings.agents.supportedAgents.requiresGguf")}
                  </span>
                ) : null}
              </div>
              <InlineCommand command={agentCommand(agent.id)} />
            </div>
          ))}
        </div>
        {visibleInfo !== null && detected.size === 0 ? (
          <p className="pt-3 text-xs text-muted-foreground">
            {t("settings.agents.quickstart.noneDetected")}
          </p>
        ) : null}
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.models.title")}
        description={t("settings.agents.models.description")}
      >
        <div className="flex flex-col gap-3 pt-2">
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.models.suffixLabel")}
            </span>
            <CommandBlock command={example("codex", MODEL_SUFFIX_FLAGS)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.models.variantLabel")}
            </span>
            <CommandBlock command={example("codex", MODEL_VARIANT_FLAGS)} />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.options.title")}
        description={t("settings.agents.options.description")}
      >
        <div className="mt-1 flex flex-col divide-y divide-border/60">
          {OPTION_ROWS.map((row) => (
            <div
              key={row.flag}
              className="grid grid-cols-[minmax(0,11rem)_1fr] items-start gap-x-5 gap-y-1 py-2.5 max-sm:grid-cols-1"
            >
              <code className="min-w-0 break-words font-mono text-xs font-medium text-foreground">
                {row.flag}
              </code>
              <span className="text-xs leading-relaxed text-muted-foreground">
                {t(row.descKey)}
              </span>
            </div>
          ))}
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.remote.title")}
        description={t("settings.agents.remote.description")}
      >
        <div className="pt-2">
          <CommandBlock command={remoteCommand} />
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.passthrough.title")}
        description={t("settings.agents.passthrough.description")}
      >
        <div className="flex flex-col gap-2 pt-2">
          {PASSTHROUGH_EXAMPLES.map(({ agent, flags }) => (
            <CommandBlock key={flags} command={example(agent, flags)} />
          ))}
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.dryRun.title")}
        description={t("settings.agents.dryRun.description")}
      >
        <div className="pt-2">
          <CommandBlock command={example("claude", DRY_RUN_FLAGS)} />
        </div>
      </SettingsSection>
    </div>
  );
}
