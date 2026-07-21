// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { isTauri } from "@/lib/api-base";
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
import { ApiProviderLogo } from "../../chat/api-provider-logo";
import { type CodingAgentsInfo, loadCodingAgents } from "../api/coding-agents";
import { isLoopbackHost, normalizeHost } from "../components/agent-command";
import { SettingsSection } from "../components/settings-section";

const DOCS_URL = "https://unsloth.ai/docs/integrations/unsloth-start";

// Backend PATH detection is only meaningful in the desktop app on a loopback
// backend; a browser loopback URL may be an SSH/port forward to another host.
function canUseLocalAgentDetection(base: string): boolean {
  if (!isTauri) return false;
  try {
    return isLoopbackHost(normalizeHost(new URL(base).hostname));
  } catch {
    return false;
  }
}

// Copy-to-clipboard state with a single timeout that a rapid re-click resets
// and unmount clears, so the "copied" tick never resets early or leaks.
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

// Each agent's `unsloth start <id>` token and display name. `logo` reuses an
// official provider asset; agents without one fall back to a monogram tile.
// Ids match the backend detection list.
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
      className="flex size-7 shrink-0 items-center justify-center rounded-md font-heading text-[11px] font-semibold text-white"
    >
      {mark}
    </span>
  );
}

/** Compact, click-to-copy command chip for an agent row. */
function InlineCommand({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);

  return (
    <button
      type="button"
      onClick={copy}
      title={copied ? t("settings.agents.copied") : t("settings.agents.copy")}
      aria-label={`${
        copied ? t("settings.agents.copied") : t("settings.agents.copy")
      }: ${command}`}
      className="inline-flex shrink-0 items-center gap-2 rounded-md border border-border bg-muted/40 py-1.5 pl-2.5 pr-2 font-mono text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-white/[0.04]"
    >
      <span className="whitespace-nowrap">{command}</span>
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        strokeWidth={2}
        className={cn(
          "size-3.5 shrink-0",
          copied ? "text-control-accent" : "text-muted-foreground",
        )}
      />
    </button>
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

const QUICKSTART_CMD = "unsloth start claude";

const MODEL_SUFFIX_CMD = `unsloth start codex \\
  --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL \\
  --context-length 32768`;

const MODEL_VARIANT_CMD = `unsloth start codex \\
  --model unsloth/gemma-4-E2B-it-GGUF \\
  --gguf-variant UD-Q4_K_XL \\
  --context-length 32768`;

const REMOTE_CMD_UNIX = `export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude`;

// PowerShell uses $env: assignments; export is POSIX-only.
const REMOTE_CMD_WINDOWS = `$env:UNSLOTH_STUDIO_URL = "https://studio.example.com"
$env:UNSLOTH_API_KEY = "sk-unsloth-..."
unsloth start claude`;

const PASSTHROUGH_CMD = `unsloth start claude --continue
unsloth start codex --persist resume --last`;

const DRY_RUN_CMD = "unsloth start claude --no-launch";

/** Monospace command block with a copy-to-clipboard button in the corner. */
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
    </div>
  );
}

export function AgentsTab() {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const [info, setInfo] = useState<CodingAgentsInfo | null>(null);

  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const localDetection = canUseLocalAgentDetection(serverUrl ?? origin);

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // Only probe PATH when detection is meaningful; a remote backend's PATH says
  // nothing about the machine the copied command will run on.
  useEffect(() => {
    if (!localDetection) {
      setInfo(null);
      return;
    }
    let cancelled = false;
    loadCodingAgents()
      .then((next) => {
        if (!cancelled) setInfo(next);
      })
      .catch(() => {
        // PATH probing is best-effort; the tab is still useful without it.
      });
    return () => {
      cancelled = true;
    };
  }, [localDetection]);

  const detected = new Set(info?.detected ?? []);
  const remoteCommand =
    deviceType === "windows" ? REMOTE_CMD_WINDOWS : REMOTE_CMD_UNIX;

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <header className="flex min-w-0 flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.agents.title")}
        </h1>
        <p className="text-xs text-muted-foreground leading-relaxed">
          {t("settings.agents.description")}
        </p>
      </header>

      <p className="text-sm text-muted-foreground leading-relaxed">
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
          <CommandBlock command={QUICKSTART_CMD} />
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
              className="flex items-center justify-between gap-4 py-2.5"
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
                  <span className="shrink-0 rounded-full bg-control-accent/10 px-2 py-1 text-[10px] leading-none font-semibold text-control-accent">
                    {t("settings.agents.quickstart.installed")}
                  </span>
                ) : null}
              </div>
              <InlineCommand command={`unsloth start ${agent.id}`} />
            </div>
          ))}
        </div>
        {localDetection && info !== null && detected.size === 0 ? (
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
            <CommandBlock command={MODEL_SUFFIX_CMD} />
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.models.variantLabel")}
            </span>
            <CommandBlock command={MODEL_VARIANT_CMD} />
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
        <div className="pt-2">
          <CommandBlock command={PASSTHROUGH_CMD} />
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.dryRun.title")}
        description={t("settings.agents.dryRun.description")}
      >
        <div className="pt-2">
          <CommandBlock command={DRY_RUN_CMD} />
        </div>
      </SettingsSection>
    </div>
  );
}
