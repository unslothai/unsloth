// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getClientPlatform } from "@/components/tauri/window-titlebar";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import {
  type BackendModelDetails,
  type GgufVariantDetail,
  type InferenceStatusResponse,
  getInferenceStatus,
  listCachedGguf,
  listGgufVariants,
  listModels,
} from "@/features/chat";
import { useHfTokenStore } from "@/features/hub";
import type { TranslationKey } from "@/i18n";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  Book03Icon,
  Copy01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiProviderLogo } from "../../chat/api-provider-logo";
import { loadCodingAgents } from "../api/coding-agents";
import {
  buildAgentCommand,
  isLoopbackHost,
  normalizeHost,
} from "../components/agent-command";
import { SettingsSection } from "../components/settings-section";
import { psSingle, shSingle } from "../components/usage-examples";

const DOCS_URL = "https://unsloth.ai/docs/integrations/unsloth-start";
const EXAMPLE_MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF";
const EXAMPLE_MODEL_VARIANT = "UD-Q4_K_XL";
const MODEL_RESULT_LIMIT = 7;
const HUGGING_FACE_REPO_PATTERN = /^[^/\\:\s]+\/[^/\\:\s]+$/;
const SEARCH_TOKEN_PATTERN = /\s+/;
const SAFE_SHELL_ARG_PATTERN = /^[A-Za-z0-9_./:@%+=,-]+$/;
const SUBAGENT_AGENT_IDS = new Set(["claude", "codex", "opencode", "pi"]);
const REMOTE_API_KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";

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

  const reset = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setCopied(false);
  };

  return { copied, copy, reset };
}

type AgentDetails = {
  id: string;
  name: string;
  docsUrl: string;
  logo?: string;
  icon?: string;
  darkIcon?: string;
  invertIconInDark?: boolean;
  color?: string;
  mark?: string;
};

type ParsedModel = {
  repo: string;
  variant: string | null;
};

const SUPPORTED_AGENTS: AgentDetails[] = [
  {
    id: "claude",
    name: "Claude Code",
    docsUrl: "https://unsloth.ai/docs/basics/claude-code",
    logo: "anthropic",
  },
  {
    id: "codex",
    name: "OpenAI Codex",
    docsUrl: "https://unsloth.ai/docs/basics/codex",
    logo: "openai",
  },
  {
    id: "hermes",
    name: "Hermes Agent",
    docsUrl: "https://unsloth.ai/docs/integrations/hermes-agent",
    icon: "hermes.svg",
    invertIconInDark: true,
  },
  {
    id: "openclaw",
    name: "OpenClaw",
    docsUrl: "https://unsloth.ai/docs/integrations/openclaw",
    icon: "openclaw.svg",
  },
  {
    id: "opencode",
    name: "OpenCode",
    docsUrl: "https://unsloth.ai/docs/integrations/opencode",
    icon: "opencode-light.svg",
    darkIcon: "opencode-dark.svg",
  },
  {
    id: "pi",
    name: "Pi Coding Agent",
    docsUrl: DOCS_URL,
    icon: "pi.svg",
  },
];

const FALLBACK_AGENT = SUPPORTED_AGENTS[0];

function detailsFor(agentId: string): AgentDetails {
  return (
    SUPPORTED_AGENTS.find((agent) => agent.id === agentId) ?? {
      id: agentId,
      name: agentId,
      docsUrl: DOCS_URL,
      color: "#64748B",
      mark: agentId.slice(0, 2),
    }
  );
}

function splitModelVariant(model: string): ParsedModel {
  const value = model.trim();
  if (
    !value ||
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.startsWith("~") ||
    (value.length >= 2 && value[1] === ":")
  ) {
    return { repo: value, variant: null };
  }

  const separator = value.lastIndexOf(":");
  if (separator < 0) {
    return { repo: value, variant: null };
  }
  const repo = value.slice(0, separator);
  const variant = value.slice(separator + 1);
  if (!(repo && variant) || variant.includes("/")) {
    return { repo: value, variant: null };
  }
  return { repo, variant };
}

function isHuggingFaceRepo(model: string): boolean {
  return HUGGING_FACE_REPO_PATTERN.test(model);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  const unitIndex = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** unitIndex;
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function discoverGgufModels(
  items: BackendModelDetails[],
  cachedRepos: string[],
): {
  models: string[];
  variants: Record<string, string>;
} {
  const models = [EXAMPLE_MODEL_REPO];
  const variants: Record<string, string> = {};
  for (const model of items) {
    if (!model.is_gguf) {
      continue;
    }
    const parsed = splitModelVariant(model.id);
    if (parsed.repo && !models.includes(parsed.repo)) {
      models.push(parsed.repo);
    }
    if (parsed.variant && !variants[parsed.repo]) {
      variants[parsed.repo] = parsed.variant;
    }
  }
  for (const repo of cachedRepos) {
    if (!models.includes(repo)) {
      models.push(repo);
    }
  }

  return { models, variants };
}

function activeGgufSelection(
  status: InferenceStatusResponse | null,
): { model: string; variant: string | null } | null {
  if (!(status?.is_gguf && status.model_identifier)) {
    return null;
  }
  const active = splitModelVariant(status.model_identifier);
  if (!active.repo) {
    return null;
  }
  return {
    model: active.repo,
    variant: isHuggingFaceRepo(active.repo)
      ? (status.gguf_variant ?? active.variant)
      : active.variant,
  };
}

/** Official provider or agent logo when available, else a monogram tile. */
function AgentIcon({
  logo,
  icon,
  darkIcon,
  invertIconInDark,
  color,
  mark,
}: {
  logo?: string;
  icon?: string;
  darkIcon?: string;
  invertIconInDark?: boolean;
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
  if (icon) {
    const iconSrc = `${import.meta.env.BASE_URL}agent-logos/${icon}`;
    const darkIconSrc = darkIcon
      ? `${import.meta.env.BASE_URL}agent-logos/${darkIcon}`
      : null;
    return (
      <span className="flex size-7 shrink-0 items-center justify-center overflow-hidden rounded-md">
        <img
          src={iconSrc}
          alt=""
          aria-hidden={true}
          className={cn(
            "size-7 object-contain",
            darkIconSrc && "dark:hidden",
            invertIconInDark && "dark:invert",
          )}
        />
        {darkIconSrc ? (
          <img
            src={darkIconSrc}
            alt=""
            aria-hidden={true}
            className="hidden size-7 object-contain dark:block"
          />
        ) : null}
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
  { flag: "--as-subagent", descKey: "settings.agents.options.asSubagent" },
  { flag: "--api-key", descKey: "settings.agents.options.apiKey" },
  { flag: "--yolo", descKey: "settings.agents.options.yolo" },
];

const REMOTE_CMD_UNIX = `export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude`;

// PowerShell uses $env: assignments; export is POSIX-only.
const REMOTE_CMD_WINDOWS = `$env:UNSLOTH_STUDIO_URL = "https://studio.example.com"
$env:UNSLOTH_API_KEY = "sk-unsloth-..."
unsloth start claude`;

// Independent alternatives, each with its own copy button (not one script).
const PASSTHROUGH_COMMANDS = [
  "unsloth start claude --continue",
  "unsloth start codex --persist resume --last",
];

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
      <output className="sr-only" aria-live="polite">
        {copied ? t("settings.agents.copied") : ""}
      </output>
    </div>
  );
}

// Quote a --model value for copy-paste. Safe identifiers are left as-is; only
// values with shell metacharacters (e.g. a local path with spaces) are quoted.
function quoteShellArg(value: string, windows: boolean): string {
  if (SAFE_SHELL_ARG_PATTERN.test(value)) {
    return value;
  }
  return windows ? `'${psSingle(value)}'` : `'${shSingle(value)}'`;
}

function SubagentSection({
  agent,
  baseCommand,
  commandModelArg,
}: {
  agent: AgentDetails;
  baseCommand: string;
  commandModelArg: string;
}) {
  const t = useT();
  const command = `${baseCommand} --as-subagent --model ${commandModelArg}`;
  const prompt =
    agent.id === "opencode"
      ? t("settings.agents.subagent.opencodePrompt")
      : t("settings.agents.subagent.defaultPrompt");
  const commandCopy = useCopyButton(command);
  const promptCopy = useCopyButton(prompt);

  if (!SUBAGENT_AGENT_IDS.has(agent.id)) {
    return null;
  }

  return (
    <div className="flex min-w-0 flex-col gap-3 rounded-lg border border-border bg-muted/10 p-3">
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium text-foreground">
          {t("settings.agents.subagent.title")}
        </span>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {t("settings.agents.subagent.description", { agent: agent.name })}
        </p>
      </div>

      <div className="flex min-w-0 flex-col gap-1.5">
        <div className="flex items-center justify-between gap-3">
          <span className="text-[11px] font-medium text-foreground">
            {t("settings.agents.subagent.setupCommand")}
          </span>
          <button
            type="button"
            onClick={commandCopy.copy}
            aria-label={t("settings.agents.subagent.copySetupCommand")}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background/70 px-2 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <HugeiconsIcon
              icon={commandCopy.copied ? Tick02Icon : Copy01Icon}
              className={cn(
                "size-3.5",
                commandCopy.copied && "text-control-accent",
              )}
            />
            {commandCopy.copied
              ? t("settings.agents.copied")
              : t("settings.agents.copy")}
          </button>
        </div>
        <code className="block min-w-0 whitespace-pre-wrap break-all rounded-md border border-border bg-background/70 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-foreground">
          {command}
        </code>
      </div>

      <div className="flex min-w-0 flex-col gap-1.5">
        <div className="flex items-center justify-between gap-3">
          <span className="text-[11px] font-medium text-foreground">
            {t("settings.agents.subagent.usagePrompt", { agent: agent.name })}
          </span>
          <button
            type="button"
            onClick={promptCopy.copy}
            aria-label={t("settings.agents.subagent.copyUsagePrompt")}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background/70 px-2 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <HugeiconsIcon
              icon={promptCopy.copied ? Tick02Icon : Copy01Icon}
              className={cn(
                "size-3.5",
                promptCopy.copied && "text-control-accent",
              )}
            />
            {promptCopy.copied
              ? t("settings.agents.copied")
              : t("settings.agents.copy")}
          </button>
        </div>
        <code className="block min-w-0 whitespace-pre-wrap break-words rounded-md border border-border bg-background/70 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-foreground">
          {prompt}
        </code>
      </div>
    </div>
  );
}

export function AgentsTab() {
  const t = useT();
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const hfToken = useHfTokenStore((s) => s.token);
  // The copied commands run on the client's machine, so quote and pick the
  // remote shell from the client platform, not the server-reported deviceType.
  const [isWindowsClient] = useState(() => {
    const p = getClientPlatform();
    return p.startsWith("win") || p.includes("windows");
  });
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const localDetection = canUseLocalAgentDetection(serverUrl ?? origin);
  const [agents, setAgents] = useState<string[]>(
    SUPPORTED_AGENTS.map((agent) => agent.id),
  );
  const [selectedAgent, setSelectedAgent] = useState(FALLBACK_AGENT.id);
  const agentSelectionChanged = useRef(false);
  const [detectedAgents, setDetectedAgents] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);
  const [models, setModels] = useState<string[]>([EXAMPLE_MODEL_REPO]);
  const [knownVariants, setKnownVariants] = useState<Record<string, string>>({
    [EXAMPLE_MODEL_REPO]: EXAMPLE_MODEL_VARIANT,
  });
  const [selectedModel, setSelectedModel] = useState(EXAMPLE_MODEL_REPO);
  const modelSelectionChanged = useRef(false);
  const [modelSearch, setModelSearch] = useState("");
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [variants, setVariants] = useState<GgufVariantDetail[]>([]);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(
    EXAMPLE_MODEL_VARIANT,
  );
  const [variantsLoading, setVariantsLoading] = useState(true);
  const [variantsFailed, setVariantsFailed] = useState(false);

  const matchingModels = useMemo(() => {
    const tokens = modelSearch
      .trim()
      .toLowerCase()
      .split(SEARCH_TOKEN_PATTERN)
      .filter(Boolean);
    const matches =
      tokens.length === 0
        ? models
        : models.filter((model) => {
            const normalizedModel = model.toLowerCase();
            return tokens.every((token) => normalizedModel.includes(token));
          });

    if (tokens.length === 0 && matches.includes(selectedModel)) {
      return [
        selectedModel,
        ...matches.filter((model) => model !== selectedModel),
      ];
    }
    return matches;
  }, [modelSearch, models, selectedModel]);

  const visibleModels = matchingModels.slice(0, MODEL_RESULT_LIMIT);
  const preferredVariant = knownVariants[selectedModel] ?? null;
  const selectedAgentDetails = detailsFor(selectedAgent);
  const commandModel = selectedVariant
    ? `${selectedModel}:${selectedVariant}`
    : selectedModel;
  const commandModelArg = quoteShellArg(commandModel, isWindowsClient);
  // Browser commands must target the Studio origin the user is viewing. The
  // desktop app instead uses the backend URL reported by /api/health; its
  // window origin is a Tauri URL and is not reachable by the CLI.
  const commandBase = buildAgentCommand(
    isTauri ? serverUrl : origin,
    REMOTE_API_KEY_PLACEHOLDER,
    isWindowsClient ? "windows" : "unix",
    selectedAgent,
  );
  const command = `${commandBase} --model ${commandModelArg}`;
  const {
    copied,
    copy: handleCopy,
    reset: resetCopied,
  } = useCopyButton(command);
  const remoteCommand = isWindowsClient ? REMOTE_CMD_WINDOWS : REMOTE_CMD_UNIX;

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // Only probe PATH when detection is meaningful; a remote backend's PATH says
  // nothing about the machine the copied command will run on.
  useEffect(() => {
    if (!localDetection) {
      return;
    }
    let cancelled = false;
    loadCodingAgents()
      .then((next) => {
        if (cancelled) {
          return;
        }
        if (next.agents.length > 0) {
          setAgents(next.agents);
          setSelectedAgent((current) => {
            if (agentSelectionChanged.current) {
              return current;
            }
            const detected = next.detected.find((agent) =>
              next.agents.includes(agent),
            );
            return (
              detected ??
              (next.agents.includes(current) ? current : next.agents[0])
            );
          });
        }
        setDetectedAgents(new Set(next.detected));
      })
      .catch(() => {
        // PATH probing is best-effort; the tab is still useful without it.
      })
      .finally(() => {
        if (!cancelled) {
          setLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [localDetection]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      listModels().catch(() => null),
      listCachedGguf().catch(() => []),
      getInferenceStatus().catch(() => null),
    ])
      .then(([info, cachedGgufs, status]) => {
        if (cancelled) {
          return;
        }
        const discovered = discoverGgufModels(
          info?.models ?? [],
          cachedGgufs.map((cached) => cached.repo_id),
        );
        const active = activeGgufSelection(status);
        const models = active
          ? [
              active.model,
              ...discovered.models.filter((model) => model !== active.model),
            ]
          : discovered.models;
        const knownVariants = active?.variant
          ? { ...discovered.variants, [active.model]: active.variant }
          : discovered.variants;
        if (active && !modelSelectionChanged.current) {
          setSelectedModel(active.model);
          setSelectedVariant(active.variant);
        }
        setModels(models);
        setKnownVariants((current) => ({
          ...current,
          ...knownVariants,
        }));
      })
      .catch(() => {
        // The example model keeps the builder useful if discovery fails.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!isHuggingFaceRepo(selectedModel)) {
      return () => {
        cancelled = true;
      };
    }

    listGgufVariants(selectedModel, hfToken || undefined)
      .then((info) => {
        if (cancelled) {
          return;
        }
        // Clear a prior failure once a later request (e.g. after adding a token) succeeds.
        setVariantsFailed(false);
        const uniqueVariants = Array.from(
          new Map(
            info.variants.map((variant) => [variant.quant, variant]),
          ).values(),
        );
        setVariants(uniqueVariants);
        setDefaultVariant(info.default_variant);
        const available = new Set(
          uniqueVariants.map((variant) => variant.quant),
        );
        const nextVariant =
          (preferredVariant && available.has(preferredVariant)
            ? preferredVariant
            : null) ??
          (info.default_variant && available.has(info.default_variant)
            ? info.default_variant
            : null) ??
          uniqueVariants[0]?.quant ??
          null;
        setSelectedVariant(nextVariant);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setVariantsFailed(true);
        setSelectedVariant(preferredVariant);
        if (preferredVariant) {
          setVariants([
            {
              filename: "",
              quant: preferredVariant,
              // API field names intentionally mirror the backend response.
              // biome-ignore lint/style/useNamingConvention: API response field
              size_bytes: 0,
            },
          ]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setVariantsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [hfToken, preferredVariant, selectedModel]);

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

      <section
        aria-label={t("settings.agents.commandBuilder")}
        className="flex w-full flex-col gap-4"
      >
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.agent")}
            </span>
            <a
              href={selectedAgentDetails.docsUrl}
              target="_blank"
              rel="noreferrer"
              aria-label={t("settings.agents.agentDocs", {
                agent: selectedAgentDetails.name,
              })}
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {t("settings.agents.docs")}
              <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
            </a>
          </div>
          <Select
            value={selectedAgent}
            onValueChange={(agent) => {
              agentSelectionChanged.current = true;
              setSelectedAgent(agent);
              resetCopied();
            }}
          >
            <SelectTrigger
              aria-label={t("settings.agents.agent")}
              className="w-full rounded-lg"
            >
              <SelectValue>
                <span className="flex min-w-0 items-center gap-2">
                  <AgentIcon
                    logo={selectedAgentDetails.logo}
                    icon={selectedAgentDetails.icon}
                    darkIcon={selectedAgentDetails.darkIcon}
                    invertIconInDark={selectedAgentDetails.invertIconInDark}
                    color={selectedAgentDetails.color}
                    mark={selectedAgentDetails.mark}
                  />
                  <span className="truncate">{selectedAgentDetails.name}</span>
                </span>
              </SelectValue>
            </SelectTrigger>
            <SelectContent align="start">
              {agents.map((agentId) => {
                const agent = detailsFor(agentId);
                return (
                  <SelectItem key={agent.id} value={agent.id}>
                    <span className="flex min-w-0 items-center gap-2">
                      <AgentIcon
                        logo={agent.logo}
                        icon={agent.icon}
                        darkIcon={agent.darkIcon}
                        invertIconInDark={agent.invertIconInDark}
                        color={agent.color}
                        mark={agent.mark}
                      />
                      <span className="truncate">{agent.name}</span>
                      {localDetection &&
                      loaded &&
                      detectedAgents.has(agent.id) ? (
                        <span className="shrink-0 rounded-full bg-control-accent/10 px-2 py-1 text-[10px] leading-none font-semibold text-control-accent">
                          {t("settings.agents.quickstart.installed")}
                        </span>
                      ) : null}
                    </span>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)_minmax(10rem,0.4fr)] items-start gap-3 max-md:grid-cols-1">
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.model")}
            </span>
            <Popover
              open={modelPickerOpen}
              onOpenChange={(open) => {
                setModelPickerOpen(open);
                if (!open) {
                  setModelSearch("");
                }
              }}
            >
              <PopoverTrigger asChild={true}>
                <button
                  type="button"
                  aria-label={t("settings.agents.model")}
                  aria-expanded={modelPickerOpen}
                  title={selectedModel}
                  className="flex h-9 w-full items-center justify-between gap-2 rounded-lg border border-border bg-background px-3 text-left transition-colors hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:border-transparent dark:bg-white/[0.06] dark:hover:bg-white/10"
                >
                  <span className="min-w-0 truncate font-mono text-xs">
                    {selectedModel}
                  </span>
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    strokeWidth={2}
                    className="size-4 shrink-0 text-muted-foreground"
                  />
                </button>
              </PopoverTrigger>
              <PopoverContent
                align="start"
                sideOffset={4}
                className="w-[var(--radix-popover-trigger-width)] max-w-[calc(100vw-2rem)] gap-0 rounded-lg p-1"
              >
                <Command
                  shouldFilter={false}
                  className="rounded-none bg-transparent p-0"
                >
                  <CommandInput
                    value={modelSearch}
                    onValueChange={setModelSearch}
                    aria-label={t("settings.agents.searchModels")}
                    placeholder={t("settings.agents.searchModels")}
                    className="font-mono text-xs"
                  />
                  <CommandList>
                    <CommandEmpty>{t("settings.agents.noModels")}</CommandEmpty>
                    {visibleModels.map((model) => (
                      <CommandItem
                        key={model}
                        value={model}
                        data-checked={model === selectedModel}
                        onSelect={() => {
                          modelSelectionChanged.current = true;
                          setSelectedModel(model);
                          setSelectedVariant(knownVariants[model] ?? null);
                          setVariants([]);
                          setDefaultVariant(null);
                          setVariantsFailed(false);
                          setVariantsLoading(isHuggingFaceRepo(model));
                          setModelSearch("");
                          setModelPickerOpen(false);
                          resetCopied();
                        }}
                        className="cursor-pointer font-mono text-xs"
                      >
                        <span className="min-w-0 truncate">{model}</span>
                      </CommandItem>
                    ))}
                  </CommandList>
                  {matchingModels.length > visibleModels.length ? (
                    <p className="border-t border-border/60 px-3 py-2 text-[11px] text-muted-foreground">
                      {t("settings.agents.showingModels", {
                        shown: visibleModels.length,
                        total: matchingModels.length,
                      })}
                    </p>
                  ) : null}
                </Command>
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.quantization")}
            </span>
            <Select
              value={selectedVariant ?? undefined}
              onValueChange={(variant) => {
                setSelectedVariant(variant);
                resetCopied();
              }}
              disabled={variantsLoading || variants.length === 0}
            >
              <SelectTrigger
                aria-label={t("settings.agents.quantization")}
                className="w-full rounded-lg font-mono text-xs"
              >
                <SelectValue
                  placeholder={
                    variantsLoading
                      ? t("settings.agents.loadingQuantizations")
                      : t("settings.agents.noQuantizations")
                  }
                >
                  {selectedVariant}
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="start">
                {variants.map((variant) => {
                  const metadata = [
                    variant.quant === defaultVariant
                      ? t("settings.agents.recommended")
                      : null,
                    variant.downloaded ? t("settings.agents.downloaded") : null,
                    formatBytes(
                      variant.download_size_bytes ?? variant.size_bytes,
                    ),
                  ].filter(Boolean);
                  return (
                    <SelectItem key={variant.quant} value={variant.quant}>
                      <span className="font-mono text-xs">{variant.quant}</span>
                      {metadata.length > 0 ? (
                        <span className="text-[10px] text-muted-foreground">
                          {metadata.join(" · ")}
                        </span>
                      ) : null}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        </div>

        {variantsFailed ? (
          <p className="text-[11px] leading-relaxed text-amber-700 dark:text-amber-400">
            {t("settings.agents.quantizationLoadError")}
          </p>
        ) : null}

        <div className="flex min-w-0 flex-col gap-2 rounded-lg border border-border bg-muted/20 p-3">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-foreground">
              {t("settings.agents.generatedCommand")}
            </span>
            <button
              type="button"
              onClick={handleCopy}
              aria-label={t("settings.agents.copyGeneratedCommand")}
              className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background/70 px-2 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={copied ? Tick02Icon : Copy01Icon}
                className={cn("size-3.5", copied && "text-control-accent")}
              />
              {copied ? t("settings.agents.copied") : t("settings.agents.copy")}
            </button>
          </div>
          <code className="block min-w-0 whitespace-pre-wrap break-all rounded-md border border-border bg-background/70 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-foreground">
            {command}
          </code>
        </div>

        <SubagentSection
          key={`${selectedAgent}:${commandModel}`}
          baseCommand={commandBase}
          commandModelArg={commandModelArg}
          agent={selectedAgentDetails}
        />

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {t("settings.agents.modelNote")}
        </p>
      </section>

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
          {PASSTHROUGH_COMMANDS.map((command) => (
            <CommandBlock key={command} command={command} />
          ))}
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
