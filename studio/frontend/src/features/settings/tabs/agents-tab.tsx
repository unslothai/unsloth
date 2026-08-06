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
  type LocalModelInfo,
  getInferenceStatus,
  listCachedGguf,
  listGgufVariants,
  listLocalModels,
  listModels,
} from "@/features/chat";
import { useHfTokenStore } from "@/features/hub";
import type { TranslationKey } from "@/i18n";
import { useT } from "@/i18n";
import { getApiBase, isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon, Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiProviderLogo } from "../../chat/api-provider-logo";
import { loadCodingAgents } from "../api/coding-agents";
import {
  buildAgentCommand,
  isLoopbackHost,
  normalizeHost,
} from "../components/agent-command";
import { SettingsSection } from "../components/settings-section";
import { psSingle, shSingle } from "../components/usage-examples";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";

const DOCS_URL = "https://unsloth.ai/docs/integrations/unsloth-start";
const EXAMPLE_MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF";
const EXAMPLE_MODEL_VARIANT = "UD-Q4_K_XL";
const MODEL_RESULT_LIMIT = 7;
const STATUS_POLL_MS = 5000;
const HUGGING_FACE_REPO_PATTERN = /^[^/\\:\s]+\/[^/\\:\s]+$/;
const SEARCH_TOKEN_PATTERN = /\s+/;
const SAFE_SHELL_ARG_PATTERN = /^[A-Za-z0-9_./:@%+=,-]+$/;
const SUBAGENT_AGENT_IDS = new Set(["claude", "codex", "opencode"]);

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

// Names are untranslated, so `settings.agents.intro` lists them all to keep them searchable.
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

function looksLikePath(value: string): boolean {
  return (
    value.includes("\\") ||
    value.startsWith("/") ||
    value.startsWith("~") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    (value.length >= 2 && value[1] === ":") ||
    value.split("/").length > 2
  );
}

// hugging face ids fold case; a path does not, since Linux paths are sensitive.
function modelKey(model: string): string {
  return looksLikePath(model) ? model : model.toLowerCase();
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
  // Hugging Face ids are case-insensitive, and the catalog and cache endpoints can
  // disagree on spelling; two rows for one repo would leave the load id on only one.
  const seen = new Set(models.map(modelKey));
  const add = (model: string) => {
    const key = modelKey(model);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    models.push(model);
  };
  for (const model of items) {
    // /api/models/list reports the backend's raw identifier, which for a native
    // grant is the host path that status deliberately withholds. The resident
    // model reaches the picker through status instead, so drop path-shaped ids
    // rather than leak one into the list and into the copied command.
    if (!model.is_gguf || looksLikePath(model.id)) {
      continue;
    }
    const parsed = splitModelVariant(model.id);
    if (parsed.repo) {
      add(parsed.repo);
    }
    if (parsed.variant && !variants[parsed.repo]) {
      variants[parsed.repo] = parsed.variant;
    }
  }
  for (const repo of cachedRepos) {
    add(repo);
  }

  return { models, variants };
}

// Scanned local GGUFs (./models, LM Studio, custom folders) that the caches above
// miss. The id is the load id, i.e. the on-disk path for anything outside the active
// cache, so label the row by repo id when there is one but keep the path to load by.
// model_format is only set by the scanners that compute it: _scan_hf_cache leaves it
// unset, so a custom scan folder holding an HF cache layout would vanish from the
// picker on an exclusive check. Treat unset as unknown and fall back to the name.
function isLocalGguf(model: LocalModelInfo): boolean {
  // The scanners set this only for a directory holding a primary, non-mmproj GGUF
  // and no other weights, so an unset format means "not GGUF", not "unknown". Do not
  // guess from the name: a safetensors folder called Foo-GGUF would load the
  // transformers backend and then fail the GGUF-only agents.
  return (model.model_format ?? "").toLowerCase() === "gguf";
}

function localGgufEntries(
  models: LocalModelInfo[],
): { id: string; label: string }[] {
  const entries: { id: string; label: string }[] = [];
  for (const model of models) {
    // partial marks an interrupted sharded download: variant discovery would treat
    // the shards it has as complete and build a command that fails on load. The
    // cached repo row still offers it, and _repo_gguf_load_id withholds the path.
    if (model.partial || !(model.id && isLocalGguf(model))) {
      continue;
    }
    // The path is the identity: two scanned models can share a basename, and it is
    // also what --model needs. The friendly name is display only.
    entries.push({
      id: model.id,
      label: model.model_id || model.display_name || model.id,
    });
  }
  return entries;
}

// First candidate the repo actually offers: an explicit pick, then the remembered
// one, then the repo default.
function pickVariant(
  available: Set<string>,
  candidates: (string | null | undefined)[],
): string | null {
  for (const candidate of candidates) {
    if (candidate && available.has(candidate)) {
      return candidate;
    }
  }
  return null;
}

function activeGgufSelection(
  status: InferenceStatusResponse | null,
): { model: string; variant: string | null; named: boolean } | null {
  if (!status?.is_gguf) {
    return null;
  }
  if (!status.model_identifier) {
    // A native file grant withholds the host path, so this GGUF is resident but
    // has no id to pass. Carry its label and attach with a bare command instead.
    return status.active_model
      ? {
          model: status.active_model,
          variant: status.gguf_variant ?? null,
          named: false,
        }
      : null;
  }
  const active = splitModelVariant(status.model_identifier);
  if (!active.repo) {
    return null;
  }
  return {
    // Status reports the quant for path loads too, whose id has no ":variant" suffix.
    model: active.repo,
    variant: status.gguf_variant ?? active.variant,
    named: true,
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
      <span className="flex size-5 shrink-0 items-center justify-center overflow-hidden rounded">
        <ApiProviderLogo providerType={logo} className="size-5 rounded" />
      </span>
    );
  }
  if (icon) {
    const iconSrc = `${import.meta.env.BASE_URL}agent-logos/${icon}`;
    const darkIconSrc = darkIcon
      ? `${import.meta.env.BASE_URL}agent-logos/${darkIcon}`
      : null;
    return (
      <span className="flex size-5 shrink-0 items-center justify-center overflow-hidden rounded">
        <img
          src={iconSrc}
          alt=""
          aria-hidden={true}
          className={cn(
            "size-5 object-contain",
            darkIconSrc && "dark:hidden",
            invertIconInDark && "dark:invert",
          )}
        />
        {darkIconSrc ? (
          <img
            src={darkIconSrc}
            alt=""
            aria-hidden={true}
            className="hidden size-5 object-contain dark:block"
          />
        ) : null}
      </span>
    );
  }
  return (
    <span
      aria-hidden={true}
      style={{ backgroundColor: color }}
      className="flex size-5 shrink-0 items-center justify-center rounded font-heading text-ui-10 font-semibold text-white"
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
const PASSTHROUGH_EXAMPLES = [
  { agent: "claude", flags: "--continue" },
  { agent: "codex", flags: "--persist resume --last" },
];

const DRY_RUN_FLAGS = "--no-launch";

/** Code box with the copy control inside it, top-right. Presentational: the
 *  copy state stays with the caller so existing resets still apply. */
function CopyableCode({
  value,
  copyLabel,
  copied,
  onCopy,
  breakAll = true,
}: {
  value: string;
  copyLabel: string;
  copied: boolean;
  onCopy: () => void;
  breakAll?: boolean;
}) {
  const t = useT();

  return (
    <div className="relative min-w-0">
      <code
        className={cn(
          "block min-w-0 whitespace-pre-wrap rounded-lg border border-border bg-background/70 py-2.5 pr-9 pl-4 font-mono text-ui-11 leading-relaxed text-foreground dark:border-transparent dark:bg-white/[0.05]",
          breakAll ? "break-all" : "break-words",
        )}
      >
        {value}
      </code>
      <button
        type="button"
        onClick={onCopy}
        aria-label={copyLabel}
        className="absolute top-1.5 right-1.5 flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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

function CommandBlock({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);

  return (
    <div className="group relative overflow-hidden rounded-xl border border-border bg-muted/40 dark:border-transparent dark:bg-white/[0.04]">
      <pre className="hover-scrollbar overflow-x-auto py-3 pr-11 pl-4 text-xs leading-relaxed text-foreground">
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

// Quote only values with shell metacharacters, e.g. a local path with spaces.
function quoteShellArg(value: string, windows: boolean): string {
  if (SAFE_SHELL_ARG_PATTERN.test(value)) {
    return value;
  }
  return windows ? `'${psSingle(value)}'` : `'${shSingle(value)}'`;
}

function SubagentSection({
  agent,
  baseCommand,
  modelArgs,
}: {
  agent: AgentDetails;
  baseCommand: string;
  modelArgs: string;
}) {
  const t = useT();
  // modelArgs is empty when attaching to a resident model that has no id to name.
  const command = `${baseCommand} --as-subagent${modelArgs ? ` ${modelArgs}` : ""}`;
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
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-col gap-1">
        <span
          data-settings-label={t("settings.agents.subagent.title")}
          className="text-xs font-medium text-foreground"
        >
          {t("settings.agents.subagent.title")}
        </span>
        <p className="text-ui-11 leading-relaxed text-muted-foreground">
          {t("settings.agents.subagent.description", { agent: agent.name })}
        </p>
      </div>

      <div className="flex min-w-0 flex-col gap-2">
        <span className="text-ui-11 font-medium text-foreground">
          {t("settings.agents.subagent.setupCommand")}
        </span>
        <CopyableCode
          value={command}
          copyLabel={t("settings.agents.subagent.copySetupCommand")}
          copied={commandCopy.copied}
          onCopy={commandCopy.copy}
        />
      </div>

      <div className="flex min-w-0 flex-col gap-2">
        <span className="text-ui-11 font-medium text-foreground">
          {t("settings.agents.subagent.usagePrompt", { agent: agent.name })}
        </span>
        <CopyableCode
          value={prompt}
          copyLabel={t("settings.agents.subagent.copyUsagePrompt")}
          copied={promptCopy.copied}
          onCopy={promptCopy.copy}
          breakAll={false}
        />
      </div>
    </div>
  );
}

export function AgentsTab() {
  const t = useT();
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const hfToken = useHfTokenStore((s) => s.token);
  const deviceType = usePlatformStore((s) => s.deviceType);
  // The remote snippet runs on the client, so use the client platform, not deviceType.
  // Anchor the match: a bare includes("win") would also match "darwin".
  const [isWindowsClient] = useState(() => {
    const p = getClientPlatform();
    return p.startsWith("win") || p.includes("windows");
  });
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  // Browser commands target the viewed origin; a desktop window origin is a Tauri URL
  // the CLI cannot reach, so use the backend URL from /api/health (getApiBase until it
  // lands). The command then runs wherever that CLI is: a loopback base is this Studio's
  // own host, so deviceType decides, and it reports wsl where the browser would claim
  // Windows; any other base is reached from the viewer's machine, so only the client
  // platform describes that shell.
  const studioBase = isTauri ? (serverUrl ?? getApiBase()) : origin;
  const isWindowsShell = isLoopbackBase(studioBase)
    ? deviceType === "windows"
    : isWindowsClient;
  const localDetection = canUseLocalAgentDetection(serverUrl ?? origin);
  const setStoredAgent = useSettingsPanelPrefsStore((s) => s.setAgentsAgent);
  const setStoredModel = useSettingsPanelPrefsStore((s) => s.setAgentsModel);
  const setStoredVariant = useSettingsPanelPrefsStore(
    (s) => s.setAgentsVariant,
  );
  // read once: these seed the controls, which write back through the handlers.
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [agents, setAgents] = useState<string[]>(
    SUPPORTED_AGENTS.map((agent) => agent.id),
  );
  const [selectedAgent, setSelectedAgent] = useState(
    storedPrefs.agentsAgent ?? FALLBACK_AGENT.id,
  );
  // a restored pick counts as explicit, or detection would overwrite it.
  const agentSelectionChanged = useRef(storedPrefs.agentsAgent != null);
  const [detectedAgents, setDetectedAgents] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);
  // list the restored model until the discovery scan confirms or retires it.
  const [models, setModels] = useState<string[]>(
    storedPrefs.agentsModel && storedPrefs.agentsModel !== EXAMPLE_MODEL_REPO
      ? [storedPrefs.agentsModel, EXAMPLE_MODEL_REPO]
      : [EXAMPLE_MODEL_REPO],
  );
  const [cachedLoadIds, setCachedLoadIds] = useState<Record<string, string>>(
    {},
  );
  // Display names for scanned models, keyed by the path that identifies them.
  const [modelLabels, setModelLabels] = useState<Record<string, string>>({});
  // The model /api/inference/status reports as resident, so the command attaches to it
  // rather than remapping to another cached copy.
  const [activeStatusModel, setActiveStatusModel] = useState<string | null>(
    null,
  );
  // Set only for a native-grant GGUF, which is resident but has no id to pass.
  const [attachOnlyModel, setAttachOnlyModel] = useState<string | null>(null);
  const [knownVariants, setKnownVariants] = useState<Record<string, string>>({
    [EXAMPLE_MODEL_REPO]: EXAMPLE_MODEL_VARIANT,
  });
  const [selectedModel, setSelectedModel] = useState(
    storedPrefs.agentsModel ?? EXAMPLE_MODEL_REPO,
  );
  const modelSelectionChanged = useRef(storedPrefs.agentsModel != null);
  // held until discovery and the first status can confirm the restored model.
  const restoredModel = useRef<string | null>(storedPrefs.agentsModel);
  const [discoveredKeys, setDiscoveredKeys] = useState<Set<string> | null>(
    null,
  );
  const [statusSettled, setStatusSettled] = useState(false);
  // The model status last reported, for the discovery scan to preserve.
  const activeModelRef = useRef<string | null>(null);
  // Only the newest status request may apply; a slow earlier one must not win.
  const statusSeq = useRef(0);
  // A quant picked by hand, scoped to its repo: polling and refetches must not
  // overwrite it, but it must not follow the selection onto a different repo.
  // seeding a restored quant here puts it through the same fetch a fresh one is.
  const chosenVariant = useRef<{ model: string; variant: string } | null>(
    storedPrefs.agentsModel && storedPrefs.agentsVariant
      ? { model: storedPrefs.agentsModel, variant: storedPrefs.agentsVariant }
      : null,
  );
  const [modelSearch, setModelSearch] = useState("");
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [variants, setVariants] = useState<GgufVariantDetail[]>([]);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(
    storedPrefs.agentsModel ? storedPrefs.agentsVariant : EXAMPLE_MODEL_VARIANT,
  );
  const [variantsLoading, setVariantsLoading] = useState(true);
  const [variantsFailed, setVariantsFailed] = useState(false);

  const labelFor = (model: string) => modelLabels[model] ?? model;
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
            // Search both, so a scanned model is findable by name and by path.
            const haystack =
              `${model} ${modelLabels[model] ?? ""}`.toLowerCase();
            return tokens.every((token) => haystack.includes(token));
          });

    if (tokens.length === 0 && matches.includes(selectedModel)) {
      return [
        selectedModel,
        ...matches.filter((model) => model !== selectedModel),
      ];
    }
    return matches;
  }, [modelLabels, modelSearch, models, selectedModel]);

  const visibleModels = matchingModels.slice(0, MODEL_RESULT_LIMIT);
  const preferredVariant = knownVariants[selectedModel] ?? null;
  const selectedAgentDetails = detailsFor(selectedAgent);
  // A GGUF outside the active cache does not resolve by repo id, so name its
  // snapshot path; `unsloth start` now also matches a path by the basename
  // /v1/models advertises for it. The resident model is exempt: it already
  // loaded by id, and cached-gguf keeps the largest copy across caches, whose
  // snapshot could switch cache or quant under it.
  const cachedLoadId =
    selectedModel === activeStatusModel
      ? null
      : (cachedLoadIds[selectedModel] ??
        cachedLoadIds[selectedModel.toLowerCase()] ??
        null);
  const modelId = cachedLoadId ?? selectedModel;
  const suffixVariant = isHuggingFaceRepo(modelId);
  const commandModel =
    selectedVariant && suffixVariant
      ? `${modelId}:${selectedVariant}`
      : modelId;
  const commandModelArg = quoteShellArg(commandModel, isWindowsShell);
  // A bare `unsloth start` attaches to whatever is loaded, which is the only way
  // to reach a native-grant GGUF: naming it would switch the server to another model.
  const attachOnly = selectedModel === attachOnlyModel;
  const modelArgs = attachOnly
    ? ""
    : selectedVariant && !suffixVariant
      ? `--model ${commandModelArg} --gguf-variant ${quoteShellArg(selectedVariant, isWindowsShell)}`
      : `--model ${commandModelArg}`;
  // No key is passed: the CLI caches an explicit one per base, overwriting a working
  // saved key. Omitting it replays the saved key; the remote section covers first setup.
  const commandOs = isWindowsShell ? "windows" : "unix";
  const commandBase = buildAgentCommand(
    studioBase,
    null,
    commandOs,
    selectedAgent,
  );
  const command = attachOnly ? commandBase : `${commandBase} ${modelArgs}`;
  // The fixed examples below target the same Studio, not a bare 127.0.0.1:8888.
  const example = (agentId: string, flags: string) =>
    `${buildAgentCommand(studioBase, null, commandOs, agentId)} ${flags}`;
  const {
    copied,
    copy: handleCopy,
    reset: resetCopied,
  } = useCopyButton(command);
  const remoteCommand = isWindowsClient ? REMOTE_CMD_WINDOWS : REMOTE_CMD_UNIX;

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // A remote backend's PATH says nothing about the machine running the copied command.
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
        // Best-effort; the tab still works without PATH detection.
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

  // a restored agent the backend no longer lists cannot build a command.
  useEffect(() => {
    if (!agentSelectionChanged.current) return;
    if (localDetection && !loaded) return;
    if (agents.includes(selectedAgent)) return;
    agentSelectionChanged.current = false;
    setStoredAgent(null);
    setSelectedAgent(
      agents.find((agent) => detectedAgents.has(agent)) ??
        agents[0] ??
        FALLBACK_AGENT.id,
    );
  }, [
    agents,
    detectedAgents,
    loaded,
    localDetection,
    selectedAgent,
    setStoredAgent,
  ]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      listModels().catch(() => null),
      listCachedGguf().catch(() => []),
      listLocalModels().catch(() => null),
    ])
      .then(([info, cachedGgufs, local]) => {
        if (cancelled) {
          return;
        }
        const localEntries = localGgufEntries(local?.models ?? []);
        const discovered = discoverGgufModels(info?.models ?? [], [
          ...cachedGgufs.map((cached) => cached.repo_id),
          ...localEntries.map((entry) => entry.id),
        ]);
        // Keep the snapshot load_id for --model while listing the model by repo id.
        const loadIds: Record<string, string> = {};
        for (const cached of cachedGgufs) {
          if (cached.load_id && cached.load_id !== cached.repo_id) {
            // Key both spellings: the merge above keeps whichever casing arrived
            // first, which may not be this endpoint's.
            loadIds[cached.repo_id] = cached.load_id;
            loadIds[cached.repo_id.toLowerCase()] = cached.load_id;
          }
        }
        const labels: Record<string, string> = {};
        for (const entry of localEntries) {
          if (entry.label !== entry.id) {
            labels[entry.id] = entry.label;
          }
        }
        setDiscoveredKeys(new Set(discovered.models.map(modelKey)));
        // Status is applied on its own schedule now, so keep whatever model it has
        // already adopted rather than dropping it when this slower scan lands.
        setModels(() => {
          const active = activeModelRef.current;
          return active && !discovered.models.includes(active)
            ? [active, ...discovered.models]
            : discovered.models;
        });
        setCachedLoadIds(loadIds);
        setModelLabels(labels);
        setKnownVariants((current) => ({
          ...current,
          ...discovered.variants,
        }));
      })
      .catch(() => {
        // The example model keeps the builder useful if discovery fails.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // List the resident model and follow it, unless the user picked one explicitly.
  const adoptActiveModel = useCallback(
    (active: { model: string; variant: string | null }) => {
      setModels((current) =>
        current.includes(active.model) ? current : [active.model, ...current],
      );
      if (active.variant) {
        setKnownVariants((current) => ({
          ...current,
          [active.model]: active.variant as string,
        }));
      }
      if (!modelSelectionChanged.current) {
        setSelectedModel(active.model);
        if (chosenVariant.current?.model !== active.model) {
          setSelectedVariant(active.variant);
        }
      }
    },
    [],
  );

  // A native-grant label only stands for whatever was resident at the time, so once
  // that model is replaced the label cannot name anything and has to go, even when
  // it was picked by hand: leaving it selected would emit it as --model.
  const retireAttachOnly = useCallback((label: string, replacement: string) => {
    setModels((current) => current.filter((model) => model !== label));
    setSelectedModel((current) => {
      if (current !== label) {
        return current;
      }
      // Drop the quant in the same transition: it belonged to the label, and an
      // explicit pick stops adoptActiveModel from correcting it afterwards.
      chosenVariant.current = null;
      setSelectedVariant(null);
      return replacement;
    });
  }, []);

  // The resident GGUF went away (unloaded, or replaced by a transformer model).
  // Following it means letting go too, or the command would name a stale model and
  // switch the shared server back. A native-grant label is not even loadable, so it
  // leaves the list entirely. An explicit pick still wins.
  const dropActiveModel = useCallback(
    (attachOnly: string | null, wasActive: string | null) => {
      if (attachOnly) {
        setModels((current) => current.filter((model) => model !== attachOnly));
        // Even a deliberate pick has to go: the label stood for a withheld path, so
        // naming it would emit --model <label>, which cannot reload anything.
        setSelectedModel((current) =>
          current === attachOnly ? EXAMPLE_MODEL_REPO : current,
        );
      }
      if (modelSelectionChanged.current || !wasActive) {
        return;
      }
      // Only the model this tab adopted by itself is dropped; anything the user
      // picked is theirs to keep.
      setSelectedModel((current) =>
        current === wasActive ? EXAMPLE_MODEL_REPO : current,
      );
      setSelectedVariant((current) => (current === null ? current : null));
    },
    [],
  );

  const applyStatus = useCallback(
    (status: InferenceStatusResponse) => {
      const active = activeGgufSelection(status);
      activeModelRef.current = active?.model ?? null;
      const wasAttachOnly = attachOnlyModel;
      setActiveStatusModel(active?.model ?? null);
      setAttachOnlyModel(active && !active.named ? active.model : null);
      if (!active) {
        dropActiveModel(wasAttachOnly, activeStatusModel);
        return;
      }
      if (wasAttachOnly && wasAttachOnly !== active.model) {
        retireAttachOnly(wasAttachOnly, active.model);
      }
      adoptActiveModel(active);
    },
    [
      activeStatusModel,
      adoptActiveModel,
      attachOnlyModel,
      dropActiveModel,
      retireAttachOnly,
    ],
  );

  // Another client, or a load that finishes after this tab opens, can change what
  // is resident on a shared server. Keep tracking it rather than pinning the model
  // seen at mount, or the command would name a stale one and switch the server
  // back, unloading it for every attached session. An explicit pick still wins.
  useEffect(() => {
    let cancelled = false;
    const sync = () => {
      const seq = ++statusSeq.current;
      getInferenceStatus()
        .then((status) => {
          if (!cancelled && seq === statusSeq.current) {
            applyStatus(status);
          }
        })
        .catch(() => {
          // A failed poll just leaves the last known selection in place.
        })
        .finally(() => {
          if (!cancelled) setStatusSettled(true);
        });
    };
    sync();
    const timer = window.setInterval(sync, STATUS_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [applyStatus]);

  // retiring a restored pick needs both reads: either alone can miss a model
  // the other knows about, and a wrong retire drops the user's choice.
  useEffect(() => {
    const restored = restoredModel.current;
    if (!(restored && discoveredKeys && statusSettled)) return;
    restoredModel.current = null;
    if (
      discoveredKeys.has(modelKey(restored)) ||
      activeModelRef.current === restored
    ) {
      return;
    }
    // an uncached pick would pin the builder to a model the CLI cannot load.
    modelSelectionChanged.current = false;
    chosenVariant.current = null;
    setStoredModel(null, null);
    setSelectedModel(EXAMPLE_MODEL_REPO);
    setSelectedVariant(EXAMPLE_MODEL_VARIANT);
  }, [discoveredKeys, statusSettled, setStoredModel]);

  useEffect(() => {
    let cancelled = false;

    // A scanned directory is not repo-shaped but still has a path to enumerate, and
    // after discovery that path IS the identity. Only a standalone .gguf file, which
    // is one quant by definition, is genuinely variantless.
    const localDir =
      cachedLoadId ??
      (looksLikePath(selectedModel) &&
      !selectedModel.toLowerCase().endsWith(".gguf")
        ? selectedModel
        : null);
    if (!(isHuggingFaceRepo(selectedModel) || localDir)) {
      // A loose .gguf is one quant already. Status can record a quant parsed from its
      // filename, and restoring that would add --gguf-variant, which a bare file path
      // cannot resolve, so it stays null here.
      const standaloneFile = selectedModel.toLowerCase().endsWith(".gguf");
      queueMicrotask(() => {
        if (cancelled) {
          return;
        }
        setVariants([]);
        setSelectedVariant(standaloneFile ? null : preferredVariant);
        setVariantsFailed(false);
        setVariantsLoading(false);
      });
      return () => {
        cancelled = true;
      };
    }

    // A programmatic model change reaches here too, so clear the previous model's
    // quants up front rather than leaving them selectable until this resolves.
    setVariants([]);
    setVariantsLoading(true);
    // Offer the quants from the same place the command loads from, not remote-only ones.
    listGgufVariants(selectedModel, hfToken || undefined, {
      preferLocalCache: localDir != null,
      localPath: localDir,
    })
      .then((info) => {
        if (cancelled) {
          return;
        }
        // Clear a prior failure once a later request (e.g. after adding a token) succeeds.
        setVariantsFailed(false);
        // Drop partial quants: an interrupted split download still lists a quant, and
        // naming it builds a command that resolves the shards it has and then fails on
        // the missing ones.
        const uniqueVariants = Array.from(
          new Map(
            info.variants
              .filter((variant) => !variant.partial)
              .map((variant) => [variant.quant, variant]),
          ).values(),
        );
        setVariants(uniqueVariants);
        const available = new Set(
          uniqueVariants.map((variant) => variant.quant),
        );
        const nextVariant =
          pickVariant(available, [
            chosenVariant.current?.model === selectedModel
              ? chosenVariant.current.variant
              : null,
            preferredVariant,
            info.default_variant,
          ]) ??
          uniqueVariants[0]?.quant ??
          null;
        setSelectedVariant(nextVariant);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setVariantsFailed(true);
        setVariants([]);
        setSelectedVariant(preferredVariant);
        if (preferredVariant) {
          setVariants([
            {
              filename: "",
              quant: preferredVariant,
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
  }, [cachedLoadId, hfToken, preferredVariant, selectedModel]);

  // No GGUF warning for `codex` (unsloth_cli's _require_gguf_for_codex): the
  // picker only ever offers GGUF models.

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-8">
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
        {/* The chip is the docs entry point, so no separate link is needed.
            No aria-label: it would replace the visible "unsloth start" as the
            accessible name, leaving voice control unable to target it. */}
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          title={t("settings.agents.readDocs")}
          className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground underline decoration-border decoration-dotted underline-offset-2 transition-colors hover:decoration-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-white/[0.08]"
        >
          unsloth start
        </a>{" "}
        {t("settings.agents.intro")}
      </p>

      <section
        aria-label={t("settings.agents.commandBuilder")}
        className="flex w-full flex-col gap-6"
      >
        {/* Keyed to this pane, not the viewport. The dialog leaves the tab
            about 440px at a 768px window, where three columns crush the agent
            and model controls; 34rem is the point all three stay usable. */}
        <div className="@container">
          <div className="grid grid-cols-1 items-start gap-3 @[34rem]:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(9rem,0.5fr)]">
            <div className="flex min-w-0 flex-col gap-1.5">
              {/* Fixed height on every column header, so the padded docs link
                  here cannot push this control below the other two. */}
              <div className="flex h-5 items-center justify-between gap-3">
                <span
                  data-settings-label={t("settings.agents.agent")}
                  className="text-xs font-medium text-foreground"
                >
                  {t("settings.agents.agent")}
                </span>
                <a
                  href={selectedAgentDetails.docsUrl}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={t("settings.agents.agentDocs", {
                    agent: selectedAgentDetails.name,
                  })}
                  className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
                  setStoredAgent(agent);
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
                      <span className="truncate">
                        {selectedAgentDetails.name}
                      </span>
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
                            <span className="shrink-0 rounded-full bg-control-accent/10 px-2 py-1 text-ui-10 leading-none font-semibold text-control-accent">
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

            <div className="flex min-w-0 flex-col gap-1.5">
              <div className="flex h-5 items-center">
                <span
                  data-settings-label={t("settings.agents.model")}
                  className="text-xs font-medium text-foreground"
                >
                  {t("settings.agents.model")}
                </span>
              </div>
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
                      {labelFor(selectedModel)}
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
                      <CommandEmpty>
                        {t("settings.agents.noModels")}
                      </CommandEmpty>
                      {visibleModels.map((model) => (
                        <CommandItem
                          key={model}
                          value={model}
                          data-checked={model === selectedModel}
                          onSelect={() => {
                            modelSelectionChanged.current = true;
                            restoredModel.current = null;
                            const variant = knownVariants[model] ?? null;
                            setSelectedModel(model);
                            setSelectedVariant(variant);
                            // a native-grant label names no path to reuse.
                            setStoredModel(
                              model === attachOnlyModel ? null : model,
                              model === attachOnlyModel ? null : variant,
                            );
                            setVariants([]);
                            setVariantsFailed(false);
                            setVariantsLoading(isHuggingFaceRepo(model));
                            setModelSearch("");
                            setModelPickerOpen(false);
                            resetCopied();
                          }}
                          className="cursor-pointer font-mono text-xs"
                        >
                          <span className="min-w-0 truncate" title={model}>
                            {labelFor(model)}
                          </span>
                        </CommandItem>
                      ))}
                    </CommandList>
                    {matchingModels.length > visibleModels.length ? (
                      <p className="border-t border-border/60 px-3 py-2 text-ui-11 text-muted-foreground">
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
              <div className="flex h-5 items-center">
                <span
                  data-settings-label={t("settings.agents.quantization")}
                  className="text-xs font-medium text-foreground"
                >
                  {t("settings.agents.quantization")}
                </span>
              </div>
              <Select
                value={selectedVariant ?? undefined}
                onValueChange={(variant) => {
                  chosenVariant.current = { model: selectedModel, variant };
                  setSelectedVariant(variant);
                  // a quant picked while following the resident model is its own.
                  if (
                    useSettingsPanelPrefsStore.getState().agentsModel ===
                    selectedModel
                  ) {
                    setStoredVariant(variant);
                  }
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
                <SelectContent align="start" className="min-w-[16rem]">
                  {variants.map((variant) => {
                    // Size only: the recommended/downloaded tags wrapped every
                    // row onto two lines and made the list hard to scan.
                    const size = formatBytes(
                      variant.download_size_bytes ?? variant.size_bytes,
                    );
                    return (
                      <SelectItem
                        key={variant.quant}
                        value={variant.quant}
                        // Stretch the item text so the size can sit flush right,
                        // giving the list a clean two-column read.
                        className="[&>span:last-child]:w-full [&>span:last-child]:justify-between"
                      >
                        <span className="font-mono text-xs whitespace-nowrap">
                          {variant.quant}
                        </span>
                        {size ? (
                          <span className="text-ui-10 whitespace-nowrap text-muted-foreground">
                            {size}
                          </span>
                        ) : null}
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {variantsFailed ? (
          <p className="text-ui-11 leading-relaxed text-amber-700 dark:text-amber-400">
            {t("settings.agents.quantizationLoadError")}
          </p>
        ) : null}

        <div className="flex min-w-0 flex-col gap-2.5">
          <span className="text-xs font-medium text-foreground">
            {t("settings.agents.generatedCommand")}
          </span>
          <CopyableCode
            value={command}
            copyLabel={t("settings.agents.copyGeneratedCommand")}
            copied={copied}
            onCopy={handleCopy}
          />
        </div>

        <SubagentSection
          key={`${selectedAgent}:${commandModel}`}
          baseCommand={commandBase}
          modelArgs={modelArgs}
          agent={selectedAgentDetails}
        />

        <p className="text-ui-11 leading-relaxed text-muted-foreground">
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
        <div className="pt-3">
          <CommandBlock command={remoteCommand} />
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.passthrough.title")}
        description={t("settings.agents.passthrough.description")}
      >
        <div className="flex flex-col gap-3 pt-3">
          {PASSTHROUGH_EXAMPLES.map(({ agent, flags }) => (
            <CommandBlock key={flags} command={example(agent, flags)} />
          ))}
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.agents.dryRun.title")}
        description={t("settings.agents.dryRun.description")}
      >
        <div className="pt-3">
          <CommandBlock command={example("claude", DRY_RUN_FLAGS)} />
        </div>
      </SettingsSection>
    </div>
  );
}
