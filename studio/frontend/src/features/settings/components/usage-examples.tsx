// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import {
  getInferenceStatus,
  isExternalModelId,
  useChatRuntimeStore,
} from "@/features/chat";
import { publicModelId } from "@/features/hub";
import { isServedByLlamaCpp } from "@/features/model-picker";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { AiSpeechIcon, MessageCircleIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  BotIcon,
  Image01Icon,
  InformationCircleIcon,
  McpServerIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { loadCodingAgents } from "../api/coding-agents";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { loadOpenAIAutoSwitchSettings } from "../api/openai-auto-switch";
import { type OpenAIModel, listOpenAIModels } from "../api/openai-models";
import { scrollIntoPane } from "../lib/scroll-into-pane";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import {
  agentRunsOnActiveModel,
  buildAgentCommand,
  compatibilityFromSources,
  fallbackAgent,
  isLoopbackHost,
  normalizeHost,
  pickCompatibleAgent,
  psSingle,
  sameBaseModelId,
  shSingle,
  statusGgufVerdict,
} from "./agent-command";
import { AgentIcon } from "./agent-command-block";
import {
  ExampleCodeBox,
  ExampleToggle,
  type ExampleOs,
  isLoopbackUrl,
  useExampleOs,
} from "./api-example-parts";
import { detailsFor } from "./coding-agent-list";
import { keylessBaseEligible } from "./keyless-example-eligibility";
import {
  readUseTunnelPref,
  subscribeUseTunnelPref,
  writeUseTunnelPref,
} from "./tunnel-preference";

type ExampleType =
  | "curl"
  | "python"
  | "javascript"
  | "curlTools"
  | "pythonTools"
  | "javascriptTools"
  | "curlAdvanced"
  | "pythonAdvanced"
  | "javascriptAdvanced";
type Os = ExampleOs;
type Variant = "plain" | "tools" | "advanced";
type Language = "curl" | "python" | "javascript";

const LANGUAGES: { id: Language; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
];

const VARIANTS: { id: Variant; labelKey: TranslationKey }[] = [
  { id: "plain", labelKey: "settings.apiKeys.requestPlain" },
  { id: "tools", labelKey: "settings.apiKeys.requestTools" },
  { id: "advanced", labelKey: "settings.apiKeys.requestAdvanced" },
];

// guards a restored pick against a snippet type dropped in a later release.
const EXAMPLE_TYPE_IDS = new Set<string>([
  "curl",
  "python",
  "javascript",
  "curlTools",
  "pythonTools",
  "javascriptTools",
  "curlAdvanced",
  "pythonAdvanced",
  "javascriptAdvanced",
] satisfies ExampleType[]);

function exampleType(language: Language, variant: Variant): ExampleType {
  if (variant === "plain") {
    return language;
  }
  return `${language}${variant === "tools" ? "Tools" : "Advanced"}`;
}

function exampleParts(type: ExampleType): [Language, Variant] {
  const language: Language = type.startsWith("curl")
    ? "curl"
    : type.startsWith("python")
      ? "python"
      : "javascript";
  const variant: Variant = type.endsWith("Tools")
    ? "tools"
    : type.endsWith("Advanced")
      ? "advanced"
      : "plain";
  return [language, variant];
}

const CURL_TYPES = new Set<ExampleType>(["curl", "curlTools", "curlAdvanced"]);
const JAVASCRIPT_TYPES = new Set<ExampleType>([
  "javascript",
  "javascriptTools",
  "javascriptAdvanced",
]);

const PROMPT = "What is Unsloth?";
const TOOLS = ["web_search", "python", "terminal"];
const ADV = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.05,
  repetition_penalty: 1.1,
  max_tokens: 1024,
} as const;

// Fallback until the backend's installed-CLI check resolves. Mirrors CODING_AGENTS in
// studio/backend/utils/coding_agents.py.
const DEFAULT_AGENTS = [
  "claude",
  "codex",
  "vscode",
  "dsh",
  "opencode",
  "hermes",
  "openclaw",
  "pi",
  "vibe",
];
const DEFAULT_AGENT = "claude";

const j = (s: string): string => JSON.stringify(s);
const toolsJson = TOOLS.map(j).join(", ");

function bodyExtraLines(variant: Variant, indent: string): string[] {
  const lines: string[] = [];
  if (variant === "advanced") {
    lines.push(`${indent}"temperature": ${ADV.temperature},`);
    lines.push(`${indent}"top_p": ${ADV.top_p},`);
    lines.push(`${indent}"top_k": ${ADV.top_k},`);
    lines.push(`${indent}"min_p": ${ADV.min_p},`);
    lines.push(`${indent}"repetition_penalty": ${ADV.repetition_penalty},`);
    lines.push(`${indent}"max_tokens": ${ADV.max_tokens},`);
    lines.push(`${indent}"enable_thinking": true,`);
  }
  if (variant !== "plain") {
    lines.push(`${indent}"enable_tools": true,`);
    lines.push(`${indent}"enabled_tools": [${toolsJson}],`);
    // The permission gate only asks over X-Unsloth-Events frames these snippets do not take.
    lines.push(`${indent}"permission_mode": "off",`);
  }
  return lines;
}

function curlBodyPretty(model: string, variant: Variant): string {
  const lines = [
    `    "model": ${j(model)},`,
    `    "messages": [{"role": "user", "content": ${j(PROMPT)}}],`,
    ...bodyExtraLines(variant, "    "),
    `    "stream": true`,
  ];
  return `{\n${lines.join("\n")}\n  }`;
}

function winBody(model: string, variant: Variant): string {
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: "user", content: PROMPT }],
  };
  if (variant === "advanced") {
    body.temperature = ADV.temperature;
    body.top_p = ADV.top_p;
    body.top_k = ADV.top_k;
    body.min_p = ADV.min_p;
    body.repetition_penalty = ADV.repetition_penalty;
    body.max_tokens = ADV.max_tokens;
    body.enable_thinking = true;
  }
  if (variant !== "plain") {
    body.enable_tools = true;
    body.enabled_tools = TOOLS;
    body.permission_mode = "off";
  }
  body.stream = true;
  return JSON.stringify(body, null, 2);
}

function curlUnix(
  base: string,
  key: string,
  model: string,
  variant: Variant,
): string {
  return `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(curlBodyPretty(model, variant))}'`;
}

function curlWindows(
  base: string,
  key: string,
  model: string,
  variant: Variant,
): string {
  return `$body = '${psSingle(winBody(model, variant))}'
Set-Content -Path body.json -Value $body -Encoding ascii
curl.exe ${base}/v1/chat/completions \`
  -H "Authorization: Bearer ${key}" \`
  -H "Content-Type: application/json" \`
  -d "@body.json"`;
}

function pythonSnippet(
  base: string,
  key: string,
  model: string,
  variant: Variant,
): string {
  const named =
    variant === "advanced"
      ? `
    temperature=${ADV.temperature},
    top_p=${ADV.top_p},
    max_tokens=${ADV.max_tokens},`
      : "";
  const extra: string[] = [];
  if (variant === "advanced") {
    extra.push(`        "top_k": ${ADV.top_k},`);
    extra.push(`        "min_p": ${ADV.min_p},`);
    extra.push(`        "repetition_penalty": ${ADV.repetition_penalty},`);
    extra.push(`        "enable_thinking": True,`);
  }
  if (variant !== "plain") {
    extra.push(`        "enable_tools": True,`);
    extra.push(`        "enabled_tools": [${toolsJson}],`);
    // biome-ignore lint/style/noUnusedTemplateLiteral: keep generated options visually uniform
    extra.push(`        "permission_mode": "off",`);
  }
  const extraBody = extra.length
    ? `
    extra_body={
${extra.join("\n")}
    },`
    : "";
  const loop =
    variant !== "plain"
      ? `for chunk in response:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")`
      : `for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")`;
  return `from openai import OpenAI

client = OpenAI(
    base_url=${j(`${base}/v1`)},
    api_key=${j(key)},
)

response = client.chat.completions.create(
    model=${j(model)},
    messages=[{"role": "user", "content": ${j(PROMPT)}}],${named}${extraBody}
    stream=True,
)
${loop}`;
}

function javascriptSnippet(
  base: string,
  key: string,
  model: string,
  variant: Variant,
): string {
  const options: string[] = [];
  if (variant === "advanced") {
    options.push(`  temperature: ${ADV.temperature},`);
    options.push(`  top_p: ${ADV.top_p},`);
    options.push(`  max_tokens: ${ADV.max_tokens},`);
  }

  // The JS SDK forwards unknown options into the body (Python needs extra_body).
  if (variant === "advanced") {
    options.push(`  top_k: ${ADV.top_k},`);
    options.push(`  min_p: ${ADV.min_p},`);
    options.push(`  repetition_penalty: ${ADV.repetition_penalty},`);
    // biome-ignore lint/style/noUnusedTemplateLiteral: keep generated options visually uniform
    options.push(`  enable_thinking: true,`);
  }
  if (variant !== "plain") {
    // biome-ignore lint/style/noUnusedTemplateLiteral: keep generated options visually uniform
    options.push(`  enable_tools: true,`);
    options.push(`  enabled_tools: [${toolsJson}],`);
    // biome-ignore lint/style/noUnusedTemplateLiteral: keep generated options visually uniform
    options.push(`  permission_mode: "off",`);
  }

  const trailingOptions = options.length ? `\n${options.join("\n")}` : "";

  return `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: ${j(`${base}/v1`)},
  apiKey: ${j(key)},
});

const response = await client.chat.completions.create({
  model: ${j(model)},
  messages: [{ role: "user", content: ${j(PROMPT)} }],${trailingOptions}
  stream: true,
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices?.[0]?.delta?.content || "");
}`;
}

function buildSnippets(
  base: string,
  key: string,
  toolsKey: string,
  model: string,
  os: Os,
): Record<ExampleType, string> {
  const curl = os === "windows" ? curlWindows : curlUnix;
  return {
    curl: curl(base, key, model, "plain"),
    python: pythonSnippet(base, key, model, "plain"),
    javascript: javascriptSnippet(base, key, model, "plain"),
    curlTools: curl(base, toolsKey, model, "tools"),
    pythonTools: pythonSnippet(base, toolsKey, model, "tools"),
    javascriptTools: javascriptSnippet(base, toolsKey, model, "tools"),
    curlAdvanced: curl(base, toolsKey, model, "advanced"),
    pythonAdvanced: pythonSnippet(base, toolsKey, model, "advanced"),
    javascriptAdvanced: javascriptSnippet(base, toolsKey, model, "advanced"),
  };
}

const KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";
const MODEL_PLACEHOLDER = "YOUR_MODEL";
// OpenAI SDKs require an api_key value even for keyless access.
const KEYLESS_KEY_PLACEHOLDER = "not-needed";
// retry while /v1 has no model because downloads and loads do not update the store.
const CATALOG_RETRY_MS = 15000;
// keep polling after a model appears because idle unloads do not update the store.
const CATALOG_IDLE_MS = 60000;

/** The Secure HTTPS switch, shared by every example card on the tab. */
export function useApiTunnelPref(): [boolean, (next: boolean) => void] {
  const useTunnel = useSyncExternalStore(
    subscribeUseTunnelPref,
    readUseTunnelPref,
    () => true,
  );
  return [useTunnel, writeUseTunnelPref];
}

// match the backend path heuristic because /v1 never advertises on-disk checkpoints.
function looksLikePath(id: string): boolean {
  return (
    id.startsWith("/") ||
    id.startsWith("~") ||
    id.startsWith(".") ||
    id.includes("\\") ||
    id.toLowerCase().endsWith(".gguf") ||
    (id.match(/\//g)?.length ?? 0) >= 2
  );
}

// return only model ids that /v1 can resolve, or null when none are available.
function useExampleModelName(keylessOnly: boolean): string | null {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  // null until /v1/models answers: "not asked" must not read as "holds nothing".
  const [catalog, setCatalog] = useState<OpenAIModel[] | null>(null);
  const [autoSwitch, setAutoSwitch] = useState(false);
  const usableCheckpoint =
    !!checkpoint &&
    !checkpoint.startsWith("external::") &&
    !looksLikePath(checkpoint);

  // biome-ignore lint/correctness/useExhaustiveDependencies: a load or unload must refetch the servable ids
  useEffect(() => {
    let cancelled = false;
    let timeoutId: number | null = null;

    const update = () => {
      // null on failure: a transient error is no evidence the server holds nothing.
      void Promise.all([
        listOpenAIModels().catch(() => null),
        loadOpenAIAutoSwitchSettings()
          .then((s) => s.enabled)
          .catch(() => null),
      ])
        .then(([models, settings]) => {
          if (cancelled) return true;
          if (models !== null) setCatalog(models);
          if (settings !== null) {
            setAutoSwitch(settings);
          }
          // biome-ignore lint/complexity/useOptionalChain: keep the explicit failed-refresh branch
          return models !== null && models.some((m) => m.loaded);
        })
        .then((resolved) => {
          if (cancelled) return;
          timeoutId = window.setTimeout(
            update,
            resolved ? CATALOG_IDLE_MS : CATALOG_RETRY_MS,
          );
        });
    };

    update();
    return () => {
      cancelled = true;
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    };
  }, [checkpoint, ggufVariant]);

  return useMemo(() => {
    const fromCatalog = (): string | null => {
      const pick =
        catalog?.find((m) => m.loaded) ??
        (!keylessOnly && autoSwitch ? catalog?.[0] : undefined);
      if (!pick) {
        return null;
      }
      return pick.quant && !pick.id.includes(":")
        ? `${pick.id}:${pick.quant}`
        : pick.id;
    };
    // The store keeps a checkpoint across unload and delete, so require the catalog to list it.
    const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));
    const backed =
      (!keylessOnly && catalog === null) ||
      (!!entry && (entry.loaded || (!keylessOnly && autoSwitch)));
    if (usableCheckpoint && checkpoint && backed) {
      if (checkpoint.includes(":")) {
        return checkpoint;
      }
      // Pin the catalog's quant: the saved one may name a deleted file.
      const quant = catalog === null ? ggufVariant : entry?.quant;
      return quant ? `${checkpoint}:${quant}` : checkpoint;
    }
    return fromCatalog();
  }, [
    autoSwitch,
    catalog,
    checkpoint,
    ggufVariant,
    keylessOnly,
    usableCheckpoint,
  ]);
}

// Only safe in the desktop app; a browser loopback URL may be a port forward.
function canUseLocalAgentDetection(base: string): boolean {
  if (!isTauri) return false;
  try {
    return isLoopbackHost(normalizeHost(new URL(base).hostname));
  } catch {
    return false;
  }
}

export function UsageExamples({
  apiKey,
  useTunnel,
  onUseTunnelChange,
  keylessScope = "off",
  keylessTools = false,
  keylessExposure = null,
  agentsExtra = null,
  agentsOnly = false,
  part = "all",
}: {
  apiKey?: string | null;
  agentsOnly?: boolean;
  part?: "all" | "chat";
  agentsExtra?: ReactNode;
  useTunnel: boolean;
  onUseTunnelChange: (next: boolean) => void;
  /** which routes keyless api access serves, so a placeholder is only used where it works */
  keylessScope?: KeylessApiAccessScope;
  keylessTools?: boolean;
  /** public tunnels and Colab never accept the dummy bearer */
  keylessExposure?: KeylessApiAccessExposure | null;
}) {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  // Refs, not ids: the API page can mount a second copy, and an id lookup found that one.
  const agentsRef = useRef<HTMLDivElement | null>(null);
  const mcpRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (agentsOnly || part !== "all") return;
    if (scrollTarget !== "api-keys-agents" && scrollTarget !== "api-keys-mcp") return;
    const target = scrollTarget;
    const section = (target === "api-keys-agents" ? agentsRef : mcpRef).current;
    if (!section) return;
    const pin = () => scrollIntoPane(section);
    const observer = new ResizeObserver(pin);
    observer.observe(section.parentElement ?? section);
    const release = () => {
      observer.disconnect();
      useSettingsDialogStore.getState().consumeScrollTarget(target);
    };
    const timer = window.setTimeout(release, 1500);
    const events = ["wheel", "touchmove", "keydown"] as const;
    for (const name of events) {
      window.addEventListener(name, release, { passive: true });
    }
    return () => {
      observer.disconnect();
      window.clearTimeout(timer);
      for (const name of events) window.removeEventListener(name, release);
    };
  }, [scrollTarget, agentsOnly, part]);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const secure = usePlatformStore((s) => s.secure);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const setStoredAgent = useSettingsPanelPrefsStore(
    (s) => s.setApiExampleAgent,
  );
  const setAgentsTabAgent = useSettingsPanelPrefsStore((s) => s.setAgentsAgent);
  // read once: these seed the controls, which write back through the handlers.
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [lang, setLang] = useState<ExampleType>(
    storedPrefs.apiExampleLang &&
      EXAMPLE_TYPE_IDS.has(storedPrefs.apiExampleLang)
      ? (storedPrefs.apiExampleLang as ExampleType)
      : "curl",
  );
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [copiedAgent, setCopiedAgent] = useState(false);
  const [agent, setAgent] = useState<string>(
    storedPrefs.apiExampleAgent ?? DEFAULT_AGENT,
  );
  const [availableAgents, setAvailableAgents] =
    useState<string[]>(DEFAULT_AGENTS);
  const [detectedAgents, setDetectedAgents] = useState<string[]>([]);
  // set on answer, so a restored agent never validates against the defaults.
  const [agentsLoaded, setAgentsLoaded] = useState(false);
  // Guards the detection effect from clobbering a pick the user made first.
  const agentPickedByUserRef = useRef(storedPrefs.apiExampleAgent != null);
  // A manual pick is kept only until the model's GGUF-ness changes under it.
  const clickedUnderGgufRef = useRef<boolean | null>(null);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  // server_url is the bind address, which an SSH forward or reverse proxy does not expose.
  const base =
    useTunnel && cloudflareUrl
      ? cloudflareUrl
      : isTauri
        ? (serverUrl ?? origin)
        : origin;
  const [os, setOs] = useExampleOs(base);
  const localAgentDetection = canUseLocalAgentDetection(base);
  const loopbackBind = isLoopbackUrl(serverUrl);
  // isServedByLlamaCpp owns which store fields count; a context length is not among them,
  // since MLX reports one too.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const loadedIsGguf = useChatRuntimeStore((s) => s.loadedIsGguf);
  // null when these fields do not describe the model the snippet names (before status, or under an
  // external selection).
  const activeCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const storeIsGguf: boolean | null =
    !activeCheckpoint || isExternalModelId(activeCheckpoint)
      ? null
      : isServedByLlamaCpp({
          loadedIsGguf,
          activeGgufVariant,
          activeNativePathToken,
          checkpoint: activeCheckpoint,
        });

  // Off the chat and hub pages the store never answers, so the server must; JSON because each part
  // can be a path or a repo id.
  const storeModelKey = JSON.stringify([
    activeCheckpoint,
    activeGgufVariant,
    activeNativePathToken,
  ]);
  // Above the derivation that checks the verdict against this model.
  const keylessBase =
    !(useTunnel && cloudflareUrl) &&
    keylessBaseEligible(base, keylessScope, keylessExposure);
  // Only the inference scope pins a keyless caller to the loaded model; full can switch.
  const model = useExampleModelName(
    keylessBase && !apiKey && keylessScope === "inference",
  );

  const [statusAnswer, setStatusAnswer] = useState<{
    key: string;
    resident: string | null;
    isGguf: boolean | null;
  } | null>(null);
  // useChatModelRuntime never polls, so only this notices a swap while the tab stays focused.
  useEffect(() => {
    let cancelled = false;
    let timeoutId: number | null = null;

    const update = () => {
      void getInferenceStatus()
        .then((status) => {
          if (cancelled) return false;
          // Both unknown, as the CLI gate reads them: no field, or is_gguf False with no model.
          const resident =
            status.active_model ?? status.model_identifier ?? null;
          const answer = statusGgufVerdict(resident, status.is_gguf);
          setStatusAnswer({ key: storeModelKey, resident, isGguf: answer });
          return answer !== null;
        })
        .catch(() => {
          // Keep the last answer: a failed probe is no evidence about the model.
          return false;
        })
        .then((resolved) => {
          if (cancelled) return;
          timeoutId = window.setTimeout(
            update,
            resolved ? CATALOG_IDLE_MS : CATALOG_RETRY_MS,
          );
        });
    };

    update();
    return () => {
      cancelled = true;
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    };
  }, [storeModelKey]);

  // status reports active_model_name raw while /v1/models uses public_model_id(), so collapse it
  // the same way or a path-loaded model never matches.
  const isGguf: boolean | null = compatibilityFromSources(
    storeIsGguf,
    statusAnswer !== null && statusAnswer.key === storeModelKey
      ? {
          isGguf: statusAnswer.isGguf,
          resident:
            statusAnswer.resident === null
              ? null
              : publicModelId(statusAnswer.resident),
        }
      : null,
    model,
  );

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // Fetch only; the auto-pick is derived below so it also reacts to model changes.
  useEffect(() => {
    // Browser loopback URLs can be port forwards, so only the desktop app uses PATH checks.
    if (!localAgentDetection) {
      setDetectedAgents([]);
      // An auto-pick was only verified against the backend PATH; never touch a manual pick.
      if (!agentPickedByUserRef.current) {
        // The effect below corrects this; isGguf here would be a stale closure.
        setAgent(DEFAULT_AGENT);
      }
      return;
    }

    let cancelled = false;
    void loadCodingAgents()
      .then((info) => {
        if (cancelled) return;
        setAvailableAgents(info.agents);
        setDetectedAgents(info.detected);
      })
      .catch(() => {
        // Best-effort: keep the default agent list and let the user pick manually.
      })
      .finally(() => {
        if (!cancelled) setAgentsLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, [localAgentDetection]);

  // Drop a restored preference this build no longer offers or that cannot run the model.
  useEffect(() => {
    if (!agentPickedByUserRef.current) return;
    if (isGguf === null) return;
    if (clickedUnderGgufRef.current === isGguf) return;
    if (localAgentDetection && !agentsLoaded) return;
    if (
      availableAgents.includes(agent) &&
      agentRunsOnActiveModel(agent, isGguf)
    ) {
      return;
    }
    const reset = fallbackAgent(isGguf, availableAgents);
    if (reset === null) return;
    agentPickedByUserRef.current = false;
    setStoredAgent(null);
    setAgent(reset);
  }, [
    agent,
    agentsLoaded,
    availableAgents,
    isGguf,
    localAgentDetection,
    setStoredAgent,
  ]);

  // Gate on detection having answered: an empty list is a valid answer.
  useEffect(() => {
    if (agentPickedByUserRef.current) return;
    if (isGguf === null) return;
    if (localAgentDetection && !agentsLoaded) return;
    const next = pickCompatibleAgent(
      detectedAgents,
      agent,
      isGguf,
      availableAgents,
    );
    if (next !== null) {
      setAgent(next);
    }
  }, [
    agent,
    agentsLoaded,
    availableAgents,
    detectedAgents,
    isGguf,
    localAgentDetection,
  ]);

  // The SDK dummy key is printed only for a transport the backend can admit.
  const key =
    apiKey || (keylessBase ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);
  // a keyless caller gets no tools until the admin grants them
  const toolsKey =
    apiKey ||
    (keylessBase && keylessTools ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);
  const agentKey =
    apiKey || (keylessBase ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);

  // Null model: nothing is servable yet, so the example names a placeholder (and the note
  // above it says so) rather than vanishing, which left the pickers looking dead.
  const snippets = useMemo(
    () => buildSnippets(base, key, toolsKey, model ?? MODEL_PLACEHOLDER, os),
    [base, key, toolsKey, model, os],
  );
  // Target the server the panel shows, not the :8888 default.
  const agentCommand = useMemo(
    () => buildAgentCommand(base, agentKey, os, agent),
    [base, agentKey, os, agent],
  );

  const shikiLang = CURL_TYPES.has(lang)
    ? os === "windows"
      ? "powershell"
      : "bash"
    : JAVASCRIPT_TYPES.has(lang)
      ? "javascript"
      : "python";

  const handleCopyUrl = async () => {
    if (cloudflareUrl && (await copyToClipboard(cloudflareUrl))) {
      setCopiedUrl(true);
      setTimeout(() => setCopiedUrl(false), 1800);
    }
  };

  const handleCopyAgent = async () => {
    if (await copyToClipboard(agentCommand)) {
      setCopiedAgent(true);
      setTimeout(() => setCopiedAgent(false), 1800);
    }
  };

  const [language, variant] = exampleParts(lang);
  const pickExample = (next: ExampleType) => {
    setLang(next);
    setStoredLang(next);
  };
  const agentDetails = detailsFor(agent);

  return (
    <section className="flex min-w-0 max-w-full flex-col gap-4">
      {agentsOnly ? null : (
        <>
          {part === "all" ? (
            <h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">
              <HugeiconsIcon icon={MessageCircleIcon} strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground" />
              {t("settings.apiKeys.usageExamples")}
            </h2>
          ) : null}
          {/* No model-auto-switch row: ModelAutoSwitchSection renders that setting just below. */}
          {cloudflareUrl ? (
            <div className="flex min-w-0 items-center justify-between gap-2 rounded-full border border-border px-4 py-2">
              <div className="flex shrink-0 items-center gap-1.5">
                <Switch
                  size="sm"
                  checked={useTunnel}
                  onCheckedChange={onUseTunnelChange}
                  aria-label={t("settings.apiKeys.secureHttps")}
                />
                <span className="text-ui-11 font-medium text-foreground">
                  {t("settings.apiKeys.secureHttps")}
                  {/* Only for a network bind not launched with --secure: its raw port stays
                      reachable. A loopback bind (the default, and Desktop) exposes nothing. */}
                  {secure || loopbackBind ? null : (
                    <Tooltip>
                      <TooltipTrigger asChild={true}>
                        {/* Inline on the baseline, like a glyph after the label. */}
                        <button
                          type="button"
                          className="ml-1.5 inline-flex align-baseline rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          aria-label={t("settings.apiKeys.secureHttpsHint")}
                        >
                          {/* Follows the UI font size, like the SettingsRow hint
                              this matches. */}
                          <HugeiconsIcon
                            icon={InformationCircleIcon}
                            className="size-[var(--ui-icon-size-hint)]"
                          />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[calc(260px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
                        {t("settings.apiKeys.secureHttpsHint")}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </span>
              </div>
              <button
                type="button"
                onClick={handleCopyUrl}
                className={cn(
                  "flex min-w-0 items-center gap-1 rounded-full px-1.5 py-1 text-ui-11 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  !useTunnel && "opacity-50",
                )}
                title={cloudflareUrl}
                aria-label={t("settings.apiKeys.copyTunnelUrl")}
              >
                <span className="truncate font-mono">{cloudflareUrl}</span>
                <HugeiconsIcon
                  icon={copiedUrl ? Tick02Icon : Copy01Icon}
                  className={cn(
                    "size-3.5 shrink-0",
                    copiedUrl && "text-emerald-600",
                  )}
                />
              </button>
            </div>
          ) : null}
          {model ? null : (
            <p className="text-ui-11 leading-relaxed text-muted-foreground">
              {t("settings.apiKeys.usageNoModel")}
            </p>
          )}
          <ExampleToggle
            label={t("settings.apiKeys.exampleRequest")}
            variant="toggle"
            value={variant}
            options={VARIANTS.map((option) => ({
              id: option.id,
              label: t(option.labelKey),
            }))}
            onChange={(next) => pickExample(exampleType(language, next))}
          />
          <ExampleCodeBox
            code={snippets[lang]}
            language={shikiLang}
            label={t("settings.apiKeys.usageExamples")}
            redactFromReload={Boolean(apiKey)}
            pills={
              <ExampleToggle
                label={t("settings.apiKeys.exampleLanguage")}
                value={language}
                options={LANGUAGES}
                onChange={(next) => pickExample(exampleType(next, variant))}
              />
            }
            os={os}
            onOsChange={setOs}
          />
        </>
      )}
      {part === "chat" ? null : (
        <div ref={agentsRef} className="flex min-w-0 scroll-mt-6 flex-col gap-2.5 pt-2">
          <div className="flex items-center justify-between gap-3">
            {agentsOnly ? (
              <span />
            ) : (
              <h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">
                <HugeiconsIcon icon={BotIcon} strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground" />
                {t("settings.tabs.agents")}
              </h2>
            )}
            {/* Agents is an owner-only tab, so others get the docs. The tab reads this pref on mount. */}
            {isOwner ? (
              <button
                type="button"
                onClick={() => {
                  setAgentsTabAgent(agent);
                  if (agentsOnly) useSettingsDialogStore.getState().openDialog("agents");
                  else useSettingsDialogStore.getState().setActiveTab("agents");
                }}
                className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                {t("settings.apiKeys.openAgentsTab")}
                <HugeiconsIcon icon={ChevronRightStandardIcon} className="size-3" />
              </button>
            ) : (
              <a
                href={agentDetails.docsUrl}
                target="_blank"
                rel="noreferrer"
                aria-label={t("settings.agents.agentDocs", {
                  agent: agentDetails.name,
                })}
                className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                {t("settings.agents.docs")}
                <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
              </a>
            )}
          </div>
          <p className="text-ui-11 leading-relaxed text-muted-foreground">
            {t("settings.apiKeys.codingAgentsHint")}
          </p>
          <div className="@container">
            <div className="grid grid-cols-1 items-start gap-3 @[34rem]:grid-cols-3">
              <div className="min-w-0">
                <Select
                  value={agent}
                  onValueChange={(id) => {
                    agentPickedByUserRef.current = true;
                    clickedUnderGgufRef.current = isGguf;
                    setAgent(id);
                    setStoredAgent(id);
                  }}
                >
                  <SelectTrigger
                    aria-label={t("settings.agents.agent")}
                    className="w-full rounded-full"
                  >
                    <SelectValue>
                      <span className="flex min-w-0 items-center gap-2">
                        <AgentIcon {...agentDetails} />
                        <span className="truncate">{agentDetails.name}</span>
                      </span>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent
                    align="start"
                    className="max-h-[min(var(--radix-select-content-available-height),calc(7*2.25rem+0.5rem))] [&_[data-slot=select-scroll-down-button]]:hidden [&_[data-slot=select-scroll-up-button]]:hidden"
                  >
                    {availableAgents.map((id) => {
                      const details = detailsFor(id);
                      return (
                        <SelectItem key={id} value={id}>
                          <span className="flex min-w-0 items-center gap-2">
                            <AgentIcon {...details} />
                            <span className="truncate">{details.name}</span>
                            {detectedAgents.includes(id) ? (
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
            </div>
          </div>
          <div className="relative min-w-0">
            <code
              className="block min-w-0 whitespace-pre-wrap break-all rounded-3xl border border-border bg-background/70 py-2.5 pr-10 pl-4 font-mono text-ui-11 leading-relaxed text-foreground dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]"
              data-reload-snapshot-sensitive={apiKey ? "" : undefined}
            >
              {agentCommand}
            </code>
            <button
              type="button"
              onClick={handleCopyAgent}
              className="absolute top-1.5 right-2 flex size-6 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              aria-label={t("settings.apiKeys.copySnippet")}
            >
              <HugeiconsIcon
                icon={copiedAgent ? Tick02Icon : Copy01Icon}
                className={cn("size-3.5", copiedAgent && "text-control-accent")}
              />
            </button>
          </div>
          {agent === "vscode" && (
            <p className="text-ui-11 leading-relaxed text-muted-foreground">
              {t("settings.agents.vscodeNote")}
            </p>
          )}
          {agentsExtra ? <div ref={mcpRef}>{agentsExtra}</div> : null}
        </div>
      )}
    </section>
  );
}
