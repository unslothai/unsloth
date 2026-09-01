// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useChatRuntimeStore } from "@/features/chat";
import { isServedByLlamaCpp } from "@/features/model-picker";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowUpRight01Icon,
  Copy01Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Streamdown } from "streamdown";
import { loadCodingAgents } from "../api/coding-agents";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { loadOpenAIAutoSwitchSettings } from "../api/openai-auto-switch";
import { type OpenAIModel, listOpenAIModels } from "../api/openai-models";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import {
  buildAgentCommand,
  isLoopbackHost,
  normalizeHost,
  psSingle,
  shSingle,
} from "./agent-command";
import { keylessBaseEligible } from "./keyless-example-eligibility";

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
type Os = "unix" | "windows";
type Variant = "plain" | "tools" | "advanced";

const TYPE_TABS: { id: ExampleType; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
  { id: "curlTools", label: "curl + tools" },
  { id: "pythonTools", label: "Python + tools" },
  { id: "javascriptTools", label: "JavaScript + tools" },
  { id: "curlAdvanced", label: "curl + advanced" },
  { id: "pythonAdvanced", label: "Python + advanced" },
  { id: "javascriptAdvanced", label: "JavaScript + advanced" },
];

// guards a restored tab against a snippet type dropped in a later release.
const EXAMPLE_TYPE_IDS = new Set<string>(TYPE_TABS.map((tab) => tab.id));

const TYPE_LABEL_KEY: Partial<Record<ExampleType, TranslationKey>> = {
  curlTools: "settings.apiKeys.exampleCurlTools",
  pythonTools: "settings.apiKeys.examplePythonTools",
  javascriptTools: "settings.apiKeys.exampleJavaScriptTools",
  curlAdvanced: "settings.apiKeys.exampleCurlAdvanced",
  pythonAdvanced: "settings.apiKeys.examplePythonAdvanced",
  javascriptAdvanced: "settings.apiKeys.exampleJavaScriptAdvanced",
};

const OS_AWARE: Record<ExampleType, boolean> = {
  curl: true,
  python: false,
  javascript: false,
  curlTools: true,
  pythonTools: false,
  javascriptTools: false,
  curlAdvanced: true,
  pythonAdvanced: false,
  javascriptAdvanced: false,
};

const CURL_TYPES = new Set<ExampleType>(["curl", "curlTools", "curlAdvanced"]);
const JAVASCRIPT_TYPES = new Set<ExampleType>([
  "javascript",
  "javascriptTools",
  "javascriptAdvanced",
]);

const PROMPT = "What is Unsloth?";
// web_search + python + terminal are the reliable built-in tools.
const TOOLS = ["web_search", "python", "terminal"];
const ADV = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.05,
  repetition_penalty: 1.1,
  max_tokens: 1024,
} as const;

const DOC_LINKS = [
  { label: "Claude Code", href: "https://unsloth.ai/docs/basics/claude-code" },
  { label: "Codex", href: "https://unsloth.ai/docs/basics/codex" },
  { label: "OpenClaw", href: "https://unsloth.ai/docs/integrations/openclaw" },
  { label: "OpenCode", href: "https://unsloth.ai/docs/integrations/opencode" },
  {
    label: "Hermes Agent",
    href: "https://unsloth.ai/docs/integrations/hermes-agent",
  },
];

// Fallback until the backend's installed-CLI check resolves. Mirrors
// CODING_AGENTS in studio/backend/utils/coding_agents.py, minus HIDDEN_AGENTS
// (see ../api/coding-agents.ts).
const DEFAULT_AGENTS = ["claude", "codex", "openclaw", "opencode", "hermes"];
// The agent selection resets to this whenever an auto-pick is no longer
// trustworthy (leaving loopback, or the only compatible detected agent
// stops being compatible) rather than lingering on a stale choice.
const DEFAULT_AGENT = "claude";
const AGENT_LABELS: Record<string, string> = {
  claude: "Claude Code",
  codex: "Codex",
  openclaw: "OpenClaw",
  opencode: "OpenCode",
  hermes: "Hermes",
};

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

  // The JS SDK forwards unknown options into the request body, so these go at the
  // top level (the Python SDK needs them under extra_body instead).
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

// every variant but "plain" asks for the server-side tools, so it needs its own key
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
// the openai sdks require some api_key, so name one rather than leave it blank
const KEYLESS_KEY_PLACEHOLDER = "not-needed";
const USE_TUNNEL_KEY = "unsloth_api_use_tunnel";
// Slow retry while /v1 has nothing to name: a download or load moves no store state.
const CATALOG_RETRY_MS = 15000;
// Slower beat once something is servable: an idle unload frees a model without
// touching the store, so residency is never settled for good.
const CATALOG_IDLE_MS = 60000;

function readUseTunnelPref(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return window.localStorage.getItem(USE_TUNNEL_KEY) !== "false";
  } catch {
    return true;
  }
}

function writeUseTunnelPref(value: boolean): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(USE_TUNNEL_KEY, value ? "true" : "false");
  } catch {
    // Non-fatal
  }
}

// A checkpoint can be an on-disk load path, which /v1 never advertises. Mirrors _looks_like_path.
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

// Same model, ignoring any ":quant" a caller pinned.
function sameBaseModelId(a: string, b: string): boolean {
  const base = (id: string) => id.trim().toLowerCase().split(":")[0];
  return (
    a.trim().toLowerCase() === b.trim().toLowerCase() || base(a) === base(b)
  );
}

// The model the examples name: always an id /v1 resolves against, null when there is none.
function useExampleModelName(keylessOnly: boolean): string | null {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  // null until /v1/models answers: "not asked yet" must not read as "holds nothing".
  const [catalog, setCatalog] = useState<OpenAIModel[] | null>(null);
  // A downloaded but unloaded model is only runnable when switching is on.
  const [autoSwitch, setAutoSwitch] = useState(false);
  const usableCheckpoint =
    !!checkpoint &&
    !checkpoint.startsWith("external::") &&
    !looksLikePath(checkpoint);

  // Always: a stored checkpoint can stop being servable without the store changing.
  // biome-ignore lint/correctness/useExhaustiveDependencies: a load or unload must refetch the servable ids
  useEffect(() => {
    let cancelled = false;
    let timeoutId: number | null = null;

    const update = () => {
      // null on failure, never [] or false: a transient error is no evidence that the
      // server holds nothing, and those negatives blanked every example while the
      // model was still servable. Keep the last answer and retry.
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
          // Resident only slows the polling; it never stops it.
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
    // Name something held here, with its quant to pin the file on disk.
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
    // The store keeps a checkpoint across an idle unload and across the model being
    // deleted, so it only names a runnable model while the catalog still lists it:
    // resident, or downloaded with switching able to reload it. A null catalog means
    // /v1/models has not answered, which is not evidence against it.
    const entry = catalog?.find((m) => sameBaseModelId(m.id, checkpoint ?? ""));
    const backed =
      (!keylessOnly && catalog === null) ||
      (!!entry && (entry.loaded || (!keylessOnly && autoSwitch)));
    if (usableCheckpoint && checkpoint && backed) {
      if (checkpoint.includes(":")) {
        return checkpoint;
      }
      // Pin the quant the catalog advertises, not the stored one: membership proves the
      // repo, and the saved quant can name a file deleted while another quant remains.
      // Fall back to the store only before /v1/models answers.
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

// Backend PATH detection is only safe in the desktop app, where the UI owns
// the local backend. A browser loopback URL may be an SSH/local port forward.
function canUseLocalAgentDetection(base: string): boolean {
  if (!isTauri) return false;
  try {
    return isLoopbackHost(normalizeHost(new URL(base).hostname));
  } catch {
    return false;
  }
}

const SHIKI_THEMES = [unslothLightTheme, unslothDarkTheme] as [
  typeof unslothLightTheme,
  typeof unslothDarkTheme,
];
const codePlugin = createCodePlugin({ themes: SHIKI_THEMES });

function HighlightedCode({
  code,
  language,
  redactFromReload,
}: {
  code: string;
  language: string;
  redactFromReload: boolean;
}) {
  const markdown = useMemo(
    () => `\`\`\`${language}\n${code}\n\`\`\``,
    [code, language],
  );
  return (
    <div
      className="max-w-full overflow-x-auto p-3 pr-16 text-ui-11 leading-relaxed [&_pre]:!m-0 [&_pre]:!whitespace-pre-wrap [&_pre]:!break-words [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-ui-11 [&_pre]:!leading-relaxed [&_code]:!text-ui-11 [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block]]:!text-ui-11"
      data-reload-snapshot-sensitive={redactFromReload ? "" : undefined}
    >
      <Streamdown
        mode="static"
        plugins={{ code: codePlugin }}
        controls={{ code: false }}
        shikiTheme={SHIKI_THEMES}
      >
        {markdown}
      </Streamdown>
    </div>
  );
}

export function UsageExamples({
  apiKey,
  keylessScope = "off",
  keylessTools = false,
  keylessExposure = null,
}: {
  apiKey?: string | null;
  /** which routes keyless api access serves, so a placeholder is only used where it works */
  keylessScope?: KeylessApiAccessScope;
  /** whether a keyless caller may drive the server-side tool loop */
  keylessTools?: boolean;
  /** public tunnels and Colab never accept the dummy bearer */
  keylessExposure?: KeylessApiAccessExposure | null;
}) {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const secure = usePlatformStore((s) => s.secure);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setApiExampleOs);
  const setStoredAgent = useSettingsPanelPrefsStore(
    (s) => s.setApiExampleAgent,
  );
  // read once: these seed the controls, which write back through the handlers.
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [lang, setLang] = useState<ExampleType>(
    storedPrefs.apiExampleLang &&
      EXAMPLE_TYPE_IDS.has(storedPrefs.apiExampleLang)
      ? (storedPrefs.apiExampleLang as ExampleType)
      : "curl",
  );
  // an explicit pick wins, since the snippet may target another machine.
  const [os, setOs] = useState<Os>(
    storedPrefs.apiExampleOs === "windows" ||
      storedPrefs.apiExampleOs === "unix"
      ? storedPrefs.apiExampleOs
      : deviceType === "windows"
        ? "windows"
        : "unix",
  );
  const [copied, setCopied] = useState(false);
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
  // True once the user has picked an agent themselves; guards the detection
  // effect below from clobbering that choice if it resolves afterward.
  const agentPickedByUserRef = useRef(storedPrefs.apiExampleAgent != null);
  const [useTunnel, setUseTunnel] = useState<boolean>(readUseTunnelPref);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl ? cloudflareUrl : (serverUrl ?? origin);
  const localAgentDetection = canUseLocalAgentDetection(base);

  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  // Fetching is the only job of this effect: populate availableAgents/
  // detectedAgents (or clear them). Which agent gets auto-picked from that
  // list is derived separately below, so it can react to the loaded model
  // changing too, not just a fresh fetch.
  useEffect(() => {
    // Browser loopback URLs can be SSH/local forwards, so only the desktop app
    // may use backend PATH checks to mark or auto-pick local agents.
    if (!localAgentDetection) {
      setDetectedAgents([]);
      // A previously auto-picked agent was only ever verified against the
      // Unsloth backend's PATH, which is meaningless now that this panel no
      // longer targets a loopback base -- don't leave it selected, but
      // never touch a choice the user made by hand.
      if (!agentPickedByUserRef.current) {
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

  // a restored agent this build no longer offers cannot build a command.
  useEffect(() => {
    if (!agentPickedByUserRef.current) return;
    if (localAgentDetection && !agentsLoaded) return;
    if (availableAgents.includes(agent)) return;
    agentPickedByUserRef.current = false;
    setStoredAgent(null);
    setAgent(DEFAULT_AGENT);
  }, [
    agent,
    agentsLoaded,
    availableAgents,
    localAgentDetection,
    setStoredAgent,
  ]);

  // Single source of truth for the auto-picked agent, re-derived whenever
  // the detected list or the loaded model's GGUF-ness changes -- in either
  // direction. `codex` needs a GGUF model (unsloth_cli's
  // _require_gguf_for_codex exits otherwise), so it's only preferred once
  // the loaded model actually qualifies; loading a GGUF model *after* a
  // non-GGUF-gated fallback picked something else re-steers back to codex
  // just as loading a non-GGUF model steers away from it. Never overrides a
  // choice the user made by hand.
  // activeGgufVariant alone only covers an HF-repo GGUF pick (a specific
  // quant variant string) -- a direct local .gguf file (custom folder /
  // LM Studio / drag-drop) is just as much a GGUF the codex preflight would
  // accept, but never has a "variant" to report, and would otherwise read as
  // non-GGUF here.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const activeNativePathToken = useChatRuntimeStore(
    (s) => s.activeNativePathToken,
  );
  const loadedIsGguf = useChatRuntimeStore((s) => s.loadedIsGguf);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  useEffect(() => {
    if (agentPickedByUserRef.current) return;
    if (detectedAgents.length === 0) return;
    const isGguf = isServedByLlamaCpp({
      loadedIsGguf,
      activeGgufVariant,
      activeNativePathToken,
      checkpoint,
    });
    const preferred = detectedAgents.find((a) => a !== "codex" || isGguf);
    if (preferred) {
      setAgent(preferred);
    } else if (agent === "codex" && !isGguf) {
      // codex was auto-picked while a GGUF model was active and it's the
      // only detected agent; now that the model isn't GGUF anymore, nothing
      // detected is actually runnable, so fall back to the default instead
      // of leaving a codex command unsloth_cli will reject.
      setAgent(DEFAULT_AGENT);
    }
  }, [
    agent,
    detectedAgents,
    activeGgufVariant,
    activeNativePathToken,
    loadedIsGguf,
    checkpoint,
  ]);

  const keylessBase =
    !(useTunnel && cloudflareUrl) &&
    keylessBaseEligible(base, keylessScope, keylessExposure);
  const model = useExampleModelName(keylessBase && !apiKey);
  // The approved SDK dummy is printed only for a transport the backend can admit.
  const key =
    apiKey || (keylessBase ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);
  // a keyless caller gets no tools until the admin grants them, so this names a real key
  const toolsKey =
    apiKey ||
    (keylessBase && keylessTools ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);
  // agent tools are client-side schemas sent through the admitted inference routes.
  const agentKey =
    apiKey || (keylessBase ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER);

  // Null model: nothing is servable, so there is no snippet worth copying.
  const snippets = useMemo(
    () => (model ? buildSnippets(base, key, toolsKey, model, os) : null),
    [base, key, toolsKey, model, os],
  );
  // Agent command must target the server the panel shows, not the :8888 default.
  const agentCommand = useMemo(
    () => buildAgentCommand(base, agentKey, os, agent),
    [base, agentKey, os, agent],
  );

  const osAware = OS_AWARE[lang];
  const shikiLang = CURL_TYPES.has(lang)
    ? os === "windows"
      ? "powershell"
      : "bash"
    : JAVASCRIPT_TYPES.has(lang)
      ? "javascript"
      : "python";

  const handleCopy = async () => {
    if (!snippets) return;
    if (await copyToClipboard(snippets[lang])) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  const handleToggleTunnel = (next: boolean) => {
    setUseTunnel(next);
    writeUseTunnelPref(next);
  };

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

  return (
    <section className="flex min-w-0 max-w-full flex-col">
      <h2 className="mb-2 text-sm font-semibold text-foreground">
        {t("settings.apiKeys.usageExamples")}
      </h2>
      <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-muted/20">
        {/* No model-auto-switch row: ModelAutoSwitchSection renders that setting just below. */}
        {cloudflareUrl ? (
          <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
            <div className="flex shrink-0 items-center gap-1.5">
              <Switch
                size="sm"
                checked={useTunnel}
                onCheckedChange={handleToggleTunnel}
                aria-label={t("settings.apiKeys.secureHttps")}
              />
              <span className="text-ui-11 font-medium text-foreground">
                {t("settings.apiKeys.secureHttps")}
              </span>
              {/* Only when not launched with --secure: the raw 0.0.0.0 port is
                  still globally reachable, so point the user at --secure. */}
              {secure ? null : (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      className="flex items-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={t("settings.apiKeys.secureHttpsHint")}
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3.5"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-[260px] text-ui-11 leading-snug">
                    {t("settings.apiKeys.secureHttpsHint")}
                  </TooltipContent>
                </Tooltip>
              )}
            </div>
            <button
              type="button"
              onClick={handleCopyUrl}
              className={cn(
                "flex min-w-0 items-center gap-1 rounded px-1.5 py-1 text-ui-11 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
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
        <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
          <div className="flex min-w-0 flex-wrap items-center gap-0.5">
            {TYPE_TABS.map((tab) => {
              const active = lang === tab.id;
              const labelKey = TYPE_LABEL_KEY[tab.id];
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => {
                    setLang(tab.id);
                    setStoredLang(tab.id);
                  }}
                  aria-pressed={active}
                  className={cn(
                    "rounded-full px-2.5 py-1 text-ui-11 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                    active
                      ? "hub-tab-toggle-pill text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {labelKey ? t(labelKey) : tab.label}
                </button>
              );
            })}
          </div>
        </div>
        {osAware ? (
          <div className="flex min-w-0 items-center gap-0.5 border-b border-border px-2 py-1.5">
            <button
              type="button"
              onClick={() => {
                setOs("unix");
                setStoredOs("unix");
              }}
              aria-pressed={os === "unix"}
              className={cn(
                "rounded-full px-2.5 py-1 text-ui-11 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                os === "unix"
                  ? "hub-tab-toggle-pill text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t("settings.apiKeys.osUnix")}
            </button>
            <button
              type="button"
              onClick={() => {
                setOs("windows");
                setStoredOs("windows");
              }}
              aria-pressed={os === "windows"}
              className={cn(
                "rounded-full px-2.5 py-1 text-ui-11 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                os === "windows"
                  ? "hub-tab-toggle-pill text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t("settings.apiKeys.osWindows")}
            </button>
          </div>
        ) : null}
        {snippets ? (
          <div className="relative min-w-0">
            <button
              type="button"
              onClick={handleCopy}
              className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-1 text-ui-11 text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              aria-label={t("settings.apiKeys.copySnippet")}
            >
              <HugeiconsIcon
                icon={copied ? Tick02Icon : Copy01Icon}
                className={cn("size-3.5", copied && "text-emerald-600")}
              />
              {copied
                ? t("settings.apiKeys.copied")
                : t("settings.apiKeys.copy")}
            </button>
            <HighlightedCode
              key={snippets[lang]}
              code={snippets[lang]}
              language={shikiLang}
              redactFromReload={Boolean(apiKey)}
            />
          </div>
        ) : (
          <div className="min-w-0 px-3 py-2.5 text-ui-11 leading-snug text-muted-foreground">
            {t("settings.apiKeys.usageNoModel")}
          </div>
        )}
        <div className="flex min-w-0 flex-col gap-1.5 border-t border-border px-3 py-2.5">
          <span className="text-ui-11 font-semibold text-foreground">
            {t("settings.apiKeys.codingAgents")}
          </span>
          <span className="text-ui-11 leading-snug text-muted-foreground">
            {t("settings.apiKeys.codingAgentsHint")}
          </span>
          <div className="flex min-w-0 flex-wrap items-center gap-1">
            {availableAgents.map((id) => {
              const installed = detectedAgents.includes(id);
              const active = agent === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => {
                    agentPickedByUserRef.current = true;
                    setAgent(id);
                    setStoredAgent(id);
                  }}
                  aria-pressed={active}
                  title={
                    installed
                      ? t("settings.apiKeys.codingAgentDetected")
                      : undefined
                  }
                  className={cn(
                    "flex items-center gap-1 rounded-full px-2.5 py-1 text-ui-11 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                    active
                      ? "hub-tab-toggle-pill text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {AGENT_LABELS[id] ?? id}
                  {installed ? (
                    <span
                      aria-hidden="true"
                      className="size-1.5 rounded-full bg-emerald-500"
                    />
                  ) : null}
                </button>
              );
            })}
          </div>
          <div className="relative mt-0.5 min-w-0">
            <code
              className="block min-w-0 overflow-x-auto rounded border border-border bg-muted/30 px-2 py-1.5 pr-14 font-mono text-ui-11 text-foreground"
              data-reload-snapshot-sensitive={apiKey ? "" : undefined}
            >
              {agentCommand}
            </code>
            <button
              type="button"
              onClick={handleCopyAgent}
              className="absolute right-1.5 top-1/2 flex -translate-y-1/2 items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-0.5 text-ui-11 text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              aria-label={t("settings.apiKeys.copySnippet")}
            >
              <HugeiconsIcon
                icon={copiedAgent ? Tick02Icon : Copy01Icon}
                className={cn("size-3.5", copiedAgent && "text-emerald-600")}
              />
            </button>
          </div>
          <span className="text-ui-11 leading-snug text-muted-foreground">
            {detectedAgents.length > 0
              ? t("settings.apiKeys.codingAgentsDetectedHint", {
                  agents: detectedAgents
                    .map((id) => AGENT_LABELS[id] ?? id)
                    .join(", "),
                })
              : t("settings.apiKeys.codingAgentsSwap")}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border px-3 py-2 text-ui-11 text-muted-foreground">
          <span>{t("settings.apiKeys.setupDocs")}</span>
          {DOC_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-0.5 rounded font-medium text-foreground underline decoration-border underline-offset-2 transition-colors hover:decoration-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {link.label}
              <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
