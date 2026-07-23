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
import {
  type OpenAIAutoSwitchSettings,
  loadOpenAIAutoSwitchSettings,
  updateOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import { buildAgentCommand, isLoopbackHost, normalizeHost } from "./agent-command";

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

const PROMPT = "Can Unsloth Studio do API calling?";
// Auto-switch demo: a second call naming a different downloaded GGUF so the
// example shows that the model field selects which model serves.
// A placeholder the user replaces with one of their downloaded GGUFs. A fixed
// repo is usually not one they have, so the resolver would fall through and the
// demo would keep serving the current model instead of switching.
const SWITCH_MODEL = "your-other-downloaded-GGUF";
const SWITCH_PROMPT = "Now answer as a different model.";
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
  { label: "Hermes Agent", href: "https://unsloth.ai/docs/integrations/hermes-agent" },
];

// Falls back to this list until the backend's installed-CLI check resolves;
// kept in sync with the `unsloth start <agent>` subcommands and with
// CODING_AGENTS in studio/backend/utils/coding_agents.py.
const DEFAULT_AGENTS = [
  "claude",
  "codex",
  "openclaw",
  "opencode",
  "hermes",
  "pi",
];
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
  pi: "Pi",
};

const j = (s: string): string => JSON.stringify(s);
const shSingle = (s: string): string => s.replace(/'/g, "'\\''");
const psSingle = (s: string): string => s.replace(/'/g, "''");
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

// A leading comment (valid in both bash and PowerShell) noting the model field
// selects the served model when auto-switch is on.
const SWITCH_NOTE =
  '# "Switch model by request" is on: set "model" to any downloaded GGUF to switch.\n';

function curlUnix(
  base: string,
  key: string,
  model: string,
  variant: Variant,
  autoSwitch: boolean,
): string {
  return `${autoSwitch ? SWITCH_NOTE : ""}curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(curlBodyPretty(model, variant))}'`;
}

function curlWindows(
  base: string,
  key: string,
  model: string,
  variant: Variant,
  autoSwitch: boolean,
): string {
  return `${autoSwitch ? SWITCH_NOTE : ""}$body = '${psSingle(winBody(model, variant))}'
Set-Content -Path body.json -Value $body -Encoding ascii
curl.exe ${base}/v1/chat/completions \`
  -H "Authorization: Bearer ${key}" \`
  -H "Content-Type: application/json" \`
  -d "@body.json"`;
}

// A second OpenAI call naming a different downloaded GGUF: with auto-switch on,
// Unsloth loads it before serving, so the model field selects the served model.
function pythonSwitchDemo(): string {
  return `

# "Switch model by request" is on: replace the model below with another GGUF you
# have downloaded and Unsloth loads it before serving. Unknown names keep serving
# the current model.
response = client.chat.completions.create(
    model=${j(SWITCH_MODEL)},
    messages=[{"role": "user", "content": ${j(SWITCH_PROMPT)}}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")`;
}

function pythonSnippet(
  base: string,
  key: string,
  model: string,
  variant: Variant,
  autoSwitch: boolean,
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
${loop}${autoSwitch ? pythonSwitchDemo() : ""}`;
}

function javascriptSnippet(
  base: string,
  key: string,
  model: string,
  variant: Variant,
  autoSwitch: boolean,
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
    options.push(`  enable_thinking: true,`);
  }
  if (variant !== "plain") {
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
}${autoSwitch ? javascriptSwitchDemo() : ""}`;
}

function javascriptSwitchDemo(): string {
  return `

// "Switch model by request" is on: replace the model below with another GGUF you
// have downloaded and Unsloth loads it before serving. Unknown names keep serving
// the current model.
const switchResponse = await client.chat.completions.create({
  model: ${j(SWITCH_MODEL)},
  messages: [{ role: "user", content: ${j(SWITCH_PROMPT)} }],
  stream: true,
});

for await (const chunk of switchResponse) {
  process.stdout.write(chunk.choices?.[0]?.delta?.content || "");
}`;
}

function buildSnippets(
  base: string,
  key: string,
  model: string,
  os: Os,
  autoSwitch: boolean,
): Record<ExampleType, string> {
  const curl = os === "windows" ? curlWindows : curlUnix;
  return {
    curl: curl(base, key, model, "plain", autoSwitch),
    python: pythonSnippet(base, key, model, "plain", autoSwitch),
    javascript: javascriptSnippet(base, key, model, "plain", autoSwitch),
    curlTools: curl(base, key, model, "tools", autoSwitch),
    pythonTools: pythonSnippet(base, key, model, "tools", autoSwitch),
    javascriptTools: javascriptSnippet(base, key, model, "tools", autoSwitch),
    curlAdvanced: curl(base, key, model, "advanced", autoSwitch),
    pythonAdvanced: pythonSnippet(base, key, model, "advanced", autoSwitch),
    javascriptAdvanced: javascriptSnippet(
      base,
      key,
      model,
      "advanced",
      autoSwitch,
    ),
  };
}

const KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";
const MODEL_FALLBACK = "unsloth/gemma-4-E4B-it-GGUF:UD-Q5_K_XL";
const USE_TUNNEL_KEY = "unsloth_api_use_tunnel";

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

function useLoadedModelName(): string {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  return useMemo(() => {
    if (!checkpoint || checkpoint.startsWith("external::")) {
      return MODEL_FALLBACK;
    }
    if (ggufVariant && !checkpoint.includes(":")) {
      return `${checkpoint}:${ggufVariant}`;
    }
    return checkpoint;
  }, [checkpoint, ggufVariant]);
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
}: {
  code: string;
  language: string;
}) {
  const markdown = useMemo(
    () => `\`\`\`${language}\n${code}\n\`\`\``,
    [code, language],
  );
  return (
    <div className="max-w-full overflow-x-auto p-3 pr-16 text-[0.6875rem] leading-relaxed [&_pre]:!m-0 [&_pre]:!whitespace-pre-wrap [&_pre]:!break-words [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-[0.6875rem] [&_pre]:!leading-relaxed [&_code]:!text-[0.6875rem] [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block]]:!text-[0.6875rem]">
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

export function UsageExamples({ apiKey }: { apiKey?: string | null }) {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const secure = usePlatformStore((s) => s.secure);
  const [lang, setLang] = useState<ExampleType>("curl");
  const [os, setOs] = useState<Os>(
    deviceType === "windows" ? "windows" : "unix",
  );
  const [copied, setCopied] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [copiedAgent, setCopiedAgent] = useState(false);
  const [agent, setAgent] = useState<string>(DEFAULT_AGENT);
  const [availableAgents, setAvailableAgents] =
    useState<string[]>(DEFAULT_AGENTS);
  const [detectedAgents, setDetectedAgents] = useState<string[]>([]);
  // True once the user has picked an agent themselves; guards the detection
  // effect below from clobbering that choice if it resolves afterward.
  const agentPickedByUserRef = useRef(false);
  const [useTunnel, setUseTunnel] = useState<boolean>(readUseTunnelPref);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl ? cloudflareUrl : (serverUrl ?? origin);
  const localAgentDetection = canUseLocalAgentDetection(base);
  // null while loading; the same setting the General tab exposes (shared cache).
  const [autoSwitch, setAutoSwitch] = useState<OpenAIAutoSwitchSettings | null>(
    null,
  );
  const [savingAutoSwitch, setSavingAutoSwitch] = useState(false);

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
      });
    return () => {
      cancelled = true;
    };
  }, [localAgentDetection]);

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
  // non-GGUF here. activeNativePathToken covers the drag-drop/picked-file
  // case; ggufContextLength is only ever populated when the backend's
  // /api/inference/status last reported is_gguf: true for the active model
  // (see applyActiveModelStatusToStore), so together these three cover every
  // path a model can be GGUF through, matching the same is_gguf-or-equivalent
  // check hasGgufSource applies to a staged pick.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const activeNativePathToken = useChatRuntimeStore((s) => s.activeNativePathToken);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  useEffect(() => {
    if (agentPickedByUserRef.current) return;
    if (detectedAgents.length === 0) return;
    const isGguf =
      activeGgufVariant != null || activeNativePathToken != null || ggufContextLength != null;
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
  }, [agent, detectedAgents, activeGgufVariant, activeNativePathToken, ggufContextLength]);

  useEffect(() => {
    let cancelled = false;
    void loadOpenAIAutoSwitchSettings()
      .then((s) => {
        if (!cancelled) setAutoSwitch(s);
      })
      .catch(() => {
        // Best-effort: leave the toggle off if the setting can't be read.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const model = useLoadedModelName();
  const key = apiKey || KEY_PLACEHOLDER;

  const autoSwitchOn = autoSwitch?.enabled ?? false;
  const snippets = useMemo(
    () => buildSnippets(base, key, model, os, autoSwitchOn),
    [base, key, model, os, autoSwitchOn],
  );
  // Agent command must target the server the panel shows, not the :8888 default.
  const agentCommand = useMemo(
    () => buildAgentCommand(base, key, os, agent),
    [base, key, os, agent],
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
    if (await copyToClipboard(snippets[lang])) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  const handleToggleTunnel = (next: boolean) => {
    setUseTunnel(next);
    writeUseTunnelPref(next);
  };

  // Same setting as the General tab; persist optimistically and revert on failure
  // so the examples reflect the live model-switch behavior.
  const handleToggleAutoSwitch = (next: boolean) => {
    const idle = autoSwitch?.autoUnloadIdleSeconds ?? 0;
    setAutoSwitch((prev) => (prev ? { ...prev, enabled: next } : prev));
    setSavingAutoSwitch(true);
    void updateOpenAIAutoSwitchSettings(next, idle)
      .then(setAutoSwitch)
      .catch(() => {
        setAutoSwitch((prev) => (prev ? { ...prev, enabled: !next } : prev));
      })
      .finally(() => setSavingAutoSwitch(false));
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
        {/* Same setting as the General tab; surfaced here so the request `model`
            actually switches the served model, which the examples below show. */}
        <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
          <div className="flex shrink-0 items-center gap-1.5">
            <Switch
              size="sm"
              checked={autoSwitchOn}
              disabled={autoSwitch === null || savingAutoSwitch}
              onCheckedChange={handleToggleAutoSwitch}
              aria-label={t("settings.general.modelAutoSwitch.enable")}
            />
            <span className="text-[0.6875rem] font-medium text-foreground">
              {t("settings.general.modelAutoSwitch.enable")}
            </span>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="flex items-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  aria-label={t(
                    "settings.general.modelAutoSwitch.enableDescription",
                  )}
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-[260px] text-[0.6875rem] leading-snug">
                {t("settings.general.modelAutoSwitch.enableDescription")}
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
        {cloudflareUrl ? (
          <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
            <div className="flex shrink-0 items-center gap-1.5">
              <Switch
                size="sm"
                checked={useTunnel}
                onCheckedChange={handleToggleTunnel}
                aria-label={t("settings.apiKeys.secureHttps")}
              />
              <span className="text-[0.6875rem] font-medium text-foreground">
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
                  <TooltipContent className="max-w-[260px] text-[0.6875rem] leading-snug">
                    {t("settings.apiKeys.secureHttpsHint")}
                  </TooltipContent>
                </Tooltip>
              )}
            </div>
            <button
              type="button"
              onClick={handleCopyUrl}
              className={cn(
                "flex min-w-0 items-center gap-1 rounded px-1.5 py-1 text-[0.6875rem] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
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
                  onClick={() => setLang(tab.id)}
                  aria-pressed={active}
                  className={cn(
                    "rounded-full px-2.5 py-1 text-[0.6875rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
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
              onClick={() => setOs("unix")}
              aria-pressed={os === "unix"}
              className={cn(
                "rounded-full px-2.5 py-1 text-[0.6875rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                os === "unix"
                  ? "hub-tab-toggle-pill text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t("settings.apiKeys.osUnix")}
            </button>
            <button
              type="button"
              onClick={() => setOs("windows")}
              aria-pressed={os === "windows"}
              className={cn(
                "rounded-full px-2.5 py-1 text-[0.6875rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                os === "windows"
                  ? "hub-tab-toggle-pill text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t("settings.apiKeys.osWindows")}
            </button>
          </div>
        ) : null}
        <div className="relative min-w-0">
          <button
            type="button"
            onClick={handleCopy}
            className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-1 text-[0.6875rem] text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-label={t("settings.apiKeys.copySnippet")}
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              className={cn("size-3.5", copied && "text-emerald-600")}
            />
            {copied ? t("settings.apiKeys.copied") : t("settings.apiKeys.copy")}
          </button>
          <HighlightedCode
            key={snippets[lang]}
            code={snippets[lang]}
            language={shikiLang}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-1.5 border-t border-border px-3 py-2.5">
          <span className="text-[0.6875rem] font-semibold text-foreground">
            {t("settings.apiKeys.codingAgents")}
          </span>
          <span className="text-[0.6875rem] leading-snug text-muted-foreground">
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
                  }}
                  aria-pressed={active}
                  title={
                    installed
                      ? t("settings.apiKeys.codingAgentDetected")
                      : undefined
                  }
                  className={cn(
                    "flex items-center gap-1 rounded-full px-2.5 py-1 text-[0.6875rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
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
            <code className="block min-w-0 overflow-x-auto rounded border border-border bg-muted/30 px-2 py-1.5 pr-14 font-mono text-[0.6875rem] text-foreground">
              {agentCommand}
            </code>
            <button
              type="button"
              onClick={handleCopyAgent}
              className="absolute right-1.5 top-1/2 flex -translate-y-1/2 items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-0.5 text-[0.6875rem] text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              aria-label={t("settings.apiKeys.copySnippet")}
            >
              <HugeiconsIcon
                icon={copiedAgent ? Tick02Icon : Copy01Icon}
                className={cn("size-3.5", copiedAgent && "text-emerald-600")}
              />
            </button>
          </div>
          <span className="text-[0.6875rem] leading-snug text-muted-foreground">
            {detectedAgents.length > 0
              ? t("settings.apiKeys.codingAgentsDetectedHint", {
                  agents: detectedAgents
                    .map((id) => AGENT_LABELS[id] ?? id)
                    .join(", "),
                })
              : t("settings.apiKeys.codingAgentsSwap")}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border px-3 py-2 text-[0.6875rem] text-muted-foreground">
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
