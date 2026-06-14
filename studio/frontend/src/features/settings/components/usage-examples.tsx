// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useChatRuntimeStore } from "@/features/chat";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon, Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";

// Example type (what to call the API with) and OS axis (curl quoting differs;
// Python is byte-identical across OSes so its OS row is hidden).
type ExampleType = "curl" | "python" | "curlTools" | "pythonTools";
type Os = "unix" | "windows";

const TYPE_TABS: { id: ExampleType; label: string; i18n?: boolean }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "curlTools", label: "curl + tools", i18n: true },
  { id: "pythonTools", label: "Python + tools", i18n: true },
];

const TYPE_LABEL_KEY: Partial<Record<ExampleType, TranslationKey>> = {
  curlTools: "settings.apiKeys.exampleCurlTools",
  pythonTools: "settings.apiKeys.examplePythonTools",
};

// curl-based examples carry the OS dimension; Python does not.
const OS_AWARE: Record<ExampleType, boolean> = {
  curl: true,
  python: false,
  curlTools: true,
  pythonTools: false,
};

const PROMPT = "Can Unsloth Studio do API calling?";
// render_html / search_knowledge_base are intentionally omitted from the
// examples; web_search + python + terminal are the reliable built-ins.
const TOOLS = ["web_search", "python", "terminal"];

const DOC_LINKS = [
  {
    label: "Claude Code",
    href: "https://unsloth.ai/docs/basics/claude-code",
  },
  {
    label: "Codex",
    href: "https://unsloth.ai/docs/basics/codex",
  },
  {
    label: "OpenClaw",
    href: "https://unsloth.ai/docs/integrations/openclaw",
  },
  {
    label: "OpenCode",
    href: "https://unsloth.ai/docs/integrations/opencode",
  },
  {
    label: "Hermes Agent",
    href: "https://unsloth.ai/docs/integrations/hermes-agent",
  },
];

// JSON-encode a string (quoted, escaped). The result is also a valid Python
// string literal, so model names with backslashes (Windows paths) or quotes
// never break the generated JSON or Python.
const j = (s: string): string => JSON.stringify(s);
// Embed inside a POSIX single-quoted string: close, escaped quote, reopen.
const shSingle = (s: string): string => s.replace(/'/g, "'\\''");
// Embed inside a PowerShell single-quoted string: '' is a literal quote.
const psSingle = (s: string): string => s.replace(/'/g, "''");
const toolsJson = TOOLS.map(j).join(", ");

// Compact one-line JSON for the Windows body file (PowerShell mangles inline
// double quotes passed to curl.exe, so the body is written to disk instead).
function winBody(model: string, tools: boolean): string {
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: "user", content: PROMPT }],
  };
  if (tools) {
    body.enable_tools = true;
    body.enabled_tools = TOOLS;
  }
  body.stream = true;
  return JSON.stringify(body);
}

function curlUnix(
  base: string,
  key: string,
  model: string,
  tools: boolean,
): string {
  const toolsLines = tools
    ? `
    "enable_tools": true,
    "enabled_tools": [${toolsJson}],`
    : "";
  const body = `{
    "model": ${j(model)},
    "messages": [{"role": "user", "content": ${j(PROMPT)}}],${toolsLines}
    "stream": true
  }`;
  return `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(body)}'`;
}

// Windows: PowerShell. curl is aliased to Invoke-WebRequest, so call curl.exe;
// pass the JSON body via a file to avoid PowerShell's native-arg quote stripping.
function curlWindows(
  base: string,
  key: string,
  model: string,
  tools: boolean,
): string {
  return `$body = '${psSingle(winBody(model, tools))}'
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
  tools: boolean,
): string {
  // enable_tools / enabled_tools are Unsloth extensions, not standard OpenAI
  // fields, so the client forwards them through extra_body.
  const toolsArg = tools
    ? `
    extra_body={
        "enable_tools": True,
        "enabled_tools": [${toolsJson}],
    },`
    : "";
  // With tools, the stream interleaves tool-lifecycle events that carry no
  // choices, so guard chunk.choices before indexing it.
  const loop = tools
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
    messages=[{"role": "user", "content": ${j(PROMPT)}}],${toolsArg}
    stream=True,
)
${loop}`;
}

function buildSnippets(
  base: string,
  key: string,
  model: string,
  os: Os,
): Record<ExampleType, string> {
  const curl = os === "windows" ? curlWindows : curlUnix;
  return {
    curl: curl(base, key, model, false),
    python: pythonSnippet(base, key, model, false),
    curlTools: curl(base, key, model, true),
    pythonTools: pythonSnippet(base, key, model, true),
  };
}

const KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";
const MODEL_FALLBACK = "unsloth/gemma-4-E4B-it-GGUF:UD-Q5_K_XL";

// Default ON: when a tunnel exists, examples should show the public base_url.
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
    // Non-fatal: the toggle still applies for this session.
  }
}

// Active local checkpoint as repo[:variant]. External (external::...) checkpoints
// don't apply to the local API, so fall back there and when nothing is loaded.
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

export function UsageExamples({ apiKey }: { apiKey?: string | null }) {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const [lang, setLang] = useState<ExampleType>("curl");
  const [os, setOs] = useState<Os>(
    deviceType === "windows" ? "windows" : "unix",
  );
  const [copied, setCopied] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [useTunnel, setUseTunnel] = useState<boolean>(readUseTunnelPref);

  // The tunnel can start after the first /api/health read, leaving cloudflareUrl
  // stuck at null; refresh on mount so a running tunnel surfaces here.
  useEffect(() => {
    void fetchDeviceType({ force: true });
  }, []);

  const model = useLoadedModelName();
  // Show the real key while it's still revealed (before "Done"); otherwise the
  // copy-and-replace placeholder.
  const key = apiKey || KEY_PLACEHOLDER;
  // Toggle on + tunnel up: use the public tunnel URL. Off: use the direct
  // host:port from the backend (the browser origin equals the tunnel URL when
  // accessed through the tunnel, so origin is only a last-resort fallback).
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl ? cloudflareUrl : (serverUrl ?? origin);

  const snippets = useMemo(
    () => buildSnippets(base, key, model, os),
    [base, key, model, os],
  );

  const osAware = OS_AWARE[lang];

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

  const handleCopyUrl = async () => {
    if (cloudflareUrl && (await copyToClipboard(cloudflareUrl))) {
      setCopiedUrl(true);
      setTimeout(() => setCopiedUrl(false), 1800);
    }
  };

  return (
    <section className="flex min-w-0 max-w-full flex-col">
      <h2 className="mb-2 text-sm font-semibold text-foreground">
        {t("settings.apiKeys.usageExamples")}
      </h2>
      <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-muted/20">
        {cloudflareUrl ? (
          <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
            <div className="flex shrink-0 items-center gap-1.5">
              <Switch
                size="sm"
                checked={useTunnel}
                onCheckedChange={handleToggleTunnel}
                aria-label={t("settings.apiKeys.cloudflareTunnel")}
              />
              <span className="text-[11px] font-medium text-foreground">
                {t("settings.apiKeys.cloudflareTunnel")}
              </span>
            </div>
            {useTunnel ? (
              <button
                type="button"
                onClick={handleCopyUrl}
                className="flex min-w-0 items-center gap-1 rounded px-1.5 py-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
            ) : null}
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
                    "rounded px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    active
                      ? "bg-background text-foreground shadow-border"
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
                "rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                os === "unix"
                  ? "bg-foreground/[0.08] text-foreground dark:bg-white/[0.12]"
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
                "rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                os === "windows"
                  ? "bg-foreground/[0.08] text-foreground dark:bg-white/[0.12]"
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
            className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-1 text-[11px] text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={t("settings.apiKeys.copySnippet")}
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              className={cn("size-3.5", copied && "text-emerald-600")}
            />
            {copied ? t("settings.apiKeys.copied") : t("settings.apiKeys.copy")}
          </button>
          <pre className="max-w-full overflow-x-auto whitespace-pre-wrap break-words p-3 pr-16 font-mono text-[11px] leading-relaxed text-foreground">
            {snippets[lang]}
          </pre>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border px-3 py-2 text-[11px] text-muted-foreground">
          <span>{t("settings.apiKeys.setupDocs")}</span>
          {DOC_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-0.5 rounded font-medium text-foreground underline decoration-border underline-offset-2 transition-colors hover:decoration-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
