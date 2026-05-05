// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  ArrowUpRight01Icon,
  Copy01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";

type Lang = "curl" | "python" | "tools";

const TABS: { id: Lang; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "tools", label: "Tools" },
];

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

function buildSnippets(base: string) {
  return {
    curl: `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'`,
    python: `from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key="sk-unsloth-YOUR_KEY",
)

response = client.chat.completions.create(
    model="current",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")`,
    tools: `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Search Python 3.13 features"}],
    "enable_tools": true,
    "enabled_tools": ["web_search", "python"],
    "stream": true
  }'`,
  };
}

export function UsageExamples() {
  const [lang, setLang] = useState<Lang>("curl");
  const [copied, setCopied] = useState(false);
  const snippets = useMemo(
    () =>
      buildSnippets(
        typeof window !== "undefined" ? window.location.origin : "",
      ),
    [],
  );

  const handleCopy = async () => {
    if (await copyToClipboard(snippets[lang])) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <section className="flex min-w-0 max-w-full flex-col">
      <h2 className="mb-2 text-sm font-semibold text-foreground">Usage examples</h2>
      <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-muted/20">
        <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
          <div className="flex min-w-0 items-center gap-0.5">
            {TABS.map((t) => {
              const active = lang === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setLang(t.id)}
                  aria-pressed={active}
                  className={cn(
                    "rounded px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    active
                      ? "bg-background text-foreground shadow-border"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {t.label}
                </button>
              );
            })}
          </div>
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Copy snippet"
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              className={cn("size-3.5", copied && "text-emerald-600")}
            />
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="max-w-full overflow-x-auto whitespace-pre-wrap break-words p-3 font-mono text-[11px] leading-relaxed text-foreground">
          {snippets[lang]}
        </pre>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border px-3 py-2 text-[11px] text-muted-foreground">
          <span>Setup docs:</span>
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
