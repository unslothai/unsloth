// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { getAuthToken } from "@/features/auth/session";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { code as codePlugin } from "@streamdown/code";
import { CheckIcon, CodeIcon, CopyIcon, LoaderIcon } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Streamdown } from "streamdown";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

interface StructuredResult {
  text: string;
  images: string[];
  sessionId: string;
}

const MAX_DISPLAY = 10_000;
const COPY_RESET_MS = 2000;
const SHIKI_THEME = ["github-light", "github-dark"] as ["github-light", "github-dark"];

function truncate(text: string): string {
  return text.length <= MAX_DISPLAY
    ? text
    : `${text.slice(0, MAX_DISPLAY)}\n... (truncated)`;
}

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) {
        clearTimeout(timer.current);
      }
    };
  }, []);

  const copy = useCallback(async () => {
    if (await copyToClipboard(text)) {
      setCopied(true);
      if (timer.current) {
        clearTimeout(timer.current);
      }
      timer.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      aria-label="Copy to clipboard"
    >
      {copied ? (
        <CheckIcon className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

/** Render code with syntax highlighting via Streamdown + shiki. No extra borders — inherits parent container. */
function HighlightedCode({ code: source, language }: { code: string; language: string }) {
  const markdown = useMemo(
    () => `\`\`\`${language}\n${truncate(source)}\n\`\`\``,
    [source, language],
  );
  return (
    <div className="max-h-48 overflow-auto text-xs [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-xs [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block]]:!border-0">
      <Streamdown
        mode="static"
        plugins={{ code: codePlugin }}
        controls={{ code: false }}
        shikiTheme={SHIKI_THEME}
      >
        {markdown}
      </Streamdown>
    </div>
  );
}

function isStructuredResult(val: unknown): val is StructuredResult {
  return (
    typeof val === "object" &&
    val !== null &&
    "text" in val &&
    "images" in val &&
    "sessionId" in val
  );
}

const PythonToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const code = (args as { code?: string })?.code ?? "";
  const firstLine = code.split("\n")[0]?.slice(0, 60) ?? "";
  const isRunning = status?.type === "running";

  let output: string;
  let images: string[] = [];
  let sessionId = "";

  if (isStructuredResult(result)) {
    output = result.text;
    images = result.images;
    sessionId = result.sessionId;
  } else if (typeof result === "string") {
    output = result;
  } else if (result) {
    output = JSON.stringify(result, null, 2);
  } else {
    output = "";
  }

  const authToken = getAuthToken();

  return (
    <ToolFallbackRoot>
      <ToolFallbackTrigger
        toolName={firstLine ? `Python: ${firstLine}` : "Python"}
        status={status}
        icon={CodeIcon}
      />
      <ToolFallbackContent>
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {/* Code + copy */}
          {code && (
            <div className="flex justify-end">
              <CopyBtn text={code} />
            </div>
          )}
          {code && <HighlightedCode code={code} language="python" />}

          {/* Output */}
          {isRunning ? (
            <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
              <LoaderIcon className="size-3.5 animate-spin" />
              <span>Running&hellip;</span>
            </div>
          ) : output ? (
            <div className="mt-2 border-t border-dashed pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">output</span>
                <CopyBtn text={output} />
              </div>
              <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
                {truncate(output)}
              </pre>
            </div>
          ) : null}

          {/* Images from Python tool execution */}
          {images.length > 0 && sessionId && (
            <div className="mt-2 flex flex-col gap-2">
              {images.map((filename) => (
                <img
                  key={filename}
                  src={`/api/inference/sandbox/${encodeURIComponent(sessionId)}/${encodeURIComponent(filename)}${authToken ? `?token=${encodeURIComponent(authToken)}` : ""}`}
                  alt={filename}
                  loading="lazy"
                  className="max-w-full rounded border border-border"
                />
              ))}
            </div>
          )}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const PythonToolUI = memo(
  PythonToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
PythonToolUI.displayName = "PythonToolUI";
