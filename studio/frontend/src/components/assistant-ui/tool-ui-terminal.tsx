// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useI18n } from "@/features/i18n";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { CheckIcon, CopyIcon, LoaderIcon, TerminalIcon } from "lucide-react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

const MAX_DISPLAY = 10_000;
const COPY_RESET_MS = 2000;

function truncate(text: string): string {
  return text.length <= MAX_DISPLAY
    ? text
    : `${text.slice(0, MAX_DISPLAY)}\n... (truncated)`;
}

function CopyBtn({ text }: { text: string }) {
  const { t } = useI18n();
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
      aria-label={t("common.copyToClipboard")}
    >
      {copied ? (
        <CheckIcon className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      {copied ? t("common.copied") : t("common.copy")}
    </button>
  );
}

const TerminalToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const { t } = useI18n();
  const command = (args as { command?: string })?.command ?? "";
  const isRunning = status?.type === "running";
  const output =
    typeof result === "string"
      ? result
      : result
        ? JSON.stringify(result, null, 2)
        : "";

  return (
    <ToolFallbackRoot>
      <ToolFallbackTrigger
        toolName={command ? `$ ${command.slice(0, 60)}` : t("assistant.tool.terminal")}
        status={status}
        icon={TerminalIcon}
      />
      <ToolFallbackContent>
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {isRunning ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <LoaderIcon className="size-3.5 animate-spin" />
              <span>{t("common.running")}&hellip;</span>
            </div>
          ) : output ? (
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">{t("assistant.tool.output")}</span>
                <CopyBtn text={output} />
              </div>
              <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
                {truncate(output)}
              </pre>
            </div>
          ) : null}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const TerminalToolUI = memo(
  TerminalToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
TerminalToolUI.displayName = "TerminalToolUI";
