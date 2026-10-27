// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSingleLine } from "@/hooks/use-single-line";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiProviderLogo } from "../../chat/api-provider-logo";

// bind feedback to the copied text so command changes cannot retain a stale tick.
export function useCopyButton(text: string) {
  const textVersion = useMemo(() => Symbol(text), [text]);
  const [copiedVersion, setCopiedVersion] = useState<symbol | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const currentVersionRef = useRef(textVersion);

  useEffect(() => {
    currentVersionRef.current = textVersion;
    return () => {
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    };
  }, [textVersion]);

  const copy = async () => {
    const requestedText = text;
    const requestedVersion = textVersion;
    if (!(await copyToClipboard(requestedText))) return;
    if (currentVersionRef.current !== requestedVersion) return;
    setCopiedVersion(requestedVersion);
    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = window.setTimeout(() => {
      setCopiedVersion(null);
      timeoutRef.current = null;
    }, 1600);
  };

  const reset = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setCopiedVersion(null);
  };

  return { copied: copiedVersion === textVersion, copy, reset };
}

export function AgentIcon({
  logo,
  icon,
  darkIcon,
  color,
  mark,
  compact = false,
}: {
  logo?: string;
  icon?: string;
  darkIcon?: string;
  color?: string;
  mark?: string;
  compact?: boolean;
}) {
  const markSize = compact ? "size-4" : "size-5";
  if (logo) {
    return (
      <span className="flex size-5 shrink-0 items-center justify-center overflow-hidden rounded-sm">
        <ApiProviderLogo
          providerType={logo}
          className={cn(markSize, "rounded-sm")}
        />
      </span>
    );
  }
  if (icon) {
    const iconSrc = `${import.meta.env.BASE_URL}agent-logos/${icon}`;
    const darkIconSrc = darkIcon
      ? `${import.meta.env.BASE_URL}agent-logos/${darkIcon}`
      : null;
    return (
      <span className="flex size-5 shrink-0 items-center justify-center overflow-hidden rounded-sm">
        <img
          src={iconSrc}
          alt=""
          aria-hidden={true}
          className={cn(
            markSize,
            "object-contain",
            darkIconSrc && "dark:hidden",
          )}
        />
        {darkIconSrc ? (
          <img
            src={darkIconSrc}
            alt=""
            aria-hidden={true}
            className={cn("hidden object-contain dark:block", markSize)}
          />
        ) : null}
      </span>
    );
  }
  return (
    <span
      aria-hidden={true}
      style={{ backgroundColor: color }}
      className="flex size-5 shrink-0 items-center justify-center rounded-sm font-heading text-ui-10 font-semibold text-white"
    >
      {mark}
    </span>
  );
}

export function CommandBlock({ command }: { command: string }) {
  const t = useT();
  const { copied, copy } = useCopyButton(command);
  const [preRef, singleLine] = useSingleLine<HTMLPreElement>();

  return (
    <div
      className={cn(
        "group relative overflow-hidden border border-border bg-muted/40 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
        singleLine ? "rounded-full" : "rounded-lg",
      )}
    >
      <pre
        ref={preRef}
        className="py-3 pr-11 pl-4 text-xs leading-relaxed text-foreground"
      >
        <code className="block font-mono whitespace-pre-wrap break-all">
          {command}
        </code>
      </pre>
      <button
        type="button"
        onClick={copy}
        aria-label={
          copied ? t("settings.agents.copied") : t("settings.agents.copy")
        }
        className="absolute top-2 right-2 flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
