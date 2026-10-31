// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The pieces the API tab's example sections (usage, audio, images and video) share.

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
import { SegmentedPill } from "@/components/segmented-control";
import { getClientPlatform } from "@/components/tauri/window-titlebar";
import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Streamdown } from "streamdown";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import { isLoopbackHost, normalizeHost } from "./agent-command";

export type ExampleOs = "unix" | "windows";

const OS_OPTIONS: { id: ExampleOs; labelKey: TranslationKey }[] = [
  { id: "unix", labelKey: "settings.apiKeys.osUnix" },
  { id: "windows", labelKey: "settings.apiKeys.osWindows" },
];

// The btn-fill grey (index.css), so a selected pill matches a standard filled button.
const LIGHT_FILL_PILL =
  "[html:not(.dark)_&]:bg-[color-mix(in_srgb,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] [html:not(.dark)_&]:shadow-none";
// Dark: the --accent pill nearly matches the example box's 5% white.
const DARK_FILL_PILL =
  "[html.dark_&]:!bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]";

// Lines that fit the example box's starting height (h-46); longer ones start there and scroll.
const START_LINES = 8;

export function isLoopbackUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  try {
    return isLoopbackHost(normalizeHost(new URL(url).hostname));
  } catch {
    return false;
  }
}

/** The platform an example's shell commands target. An explicit pick wins (the snippet may
 *  run on another machine). Otherwise a loopback base is this machine, where deviceType tells
 *  WSL from Windows, and any other base is pasted where the user is: the client platform. */
export function useExampleOs(
  base: string,
): [ExampleOs, (os: ExampleOs) => void] {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setApiExampleOs);
  const [override, setOverride] = useState<ExampleOs | null>(() => {
    const stored = useSettingsPanelPrefsStore.getState().apiExampleOs;
    return stored === "windows" || stored === "unix" ? stored : null;
  });
  // Anchor the match: a bare includes("win") would also match "darwin".
  const [isWindowsClient] = useState(() => {
    const p = getClientPlatform();
    return p.startsWith("win") || p.includes("windows");
  });
  const os: ExampleOs =
    override ??
    ((isLoopbackUrl(base) ? deviceType === "windows" : isWindowsClient)
      ? "windows"
      : "unix");
  const pick = (next: ExampleOs) => {
    setOverride(next);
    setStoredOs(next);
  };
  return [os, pick];
}

/** A labelled control in the Agents tab's style: fixed-height header, then the control. */
export function ExampleField({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <div className="flex h-5 items-center">
        <span className="truncate text-xs font-medium text-foreground">
          {label}
        </span>
      </div>
      {children}
    </div>
  );
}

/** A code box header's choice: "toggle" (the Agents tab's) for what the example does,
 *  "pills" (as the usage examples' were) for the language and platform it is written for. */
export function ExampleToggle<T extends string>({
  label,
  value,
  options,
  onChange,
  variant = "pills",
}: {
  label: string;
  value: T;
  options: { id: T; label: string }[];
  onChange: (next: T) => void;
  variant?: "toggle" | "pills";
}) {
  const pillId = useId();
  const toggle = variant === "toggle";
  return (
    <fieldset className="flex min-w-0">
      <legend className="sr-only">{label}</legend>
      <div
        className={
          toggle
            ? "hub-tab-toggle flex h-9 w-fit max-w-full min-w-0 items-center rounded-full"
            : "example-pills isolate flex min-w-0 flex-wrap items-center gap-0.5"
        }
      >
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            aria-pressed={value === option.id}
            className={cn(
              "relative whitespace-nowrap rounded-full font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              toggle
                ? "inline-flex h-9 items-center px-3 text-ui-12"
                : "px-2.5 py-1 text-ui-11",
              value === option.id
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {value === option.id && (
              <SegmentedPill
                group={pillId}
                // With no track behind them, the pills take a filled button's grey in light mode, not a
                // white lifted pill.
                className={
                  toggle ? undefined : cn(LIGHT_FILL_PILL, DARK_FILL_PILL)
                }
              />
            )}
            <span className="relative truncate">{option.label}</span>
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function PlatformToggle({
  value,
  onChange,
}: {
  value: ExampleOs;
  onChange: (os: ExampleOs) => void;
}) {
  const t = useT();
  return (
    <ExampleToggle
      label={t("settings.apiKeys.examplePlatform")}
      value={value}
      options={OS_OPTIONS.map((o) => ({ id: o.id, label: t(o.labelKey) }))}
      onChange={onChange}
    />
  );
}

const SHIKI_THEMES = [unslothLightTheme, unslothDarkTheme] as [
  typeof unslothLightTheme,
  typeof unslothDarkTheme,
];
const codePlugin = createCodePlugin({ themes: SHIKI_THEMES });

export function HighlightedCode({
  code,
  language,
  label,
  redactFromReload,
}: {
  code: string;
  language: string;
  label: string;
  redactFromReload: boolean;
}) {
  const markdown = useMemo(
    () => `\`\`\`${language}\n${code}\n\`\`\``,
    [code, language],
  );
  return (
    // Intrinsic code width keeps scrolling on the focusable section.
    <section
      // Safari never focuses a scroller on its own, so long examples need a tab stop.
      // biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard-scrollable region
      tabIndex={0}
      aria-label={label}
      className="max-h-96 max-w-full hover-scrollbar overflow-auto [mask-image:linear-gradient(to_bottom,transparent,#000_0.75rem)] p-3 pr-16 outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset text-ui-11 leading-relaxed [&_pre]:!m-0 [&_pre]:!whitespace-pre [&_pre]:!break-normal [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-ui-11 [&_pre]:!leading-relaxed [&_code]:!text-ui-11 [&_[data-streamdown=code-block]]:!min-w-max [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block]]:!text-ui-11"
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
    </section>
  );
}

/** An example's code under one header row of its choices and, at the end, the Platform pills,
 *  with an icon copy on the first code line. No language label (the Language pills say it); starts about eight lines tall
 *  and drags taller from its grip. */
export function ExampleCodeBox({
  code,
  language,
  label,
  redactFromReload,
  pills,
  os,
  onOsChange,
  fitContent = false,
}: {
  code: string;
  language: string;
  label: string;
  redactFromReload: boolean;
  pills: ReactNode;
  os: ExampleOs;
  onOsChange: (os: ExampleOs) => void;
  /** Show the whole example up to most of the viewport instead of starting eight lines tall. */
  fitContent?: boolean;
}) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  const tall = code.split("\n").length > START_LINES;

  const handleCopy = async () => {
    if (await copyToClipboard(code)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  const startResize = (event: ReactPointerEvent<HTMLSpanElement>) => {
    const scroller = boxRef.current?.querySelector("section");
    if (!scroller || event.button !== 0) {
      return;
    }
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const startY = event.clientY;
    const startHeight = scroller.getBoundingClientRect().height;
    const move = (e: PointerEvent) => {
      // 96px floor matches min-h-24.
      setHeight(Math.max(96, Math.round(startHeight + e.clientY - startY)));
    };
    const stop = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  };

  return (
    <div className="relative min-w-0 overflow-hidden rounded-xl border border-border bg-background/70 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]">
      {/* Square scroller: this overflow-hidden rounds its scrollbars off at the corners. */}
      <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 px-2 pt-1.5">
        {pills}
        <div className="ml-auto">
          <PlatformToggle value={os} onChange={onOsChange} />
        </div>
      </div>
      {/* HighlightedCode's max-h-96 is lifted so a drag can grow the box past it. */}
      <div
        ref={boxRef}
        style={
          height === null
            ? undefined
            : ({ "--example-h": `${height}px` } as CSSProperties)
        }
        className={cn(
          "relative [&>section]:max-h-none [&>section]:min-h-24 [&>section]:pr-8 [&_[data-streamdown=code-block-header]]:hidden",
          height === null
            ? fitContent
              ? "[&>section]:max-h-[60dvh]"
              : tall && "[&>section]:h-46"
            : "[&>section]:h-(--example-h)",
        )}
      >
        {/* On the first code line, clear of the vertical scrollbar. */}
        <button
          type="button"
          onClick={handleCopy}
          className="absolute top-2.5 right-3 z-10 flex size-6 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          aria-label={
            copied
              ? t("settings.apiKeys.copied")
              : t("settings.apiKeys.copySnippet")
          }
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            className={cn("size-3.5", copied && "text-control-accent")}
          />
        </button>
        <HighlightedCode
          key={code}
          code={code}
          language={language}
          label={label}
          redactFromReload={redactFromReload}
        />
      </div>
      {/* Our own handle, not CSS resize: the native grabber can't be hidden without custom
          scrollbars and the rounded corner clips it into a stray arc. Inset past the curve.
          Mouse-only, like the native one; the keyboard still scrolls the region. */}
      <span
        aria-hidden="true"
        onPointerDown={startResize}
        className="absolute right-1 bottom-1 flex size-5 cursor-ns-resize touch-none items-center justify-center text-muted-foreground/60 transition-colors hover:text-muted-foreground"
      >
        <svg aria-hidden="true" viewBox="0 0 10 10" className="size-2.5">
          <path
            d="M9 3 3 9M9 6.5 6.5 9"
            stroke="currentColor"
            strokeWidth="1.25"
            strokeLinecap="round"
            fill="none"
          />
        </svg>
      </span>
    </div>
  );
}
