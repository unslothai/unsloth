// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Popover, PopoverAnchor, PopoverContent } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ATTACHMENT_KIND_ICON_CLASS, ATTACHMENT_KIND_ICONS, attachmentFileKind } from "@/features/chat";
import { useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  MoreHorizontalIcon,
  ArrowUpRight01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { type FinishedDownload, mountDownloadsButton, useDownloadActivity } from "./download-activity";
import { formatSize, revealLabelKey } from "./download-format";
import { type Target, openTarget, revealTarget } from "./download-open";
import { type DownloadItem, useBrowserHistoryStore } from "./history-store";
import { nativeDownloadsExist } from "./native-downloads";
import { useBrowserStore } from "./store";

const RECENT_COUNT = 8;
// Long enough to read and reach for Open; hovering keeps it up.
const COMPLETE_SHOWN_MS = 6000;

const ROUND_ACTION =
  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_7%,transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-40 data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_8%,transparent)]";

/** Desktop downloads missing from disk, by native id. Checked as the list opens, on window focus, and
 *  after an action fails, as the Downloads page does. */
function useMissing(downloads: DownloadItem[], enabled: boolean) {
  const [missing, setMissing] = useState<ReadonlySet<string>>(() => new Set());
  const [checks, setChecks] = useState(0);
  // Keyed by the ids, not the array, which is new on every render.
  const ids = downloads.flatMap((item) => (item.nativeId ? [item.nativeId] : [])).join("\n");
  useEffect(() => {
    if (!enabled || !isTauri || !ids) return;
    const list = ids.split("\n");
    let live = true;
    const check = () =>
      nativeDownloadsExist(list).then(
        (found) => live && setMissing(new Set(list.filter((_, index) => !found[index]))),
        () => undefined,
      );
    void check();
    window.addEventListener("focus", check);
    return () => {
      live = false;
      window.removeEventListener("focus", check);
    };
  }, [ids, enabled, checks]);
  return { missing, recheck: () => setChecks((count) => count + 1) };
}

function KindTile({ name, contentType, muted }: { name: string; contentType: string; muted?: boolean }) {
  const kind = attachmentFileKind(name, contentType);
  return (
    <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-[color-mix(in_oklab,var(--foreground)_6%,transparent)]">
      <HugeiconsIcon
        icon={ATTACHMENT_KIND_ICONS[kind]}
        strokeWidth={1.75}
        className={cn("size-4.5", muted ? "text-muted-foreground" : ATTACHMENT_KIND_ICON_CLASS[kind])}
      />
    </span>
  );
}

function DownloadRow({
  item,
  missing,
  onFailed,
  onDone,
}: {
  item: DownloadItem;
  missing: boolean;
  /** After Open or Show in folder fails, so the row can learn the file is gone. */
  onFailed: () => void;
  onDone: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const failed = (key: "browser.downloads.openFailed" | "browser.pages.revealFailed") => (name: string) => {
    toast.error(t(key, { name }));
    onFailed();
  };
  const target: Target = { ...item, keptId: item.id };
  const open = missing ? undefined : openTarget(target, failed("browser.downloads.openFailed"));
  const reveal = missing ? undefined : revealTarget(target, failed("browser.pages.revealFailed"));
  const time = new Intl.DateTimeFormat(locale, { timeStyle: "short" }).format(item.downloadedAt);
  const status = t(missing ? "browser.downloads.missing" : "browser.downloads.downloaded");
  const revealLabel = t(revealLabelKey());
  const run = (action: (() => void) | undefined) => () => {
    action?.();
    onDone();
  };
  return (
    <li className="group/row flex items-center gap-3 rounded-2xl px-2 py-1.5 hover:bg-[color-mix(in_oklab,var(--foreground)_4%,transparent)]">
      <button
        type="button"
        onClick={run(open)}
        disabled={!open}
        title={item.name}
        className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-start focus-visible:outline-none disabled:cursor-default"
      >
        <KindTile name={item.name} contentType={item.contentType} muted={missing} />
        <span className="flex min-w-0 flex-col">
          <span className={cn("truncate text-ui-14", missing ? "text-muted-foreground" : "text-foreground")}>
            {item.name}
          </span>
          <span className="truncate text-ui-12 text-muted-foreground tabular-nums">
            {[status, item.size > 0 ? formatSize(item.size, locale) : null, time].filter(Boolean).join(" · ")}
          </span>
        </span>
      </button>
      {isTauri ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button type="button" aria-label={revealLabel} disabled={!reveal} onClick={run(reveal)} className={ROUND_ACTION}>
              <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-4.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="tooltip-compact">{revealLabel}</TooltipContent>
        </Tooltip>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button type="button" aria-label={t("browser.more")} className={ROUND_ACTION}>
            <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-4.5" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="browser-menu min-w-48 rounded-[20px] p-1.5">
          <DropdownMenuItem disabled={!open} onSelect={run(open)}>
            <HugeiconsIcon icon={ArrowUpRight01Icon} strokeWidth={1.75} className="size-4" />
            {t("browser.downloads.open")}
          </DropdownMenuItem>
          {isTauri ? (
            <DropdownMenuItem disabled={!reveal} onSelect={run(reveal)}>
              <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-4" />
              {revealLabel}
            </DropdownMenuItem>
          ) : null}
          {item.url ? (
            <DropdownMenuItem
              onSelect={() =>
                void copyToClipboard(item.url ?? "").then((ok) => ok && toast.success(t("browser.linkCopied")))
              }
            >
              <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-4" />
              {t("browser.downloads.copyLink")}
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuItem onSelect={() => useBrowserHistoryStore.getState().removeDownload(item.id)}>
            <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-4" />
            {t("browser.pages.removeFromHistory")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  );
}

function RecentDownloads({ open, onDone }: { open: boolean; onDone: () => void }) {
  const t = useT();
  const downloads = useBrowserHistoryStore((state) => state.downloads);
  const active = useDownloadActivity((state) => state.active);
  const recent = downloads.slice(0, RECENT_COUNT);
  const { missing, recheck } = useMissing(recent, open);
  const running = Object.entries(active);
  return (
    <>
      <div className="flex items-center justify-between gap-2 pr-1 pl-2">
        <span className="text-ui-14 font-medium text-foreground">{t("browser.downloads.title")}</span>
        <button
          type="button"
          onClick={() => {
            useBrowserStore.getState().openInternal("downloads");
            onDone();
          }}
          className="h-7 cursor-pointer rounded-full px-2.5 text-ui-12p5 text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_7%,transparent)] hover:text-foreground"
        >
          {t("browser.downloads.showAll")}
        </button>
      </div>
      {running.length === 0 && recent.length === 0 ? (
        <p className="px-2 py-6 text-center text-ui-13 text-muted-foreground">{t("browser.pages.noDownloads")}</p>
      ) : (
        <ul className="flex flex-col">
          {running.map(([key, name]) => (
            <li key={key} className="flex items-center gap-3 px-2 py-1.5">
              <KindTile name={name} contentType="" />
              <span className="flex min-w-0 flex-col">
                <span className="truncate text-ui-14 text-foreground">{name}</span>
                <span className="text-ui-12 text-muted-foreground">{t("browser.downloads.inProgress")}</span>
              </span>
            </li>
          ))}
          {recent.map((item) => (
            <DownloadRow
              key={item.id}
              item={item}
              missing={item.nativeId !== undefined && missing.has(item.nativeId)}
              onFailed={recheck}
              onDone={onDone}
            />
          ))}
        </ul>
      )}
    </>
  );
}

function FinishedNotice({ finished, onDone }: { finished: FinishedDownload; onDone: () => void }) {
  const t = useT();
  const failed = (key: "browser.downloads.openFailed" | "browser.pages.revealFailed") => (name: string) =>
    toast.error(t(key, { name }));
  const target: Target = { ...finished, keptId: finished.historyId ?? finished.key };
  const open = finished.failed ? undefined : openTarget(target, failed("browser.downloads.openFailed"));
  const reveal = finished.failed ? undefined : revealTarget(target, failed("browser.pages.revealFailed"));
  const run = (action: () => void) => () => {
    action();
    onDone();
  };
  return (
    <div className="flex flex-col gap-3 px-1">
      <div className="flex min-w-0 items-start gap-2.5">
        <HugeiconsIcon
          icon={finished.failed ? Alert02Icon : CheckmarkCircle02Icon}
          strokeWidth={2}
          className={cn("mt-0.5 size-5 shrink-0", finished.failed ? "text-destructive" : "text-primary")}
        />
        <span className="flex min-w-0 flex-col">
          <span className="text-ui-14 font-medium text-foreground">
            {t(finished.failed ? "browser.downloads.failed" : "browser.downloads.complete")}
          </span>
          <span className="truncate text-ui-13 text-muted-foreground" title={finished.name}>
            {finished.name}
          </span>
        </span>
      </div>
      {open || reveal ? (
        <div className="flex justify-end gap-2">
          {reveal ? (
            <button
              type="button"
              onClick={run(reveal)}
              className="h-8 cursor-pointer rounded-full bg-[color-mix(in_oklab,var(--foreground)_8%,transparent)] px-3.5 text-ui-13 text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_12%,transparent)]"
            >
              {t(revealLabelKey())}
            </button>
          ) : null}
          {open ? (
            <button
              type="button"
              onClick={run(open)}
              className="h-8 cursor-pointer rounded-full bg-foreground px-3.5 text-ui-13 text-background transition-opacity hover:opacity-85"
            >
              {t("browser.downloads.open")}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** The download glyph while a file downloads: its arrow inside a spinning ring, the tray bar below. */
function DownloadingIcon() {
  return (
    <svg
      aria-hidden={true}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-5"
    >
      <path d="M12 13.5V6M15.25 10.5c0 0-2.4 3.25-3.25 3.25s-3.25-3.25-3.25-3.25" />
      <path d="M6.5 21.5h11" />
      <circle cx="12" cy="10" r="8.5" strokeOpacity={0.25} />
      <circle
        cx="12"
        cy="10"
        r="8.5"
        strokeDasharray="17 36.4"
        className="animate-spin [animation-duration:1.4s] [transform-box:view-box] [transform-origin:12px_10px]"
      />
    </svg>
  );
}

/** Toolbar Downloads button: a ring while files download, a notice when one finishes, recent downloads on click. */
/** `visible`: false while its panel stays mounted off screen, so results go to a toast instead.
 *  `idleHidden`: file tabs keep their own save button, so the glyph shows only while a download runs
 *  or its result is up. It stays registered meanwhile, so a quick save still finds it. */
export function DownloadsButton({
  className,
  visible = true,
  idleHidden = false,
}: { className?: string; visible?: boolean; idleHidden?: boolean }) {
  const t = useT();
  const active = useDownloadActivity((state) => Object.keys(state.active).length > 0);
  const finished = useDownloadActivity((state) => state.finished);
  const finishedSequence = useDownloadActivity((state) => state.finishedSequence);
  // A notice another toolbar's button was showing (the tab changed) carries on here.
  const [mode, setMode] = useState<"closed" | "list" | "finished">(() =>
    visible && finished ? "finished" : "closed",
  );
  const [hovered, setHovered] = useState(false);
  const anchorRef = useRef<HTMLSpanElement>(null);
  // Only finishes after mount pop the notice. An open list shows recorded ones itself; the rest still need it.
  const [seen, setSeen] = useState(finishedSequence);
  if (finishedSequence !== seen) {
    setSeen(finishedSequence);
    if (visible && finished && (mode !== "list" || !finished.historyId)) {
      setMode("finished");
      setHovered(false);
    }
  }

  // Off screen its popover, portaled out of the panel, would stay up over the next page.
  if (!visible && mode !== "closed") {
    setMode("closed");
    setHovered(false);
  }

  useEffect(() => (visible ? mountDownloadsButton() : undefined), [visible]);

  const close = () => {
    setMode("closed");
    setHovered(false);
    useDownloadActivity.getState().dismissFinished();
  };

  // Hovering holds the notice; leaving starts the countdown again.
  useEffect(() => {
    if (mode !== "finished" || hovered) return;
    const timer = window.setTimeout(() => {
      setMode("closed");
      useDownloadActivity.getState().dismissFinished();
    }, COMPLETE_SHOWN_MS);
    return () => window.clearTimeout(timer);
  }, [mode, hovered, finishedSequence]);
  const label = t(active ? "browser.downloads.inProgressLabel" : "browser.downloads.title");
  const lit = active || mode === "finished";

  return (
    <Popover open={mode !== "closed"} onOpenChange={(open) => (open ? setMode("list") : close())}>
      <PopoverAnchor asChild={true}>
        <span
          ref={anchorRef}
          className={cn(
            "relative flex shrink-0",
            idleHidden && !active && !finished && mode === "closed" && "hidden",
          )}
        >
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={label}
                aria-expanded={mode !== "closed"}
                onClick={() => (mode === "list" ? close() : setMode("list"))}
                className={cn(className, lit && "text-primary hover:text-primary")}
              >
                {active ? (
                  <DownloadingIcon />
                ) : (
                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4.5" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="tooltip-compact">
              {label}
            </TooltipContent>
          </Tooltip>
        </span>
      </PopoverAnchor>
      <PopoverContent
        align="end"
        sideOffset={8}
        onOpenAutoFocus={(event) => mode === "finished" && event.preventDefault()}
        // The button is only the anchor: a press on it would dismiss here, then its click reopen.
        onInteractOutside={(event) => {
          if (anchorRef.current?.contains(event.target as Node)) event.preventDefault();
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={cn(
          "browser-menu gap-2 rounded-[22px] p-2",
          mode === "finished" ? "w-80 p-3" : "w-96 max-h-[min(32rem,var(--radix-popover-content-available-height))]",
        )}
      >
        {mode === "finished" && finished ? (
          <FinishedNotice finished={finished} onDone={close} />
        ) : (
          <RecentDownloads open={mode === "list"} onDone={close} />
        )}
      </PopoverContent>
    </Popover>
  );
}
