// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A live log, drawn like Settings > Logs: framed mono text that follows the tail, a wrap
// toggle and a jump back to the latest line once the reader scrolls up. Settings' own
// viewer (debugging-tab.tsx) is still inline and can move onto this.

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { TextWrapIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowDownIcon } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const FOLLOW_THRESHOLD_PX = 40;

export function LogPane({
  text,
  empty,
  className,
}: {
  /** The whole log as one string; lines joined with "\n". */
  text: string;
  /** Shown muted while there is nothing yet. */
  empty?: string;
  /** Height of the frame; defaults to Settings' pane. */
  className?: string;
}): ReactElement {
  const t = useT();
  const paneRef = useRef<HTMLPreElement>(null);
  // The ref is the source of truth so scrolling does not re-render; the state drives the button.
  const pinnedRef = useRef(true);
  const [following, setFollowing] = useState(true);
  const [wrap, setWrap] = useState(true);

  const scrollToBottom = useCallback(() => {
    const pane = paneRef.current;
    if (!pane) return;
    pane.scrollTop = pane.scrollHeight;
    pinnedRef.current = true;
    setFollowing(true);
  }, []);

  // Wrapping changes the scroll height without changing the text, so it re-pins too.
  useEffect(() => {
    const pane = paneRef.current;
    if (pane && pinnedRef.current) pane.scrollTop = pane.scrollHeight;
  }, [text, wrap]);

  const onScroll = useCallback(() => {
    const pane = paneRef.current;
    if (!pane) return;
    const pinned =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight <
      FOLLOW_THRESHOLD_PX;
    if (pinned !== pinnedRef.current) {
      pinnedRef.current = pinned;
      setFollowing(pinned);
    }
  }, []);

  return (
    <div className="relative">
      {/* Border on the frame, scrolling inside it: keeps the scrollbar off the rounded corners. */}
      <div
        className={cn(
          "flex rounded-lg border border-border/60 bg-muted/20 py-1 pr-0 pl-1 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
          className ?? "h-[min(26rem,45vh)]",
        )}
      >
        {/* One text surface, not an element per line, so a busy log repaints cheaply. */}
        <pre
          ref={paneRef}
          onScroll={onScroll}
          data-testid="log-pane"
          className={cn(
            "h-full w-full overflow-auto [overflow-anchor:none] scroll-rounded rounded-md py-2 pr-11 pl-2.5 font-mono text-ui-11 leading-[1.55] text-foreground/90",
            wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre",
            !text && "text-muted-foreground",
          )}
        >
          {text || empty}
        </pre>
      </div>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-pressed={wrap}
            aria-label={t("settings.debugging.wrapLines")}
            onClick={() => setWrap((previous) => !previous)}
            className={cn(
              "absolute top-1.5 right-3.5 size-7 bg-background/80 shadow-sm backdrop-blur-sm hover:bg-background [&_svg]:size-3.5",
              wrap ? "text-foreground" : "text-muted-foreground",
            )}
          >
            <HugeiconsIcon strokeWidth={1.75} icon={TextWrapIcon} />
          </Button>
        </TooltipTrigger>
        <TooltipContent>{t("settings.debugging.wrapLines")}</TooltipContent>
      </Tooltip>
      {!following && text ? (
        <Button
          size="xs"
          variant="dark"
          onClick={scrollToBottom}
          className="absolute right-4 bottom-3 rounded-full pr-2.5 pl-2 shadow-md"
        >
          <ArrowDownIcon strokeWidth={2} />
          {t("settings.debugging.jumpToLatest")}
        </Button>
      ) : null}
    </div>
  );
}
