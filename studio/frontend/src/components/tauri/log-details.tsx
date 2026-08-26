// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isFollowingTail } from "@/components/tauri/log-follow";

import { ChevronDown as ChevronDownIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useLayoutEffect, useRef } from "react";

/**
 * The collapsed log tail every long-running full-window screen shows under its status
 * line. Shared so install, repair and update stay one control rather than three copies
 * that drift apart; only the noun in the toggle and the lines themselves differ.
 *
 * The log follows its own tail, but yields: scrolling up to read something pins the view
 * there however many lines arrive next, and scrolling back to the bottom resumes the
 * follow. Reopening the panel always starts at the latest line.
 */
export function LogDetails({
  label,
  lines,
}: {
  /** Noun phrase completing "Show ..." / "Hide ...", e.g. "installation details". */
  label: string;
  lines: string[];
}) {
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const logRef = useRef<HTMLPreElement>(null);
  // A ref, not state: nothing renders from it, and a re-render per scroll event while a
  // log is streaming is exactly what this component cannot afford.
  const following = useRef(true);

  const text = lines.join("\n");

  // Layout effect, not a passive one: the browser must not paint the new lines at the old
  // offset first, or a fast-appending log visibly judders on every update.
  // biome-ignore lint/correctness/useExhaustiveDependencies: text is the trigger, not a read - new lines changed the DOM, which is what there is to react to
  useLayoutEffect(() => {
    if (!following.current) {
      return;
    }
    const log = logRef.current;
    if (!log) {
      return;
    }
    log.scrollTop = log.scrollHeight;
  }, [text]);

  function handleScroll() {
    const log = logRef.current;
    if (!log) {
      return;
    }
    following.current = isFollowingTail(log);
  }

  function handleToggle() {
    // Closed <details> content has no layout, so scrollTop cannot be set while it is
    // hidden and the effect above no-ops for every line that arrives meanwhile. Catch up
    // on open, which is also the moment a stale pinned offset is least worth keeping.
    if (!detailsRef.current?.open) {
      return;
    }
    following.current = true;
    const log = logRef.current;
    if (log) {
      log.scrollTop = log.scrollHeight;
    }
  }

  if (lines.length === 0) {
    return null;
  }

  return (
    <details
      ref={detailsRef}
      onToggle={handleToggle}
      className="group mt-2 w-full max-w-sm text-left"
    >
      <summary className="mx-auto flex w-fit cursor-pointer list-none items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        <span className="group-open:hidden">Show {label}</span>
        <span className="hidden group-open:inline">Hide {label}</span>
        <HugeiconsIcon
          icon={ChevronDownIcon}
          aria-hidden="true"
          strokeWidth={1.5}
          className="size-[13px] shrink-0 transition-transform group-open:rotate-180"
        />
      </summary>
      <pre
        ref={logRef}
        onScroll={handleScroll}
        className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground"
      >
        {text}
      </pre>
    </details>
  );
}
