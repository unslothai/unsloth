// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isFollowingTail } from "@/components/tauri/log-follow";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";

import { HugeiconsIcon } from "@hugeicons/react";
import { useLayoutEffect, useRef } from "react";

/** Shared log tail; scrolling up pins the view, scrolling back to the bottom resumes follow. */
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
  // A ref: re-rendering per scroll event while a log streams is too costly.
  const following = useRef(true);

  const text = lines.join("\n");

  // Layout effect so new lines never paint at the old offset first.
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
    // Closed <details> has no layout, so scrollTop cannot be set while hidden; catch up on open.
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
          icon={ChevronDownStandardIcon}
          aria-hidden="true"
          strokeWidth={1.5}
          className="size-[calc(13px*var(--ui-space-scale,1))] shrink-0 transition-transform group-open:rotate-180"
        />
      </summary>
      <pre
        ref={logRef}
        onScroll={handleScroll}
        className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words scroll-rounded rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground"
      >
        {text}
      </pre>
    </details>
  );
}
