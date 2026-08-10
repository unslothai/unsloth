// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown as ChevronDownIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";

// A wrapped line lands a few pixels off the bottom, which is not scrolling away.
const TAIL_SLACK_PX = 24;

function isAtTail(pane: HTMLPreElement): boolean {
  return (
    pane.scrollHeight - pane.scrollTop - pane.clientHeight <= TAIL_SLACK_PX
  );
}

/** The install, repair and update log, following its newest line unless the reader scrolls up. */
export function SetupLogDetails({
  label,
  lines,
}: {
  label: string;
  lines: string[];
}) {
  if (lines.length === 0) {
    return null;
  }
  return <SetupLog label={label} lines={lines} />;
}

// Separate component so a retry, which empties the log and fills it again, discards this
// state with the <details> it mirrors instead of leaving the new one a stale `open`.
function SetupLog({ label, lines }: { label: string; lines: string[] }) {
  // State, not a ref, so the effect below re-runs once the pane is mounted.
  const [pane, setPane] = useState<HTMLPreElement | null>(null);
  const [open, setOpen] = useState(false);
  const [following, setFollowing] = useState(true);
  const text = lines.join("\n");

  // `open` is a dependency because a collapsed <details> has no layout to scroll, so a
  // log opened after its last line arrived has to be scrolled on the open itself.
  // biome-ignore lint/correctness/useExhaustiveDependencies: new output is what the pane has to follow, and it reaches the effect as rendered text
  useEffect(() => {
    if (!(pane && open && following)) {
      return;
    }
    pane.scrollTop = pane.scrollHeight;
  }, [pane, open, following, text]);

  return (
    <details
      className="group mt-2 w-full max-w-sm text-left"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="mx-auto flex w-fit cursor-pointer list-none items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        <span className="group-open:hidden">Show {label} details</span>
        <span className="hidden group-open:inline">Hide {label} details</span>
        <HugeiconsIcon
          icon={ChevronDownIcon}
          aria-hidden="true"
          strokeWidth={1.5}
          className="size-[13px] shrink-0 transition-transform group-open:rotate-180"
        />
      </summary>
      <pre
        ref={setPane}
        onScroll={(event) => setFollowing(isAtTail(event.currentTarget))}
        className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground"
      >
        {text}
      </pre>
    </details>
  );
}
