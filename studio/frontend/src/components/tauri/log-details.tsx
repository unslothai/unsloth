// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown as ChevronDownIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/**
 * The collapsed log tail every long-running full-window screen shows under its status
 * line. Shared so install, repair and update stay one control rather than three copies
 * that drift apart; only the noun in the toggle and the lines themselves differ.
 */
export function LogDetails({
  label,
  lines,
}: {
  /** Noun phrase completing "Show ..." / "Hide ...", e.g. "installation details". */
  label: string;
  lines: string[];
}) {
  if (lines.length === 0) {
    return null;
  }

  return (
    <details className="group mt-2 w-full max-w-sm text-left">
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
      <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground">
        {lines.join("\n")}
      </pre>
    </details>
  );
}
