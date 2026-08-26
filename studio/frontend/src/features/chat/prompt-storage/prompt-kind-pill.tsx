// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";

// Both kinds share one submenu but behave very differently on select: a prompt
// fills the composer, a list queues and sends N prompts.
export function PromptKindPill({
  kind,
  count,
}: {
  kind: "prompt" | "list";
  count?: number;
}): ReactElement {
  return (
    <span className="ml-auto shrink-0 rounded-full bg-muted px-1.5 py-px text-ui-11 font-medium text-muted-foreground">
      {kind === "list" ? `List · ${count ?? 0}` : "Prompt"}
    </span>
  );
}
