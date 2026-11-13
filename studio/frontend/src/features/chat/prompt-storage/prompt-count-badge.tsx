// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";

export function PromptCountBadge({ count }: { count: number }): ReactElement {
  return (
    <span
      className="ml-auto flex h-4 min-w-4 shrink-0 items-center justify-center rounded-full bg-muted px-1 text-ui-11 font-medium tabular-nums text-muted-foreground"
      title={`${count} prompts`}
      aria-label={`${count} prompts`}
    >
      {count}
    </span>
  );
}
