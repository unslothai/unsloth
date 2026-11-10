// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";

/** Pill marking a recently shipped surface; drop the usage when it stops being new. */
export function NewBadge({ className }: { className?: string }): ReactElement {
  const t = useT();
  return (
    <Badge
      variant="secondary"
      className={cn(
        "h-4 px-1.5 text-ui-10 font-semibold uppercase tracking-[0.06em] text-primary",
        className,
      )}
    >
      {t("shell.navigation.newBadge")}
    </Badge>
  );
}
