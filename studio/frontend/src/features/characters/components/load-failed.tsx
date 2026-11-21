// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";

export function LoadFailed({
  onRetry,
  compact = false,
  className,
}: {
  onRetry: () => void;
  compact?: boolean;
  className?: string;
}) {
  const t = useT();
  return (
    <div
      className={cn(
        "flex flex-col items-center text-center text-muted-foreground",
        compact ? "gap-2 py-6 text-ui-14" : "mt-16 gap-3",
        className,
      )}
      role="alert"
    >
      {compact ? t("characters.loadFailed") : <p>{t("characters.loadFailed")}</p>}
      <Button variant="outline" size={compact ? "sm" : undefined} onClick={onRetry}>
        {t("characters.retry")}
      </Button>
    </div>
  );
}
