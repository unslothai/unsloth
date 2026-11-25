// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";

export function ErrorDetails({
  message,
  onToggle,
}: { message: string; onToggle?: (open: boolean) => void }) {
  const t = useT();
  const label = t("common.technicalDetails");
  return (
    <details
      className="min-w-0 max-w-full"
      onToggle={(event) => onToggle?.(event.currentTarget.open)}
    >
      <summary className="cursor-pointer font-medium text-xs">{label}</summary>
      <pre
        tabIndex={0}
        aria-label={label}
        className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-background/50 p-3 font-mono text-xs [overflow-wrap:anywhere]"
      >
        {message}
      </pre>
    </details>
  );
}
