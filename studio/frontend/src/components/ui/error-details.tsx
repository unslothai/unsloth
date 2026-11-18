// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Shared by inline errors and model-operation toasts. The complete diagnostic stays selectable. */
export function ErrorDetails({ message, onToggle }: { message: string; onToggle?: (open: boolean) => void }) {
  return (
    <details className="min-w-0 max-w-full" onToggle={(event) => onToggle?.(event.currentTarget.open)}>
      <summary className="cursor-pointer text-xs font-medium">Technical details</summary>
      <pre tabIndex={0} aria-label="Error technical details"
        className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap [overflow-wrap:anywhere] rounded-lg bg-background/50 p-3 font-mono text-xs">{message}</pre>
    </details>
  );
}
