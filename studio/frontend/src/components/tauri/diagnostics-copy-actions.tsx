// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";

import { useState } from "react";

/**
 * The action row every failure state ends in: copy the support report, plus whatever
 * recovery buttons that state offers, passed as children so the caller keeps ownership
 * of them. When the clipboard refuses — a headless session, a denied permission — the
 * report is rendered into a selectable textarea so the user can still hand it over.
 */
export function DiagnosticsCopyActions({
  onCopyDiagnostics,
  children,
}: {
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
  children: React.ReactNode;
}) {
  const [copying, setCopying] = useState(false);
  const [manualReport, setManualReport] = useState<string | null>(null);
  const [manualMessage, setManualMessage] = useState<string | null>(null);

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(result.error ?? "Clipboard copy failed. Select and copy the diagnostics below.");
      }
    } catch (error) {
      setManualReport(null);
      setManualMessage(`Diagnostics copy failed: ${String(error)}`);
    } finally {
      setCopying(false);
    }
  }

  return (
    <div className="mt-4 flex w-full flex-col items-center gap-3">
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button
          variant="muted"
          size="hero"
          onClick={() => void handleCopyDiagnostics()}
        >
          {copying ? "Copying..." : "Copy Diagnostics"}
        </Button>
        {children}
      </div>
      {manualMessage && (
        <p className="max-w-md text-center text-xs text-destructive">{manualMessage}</p>
      )}
      {manualReport && (
        // text-left, not the inherited centering: a diagnostics report is read line by
        // line, and both screens center everything else in their column.
        <textarea
          readOnly
          value={manualReport}
          onFocus={(event) => event.currentTarget.select()}
          className="h-32 w-full max-w-md resize-none rounded-lg border border-border/50 bg-muted/30 p-2 text-left font-mono text-ui-10 text-muted-foreground"
        />
      )}
    </div>
  );
}
