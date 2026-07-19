// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import type { LlamaApplyTarget } from "@/hooks/use-llama-update-check";
import { useCallback, useRef, useState, type ReactElement } from "react";

interface LlamaUpdateConfirmGate {
  /**
   * Open the prompt for `target`; resolve true only on explicit accept, false on
   * cancel/dismiss. Pass as the `apply()` gate so the destructive swap never runs
   * without a visible, accepted build + target host.
   */
  requestConfirm: (target: LlamaApplyTarget) => Promise<boolean>;
  /** Render once in the tree; inert until requestConfirm opens it. */
  dialog: ReactElement;
}

/**
 * Accept/cancel gate for the llama.cpp host-binary swap: shows the exact build
 * (from -> to) and target machine, resolving true only on explicit accept.
 */
export function useLlamaUpdateConfirmGate(): LlamaUpdateConfirmGate {
  const [target, setTarget] = useState<LlamaApplyTarget | null>(null);
  const resolveRef = useRef<((accepted: boolean) => void) | null>(null);

  const decide = useCallback((accepted: boolean) => {
    const resolve = resolveRef.current;
    resolveRef.current = null;
    setTarget(null);
    resolve?.(accepted);
  }, []);

  const requestConfirm = useCallback((next: LlamaApplyTarget) => {
    // A still-open prior prompt (e.g. a double click) resolves as declined.
    resolveRef.current?.(false);
    setTarget(next);
    return new Promise<boolean>((resolve) => {
      resolveRef.current = resolve;
    });
  }, []);

  const machineLabel = target?.machine?.hostname?.trim() || "this machine";

  const dialog = (
    <AlertDialog
      open={target !== null}
      onOpenChange={(open) => {
        if (!open) decide(false);
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Update llama.cpp?</AlertDialogTitle>
          <AlertDialogDescription>
            This downloads and swaps the llama.cpp binary on {machineLabel}. It
            replaces the running build, so only continue if you started this
            update.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="min-w-0 rounded-[12px] bg-muted/50 px-3 py-2.5 text-left">
          <p className="text-[13px] font-medium text-foreground">
            {target?.fromTag ?? "unknown"} &rarr;{" "}
            <span className="text-foreground">
              {target?.toTag ?? "the latest build"}
            </span>
          </p>
          {target?.machine?.hostname && (
            <p className="mt-0.5 break-all text-[11.5px] leading-[16px] text-muted-foreground">
              {target.machine.hostname}
              {target.machine.platform ? ` · ${target.machine.platform}` : ""}
            </p>
          )}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => decide(false)}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => decide(true)}>
            Update
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  return { requestConfirm, dialog };
}
