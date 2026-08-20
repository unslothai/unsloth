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
import { PackageIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useLlmCompressorConsentStore } from "../stores/llm-compressor-consent-store";

/** Consent dialog before Studio installs llm-compressor for FP8/FP4 export. */
export function LlmCompressorConsentDialog() {
  const open = useLlmCompressorConsentStore((s) => s.open);
  const probe = useLlmCompressorConsentStore((s) => s.probe);
  const resolve = useLlmCompressorConsentStore((s) => s.resolve);

  const blocked = Boolean(probe?.blocked_reason);
  const summary =
    probe?.install_summary ??
    "This export needs llm-compressor, which is not installed in this environment.";

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) resolve(false);
      }}
    >
      <AlertDialogContent className="max-w-lg">
        <AlertDialogHeader className="min-w-0">
          <div className="flex w-full min-w-0 items-start gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400">
              <HugeiconsIcon icon={PackageIcon} className="size-5" />
            </div>
            <div className="min-w-0 flex-1 space-y-3">
              <div className="space-y-1">
                <AlertDialogTitle>
                  {blocked ? "Cannot install llm-compressor" : "Install llm-compressor?"}
                </AlertDialogTitle>
                <AlertDialogDescription asChild>
                  <div className="space-y-2 text-sm text-muted-foreground">
                    {blocked ? (
                      <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">
                        {probe?.blocked_reason}
                      </p>
                    ) : (
                      <>
                        <p>{summary}</p>
                        {probe?.consent_kind === "workspace" ? (
                          <p className="rounded-md border bg-muted/40 px-3 py-2 font-mono text-xs text-foreground">
                            {probe.workspace_install_command}
                          </p>
                        ) : probe?.consent_kind === "shadow" ? (
                          <p className="text-xs">
                            Target:{" "}
                            <span className="font-mono text-foreground">
                              {probe.shadow_path}
                            </span>
                          </p>
                        ) : null}
                        <p className="text-xs">
                          Interpreter:{" "}
                          <span className="font-mono text-foreground">
                            {probe?.python_executable}
                          </span>
                        </p>
                      </>
                    )}
                  </div>
                </AlertDialogDescription>
              </div>
            </div>
          </div>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          {!blocked ? (
            <AlertDialogAction onClick={() => resolve(true)}>
              Install and continue
            </AlertDialogAction>
          ) : null}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
