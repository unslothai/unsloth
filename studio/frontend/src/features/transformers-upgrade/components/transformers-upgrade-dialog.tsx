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
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { PackageIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";

/** Last path segment of the model id, for display. */
function modelDisplayName(modelName: string | null): string {
  if (!modelName) return "This model";
  return modelName.split("/").pop() || modelName;
}

/** App-wide consent dialog for models whose architecture needs a newer transformers
 *  than any installed one (validate: requires_transformers_upgrade). On Install it
 *  runs the sidecar install itself and resolves the paused load on success, so the
 *  load continues automatically. Mounted once in the root layout, next to the
 *  remote-code consent dialog. */
export function TransformersUpgradeDialog() {
  const open = useTransformersUpgradeDialogStore((s) => s.open);
  const modelName = useTransformersUpgradeDialogStore((s) => s.modelName);
  const upgrade = useTransformersUpgradeDialogStore((s) => s.upgrade);
  const phase = useTransformersUpgradeDialogStore((s) => s.phase);
  const errorMessage = useTransformersUpgradeDialogStore((s) => s.errorMessage);
  const install = useTransformersUpgradeDialogStore((s) => s.install);
  const resolve = useTransformersUpgradeDialogStore((s) => s.resolve);

  const displayName = modelDisplayName(modelName);
  const modelType = upgrade?.model_type ?? "unknown";
  const version = upgrade?.pypi_version ?? null;
  // Installable only from a released PyPI version; main-branch (dev) builds are
  // never offered, matching the backend's install-latest-transformers contract.
  const installable = Boolean(upgrade?.supported_in_pypi && version);
  const devOnly = !installable && Boolean(upgrade?.supported_in_main);
  const installing = phase === "installing";

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        // The dialog stays up while the install runs; Cancel is disabled below
        // and an Escape/overlay dismiss must not abandon an in-flight install.
        if (!next && !installing) resolve(false);
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
                <AlertDialogTitle>New model architecture</AlertDialogTitle>
                <AlertDialogDescription>
                  <span className="font-medium text-foreground">
                    {displayName}
                  </span>{" "}
                  uses the{" "}
                  <span className="font-mono text-foreground">{modelType}</span>{" "}
                  architecture, which your installed transformers does not
                  support yet.{" "}
                  {installable ? (
                    <>
                      Install transformers{" "}
                      <span className="font-medium text-foreground">
                        {version}
                      </span>{" "}
                      from PyPI to load it. The install runs once and can take
                      a minute; loading continues automatically afterwards.
                    </>
                  ) : devOnly ? (
                    <>
                      It is currently only available on the transformers
                      development branch (main). Studio does not install
                      development builds; support arrives with the next
                      transformers release on PyPI.
                    </>
                  ) : (
                    <>
                      No released transformers version supports it yet, so it
                      cannot be loaded.
                    </>
                  )}
                </AlertDialogDescription>
              </div>

              {phase === "error" && errorMessage ? (
                <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">
                  {errorMessage}
                </p>
              ) : null}

              {installing ? (
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Spinner className="size-3.5" />
                  Installing transformers {version}... This can take a minute.
                </p>
              ) : null}
            </div>
          </div>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={installing}>Cancel</AlertDialogCancel>
          {installable ? (
            <AlertDialogAction
              disabled={installing}
              className={cn(installing && "pointer-events-none")}
              onClick={(event) => {
                // Keep the dialog open while the install runs; the store closes
                // it (and resumes the paused load) on success.
                event.preventDefault();
                void install();
              }}
            >
              {installing ? (
                <>
                  <Spinner className="size-4" />
                  Installing...
                </>
              ) : phase === "error" ? (
                "Retry install"
              ) : (
                `Install transformers ${version}`
              )}
            </AlertDialogAction>
          ) : null}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
