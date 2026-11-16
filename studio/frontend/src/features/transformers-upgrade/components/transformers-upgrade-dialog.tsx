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
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, PackageIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  upgradeDialogActions,
  upgradeInstallVersion,
} from "../lib/upgrade-dialog-actions";
import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";

function modelDisplayName(modelName: string | null): string {
  if (!modelName) return "This model";
  return modelName.split("/").pop() || modelName;
}

export function TransformersUpgradeDialog() {
  const open = useTransformersUpgradeDialogStore((s) => s.open);
  const modelName = useTransformersUpgradeDialogStore((s) => s.modelName);
  const upgrade = useTransformersUpgradeDialogStore((s) => s.upgrade);
  const phase = useTransformersUpgradeDialogStore((s) => s.phase);
  const errorMessage = useTransformersUpgradeDialogStore((s) => s.errorMessage);
  const trustRemoteCodeFallback = useTransformersUpgradeDialogStore(
    (s) => s.trustRemoteCodeFallback,
  );
  const install = useTransformersUpgradeDialogStore((s) => s.install);
  const resolve = useTransformersUpgradeDialogStore((s) => s.resolve);

  const displayName = modelDisplayName(modelName);
  const modelType = upgrade?.model_type ?? "unknown";
  const version = upgradeInstallVersion(upgrade);
  const { installable, fromMain, devOnly, customCode } = upgradeDialogActions({
    upgrade,
    phase,
    trustRemoteCodeFallback,
  });
  const installing = phase === "installing";

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        // Escape/overlay dismiss must not abandon an in-flight install.
        if (!next && !installing) resolve(false);
      }}
    >
      {/* A plain max-w loses to the size rule, whose 28rem can't fit a dev version's button row. */}
      <AlertDialogContent className="sm:!max-w-[calc(34rem*var(--ui-space-scale,1))]">
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
                  uses{" "}
                  <span className="font-mono text-foreground">{modelType}</span>
                  {installable ? (
                    <>
                      , which needs transformers{" "}
                      <span className="font-medium text-foreground">
                        {version}
                      </span>
                      {fromMain ? " (development build)" : ""}. Updating takes
                      about a minute.
                    </>
                  ) : devOnly ? (
                    <>
                      , which only the transformers development build supports.
                      Its version can't be checked right now; try again later.
                    </>
                  ) : (
                    <>, which no transformers version supports yet.</>
                  )}
                  {customCode && !installable
                    ? " You can still load it with the model's own code."
                    : null}
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
                  Updating transformers to {version}...
                </p>
              ) : null}
            </div>
          </div>
        </AlertDialogHeader>

        <AlertDialogFooter className="sm:flex-wrap">
          <AlertDialogCancel disabled={installing}>Cancel</AlertDialogCancel>
          {installable ? (
            <>
              {customCode ? (
                // Installing forces 16-bit, so this is a 4-bit run's only path; the code scan follows.
                <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
                    <Button variant="outline">
                      Other
                      <HugeiconsIcon
                        icon={ArrowDown01Icon}
                        className="size-4"
                      />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-auto">
                    <DropdownMenuItem onSelect={() => resolve(true)}>
                      Continue with custom code
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : null}
              <AlertDialogAction
                disabled={installing}
                className={cn(installing && "pointer-events-none")}
                onClick={(event) => {
                  // Keep the dialog open; the store closes it on success.
                  event.preventDefault();
                  void install();
                }}
              >
                {installing ? (
                  <>
                    <Spinner className="size-4" />
                    Updating...
                  </>
                ) : phase === "error" ? (
                  "Retry update"
                ) : (
                  `Update transformers to ${version}`
                )}
              </AlertDialogAction>
            </>
          ) : customCode ? (
            <AlertDialogAction onClick={() => resolve(true)}>
              Continue with custom code
            </AlertDialogAction>
          ) : null}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
