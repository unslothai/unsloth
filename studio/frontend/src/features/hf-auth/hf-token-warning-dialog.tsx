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
import { AlertTriangle } from "lucide-react";
import { useHfTokenWarningStore } from "./store";

export function HfTokenWarningDialog() {
  const open = useHfTokenWarningStore((state) => state.open);
  const resolve = useHfTokenWarningStore((state) => state.resolve);

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) resolve("cancel");
      }}
    >
      <AlertDialogContent className="max-w-md">
        <AlertDialogHeader>
          <div className="flex items-start gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400">
              <AlertTriangle className="size-5" />
            </div>
            <div className="space-y-1 text-left">
              <AlertDialogTitle>Hugging Face token is invalid</AlertDialogTitle>
              <AlertDialogDescription>
                Hugging Face rejected the saved token. Replace it to access
                private or gated repositories, or continue without it for public
                and fully downloaded models.
              </AlertDialogDescription>
            </div>
          </div>
        </AlertDialogHeader>
        <AlertDialogFooter className="sm:justify-between">
          <AlertDialogCancel onClick={() => resolve("cancel")}>
            Cancel
          </AlertDialogCancel>
          <div className="flex flex-col-reverse gap-2 sm:flex-row">
            <Button variant="outline" onClick={() => resolve("anonymous")}>
              Continue without token
            </Button>
            <AlertDialogAction onClick={() => resolve("replace")}>
              Replace token
            </AlertDialogAction>
          </div>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
