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
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingTrustRemoteCodeDialogStore } from "../stores/training-trust-remote-code-dialog-store";

export function TrainingTrustRemoteCodeDialog() {
  const open = useTrainingTrustRemoteCodeDialogStore((state) => state.open);
  const resolve = useTrainingTrustRemoteCodeDialogStore(
    (state) => state.resolve,
  );
  const dialogModelName = useTrainingTrustRemoteCodeDialogStore(
    (state) => state.modelName,
  );
  const selectedModel = useTrainingConfigStore((state) => state.selectedModel);
  const modelName = dialogModelName || selectedModel;
  const displayName = modelName?.split("/").pop() || "This model";

  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) resolve(false);
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Enable custom code for training?</AlertDialogTitle>
          <AlertDialogDescription>
            {displayName} declares custom Python code in its repository.
            Continue only if you trust the model source.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={() => resolve(true)}>
            Enable and start
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
