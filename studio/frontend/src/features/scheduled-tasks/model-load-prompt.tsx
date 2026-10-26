// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useId, useState } from "react";
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
import { Checkbox } from "@/components/ui/checkbox";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import type { FinishedScheduledTaskRun } from "./api";
import { useScheduledTasksStore } from "./store";

function modelName(model: string): string {
  return model.split(/[\\/]/).filter(Boolean).pop() ?? model;
}

/** Asks whether a task may load its model, which replaces the one in Chat: once, or for good. */
export default function ModelLoadPrompt({
  run,
  onDone,
}: {
  run: FinishedScheduledTaskRun;
  onDone: () => void;
}) {
  const t = useT();
  const checkboxId = useId();
  const [always, setAlways] = useState(false);
  const model = modelName(run.taskModel ?? "");

  async function load() {
    onDone();
    const { update, runNow } = useScheduledTasksStore.getState();
    try {
      if (always) await update(run.taskId, { allowModelLoad: true });
      await runNow(run.taskId, true);
    } catch (error) {
      toast.error(t("scheduledTasks.errors.runFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  }

  return (
    <AlertDialog open={true} onOpenChange={(open) => !open && onDone()}>
      <AlertDialogContent onOverlayClick={onDone}>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {t("scheduledTasks.loadPrompt.title", { model })}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {t("scheduledTasks.loadPrompt.description", { title: run.taskTitle, model })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <label htmlFor={checkboxId} className="flex cursor-pointer items-start gap-2.5 text-ui-14">
          <Checkbox
            id={checkboxId}
            checked={always}
            onCheckedChange={(on) => setAlways(on === true)}
            className="mt-0.5"
          />
          {t("scheduledTasks.loadPrompt.always")}
        </label>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("scheduledTasks.loadPrompt.notNow")}</AlertDialogCancel>
          <AlertDialogAction onClick={() => void load()}>
            {t("scheduledTasks.loadPrompt.load")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
