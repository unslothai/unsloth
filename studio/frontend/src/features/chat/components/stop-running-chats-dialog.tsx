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
import { useStopRunningChatsDialogStore } from "../stores/stop-running-chats-dialog-store";

/**
 * Confirmation for applying a model or reload-required setting while chats are generating.
 * They share one llama-server, so the swap ends all of them: name them and make the user
 * opt in rather than truncating silently.
 */
export function StopRunningChatsDialog() {
  const open = useStopRunningChatsDialogStore((s) => s.open);
  const count = useStopRunningChatsDialogStore((s) => s.count);
  const titles = useStopRunningChatsDialogStore((s) => s.titles);
  const action = useStopRunningChatsDialogStore((s) => s.action);
  const resolve = useStopRunningChatsDialogStore((s) => s.resolve);

  const plural = count === 1 ? "chat" : "chats";
  const shown = titles.slice(0, 5);
  const remaining = Math.max(0, titles.length - shown.length);

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        // Escape / overlay click must resolve, or the caller's await hangs.
        if (!next) resolve(false);
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            Stop {count} running {plural}?
          </AlertDialogTitle>
          <AlertDialogDescription>
            {action ? `${action} reloads the model, ` : "Reloading the model "}
            which every open conversation shares, so{" "}
            {count === 1 ? "this" : "these"} {plural} will stop generating.
            Replies produced so far are kept.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {shown.length > 0 && (
          <ul className="max-h-40 overflow-y-auto rounded-md border bg-muted/40 px-3 py-2 text-sm">
            {shown.map((title) => (
              <li key={title} className="truncate py-0.5">
                {title}
              </li>
            ))}
            {remaining > 0 && (
              <li className="py-0.5 text-muted-foreground">
                and {remaining} more
              </li>
            )}
          </ul>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => resolve(false)}>
            Keep generating
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => resolve(true)}>
            Stop and reload
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
