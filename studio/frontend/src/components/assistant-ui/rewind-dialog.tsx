// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui, useAuiState } from "@assistant-ui/react";
import { type FC, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  collectRewindTurnIds,
  defaultRewindMode,
  isThreadIncognito,
  type RewindMode,
  type RewindResult,
  rewindChangedCount,
  rewindChatThread,
  rewindPickerEntries,
  rewindSkippedCount,
  rewindTouchesConversation,
  rewindTouchesFiles,
  useRewindDialogStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";

type Translate = ReturnType<typeof useT>;

type DryRun =
  | { state: "loading" }
  | { state: "done"; result: RewindResult | null }
  | { state: "error"; message: string };

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

function toastRestoreResult(
  t: Translate,
  remoteId: string,
  result: RewindResult,
) {
  const changed = rewindChangedCount(result);
  const skipped = rewindSkippedCount(result) + result.errors.length;
  if (changed === 0 && skipped > 0) {
    // Every file was skipped: a failure, so the caller stops before moving the conversation.
    toast.error(t("rewind.restoreFailed"), {
      description: t("rewind.skippedCount", { count: skipped }),
    });
    return false;
  }
  const title =
    changed === 0
      ? t("rewind.nothingRestored")
      : changed === 1
        ? t("rewind.restoredOne")
        : t("rewind.restoredMany", { count: changed });
  const undoTurnId = result.undo_turn_id;
  toast.success(title, {
    ...(skipped > 0
      ? { description: t("rewind.skippedCount", { count: skipped }) }
      : {}),
    ...(undoTurnId
      ? {
          action: {
            label: t("rewind.undo"),
            onClick: () => {
              rewindChatThread(remoteId, {
                turn_ids: [undoTurnId],
                dry_run: false,
              }).then(
                (undo) => {
                  const skippedUndo =
                    rewindSkippedCount(undo) + undo.errors.length;
                  if (rewindChangedCount(undo) === 0) {
                    toast.error(t("rewind.undoFailed"), {
                      ...(skippedUndo > 0
                        ? {
                            description: t("rewind.skippedCount", {
                              count: skippedUndo,
                            }),
                          }
                        : {}),
                    });
                  } else {
                    toast.success(t("rewind.undone"), {
                      ...(skippedUndo > 0
                        ? {
                            description: t("rewind.skippedCount", {
                              count: skippedUndo,
                            }),
                          }
                        : {}),
                    });
                  }
                },
                (error: unknown) =>
                  toast.error(t("rewind.undoFailed"), {
                    description: errorMessage(error),
                  }),
              );
            },
          },
        }
      : {}),
  });
  return true;
}

const FileList: FC<{ label: string; files: string[] }> = ({ label, files }) =>
  files.length === 0 ? null : (
    <div className="grid gap-1">
      <p className="text-muted-foreground text-xs">{label}</p>
      <ul className="max-h-24 overflow-y-auto font-mono text-xs">
        {files.map((file) => (
          <li key={file} className="truncate" title={file}>
            {file}
          </li>
        ))}
      </ul>
    </div>
  );

/** Mounted once per Thread; handles requests addressed to this thread's list item. */
export const RewindDialog: FC = () => {
  const threadKey = useAuiState(({ threadListItem }) => threadListItem.id);
  const request = useRewindDialogStore((s) => s.request);
  const closeRewind = useRewindDialogStore((s) => s.closeRewind);
  // Unmounts on close, so each request starts from a fresh read of the branch.
  return request !== null && request.threadKey === threadKey ? (
    <RewindFlow
      key={request.messageId ?? ""}
      initialMessageId={request.messageId}
      onClose={closeRewind}
    />
  ) : null;
};

const RewindFlow: FC<{
  initialMessageId: string | null;
  onClose: () => void;
}> = ({ initialMessageId, onClose }) => {
  const aui = useAui();
  const [messageId, setMessageId] = useState(initialMessageId);
  // Read once: subscribing would re-render this dialog on every streamed token.
  const branch = useMemo(() => aui.thread().getState().messages, [aui]);
  const entries = useMemo(() => rewindPickerEntries(branch), [branch]);

  if (messageId === null) {
    return (
      <RewindPicker
        entries={entries}
        onPick={setMessageId}
        onClose={onClose}
      />
    );
  }
  return (
    <RewindChoices
      key={messageId}
      messageId={messageId}
      preview={entries.find((entry) => entry.id === messageId)?.text ?? ""}
      turnIds={collectRewindTurnIds(branch, messageId)}
      onBack={initialMessageId === null ? () => setMessageId(null) : undefined}
      onClose={onClose}
    />
  );
};

const RewindPicker: FC<{
  entries: { id: string; preview: string; text: string }[];
  onPick: (id: string) => void;
  onClose: () => void;
}> = ({ entries, onPick, onClose }) => {
  const t = useT();
  return (
    <Dialog open={true} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-ui-21">{t("rewind.title")}</DialogTitle>
          <DialogDescription>{t("rewind.pickDescription")}</DialogDescription>
        </DialogHeader>
        {entries.length === 0 ? (
          <p className="text-muted-foreground">{t("rewind.pickEmpty")}</p>
        ) : (
          <ul className="grid max-h-80 gap-1 overflow-y-auto">
            {entries.map((entry) => (
              <li key={entry.id}>
                <button
                  type="button"
                  onClick={() => onPick(entry.id)}
                  className="w-full truncate rounded-md px-3 py-2 text-left hover:bg-accent focus-visible:bg-accent outline-none"
                >
                  {entry.preview || (
                    <span className="text-muted-foreground">
                      {t("rewind.attachmentOnly")}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </DialogContent>
    </Dialog>
  );
};

const RewindChoices: FC<{
  messageId: string;
  preview: string;
  turnIds: string[];
  onBack?: () => void;
  onClose: () => void;
}> = ({ messageId, preview, turnIds, onBack, onClose }) => {
  const t = useT();
  const aui = useAui();
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  // A temporary chat keeps no file history (and has no row the route can find).
  const tracked = Boolean(remoteId) && !isThreadIncognito(remoteId ?? "");
  const [dryRun, setDryRun] = useState<DryRun>(
    tracked && turnIds.length > 0
      ? { state: "loading" }
      : { state: "done", result: null },
  );
  const [mode, setMode] = useState<RewindMode>("conversation");
  const [busy, setBusy] = useState(false);
  const turnKey = turnIds.join("\n");

  useEffect(() => {
    if (!remoteId || !tracked || turnKey === "") return;
    let cancelled = false;
    rewindChatThread(remoteId, { turn_ids: turnKey.split("\n"), dry_run: true }).then(
      (result) => {
        if (cancelled) return;
        setDryRun({ state: "done", result });
        setMode(defaultRewindMode(rewindChangedCount(result) > 0));
      },
      (error: unknown) => {
        if (!cancelled) setDryRun({ state: "error", message: errorMessage(error) });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [remoteId, tracked, turnKey]);

  const result = dryRun.state === "done" ? dryRun.result : null;
  // Rows can exist with nothing left to put back (right after a rewind): that is no change.
  const hasChanges = result !== null && rewindChangedCount(result) > 0;

  const rewind = async (chosen: RewindMode) => {
    if (aui.thread().getState().isRunning) {
      toast.error(t("rewind.waitForResponse"));
      onClose();
      return;
    }
    if (rewindTouchesFiles(chosen) && remoteId) {
      setBusy(true);
      try {
        const restored = await rewindChatThread(remoteId, {
          turn_ids: turnIds,
          dry_run: false,
        });
        if (!toastRestoreResult(t, remoteId, restored)) {
          setBusy(false);
          return;
        }
      } catch (error) {
        setBusy(false);
        toast.error(t("rewind.restoreFailed"), {
          description: errorMessage(error),
        });
        return;
      }
    }
    if (rewindTouchesConversation(chosen)) {
      try {
        aui.thread().message({ id: messageId }).composer().beginEdit();
      } catch {
        // The message left the branch while the dialog was open.
      }
    }
    onClose();
  };

  // Only the conversation can go back: no choice to make, so no dialog.
  const conversationOnly = dryRun.state !== "loading" && !hasChanges;
  // biome-ignore lint/correctness/useExhaustiveDependencies: fires once, when the check settles.
  useEffect(() => {
    if (!conversationOnly) return;
    if (dryRun.state === "error") {
      toast.error(t("rewind.checkFailed", { error: dryRun.message }));
    }
    // Files changed since, but none can be put back: say so, or the rewind reads as a full restore.
    const skipped = result ? result.conflicts.length + result.untracked.length : 0;
    if (skipped > 0) {
      toast.warning(t("rewind.filesNotRestored", { count: skipped }));
    }
    void rewind("conversation");
  }, [conversationOnly]);

  if (!hasChanges || result === null) return null;

  const options: { value: RewindMode; label: string }[] = [
    { value: "both", label: t("rewind.modeBoth") },
    { value: "conversation", label: t("rewind.modeConversation") },
    { value: "files", label: t("rewind.modeFiles") },
  ];

  return (
    <Dialog open={true} onOpenChange={(next) => !next && !busy && onClose()}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-4 sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-ui-21">
            {t("rewind.confirmTitle")}
          </DialogTitle>
        </DialogHeader>
        <p className="line-clamp-3 rounded-lg bg-muted px-4 py-3 text-sm wrap-break-word">
          {preview || (
            <span className="text-muted-foreground">
              {t("rewind.attachmentOnly")}
            </span>
          )}
        </p>
        <div className="grid gap-2">
          <Label htmlFor="rewind-restore">{t("rewind.restoreLabel")}</Label>
          <Select
            value={mode}
            onValueChange={(value) => setMode(value as RewindMode)}
            disabled={busy}
          >
            <SelectTrigger id="rewind-restore" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2 text-sm">
          <p>
            {t("rewind.summary", {
              restore: result.restored.length,
              remove: result.deleted.length,
              recreate: result.recreated.length,
            })}
          </p>
          <FileList label={t("rewind.conflicts")} files={result.conflicts} />
          <FileList label={t("rewind.untracked")} files={result.untracked} />
          <DialogDescription className="text-xs">
            {t("rewind.scopeNote")}
          </DialogDescription>
        </div>
        <DialogFooter>
          {onBack ? (
            <Button variant="ghost" onClick={onBack} disabled={busy}>
              {t("rewind.back")}
            </Button>
          ) : null}
          <Button variant="outline" onClick={onClose} disabled={busy}>
            {t("rewind.cancel")}
          </Button>
          <Button onClick={() => void rewind(mode)} disabled={busy}>
            {t("rewind.confirm")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
