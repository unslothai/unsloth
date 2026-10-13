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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  collectRewindTurnIds,
  defaultRewindMode,
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
                () => toast.success(t("rewind.undone")),
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

/** Mounted once per Thread; opens for requests addressed to this thread's list item. */
export const RewindDialog: FC = () => {
  const threadKey = useAuiState(({ threadListItem }) => threadListItem.id);
  const request = useRewindDialogStore((s) => s.request);
  const closeRewind = useRewindDialogStore((s) => s.closeRewind);
  const open = request !== null && request.threadKey === threadKey;
  return (
    <Dialog open={open} onOpenChange={(next) => !next && closeRewind()}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md">
        {/* Content unmounts on close, so each open starts from a fresh read of the branch. */}
        {open ? (
          <RewindDialogBody
            initialMessageId={request.messageId}
            onClose={closeRewind}
          />
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

const RewindDialogBody: FC<{
  initialMessageId: string | null;
  onClose: () => void;
}> = ({ initialMessageId, onClose }) => {
  const t = useT();
  const aui = useAui();
  const [messageId, setMessageId] = useState(initialMessageId);
  // Read once: subscribing would re-render this dialog on every streamed token.
  const branch = useMemo(() => aui.thread().getState().messages, [aui]);
  const entries = useMemo(() => rewindPickerEntries(branch), [branch]);

  if (messageId === null) {
    return (
      <>
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
                  onClick={() => setMessageId(entry.id)}
                  className="w-full truncate rounded-[12px] px-3 py-2 text-left hover:bg-accent focus-visible:bg-accent outline-none"
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
      </>
    );
  }
  return (
    <RewindChoices
      key={messageId}
      messageId={messageId}
      turnIds={collectRewindTurnIds(branch, messageId)}
      onBack={initialMessageId === null ? () => setMessageId(null) : undefined}
      onClose={onClose}
    />
  );
};

const RewindChoices: FC<{
  messageId: string;
  turnIds: string[];
  onBack?: () => void;
  onClose: () => void;
}> = ({ messageId, turnIds, onBack, onClose }) => {
  const t = useT();
  const aui = useAui();
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const [dryRun, setDryRun] = useState<DryRun>(
    remoteId && turnIds.length > 0
      ? { state: "loading" }
      : { state: "done", result: null },
  );
  const [mode, setMode] = useState<RewindMode>("conversation");
  const [busy, setBusy] = useState(false);
  const turnKey = turnIds.join("\n");

  useEffect(() => {
    if (!remoteId || turnKey === "") return;
    let cancelled = false;
    rewindChatThread(remoteId, { turn_ids: turnKey.split("\n"), dry_run: true }).then(
      (result) => {
        if (cancelled) return;
        setDryRun({ state: "done", result });
        setMode(defaultRewindMode(result.has_changes));
      },
      (error: unknown) => {
        if (!cancelled) setDryRun({ state: "error", message: errorMessage(error) });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [remoteId, turnKey]);

  const result = dryRun.state === "done" ? dryRun.result : null;
  const hasChanges = Boolean(result?.has_changes);

  const confirm = async () => {
    if (aui.thread().getState().isRunning) {
      toast.error(t("rewind.waitForResponse"));
      return;
    }
    if (rewindTouchesFiles(mode) && remoteId) {
      setBusy(true);
      try {
        const restored = await rewindChatThread(remoteId, {
          turn_ids: turnIds,
          dry_run: false,
        });
        toastRestoreResult(t, remoteId, restored);
      } catch (error) {
        setBusy(false);
        toast.error(t("rewind.restoreFailed"), {
          description: errorMessage(error),
        });
        return;
      }
    }
    if (rewindTouchesConversation(mode)) {
      try {
        aui.thread().message({ id: messageId }).composer().beginEdit();
      } catch {
        // The message left the branch while the dialog was open.
      }
    }
    onClose();
  };

  const options: { value: RewindMode; label: string; files: boolean }[] = [
    { value: "both", label: t("rewind.modeBoth"), files: true },
    { value: "conversation", label: t("rewind.modeConversation"), files: false },
    { value: "files", label: t("rewind.modeFiles"), files: true },
  ];

  return (
    <>
      <DialogHeader>
        <DialogTitle className="text-ui-21">{t("rewind.title")}</DialogTitle>
        <DialogDescription>{t("rewind.chooseDescription")}</DialogDescription>
      </DialogHeader>
      <RadioGroup
        value={mode}
        onValueChange={(value) => setMode(value as RewindMode)}
        disabled={busy}
      >
        {options.map((option) => (
          <label
            key={option.value}
            className="flex cursor-pointer items-center gap-3 has-[button:disabled]:cursor-not-allowed has-[button:disabled]:opacity-50"
          >
            <RadioGroupItem
              value={option.value}
              disabled={option.files && !hasChanges}
            />
            {option.label}
          </label>
        ))}
      </RadioGroup>
      <div className="grid gap-2 text-sm">
        {dryRun.state === "loading" ? (
          <p className="text-muted-foreground">{t("rewind.checking")}</p>
        ) : dryRun.state === "error" ? (
          <p className="text-destructive">
            {t("rewind.checkFailed", { error: dryRun.message })}
          </p>
        ) : !hasChanges || !result ? (
          <p className="text-muted-foreground">{t("rewind.noFileChanges")}</p>
        ) : (
          <>
            <p>
              {t("rewind.summary", {
                restore: result.restored.length,
                remove: result.deleted.length,
                recreate: result.recreated.length,
              })}
            </p>
            <FileList label={t("rewind.conflicts")} files={result.conflicts} />
            <FileList label={t("rewind.untracked")} files={result.untracked} />
          </>
        )}
        <p className="text-muted-foreground text-xs">{t("rewind.scopeNote")}</p>
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
        <Button
          onClick={() => void confirm()}
          disabled={busy || dryRun.state === "loading"}
        >
          {t("rewind.confirm")}
        </Button>
      </DialogFooter>
    </>
  );
};
