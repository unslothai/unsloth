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
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useEffect, useRef, useState } from "react";

interface NameDialogProps {
  open: boolean;
  title: string;
  submitLabel: string;
  initialValue: string;
  onSubmit: (name: string) => Promise<void>;
  onOpenChange: (open: boolean) => void;
}

/** One text field, used for New folder and every Rename. */
export function NameDialog({ open, title, onOpenChange, ...form }: NameDialogProps) {
  const [busy, setBusy] = useState(false);
  return (
    <Dialog open={open} onOpenChange={(next) => !busy && onOpenChange(next)}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        {/* Content unmounts on close, so every open starts from initialValue. */}
        <NameForm {...form} busy={busy} setBusy={setBusy} onClose={() => onOpenChange(false)} />
      </DialogContent>
    </Dialog>
  );
}

function NameForm({
  submitLabel,
  initialValue,
  onSubmit,
  busy,
  setBusy,
  onClose,
}: Pick<NameDialogProps, "submitLabel" | "initialValue" | "onSubmit"> & {
  busy: boolean;
  setBusy: (busy: boolean) => void;
  onClose: () => void;
}) {
  const [value, setValue] = useState(initialValue);
  const input = useRef<HTMLInputElement>(null);

  // Select the stem so a rename keeps its extension by default.
  useEffect(() => {
    const dot = initialValue.lastIndexOf(".");
    const frame = requestAnimationFrame(() =>
      input.current?.setSelectionRange(0, dot > 0 ? dot : initialValue.length),
    );
    return () => cancelAnimationFrame(frame);
  }, [initialValue]);

  const trimmed = value.trim();
  async function submit() {
    if (!trimmed || busy) return;
    setBusy(true);
    try {
      await onSubmit(trimmed);
      onClose();
    } catch {
      // The caller toasted; keep the dialog open so the name can be fixed.
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <Input
        ref={input}
        autoFocus
        value={value}
        maxLength={255}
        onChange={(event) => setValue(event.target.value)}
        aria-label="Name"
      />
      <DialogFooter className="mt-5">
        <Button type="button" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button type="submit" disabled={!trimmed || busy}>
          {submitLabel}
        </Button>
      </DialogFooter>
    </form>
  );
}

export function ConfirmDeleteDialog({
  open,
  title,
  description,
  confirmLabel,
  onConfirm,
  onOpenChange,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  onConfirm: () => void;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** A note that could not be saved on the way out: try again, keep editing, or close without it. */
export function UnsavedChangesDialog({
  error,
  onRetry,
  onDiscard,
  onKeepEditing,
}: {
  /** Why the save failed; the dialog shows while this is set. */
  error: string | null;
  onRetry: () => void;
  onDiscard: () => void;
  onKeepEditing: () => void;
}) {
  const reason = error && !/[.!?]$/.test(error) ? `${error}.` : error;
  return (
    <AlertDialog open={error !== null} onOpenChange={(open) => !open && onKeepEditing()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Your changes weren't saved</AlertDialogTitle>
          <AlertDialogDescription>
            {reason} Try again, or discard your changes to close the file.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep editing</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onDiscard}>
            Discard changes
          </AlertDialogAction>
          <AlertDialogAction onClick={onRetry}>Try again</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
