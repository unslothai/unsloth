// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";

import {
  CUSTOM_SECTION_NAME_MAX,
  normalizeSectionName,
} from "../stores/sidebar-organization-store";

// Names a custom sidebar section: a new one, or an existing one being renamed. The caller owns
// what the name is for, so this only collects it.
export function SectionNameDialog({
  open,
  mode,
  initialName = "",
  onOpenChange,
  onSubmit,
}: {
  open: boolean;
  mode: "create" | "rename";
  initialName?: string;
  onOpenChange: (open: boolean) => void;
  onSubmit: (name: string) => void;
}) {
  // What the open dialog was opened for, kept through the close animation: the caller clears its
  // state on close, which would otherwise flip a rename to "New section" as it fades out.
  const [shown, setShown] = useState({ mode, initialName });
  if (open && (shown.mode !== mode || shown.initialName !== initialName)) {
    setShown({ mode, initialName });
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md">
        {/* Content unmounts on close, so each open starts from its own name, not the last draft. */}
        <SectionNameForm
          mode={shown.mode}
          initialName={shown.initialName}
          onCancel={() => onOpenChange(false)}
          onSubmit={(name) => {
            onSubmit(name);
            onOpenChange(false);
          }}
        />
      </DialogContent>
    </Dialog>
  );
}

function SectionNameForm({
  mode,
  initialName,
  onCancel,
  onSubmit,
}: {
  mode: "create" | "rename";
  initialName: string;
  onCancel: () => void;
  onSubmit: (name: string) => void;
}) {
  const t = useT();
  const [name, setName] = useState(initialName);
  const clean = normalizeSectionName(name);
  const unchanged = mode === "rename" && clean === normalizeSectionName(initialName);

  function submit() {
    if (!clean || unchanged) return;
    onSubmit(clean);
  }

  return (
    <>
      <DialogHeader>
        <DialogTitle className="text-ui-21">
          {mode === "create"
            ? t("shell.sections.createTitle")
            : t("shell.sections.renameTitle")}
        </DialogTitle>
        <DialogDescription>
          {mode === "create"
            ? t("shell.sections.createDescription")
            : t("shell.sections.renameDescription")}
        </DialogDescription>
      </DialogHeader>
      <input
        value={name}
        onChange={(event) => setName(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.nativeEvent.isComposing) {
            event.preventDefault();
            submit();
          }
        }}
        onFocus={(event) => {
          if (mode === "rename") event.currentTarget.select();
        }}
        autoFocus
        maxLength={CUSTOM_SECTION_NAME_MAX}
        placeholder={t("shell.sections.namePlaceholder")}
        aria-label={t("shell.sections.namePlaceholder")}
        className="w-full rounded-[14px] border border-border bg-background px-4 py-3 text-base outline-none transition-colors placeholder:text-muted-foreground focus:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus:border-ring"
      />
      <DialogFooter className="flex-wrap gap-2 sm:justify-end">
        <Button type="button" variant="ghost" onClick={onCancel}>
          {t("common.cancel")}
        </Button>
        <Button type="button" onClick={submit} disabled={!clean || unchanged}>
          {mode === "create" ? t("shell.sections.create") : t("common.save")}
        </Button>
      </DialogFooter>
    </>
  );
}
