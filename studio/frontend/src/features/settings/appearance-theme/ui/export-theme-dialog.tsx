// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useId, useState } from "react";

import { exportThemeFile } from "../file";
import { META_LIMITS } from "../domain/schema";

function ExportThemeForm({
  busy,
  setBusy,
  onClose,
}: {
  busy: boolean;
  setBusy: (busy: boolean) => void;
  onClose: () => void;
}) {
  const t = useT();
  const id = useId();
  const [name, setName] = useState("");
  const [author, setAuthor] = useState("");
  const [includeLayout, setIncludeLayout] = useState(false);
  const [includeAssets, setIncludeAssets] = useState(false);

  async function download() {
    if (busy) return;
    setBusy(true);
    try {
      const result = await exportThemeFile({
        name:
          name.trim() || t("settings.appearance.themeFile.exportNamePlaceholder"),
        author: author.trim() || undefined,
        includeLayout,
        includeAssets,
      });
      // Cancelled leaves the dialog open, with what the user typed.
      if (result.status === "cancelled") return;
      onClose();
      toast.success(
        t("settings.appearance.themeFile.exported", { name: result.fileName }),
      );
      if (result.backgroundOmitted) {
        toast.warning(
          t("settings.appearance.themeFile.exportBackgroundOmitted"),
        );
      }
    } catch {
      toast.error(t("settings.appearance.themeFile.exportFailed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="flex flex-col gap-6"
      onSubmit={(event) => {
        event.preventDefault();
        void download();
      }}
    >
      <DialogHeader>
        <DialogTitle>{t("settings.appearance.themeFile.exportTitle")}</DialogTitle>
        <DialogDescription>
          {t("settings.appearance.themeFile.exportDescription")}
        </DialogDescription>
      </DialogHeader>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`${id}-name`}>
            {t("settings.appearance.themeFile.exportName")}
          </Label>
          <Input
            id={`${id}-name`}
            autoFocus={true}
            value={name}
            maxLength={META_LIMITS.name}
            placeholder={t("settings.appearance.themeFile.exportNamePlaceholder")}
            disabled={busy}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`${id}-author`}>
            {t("settings.appearance.themeFile.exportAuthor")}
          </Label>
          <Input
            id={`${id}-author`}
            value={author}
            maxLength={META_LIMITS.author}
            disabled={busy}
            onChange={(event) => setAuthor(event.target.value)}
          />
        </div>
        <div className="flex items-center justify-between gap-3">
          <Label htmlFor={`${id}-layout`}>
            {t("settings.appearance.themeFile.exportIncludeLayout")}
          </Label>
          <Switch
            id={`${id}-layout`}
            checked={includeLayout}
            onCheckedChange={setIncludeLayout}
            disabled={busy}
          />
        </div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between gap-3">
            <Label htmlFor={`${id}-assets`}>
              {t("settings.appearance.themeFile.exportIncludeAssets")}
            </Label>
            <Switch
              id={`${id}-assets`}
              aria-describedby={`${id}-assets-note`}
              checked={includeAssets}
              onCheckedChange={setIncludeAssets}
              disabled={busy}
            />
          </div>
          <p id={`${id}-assets-note`} className="text-xs text-muted-foreground">
            {t("settings.appearance.themeFile.exportIncludeAssetsNote")}
          </p>
        </div>
      </div>

      <DialogFooter>
        <Button type="button" variant="outline" disabled={busy} onClick={onClose}>
          {t("common.cancel")}
        </Button>
        <Button type="submit" variant="dark" disabled={busy}>
          {t("settings.appearance.themeFile.exportAction")}
        </Button>
      </DialogFooter>
    </form>
  );
}

export function ExportThemeDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (busy && !next) return;
        onOpenChange(next);
      }}
    >
      <DialogContent showCloseButton={!busy}>
        <ExportThemeForm
          busy={busy}
          setBusy={setBusy}
          onClose={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
