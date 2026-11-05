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
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { ImageUpload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPersona, removePersonaAvatar, updatePersona, uploadPersonaAvatar } from "../api";
import { useCharactersStore } from "../store";
import type { Persona } from "../types";
import { Portrait } from "./portrait";

type PersonaDialogProps = {
  open: boolean;
  persona: Persona | null;
  onClose: () => void;
};

export function PersonaDialog({ open, persona, onClose }: PersonaDialogProps) {
  const t = useT();
  const [name, setName] = useState("");
  const [about, setAbout] = useState("");
  const [isDefault, setIsDefault] = useState(false);
  const [picture, setPicture] = useState<File | null | "remove">(null);
  const [saving, setSaving] = useState(false);
  // A persona this dialog created, kept so a retry after a failed portrait upload updates it.
  const [created, setCreated] = useState<Persona | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const previewUrl = useMemo(
    () => (picture instanceof File ? URL.createObjectURL(picture) : null),
    [picture],
  );
  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);
  useEffect(() => {
    if (!open) return;
    setName(persona?.name ?? "");
    setAbout(persona?.description ?? "");
    setIsDefault(persona?.isDefault ?? false);
    setPicture(null);
    setCreated(null);
  }, [open, persona]);

  async function save() {
    if (!name.trim()) return;
    setSaving(true);
    try {
      const existing = persona ?? created;
      let saved = existing
        ? await updatePersona(existing.id, { name: name.trim(), description: about, ...(isDefault ? { isDefault: true } : {}) })
        : await createPersona({ name: name.trim(), description: about, isDefault });
      if (!existing) {
        setCreated(saved);
        useCharactersStore.getState().upsertPersona(saved);
      }
      if (picture instanceof File) saved = await uploadPersonaAvatar(saved.id, picture);
      else if (picture === "remove" && saved.hasAvatar) saved = await removePersonaAvatar(saved.id);
      useCharactersStore.getState().upsertPersona(saved);
      if (saved.isDefault) await useCharactersStore.getState().refreshPersonas().catch(() => undefined);
      onClose();
    } catch (error) {
      toast.error(t("characters.personas.saveFailed"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setSaving(false);
    }
  }

  const showImage = picture === "remove" ? false : Boolean(previewUrl) || Boolean(persona?.hasAvatar);
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{persona ? t("characters.personas.editTitle") : t("characters.personas.newTitle")}</DialogTitle>
          <DialogDescription>{t("characters.personas.description")}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="flex items-center gap-4">
            <Portrait
              kind="persona"
              id={persona?.id}
              name={name || "?"}
              hasImage={showImage}
              rev={persona?.avatarRev ?? 0}
              imageUrl={previewUrl ?? (picture === "remove" ? null : undefined)}
              className="size-16 rounded-2xl text-[1.6rem]"
            />
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="outline" size="sm" onClick={() => fileRef.current?.click()}>
                <HugeiconsIcon icon={ImageUpload01Icon} strokeWidth={1.75} className="size-icon" />
                {showImage ? t("characters.editor.changePicture") : t("characters.editor.uploadPicture")}
              </Button>
              {showImage ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => setPicture("remove")}>
                  {t("characters.editor.removePicture")}
                </Button>
              ) : null}
              <input
                ref={fileRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) setPicture(file);
                  event.target.value = "";
                }}
              />
            </div>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="persona-name">{t("characters.personas.name")}</Label>
            <Input
              id="persona-name"
              value={name}
              maxLength={200}
              autoFocus={true}
              placeholder={t("characters.personas.namePlaceholder")}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="persona-about">{t("characters.personas.about")}</Label>
            <Textarea
              id="persona-about"
              value={about}
              rows={5}
              placeholder={t("characters.personas.aboutHint")}
              onChange={(event) => setAbout(event.target.value)}
            />
          </div>
          <label className="flex items-center justify-between gap-3 text-ui-14">
            <span>{t("characters.personas.makeDefault")}</span>
            <Switch checked={isDefault} onCheckedChange={setIsDefault} disabled={persona?.isDefault} />
          </label>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={onClose}>
              {t("characters.cancel")}
            </Button>
            <Button type="submit" variant="dark" disabled={!name.trim() || saving}>
              {t("characters.personas.save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
