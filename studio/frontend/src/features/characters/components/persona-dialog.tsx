// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { errorMessage } from "@/lib/error-message";
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
import { openFilePicker } from "@/lib/open-file-picker";
import { toast } from "@/lib/toast";
import { ImageUpload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useObjectUrl } from "@/hooks/use-object-url";
import { useEffect, useState } from "react";
import { createPersona, removePersonaAvatar, updatePersona, uploadPersonaAvatar } from "../api";
import { applyAvatarChoice, AVATAR_ACCEPT, avatarFits, type AvatarChoice } from "../avatar-upload";
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
  const [avatar, setAvatar] = useState<AvatarChoice>(null);
  const [saving, setSaving] = useState(false);
  // A persona this dialog created, kept so a retry after a failed portrait upload updates it.
  const [created, setCreated] = useState<Persona | null>(null);
  const previewUrl = useObjectUrl(avatar instanceof File ? avatar : null);
  useEffect(() => {
    if (!open) return;
    setName(persona?.name ?? "");
    setAbout(persona?.description ?? "");
    setIsDefault(persona?.isDefault ?? false);
    setAvatar(null);
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
      saved = await applyAvatarChoice(saved, avatar, uploadPersonaAvatar, removePersonaAvatar);
      useCharactersStore.getState().upsertPersona(saved);
      if (saved.isDefault) await useCharactersStore.getState().refreshPersonas().catch(() => undefined);
      onClose();
    } catch (error) {
      toast.error(t("characters.personas.saveFailed"), {
        description: errorMessage(error),
      });
    } finally {
      setSaving(false);
    }
  }

  const showAvatar = avatar === "remove" ? false : Boolean(previewUrl) || Boolean(persona?.hasAvatar);
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
              subject={{ id: persona?.id, name: name || "?", hasAvatar: showAvatar, avatarRev: persona?.avatarRev ?? 0 }}
              imageUrl={previewUrl ?? (avatar === "remove" ? null : undefined)}
              className="size-16 rounded-2xl text-[1.6rem]"
            />
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  openFilePicker(AVATAR_ACCEPT, ([file]) => {
                    if (file && avatarFits(file, t)) setAvatar(file);
                  })
                }
              >
                <HugeiconsIcon icon={ImageUpload01Icon} strokeWidth={1.75} className="size-icon" />
                {showAvatar ? t("characters.editor.changePicture") : t("characters.editor.uploadPicture")}
              </Button>
              {showAvatar ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => setAvatar("remove")}>
                  {t("characters.editor.removePicture")}
                </Button>
              ) : null}
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
              {t("common.cancel")}
            </Button>
            <Button type="submit" variant="dark" disabled={!name.trim() || saving}>
              {t("common.save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
