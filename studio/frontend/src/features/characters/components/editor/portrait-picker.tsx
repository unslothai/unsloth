// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useNativeFileDrop } from "@/features/native-intents";
import { useT } from "@/i18n";
import { openFilePicker } from "@/lib/open-file-picker";
import { cn } from "@/lib/utils";
import { ImageUpload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { PictureChoice } from "../../hooks/use-character-draft";
import { PICTURE_ACCEPT, pictureFits } from "../../images";
import type { Character } from "../../types";
import { Portrait } from "../portrait";

/** The editor's portrait: click or drop an image to replace it, or remove it. */
export function PortraitPicker({
  original,
  name,
  picture,
  previewUrl,
  onChange,
}: {
  original: Character | null;
  name: string;
  picture: PictureChoice;
  previewUrl: string | null;
  onChange: (picture: PictureChoice) => void;
}) {
  const t = useT();
  const choose = (file: File | undefined) => {
    if (file && pictureFits(file, t)) onChange(file);
  };
  // The desktop app included.
  const drop = useNativeFileDrop({
    onFiles: (files) => choose(files[0]),
    accept: ".png,.jpg,.jpeg,.webp,.gif",
    multiple: false,
  });
  const hasImage = picture === "remove" ? false : Boolean(previewUrl) || Boolean(original?.hasAvatar);
  return (
    <>
      <div
        className={cn(
          "group/picture relative overflow-hidden rounded-[24px] ring-1 ring-black/[0.05] max-md:mx-auto max-md:w-52 dark:ring-white/[0.06]",
          drop.dragging && "ring-2 ring-primary",
        )}
        ref={drop.ref}
        {...drop.dragHandlers}
      >
        <Portrait
          id={original?.id}
          name={name || "?"}
          hasImage={hasImage}
          rev={original?.avatarRev ?? 0}
          imageUrl={previewUrl ?? (picture === "remove" ? null : undefined)}
          variant="card"
          className="aspect-[2/3] w-full"
        />
        <button
          type="button"
          onClick={() => openFilePicker(PICTURE_ACCEPT, ([file]) => choose(file))}
          className="absolute inset-0 flex cursor-pointer flex-col items-center justify-end gap-1 bg-gradient-to-t from-black/60 via-black/0 to-transparent pb-4 text-white opacity-0 transition-opacity focus-visible:opacity-100 focus-visible:outline-none group-hover/picture:opacity-100"
        >
          <HugeiconsIcon icon={ImageUpload01Icon} strokeWidth={1.6} className="size-6" />
          <span className="text-ui-13 font-medium">
            {drop.dragging
              ? t("characters.editor.dropPicture")
              : hasImage
                ? t("characters.editor.changePicture")
                : t("characters.editor.uploadPicture")}
          </span>
        </button>
      </div>
      {hasImage ? (
        <Button variant="ghost" size="sm" className="-mt-2 self-center" onClick={() => onChange("remove")}>
          {t("characters.editor.removePicture")}
        </Button>
      ) : null}
    </>
  );
}
