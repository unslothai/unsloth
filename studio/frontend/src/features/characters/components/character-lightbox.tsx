// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { downloadBlobStreaming, isDownloadCancelled } from "@/lib/native-files";
import { Cancel01Icon, Download01Icon, Edit03Icon, UserCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Dialog as DialogPrimitive } from "radix-ui";
import { fetchImage } from "../api";
import { initials, nameHue, useImageUrl } from "../images";
import { useCharacterLightbox } from "../lightbox-store";

/** A character's or persona's picture at full size, with its name and the actions that go with it. */
export function CharacterLightbox() {
  const t = useT();
  const navigate = useNavigate();
  const subject = useCharacterLightbox((s) => s.subject);
  const close = useCharacterLightbox((s) => s.close);
  const url = useImageUrl(subject?.kind ?? "character", subject?.id, subject?.rev ?? 0, Boolean(subject?.hasImage));
  const hue = nameHue(subject?.name ?? "");

  async function download() {
    if (!subject) return;
    try {
      const blob = await fetchImage(subject.kind, subject.id);
      const extension = blob.type.split("/")[1]?.replace("jpeg", "jpg") || "png";
      await downloadBlobStreaming(blob, `${subject.name.replace(/[^\w\- ]+/g, "").trim() || "picture"}.${extension}`);
    } catch (error) {
      if (!isDownloadCancelled(error)) throw error;
    }
  }

  return (
    <DialogPrimitive.Root open={subject !== null} onOpenChange={(open) => !open && close()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[80] bg-black/80 backdrop-blur-[3px] data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=closed]:animate-out data-[state=closed]:fade-out-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed inset-0 z-[81] flex flex-col items-center justify-center gap-4 p-4 outline-none data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-[0.97] sm:p-8"
          onClick={(event) => {
            if (event.target === event.currentTarget) close();
          }}
        >
          <DialogPrimitive.Title className="sr-only">{subject?.name ?? ""}</DialogPrimitive.Title>
          {subject ? (
            <>
              <figure className="relative flex max-h-[calc(100dvh-9rem)] min-h-0 max-w-full flex-col items-center">
                {url ? (
                  <img
                    src={url}
                    alt={subject.name}
                    className="max-h-[calc(100dvh-9rem)] max-w-[min(92vw,900px)] rounded-[22px] object-contain shadow-[0_30px_80px_-20px_rgb(0_0_0/0.8)]"
                    data-testid="character-lightbox-image"
                  />
                ) : (
                  <div
                    className="flex aspect-[2/3] h-[min(70dvh,560px)] items-center justify-center rounded-[22px] font-heading text-7xl font-semibold text-white/95"
                    style={{ background: `linear-gradient(150deg, hsl(${hue} 42% 62%), hsl(${(hue + 28) % 360} 46% 40%))` }}
                  >
                    {initials(subject.name)}
                  </div>
                )}
              </figure>
              <div className="flex max-w-[min(92vw,900px)] flex-wrap items-center justify-center gap-x-4 gap-y-2 text-white">
                <div className="flex min-w-0 flex-col items-center text-center sm:items-start sm:text-left">
                  <span className="font-heading text-ui-20 font-semibold">{subject.name}</span>
                  {subject.caption ? <span className="line-clamp-1 text-ui-13 text-white/65">{subject.caption}</span> : null}
                </div>
                <div className="flex items-center gap-1.5">
                  {subject.kind === "character" ? (
                    <>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="rounded-full text-white hover:bg-neutral-800 hover:text-white"
                        onClick={() => {
                          close();
                          void navigate({ to: "/characters/$characterId", params: { characterId: subject.id } });
                        }}
                      >
                        <HugeiconsIcon icon={UserCircleIcon} strokeWidth={1.75} className="size-icon" />
                        {t("characters.lightbox.openProfile")}
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="rounded-full text-white hover:bg-neutral-800 hover:text-white"
                        onClick={() => {
                          close();
                          void navigate({ to: "/characters/$characterId/edit", params: { characterId: subject.id } });
                        }}
                      >
                        <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                        {t("characters.lightbox.edit")}
                      </Button>
                    </>
                  ) : null}
                  {subject.hasImage ? (
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="rounded-full text-white hover:bg-neutral-800 hover:text-white"
                      aria-label={t("characters.lightbox.download")}
                      title={t("characters.lightbox.download")}
                      onClick={() => void download()}
                    >
                      <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
                    </Button>
                  ) : null}
                </div>
              </div>
              <DialogPrimitive.Close asChild={true}>
                <button
                  type="button"
                  aria-label={t("characters.lightbox.close")}
                  className="fixed right-4 top-4 flex size-10 cursor-pointer items-center justify-center rounded-full bg-neutral-800 text-white shadow-lg transition-colors hover:bg-neutral-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70"
                >
                  <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.9} className="size-5" />
                </button>
              </DialogPrimitive.Close>
            </>
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
