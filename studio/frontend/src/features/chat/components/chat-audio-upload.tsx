// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRef, useState } from "react";
import { Cancel01Icon, Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { AUDIO_PICKER_ACCEPT } from "@/lib/audio-utils";
import { cn } from "@/lib/utils";

export interface ChatAudioUploadProps {
  model: string;
  language: string;
  busy: boolean;
  disabled?: boolean;
  onFileSelected: (file: File) => void;
  onCancel: () => void;
  className?: string;
}

export function ChatAudioUpload({
  model,
  language,
  busy,
  disabled = false,
  onFileSelected,
  onCancel,
  className,
}: ChatAudioUploadProps) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const title = t("settings.voice.dictation.audioUploadTitle");
  const cancel = t("settings.voice.dictation.audioUploadCancel");

  return (
    <Dialog open={open && !busy} onOpenChange={setOpen}>
      {busy ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className={cn("text-muted-foreground", className)}
          aria-label={cancel}
          title={cancel}
          onClick={onCancel}
        >
          <Spinner label={t("settings.voice.dictation.audioUploadTranscribing")} />
          <HugeiconsIcon icon={Cancel01Icon} className="size-3" aria-hidden="true" />
        </Button>
      ) : (
        <DialogTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={cn("text-muted-foreground", className)}
            aria-label={title}
            title={title}
            disabled={disabled}
          >
            <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} aria-hidden="true" />
          </Button>
        </DialogTrigger>
      )}
      <DialogContent showCloseButton={false} className="max-sm:content-start">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            {t("settings.voice.dictation.audioUploadDescription")}
          </DialogDescription>
        </DialogHeader>
        <dl className="grid min-w-0 gap-3 text-sm">
          <div>
            <dt className="text-muted-foreground">
              {t("settings.voice.dictation.sttModelLabel")}
            </dt>
            <dd className="break-words">
              <bdi>{model}</bdi>
            </dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("settings.voice.dictation.languageLabel")}
            </dt>
            <dd className="break-words">
              <bdi>{language}</bdi>
            </dd>
          </div>
        </dl>
        <input
          ref={inputRef}
          type="file"
          accept={AUDIO_PICKER_ACCEPT}
          className="hidden"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (!file || disabled || busy) return;
            setOpen(false);
            onFileSelected(file);
          }}
        />
        <Button
          type="button"
          variant="outline"
          className="h-auto min-h-9 whitespace-normal py-2"
          onClick={() => {
            setOpen(false);
            useSettingsDialogStore.getState().openDialog("voice");
          }}
        >
          {t("settings.voice.dictation.sttOpenVoiceSettings")}
        </Button>
        <DialogFooter>
          <DialogClose asChild>
            <Button type="button" variant="ghost">
              {t("common.cancel")}
            </Button>
          </DialogClose>
          <Button
            type="button"
            className="h-auto min-h-9 whitespace-normal py-2"
            disabled={disabled || busy}
            onClick={() => inputRef.current?.click()}
          >
            {t("settings.voice.dictation.audioUploadChooseFile")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
