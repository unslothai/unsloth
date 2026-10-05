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
import { Spinner } from "@/components/ui/spinner";
import { sttModelName, useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { AUDIO_PICKER_ACCEPT } from "@/lib/audio-utils";
import { useEffect, useMemo, useRef } from "react";
import type { useChatAudioUpload } from "../hooks/use-chat-audio-upload";
import { currentRecordingPickerPlatform } from "../utils/dictation-entry";

export interface ChatAudioUploadProps {
  audioUpload: ReturnType<typeof useChatAudioUpload>;
}

export function ChatAudioUpload({ audioUpload }: ChatAudioUploadProps) {
  const t = useT();
  const recordInputRef = useRef<HTMLInputElement>(null);
  const chooseInputRef = useRef<HTMLInputElement>(null);
  const platform = useMemo(() => currentRecordingPickerPlatform(), []);
  const { dialogOpen, pickerCancelled } = audioUpload;
  useEffect(() => {
    if (!dialogOpen) return;
    const cancelled = () => pickerCancelled();
    const inputs = [recordInputRef.current, chooseInputRef.current];
    for (const input of inputs) input?.addEventListener("cancel", cancelled);
    return () => {
      for (const input of inputs)
        input?.removeEventListener("cancel", cancelled);
    };
  }, [dialogOpen, pickerCancelled]);
  const languageName =
    audioUpload.language === "auto"
      ? t("settings.voice.dictation.audioUploadAutomatic")
      : (new Intl.DisplayNames(undefined, { type: "language" }).of(
          audioUpload.language,
        ) ?? audioUpload.language);
  const pickerDisabled =
    audioUpload.readiness.state === "checking" ||
    audioUpload.readiness.state === "downloading" ||
    audioUpload.readiness.state === "unavailable";

  const launch = (input: HTMLInputElement | null) => {
    if (!input || !audioUpload.snapshotForPicker()) return;
    input.click();
  };

  const acceptFile = (input: HTMLInputElement) => {
    const file = input.files?.[0];
    input.value = "";
    if (file) audioUpload.selectFile(file);
  };

  const readinessText = (() => {
    switch (audioUpload.readiness.state) {
      case "ready":
        return t("settings.voice.dictation.audioUploadModelReady");
      case "missing":
        return t("settings.voice.dictation.sttNotDownloaded");
      case "downloading":
        return audioUpload.readiness.progress === null
          ? t("settings.voice.dictation.sttDownloadChecking")
          : t("settings.voice.dictation.sttDownloading", {
              progress: audioUpload.readiness.progress,
            });
      case "unavailable":
        return t("settings.voice.dictation.sttUnavailable");
      case "error":
        return t("settings.voice.dictation.sttDownloadStatusFailed");
      case "checking":
      case "idle":
        return t("settings.voice.dictation.sttChecking");
    }
  })();

  return (
    <Dialog
      open={audioUpload.dialogOpen && !audioUpload.busy}
      onOpenChange={(open) => {
        if (!open) audioUpload.closeDialog();
      }}
    >
      <DialogContent
        showCloseButton={false}
        className="max-sm:content-start sm:max-w-lg"
      >
        <DialogHeader>
          <DialogTitle>
            {t("settings.voice.dictation.audioUploadTitle")}
          </DialogTitle>
          <DialogDescription>
            {t("settings.voice.dictation.audioUploadDescription")}
          </DialogDescription>
        </DialogHeader>

        {platform === "ios" ? (
          <p className="rounded-2xl bg-muted px-3 py-2 text-sm text-muted-foreground">
            {t("settings.voice.dictation.audioUploadIphoneHint")}
          </p>
        ) : null}

        <dl className="grid min-w-0 gap-3 rounded-2xl bg-muted/60 p-3 text-sm">
          <div>
            <dt className="text-muted-foreground">
              {t("settings.voice.dictation.sttModelLabel")}
            </dt>
            <dd className="break-words font-medium">
              <bdi>{sttModelName(audioUpload.model)}</bdi>
            </dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("settings.voice.dictation.languageLabel")}
            </dt>
            <dd className="break-words font-medium">
              <bdi>{languageName}</bdi>
            </dd>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            {audioUpload.readiness.state === "checking" ||
            audioUpload.readiness.state === "downloading" ? (
              <Spinner className="size-3.5" />
            ) : null}
            <span>{readinessText}</span>
          </div>
        </dl>

        <p className="text-sm text-muted-foreground">
          {t("settings.voice.dictation.audioUploadServerModelNote")}
        </p>

        {audioUpload.failureMessage ? (
          <div
            className="rounded-2xl border border-destructive/30 bg-destructive/5 p-3 text-sm"
            role="alert"
          >
            <p className="font-medium text-destructive">
              {t("settings.voice.dictation.audioUploadRetryTitle", {
                file: audioUpload.failedFileName ?? "",
              })}
            </p>
            <p className="mt-1 break-words text-muted-foreground">
              {audioUpload.failureMessage}
            </p>
          </div>
        ) : null}

        <input
          ref={recordInputRef}
          type="file"
          accept="audio/*"
          capture="user"
          className="hidden"
          onChange={(event) => acceptFile(event.currentTarget)}
        />
        <input
          ref={chooseInputRef}
          type="file"
          accept={AUDIO_PICKER_ACCEPT}
          className="hidden"
          onChange={(event) => acceptFile(event.currentTarget)}
        />

        <Button
          type="button"
          variant="outline"
          className="h-auto min-h-9 whitespace-normal py-2"
          onClick={() => useSettingsDialogStore.getState().openDialog("voice")}
        >
          {t("settings.voice.dictation.sttOpenVoiceSettings")}
        </Button>

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            type="button"
            variant="ghost"
            onClick={audioUpload.closeDialog}
          >
            {t("common.cancel")}
          </Button>
          {audioUpload.readiness.state === "error" ? (
            <Button
              type="button"
              variant="outline"
              onClick={() => void audioUpload.refreshReadiness()}
            >
              {t("settings.voice.dictation.sttRetry")}
            </Button>
          ) : null}
          {audioUpload.failureMessage ? (
            <Button
              type="button"
              disabled={pickerDisabled}
              onClick={audioUpload.retry}
            >
              {t("settings.voice.dictation.audioUploadRetry")}
            </Button>
          ) : (
            <>
              {platform === "android" ? (
                <Button
                  type="button"
                  disabled={pickerDisabled}
                  onClick={() => launch(recordInputRef.current)}
                >
                  {t("settings.voice.dictation.audioUploadRecord")}
                </Button>
              ) : null}
              <Button
                type="button"
                variant={platform === "android" ? "outline" : "default"}
                disabled={pickerDisabled}
                onClick={() => launch(chooseInputRef.current)}
              >
                {t("settings.voice.dictation.audioUploadChooseFile")}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
