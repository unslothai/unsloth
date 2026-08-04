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
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { startSttDownload } from "@/features/chat";
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { useT } from "@/i18n";
import { MicIcon } from "@/lib/mic-icon";
import { toast } from "@/lib/toast";
import type { ReactNode } from "react";
import { trackSttDownload } from "../lib/stt-download-mirror";
import { useSttDownloadPromptStore } from "../stores/stt-download-prompt-store";
import {
  type SttModel,
  sttModelName,
  sttModelSize,
} from "../stores/voice-settings-store";

/** Emphasise the model name in translated copy; plain text if absent. */
function highlightModel(text: string, model: string): ReactNode {
  const at = model ? text.indexOf(model) : -1;
  if (at === -1) return text;
  return (
    <>
      {text.slice(0, at)}
      <span className="font-medium text-foreground">{model}</span>
      {text.slice(at + model.length)}
    </>
  );
}

/**
 * App-level confirmation for a dictation model download. Mounted once so the
 * mic can raise it before Voice settings is ever opened.
 */
export function SttDownloadPrompt() {
  const t = useT();
  const pendingModel = useSttDownloadPromptStore((s) => s.pendingModel);
  const dismiss = useSttDownloadPromptStore((s) => s.dismiss);
  const hfToken = useHfTokenStore((state) => state.token);

  const confirm = async (model: SttModel) => {
    try {
      await startSttDownload(model, hfApiToken(hfToken));
      // Progress goes to the shared download panel; the model loads itself
      // when it lands.
      trackSttDownload(model);
    } catch (error) {
      toast.error(t("settings.voice.dictation.sttDownloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const size = pendingModel ? sttModelSize(pendingModel) : "";
  return (
    <AlertDialog
      open={pendingModel !== null}
      onOpenChange={(open) => {
        if (!open) dismiss();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia>
            {/* The glyph fills its viewBox, so it needs to sit smaller than the
                padded hugeicons the media circle is sized for. */}
            <MicIcon className="text-muted-foreground size-5" />
          </AlertDialogMedia>
          <AlertDialogTitle>
            {t("settings.voice.dictation.sttDownloadConfirmTitle", {
              model: sttModelName(pendingModel ?? ""),
            })}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {highlightModel(
              pendingModel && size
                ? t("settings.voice.dictation.sttDownloadConfirmBody", {
                    model: sttModelName(pendingModel),
                    size,
                  })
                : t("settings.voice.dictation.sttDownloadConfirmBodyUnsized", {
                    model: sttModelName(pendingModel ?? ""),
                  }),
              sttModelName(pendingModel ?? ""),
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={(event) => {
              event.preventDefault();
              const model = pendingModel;
              dismiss();
              if (model) void confirm(model);
            }}
          >
            {t("settings.voice.dictation.sttDownload")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
