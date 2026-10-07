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
import { audioCppSizeLabel } from "@/features/audio/audio-cpp-catalog";
import { startSttDownload } from "@/features/chat";
import { hfApiToken, listGgufVariants, useHfTokenStore } from "@/features/hub";
import { useT } from "@/i18n";
import { MicIcon } from "@/lib/mic-icon";
import { toast } from "@/lib/toast";
import { type ReactNode, useEffect, useState } from "react";
import { trackSttDownload } from "../lib/stt-download-mirror";
import {
  type SttDownloadRequest,
  useSttDownloadPromptStore,
} from "../stores/stt-download-prompt-store";
import {
  sttModelName,
  sttModelSize,
  useVoiceSettingsStore,
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
  const pending = useSttDownloadPromptStore((s) => s.pending);
  const dismiss = useSttDownloadPromptStore((s) => s.dismiss);
  const hfToken = useHfTokenStore((state) => state.token);
  const pendingModel = pending?.model ?? null;
  const variant = pending?.ggufVariant ?? null;
  // A package folder has no curated size: read its quant's from the cached listing.
  const [listedSize, setListedSize] = useState<{
    model: string;
    variant: string | null;
    size: string;
  } | null>(null);
  useEffect(() => {
    if (!pendingModel || sttModelSize(pendingModel)) return;
    let cancelled = false;
    listGgufVariants(pendingModel, hfApiToken(hfToken))
      .then((listing) => {
        const row = listing.variants.find(
          (candidate) =>
            candidate.quant === (variant ?? listing.default_variant),
        );
        if (cancelled || !row) return;
        setListedSize({
          model: pendingModel,
          variant,
          size: audioCppSizeLabel(row.download_size_bytes ?? row.size_bytes),
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [pendingModel, variant, hfToken]);

  const confirm = async (request: SttDownloadRequest) => {
    // Only on accept: a cancel must not leave the engine changed.
    if (request.selectLocalEngine) {
      useVoiceSettingsStore.getState().setDictationEngine("model");
    }
    try {
      await startSttDownload(
        request.model,
        hfApiToken(hfToken),
        undefined,
        request.ggufVariant,
      );
      // Progress goes to the shared download panel; the model loads itself when it lands.
      trackSttDownload(request.model, {
        ggufVariant: request.ggufVariant ?? null,
      });
    } catch (error) {
      toast.error(t("settings.voice.dictation.sttDownloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const size = pendingModel
    ? sttModelSize(pendingModel) ||
      (listedSize?.model === pendingModel && listedSize.variant === variant
        ? listedSize.size
        : "")
    : "";
  return (
    <AlertDialog
      open={pending !== null}
      onOpenChange={(open) => {
        if (!open) dismiss();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          {/* The glyph fills its viewBox, unlike the padded hugeicons the
              default circle is sized for, so both come down together. */}
          <AlertDialogMedia className="size-12">
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
              const request = pending;
              dismiss();
              if (request) void confirm(request);
            }}
          >
            {t("settings.voice.dictation.sttDownload")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
