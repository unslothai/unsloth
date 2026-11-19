// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { startSttDownload } from "@/features/chat/adapters/studio-model-dictation-adapter";
import {
  hfApiToken,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import type { SttDownloadRequest } from "../stores/stt-download-prompt-store";
import { useVoiceSettingsStore } from "../stores/voice-settings-store";
import { trackSttDownload } from "./stt-download-mirror";

export async function downloadSttNow(
  request: SttDownloadRequest,
): Promise<void> {
  if (request.selectLocalEngine) {
    useVoiceSettingsStore.getState().setDictationEngine("model");
  }
  const hfToken = useHfTokenStore.getState().token;
  try {
    const download = await startSttDownload(
      request.model,
      hfApiToken(hfToken),
      undefined,
      request.ggufVariant,
    );
    trackSttDownload(request.model, {
      ggufVariant: request.ggufVariant ?? null,
      downloadId: download.download_id,
    });
  } catch (error) {
    toast.error(translate("settings.voice.dictation.sttDownloadFailed"), {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}
