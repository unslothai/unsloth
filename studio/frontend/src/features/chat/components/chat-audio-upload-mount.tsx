// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense } from "react";
import type { useChatAudioUpload } from "../hooks/use-chat-audio-upload";

const ChatAudioUpload = lazy(() =>
  import("./chat-audio-upload").then((module) => ({
    default: module.ChatAudioUpload,
  })),
);

export function ChatAudioUploadMount({
  audioUpload,
}: {
  audioUpload: ReturnType<typeof useChatAudioUpload>;
}) {
  if (!audioUpload.dialogOpen) return null;
  return (
    <Suspense fallback={null}>
      <ChatAudioUpload audioUpload={audioUpload} />
    </Suspense>
  );
}
