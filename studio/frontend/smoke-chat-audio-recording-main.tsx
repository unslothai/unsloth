// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  useAuiState,
  useLocalRuntime,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ChatAudioRecording } from "@/components/assistant-ui/chat-audio-recording";
// eslint-disable-next-line no-restricted-imports -- Test the production attachment adapter directly.
import { AudioAttachmentAdapter } from "@/features/chat/audio-attachment-adapter";
import "./src/index.css";

/** Render the real recording control and attachment adapter without loading a model. */
function Composer() {
  const [visible, setVisible] = useState(true);
  const attachments = useAuiState((state) => state.composer.attachments);
  return (
    <>
      <button type="button" onClick={() => setVisible(!visible)}>
        Switch chat
      </button>
      <ComposerPrimitive.Root>
        <ComposerPrimitive.If dictation={false}>
          {visible && <ChatAudioRecording />}
          <span>Dictation control</span>
        </ComposerPrimitive.If>
        <output aria-label="Attachments">
          {JSON.stringify(
            attachments.map((item) => ({
              name: item.name,
              type: item.contentType,
              size: item.file?.size,
            })),
          )}
        </output>
      </ComposerPrimitive.Root>
    </>
  );
}

/** Host a composer with the same audio adapter as the application. */
function App() {
  const runtime = useLocalRuntime(
    { async *run() {} },
    {
      adapters: { attachments: new AudioAttachmentAdapter() },
    },
  );
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <Composer />
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
