// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Production settings, preview, preferences and assistant-ui input. No backend.
import { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  useLocalRuntime,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { ComposerSettings } from "@/features/settings/components/composer-settings";
import { ComposerDraftPreview } from "@/components/assistant-ui/composer-draft-preview";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import {
  composerSubmitIntent,
  composerFollowUpBehavior,
} from "@/features/chat/utils/composer-preferences";
import "./src/index.css";

function Draft() {
  const aui = useAui();
  const text = useAuiState((s) => s.composer.text);
  const prefs = useChatPreferencesStore();
  const [sent, setSent] = useState<{ text: string; behavior: string }[]>([]);
  return (
    <div className="mt-8">
      <ComposerPrimitive.Root
        className="rounded-3xl border border-border bg-background p-5 shadow-sm"
        onSubmit={(e) => e.preventDefault()}
      >
        <ComposerDraftPreview text={text} />
        <ComposerPrimitive.Input
          aria-label="Message"
          placeholder="Ask anything"
          submitMode="none"
          className="min-h-24 w-full resize-none bg-transparent text-sm outline-none"
          onKeyDown={(e) => {
            const intent = composerSubmitIntent(
              { ...e, isComposing: e.nativeEvent.isComposing },
              prefs.sendShortcut,
              text,
            );
            if (!intent) return;
            e.preventDefault();
            setSent((old) => [
              ...old,
              {
                text: aui.composer().getState().text,
                behavior: composerFollowUpBehavior(
                  prefs.followUpBehavior,
                  intent,
                ),
              },
            ]);
            aui.composer().setText("");
          }}
        />
        <div className="mt-3 text-xs text-muted-foreground">
          Try your send shortcut here
        </div>
      </ComposerPrimitive.Root>
      <output aria-label="Submitted messages" className="sr-only">
        {JSON.stringify(sent)}
      </output>
    </div>
  );
}
function App() {
  const runtime = useLocalRuntime({
    async *run() {
      yield { content: [{ type: "text", text: "Fixture" }] };
    },
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <main className="mx-auto max-w-4xl px-5 py-12">
          <ComposerSettings />
          <Draft />
        </main>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
