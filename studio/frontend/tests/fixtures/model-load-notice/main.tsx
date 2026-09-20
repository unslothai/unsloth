// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* eslint-disable no-restricted-imports -- Exercise the real hook and UI without the feature barrel. */

import { Toaster } from "@/components/ui/sonner";
import {
  ModelLoadDescription,
  ModelLoadInlineStatus,
} from "@/features/chat/components/model-load-status";
import { mediaModelLoadToastOptions } from "@/features/chat/lib/model-load-toast-options";
import { useChatModelRuntime } from "@/features/chat/hooks/use-chat-model-runtime";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { toast } from "@/lib/toast";
import { createRoot } from "react-dom/client";
import "./styles.css";

localStorage.setItem("unsloth_auth_token", "model-load-fixture");
useChatRuntimeStore.setState({ settingsHydrated: true });
// eslint-disable-next-line react-refresh/only-export-components -- Standalone browser fixture.
function App() {
  const runtime = useChatModelRuntime();
  const loading = useChatRuntimeStore((s) => s.modelLoading);
  return (
    <main className="p-8">
      <button
        type="button"
        onClick={() =>
          runtime.selectModel({
            id: "fixture/model",
            isGguf: true,
            isDownloaded: !new URLSearchParams(location.search).has("download"),
            ggufVariant: "Q4_K_M",
            forceReload: true,
          })
        }
      >
        Load model
      </button>
      <button
        type="button"
        onClick={() => {
          const id = toast(
            null,
            mediaModelLoadToastOptions({
              description: (
                <ModelLoadDescription
                  title="Downloading model requirements…"
                  progressPercent={20}
                  progressLabel="200 MB of 1 GB"
                />
              ),
              onCancel: () => toast.dismiss(id),
              onHide: () => toast.dismiss(id),
            }),
          );
        }}
      >
        Show media model load
      </button>
      <p data-testid="lifecycle">{loading ? "Busy" : "Idle"}</p>
      <div data-testid="inline-status">
        {runtime.loadingModel && runtime.loadToastDismissed ? (
          <ModelLoadInlineStatus
            label="Loading model…"
            title="Loading model"
            progressPercent={runtime.loadProgress?.percent}
            progressLabel={runtime.loadProgress?.label}
            onStop={runtime.cancelLoading}
          />
        ) : null}
      </div>
      <Toaster />
    </main>
  );
}
const root = document.getElementById("root");
if (!root) {
  throw new Error("Missing fixture root");
}
createRoot(root).render(<App />);
