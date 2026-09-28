// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDeferredValue, type ComponentProps } from "react";
import { Streamdown } from "streamdown";
import { useChatPreferencesStore } from "@/features/chat";
import { useT } from "@/i18n";

// Render draft URLs without navigation or image requests.
const components: NonNullable<ComponentProps<typeof Streamdown>["components"]> =
  {
    a: ({ children }) => (
      <span className="text-primary underline">{children}</span>
    ),
    img: ({ alt }) => (
      <span className="text-muted-foreground">[{alt || "image"}]</span>
    ),
  };

export function ComposerDraftPreview({ text }: { text: string }) {
  const plain = useChatPreferencesStore((s) => s.plainTextComposer);
  const draft = useDeferredValue(text);
  const t = useT();
  if (plain || !text.trim() || !draft.trim()) return null;
  return (
    <section
      aria-label={t("composerSettings.preview")}
      className="mb-3 min-w-0 rounded-xl border border-border/60 bg-muted/20 p-3 text-sm"
    >
      {/* Scroller inset by the section's padding: a scrollbar flush with a rounded
          edge squares the corners on its side. See ScrollPane. */}
      <div className="max-h-48 min-w-0 overflow-auto">
      <div className="mb-2 text-xs text-muted-foreground">
        {t("composerSettings.preview")}
      </div>
      <Streamdown
        mode="static"
        controls={false}
        components={components}
        skipHtml
        className="min-w-0 space-y-2 [overflow-wrap:anywhere] [&_pre]:whitespace-pre-wrap [&_table]:block [&_table]:overflow-x-auto"
      >
        {draft}
      </Streamdown>
      </div>
    </section>
  );
}
