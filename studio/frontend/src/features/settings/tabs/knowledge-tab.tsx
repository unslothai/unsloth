// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { KnowledgeBaseManager } from "@/features/rag";
import { useT } from "@/i18n";

/** The chat composer's knowledge base dialog, as a page: same views, same API, no new state. */
export function KnowledgeTab() {
  const t = useT();
  return (
    <div className="settings-page">
      <KnowledgeBaseManager
        open={true}
        renderHeader={(title, description, inKnowledgeBase) => (
          <header className="flex flex-col gap-1">
            <h1 className="text-xl font-semibold font-heading">
              {inKnowledgeBase ? title : t("settings.tabs.knowledge")}
            </h1>
            <p className="text-sm text-muted-foreground">{description}</p>
          </header>
        )}
      />
    </div>
  );
}
