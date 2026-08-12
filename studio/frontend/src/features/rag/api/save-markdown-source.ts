// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { invalidateProjectSources, uploadProjectDocument } from "./rag-api";

export async function saveMarkdownAsProjectSource(
  projectId: string,
  markdown: string,
  title: string,
): Promise<void> {
  const name =
    title.replace(/[\\/:*?"<>|]/g, "_").trim().slice(0, 80) || "chat";
  const file = new File([markdown], `${name}.md`, { type: "text/markdown" });
  try {
    await uploadProjectDocument(projectId, file);
    invalidateProjectSources(projectId);
    toast.success("Saved to project sources.");
  } catch (error) {
    toast.error("Failed to save to project sources.", {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}
