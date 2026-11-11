// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as Dropzone from "../src/features/rag/components/project-source-dropzone.tsx";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";
import { recordedToasts, toast } from "./helpers/store-stubs/toast.ts";

const announced: string[] = [];
const ragApi = {
  announceProjectSourcesUpdated: (projectId: string) => announced.push(projectId),
  invalidateProjectSources: () => undefined,
  noteProjectWork: () => undefined,
  uploadProjectDocument: async (_projectId: string, file: File) => ({
    documentId: `doc-${file.name}`,
    jobId: `job-${file.name}`,
    filename: file.name,
  }),
  getJob: async (jobId: string) =>
    jobId === "job-scan.pdf"
      ? { id: jobId, status: "failed", error: "No text could be extracted" }
      : { id: jobId, status: "completed" },
};

const { uploadStagedSources } = loadWithStubs<typeof Dropzone>(
  new URL("../src/features/rag/components/project-source-dropzone.tsx", import.meta.url),
  {
    react: {},
    "react/jsx-runtime": stubJsxRuntime(),
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": {},
    "@/lib/utils": { cn: () => "" },
    "@/lib/toast": { toast },
    "@/features/native-intents": {},
    "@/features/chat/stores/chat-runtime-store": {},
    "./vision-overrides": {
      resolveVisionOverrides: async () => ({ ocr: undefined, caption: undefined }),
    },
    "../api/rag-api": ragApi,
    "./rag-api": ragApi,
  },
  { relativePassthrough: true },
);

test("a file added at project creation that fails to index is reported by name", async () => {
  const staged = ["notes.txt", "scan.pdf"].map((name) => {
    const file = new File(["x"], name);
    return { id: name, name, size: file.size, upload: file };
  });
  await uploadStagedSources("p1", staged as never);
  for (let i = 0; i < 60 && announced.length < 3; i++) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.deepEqual(
    recordedToasts.map((t) => [t.kind, t.message, t.description]),
    [["error", "Couldn't index scan.pdf", "No text could be extracted"]],
  );
});
