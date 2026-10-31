// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// file-kind.ts asks the document viewer and the chat feature about names; stub both with the
// answers for the plain names these tests use.
const FILE_VIEWER_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const documentKind = () => null; export const isMarkdown = (name) => /\\.md$/i.test(name);");

const CHAT_STUB =
  "data:text/javascript," +
  encodeURIComponent(
    "export const useChatArtifactsStore = { getState: () => ({ closeArtifactSurface() {} }) };" +
      " export const attachmentTextLanguage = (name) => (/\\.(tsx?|jsx?|py)$/i.test(name) ? \"code\" : null);",
  );

export function resolve(specifier, context, next) {
  if (specifier === "@/components/file-viewer") return { url: FILE_VIEWER_STUB, shortCircuit: true };
  if (specifier === "@/features/chat" && context.parentURL?.includes("/src/features/browser/file-kind")) {
    return { url: CHAT_STUB, shortCircuit: true };
  }
  return next(specifier, context);
}
