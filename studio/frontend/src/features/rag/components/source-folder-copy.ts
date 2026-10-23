// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SOURCE_FOLDER_DESCRIPTION =
  "Keep supported documents in sync for chats to search. Linking does not grant access to edit the original files.";

export function sourceFolderUnavailableMessage(isDesktop: boolean): string {
  return isDesktop
    ? "Folder selection is unavailable with this connection. You can still upload documents."
    : "Open Unsloth Desktop to add a source folder. You can still upload documents.";
}
