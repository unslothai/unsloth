// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SOURCE_FOLDER_DESCRIPTION =
  "Keep supported documents in sync for chats to search. Choose a folder on this server or link one from Desktop. Linking does not grant access to edit the original files.";

export function sourceFolderUnavailableMessage(isDesktop: boolean): string {
  return isDesktop
    ? "Local folder selection is unavailable with this connection. Use Browse server to choose a folder on the Studio host."
    : "Use Browse server to choose a folder on the Studio host, or open Unsloth Desktop to link a local folder.";
}
