// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { startCharacterChat } from "../actions";
import { CharacterPickerDialog } from "./character-picker";

/** "Chat with a character" from the composer's + menu: pick one (or several), and the chat opens. */
export function CharacterChatPicker({ onClose }: { onClose: () => void }) {
  const t = useT();
  const navigate = useNavigate();
  return (
    <CharacterPickerDialog
      open={true}
      mode="start"
      onClose={onClose}
      onPick={(ids) => {
        onClose();
        void startCharacterChat(ids)
          .then((threadId) => navigate({ to: "/chat", search: { thread: threadId } }))
          .catch((error) =>
            toast.error(t("characters.chat.startFailed"), {
              description: error instanceof Error ? error.message : String(error),
            }),
          );
      }}
    />
  );
}
