// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useOpenCharacterChat } from "../actions";
import { CharacterPickerDialog } from "./character-picker";

/** "Chat with a character" from the composer's + menu: pick one (or several), and the chat opens. */
export function CharacterChatPicker({ onClose }: { onClose: () => void }) {
  const { open } = useOpenCharacterChat();
  return (
    <CharacterPickerDialog
      open={true}
      mode="start"
      onClose={onClose}
      onPick={(ids) => {
        onClose();
        void open(ids);
      }}
    />
  );
}
