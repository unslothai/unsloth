// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ActionBarPrimitive } from "@assistant-ui/react";
import { createContext, useContext, type FC, type PropsWithChildren } from "react";
import { UserMessageTime } from "./user-message-time";

/** Use the message's full width, independently of the bubble's 80% limit. */
export const UserMessageFooter: FC<PropsWithChildren> = ({ children }) => (
  <div className="aui-user-message-footer mt-1 flex min-h-8 w-full min-w-0 flex-wrap items-center justify-end gap-y-1">
    {children}
  </div>
);

/** False where the message already shows its time elsewhere, as a character chat's name line does. */
export const UserMessageTimeShownContext = createContext(true);

export const UserMessageActionBar: FC<PropsWithChildren> = ({ children }) => {
  const showTime = useContext(UserMessageTimeShownContext);
  return (
    <ActionBarPrimitive.Root
      autohide="always"
      className="aui-user-action-bar-root contents text-chat-icon-fg"
    >
      {showTime ? <UserMessageTime /> : null}
      <div className="aui-user-action-controls flex shrink-0 gap-1 [&_button]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover [&_button:hover]:text-chat-icon-fg-hover">
        {children}
      </div>
    </ActionBarPrimitive.Root>
  );
};
