// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FC } from "react";

/** Microphone icon used by the chat composer and the Voice settings tab. */
export const MicIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="currentColor"
    xmlns="http://www.w3.org/2000/svg"
    focusable="false"
    aria-hidden={true}
  >
    <path d="M18.585 13.412a.9.9 0 0 1 1.668.676 8.91 8.91 0 0 1-7.353 5.516V22a.9.9 0 0 1-1.8 0v-2.396a8.91 8.91 0 0 1-7.352-5.516.9.9 0 0 1 1.668-.676 7.104 7.104 0 0 0 13.169 0" />
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      transform="translate(0 -0.425) scale(1 1.05)"
      d="M12 1.35a4.9 4.9 0 0 1 4.9 4.9v4.483a4.9 4.9 0 0 1-9.8 0V6.25a4.9 4.9 0 0 1 4.9-4.9m0 1.8a3.1 3.1 0 0 0-3.1 3.1v4.483a3.1 3.1 0 1 0 6.2 0V6.25a3.1 3.1 0 0 0-3.1-3.1"
    />
  </svg>
);
