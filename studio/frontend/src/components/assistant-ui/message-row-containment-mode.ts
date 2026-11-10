// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ContainmentMode,
  defineContainmentFlag,
} from "./containment-flag.ts";

export type MessageRowContainmentMode = ContainmentMode;

/* Off on WebKitGTK: message-window-controller.ts already parks distant rows there. */
export const MESSAGE_ROW_CONTAINMENT = defineContainmentFlag({
  attribute: "data-message-row-containment",
  global: "__UNSLOTH_MESSAGE_ROW_CONTAINMENT__",
});

export const MESSAGE_ROW_CONTAINMENT_ATTRIBUTE = MESSAGE_ROW_CONTAINMENT.attribute;
export const MESSAGE_ROW_CONTAINMENT_ON = MESSAGE_ROW_CONTAINMENT.on;
