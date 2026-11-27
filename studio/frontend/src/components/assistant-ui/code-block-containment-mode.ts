// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ContainmentMode,
  defineContainmentFlag,
} from "./containment-flag.ts";

export type CodeBlockContainmentMode = ContainmentMode;

export const CODE_BLOCK_CONTAINMENT = defineContainmentFlag({
  attribute: "data-code-block-containment",
  global: "__UNSLOTH_CODE_BLOCK_CONTAINMENT__",
});

export const SHIP_DEFAULT = CODE_BLOCK_CONTAINMENT.shipDefault;
export const CODE_BLOCK_CONTAINMENT_ATTRIBUTE = CODE_BLOCK_CONTAINMENT.attribute;
export const CODE_BLOCK_CONTAINMENT_ON = CODE_BLOCK_CONTAINMENT.on;
export const CODE_BLOCK_CONTAINMENT_GLOBAL = CODE_BLOCK_CONTAINMENT.global;
export const FENCE_HEIGHT_PROPERTY = "--unsloth-fence-height";

export const resolveCodeBlockContainmentMode = CODE_BLOCK_CONTAINMENT.resolve;
export const codeBlockContainmentMode = CODE_BLOCK_CONTAINMENT.mode;
export const installCodeBlockContainmentWatcher =
  CODE_BLOCK_CONTAINMENT.installWatcher;
