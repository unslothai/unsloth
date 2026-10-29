// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ComponentProps } from "react";

export const RefreshGlyph = (
  props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">,
) => <HugeiconsIcon icon={Refresh01Icon} strokeWidth={2} {...props} />;
