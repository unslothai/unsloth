// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InternetIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ComponentProps } from "react";

// The globe as a component, for slots that take one. Stroke 2 matches the lucide icons beside it.
export const InternetGlyph = (
  props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">,
) => <HugeiconsIcon icon={InternetIcon} strokeWidth={2} {...props} />;
