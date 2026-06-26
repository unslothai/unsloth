// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TestTube01Icon } from "@hugeicons/core-free-icons";

// Slice to the first 3 paths to drop TestTube01Icon's two interior bubbles,
// keeping the outline + cap + liquid line. Shared so Train stays identical app-wide.
export const TrainIcon = TestTube01Icon.slice(0, 3) as typeof TestTube01Icon;
