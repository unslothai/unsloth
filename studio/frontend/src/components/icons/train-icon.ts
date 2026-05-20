// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TestTube01Icon } from "@hugeicons/core-free-icons";

// Hugeicons' TestTube01Icon ships with two interior bubbles (paths #4
// and #5 of the 5-path definition). Slicing to the first three paths
// keeps the test-tube outline + horizontal cap + liquid line, dropping
// the bubbles. Shared between the left sidebar's Train nav item and
// the hub's Train action buttons so the icon stays identical across
// the app.
export const TrainIcon = TestTube01Icon.slice(0, 3) as typeof TestTube01Icon;
