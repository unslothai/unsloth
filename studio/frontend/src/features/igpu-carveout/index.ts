// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  dismissCarveoutAdviceForModel,
  IGPU_CARVEOUT_NOTICE_DURATION_MS,
  IGPU_CARVEOUT_NOTICE_TITLE,
  IGPU_CARVEOUT_TOAST_ID,
  showCarveoutAdvice,
} from "./igpu-carveout-toast";
export { parseCarveoutAdvice, type IgpuCarveoutAdvice } from "./types";
