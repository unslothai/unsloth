// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useTrainingConfigStore } from "@/features/training";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";

export function useMlxTrainingConfigPolicy(): void {
  const isMac = usePlatformStore((state) => state.deviceType === "mac");
  const { loraVariant, packing } = useTrainingConfigStore(
    useShallow((state) => ({
      loraVariant: state.loraVariant,
      packing: state.packing,
    })),
  );

  // Only loftq: MLX trains DoRA, so clearing a `dora` selection here would
  // silently substitute plain LoRA for the run the user asked for.
  useEffect(() => {
    if (isMac && loraVariant === "loftq") {
      useTrainingConfigStore.setState({ loraVariant: "lora" });
    }
  }, [isMac, loraVariant]);

  useEffect(() => {
    if (isMac && packing) {
      useTrainingConfigStore.setState({ packing: false });
    }
  }, [isMac, packing]);
}
