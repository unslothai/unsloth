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

  useEffect(() => {
    if (isMac && (loraVariant === "loftq" || loraVariant === "dora")) {
      useTrainingConfigStore.setState({ loraVariant: "lora" });
    }
  }, [isMac, loraVariant]);

  useEffect(() => {
    if (isMac && packing) {
      useTrainingConfigStore.setState({ packing: false });
    }
  }, [isMac, packing]);
}
