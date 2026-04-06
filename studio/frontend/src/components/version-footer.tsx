// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface HardwareInfo {
  gpu: {
    gpu_name: string | null;
    vram_total_gb: number | null;
    vram_free_gb: number | null;
  };
  versions: {
    unsloth: string | null;
    torch: string | null;
    transformers: string | null;
    cuda: string | null;
  };
}

export function VersionFooter({ className }: { className?: string }) {
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    async function fetchVersion() {
      try {
        const response = await fetch("/api/system/hardware");
        if (!response.ok) return;
        const data: HardwareInfo = await response.json();
        setVersion(data.versions.unsloth);
      } catch (error) {
        console.error("Failed to fetch version:", error);
      }
    }

    fetchVersion();
  }, []);

  if (!version) {
    return null;
  }

  return (
    <footer
      className={cn(
        "fixed bottom-0 left-0 right-0 z-10 flex items-center justify-center py-2 text-xs text-muted-foreground",
        className
      )}
    >
      <span>
        v{version} <span className="font-semibold">BETA</span>
      </span>
    </footer>
  );
}
