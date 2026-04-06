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

export function VersionBadge({ className }: { className?: string }) {
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/system/hardware")
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch");
        return response.json();
      })
      .then((data: HardwareInfo) => setVersion(data.versions.unsloth))
      .catch(() => {});
  }, []);

  if (!version) {
    return null;
  }

  return (
    <div
      className={cn(
        "fixed bottom-3 right-3 z-10 flex items-center gap-1 text-[11px] opacity-60 hover:opacity-100 transition-opacity",
        className
      )}
    >
      <span className="font-mono text-muted-foreground">v{version}</span>
      <span className="font-medium text-muted-foreground">Beta</span>
    </div>
  );
}
