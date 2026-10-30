// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  svgToPng,
  svgToString,
} from "@/features/benchmarks/components/chart-export";
import { useT } from "@/i18n";
import { DownloadCancelledError, downloadFile } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement, ReactNode, RefObject } from "react";

/** Config sweeps' Export menu for a chart: the titled SVG behind `svgRef` as SVG or PNG. */
export function ChartSaveMenu({
  svgRef,
  fileBase,
  children,
}: {
  svgRef: RefObject<SVGSVGElement | null>;
  fileBase: string;
  /** Extra items above the two saves. */
  children?: ReactNode;
}): ReactElement {
  const t = useT();
  const save = async (content: string | Blob, name: string, mime: string) => {
    try {
      await downloadFile(content, name, mime);
    } catch (err) {
      if (!(err instanceof DownloadCancelledError))
        toast.error(t("benchmark.chart.saveFailed"), {
          description: err instanceof Error ? err.message : String(err),
        });
    }
  };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <Button variant="ghost" size="sm" className="h-8 gap-1.5 rounded-full">
          <HugeiconsIcon
            icon={Download04Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
          {t("benchmark.history.export")}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {children}
        <DropdownMenuItem
          onSelect={() =>
            svgRef.current &&
            void save(
              svgToString(svgRef.current),
              `${fileBase}.svg`,
              "image/svg+xml",
            )
          }
        >
          {t("benchmark.chart.saveSvg")}
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={async () => {
            if (!svgRef.current) return;
            try {
              await save(
                await svgToPng(svgRef.current),
                `${fileBase}.png`,
                "image/png",
              );
            } catch (err) {
              toast.error(t("benchmark.chart.pngFailed"), {
                description: err instanceof Error ? err.message : String(err),
              });
            }
          }}
        >
          {t("benchmark.chart.savePng")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
