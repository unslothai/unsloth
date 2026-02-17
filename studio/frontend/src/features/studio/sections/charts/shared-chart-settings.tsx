import { DropdownMenuCheckboxItem, DropdownMenuLabel, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { ReactElement } from "react";
import type { OutlierMode, ScaleMode, ViewSettingsState } from "./types";

export function SharedChartSettings({
  view,
  scale,
  setScale,
  outlierMode,
  setOutlierMode,
}: {
  view: ViewSettingsState;
  scale: ScaleMode;
  setScale: (value: ScaleMode) => void;
  outlierMode: OutlierMode;
  setOutlierMode: (value: OutlierMode) => void;
}): ReactElement {
  return (
    <>
      <DropdownMenuSeparator />
      <DropdownMenuLabel className="text-xs">View</DropdownMenuLabel>
      <div className="flex flex-col gap-1.5 px-2 py-1.5">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Window (steps)</Label>
          <span className="text-xs tabular-nums text-muted-foreground">
            {view.effectiveWindowSize}
          </span>
        </div>
        <Slider
          value={[view.effectiveWindowSize]}
          onValueChange={([v]) => view.setWindowSize(Math.max(1, Math.round(v)))}
          min={view.minWindow}
          max={Math.max(view.minWindow, view.allStepsLength)}
          step={1}
        />
      </div>
      <div className="flex flex-col gap-1.5 px-2 py-1.5">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Pan</Label>
          <span className="text-xs tabular-nums text-muted-foreground">
            {view.effectivePanOffset}
          </span>
        </div>
        <Slider
          value={[view.effectivePanOffset]}
          onValueChange={([v]) => view.setPanOffset(Math.max(0, Math.round(v)))}
          min={0}
          max={Math.max(0, view.maxPanOffset)}
          step={1}
        />
      </div>
      <DropdownMenuSeparator />
      <DropdownMenuLabel className="text-xs">Y Scale</DropdownMenuLabel>
      <DropdownMenuCheckboxItem
        checked={scale === "linear"}
        onCheckedChange={(checked) => checked && setScale("linear")}
      >
        Linear
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={scale === "log"}
        onCheckedChange={(checked) => checked && setScale("log")}
      >
        Log (log1p)
      </DropdownMenuCheckboxItem>
      <DropdownMenuSeparator />
      <DropdownMenuLabel className="text-xs">Outliers</DropdownMenuLabel>
      <DropdownMenuCheckboxItem
        checked={outlierMode === "none"}
        onCheckedChange={(checked) => checked && setOutlierMode("none")}
      >
        No clipping
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={outlierMode === "p99"}
        onCheckedChange={(checked) => checked && setOutlierMode("p99")}
      >
        Clip above p99
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={outlierMode === "p95"}
        onCheckedChange={(checked) => checked && setOutlierMode("p95")}
      >
        Clip above p95
      </DropdownMenuCheckboxItem>
    </>
  );
}
