// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { useChartPreferencesStore } from "./chart-preferences-store";
import type { OutlierMode, ScaleMode } from "./types";

function ChoiceButtons<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { label: string; value: T }[];
  value: T;
  onChange: (value: T) => void;
}): ReactElement {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((option) => (
        <Button
          key={option.value}
          type="button"
          size="xs"
          variant={value === option.value ? "secondary" : "outline"}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  );
}

function SettingRow({
  label,
  description,
  control,
}: {
  label: string;
  description?: string;
  control: ReactElement;
}): ReactElement {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <Label className="text-sm">{label}</Label>
        {description ? (
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        ) : null}
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}

function ScaleSection({
  title,
  scale,
  setScale,
  outlierMode,
  setOutlierMode,
}: {
  title: string;
  scale: ScaleMode;
  setScale: (value: ScaleMode) => void;
  outlierMode: OutlierMode;
  setOutlierMode: (value: OutlierMode) => void;
}): ReactElement {
  return (
    <div className="space-y-3">
      <div>
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">Scale and cleanup</p>
      </div>
      <ChoiceButtons
        options={[
          { label: "Linear", value: "linear" },
          { label: "Log", value: "log" },
        ]}
        value={scale}
        onChange={setScale}
      />
      <ChoiceButtons
        options={[
          { label: "No clip", value: "none" },
          { label: "Clip p99", value: "p99" },
          { label: "Clip p95", value: "p95" },
        ]}
        value={outlierMode}
        onChange={setOutlierMode}
      />
    </div>
  );
}

export function ChartSettingsSheet(): ReactElement {
  const [open, setOpen] = useState(false);
  const {
    availableSteps,
    windowSize,
    smoothing,
    showRaw,
    showSmoothed,
    showAvgLine,
    lossScale,
    lrScale,
    gradScale,
    lossOutlierMode,
    gradOutlierMode,
    lrOutlierMode,
    setWindowSize,
    setSmoothing,
    setShowRaw,
    setShowSmoothed,
    setShowAvgLine,
    setLossScale,
    setLrScale,
    setGradScale,
    setLossOutlierMode,
    setGradOutlierMode,
    setLrOutlierMode,
    resetPreferences,
  } = useChartPreferencesStore(
    useShallow((state) => ({
      availableSteps: state.availableSteps,
      windowSize: state.windowSize,
      smoothing: state.smoothing,
      showRaw: state.showRaw,
      showSmoothed: state.showSmoothed,
      showAvgLine: state.showAvgLine,
      lossScale: state.lossScale,
      lrScale: state.lrScale,
      gradScale: state.gradScale,
      lossOutlierMode: state.lossOutlierMode,
      gradOutlierMode: state.gradOutlierMode,
      lrOutlierMode: state.lrOutlierMode,
      setWindowSize: state.setWindowSize,
      setSmoothing: state.setSmoothing,
      setShowRaw: state.setShowRaw,
      setShowSmoothed: state.setShowSmoothed,
      setShowAvgLine: state.setShowAvgLine,
      setLossScale: state.setLossScale,
      setLrScale: state.setLrScale,
      setGradScale: state.setGradScale,
      setLossOutlierMode: state.setLossOutlierMode,
      setGradOutlierMode: state.setGradOutlierMode,
      setLrOutlierMode: state.setLrOutlierMode,
      resetPreferences: state.resetPreferences,
    })),
  );

  const minWindow = Math.min(10, Math.max(1, availableSteps));
  const effectiveWindowSize =
    windowSize == null ? Math.max(availableSteps, 1) : windowSize;
  const showingAll =
    availableSteps > 0 &&
    (windowSize == null || effectiveWindowSize >= availableSteps);
  const sliderMax = Math.max(minWindow, availableSteps || 1);

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        className="rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
        onClick={() => setOpen(true)}
        aria-label="Open chart settings"
      >
        <HugeiconsIcon icon={Settings02Icon} className="size-4" />
      </Button>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          className="w-full sm:max-w-md"
          overlayClassName="bg-transparent backdrop-blur-0"
        >
          <SheetHeader className="pb-4">
            <SheetTitle>Chart Settings</SheetTitle>
            <SheetDescription>
              Tune chart presentation while training keeps running.
            </SheetDescription>
          </SheetHeader>
          <div className="flex-1 space-y-6 overflow-y-auto px-6 pb-6">
            <div className="space-y-3">
              <div>
                <p className="text-sm font-medium">View window</p>
                <p className="text-xs text-muted-foreground">
                  Show latest steps only or the full history.
                </p>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Window</span>
                  <span className="tabular-nums">
                    {showingAll ? "All" : effectiveWindowSize}
                  </span>
                </div>
                <Slider
                  value={[effectiveWindowSize]}
                  onValueChange={([value]) => setWindowSize(value)}
                  min={minWindow}
                  max={sliderMax}
                  step={1}
                  disabled={availableSteps <= 1}
                />
              </div>
            </div>
            <Separator />
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium">Training loss</p>
                <p className="text-xs text-muted-foreground">
                  Control overlays and EMA smoothing.
                </p>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Smoothing</span>
                  <span className="tabular-nums">{smoothing.toFixed(2)}</span>
                </div>
                <Slider
                  value={[smoothing]}
                  onValueChange={([value]) => setSmoothing(value)}
                  min={0}
                  max={0.9}
                  step={0.01}
                />
                <p className="text-[11px] text-muted-foreground">
                  Move right for more smoothing. `0` = raw.
                </p>
              </div>
              <SettingRow
                label="Show raw loss"
                control={
                  <Switch checked={showRaw} onCheckedChange={setShowRaw} />
                }
              />
              <SettingRow
                label="Show smoothed loss"
                control={
                  <Switch
                    checked={showSmoothed}
                    onCheckedChange={setShowSmoothed}
                  />
                }
              />
              <SettingRow
                label="Show average line"
                control={
                  <Switch
                    checked={showAvgLine}
                    onCheckedChange={setShowAvgLine}
                  />
                }
              />
            </div>
            <Separator />
            <ScaleSection
              title="Loss axis"
              scale={lossScale}
              setScale={setLossScale}
              outlierMode={lossOutlierMode}
              setOutlierMode={setLossOutlierMode}
            />
            <Separator />
            <ScaleSection
              title="Gradient norm axis"
              scale={gradScale}
              setScale={setGradScale}
              outlierMode={gradOutlierMode}
              setOutlierMode={setGradOutlierMode}
            />
            <Separator />
            <ScaleSection
              title="Learning rate axis"
              scale={lrScale}
              setScale={setLrScale}
              outlierMode={lrOutlierMode}
              setOutlierMode={setLrOutlierMode}
            />
          </div>
          <SheetFooter className="mt-0 border-t border-border/60 bg-background/70 sm:flex-row sm:justify-between">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={resetPreferences}
            >
              Reset defaults
            </Button>
            <Button type="button" size="sm" onClick={() => setOpen(false)}>
              Done
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  );
}
