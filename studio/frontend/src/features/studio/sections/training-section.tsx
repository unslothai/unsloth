import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { useWizardStore } from "@/stores/training";
import {
  Archive04Icon,
  ArrowDown01Icon,
  ChartAverageIcon,
  CleanIcon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

const chartConfig = {
  loss: { label: "Loss", color: "#3b82f6" },
} satisfies ChartConfig;

const placeholderData = [
  { step: 0, loss: 2.5 },
  { step: 10, loss: 2.1 },
  { step: 20, loss: 1.7 },
  { step: 30, loss: 1.3 },
  { step: 40, loss: 1.0 },
  { step: 50, loss: 0.8 },
];

export function TrainingSection() {
  const store = useWizardStore();
  const [logOpen, setLogOpen] = useState(false);

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title="Training"
      description="Monitor and control training"
      accent="blue"
      className="lg:col-span-4 min-h-[450px]"
    >
      <div className="flex flex-col gap-4">
        {/* Loss chart */}
        <div className="relative  ">
          <ChartContainer
            config={chartConfig}
            className="min-h-[180px] w-full relative right-8 w-full blur "
          >
            <LineChart data={placeholderData} accessibilityLayer={true}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
            <HugeiconsIcon
              icon={ChartAverageIcon}
              className="size-5 text-muted-foreground/50"
            />
            <p className="text-sm font-medium text-muted-foreground">
              No training data yet
            </p>
            <p className="text-xs text-muted-foreground/60">
              Start training to see loss progress
            </p>
          </div>
        </div>

        {/* Start/Stop */}
        <Button
          className="w-full cursor-pointer bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600"
          onClick={() => store.setIsTraining(true)}
        >
          <HugeiconsIcon icon={Rocket01Icon} className="size-4" /> Start
          Training
        </Button>

        {/* Save / Clear */}
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm" className="cursor-pointer">
            <HugeiconsIcon icon={Archive04Icon} className="size-3.5" /> Save
            Config
          </Button>
          <Button variant="outline" size="sm" className="cursor-pointer">
            <HugeiconsIcon icon={CleanIcon} className="size-3.5" /> Clear
          </Button>
        </div>

        {/* Logging */}
        <Collapsible open={logOpen} onOpenChange={setLogOpen}>
          <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              className={`size-3.5 transition-transform ${logOpen ? "rotate-180" : ""}`}
            />
            Logging
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 flex flex-col gap-3">
            {/* W&B */}
            <div className="flex items-center gap-2">
              <Checkbox
                id="wandb"
                checked={store.enableWandb}
                onCheckedChange={(v) => store.setEnableWandb(!!v)}
              />
              <label
                htmlFor="wandb"
                className="text-xs cursor-pointer text-muted-foreground"
              >
                Weights & Biases
              </label>
            </div>
            {store.enableWandb && (
              <div className="flex flex-col gap-2 pl-6">
                <Input
                  placeholder="W&B API Token"
                  type="password"
                  value={store.wandbToken}
                  onChange={(e) => store.setWandbToken(e.target.value)}
                />
                <Input
                  placeholder="Project name"
                  value={store.wandbProject}
                  onChange={(e) => store.setWandbProject(e.target.value)}
                />
              </div>
            )}

            {/* TensorBoard */}
            <div className="flex items-center gap-2">
              <Checkbox
                id="tensorboard"
                checked={store.enableTensorboard}
                onCheckedChange={(v) => store.setEnableTensorboard(!!v)}
              />
              <label
                htmlFor="tensorboard"
                className="text-xs cursor-pointer text-muted-foreground"
              >
                TensorBoard
              </label>
            </div>
            {store.enableTensorboard && (
              <div className="flex flex-col gap-2 pl-6">
                <Input
                  placeholder="Log directory"
                  value={store.tensorboardDir}
                  onChange={(e) => store.setTensorboardDir(e.target.value)}
                />
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-muted-foreground">
                    Log frequency
                  </span>
                  <Input
                    type="number"
                    value={store.logFrequency}
                    onChange={(e) =>
                      store.setLogFrequency(Number(e.target.value))
                    }
                    className="w-24"
                  />
                </div>
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>
      </div>
    </SectionCard>
  );
}
