import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { SectionCard } from "@/components/section-card";
import { MODELS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon, PackageIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import { ExportDialog } from "./components/export-dialog";
import { MethodPicker } from "./components/method-picker";
import { QuantPicker } from "./components/quant-picker";
import {
  type ExportMethod,
  GUIDE_STEPS,
  METHOD_LABELS,
  getEstimatedSize,
} from "./constants";

const anim = {
  initial: { height: 0, opacity: 0 },
  animate: { height: "auto" as const, opacity: 1 },
  exit: { height: 0, opacity: 0 },
  transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] as const },
};

export function ExportPage() {
  const store = useWizardStore();
  const isAdapter = store.trainingMethod === "lora" || store.trainingMethod === "qlora";
  const modelInfo = useMemo(
    () => MODELS.find((m) => m.id === store.selectedModel),
    [store.selectedModel],
  );

  const checkpoints = useMemo(() => {
    if (isAdapter) {
      const interval = store.saveSteps > 0 ? store.saveSteps : 100;
      const total = store.trainingMetrics?.totalSteps ?? 500;
      const entries: { value: string; label: string; detail: string }[] = [];
      for (let step = interval; step <= total; step += interval) {
        const loss = (1.5 - (step / total) * 0.7 + Math.random() * 0.05).toFixed(2);
        entries.push({
          value: `checkpoint-${step}`,
          label: `checkpoint-${step}`,
          detail: step === total ? `Best Loss: ${loss}` : `Loss: ${loss}`,
        });
      }
      return entries.reverse();
    }
    return [{ value: "final-model", label: "Final Model", detail: "Full fine-tuned weights" }];
  }, [isAdapter, store.saveSteps, store.trainingMetrics?.totalSteps]);

  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  const [exportMethod, setExportMethod] = useState<ExportMethod | null>(null);
  const [quantLevels, setQuantLevels] = useState<string[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);

  const [destination, setDestination] = useState<"local" | "hub">("local");
  const [hfUsername, setHfUsername] = useState("");
  const [modelName, setModelName] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);

  const handleMethodChange = (method: ExportMethod) => {
    setExportMethod(method);
    if (method !== "gguf") setQuantLevels([]);
  };

  const estimatedSize = getEstimatedSize(exportMethod, quantLevels);
  const canExport = checkpoint && exportMethod && (exportMethod !== "gguf" || quantLevels.length > 0);
  const baseModelName = modelInfo?.name ?? store.selectedModel ?? "—";

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8 flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">Export Model</h1>
          <p className="text-sm text-muted-foreground">Export your fine-tuned model for deployment</p>
        </div>

        <SectionCard
          icon={<HugeiconsIcon icon={PackageIcon} className="size-5" />}
          title="Export Configuration"
          description="Select checkpoint, method, and quantization"
          accent="emerald"
          featured
          className="shadow-border ring-1 ring-border"
        >
          {/* Top row: Checkpoint + metadata | Guide */}
          <div className="grid grid-cols-2 gap-8">
            <div className="flex flex-col gap-4 ">
              <div className="flex flex-col gap-2">
                <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  {isAdapter ? "Checkpoint" : "Model"}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button type="button" className="text-foreground/70 hover:text-foreground">
                        <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Choose a saved checkpoint to export. Lower loss generally means better quality.{" "}
                      <a href="https://unsloth.ai/docs/basics/inference-and-deployment" target="_blank" rel="noopener noreferrer" className="text-primary underline">Read more</a>
                    </TooltipContent>
                  </Tooltip>
                </label>
                <Select value={checkpoint ?? ""} onValueChange={setCheckpoint}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={isAdapter ? "Select a checkpoint…" : "Select model…"} />
                  </SelectTrigger>
                  <SelectContent>
                    {checkpoints.map((cp) => (
                      <SelectItem key={cp.value} value={cp.value}>
                        <span className="flex items-center gap-2">
                          {cp.label}
                          <span className="text-muted-foreground text-xs">{cp.detail}</span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="rounded-xl bg-muted/50 p-3 flex flex-col gap-2">
                <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Training Info</span>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Base Model</span>
                    <span className="font-medium">{baseModelName}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Method</span>
                    <span className="font-medium">{METHOD_LABELS[store.trainingMethod] ?? store.trainingMethod}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Checkpoints</span>
                    <span className="font-medium">{checkpoints.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Epochs</span>
                    <span className="font-medium">{store.epochs}</span>
                  </div>
                  {isAdapter && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">LoRA Rank</span>
                      <span className="font-medium">{store.loraRank}</span>
                    </div>
                  )}
                  {modelInfo?.params && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Params</span>
                      <span className="font-medium">{modelInfo.params}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-col  gap-2.5">
              <span className="text-xs font-medium text-muted-foreground">Quick Guide</span>
              <ol className="flex flex-col gap-3">
                {GUIDE_STEPS.map((step, i) => (
                  <li key={step} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold">
                      {i + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <MethodPicker value={exportMethod} onChange={handleMethodChange} />

          <AnimatePresence>
            {exportMethod === "gguf" && (
              <motion.div {...anim} className="overflow-hidden">
                <QuantPicker value={quantLevels} onChange={setQuantLevels} />
              </motion.div>
            )}
          </AnimatePresence>

          <Separator />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
              <span>Est. size: {estimatedSize} · Free disk space: 120 GB</span>
            </div>
            <Button disabled={!canExport} onClick={() => setDialogOpen(true)}>
              Export Model
            </Button>
          </div>
        </SectionCard>
      </main>

      <ExportDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        checkpoint={checkpoint}
        exportMethod={exportMethod}
        quantLevels={quantLevels}
        estimatedSize={estimatedSize}
        baseModelName={baseModelName}
        isAdapter={isAdapter}
        destination={destination}
        onDestinationChange={setDestination}
        hfUsername={hfUsername}
        onHfUsernameChange={setHfUsername}
        modelName={modelName}
        onModelNameChange={setModelName}
        hfToken={store.hfToken}
        onHfTokenChange={store.setHfToken}
        privateRepo={privateRepo}
        onPrivateRepoChange={setPrivateRepo}
      />
    </div>
  );
}
