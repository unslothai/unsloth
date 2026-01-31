import {
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CONTEXT_LENGTHS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";

export function HyperparametersStep() {
  const {
    trainingMethod,
    epochs,
    setEpochs,
    contextLength,
    setContextLength,
    learningRate,
    setLearningRate,
    loraRank,
    setLoraRank,
    loraAlpha,
    setLoraAlpha,
    loraDropout,
    setLoraDropout,
  } = useWizardStore(
    useShallow((s) => ({
      trainingMethod: s.trainingMethod,
      epochs: s.epochs,
      setEpochs: s.setEpochs,
      contextLength: s.contextLength,
      setContextLength: s.setContextLength,
      learningRate: s.learningRate,
      setLearningRate: s.setLearningRate,
      loraRank: s.loraRank,
      setLoraRank: s.setLoraRank,
      loraAlpha: s.loraAlpha,
      setLoraAlpha: s.setLoraAlpha,
      loraDropout: s.loraDropout,
      setLoraDropout: s.setLoraDropout,
    })),
  );

  const showLoraParams =
    trainingMethod === "lora" || trainingMethod === "qlora";

  return (
    <FieldGroup>
      <FieldSet>
        <FieldLegend variant="label">Training</FieldLegend>
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
              Epochs
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-muted-foreground/50 hover:text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3.5"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Number of times to iterate over the entire dataset.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </FieldLabel>
            <div className="flex items-center gap-3">
              <Slider
                value={[epochs]}
                onValueChange={([v]) => setEpochs(v)}
                min={1}
                max={10}
                step={1}
                className="w-40"
              />
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                min={1}
                max={10}
                step={1}
                className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
              Context Length
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-muted-foreground/50 hover:text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3.5"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Maximum number of tokens per training sample.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </FieldLabel>
            <Select
              value={String(contextLength)}
              onValueChange={(v) => setContextLength(Number(v))}
            >
              <SelectTrigger className="w-32 font-mono">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CONTEXT_LENGTHS.map((len) => (
                  <SelectItem key={len} value={String(len)}>
                    {len.toLocaleString()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
              Learning Rate
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-muted-foreground/50 hover:text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3.5"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Step size for weight updates. Lower = slower but more stable.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </FieldLabel>
            <Input
              type="number"
              step="0.00001"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-32 font-mono"
            />
          </div>
        </div>
      </FieldSet>

      {showLoraParams && (
        <>
          <Separator />
          <FieldSet>
            <FieldLegend variant="label">LoRA Parameters</FieldLegend>
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Rank (r)
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Dimension of the low-rank matrices. Higher = more
                      capacity.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </TooltipContent>
                  </Tooltip>
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraRank]}
                    onValueChange={([v]) => setLoraRank(v)}
                    min={4}
                    max={128}
                    step={4}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraRank}
                    onChange={(e) => setLoraRank(Number(e.target.value))}
                    min={4}
                    max={128}
                    step={4}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Alpha
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Scaling factor. Typically set to 2x the rank value.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </TooltipContent>
                  </Tooltip>
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraAlpha]}
                    onValueChange={([v]) => setLoraAlpha(v)}
                    min={8}
                    max={256}
                    step={8}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    min={8}
                    max={256}
                    step={8}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Dropout
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Probability of dropping neurons during training for
                      regularization.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </TooltipContent>
                  </Tooltip>
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraDropout]}
                    onValueChange={([v]) => setLoraDropout(v)}
                    min={0}
                    max={0.5}
                    step={0.01}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraDropout}
                    onChange={(e) => setLoraDropout(Number(e.target.value))}
                    min={0}
                    max={0.5}
                    step={0.01}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>
            </div>
          </FieldSet>
        </>
      )}
    </FieldGroup>
  );
}
