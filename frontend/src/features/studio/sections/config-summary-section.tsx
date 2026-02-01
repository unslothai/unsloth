import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { useWizardStore } from "@/stores/training";
import { Settings02Icon, StopIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function ConfigSummarySection() {
  const store = useWizardStore();

  const items = [
    {
      section: "Model",
      rows: [
        ["Model", store.selectedModel ?? "—"],
        ["Type", store.modelType ?? "—"],
        ["Method", store.trainingMethod],
      ],
    },
    {
      section: "Dataset",
      rows: [
        ["Source", store.datasetSource],
        ["Dataset", store.dataset ?? store.uploadedFile ?? "—"],
        ["Format", store.datasetFormat],
      ],
    },
    {
      section: "Hyperparams",
      rows: [
        ["Epochs", store.epochs],
        ["Batch size", store.batchSize],
        ["Learning rate", store.learningRate],
        ["Max steps", store.maxSteps],
        ["Context length", store.contextLength],
        ["Warmup steps", store.warmupSteps],
      ],
    },
    ...(store.trainingMethod !== "full"
      ? [
          {
            section: "LoRA",
            rows: [
              ["Rank", store.loraRank],
              ["Alpha", store.loraAlpha],
              ["Dropout", store.loraDropout],
              ["Variant", store.loraVariant],
            ],
          },
        ]
      : []),
  ];

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={Settings02Icon} className="size-5" />}
      title="Config"
      description="Training configuration"
      accent="indigo"
      className="lg:col-span-4"
    >
      <div className="flex flex-col gap-4">
        {items.map((group) => (
          <div key={group.section} className="flex flex-col gap-1">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {group.section}
            </p>
            {group.rows.map(([label, value]) => (
              <div key={String(label)} className="flex justify-between text-xs">
                <span className="text-muted-foreground">{String(label)}</span>
                <span className="font-medium tabular-nums">
                  {String(value)}
                </span>
              </div>
            ))}
          </div>
        ))}

        <Button
          variant="destructive"
          className="mt-2 w-full cursor-pointer"
          onClick={() => store.setIsTraining(false)}
        >
          <HugeiconsIcon icon={StopIcon} className="size-4" /> Stop Training
        </Button>
      </div>
    </SectionCard>
  );
}
