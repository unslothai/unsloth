import { useWizardStore } from "@/stores/training";
import type { TrainingMetrics } from "@/types/training";
import { type ReactElement, useEffect, useRef } from "react";
import { ChartsSection } from "./sections/charts-section";
import { ProgressSection } from "./sections/progress-section";

function createInitialMetrics(
  totalSteps: number,
  totalEpochs: number,
  lr: number,
): TrainingMetrics {
  return {
    currentStep: 0,
    totalSteps,
    currentEpoch: 0,
    totalEpochs,
    currentLoss: 2.5,
    currentLR: lr * 0.1,
    gradNorm: 0,
    samplesPerSecond: 0,
    lossHistory: [],
    lrHistory: [],
    gradNormHistory: [],
    gpuUtil: 0,
    gpuTemp: 45,
    gpuVramUsed: 0,
    gpuVramTotal: 24,
    gpuPower: 50,
    elapsed: 0,
    status: "warmup",
  };
}

export function TrainingView(): ReactElement {
  const { maxSteps, epochs, learningRate, warmupSteps, setTrainingMetrics } =
    useWizardStore();
  const metricsRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chartsRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const totalSteps = maxSteps || 500;
    const totalEpochs = epochs || 3;
    const peakLR = learningRate;
    const warmup = warmupSteps || 20;

    setTrainingMetrics(createInitialMetrics(totalSteps, totalEpochs, peakLR));

    let step = 0;
    let elapsed = 0;

    const computeStep = () => {
      step++;
      if (step > totalSteps) {
        return null;
      }
      elapsed++;

      let lr: number;
      if (step < warmup) {
        lr = peakLR * (step / warmup);
      } else {
        const progress = (step - warmup) / (totalSteps - warmup);
        lr = peakLR * 0.5 * (1 + Math.cos(Math.PI * progress));
      }

      const baseLoss = 2.5 * Math.exp((-3 * step) / totalSteps) + 0.3;
      const noise = (Math.random() - 0.5) * 0.08;
      const loss = Math.max(0.1, baseLoss + noise);
      const status =
        step < warmup ? "warmup" : step % 100 === 0 ? "saving" : "training";
      const gradNorm = +(
        1.2 * Math.exp(-step / totalSteps) +
        0.1 +
        (Math.random() - 0.5) * 0.05
      ).toFixed(3);

      return {
        step,
        elapsed,
        lr,
        loss: +loss.toFixed(4),
        status: status as TrainingMetrics["status"],
        gradNorm,
      };
    };

    // Top card values — update every 1s
    metricsRef.current = setInterval(() => {
      const s = computeStep();
      if (!s) {
        if (metricsRef.current) {
          clearInterval(metricsRef.current);
        }
        if (chartsRef.current) {
          clearInterval(chartsRef.current);
        }
        return;
      }

      const prev = useWizardStore.getState().trainingMetrics;
      setTrainingMetrics({
        currentStep: s.step,
        totalSteps,
        currentEpoch:
          Math.floor((s.step / totalSteps) * totalEpochs * 100) / 100,
        totalEpochs,
        currentLoss: s.loss,
        currentLR: s.lr,
        gradNorm: s.gradNorm,
        samplesPerSecond: +(12 + (Math.random() - 0.5) * 2).toFixed(1),
        lossHistory: prev?.lossHistory ?? [],
        lrHistory: prev?.lrHistory ?? [],
        gradNormHistory: prev?.gradNormHistory ?? [],
        gpuUtil: Math.min(99, 85 + Math.round((Math.random() - 0.5) * 10)),
        gpuTemp: Math.min(89, 68 + Math.round((Math.random() - 0.5) * 6)),
        gpuVramUsed: +(18.2 + (Math.random() - 0.5) * 0.4).toFixed(1),
        gpuVramTotal: 24,
        gpuPower: Math.round(280 + (Math.random() - 0.5) * 30),
        elapsed: s.elapsed,
        status: s.status,
      });
    }, 1000);

    // Chart history — update every 5s
    chartsRef.current = setInterval(() => {
      const prev = useWizardStore.getState().trainingMetrics;
      if (!prev || prev.currentStep === 0) {
        return;
      }

      setTrainingMetrics({
        ...prev,
        lossHistory: [
          ...prev.lossHistory,
          { step: prev.currentStep, loss: prev.currentLoss },
        ],
        lrHistory: [
          ...prev.lrHistory,
          { step: prev.currentStep, lr: prev.currentLR },
        ],
        gradNormHistory: [
          ...prev.gradNormHistory,
          { step: prev.currentStep, gradNorm: prev.gradNorm },
        ],
      });
    }, 5000);

    return () => {
      if (metricsRef.current) {
        clearInterval(metricsRef.current);
      }
      if (chartsRef.current) {
        clearInterval(chartsRef.current);
      }
    };
  }, [epochs, learningRate, maxSteps, setTrainingMetrics, warmupSteps]);

  return (
    <div className="flex flex-col gap-6">
      <ProgressSection />
      <ChartsSection />
    </div>
  );
}
