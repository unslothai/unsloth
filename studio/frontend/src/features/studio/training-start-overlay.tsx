// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  AnimatedSpan,
  Terminal,
  TypingAnimation,
} from "@/components/ui/terminal";
import { useTrainingActions, useTrainingRuntimeStore } from "@/features/training";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState, type ReactElement } from "react";

type TrainingStartOverlayProps = {
  message: string
  currentStep: number
}

export function TrainingStartOverlay({
  message,
  currentStep,
}: TrainingStartOverlayProps): ReactElement {
  const { stopTrainingRun, dismissTrainingRun } = useTrainingActions();
  const isStarting = useTrainingRuntimeStore((s) => s.isStarting);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);

  useEffect(() => {
    if (!isStarting) {
      setCancelRequested(false);
    }
  }, [isStarting]);

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="pointer-events-auto relative flex w-[860px] max-w-[calc(100%-2rem)] flex-col items-center gap-4">
        <img
          src="/unsloth-gem.png"
          alt="Unsloth mascot"
          className="size-24 object-contain"
        />
        <div className="relative w-full">
          <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-3 top-3 z-10 size-7 cursor-pointer rounded-full text-muted-foreground/60 hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setCancelDialogOpen(true)}
              disabled={cancelRequested}
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            </Button>
            <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
              <AlertDialogHeader>
                <AlertDialogTitle>Cancel Training</AlertDialogTitle>
                <AlertDialogDescription>
                  Do you want to cancel the current training run?
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Continue Training</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={() => {
                    setCancelRequested(true);
                    setCancelDialogOpen(false);
                    useTrainingRuntimeStore.getState().setStopRequested(true);
                    void stopTrainingRun(false).then((ok) => {
                      if (ok) {
                        void dismissTrainingRun();
                      } else {
                        setCancelRequested(false);
                      }
                    });
                  }}
                >
                  Cancel Training
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Terminal
            className="w-full min-h-[390px] rounded-2xl px-7 py-6 text-left"
            startOnView={false}
          >
          <TypingAnimation
            duration={36}
            className="bg-gradient-to-r from-emerald-300 via-lime-300 to-teal-300 bg-clip-text font-semibold text-transparent"
          >
            {"> unsloth training starts..."}
          </TypingAnimation>
          <AnimatedSpan className="my-2">
            <pre className="whitespace-pre text-left text-muted-foreground">{`==((====))==
\\\\   /|
O^O/ \\_/ \\
\\        /
 "-____-"`}</pre>
          </AnimatedSpan>
          <TypingAnimation duration={44}>
            {"> Preparing model and dataset..."}
          </TypingAnimation>
          <TypingAnimation duration={44}>
            {"> We are getting everything ready for your run..."}
          </TypingAnimation>
          <AnimatedSpan className="mt-2 text-muted-foreground">
            {`> ${message || "starting training..."} | waiting for first step... (${currentStep})`}
          </AnimatedSpan>
          </Terminal>
        </div>
      </div>
    </div>
  )
}
