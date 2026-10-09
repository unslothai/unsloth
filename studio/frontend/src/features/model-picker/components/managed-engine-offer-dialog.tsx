// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Spinner } from "@/components/ui/spinner";
import { useState } from "react";
import { isEngineReady } from "../api/engines";
import { useManagedEngineOfferStore } from "../hooks/managed-engine-offer";
import { useEngines } from "../hooks/use-engines";
import { EngineInstall } from "./inference-engines";

const ENGINE_NAMES = { vllm: "vLLM", sglang: "SGLang" };
const QUANTIZATION_NAMES: Record<string, string> = {
  "compressed-tensors": "compressed-tensors",
  awq: "AWQ",
  gptq: "GPTQ",
};

/** Root-mounted: a Default-engine load only vLLM / SGLang can run waits here for an engine. */
export function ManagedEngineOfferDialog() {
  const open = useManagedEngineOfferStore((s) => s.open);
  const modelName = useManagedEngineOfferStore((s) => s.modelName);
  const offer = useManagedEngineOfferStore((s) => s.offer);
  const resolve = useManagedEngineOfferStore((s) => s.resolve);
  const { engines } = useEngines(open);
  const offered = (offer?.engines ?? []).flatMap((name) => {
    const engine = engines.find((row) => row.engine === name);
    return engine ? [engine] : [];
  });
  const ready = offered.filter((engine) => isEngineReady(engine));
  // An install started here stays mounted until its success hands the load to it.
  const [installing, setInstalling] = useState<readonly string[]>([]);
  const running = offered
    .filter((engine) => engine.job.state === "running")
    .map((engine) => engine.engine);
  const started = running.filter((name) => !installing.includes(name));
  if (!open && installing.length > 0) setInstalling([]);
  else if (open && started.length > 0)
    setInstalling([...installing, ...started]);
  const names = (offer?.engines ?? []).map((name) => ENGINE_NAMES[name]);
  const displayName = modelName?.split("/").pop() || "This model";
  const quantization =
    QUANTIZATION_NAMES[offer?.quantization ?? ""] ?? offer?.quantization;

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) resolve(null);
      }}
    >
      <AlertDialogContent className="max-w-lg">
        <AlertDialogHeader className="min-w-0">
          <AlertDialogTitle>
            {displayName} needs {names.join(" or ")}
          </AlertDialogTitle>
          <AlertDialogDescription>
            This model is quantized with {quantization}, which Unsloth's default
            engine cannot run. {names.join(" and ")}{" "}
            {names.length > 1 ? "can each" : "can"} run it on this computer
            {ready.length > 0 ? "." : " once installed."}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {offered.length === 0 ? (
          <p className="flex items-center gap-2 text-ui-12 text-muted-foreground">
            <Spinner className="size-3.5" />
            Checking installed engines...
          </p>
        ) : (
          offered
            .filter(
              (engine) =>
                ready.length === 0 || installing.includes(engine.engine),
            )
            .map((engine) => (
              <section
                key={engine.engine}
                aria-label={ENGINE_NAMES[engine.engine]}
                className="space-y-2 rounded-lg border p-3"
              >
                <p className="text-ui-13 font-medium">
                  {ENGINE_NAMES[engine.engine]}
                </p>
                <EngineInstall
                  engine={engine}
                  onUse={() => resolve(engine.engine)}
                />
              </section>
            ))
        )}
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          {ready.map((engine) => (
            <AlertDialogAction
              key={engine.engine}
              onClick={() => resolve(engine.engine)}
            >
              Load with {ENGINE_NAMES[engine.engine]}
            </AlertDialogAction>
          ))}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
