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
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { CpuIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
  "auto-round": "AutoRound",
};

/** Root-mounted: a Default-engine load only vLLM / SGLang can run waits here for an engine. */
export function ManagedEngineOfferDialog() {
  const open = useManagedEngineOfferStore((s) => s.open);
  const modelName = useManagedEngineOfferStore((s) => s.modelName);
  const offer = useManagedEngineOfferStore((s) => s.offer);
  const resolve = useManagedEngineOfferStore((s) => s.resolve);
  const { engines, error } = useEngines(open);
  const offered = (offer?.engines ?? []).flatMap((name) => {
    const engine = engines.find((row) => row.engine === name);
    return engine ? [engine] : [];
  });
  const ready = offered.filter((engine) => isEngineReady(engine));
  // Install sections stay mounted while open, so even an install that finishes between polls
  // still reaches its success effect and hands the load to that engine.
  const [shown, setShown] = useState<readonly string[]>([]);
  const unshown = offered
    .filter((engine) => !shown.includes(engine.engine))
    .map((engine) => engine.engine);
  if (!open && shown.length > 0) setShown([]);
  else if (open && ready.length === 0 && unshown.length > 0)
    setShown([...shown, ...unshown]);
  const visible = offered.filter(
    (engine) => ready.length === 0 || shown.includes(engine.engine),
  );
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
      <AlertDialogContent size="sm" className="data-[size=sm]:max-w-md">
        <AlertDialogHeader className="min-w-0">
          <AlertDialogMedia className="size-12">
            <HugeiconsIcon
              icon={CpuIcon}
              strokeWidth={1.75}
              className="size-5 text-muted-foreground"
            />
          </AlertDialogMedia>
          <AlertDialogTitle className="min-w-0 break-words">
            {displayName} needs {names.join(" or ")}
          </AlertDialogTitle>
          <AlertDialogDescription>
            The default engine cannot run {quantization} models.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {offered.length === 0 ? (
          error ? (
            // The check keeps retrying; say why it is stuck rather than spinning silently.
            <p role="alert" className="text-center text-ui-12 text-destructive">
              Could not check installed engines: {error}
            </p>
          ) : (
            <p className="flex items-center justify-center gap-2 text-ui-12 text-muted-foreground">
              <Spinner className="size-3.5" />
              Checking installed engines...
            </p>
          )
        ) : (
          <div className="space-y-3">
            {visible.map((engine) => (
              <section
                key={engine.engine}
                aria-label={ENGINE_NAMES[engine.engine]}
                // The title already names a lone engine; a choice of two needs each labelled.
                className={cn(
                  "space-y-2",
                  visible.length > 1 && "rounded-2xl border p-3",
                )}
              >
                {visible.length > 1 && (
                  <p className="text-center text-ui-13 font-medium">
                    {ENGINE_NAMES[engine.engine]}
                  </p>
                )}
                <EngineInstall
                  engine={engine}
                  centered={true}
                  onUse={() => resolve(engine.engine)}
                />
              </section>
            ))}
          </div>
        )}
        <AlertDialogFooter className="group-data-[size=sm]/alert-dialog-content:flex sm:justify-center">
          <AlertDialogCancel variant="ghost">Cancel</AlertDialogCancel>
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
