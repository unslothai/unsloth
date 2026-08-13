import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import {
  type PlatformPipeline,
  getPipelineDsl,
  isPlatformApiError,
  listPipelines,
  mapPipelineToDatasetFields,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useState } from "react";

export function PlatformPipelineSelect({
  disabled,
  onChange,
  value,
}: {
  disabled?: boolean;
  onChange: (pipelineId: string) => void;
  value: string;
}) {
  const [pipelines, setPipelines] = useState<PlatformPipeline[]>([]);
  const [state, setState] = useState<
    "loading" | "ready" | "runtime-disabled" | "error"
  >("loading");
  const [validating, setValidating] = useState(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    setState("loading");
    try {
      setPipelines(await listPipelines(signal));
      setState("ready");
    } catch (error) {
      if (isPlatformApiError(error) && error.isAbort) return;
      setState(
        isPlatformApiError(error) && error.httpStatus === 404
          ? "runtime-disabled"
          : "error",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return (
    <div className="grid gap-2">
      <Label htmlFor="kb-platform-pipeline">Pipeline</Label>
      {state === "loading" ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Spinner /> Pipeline kataloğu yükleniyor…
        </div>
      ) : state === "runtime-disabled" ? (
        <p role="status" className="text-sm text-amber-600">
          Pipeline seçimi aktif hybrid proxy runtime’ında kullanılamıyor (route
          HTTP 404). Dataset yerleşik parser ile oluşturulacak.
        </p>
      ) : state === "error" ? (
        <div className="flex items-center gap-2">
          <p className="text-sm text-destructive">
            Pipeline kataloğu okunamadı.
          </p>
          <Button size="sm" variant="outline" onClick={() => void load()}>
            Yeniden dene
          </Button>
        </div>
      ) : (
        <>
          <select
            id="kb-platform-pipeline"
            className="h-9 rounded-full border bg-background px-3 text-sm"
            value={value}
            disabled={disabled || validating || pipelines.length === 0}
            onChange={(event) => {
              const next = event.target.value;
              // Validate and normalize against the exact dataset contract before
              // the selection leaves this UI boundary.
              const mapped = mapPipelineToDatasetFields(next);
              if (!mapped) {
                onChange("");
                return;
              }
              setValidating(true);
              void getPipelineDsl(mapped.pipeline_id)
                .then(() => onChange(mapped.pipeline_id))
                .catch((error: unknown) => {
                  if (!isPlatformApiError(error) || !error.isAbort)
                    setState("error");
                })
                .finally(() => setValidating(false));
            }}
          >
            <option value="">Yerleşik parser</option>
            {pipelines.map((pipeline) => (
              <option key={pipeline.id} value={pipeline.id}>
                {pipeline.title}
              </option>
            ))}
          </select>
          {validating && (
            <p className="text-xs text-muted-foreground">
              Pipeline sözleşmesi doğrulanıyor…
            </p>
          )}
          {pipelines.length === 0 && (
            <p className="text-sm text-muted-foreground">
              Pipeline bulunamadı.
            </p>
          )}
        </>
      )}
    </div>
  );
}
