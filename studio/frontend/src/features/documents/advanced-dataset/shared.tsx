import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { isPlatformApiError } from "@/integrations/platform-backend";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef, useState } from "react";

export type AsyncState = "loading" | "ready" | "empty" | "error" | "permission";

export function errorState(
  error: unknown,
): Exclude<AsyncState, "loading" | "ready" | "empty"> {
  return isPlatformApiError(error) &&
    (error.httpStatus === 401 ||
      error.httpStatus === 403 ||
      Number(error.code) === 109 ||
      Number(error.code) === 401 ||
      Number(error.code) === 403)
    ? "permission"
    : "error";
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function useAbortableLoad<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  isEmpty?: (value: T) => boolean,
) {
  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<AsyncState>("loading");
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const isEmptyRef = useRef(isEmpty);
  isEmptyRef.current = isEmpty;

  const load = useCallback(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    setState("loading");
    setError(null);
    void loader(controller.signal)
      .then((value) => {
        if (generation.current !== current) return;
        setData(value);
        setState(isEmptyRef.current?.(value) ? "empty" : "ready");
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || generation.current !== current) return;
        setError(errorMessage(cause));
        setState(errorState(cause));
      });
    return controller;
  }, [loader]);

  useEffect(() => {
    const controller = load();
    return () => {
      generation.current += 1;
      controller.abort();
    };
  }, [load]);

  return { data, error, load, setData, setState, state };
}

export function PanelState({
  state,
  error,
  empty = "Henüz kayıt yok.",
  onRetry,
}: {
  state: AsyncState;
  error?: string | null;
  empty?: string;
  onRetry?: () => void;
}) {
  if (state === "ready") return null;
  if (state === "loading")
    return (
      <div
        role="status"
        className="flex min-h-40 items-center justify-center gap-2 text-sm text-muted-foreground"
      >
        <Spinner /> Yükleniyor…
      </div>
    );
  const title =
    state === "permission"
      ? "Bu alan için yetkiniz yok."
      : state === "empty"
        ? empty
        : "Veri yüklenemedi.";
  return (
    <div
      className={cn(
        "flex min-h-40 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed p-6 text-center",
        state === "error" && "border-destructive/40",
      )}
    >
      <p
        role={state === "error" ? "alert" : "status"}
        className="text-sm font-medium"
      >
        {title}
      </p>
      {error ? (
        <p className="max-w-xl text-xs text-muted-foreground">{error}</p>
      ) : null}
      {onRetry && state !== "permission" ? (
        <Button size="sm" variant="outline" onClick={onRetry}>
          Yeniden dene
        </Button>
      ) : null}
    </div>
  );
}

export function Field({
  label,
  children,
  hint,
}: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="grid min-w-0 gap-1.5 text-xs font-medium text-muted-foreground">
      {label}
      {children}
      {hint ? <span className="font-normal">{hint}</span> : null}
    </label>
  );
}

export const inputClass =
  "h-9 min-w-0 w-full rounded-xl border bg-background px-3 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring";
export const textareaClass =
  "min-h-28 min-w-0 w-full rounded-xl border bg-background px-3 py-2 font-mono text-xs text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring";

export function SectionCard({
  title,
  description,
  children,
  actions,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <section className="min-w-0 overflow-hidden rounded-2xl border bg-card p-3 shadow-sm sm:p-4">
      <header className="mb-4 flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">{title}</h3>
          {description ? (
            <p className="mt-1 text-xs text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {actions ? (
          <div className="flex max-w-full shrink-0 flex-wrap items-center gap-2">
            {actions}
          </div>
        ) : null}
      </header>
      {children}
    </section>
  );
}
