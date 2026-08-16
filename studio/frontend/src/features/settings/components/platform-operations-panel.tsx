import { Button } from "@/components/ui/button";
import {
  type PlatformOperationsStatus,
  type PlatformUsageStats,
  getPlatformOperationsStatus,
  getPlatformUiError,
  getPlatformUsageStats,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useRef, useState } from "react";
import { SettingsSection } from "./settings-section";

function latest(series: { value: number }[]): number {
  return series.at(-1)?.value ?? 0;
}

export function PlatformOperationsPanel() {
  const [status, setStatus] = useState<PlatformOperationsStatus | null>(null);
  const [stats, setStats] = useState<PlatformUsageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadController = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    const [statusResult, statsResult] = await Promise.allSettled([
      getPlatformOperationsStatus(signal),
      getPlatformUsageStats({ signal }),
    ]);
    if (signal?.aborted) return;
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    if (statsResult.status === "fulfilled") setStats(statsResult.value);
    const failure =
      statusResult.status === "rejected"
        ? statusResult.reason
        : statsResult.status === "rejected"
          ? statsResult.reason
          : null;
    if (failure) setError(getPlatformUiError(failure).message);
    setLoading(false);
  }, []);

  const reload = useCallback(() => {
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;
    void load(controller.signal);
  }, [load]);

  useEffect(() => {
    reload();
    return () => loadController.current?.abort();
  }, [reload]);

  return (
    <SettingsSection title="Operasyon görünümü">
      <div className="flex items-center justify-between gap-3 py-2">
        <p className="text-xs text-muted-foreground">
          Bağımlılık durumu ve son yedi günlük toplu kullanım metrikleri.
        </p>
        <Button
          size="sm"
          variant="outline"
          disabled={loading}
          onClick={reload}
        >
          Yenile
        </Button>
      </div>
      {loading && !status && !stats ? (
        <div
          aria-label="Operasyon bilgileri yükleniyor"
          className="h-24 animate-pulse rounded-md bg-muted/40"
        />
      ) : (
        <div className="grid gap-4">
          {status && (
            <div className="grid gap-2 sm:grid-cols-2">
              {status.services.map((service) => (
                <div
                  key={service.id}
                  className="rounded-lg border border-border/60 p-3"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="capitalize text-sm font-medium">
                      {service.label}
                    </span>
                    <span
                      className={
                        service.status === "healthy"
                          ? "text-xs text-emerald-600"
                          : "text-xs text-amber-600"
                      }
                    >
                      {service.status === "healthy" ? "Sağlıklı" : "Kısıtlı"}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {[
                      service.type,
                      service.latencyMs !== null
                        ? `${service.latencyMs.toFixed(1)} ms`
                        : null,
                    ]
                      .filter(Boolean)
                      .join(" · ") || "Ayrıntı yok"}
                  </p>
                </div>
              ))}
              <div className="rounded-lg border border-border/60 p-3">
                <div className="text-sm font-medium">Görev yürütücüleri</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {status.taskExecutorCount} etkin yürütücü
                </p>
              </div>
            </div>
          )}
          {stats && (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {[
                ["Görüntüleme", latest(stats.pageViews)],
                ["Kullanıcı", latest(stats.uniqueVisitors)],
                ["Tur", latest(stats.rounds)],
                ["Token (bin)", latest(stats.tokensThousands)],
                ["Token/sn", latest(stats.speed)],
                ["Olumlu oy", latest(stats.thumbsUp)],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg bg-muted/30 p-3">
                  <div className="text-xs text-muted-foreground">{label}</div>
                  <div className="mt-1 font-mono text-sm tabular-nums">
                    {Number(value).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
          {!status && !stats && !error && (
            <p className="py-4 text-center text-sm text-muted-foreground">
              Henüz operasyon verisi yok.
            </p>
          )}
        </div>
      )}
      {error && (
        <p role="alert" className="py-2 text-xs text-destructive">
          {error}
        </p>
      )}
    </SettingsSection>
  );
}
