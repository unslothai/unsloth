import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { usePlatformSessionStore } from "@/integrations/platform-backend";
import { downloadFile } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";
import {
  dryRunPlatformChatMigration,
  runPlatformChatMigration,
  serializePlatformChatMigrationExport,
  type PlatformChatMigrationPlan,
  type PlatformChatMigrationProgress,
} from "./platform-chat-migration";

export function PlatformChatMigrationPanel() {
  const user = usePlatformSessionStore((state) => state.user);
  const ownerId = user?.id || user?.email || "signed-in-user";
  const [plan, setPlan] = useState<PlatformChatMigrationPlan | null>(null);
  const [progress, setProgress] = useState<PlatformChatMigrationProgress | null>(null);
  const [busy, setBusy] = useState<"scan" | "migrate" | "export" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const scan = async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy("scan");
    setError(null);
    try {
      setPlan(await dryRunPlatformChatMigration(ownerId, controller.signal));
    } catch (scanError) {
      if (!controller.signal.aborted) {
        setError(scanError instanceof Error ? scanError.message : String(scanError));
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      if (!controller.signal.aborted) setBusy(null);
    }
  };

  const exportPlan = async () => {
    if (!plan) return;
    setBusy("export");
    try {
      await downloadFile(
        serializePlatformChatMigrationExport(plan),
        `rag-platform-chat-migration-${new Date().toISOString().slice(0, 10)}.json`,
        "application/json;charset=utf-8",
      );
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : String(exportError));
    } finally {
      setBusy(null);
    }
  };

  const migrate = async () => {
    if (!plan || plan.totals.pending === 0) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy("migrate");
    setError(null);
    setProgress({ completed: 0, total: plan.projects.length + plan.threads.length, current: "Hazırlanıyor" });
    try {
      const result = await runPlatformChatMigration(plan, {
        signal: controller.signal,
        onProgress: setProgress,
      });
      if (result.failures.length > 0) {
        setError(`${result.failures.length} kayıt taşınamadı. Yeniden çalıştırarak güvenle devam edebilirsiniz.`);
      } else if (result.aborted) {
        toast.info("Migration durduruldu; tamamlanan kayıtlar korundu.");
      } else {
        toast.success("Desteklenen sohbet kayıtları Rag Platform'a taşındı.");
      }
      if (!controller.signal.aborted) {
        setPlan(await dryRunPlatformChatMigration(ownerId, controller.signal));
      }
    } catch (migrationError) {
      if (!controller.signal.aborted) {
        setError(migrationError instanceof Error ? migrationError.message : String(migrationError));
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      setBusy(null);
    }
  };

  return (
    <div className="space-y-3 rounded-xl border bg-muted/15 p-4" aria-live="polite">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">Rag Platform sohbet migration</h3>
        <p className="text-xs text-muted-foreground">
          Önce dry-run raporu üretin ve export alın. Eski veri doğrulama tamamlanmadan silinmez.
        </p>
      </div>

      {busy === "scan" ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Spinner className="size-4" /> Eski kayıtlar taranıyor…
        </div>
      ) : plan ? (
        <div className="grid gap-2 text-xs sm:grid-cols-4">
          <span>Proje: {plan.totals.projects}</span>
          <span>Sohbet: {plan.totals.threads}</span>
          <span>Mesaj/export: {plan.totals.messages}</span>
          <span>Bekleyen: {plan.totals.pending}</span>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">Henüz dry-run yapılmadı.</p>
      )}

      {plan?.snapshot.sourceWarnings.map((warning) => (
        <p key={warning} className="text-xs text-amber-700 dark:text-amber-300">{warning}</p>
      ))}
      {plan?.unsupported.map((item) => (
        <p key={`${item.kind}-${item.reason}`} className="text-xs text-muted-foreground">
          {item.count} kayıt: {item.reason}
        </p>
      ))}
      {progress && busy === "migrate" ? (
        <p className="text-xs text-muted-foreground">
          {progress.completed}/{progress.total} · {progress.current}
        </p>
      ) : null}
      {error ? <p role="alert" className="text-xs text-destructive">{error}</p> : null}

      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" onClick={() => void scan()} disabled={busy !== null}>
          Dry-run
        </Button>
        <Button variant="outline" size="sm" onClick={() => void exportPlan()} disabled={!plan || busy !== null}>
          Export al
        </Button>
        <Button size="sm" onClick={() => void migrate()} disabled={!plan || plan.totals.pending === 0 || busy !== null}>
          {plan?.totals.alreadyMigrated ? "Devam et" : "Migration başlat"}
        </Button>
        {busy === "migrate" ? (
          <Button variant="destructive" size="sm" onClick={() => controllerRef.current?.abort()}>
            Durdur
          </Button>
        ) : null}
      </div>
    </div>
  );
}
