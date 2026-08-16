import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  type PlatformLangfuseConfig,
  createPlatformLangfuseConfig,
  deletePlatformLangfuseConfig,
  getPlatformLangfuseConfig,
  getPlatformUiError,
  updatePlatformLangfuseConfig,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useRef, useState } from "react";

export function PlatformLangfuseSettings() {
  const [config, setConfig] = useState<PlatformLangfuseConfig | null>(null);
  const [host, setHost] = useState("");
  const [publicKey, setPublicKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadController = useRef<AbortController | null>(null);
  const mutationController = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const next = await getPlatformLangfuseConfig(signal);
      setConfig(next);
      setHost(next?.host ?? "");
    } catch (loadError) {
      if (!signal?.aborted) setError(getPlatformUiError(loadError).message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  const reload = useCallback(() => {
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;
    void load(controller.signal);
  }, [load]);

  useEffect(() => {
    reload();
    return () => {
      loadController.current?.abort();
      mutationController.current?.abort();
      setSecretKey("");
    };
  }, [reload]);

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    mutationController.current?.abort();
    const controller = new AbortController();
    mutationController.current = controller;
    setSaving(true);
    setError(null);
    try {
      const input = { host, publicKey, secretKey };
      const next = config
        ? await updatePlatformLangfuseConfig(input, controller.signal)
        : await createPlatformLangfuseConfig(input, controller.signal);
      setConfig(next);
      setEditing(false);
      setPublicKey("");
      setSecretKey("");
    } catch (saveError) {
      if (!controller.signal.aborted)
        setError(getPlatformUiError(saveError).message);
    } finally {
      if (!controller.signal.aborted) setSaving(false);
    }
  };

  const remove = async () => {
    mutationController.current?.abort();
    const controller = new AbortController();
    mutationController.current = controller;
    setSaving(true);
    setError(null);
    try {
      await deletePlatformLangfuseConfig(controller.signal);
      setConfig(null);
      setHost("");
      setPublicKey("");
      setSecretKey("");
      setConfirmDelete(false);
    } catch (deleteError) {
      if (!controller.signal.aborted)
        setError(getPlatformUiError(deleteError).message);
    } finally {
      if (!controller.signal.aborted) setSaving(false);
    }
  };

  const showForm = !config || editing;
  return (
    <section
      className="mt-8 flex flex-col gap-4 rounded-xl border border-border/70 p-4"
      aria-labelledby="langfuse-title"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2
            id="langfuse-title"
            className="text-sm font-semibold text-foreground"
          >
            Langfuse gözlemlenebilirliği
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            İzleme bağlantısını doğrulayın. Gizli anahtar hiçbir zaman kalıcı
            frontend deposuna yazılmaz.
          </p>
        </div>
        {config && !editing && (
          <Button size="sm" variant="outline" onClick={() => setEditing(true)}>
            Yeniden yapılandır
          </Button>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 text-xs text-destructive"
        >
          <span>{error}</span>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading || saving}
            onClick={reload}
          >
            Yeniden yükle
          </Button>
        </div>
      )}
      {loading ? (
        <div
          aria-label="Langfuse yapılandırması yükleniyor"
          className="h-20 animate-pulse rounded-md bg-muted/40"
        />
      ) : showForm ? (
        <form className="grid gap-3" onSubmit={(event) => void save(event)}>
          <Input
            type="url"
            value={host}
            onChange={(event) => setHost(event.target.value)}
            placeholder="https://langfuse.example.com"
            aria-label="Langfuse adresi"
            autoComplete="off"
            required={true}
          />
          <Input
            value={publicKey}
            onChange={(event) => setPublicKey(event.target.value)}
            placeholder="Public key"
            aria-label="Langfuse public key"
            autoComplete="off"
            required={true}
          />
          <Input
            type="password"
            value={secretKey}
            onChange={(event) => setSecretKey(event.target.value)}
            placeholder="Secret key"
            aria-label="Langfuse secret key"
            autoComplete="new-password"
            required={true}
          />
          <div className="flex justify-end gap-2">
            {config && (
              <Button
                type="button"
                variant="ghost"
                disabled={saving}
                onClick={() => setEditing(false)}
              >
                İptal
              </Button>
            )}
            <Button type="submit" size="sm" disabled={saving}>
              {saving ? "Doğrulanıyor…" : config ? "Güncelle" : "Bağla"}
            </Button>
          </div>
        </form>
      ) : (
        <div className="grid gap-2 text-xs text-muted-foreground">
          <div>
            <span className="font-medium text-foreground">Host:</span>{" "}
            {config.host}
          </div>
          <div>
            <span className="font-medium text-foreground">Public key:</span>{" "}
            <code>{config.maskedPublicKey}</code>
          </div>
          <div>
            <span className="font-medium text-foreground">Proje:</span>{" "}
            {config.projectName ?? config.projectId ?? "Bağlı"}
          </div>
          {confirmDelete ? (
            <div className="mt-2 flex items-center justify-between gap-3 rounded-md border border-destructive/20 bg-destructive/5 p-3">
              <span>Langfuse bağlantısı kaldırılsın mı?</span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={saving}
                  onClick={() => setConfirmDelete(false)}
                >
                  Vazgeç
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  disabled={saving}
                  onClick={() => void remove()}
                >
                  {saving ? "Kaldırılıyor…" : "Kaldır"}
                </Button>
              </div>
            </div>
          ) : (
            <Button
              className="mt-2 self-start"
              size="sm"
              variant="ghost"
              onClick={() => setConfirmDelete(true)}
            >
              Bağlantıyı kaldır
            </Button>
          )}
        </div>
      )}
    </section>
  );
}
