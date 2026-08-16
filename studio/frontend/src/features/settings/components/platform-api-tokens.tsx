import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  type PlatformApiToken,
  type PlatformCreatedApiToken,
  createPlatformApiToken,
  getPlatformUiError,
  listPlatformApiTokens,
  revokePlatformApiToken,
} from "@/integrations/platform-backend";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";

function TokenReveal({
  created,
  onDone,
}: {
  created: PlatformCreatedApiToken;
  onDone: () => void;
}) {
  const credentials = [
    { label: "API token", value: created.token },
    ...(created.compatibilityToken
      ? [
          {
            label: "Uyumluluk token'ı",
            value: created.compatibilityToken,
          },
        ]
      : []),
  ];
  return (
    <section className="flex flex-col gap-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4">
      <div>
        <h2 className="text-sm font-semibold text-foreground">
          Yeni erişim token'ı oluşturuldu
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Şimdi kopyalayın. Bu değerler daha sonra yeniden gösterilmez.
        </p>
      </div>
      {credentials.map((credential) => (
        <div key={credential.label} className="flex flex-col gap-1">
          <span className="text-xs font-medium text-muted-foreground">
            {credential.label}
          </span>
          <button
            type="button"
            onClick={() => void copyToClipboard(credential.value)}
            className="flex min-w-0 items-center gap-3 rounded-md border border-border bg-background px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <code className="min-w-0 flex-1 break-all text-xs text-foreground">
              {credential.value}
            </code>
            <HugeiconsIcon icon={Copy01Icon} className="size-4 shrink-0" />
          </button>
        </div>
      ))}
      <Button type="button" size="sm" className="self-end" onClick={onDone}>
        Bitti
      </Button>
    </section>
  );
}

export function PlatformApiTokens() {
  const [tokens, setTokens] = useState<PlatformApiToken[]>([]);
  const [created, setCreated] = useState<PlatformCreatedApiToken | null>(null);
  const [loading, setLoading] = useState(true);
  const [mutating, setMutating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<PlatformApiToken | null>(
    null,
  );
  const loadController = useRef<AbortController | null>(null);
  const mutationController = useRef<AbortController | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      setTokens(await listPlatformApiTokens(signal));
    } catch (loadError) {
      if (signal?.aborted) return;
      setError(getPlatformUiError(loadError).message);
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
    };
  }, [reload]);

  const create = async () => {
    mutationController.current?.abort();
    const controller = new AbortController();
    mutationController.current = controller;
    setMutating(true);
    setError(null);
    try {
      const result = await createPlatformApiToken(controller.signal);
      setCreated(result);
      await load(controller.signal);
    } catch (createError) {
      if (!controller.signal.aborted) {
        setError(getPlatformUiError(createError).message);
      }
    } finally {
      if (!controller.signal.aborted) setMutating(false);
    }
  };

  const revoke = async () => {
    if (!revokeTarget) return;
    mutationController.current?.abort();
    const controller = new AbortController();
    mutationController.current = controller;
    setMutating(true);
    setError(null);
    try {
      await revokePlatformApiToken(revokeTarget.revokeKey, controller.signal);
      setRevokeTarget(null);
      await load(controller.signal);
    } catch (revokeError) {
      if (!controller.signal.aborted) {
        setError(getPlatformUiError(revokeError).message);
      }
    } finally {
      if (!controller.signal.aborted) setMutating(false);
    }
  };

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold font-heading">API token'ları</h1>
          <p className="mt-1 text-xs text-muted-foreground">
            Rag Platform API erişimini oluşturun ve iptal edin.
          </p>
        </div>
        <Button size="sm" onClick={() => void create()} disabled={mutating}>
          {mutating ? "İşleniyor…" : "Token oluştur"}
        </Button>
      </header>

      {created && (
        <TokenReveal created={created} onDone={() => setCreated(null)} />
      )}

      {error && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 rounded-md border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive"
        >
          <span>{error}</span>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading || mutating}
            onClick={reload}
          >
            Listeyi yenile
          </Button>
        </div>
      )}

      <section aria-label="API token listesi">
        {loading ? (
          <div
            aria-label="API token'ları yükleniyor"
            className="flex flex-col gap-2"
          >
            {[0, 1].map((item) => (
              <div
                key={item}
                className="h-14 animate-pulse rounded-md bg-muted/40"
              />
            ))}
          </div>
        ) : error && tokens.length === 0 ? (
          <div className="rounded-lg border border-dashed border-destructive/30 p-6 text-center text-sm text-muted-foreground">
            Token listesi yüklenemedi. Yukarıdaki bağlantı veya yetki hatasını
            giderip yeniden deneyin.
          </div>
        ) : tokens.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
            Henüz API token'ı yok.
          </div>
        ) : (
          <div className="flex flex-col divide-y divide-border/60">
            {tokens.map((token) => (
              <div key={token.id} className="flex items-center gap-4 py-3">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-foreground">
                    {token.label}
                  </div>
                  <div className="mt-0.5 flex flex-wrap gap-x-3 text-xs text-muted-foreground">
                    <code>{token.maskedToken}</code>
                    <span>
                      {token.createdAt
                        ? new Date(token.createdAt).toLocaleString()
                        : "Oluşturma zamanı bilinmiyor"}
                    </span>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={mutating}
                  onClick={() => setRevokeTarget(token)}
                  aria-label={`${token.label} token'ını iptal et`}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </section>

      <Dialog
        open={revokeTarget !== null}
        onOpenChange={(open) => !open && !mutating && setRevokeTarget(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>API token'ı iptal edilsin mi?</DialogTitle>
            <DialogDescription>
              Bu token'ı kullanan istemciler erişimini hemen kaybeder. Bu işlem
              geri alınamaz.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={mutating}
              onClick={() => setRevokeTarget(null)}
            >
              İptal
            </Button>
            <Button
              variant="destructive"
              disabled={mutating}
              onClick={() => void revoke()}
            >
              {mutating ? "İptal ediliyor…" : "Token'ı iptal et"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
