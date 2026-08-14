import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  fetchDocumentArtifact,
  fetchDocumentImage,
  fetchDocumentPreview,
  hasControlCharacters,
  isInlineSafeContentType,
  listDocumentThumbnails,
  type PlatformAsset,
  type PlatformDocument,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { Download, FileWarning, ImageIcon, PackageOpen } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type AssetMode = "preview" | "media";

interface DocumentAssetDialogProps {
  document: PlatformDocument | null;
  mode: AssetMode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function assetUrl(asset: PlatformAsset | null): string | null {
  return asset ? URL.createObjectURL(asset.blob) : null;
}

function safeDownloadName(value: string): string {
  const sanitized = Array.from(value.replace(/[\\/:*?"<>|]/g, "_"), (character) =>
    hasControlCharacters(character) ? "_" : character,
  ).join("");
  return sanitized.slice(0, 180) || "document";
}

function triggerDownload(asset: PlatformAsset, name: string) {
  const url = URL.createObjectURL(asset.blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = safeDownloadName(name);
  anchor.rel = "noopener";
  anchor.click();
  URL.revokeObjectURL(url);
}

function AssetPreview({ asset, name }: { asset: PlatformAsset; name: string }) {
  const url = useMemo(() => assetUrl(asset), [asset]);
  const [text, setText] = useState<string | null>(null);
  const normalizedType = asset.contentType.split(";", 1)[0]?.toLowerCase() ?? "";

  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);

  useEffect(() => {
    let active = true;
    if (normalizedType.startsWith("text/")) {
      asset.blob.text().then((value) => {
        if (active) setText(value.slice(0, 250_000));
      });
    } else {
      setText(null);
    }
    return () => {
      active = false;
    };
  }, [asset, normalizedType]);

  if (normalizedType.startsWith("image/") && url) {
    return (
      <div className="flex min-h-80 items-center justify-center rounded-2xl bg-black/[0.03] p-4 dark:bg-white/[0.03]">
        <img src={url} alt={name} className="max-h-[62vh] max-w-full rounded-xl object-contain" />
      </div>
    );
  }
  if (normalizedType === "application/pdf" && url) {
    return <iframe title={`${name} önizlemesi`} src={url} className="h-[62vh] w-full rounded-2xl border" />;
  }
  if (normalizedType.startsWith("text/")) {
    return (
      <pre className="max-h-[62vh] overflow-auto whitespace-pre-wrap rounded-2xl border bg-muted/30 p-5 font-mono text-xs leading-6">
        {text ?? "Metin hazırlanıyor…"}
      </pre>
    );
  }
  return (
    <div className="flex min-h-72 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed text-center">
      <FileWarning className="size-8 text-muted-foreground" />
      <div>
        <p className="font-medium">Bu biçim tarayıcıda güvenle gösterilmiyor</p>
        <p className="mt-1 text-sm text-muted-foreground">Dosyayı indirerek yerel uygulamanızda açabilirsiniz.</p>
      </div>
      <Button variant="outline" onClick={() => triggerDownload(asset, name)}>
        <Download /> İndir
      </Button>
    </div>
  );
}

export function DocumentInlinePreview({ document }: { document: PlatformDocument }) {
  const [asset, setAsset] = useState<PlatformAsset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  const load = useCallback(() => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    setError(null);
    setAsset(null);
    void fetchDocumentPreview(document.id, controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) setAsset(next);
      })
      .catch((loadError: unknown) => {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
        if (requestRef.current === controller) requestRef.current = null;
      });
  }, [document.id]);

  useEffect(() => {
    load();
    return () => requestRef.current?.abort();
  }, [load]);

  if (loading) {
    return <div className="flex min-h-64 items-center justify-center"><Spinner className="size-5" /></div>;
  }
  if (error) {
    return (
      <div className="flex min-h-56 flex-col items-center justify-center gap-3 rounded-2xl bg-destructive/5 p-5 text-center">
        <FileWarning className="size-6 text-destructive" />
        <div>
          <p className="font-medium">İçerik açılamadı</p>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">{error}</p>
        </div>
        <Button size="sm" variant="outline" onClick={load}>Yeniden dene</Button>
      </div>
    );
  }
  if (!asset) {
    return (
      <div className="flex min-h-56 items-center justify-center text-sm text-muted-foreground">
        Önizlenebilir içerik bulunamadı.
      </div>
    );
  }
  return <AssetPreview asset={asset} name={document.name} />;
}

export function DocumentAssetDialog({
  document,
  mode,
  open,
  onOpenChange,
}: DocumentAssetDialogProps) {
  const [asset, setAsset] = useState<PlatformAsset | null>(null);
  const [thumbnail, setThumbnail] = useState<PlatformAsset | null>(null);
  const [artifact, setArtifact] = useState<PlatformAsset | null>(null);
  const [artifactName, setArtifactName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const contentRequestRef = useRef<AbortController | null>(null);
  const artifactRequestRef = useRef<AbortController | null>(null);

  const loadPreview = useCallback(async (signal: AbortSignal) => {
    if (!document) return;
    setLoading(true);
    setError(null);
    try {
      const next = await fetchDocumentPreview(document.id, signal);
      setAsset(next);
    } catch (loadError) {
      if (!signal.aborted) setError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      if (!signal.aborted) setLoading(false);
    }
  }, [document]);

  const loadMedia = useCallback(async (signal: AbortSignal) => {
    if (!document) return;
    setLoading(true);
    setError(null);
    try {
      const thumbnails = await listDocumentThumbnails([document.id], signal);
      const value = thumbnails[document.id];
      if (!value) {
        setThumbnail(null);
        return;
      }
      if (value.startsWith("data:image/")) {
        const response = await fetch(value, { signal });
        setThumbnail({ blob: await response.blob(), contentType: response.headers.get("content-type") ?? "image/png", disposition: null });
        return;
      }
      const imageId = value.includes("/documents/images/")
        ? decodeURIComponent(value.split("/documents/images/")[1] ?? "")
        : `${document.datasetId}-${value}`;
      setThumbnail(await fetchDocumentImage(imageId, signal));
    } catch (loadError) {
      if (!signal.aborted) setError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      if (!signal.aborted) setLoading(false);
    }
  }, [document]);

  useEffect(() => {
    if (!open || !document) {
      contentRequestRef.current?.abort();
      artifactRequestRef.current?.abort();
      return;
    }
    contentRequestRef.current?.abort();
    const controller = new AbortController();
    contentRequestRef.current = controller;
    setAsset(null);
    setThumbnail(null);
    setArtifact(null);
    if (mode === "preview") void loadPreview(controller.signal);
    else void loadMedia(controller.signal);
    return () => {
      controller.abort();
      if (contentRequestRef.current === controller) contentRequestRef.current = null;
    };
  }, [document, loadMedia, loadPreview, mode, open]);

  useEffect(() => () => {
    contentRequestRef.current?.abort();
    artifactRequestRef.current?.abort();
  }, []);

  const loadArtifact = async () => {
    const name = artifactName.trim();
    if (!name || name.includes("/") || name.includes("\\") || hasControlCharacters(name)) {
      toast.error("Geçerli ve yalnızca dosya adı içeren bir artifact adı girin.");
      return;
    }
    artifactRequestRef.current?.abort();
    const controller = new AbortController();
    artifactRequestRef.current = controller;
    setLoading(true);
    try {
      const next = await fetchDocumentArtifact(name, controller.signal);
      setArtifact(next);
      if (!isInlineSafeContentType(next.contentType)) {
        toast.info("Bu artifact güvenlik nedeniyle yalnızca indirilebilir.");
      }
    } catch (loadError) {
      toast.error("Artifact alınamadı", {
        description: loadError instanceof Error ? loadError.message : String(loadError),
      });
    } finally {
      if (!controller.signal.aborted) setLoading(false);
      if (artifactRequestRef.current === controller) artifactRequestRef.current = null;
    }
  };

  const mediaAsset = thumbnail ?? artifact;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>{mode === "preview" ? "Belge önizlemesi" : "Belge medyası"}</DialogTitle>
            {document?.suffix ? <Badge variant="outline">{document.suffix.toUpperCase()}</Badge> : null}
          </div>
          <DialogDescription>{document?.name ?? "Belge"}</DialogDescription>
        </DialogHeader>

        {mode === "media" ? (
          <div className="flex flex-wrap items-center gap-2 rounded-2xl border bg-muted/20 p-3">
            <PackageOpen className="size-4 text-muted-foreground" />
            <Input
              value={artifactName}
              onChange={(event) => setArtifactName(event.target.value)}
              placeholder="artifact-dosya-adi.pdf"
              aria-label="Artifact dosya adı"
              className="min-w-64 flex-1"
            />
            <Button variant="outline" onClick={() => void loadArtifact()} disabled={loading}>
              Artifact aç
            </Button>
          </div>
        ) : null}

        {loading ? (
          <div className="flex min-h-80 items-center justify-center"><Spinner className="size-6" /></div>
        ) : error ? (
          <div className="flex min-h-72 flex-col items-center justify-center gap-3 rounded-2xl border border-destructive/30 bg-destructive/5 p-6 text-center">
            <FileWarning className="size-7 text-destructive" />
            <p className="font-medium">İçerik açılamadı</p>
            <p className="max-w-lg text-sm text-muted-foreground">{error}</p>
            <Button variant="outline" onClick={() => {
              contentRequestRef.current?.abort();
              const controller = new AbortController();
              contentRequestRef.current = controller;
              void (mode === "preview" ? loadPreview(controller.signal) : loadMedia(controller.signal));
            }}>Yeniden dene</Button>
          </div>
        ) : mode === "preview" && asset ? (
          <AssetPreview asset={asset} name={document?.name ?? "document"} />
        ) : mode === "media" && mediaAsset ? (
          <div className="space-y-3">
            <AssetPreview asset={mediaAsset} name={artifact ? artifactName : `${document?.name ?? "Belge"} küçük resmi`} />
            <Button variant="outline" onClick={() => triggerDownload(mediaAsset, artifact ? artifactName : `${document?.name ?? "thumbnail"}.png`)}>
              <Download /> Medyayı indir
            </Button>
          </div>
        ) : (
          <div className="flex min-h-72 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed text-center">
            <ImageIcon className="size-8 text-muted-foreground" />
            <p className="font-medium">Bu belge için küçük resim bulunamadı</p>
            <p className="text-sm text-muted-foreground">Üretilen bir artifact varsa dosya adıyla güvenli olarak açabilirsiniz.</p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
