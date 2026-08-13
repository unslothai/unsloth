import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  type ModelSelector,
  type PlatformModel,
  chatToModel,
  createEmbeddings,
  isPlatformApiError,
  ocrFile,
  parseFile,
  rerankDocuments,
  synthesizeSpeech,
  transcribeAudio,
} from "@/integrations/platform-backend";
import { useEffect, useMemo, useRef, useState } from "react";

type ToolId =
  | "chat"
  | "embedding"
  | "rerank"
  | "transcription"
  | "speech"
  | "ocr"
  | "parse";

const MAX_TEXT_LENGTH = 32_000;
const MAX_FILE_BYTES = 10 * 1024 * 1024;

const TOOLS: Array<{
  capability: string[];
  file: boolean;
  id: ToolId;
  label: string;
}> = [
  { id: "chat", label: "Chat to model", capability: ["chat"], file: false },
  {
    id: "embedding",
    label: "Embedding",
    capability: ["embedding"],
    file: false,
  },
  { id: "rerank", label: "Rerank", capability: ["rerank"], file: false },
  {
    id: "transcription",
    label: "Audio transcription",
    capability: ["speech2text", "asr"],
    file: true,
  },
  { id: "speech", label: "Audio speech", capability: ["tts"], file: false },
  { id: "ocr", label: "OCR", capability: ["ocr"], file: true },
  { id: "parse", label: "File parse", capability: ["doc_parse"], file: true },
];

function selector(model: PlatformModel): ModelSelector {
  return {
    instanceName: model.instanceName,
    modelId: model.id,
    modelName: model.name,
    providerName: model.providerName,
  };
}

function fileAllowed(tool: ToolId, file: File): boolean {
  if (tool === "transcription") return file.type.startsWith("audio/");
  return (
    file.type.startsWith("image/") ||
    file.type === "application/pdf" ||
    file.type === "text/plain" ||
    file.type === "application/octet-stream"
  );
}

async function fileToBase64(file: File, signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    const cleanup = () => signal.removeEventListener("abort", abort);
    const abort = () => {
      cleanup();
      reader.abort();
      reject(new DOMException("Aborted", "AbortError"));
    };
    if (signal.aborted) {
      abort();
      return;
    }
    signal.addEventListener("abort", abort, { once: true });
    reader.onerror = () => {
      cleanup();
      reject(reader.error ?? new Error("Dosya okunamadı."));
    };
    reader.onabort = () => {
      cleanup();
      reject(new DOMException("Aborted", "AbortError"));
    };
    reader.onload = () => {
      cleanup();
      const value = typeof reader.result === "string" ? reader.result : "";
      resolve(value.slice(value.indexOf(",") + 1));
    };
    reader.readAsDataURL(file);
  });
}

export function PlatformModelTools({ models }: { models: PlatformModel[] }) {
  const [toolId, setToolId] = useState<ToolId>("chat");
  const [modelId, setModelId] = useState("");
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File>();
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [audioUrl, setAudioUrl] = useState("");
  const audioUrlRef = useRef("");
  const abortRef = useRef<AbortController | undefined>(undefined);

  const tool = TOOLS.find((item) => item.id === toolId) ?? TOOLS[0];
  const compatibleModels = useMemo(
    () =>
      models.filter((model) =>
        tool.capability.some((capability) =>
          model.capabilities.includes(capability),
        ),
      ),
    [models, tool],
  );
  const selectedModel =
    compatibleModels.find((model) => model.id === modelId) ??
    compatibleModels[0];

  useEffect(() => {
    setModelId("");
    setFile(undefined);
    setInput("");
    setResult("");
    setError("");
  }, [toolId]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  const replaceAudioUrl = (blob?: Blob) => {
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    const next = blob ? URL.createObjectURL(blob) : "";
    audioUrlRef.current = next;
    setAudioUrl(next);
  };

  const run = async () => {
    if (!selectedModel || running) return;
    if (!tool.file && (!input.trim() || input.length > MAX_TEXT_LENGTH)) {
      setError(
        `Metin 1–${MAX_TEXT_LENGTH.toLocaleString()} karakter olmalıdır.`,
      );
      return;
    }
    if (tool.file && !file) {
      setError("Dosya seçin.");
      return;
    }
    if (file && (file.size > MAX_FILE_BYTES || !fileAllowed(tool.id, file))) {
      setError("Dosya türü desteklenmiyor veya 10 MB sınırını aşıyor.");
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError("");
    setResult("");
    replaceAudioUrl();
    try {
      const modelSelector = selector(selectedModel);
      if (tool.id === "chat") {
        const response = await chatToModel(
          modelSelector,
          [{ role: "user", content: input.trim() }],
          controller.signal,
        );
        setResult(
          [
            response.reasoning && `Reasoning:\n${response.reasoning}`,
            response.answer,
          ]
            .filter(Boolean)
            .join("\n\n"),
        );
      } else if (tool.id === "embedding") {
        const response = await createEmbeddings(
          modelSelector,
          input.split("\n").filter(Boolean),
          0,
          controller.signal,
        );
        // Deliberately expose dimensions and a tiny sample, never full vectors.
        setResult(
          response
            .map(
              (item) =>
                `#${item.index}: dimension=${item.vector.length}, tokens=${item.tokenCount}, sample=[${item.vector
                  .slice(0, 8)
                  .join(", ")}${item.vector.length > 8 ? ", …" : ""}]`,
            )
            .join("\n"),
        );
      } else if (tool.id === "rerank") {
        const [query, ...documents] = input.split("\n").filter(Boolean);
        if (!query || documents.length === 0) {
          setError("İlk satıra sorgu, sonraki satırlara dokümanları yazın.");
          return;
        }
        const response = await rerankDocuments(
          modelSelector,
          query,
          documents,
          documents.length,
          controller.signal,
        );
        setResult(
          response
            .map((item) => `document[${item.index}]: ${item.relevanceScore}`)
            .join("\n"),
        );
      } else if (tool.id === "speech") {
        replaceAudioUrl(
          await synthesizeSpeech(
            modelSelector,
            input.trim(),
            controller.signal,
          ),
        );
        setResult("Ses üretildi; oynatıcı geçici bir object URL kullanıyor.");
      } else {
        const encoded = await fileToBase64(file as File, controller.signal);
        if (tool.id === "transcription") {
          setResult(
            await transcribeAudio(
              modelSelector,
              encoded,
              [],
              controller.signal,
            ),
          );
        } else if (tool.id === "ocr") {
          setResult(await ocrFile(modelSelector, encoded, controller.signal));
        } else {
          setResult(
            `Task ID: ${await parseFile(modelSelector, encoded, controller.signal)}`,
          );
        }
      }
    } catch (runError) {
      if (isPlatformApiError(runError) && runError.isAbort) {
        setError("İstek iptal edildi.");
      } else if (
        runError instanceof DOMException &&
        runError.name === "AbortError"
      ) {
        setError("İstek iptal edildi.");
      } else {
        setError(
          runError instanceof Error
            ? runError.message
            : "Rag Platform aracı çalıştırılamadı.",
        );
      }
    } finally {
      abortRef.current = undefined;
      setRunning(false);
      setFile(undefined);
    }
  };

  return (
    <section className="space-y-4 overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12] p-4">
      <div className="flex min-w-0 flex-col gap-0.5">
        <h3 className="text-sm font-medium text-foreground">
          Yetkili model araçları
        </h3>
        <p className="text-xs leading-snug text-muted-foreground">
          İstekler oturum yetkisiyle çalışır; dosya ve sonuçlar kalıcı
          depolamaya yazılmaz.
        </p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="platform-model-tool">Tool</Label>
          <Select
            value={toolId}
            disabled={running}
            onValueChange={(value) => setToolId(value as ToolId)}
          >
            <SelectTrigger id="platform-model-tool" className="h-9 w-full">
              <SelectValue placeholder="Choose a tool" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {TOOLS.map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="platform-tool-model">Model</Label>
          <Select
            value={selectedModel?.id ?? ""}
            disabled={running || compatibleModels.length === 0}
            onValueChange={setModelId}
          >
            <SelectTrigger id="platform-tool-model" className="h-9 w-full">
              <SelectValue
                placeholder={
                  compatibleModels.length
                    ? "Choose a model"
                    : "No compatible model"
                }
              />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {compatibleModels.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name} — {model.providerName}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </div>

      {compatibleModels.length === 0 ? (
        <p role="status" className="text-sm text-amber-600">
          Bu araç {tool.capability.join(" / ")} capability’sine sahip etkin bir
          model gerektirir. Provider Models bölümünden model ekleyin.
        </p>
      ) : tool.file ? (
        <input
          aria-label="Araç dosyası"
          type="file"
          disabled={running}
          accept={
            tool.id === "transcription"
              ? "audio/*"
              : "image/*,application/pdf,text/plain"
          }
          onChange={(event) => setFile(event.target.files?.[0])}
        />
      ) : (
        <div className="space-y-1">
          <Textarea
            aria-label="Araç girdisi"
            rows={5}
            maxLength={MAX_TEXT_LENGTH}
            disabled={running}
            placeholder={
              tool.id === "rerank"
                ? "İlk satır: sorgu\nSonraki satırlar: dokümanlar"
                : tool.id === "embedding"
                  ? "Her satıra bir metin"
                  : "Metin girin"
            }
            value={input}
            onChange={(event) => setInput(event.target.value)}
          />
          <div className="text-right text-xs text-muted-foreground">
            {input.length}/{MAX_TEXT_LENGTH}
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <Button disabled={running || !selectedModel} onClick={() => void run()}>
          {running ? <Spinner /> : "Çalıştır"}
        </Button>
        {running && (
          <Button variant="outline" onClick={() => abortRef.current?.abort()}>
            İptal et
          </Button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {audioUrl && <audio controls={true} src={audioUrl} className="w-full" />}
      {result && (
        <pre className="max-h-72 overflow-auto rounded-lg bg-muted p-3 text-xs whitespace-pre-wrap break-words">
          {result}
        </pre>
      )}
    </section>
  );
}
