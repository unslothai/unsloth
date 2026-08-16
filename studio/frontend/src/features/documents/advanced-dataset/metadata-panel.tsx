import { Button } from "@/components/ui/button";
import { PlatformPipelineSelect } from "@/features/rag/components/platform-pipeline-select";
import {
  type PlatformDatasetChunkMethod,
  type PlatformDatasetDto,
  type PlatformMetadataConfig,
  batchUpdateDatasetDocumentStatus,
  batchUpdateDatasetMetadata,
  getDatasetMetadataConfig,
  getDatasetMetadataSummary,
  getFlattenedDatasetMetadata,
  getPlatformDataset,
  mapPipelineToDatasetFields,
  updateDatasetMetadataConfig,
  updateDocumentMetadataConfig,
  updatePlatformDataset,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useMemo, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  textareaClass,
  useAbortableLoad,
} from "./shared";

interface MetadataBundle {
  config: PlatformMetadataConfig;
  dataset: PlatformDatasetDto;
  flattened: Record<string, unknown>;
  summary: Record<string, unknown>;
}

function parseIds(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}
const CHUNK_METHODS = new Set<PlatformDatasetChunkMethod>([
  "naive",
  "book",
  "email",
  "laws",
  "manual",
  "one",
  "paper",
  "picture",
  "presentation",
  "qa",
  "table",
  "tag",
  "resume",
]);
function parseJson(value: string, label: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    throw new Error(`${label} geçerli JSON olmalı.`);
  }
}

export default function MetadataPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const loader = useCallback(
    async (signal: AbortSignal): Promise<MetadataBundle> => {
      const [config, dataset, flattened, summary] = await Promise.all([
        getDatasetMetadataConfig(datasetId, signal),
        getPlatformDataset(datasetId, signal),
        getFlattenedDatasetMetadata([datasetId], signal),
        getDatasetMetadataSummary(datasetId, [], signal),
      ]);
      return { config, dataset, flattened, summary: summary.summary };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(loader);
  const [schemaDraft, setSchemaDraft] = useState("");
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [documentIds, setDocumentIds] = useState("");
  const [metadataKey, setMetadataKey] = useState("");
  const [metadataValue, setMetadataValue] = useState('"değer"');
  const [documentConfig, setDocumentConfig] = useState(
    '{\n  "metadata": {}\n}',
  );
  const [busy, setBusy] = useState<string | null>(null);
  const schema =
    schemaDraft ||
    (loaded.data ? JSON.stringify(loaded.data.config, null, 2) : "");
  const selectedPipeline =
    pipelineId ??
    (typeof loaded.data?.dataset.pipeline_id === "string"
      ? loaded.data.dataset.pipeline_id
      : "");
  const ids = useMemo(() => parseIds(documentIds), [documentIds]);

  const run = async (
    name: string,
    action: () => Promise<unknown>,
    success: string,
  ) => {
    setBusy(name);
    try {
      await action();
      toast.success(success);
      loaded.load();
    } catch (error) {
      toast.error("İşlem tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(null);
    }
  };

  if (!datasetId)
    return <PanelState state="empty" empty="Önce bir dataset seçin." />;
  if (loaded.state !== "ready" || !loaded.data)
    return (
      <PanelState
        state={loaded.state}
        error={loaded.error}
        onRetry={loaded.load}
      />
    );
  const data = loaded.data;
  return (
    <div className="grid min-w-0 gap-4 pb-5">
      <div className="grid min-w-0 gap-4 xl:grid-cols-2">
        <SectionCard
          title="Metadata şeması"
          description="Dataset alanlarını backend metadata/config sözleşmesine göre yönetir."
          actions={
            <Button
              size="sm"
              disabled={busy !== null}
              onClick={() =>
                void run(
                  "schema",
                  async () => {
                    const parsed = parseJson(
                      schema,
                      "Metadata şeması",
                    ) as PlatformMetadataConfig;
                    if (
                      !Array.isArray(parsed.metadata) ||
                      !Array.isArray(parsed.built_in_metadata)
                    )
                      throw new Error(
                        "metadata ve built_in_metadata listeleri zorunludur.",
                      );
                    await updateDatasetMetadataConfig(datasetId, parsed);
                  },
                  "Metadata şeması kaydedildi.",
                )
              }
            >
              Kaydet
            </Button>
          }
        >
          <textarea
            aria-label="Metadata şeması JSON"
            className={textareaClass}
            value={schema}
            onChange={(event) => setSchemaDraft(event.target.value)}
          />
        </SectionCard>
        <SectionCard
          title="Metadata görünümü"
          description="Flattened alanlar ve belge metadata özeti salt okunur gösterilir."
        >
          <div className="grid min-w-0 gap-3 md:grid-cols-2">
            <pre className="max-h-72 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
              {JSON.stringify(loaded.data.flattened, null, 2)}
            </pre>
            <pre className="max-h-72 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
              {JSON.stringify(loaded.data.summary, null, 2)}
            </pre>
          </div>
        </SectionCard>
      </div>
      <SectionCard
        title="Pipeline / parser eşlemesi"
        description="Faz 3 pipeline sözleşmesini dataset yapılandırmasına bağlar."
        actions={
          <Button
            size="sm"
            disabled={busy !== null}
            onClick={() =>
              void run(
                "pipeline",
                async () => {
                  const mapped = mapPipelineToDatasetFields(selectedPipeline);
                  const rawParserId = data.dataset.parser_id;
                  const parserId =
                    typeof rawParserId === "string" &&
                    CHUNK_METHODS.has(rawParserId as PlatformDatasetChunkMethod)
                      ? (rawParserId as PlatformDatasetChunkMethod)
                      : "naive";
                  await updatePlatformDataset(
                    datasetId,
                    mapped ?? {
                      pipeline_id: "",
                      parse_type: 1,
                      parser_id: parserId,
                    },
                  );
                },
                "Pipeline seçimi güncellendi.",
              )
            }
          >
            Uygula
          </Button>
        }
      >
        <div className="min-w-0 max-w-xl">
          <PlatformPipelineSelect
            value={selectedPipeline}
            onChange={setPipelineId}
            disabled={busy !== null}
          />
        </div>
      </SectionCard>
      <SectionCard
        title="Toplu belge metadata işlemleri"
        description="Belge kimlikleri açıkça seçilir; toplu değişiklik ve durum güncellemesi onaylı aksiyonlardır."
      >
        <div className="grid min-w-0 gap-3 lg:grid-cols-3">
          <Field label="Belge kimlikleri" hint="Virgülle ayırın.">
            <input
              className={inputClass}
              value={documentIds}
              onChange={(event) => setDocumentIds(event.target.value)}
              placeholder="doc-1, doc-2"
            />
          </Field>
          <Field label="Metadata anahtarı">
            <input
              className={inputClass}
              value={metadataKey}
              onChange={(event) => setMetadataKey(event.target.value)}
              placeholder="department"
            />
          </Field>
          <Field label="Metadata değeri (JSON)">
            <input
              className={inputClass}
              value={metadataValue}
              onChange={(event) => setMetadataValue(event.target.value)}
            />
          </Field>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            size="sm"
            disabled={busy !== null || ids.length === 0 || !metadataKey.trim()}
            onClick={() =>
              void run(
                "batch",
                () =>
                  batchUpdateDatasetMetadata(datasetId, {
                    selector: { document_ids: ids, metadata_condition: {} },
                    updates: [
                      {
                        key: metadataKey.trim(),
                        value: parseJson(metadataValue, "Metadata değeri"),
                      },
                    ],
                    deletes: [],
                  }),
                "Belge metadata alanları güncellendi.",
              )
            }
          >
            Metadata uygula
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null || ids.length === 0}
            onClick={() =>
              void run(
                "enable",
                () => batchUpdateDatasetDocumentStatus(datasetId, ids, 1),
                "Belgeler etkinleştirildi.",
              )
            }
          >
            Belgeleri etkinleştir
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null || ids.length === 0}
            onClick={() =>
              void run(
                "disable",
                () => batchUpdateDatasetDocumentStatus(datasetId, ids, 0),
                "Belgeler devre dışı bırakıldı.",
              )
            }
          >
            Belgeleri devre dışı bırak
          </Button>
        </div>
      </SectionCard>
      <SectionCard
        title="Tek belge metadata yapılandırması"
        description="Belgeye ait serbest biçimli metadata config sözleşmesini doğrulanmış JSON olarak gönderir."
        actions={
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null || ids.length !== 1}
            onClick={() =>
              void run(
                "doc-config",
                () =>
                  updateDocumentMetadataConfig(
                    datasetId,
                    ids[0],
                    parseJson(
                      documentConfig,
                      "Belge metadata config",
                    ) as Record<string, unknown>,
                  ),
                "Belge metadata yapılandırması kaydedildi.",
              )
            }
          >
            Seçili belgeye kaydet
          </Button>
        }
      >
        <textarea
          aria-label="Belge metadata config JSON"
          className={textareaClass}
          value={documentConfig}
          onChange={(event) => setDocumentConfig(event.target.value)}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          Bu aksiyon için tam olarak bir belge kimliği girilmelidir.
        </p>
      </SectionCard>
    </div>
  );
}
