import { Button } from "@/components/ui/button";
import {
  type PlatformArtifactGraph,
  type PlatformDatasetSearchResponse,
  type PlatformGraphData,
  getDatasetArtifactGraph,
  getDatasetGraph,
  searchDatasets,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useMemo, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  useAbortableLoad,
} from "./shared";

function summarize(value: unknown, limit: number): unknown {
  if (Array.isArray(value))
    return value.slice(0, limit).map((item) => summarize(item, limit));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .slice(0, limit)
        .map(([key, child]) => [key, summarize(child, limit)]),
    );
  return value;
}

export default function GraphPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const [detail, setDetail] = useState(50);
  const [node, setNode] = useState("");
  const [question, setQuestion] = useState("");
  const [searchResult, setSearchResult] =
    useState<PlatformDatasetSearchResponse | null>(null);
  const loader = useCallback(
    async (
      signal: AbortSignal,
    ): Promise<{
      dataset: PlatformGraphData;
      artifacts: PlatformArtifactGraph;
    }> => {
      const [dataset, artifacts] = await Promise.all([
        getDatasetGraph(datasetId, signal),
        getDatasetArtifactGraph(datasetId, undefined, signal),
      ]);
      return { dataset, artifacts };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(loader);
  const visibleArtifacts = useMemo(
    () =>
      loaded.data
        ? {
            entities: loaded.data.artifacts.entities.slice(0, detail),
            relations: loaded.data.artifacts.relations.slice(0, detail),
          }
        : null,
    [detail, loaded.data],
  );
  const expand = async () => {
    if (!node.trim()) return;
    loaded.setState("loading");
    try {
      const artifacts = await getDatasetArtifactGraph(datasetId, node.trim());
      loaded.setData((current) =>
        current ? { ...current, artifacts } : { dataset: {}, artifacts },
      );
      loaded.setState("ready");
    } catch {
      loaded.setState("error");
    }
  };
  if (loaded.state !== "ready" || !loaded.data)
    return (
      <PanelState
        state={loaded.state}
        error={loaded.error}
        onRetry={loaded.load}
      />
    );
  return (
    <div className="grid min-w-0 gap-4 pb-5">
      <SectionCard
        title="Dataset araması"
        description="Aktif çoklu-dataset arama sözleşmesini mevcut dataset kapsamıyla çalıştırır."
      >
        <div className="flex min-w-0 max-w-2xl flex-col items-stretch gap-2 sm:flex-row sm:items-end">
          <Field label="Soru">
            <input
              className={inputClass}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Bu datasette ne var?"
            />
          </Field>
          <Button
            className="sm:shrink-0"
            size="sm"
            disabled={!question.trim()}
            onClick={() =>
              void searchDatasets([datasetId], question.trim())
                .then(setSearchResult)
                .catch((error: unknown) =>
                  toast.error("Dataset araması tamamlanamadı", {
                    description:
                      error instanceof Error ? error.message : String(error),
                  }),
                )
            }
          >
            Ara
          </Button>
        </div>
        {searchResult ? (
          <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
            {JSON.stringify(searchResult, null, 2)}
          </pre>
        ) : null}
      </SectionCard>
      <SectionCard
        title="Dataset knowledge graph"
        description="Büyük graph yanıtları detay seviyesine göre sınırlandırılır; ham vektör alanları frontend'e taşınmaz."
        actions={
          <select
            aria-label="Grafik detay seviyesi"
            className={inputClass}
            value={detail}
            onChange={(event) => setDetail(Number(event.target.value))}
          >
            <option value={20}>20 öğe</option>
            <option value={50}>50 öğe</option>
            <option value={100}>100 öğe</option>
          </select>
        }
      >
        <pre className="max-h-[32rem] overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
          {JSON.stringify(summarize(loaded.data.dataset, detail), null, 2)}
        </pre>
      </SectionCard>
      <SectionCard
        title="Artifact graph"
        description={`${loaded.data.artifacts.entities.length} entity, ${loaded.data.artifacts.relations.length} ilişki. Bir node girilirse backend incremental alt grafik döndürür.`}
      >
        <div className="mb-3 flex min-w-0 max-w-xl flex-col items-stretch gap-2 sm:flex-row sm:items-end">
          <Field label="Merkez node">
            <input
              className={inputClass}
              value={node}
              onChange={(event) => setNode(event.target.value)}
              placeholder="entity slug"
            />
          </Field>
          <Button
            className="sm:shrink-0"
            size="sm"
            onClick={() => void expand()}
            disabled={!node.trim()}
          >
            Alt grafiği getir
          </Button>
        </div>
        <pre className="max-h-[32rem] overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
          {JSON.stringify(visibleArtifacts, null, 2)}
        </pre>
      </SectionCard>
    </div>
  );
}
