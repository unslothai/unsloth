import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  Binary,
  Braces,
  FileStack,
  Network,
  Tags,
  WandSparkles,
} from "lucide-react";
import { Suspense, lazy, useState } from "react";

const MetadataPanel = lazy(() => import("./advanced-dataset/metadata-panel"));
const TagsPanel = lazy(() => import("./advanced-dataset/tags-panel"));
const GraphPanel = lazy(() => import("./advanced-dataset/graph-panel"));
const ArtifactsPanel = lazy(() => import("./advanced-dataset/artifacts-panel"));
const IndexingPanel = lazy(() => import("./advanced-dataset/indexing-panel"));
const SkillsPanel = lazy(() => import("./advanced-dataset/skills-panel"));

type AdvancedTab =
  | "metadata"
  | "tags"
  | "graph"
  | "artifacts"
  | "indexing"
  | "skills";

const tabs: Array<{
  id: AdvancedTab;
  label: string;
  icon: typeof Braces;
  experimental?: boolean;
}> = [
  { id: "metadata", label: "Metadata", icon: Braces },
  { id: "tags", label: "Etiketler", icon: Tags },
  { id: "graph", label: "Grafik", icon: Network, experimental: true },
  { id: "artifacts", label: "Artifact", icon: FileStack, experimental: true },
  { id: "indexing", label: "İndeks & ingestion", icon: Binary },
  { id: "skills", label: "Beceriler", icon: WandSparkles, experimental: true },
];

export default function AdvancedDatasetWorkspace({
  datasetId,
  datasetName,
}: { datasetId: string; datasetName: string }) {
  const [tab, setTab] = useState<AdvancedTab>("metadata");
  const props = { datasetId, datasetName };
  return (
    <section
      data-testid="advanced-dataset-workspace"
      className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden px-3 pb-4 pt-3 sm:px-6 sm:pb-6 sm:pt-4 lg:px-8"
    >
      <div className="mx-auto flex min-h-0 min-w-0 w-full max-w-[var(--hub-measure)] flex-1 flex-col gap-3 sm:gap-4">
        <header className="flex min-w-0 shrink-0 flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Gelişmiş dataset yönetimi
            </p>
            <h2 className="mt-1 break-words text-xl font-semibold">
              {datasetName || "Dataset"}
            </h2>
          </div>
          <span className="rounded-full border px-2.5 py-1 text-[11px] text-muted-foreground">
            Aktif runtime sözleşmesi
          </span>
        </header>
        <div
          role="tablist"
          aria-label="Gelişmiş dataset alanları"
          className="grid shrink-0 grid-cols-2 gap-1 rounded-2xl border bg-muted/35 p-1 sm:grid-cols-3 xl:grid-cols-6"
        >
          {tabs.map((item) => {
            const Icon = item.icon;
            return (
              <button
                type="button"
                key={item.id}
                role="tab"
                aria-selected={tab === item.id}
                onClick={() => setTab(item.id)}
                className={cn(
                  "flex min-w-0 items-center justify-center gap-1.5 rounded-xl px-2 py-2 text-xs font-medium leading-tight transition-colors sm:gap-2 sm:px-3",
                  tab === item.id
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="size-3.5" />
                <span className="min-w-0 truncate">{item.label}</span>
                {item.experimental ? (
                  <span className="shrink-0 rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[9px] uppercase text-amber-700 dark:text-amber-300">
                    deneysel
                  </span>
                ) : null}
              </button>
            );
          })}
        </div>
        <div
          role="tabpanel"
          className="min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto pr-1 [scrollbar-gutter:stable]"
        >
          <Suspense
            fallback={
              <div
                role="status"
                className="flex min-h-56 items-center justify-center gap-2 text-sm text-muted-foreground"
              >
                <Spinner /> Bölüm yükleniyor…
              </div>
            }
          >
            {tab === "metadata" ? <MetadataPanel {...props} /> : null}
            {tab === "tags" ? <TagsPanel {...props} /> : null}
            {tab === "graph" ? <GraphPanel {...props} /> : null}
            {tab === "artifacts" ? <ArtifactsPanel {...props} /> : null}
            {tab === "indexing" ? <IndexingPanel {...props} /> : null}
            {tab === "skills" ? <SkillsPanel {...props} /> : null}
          </Suspense>
        </div>
      </div>
    </section>
  );
}
