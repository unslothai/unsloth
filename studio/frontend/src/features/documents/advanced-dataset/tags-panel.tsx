import { Button } from "@/components/ui/button";
import {
  type PlatformTagCount,
  aggregateDatasetTags,
  listDatasetTags,
  removeDatasetTags,
  renameDatasetTag,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  useAbortableLoad,
} from "./shared";

export default function TagsPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const loader = useCallback(
    async (signal: AbortSignal) => {
      const [tags, aggregate] = await Promise.all([
        listDatasetTags(datasetId, signal),
        aggregateDatasetTags([datasetId], signal),
      ]);
      return { tags, aggregate };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(
    loader,
    (value) => value.tags.length === 0 && value.aggregate.length === 0,
  );
  const [fromTag, setFromTag] = useState("");
  const [toTag, setToTag] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const run = async (action: () => Promise<unknown>, success: string) => {
    setBusy(true);
    try {
      await action();
      toast.success(success);
      setSelected([]);
      loaded.load();
    } catch (error) {
      toast.error("Etiket işlemi tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };
  if (loaded.state !== "ready" && loaded.state !== "empty")
    return (
      <PanelState
        state={loaded.state}
        error={loaded.error}
        onRetry={loaded.load}
      />
    );
  const tags: PlatformTagCount[] = loaded.data?.tags ?? [];
  return (
    <div className="grid min-w-0 gap-4 pb-5">
      <SectionCard
        title="Dataset etiketleri"
        description="Etiket sayıları hem dataset listesi hem aggregation sözleşmesinden doğrulanır."
        actions={
          <Button size="sm" variant="outline" onClick={loaded.load}>
            Yenile
          </Button>
        }
      >
        {tags.length === 0 ? (
          <PanelState state="empty" empty="Bu dataset için etiket yok." />
        ) : (
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <label
                key={tag.key}
                className="flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-xs"
              >
                <input
                  type="checkbox"
                  checked={selected.includes(tag.key)}
                  onChange={(event) =>
                    setSelected((current) =>
                      event.target.checked
                        ? [...current, tag.key]
                        : current.filter((item) => item !== tag.key),
                    )
                  }
                />
                <span>{tag.key}</span>
                <span className="text-muted-foreground">{tag.count}</span>
              </label>
            ))}
          </div>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="destructive"
            disabled={busy || selected.length === 0}
            onClick={() =>
              void run(
                () => removeDatasetTags(datasetId, selected),
                `${selected.length} etiket kaldırıldı.`,
              )
            }
          >
            Seçilenleri kaldır
          </Button>
        </div>
      </SectionCard>
      <SectionCard
        title="Etiketi yeniden adlandır"
        description="Tüm dataset belgelerinde kaynak etiketi hedef etiketle değiştirir."
      >
        <div className="grid min-w-0 gap-3 sm:grid-cols-2">
          <Field label="Mevcut etiket">
            <input
              className={inputClass}
              value={fromTag}
              onChange={(event) => setFromTag(event.target.value)}
            />
          </Field>
          <Field label="Yeni etiket">
            <input
              className={inputClass}
              value={toTag}
              onChange={(event) => setToTag(event.target.value)}
            />
          </Field>
        </div>
        <Button
          className="mt-3"
          size="sm"
          disabled={busy || !fromTag.trim() || !toTag.trim()}
          onClick={() =>
            void run(
              () => renameDatasetTag(datasetId, fromTag.trim(), toTag.trim()),
              "Etiket yeniden adlandırıldı.",
            )
          }
        >
          Yeniden adlandır
        </Button>
      </SectionCard>
      <SectionCard title="Aggregation sonucu">
        <pre className="max-h-72 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
          {JSON.stringify(loaded.data?.aggregate ?? [], null, 2)}
        </pre>
      </SectionCard>
    </div>
  );
}
