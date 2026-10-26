// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useHfTokenStore } from "@/features/hub";
import {
  type PreviewToken,
  type TokenPreviewResponse,
  type TokenPreviewSample,
  type TrainingConfigState,
  fetchTokenPreview,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { ShuffleIcon, TextSelectIcon } from "lucide-react";
import { type ReactNode, memo, useEffect, useMemo, useState } from "react";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";

// Neighbouring tokens alternate shades so their boundaries stay visible.
const TRAINED_SHADES = [
  "bg-emerald-500/20 text-foreground",
  "bg-sky-500/20 text-foreground",
  "bg-amber-500/25 text-foreground",
  "bg-violet-500/20 text-foreground",
];
const MASKED_SHADES = [
  "bg-muted text-muted-foreground/80",
  "bg-muted/50 text-muted-foreground/80",
];
const TEXT_DEBOUNCE_MS = 300;

type PreviewState = {
  data: TokenPreviewResponse | null;
  error: string | null;
  loading: boolean;
};

const IDLE: PreviewState = { data: null, error: null, loading: false };

function visibleText(text: string): string {
  if (text === "") {
    return "∅";
  }
  return text.replace(/\t/g, "→").replace(/\n/g, "↵\n");
}

const TokenStream = memo(function TokenStream({
  tokens,
  showIds,
}: {
  tokens: PreviewToken[];
  showIds: boolean;
}) {
  return (
    <div className="font-mono text-ui-12p5 leading-6 whitespace-pre-wrap break-words">
      {tokens.map(([id, text, trained], index) => (
        <span
          // biome-ignore lint/suspicious/noArrayIndexKey: tokens are positional
          key={index}
          title={`#${index}  id ${id}`}
          className={cn(
            "rounded-[3px] px-px",
            trained
              ? TRAINED_SHADES[index % TRAINED_SHADES.length]
              : MASKED_SHADES[index % MASKED_SHADES.length],
          )}
        >
          {visibleText(text)}
          {showIds ? (
            <sub className="ml-0.5 text-ui-9 opacity-60">{id}</sub>
          ) : null}
        </span>
      ))}
    </div>
  );
});

function Legend() {
  const t = useT();
  return (
    <div className="flex items-center gap-3 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <span className="size-3 rounded-[3px] bg-emerald-500/30" />
        {t("studio.dataset.tokenPreview.trained")}
      </span>
      <span className="flex items-center gap-1.5">
        <span className="size-3 rounded-[3px] bg-muted ring-1 ring-border" />
        {t("studio.dataset.tokenPreview.masked")}
      </span>
    </div>
  );
}

function SampleCard({
  sample,
  label,
  showIds,
  maskingApplied,
}: {
  sample: TokenPreviewSample;
  label?: string;
  showIds: boolean;
  maskingApplied: boolean;
}) {
  const t = useT();
  const percent =
    sample.num_tokens > 0
      ? Math.round((100 * sample.num_trained) / sample.num_tokens)
      : 0;
  const copyIds = async () => {
    if (await copyToClipboard(JSON.stringify(sample.input_ids))) {
      toast.success(t("studio.dataset.tokenPreview.copied"));
    }
  };
  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-card p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {label ? <span className="font-medium">{label}</span> : null}
        <span className="text-muted-foreground">
          {t("studio.dataset.tokenPreview.stats", {
            tokens: sample.num_tokens.toLocaleString(),
            trained: sample.num_trained.toLocaleString(),
            percent,
          })}
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto h-6 px-2 text-xs"
          onClick={copyIds}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3" />
          {t("studio.dataset.tokenPreview.copyIds")}
        </Button>
      </div>
      {maskingApplied && sample.num_trained === 0 ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("studio.dataset.tokenPreview.noTrained")}
        </p>
      ) : null}
      {sample.truncated_from ? (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          {t("studio.dataset.tokenPreview.truncated", {
            max: sample.num_tokens.toLocaleString(),
            full: sample.truncated_from.toLocaleString(),
          })}
        </p>
      ) : null}
      <TokenStream tokens={sample.tokens} showIds={showIds} />
      {sample.num_hidden > 0 ? (
        <p className="text-xs text-muted-foreground">
          {t("studio.dataset.tokenPreview.hiddenTokens", {
            count: sample.num_hidden.toLocaleString(),
          })}
        </p>
      ) : null}
    </div>
  );
}

function MaskingSummary({
  data,
  requested,
}: {
  data: TokenPreviewResponse;
  requested: boolean;
}) {
  const t = useT();
  if (data.masking.applied) {
    return (
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="secondary">
          {t("studio.dataset.tokenPreview.completionsOnly")}
        </Badge>
        {data.masking.response_part ? (
          <span className="text-muted-foreground">
            {t("studio.dataset.tokenPreview.responseMarker")}{" "}
            <code className="rounded-sm bg-muted px-1 py-0.5 font-mono">
              {JSON.stringify(data.masking.response_part)}
            </code>
          </span>
        ) : null}
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge variant="outline">
        {t("studio.dataset.tokenPreview.allTokens")}
      </Badge>
      {requested && data.final_format !== "raw_text" ? (
        <span className="text-amber-600 dark:text-amber-400">
          {t("studio.dataset.tokenPreview.maskingNotApplied")}
        </span>
      ) : null}
    </div>
  );
}

export function TokenPreviewDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {open ? <TokenPreviewContent /> : null}
    </Dialog>
  );
}

type PreviewQuery = {
  config: TrainingConfigState;
  hfToken: string | null;
  text?: string;
  finalFormat?: string | null;
  seed?: number;
};

/** The latest preview for `query`; a newer query aborts the request before it. */
function useTokenPreview(query: PreviewQuery | null, debounceMs = 0) {
  const [state, setState] = useState<PreviewState>(IDLE);
  useEffect(() => {
    if (!query) {
      return;
    }
    const controller = new AbortController();
    setState((prev) => ({ ...prev, error: null, loading: true }));
    const timer = window.setTimeout(() => {
      fetchTokenPreview({ ...query, signal: controller.signal })
        .then((data) => setState({ data, error: null, loading: false }))
        .catch((error: unknown) => {
          if (controller.signal.aborted) {
            return;
          }
          setState({
            data: null,
            error: error instanceof Error ? error.message : String(error),
            loading: false,
          });
        });
    }, debounceMs);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query, debounceMs]);
  return state;
}

function LoadingRow() {
  const t = useT();
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
      <Spinner className="size-4" />
      {t("studio.dataset.tokenPreview.loading")}
    </div>
  );
}

function DatasetSamples({
  state,
  showIds,
  onShuffle,
}: {
  state: PreviewState;
  showIds: boolean;
  onShuffle: () => void;
}) {
  const t = useT();
  if (state.error) {
    return (
      <div className="flex flex-col items-start gap-3 py-6">
        <p className="text-sm text-destructive">{state.error}</p>
        <Button variant="outline" size="sm" onClick={onShuffle}>
          {t("studio.dataset.tokenPreview.retry")}
        </Button>
      </div>
    );
  }
  if (!state.data || state.loading) {
    return <LoadingRow />;
  }
  const maskingApplied = state.data.masking.applied;
  return (
    <div className="flex flex-col gap-3">
      {state.data.samples.map((sample, index) => (
        <SampleCard
          // biome-ignore lint/suspicious/noArrayIndexKey: samples are positional
          key={index}
          sample={sample}
          label={t("studio.dataset.tokenPreview.sample", { index: index + 1 })}
          showIds={showIds}
          maskingApplied={maskingApplied}
        />
      ))}
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={onShuffle}>
          <ShuffleIcon className="size-3.5" />
          {t("studio.dataset.tokenPreview.shuffle")}
        </Button>
      </div>
    </div>
  );
}

function TextPlayground({
  text,
  onTextChange,
  state,
  exampleText,
  showIds,
}: {
  text: string;
  onTextChange: (text: string) => void;
  state: PreviewState;
  exampleText: string | null;
  showIds: boolean;
}) {
  const t = useT();
  const sample = text ? state.data?.samples[0] : undefined;
  let result: ReactNode = null;
  if (state.error && text) {
    result = <p className="text-sm text-destructive">{state.error}</p>;
  } else if (sample) {
    result = (
      <SampleCard
        sample={sample}
        showIds={showIds}
        maskingApplied={state.data?.masking.applied ?? false}
      />
    );
  } else if (state.loading && !state.data) {
    result = <LoadingRow />;
  }
  return (
    <>
      <Textarea
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder={t("studio.dataset.tokenPreview.textPlaceholder")}
        className="min-h-28 font-mono text-ui-12p5"
      />
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-muted-foreground">
          {t("studio.dataset.tokenPreview.textHint")}
        </p>
        {exampleText ? (
          <Button
            variant="outline"
            size="sm"
            className="ml-auto"
            onClick={() => onTextChange(exampleText)}
          >
            {t("studio.dataset.tokenPreview.insertExample")}
          </Button>
        ) : null}
      </div>
      {result}
    </>
  );
}

function TokenPreviewContent() {
  const t = useT();
  const hfToken = useHfTokenStore((state) => state.token.trim() || null);
  // A snapshot: the dialog is modal, so the configuration cannot change underneath it.
  const [config] = useState(() => useTrainingConfigStore.getState());
  const multimodal = config.isDatasetImage === true || config.isDatasetAudio;
  const [tab, setTab] = useState(multimodal ? "text" : "dataset");
  const [showIds, setShowIds] = useState(false);
  const [seed, setSeed] = useState(0);
  const [text, setText] = useState("");
  const datasetQuery = useMemo(
    () => (multimodal ? null : { config, hfToken, seed }),
    [config, hfToken, seed, multimodal],
  );
  const datasetState = useTokenPreview(datasetQuery);
  const finalFormat = datasetState.data?.final_format ?? null;
  const textQuery = useMemo(
    () => (tab === "text" ? { config, hfToken, text, finalFormat } : null),
    [config, hfToken, text, tab, finalFormat],
  );
  const textState = useTokenPreview(textQuery, TEXT_DEBOUNCE_MS);
  const summary = datasetState.data ?? textState.data;

  return (
    <DialogContent className="sm:max-w-4xl w-[90vw] max-sm:w-[calc(100vw-1rem)] max-h-[88dvh] flex flex-col gap-0 p-0 overflow-hidden rounded-3xl corner-squircle">
      <DialogHeader className="px-6 max-sm:px-4 pt-5 pb-3 shrink-0">
        <div className="flex items-center gap-3 pr-10">
          <div className="rounded-md corner-squircle p-2 ring-1 ring-emerald-200 bg-emerald-50 text-emerald-600 dark:ring-emerald-800 dark:bg-emerald-950 dark:text-emerald-400 shrink-0">
            <TextSelectIcon className="size-4" />
          </div>
          <div className="min-w-0">
            <DialogTitle className="font-heading text-lg font-semibold tracking-tight">
              {t("studio.dataset.tokenPreview.title")}
            </DialogTitle>
            <DialogDescription className="truncate text-xs">
              {t("studio.dataset.tokenPreview.description", {
                model: config.selectedModel ?? "",
              })}
            </DialogDescription>
          </div>
        </div>
      </DialogHeader>

      <Tabs
        value={tab}
        onValueChange={setTab}
        className="flex min-h-0 flex-1 flex-col px-6 max-sm:px-4 pb-5"
      >
        <div className="flex flex-wrap items-center gap-3 pb-2">
          <TabsList>
            <TabsTrigger value="dataset">
              {t("studio.dataset.tokenPreview.datasetTab")}
            </TabsTrigger>
            <TabsTrigger value="text">
              {t("studio.dataset.tokenPreview.textTab")}
            </TabsTrigger>
          </TabsList>
          <Legend />
          <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            <Switch
              id="token-preview-show-ids"
              size="sm"
              checked={showIds}
              onCheckedChange={setShowIds}
            />
            <label htmlFor="token-preview-show-ids">
              {t("studio.dataset.tokenPreview.showIds")}
            </label>
          </div>
        </div>
        {summary ? (
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3">
            <MaskingSummary
              data={summary}
              requested={config.trainOnCompletions}
            />
            <span className="text-xs text-muted-foreground">
              {t("studio.dataset.tokenPreview.vocab", {
                count: summary.vocab_size.toLocaleString(),
              })}
            </span>
          </div>
        ) : null}

        {config.packing ? (
          <p className="pb-3 text-xs text-muted-foreground">
            {t("studio.dataset.tokenPreview.packing")}
          </p>
        ) : null}

        <TabsContent value="dataset" className="min-h-0 flex-1 overflow-auto">
          {multimodal ? (
            <p className="py-6 text-sm text-muted-foreground">
              {t("studio.dataset.tokenPreview.multimodal")}
            </p>
          ) : (
            <DatasetSamples
              state={datasetState}
              showIds={showIds}
              onShuffle={() => setSeed((value) => value + 1)}
            />
          )}
        </TabsContent>

        <TabsContent
          value="text"
          className="flex min-h-0 flex-1 flex-col gap-3 overflow-auto"
        >
          <TextPlayground
            text={text}
            onTextChange={setText}
            state={textState}
            exampleText={summary?.example_text ?? null}
            showIds={showIds}
          />
        </TabsContent>
      </Tabs>
    </DialogContent>
  );
}
