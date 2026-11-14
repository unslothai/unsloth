// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type NpuCatalog,
  type NpuModel,
  npuCapabilities,
  npuContextLabel,
  npuReadmeNote,
  npuResumeLabel,
  npuSizeLabel,
} from "@/features/npu";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { formatCompact } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  ChipIcon,
  Download01Icon,
  FavouriteIcon,
  PackageIcon,
} from "@hugeicons/core-free-icons";
import { useSelectedModelMetadata } from "../hooks/use-selected-model-metadata";
import {
  BaseModelSearchChip,
  CopyRepoButton,
  InspectorDownloadSlot,
  StatGrid,
  StatRow,
  ViewRepositoryButton,
} from "./model-inspector";
import { ModelReadme } from "./model-readme";
import { NpuModelActions } from "./npu-catalog-list";
import { OwnerAvatar } from "./owner-avatar";
import { CapabilityPill } from "./shared";

export function NpuModelDetail({
  model,
  catalog,
  onRun,
  onSearchHub,
}: {
  model: NpuModel;
  catalog: NpuCatalog;
  onRun: () => void;
  onSearchHub?: (query: string) => void;
}) {
  const repo = model.hf_repo ?? null;
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const { result: metadata } = useSelectedModelMetadata(repo, {
    accessToken: hfToken || undefined,
    enabled: true,
    online,
  });
  const [owner, repoName] = repo ? repo.split("/") : ["FastFlowLM", model.id];
  const sizeLabel = npuSizeLabel(model.size_gb);
  const contextLabel = npuContextLabel(model.max_context_length);
  const readmeNote = npuReadmeNote(model);
  const resumeLabel =
    model.id in catalog.downloads ? null : npuResumeLabel(model);

  return (
    <div data-testid="hub-npu-detail" className="flex flex-col">
      <div className="pb-2 pt-1">
        <div className="flex items-center gap-4">
          <OwnerAvatar
            owner={owner}
            repoName={repoName}
            className="size-[calc(60px*var(--ui-space-scale,1))] rounded-lg text-ui-19"
          />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-1.5">
              <h2 className="truncate text-ui-25 font-semibold leading-ui-31 tracking-normal text-foreground">
                {model.id}
              </h2>
              {repo && (
                <div className="flex shrink-0 items-center gap-0.5">
                  <CopyRepoButton repoId={repo} />
                  <ViewRepositoryButton repoId={repo} isDataset={false} />
                </div>
              )}
            </div>
            <div className="mt-0.5 truncate text-ui-15 leading-ui-24 text-muted-foreground">
              {repo ?? "FastFlowLM"}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-1.5">
          <span className="inline-flex h-6 items-center gap-1.5 rounded-full bg-muted px-2.5 text-ui-11p5 font-medium text-foreground dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]">
            <span
              aria-hidden={true}
              className="size-1.5 rounded-full bg-format-npu"
            />
            NPU
          </span>
          {npuCapabilities(model).map((capability) => (
            <CapabilityPill key={capability.key} capability={capability} />
          ))}
          {metadata?.baseModel && (
            <BaseModelSearchChip
              baseModel={metadata.baseModel}
              searchTerm={metadata.baseModel}
              onSearchHub={onSearchHub}
            />
          )}
        </div>
      </div>

      <InspectorDownloadSlot>
        <div className="flex flex-wrap items-center gap-3">
          <NpuModelActions model={model} catalog={catalog} onRun={onRun} />
          {resumeLabel && (
            <span className="text-ui-12 text-muted-foreground">
              {resumeLabel}
            </span>
          )}
        </div>
      </InspectorDownloadSlot>

      <div className="pb-5 pt-5">
        <StatGrid>
          <StatRow label="Checkpoint" value={model.checkpoint} icon={ChipIcon} />
          {sizeLabel && (
            <StatRow label="Size" value={sizeLabel} icon={PackageIcon} />
          )}
          {contextLabel && (
            <StatRow
              label="Max context"
              value={contextLabel}
              icon={ArrowExpand01Icon}
            />
          )}
          {metadata && (
            <>
              <StatRow
                label="Downloads"
                value={formatCompact(metadata.downloads)}
                icon={Download01Icon}
              />
              <StatRow
                label="Likes"
                value={formatCompact(metadata.likes)}
                icon={FavouriteIcon}
              />
            </>
          )}
        </StatGrid>
      </div>

      <div className="max-w-[var(--hub-readme-measure)] space-y-4 pt-4">
        {repo ? (
          <>
            {readmeNote && (
              <p
                data-testid="hub-npu-readme-note"
                className="text-ui-12 text-muted-foreground"
              >
                {readmeNote}
              </p>
            )}
            <ModelReadme repoId={repo} revision={model.hf_revision ?? null} />
          </>
        ) : (
          <p className="text-ui-13 text-muted-foreground">
            No model card for this model.
          </p>
        )}
      </div>
    </div>
  );
}
