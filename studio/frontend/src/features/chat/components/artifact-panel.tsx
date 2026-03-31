// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { downloadTextFile } from "@/lib/download";
import {
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  XIcon,
} from "lucide-react";
import { type FC, useEffect, useRef, useState } from "react";
import {
  type Artifact,
  useArtifactStore,
} from "../stores/artifact-store";

const COPY_RESET_MS = 2000;

const ArtifactTab: FC<{ artifact: Artifact; isActive: boolean }> = ({
  artifact,
  isActive,
}) => {
  const setActive = useArtifactStore((s) => s.setActiveArtifact);
  const remove = useArtifactStore((s) => s.removeArtifact);

  return (
    <div
      role="tab"
      tabIndex={0}
      onClick={() => setActive(artifact.id)}
      onKeyDown={(e) => e.key === "Enter" && setActive(artifact.id)}
      className={`group flex cursor-pointer items-center gap-1.5 rounded-t-md border-b-2 px-3 py-1.5 text-xs font-medium transition-colors ${
        isActive
          ? "border-primary bg-background text-foreground"
          : "border-transparent text-muted-foreground hover:text-foreground"
      }`}
    >
      <span className="max-w-24 truncate">{artifact.title}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          remove(artifact.id);
        }}
        className="size-4 rounded opacity-0 hover:bg-destructive/20 group-hover:opacity-100"
      >
        <XIcon className="size-3" />
      </button>
    </div>
  );
};

export const ArtifactPanel: FC = () => {
  const artifacts = useArtifactStore((s) => s.artifacts);
  const activeId = useArtifactStore((s) => s.activeArtifactId);
  const panelOpen = useArtifactStore((s) => s.panelOpen);
  const setPanelOpen = useArtifactStore((s) => s.setPanelOpen);
  const setVersion = useArtifactStore((s) => s.setActiveVersion);
  const updateContent = useArtifactStore((s) => s.updateArtifactContent);

  const [copied, setCopied] = useState(false);
  const [localValue, setLocalValue] = useState("");
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const active = artifacts.find((a) => a.id === activeId) ?? artifacts[0];
  const viewedContent = active
    ? (active.history[active.activeVersion] ?? active.content)
    : "";

  // Sync local editor value when switching artifacts or versions
  useEffect(() => {
    if (active) setLocalValue(viewedContent);
  }, [active?.id, active?.activeVersion, viewedContent]);

  // Cleanup copy timer on unmount
  useEffect(() => {
    return () => {
      if (resetRef.current) clearTimeout(resetRef.current);
    };
  }, []);

  if (!panelOpen || artifacts.length === 0 || !active) return null;

  const handleCopy = () => {
    if (copyToClipboard(localValue)) {
      setCopied(true);
      if (resetRef.current) clearTimeout(resetRef.current);
      resetRef.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  };

  const handleDownload = () => {
    const ext = active.language === "html"
      ? ".html"
      : active.language === "svg"
        ? ".svg"
        : active.language
          ? `.${active.language}`
          : ".txt";
    downloadTextFile(`${active.title}${ext}`, localValue);
  };

  const canPrev = active.activeVersion > 0;
  const canNext = active.activeVersion < active.history.length - 1;

  return (
    <div className="flex h-full flex-col border-l bg-background">
      {/* Tab bar */}
      <div className="flex shrink-0 items-center border-b px-2">
        <div className="flex flex-1 items-center gap-0.5 overflow-x-auto">
          {artifacts.map((a) => (
            <ArtifactTab
              key={a.id}
              artifact={a}
              isActive={a.id === active.id}
            />
          ))}
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="size-7 shrink-0"
          onClick={() => setPanelOpen(false)}
        >
          <XIcon className="size-4" />
        </Button>
      </div>

      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-1 border-b px-3 py-1.5">
        <span className="flex-1 truncate text-xs font-medium">{active.title}</span>
        {active.history.length > 1 && (
          <div className="flex items-center gap-0.5 text-xs text-muted-foreground">
            <button
              type="button"
              disabled={!canPrev}
              onClick={() => setVersion(active.id, active.activeVersion - 1)}
              className="p-0.5 disabled:opacity-30"
            >
              <ChevronLeftIcon className="size-3.5" />
            </button>
            <span className="tabular-nums">
              v{active.activeVersion + 1}/{active.history.length}
            </span>
            <button
              type="button"
              disabled={!canNext}
              onClick={() => setVersion(active.id, active.activeVersion + 1)}
              className="p-0.5 disabled:opacity-30"
            >
              <ChevronRightIcon className="size-3.5" />
            </button>
          </div>
        )}
        <button type="button" onClick={handleCopy} className="p-1 text-muted-foreground hover:text-foreground">
          {copied ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
        </button>
        <button type="button" onClick={handleDownload} className="p-1 text-muted-foreground hover:text-foreground">
          <DownloadIcon className="size-3.5" />
        </button>
      </div>

      {/* Editor */}
      <div className="flex-1 overflow-auto">
        <textarea
          ref={textareaRef}
          value={localValue}
          onChange={(e) => setLocalValue(e.target.value)}
          onBlur={() => {
            if (localValue !== viewedContent) {
              updateContent(active.id, localValue);
            }
          }}
          className="h-full w-full resize-none bg-transparent p-4 font-mono text-xs leading-relaxed outline-none"
          spellCheck={false}
        />
      </div>
    </div>
  );
};
