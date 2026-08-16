"use client";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import {
  generatePlatformMindMapForChat,
  getPlatformRecommendationsForChat,
  submitPlatformMessageFeedbackForChat,
} from "@/features/chat/api/platform-chat-adapter";
import {
  isPlatformChatPersistenceEnabled,
  type PlatformMindMapNode,
} from "@/integrations/platform-backend";
import { useAui, useAuiState, useMessage } from "@assistant-ui/react";
import {
  Download,
  Network,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { type FC, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function textContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((part) => {
      const item = record(part);
      return item?.type === "text" && typeof item.text === "string"
        ? [item.text]
        : [];
    })
    .join("\n")
    .trim();
}

function metadataValue(metadata: unknown, key: string): unknown {
  const value = record(metadata);
  return record(value?.custom)?.[key] ?? value?.[key];
}

function usePlatformMessageContext() {
  const aui = useAui();
  const message = useMessage();
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const question = useMemo(() => {
    const messages = aui.thread().getState().messages;
    const currentIndex = messages.findIndex((item) => item.id === message.id);
    const prior = messages
      .slice(0, currentIndex < 0 ? messages.length : currentIndex)
      .reverse()
      .find((item) => item.role === "user");
    return textContent(prior?.content);
  }, [aui, message.id]);
  const platformMessageId = metadataValue(
    message.metadata,
    "platformMessageId",
  );
  return {
    aui,
    question,
    sessionId: remoteId ?? aui.threadListItem().getState().id ?? null,
    platformMessageId:
      typeof platformMessageId === "string" && platformMessageId
        ? platformMessageId
        : null,
  };
}

export const MindMapTree: FC<{ node: PlatformMindMapNode }> = ({ node }) => (
  <ul className="space-y-2" aria-label="Mindmap dalları">
    <li>
      <div className="rounded-xl border bg-muted/35 px-3 py-2 font-medium">
        {node.label}
      </div>
      {node.children.length > 0 && (
        <div className="ml-4 mt-2 border-l pl-3">
          {node.children.map((child) => (
            <MindMapTree key={child.id} node={child} />
          ))}
        </div>
      )}
    </li>
  </ul>
);

export const PlatformChatEnrichments: FC = () => {
  const { aui, question, sessionId } = usePlatformMessageContext();
  const [mindMapOpen, setMindMapOpen] = useState(false);
  const [mindMap, setMindMap] = useState<PlatformMindMapNode | null>(null);
  const [mindMapError, setMindMapError] = useState<string | null>(null);
  const [mindMapLoading, setMindMapLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<string[] | null>(null);
  const [recommendationError, setRecommendationError] = useState<string | null>(
    null,
  );
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const controllers = useRef(new Set<AbortController>());

  useEffect(
    () => () => {
      for (const controller of controllers.current) controller.abort();
      controllers.current.clear();
    },
    [],
  );

  if (!isPlatformChatPersistenceEnabled() || !sessionId || !question)
    return null;

  const loadMindMap = async () => {
    setMindMapOpen(true);
    if (mindMap || mindMapLoading) return;
    const controller = new AbortController();
    controllers.current.add(controller);
    setMindMapLoading(true);
    setMindMapError(null);
    try {
      setMindMap(
        await generatePlatformMindMapForChat(
          sessionId,
          question,
          controller.signal,
        ),
      );
    } catch (error) {
      if (!controller.signal.aborted) {
        setMindMapError(
          error instanceof Error ? error.message : "Mindmap oluşturulamadı.",
        );
      }
    } finally {
      controllers.current.delete(controller);
      if (!controller.signal.aborted) setMindMapLoading(false);
    }
  };

  const loadRecommendations = async () => {
    if (recommendationLoading) return;
    if (recommendations !== null) {
      setRecommendations(null);
      return;
    }
    const controller = new AbortController();
    controllers.current.add(controller);
    setRecommendationLoading(true);
    setRecommendationError(null);
    try {
      setRecommendations(
        await getPlatformRecommendationsForChat(question, controller.signal),
      );
    } catch (error) {
      if (!controller.signal.aborted) {
        setRecommendationError(
          error instanceof Error ? error.message : "Öneriler alınamadı.",
        );
      }
    } finally {
      controllers.current.delete(controller);
      if (!controller.signal.aborted) setRecommendationLoading(false);
    }
  };

  const exportMindMap = () => {
    if (!mindMap) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(mindMap, null, 2)], {
        type: "application/json",
      }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `rag-platform-mindmap-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-3 space-y-2" aria-label="Rag Platform yanıt araçları">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => void loadMindMap()}
        >
          <Network className="size-4" aria-hidden="true" />
          {mindMapLoading ? "Mindmap hazırlanıyor" : "Mindmap"}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => void loadRecommendations()}
        >
          <Sparkles className="size-4" aria-hidden="true" />
          {recommendationLoading ? "Öneriler hazırlanıyor" : "Takip önerileri"}
        </Button>
      </div>

      {recommendationError && (
        <p role="alert" className="text-sm text-destructive">
          {recommendationError}
        </p>
      )}
      {recommendations !== null && !recommendationError && (
        <div
          className="flex flex-wrap gap-2"
          aria-label="Takip sorusu önerileri"
        >
          {recommendations.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Takip önerisi bulunamadı.
            </p>
          ) : (
            recommendations.map((recommendation) => (
              <Button
                key={recommendation}
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => {
                  aui.composer().setText(recommendation);
                }}
              >
                {recommendation}
              </Button>
            ))
          )}
        </div>
      )}

      <Dialog open={mindMapOpen} onOpenChange={setMindMapOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Mindmap</DialogTitle>
            <DialogDescription>
              Yanıta bağlı bilgi dallarını inceleyin veya JSON olarak dışa
              aktarın.
            </DialogDescription>
          </DialogHeader>
          {mindMapLoading ? (
            <p role="status">Mindmap hazırlanıyor…</p>
          ) : mindMapError ? (
            <div role="alert" className="space-y-3 text-destructive">
              <p>{mindMapError}</p>
              <Button
                variant="outline"
                onClick={() => {
                  setMindMap(null);
                  void loadMindMap();
                }}
              >
                Yeniden dene
              </Button>
            </div>
          ) : mindMap ? (
            <div className="max-h-[55dvh] overflow-auto pr-2">
              <MindMapTree node={mindMap} />
            </div>
          ) : (
            <p className="text-muted-foreground">Mindmap verisi bulunamadı.</p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={!mindMap}
              onClick={exportMindMap}
            >
              <Download className="size-4" aria-hidden="true" />
              JSON dışa aktar
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export const PlatformFeedbackActions: FC = () => {
  const { sessionId, platformMessageId } = usePlatformMessageContext();
  const metadata = useMessage().metadata;
  const initial = metadataValue(metadata, "platformThumbup");
  const [selection, setSelection] = useState<boolean | null>(
    typeof initial === "boolean" ? initial : null,
  );
  const [negativeOpen, setNegativeOpen] = useState(false);
  const [feedback, setFeedback] = useState(
    typeof metadataValue(metadata, "platformFeedback") === "string"
      ? String(metadataValue(metadata, "platformFeedback"))
      : "",
  );
  const [saving, setSaving] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  if (!isPlatformChatPersistenceEnabled() || !sessionId || !platformMessageId) {
    return null;
  }

  const submit = async (thumbup: boolean, detail?: string) => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setSaving(true);
    try {
      await submitPlatformMessageFeedbackForChat(
        sessionId,
        platformMessageId,
        { thumbup, ...(detail?.trim() ? { feedback: detail.trim() } : {}) },
        next.signal,
      );
      setSelection(thumbup);
      setNegativeOpen(false);
      toast.success("Geri bildiriminiz kaydedildi.");
    } catch (error) {
      if (!next.signal.aborted) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Geri bildirim kaydedilemedi.",
        );
      }
    } finally {
      if (controller.current === next) controller.current = null;
      if (!next.signal.aborted) setSaving(false);
    }
  };

  return (
    <>
      <TooltipIconButton
        tooltip="Yararlı"
        aria-label="Yanıt yararlı"
        aria-pressed={selection === true}
        disabled={saving}
        onClick={() => void submit(true)}
      >
        <ThumbsUp className="size-icon" />
      </TooltipIconButton>
      <TooltipIconButton
        tooltip="Yararlı değil"
        aria-label="Yanıt yararlı değil"
        aria-pressed={selection === false}
        disabled={saving}
        onClick={() => setNegativeOpen(true)}
      >
        <ThumbsDown className="size-icon" />
      </TooltipIconButton>
      <Dialog open={negativeOpen} onOpenChange={setNegativeOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Yanıt geri bildirimi</DialogTitle>
            <DialogDescription>
              İsterseniz yanıtın neden yararlı olmadığını açıklayın.
            </DialogDescription>
          </DialogHeader>
          <Textarea
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="İsteğe bağlı açıklama"
            maxLength={2_000}
          />
          <DialogFooter>
            <Button
              disabled={saving}
              onClick={() => void submit(false, feedback)}
            >
              {saving ? "Kaydediliyor" : "Gönder"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
