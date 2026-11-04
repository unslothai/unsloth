// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRouter } from "@tanstack/react-router";
import {
  ArrowLeftIcon,
  Maximize2Icon,
  Minimize2Icon,
  SparklesIcon,
} from "lucide-react";
import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  FlowDocument,
  FlowKind,
  RunStatus,
  StoryManifest,
  WorldSpec,
} from "../types";
import { ChatPlayer } from "./chat/chat-player";
import { type ChatEntry, chatEntry } from "./chat/chat-state";
import { type PlayerApi, clearMediaUrls, playerApi } from "./player-api";
import { StoryPlayer, storySeed } from "./story/story-player";
import type { WorldGameHandle } from "./world/phaser-game";

// The world player pulls in Phaser through its own import(); keep it out of the story path entirely.
const WorldPlayer = lazy(() =>
  import("./world/world-player").then((m) => ({ default: m.WorldPlayer })),
);

type Loaded =
  | { kind: "story"; manifest: StoryManifest }
  | { kind: "world"; spec: WorldSpec }
  | { kind: "chat"; entry: ChatEntry }
  | { kind: "pipeline" };

const GENERATORS: Record<string, string> = {
  story: "story.generator",
  world: "world.generator",
};

export interface FlowPlayerPageProps {
  flowId: string;
  api?: PlayerApi;
  onExit?: () => void;
  onWorldReady?: (game: WorldGameHandle) => void;
}

export function FlowPlayerPage({
  flowId,
  api = playerApi,
  onExit,
  onWorldReady,
}: FlowPlayerPageProps) {
  const router = useRouter({ warn: false }) as
    ReturnType<typeof useRouter> | undefined;
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [doc, setDoc] = useState<FlowDocument | null>(null);
  const [name, setName] = useState("");
  const [version, setVersion] = useState(0);
  const [rerolling, setRerolling] = useState<RunStatus | "starting" | null>(
    null,
  );
  const [full, setFull] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  const exit = useCallback(() => {
    if (onExit) onExit();
    else router?.history.push(`/flows/${encodeURIComponent(flowId)}`);
  }, [onExit, router, flowId]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` re-fetches after a re-roll
  useEffect(() => {
    let live = true;
    setError(null);
    (async () => {
      try {
        const info = await api.flowInfo(flowId);
        if (!live) return;
        setName(info.name);
        setDoc(info.doc ?? null);
        const kind: FlowKind = info.kind;
        if (kind === "story") {
          const manifest = await api.story(flowId);
          if (live) setLoaded({ kind: "story", manifest });
        } else if (kind === "world") {
          const spec = await api.world(flowId);
          if (live) setLoaded({ kind: "world", spec });
        } else if (live) {
          const entry = chatEntry(info.doc);
          setLoaded(entry ? { kind: "chat", entry } : { kind: "pipeline" });
        }
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
    };
  }, [api, flowId, version]);

  useEffect(() => () => clearMediaUrls(), []);

  useEffect(() => {
    const onFs = () => setFull(document.fullscreenElement === box.current);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const generator = doc?.nodes.find(
    (n) => n.type === GENERATORS[doc.kind] && !n.disabled,
  );
  const reroll = useCallback(async () => {
    setRerolling("starting");
    try {
      const seed = Math.floor(Math.random() * 2 ** 31);
      await api.reroll(flowId, seed, (s) => setRerolling(s));
      setLoaded(null);
      setVersion((v) => v + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRerolling(null);
    }
  }, [api, flowId]);

  const toggleFull = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void box.current?.requestFullscreen?.();
  };

  return (
    // Like the editor: the shell gives this route no top inset, so the game starts below the titlebar
    // and the window buttons and app sidebar toggle never sit on top of its HUD.
    <div className="h-full w-full bg-black pt-[var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))]">
      <div
        ref={box}
        className="relative h-full min-h-120 w-full overflow-hidden bg-black"
        data-testid="flow-player-page"
      >
        {error ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center text-white">
            <p className="font-semibold text-xl">
              {name || "This flow"} is not ready to play
            </p>
            <p className="max-w-lg text-white/65">{error}</p>
            <p className="max-w-lg text-sm text-white/45">
              Run the flow in the editor first (it writes the story or builds
              the world), then press Play again.
            </p>
            <button
              type="button"
              onClick={exit}
              className="mt-2 flex items-center gap-2 rounded-full bg-white px-5 py-2 font-medium text-black"
            >
              <ArrowLeftIcon className="size-4" /> Back to the editor
            </button>
          </div>
        ) : loaded ? (
          loaded.kind === "story" ? (
            <StoryPlayer
              key={version}
              flowId={flowId}
              api={api}
              manifest={loaded.manifest}
              seed={storySeed(flowId) + version}
              canReroll={Boolean(generator)}
              onReroll={reroll}
              onExit={exit}
            />
          ) : loaded.kind === "world" ? (
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center text-white/60">
                  Starting the world...
                </div>
              }
            >
              <WorldPlayer
                key={version}
                onReady={onWorldReady}
                flowId={flowId}
                api={api}
                spec={loaded.spec}
                canReroll={Boolean(generator)}
                onReroll={reroll}
                onExit={exit}
              />
            </Suspense>
          ) : loaded.kind === "chat" ? (
            <ChatPlayer
              key={version}
              flowId={flowId}
              api={api}
              entry={loaded.entry}
              title={name || "Chat"}
              onExit={exit}
            />
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-4 text-white">
              <p className="text-lg">This flow is a pipeline, not a game.</p>
              <button
                type="button"
                onClick={exit}
                className="rounded-full bg-white px-5 py-2 font-medium text-black"
              >
                Back to the editor
              </button>
            </div>
          )
        ) : (
          <div className="flex h-full items-center justify-center gap-3 text-white/70">
            <SparklesIcon className="size-5 animate-pulse text-amber-200" />{" "}
            Loading {name || "the game"}...
          </div>
        )}
        {rerolling ? (
          <div className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-black/75 text-white backdrop-blur-sm">
            <SparklesIcon className="size-7 animate-pulse text-fuchsia-300" />
            <p className="text-lg">
              Dreaming up a new {doc?.kind === "world" ? "world" : "story"}...
            </p>
            <p className="text-sm text-white/50">
              {rerolling === "starting" ? "Starting" : rerolling}
            </p>
          </div>
        ) : null}
        <button
          type="button"
          onClick={toggleFull}
          title={full ? "Exit full screen" : "Full screen"}
          aria-label={full ? "Exit full screen" : "Full screen"}
          className="absolute right-3 bottom-3 z-40 rounded-full bg-black/40 p-2 text-white/60 backdrop-blur hover:bg-black/60 hover:text-white"
        >
          {full ? (
            <Minimize2Icon className="size-4" />
          ) : (
            <Maximize2Icon className="size-4" />
          )}
        </button>
      </div>
    </div>
  );
}
