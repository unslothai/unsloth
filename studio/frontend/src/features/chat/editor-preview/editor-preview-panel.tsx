// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { readFrames } from "./frame-stream";

const editors = {
  blender: "Blender", godot: "Godot", unity: "Unity",
  unreal: "Unreal Engine", roblox: "Roblox Studio",
};
type Editor = keyof typeof editors;
type Session = { id: string; pairing: string };
type Gesture = Partial<Record<"orbit_x" | "orbit_y" | "pan_x" | "pan_y" | "zoom", number>>;

async function request(path: string, init?: RequestInit) {
  const response = await authFetch("/api/editor-preview" + path, init, { retryNetworkErrors: false });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === "string" ? error.detail : "Editor preview request failed.");
  }
  return response;
}

function close(id: string) {
  void request("/sessions/" + id, { method: "DELETE", keepalive: true }).catch(() => undefined);
}

export function EditorPreviewPanel({ onClose }: { onClose: () => void }) {
  const [editor, setEditor] = useState<Editor>("blender");
  const [session, setSession] = useState<Session>();
  const [status, setStatus] = useState<"waiting" | "live" | "paused" | "ended">("waiting");
  const [showFrame, setShowFrame] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const active = useRef<Session | undefined>(undefined);
  const mounted = useRef(false);
  const surface = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ id: number; x: number; y: number } | null>(null);
  const pending = useRef<Gesture>({});
  const timer = useRef(0);
  const sending = useRef(false);

  const disconnect = useCallback(() => {
    if (active.current) close(active.current.id);
    active.current = undefined;
    drag.current = null;
    pending.current = {};
    window.clearTimeout(timer.current);
    timer.current = 0;
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      disconnect();
    };
  }, [disconnect]);

  useEffect(() => {
    if (!session) return;
    const controller = new AbortController();
    let seen = 0;
    let drawing = false;
    let latest: Uint8Array<ArrayBuffer> | undefined;
    async function paint() {
      if (drawing) return;
      drawing = true;
      try {
        while (latest && !controller.signal.aborted) {
          const bytes = latest;
          latest = undefined;
          const bitmap = await createImageBitmap(new Blob([bytes], {
            type: bytes[0] === 0x89 ? "image/png" : "image/jpeg",
          })).catch(() => null);
          if (!bitmap) continue;
          const canvas = surface.current;
          if (canvas && !controller.signal.aborted) {
            if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
            if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
            canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
            seen = performance.now();
            setShowFrame(true);
            setStatus("live");
          }
          bitmap.close();
        }
      } finally { drawing = false; }
    }
    const watch = window.setInterval(() => {
      if (seen && performance.now() - seen > 2000) {
        drag.current = null;
        pending.current = {};
        setStatus((current) => current === "live" ? "paused" : current);
      }
    }, 500);
    void request("/sessions/" + session.id + "/frames", { signal: controller.signal }).then(async (response) => {
      if (!response.body) throw new Error("The viewport stream has no response body.");
      await readFrames(response.body, (frame) => {
        latest = frame;
        void paint();
      });
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Viewport disconnected.");
    }).finally(() => {
      if (!controller.signal.aborted) setStatus("ended");
    });
    return () => {
      controller.abort();
      window.clearInterval(watch);
    };
  }, [session]);

  const input = useCallback((delta: Gesture) => {
    const target = active.current;
    if (!target || status !== "live") return;
    for (const [key, amount] of Object.entries(delta)) {
      const axis = key as keyof Gesture;
      pending.current[axis] = Math.max(-.25, Math.min(.25, (pending.current[axis] ?? 0) + amount));
    }
    if (timer.current || sending.current) return;
    function flush(id: string) {
      timer.current = 0;
      if (active.current?.id !== id || !Object.keys(pending.current).length) return;
      const body = JSON.stringify(pending.current);
      pending.current = {};
      sending.current = true;
      void request("/sessions/" + id + "/input", {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      }).catch(() => undefined).finally(() => {
        sending.current = false;
        const next = active.current?.id;
        if (next && Object.keys(pending.current).length && !timer.current) {
          timer.current = window.setTimeout(() => flush(next), 30);
        }
      });
    }
    timer.current = window.setTimeout(() => flush(target.id), 30);
  }, [status]);

  useEffect(() => {
    const canvas = surface.current;
    if (!canvas || status !== "live") return;
    function wheel(event: WheelEvent) {
      event.preventDefault();
      const scale = event.deltaMode === 1 ? 12 : event.deltaMode === 2 ? 200 : 1;
      input({ zoom: event.deltaY * scale / 600 });
    }
    canvas.addEventListener("wheel", wheel, { passive: false });
    return () => canvas.removeEventListener("wheel", wheel);
  }, [status, input]);

  async function start() {
    disconnect();
    setSession(undefined);
    setStatus("waiting");
    setShowFrame(false);
    setCopied(false);
    setBusy(true);
    setError("");
    try {
      const response = await request("/sessions/" + editor, { method: "POST" });
      const next = (await response.json()) as Session;
      if (!mounted.current) close(next.id);
      else {
        active.current = next;
        setSession(next);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start preview.");
    } finally { setBusy(false); }
  }

  async function download() {
    setError("");
    try {
      const response = await request("/adapters/" + editor);
      const href = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = href;
      link.download = "unsloth-preview-" + editor + ".zip";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 2000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not download adapter.");
    }
  }

  return (
    <section aria-label="Editor preview" className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background">
      <header className="flex items-center justify-between gap-3 border-b px-4 py-3">
        <h2 className="text-sm font-semibold">Editor preview</h2>
        <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
      </header>
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <select aria-label="Editor" className="rounded-md border bg-background px-2 py-1.5 text-sm"
          value={editor} disabled={busy || Boolean(session && status !== "ended")}
          onChange={(event) => setEditor(event.currentTarget.value as Editor)}>
          {Object.entries(editors).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
        </select>
        <Button size="sm" variant="secondary" onClick={() => void download()}>Download adapter</Button>
        {!session || status === "ended"
          ? <Button size="sm" disabled={busy} onClick={() => void start()}>Start preview</Button>
          : <Button size="sm" variant="outline" onClick={() => {
              disconnect();
              setSession(undefined);
              setShowFrame(false);
              setStatus("waiting");
            }}>Disconnect</Button>}
        {session && status !== "ended" ? <Button size="sm" variant="outline" onClick={() => {
          void navigator.clipboard.writeText(session.pairing).then(() => setCopied(true)).catch(() => {
            setError("Select and copy the pairing code below.");
            setCopied(false);
          });
        }}>{copied ? "Copied" : "Copy pairing code"}</Button> : null}
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[#171a1e]">
        <canvas ref={surface} aria-label="Interactive editor viewport" tabIndex={status === "live" ? 0 : -1}
          className={"absolute inset-0 h-full w-full touch-none object-contain " +
            (showFrame ? "cursor-grab active:cursor-grabbing" : "invisible")}
          onPointerDown={(event) => {
            if (status !== "live" || event.button !== 0) return;
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
          }}
          onPointerMove={(event) => {
            const previous = drag.current;
            if (!previous || previous.id !== event.pointerId) return;
            const bounds = event.currentTarget.getBoundingClientRect();
            const x = (event.clientX - previous.x) / Math.max(1, bounds.width);
            const y = (event.clientY - previous.y) / Math.max(1, bounds.height);
            drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
            input(event.shiftKey ? { pan_x: x, pan_y: y } : { orbit_x: x, orbit_y: y });
          }}
          onPointerUp={() => { drag.current = null; }}
          onPointerCancel={() => { drag.current = null; }}
          onLostPointerCapture={() => { drag.current = null; }}
          onKeyDown={(event) => {
            const delta: Record<string, Gesture> = {
              ArrowLeft: event.shiftKey ? { pan_x: -.04 } : { orbit_x: -.04 },
              ArrowRight: event.shiftKey ? { pan_x: .04 } : { orbit_x: .04 },
              ArrowUp: event.shiftKey ? { pan_y: -.04 } : { orbit_y: -.04 },
              ArrowDown: event.shiftKey ? { pan_y: .04 } : { orbit_y: .04 },
              "+": { zoom: -.1 }, "=": { zoom: -.1 }, "-": { zoom: .1 },
            };
            if (delta[event.key]) {
              event.preventDefault();
              input(delta[event.key]);
            } else if (event.key === "Escape") {
              drag.current = null;
              event.currentTarget.blur();
            }
          }}
        />
        {!showFrame ? <p className="px-5 text-center text-sm text-white/75">
          {session && status === "waiting" ? "Paste the pairing code into the editor adapter." : "Choose an editor and start a local preview."}
        </p> : null}
      </div>
      <footer className="shrink-0 border-t px-4 py-2 text-xs text-muted-foreground">
        <p role="status">{status === "live" ? "Live · Drag to orbit · Shift-drag to pan · Scroll to zoom"
          : status === "paused" ? "Waiting for the editor to reconnect"
          : status === "ended" ? "Disconnected"
          : session ? "Waiting for " + editors[editor] : "Local viewport"}</p>
        {session && status !== "ended" && (status === "waiting" || error) ?
          <input aria-label="Pairing code" readOnly value={session.pairing}
            className="mt-2 w-full rounded border bg-background px-2 py-1 font-mono text-xs"
            onFocus={(event) => event.currentTarget.select()} /> : null}
        {error ? <p role="alert" className="mt-1 text-destructive">{error}</p> : null}
      </footer>
    </section>
  );
}
