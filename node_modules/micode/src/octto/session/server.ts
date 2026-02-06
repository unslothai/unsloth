// src/octto/session/server.ts
import type { Server, ServerWebSocket } from "bun";

import { config } from "../../utils/config";
import { getHtmlBundle } from "../ui";
import type { SessionStore } from "./sessions";
import type { WsClientMessage } from "./types";

interface WsData {
  sessionId: string;
}

export async function createServer(
  sessionId: string,
  store: SessionStore,
): Promise<{ server: Server<WsData>; port: number }> {
  const htmlBundle = getHtmlBundle();

  const server = Bun.serve<WsData>({
    port: 0, // Random available port
    hostname: config.octto.allowRemoteBind ? config.octto.bindAddress : "127.0.0.1",
    fetch(req, server) {
      const url = new URL(req.url);

      // WebSocket upgrade
      if (url.pathname === "/ws") {
        const success = server.upgrade(req, {
          data: { sessionId },
        });
        if (success) {
          return undefined;
        }
        return new Response("WebSocket upgrade failed", { status: 400 });
      }

      // Serve the bundled HTML app
      if (url.pathname === "/" || url.pathname === "/index.html") {
        return new Response(htmlBundle, {
          headers: {
            "Content-Type": "text/html; charset=utf-8",
          },
        });
      }

      return new Response("Not Found", { status: 404 });
    },
    websocket: {
      open(ws: ServerWebSocket<WsData>) {
        const { sessionId } = ws.data;
        store.handleWsConnect(sessionId, ws);
      },
      close(ws: ServerWebSocket<WsData>) {
        const { sessionId } = ws.data;
        store.handleWsDisconnect(sessionId);
      },
      message(ws: ServerWebSocket<WsData>, message: string | Buffer) {
        const { sessionId } = ws.data;

        let parsed: WsClientMessage;
        try {
          parsed = JSON.parse(message.toString()) as WsClientMessage;
        } catch (error) {
          console.error("[octto] Failed to parse WebSocket message:", error);
          ws.send(
            JSON.stringify({
              type: "error",
              error: "Invalid message format",
              details: error instanceof Error ? error.message : "Parse failed",
            }),
          );
          return;
        }

        store.handleWsMessage(sessionId, parsed);
      },
    },
  });

  // Port is always defined when using port: 0
  const port = server.port;
  if (port === undefined) {
    throw new Error("Failed to get server port");
  }

  return {
    server,
    port,
  };
}
