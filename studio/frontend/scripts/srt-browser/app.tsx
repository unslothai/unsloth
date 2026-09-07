import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { PermissionModeDropdown } from "../../src/features/chat/permission-mode-select";
import { useChatRuntimeStore } from "../../src/features/chat/stores/chat-runtime-store";
import {
  toolExecutionRecordLabel,
  parseBackendExecutionRecord,
} from "../../src/features/chat/types/api";
import "../../src/index.css";
const nativeFetch = window.fetch.bind(window);
window.fetch = (input, init) =>
  nativeFetch(
    typeof input === "string" && input.startsWith("/api/")
      ? "http://127.0.0.1:5198" + input
      : input,
    init,
  );
function App() {
  const [output, setOutput] = useState("");
  const [label, setLabel] = useState("");
  async function run(kind: string) {
    const state = useChatRuntimeStore.getState();
    const body = await nativeFetch("http://127.0.0.1:5198/fixture/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind,
        mode: state.toolExecutionMode,
        session: state.toolIsolationUiSessionId,
        grant: state.limitedToolGrant?.grant,
      }),
    }).then((r) => r.json());
    (window as any).lastExecution = body;
    setOutput(body.result);
    setLabel(
      toolExecutionRecordLabel(
        parseBackendExecutionRecord(body.records?.[0]),
      ) ?? "No execution record",
    );
  }
  return (
    <main style={{ padding: 40, maxWidth: 1100 }}>
      <h1>Production API and selected native tools</h1>
      <p>
        Isolated test app: production consent controls, capability/grant
        endpoints and execute_tool. Fixed benign payloads; test authentication
        identity.
      </p>
      <PermissionModeDropdown />
      <div style={{ marginTop: 24, display: "flex", gap: 12 }}>
        <button onClick={() => run("python")}>Run fixed Python</button>
        <button onClick={() => run("terminal")}>Run fixed Terminal</button>
      </div>
      <h2>{label}</h2>
      <pre style={{ whiteSpace: "pre-wrap" }}>{output}</pre>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
