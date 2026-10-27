// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// matches AppSidebar's scroller root and unstable renderItem identity.
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { ProgressiveRows } from "@/components/progressive-rows";
import "./styles.css";

const params = new URLSearchParams(location.search);
const height = Number(params.get("height") ?? 600);

type Item = { id: string };
const makeItems = (count: number): Item[] =>
  Array.from({ length: count }, (_, i) => ({ id: `chat-${i}` }));

declare global {
  interface Window {
    rowRenders: number;
    fixture: {
      setCount: (count: number) => void;
      setMounted: (mounted: boolean) => void;
      // uses new item objects with stable ids, matching a chat-list refetch.
      refresh: () => void;
    };
  }
}
window.rowRenders = 0;
// tracks renderItem calls to detect unnecessary row redraws.
const countRowRender = () => {
  window.rowRenders += 1;
};

function App() {
  const [items, setItems] = useState(() =>
    makeItems(Number(params.get("count") ?? 1000)),
  );
  const [mounted, setMounted] = useState(true);
  useEffect(() => {
    window.fixture = {
      setCount: (count) => setItems(makeItems(count)),
      setMounted,
      refresh: () => setItems((current) => current.map(({ id }) => ({ id }))),
    };
  }, []);
  return (
    <div data-sidebar="content" style={{ height, overflowY: "auto" }}>
      {mounted ? (
        <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
          <ProgressiveRows
            items={items}
            pageSize={50}
            renderItem={(item) => {
              countRowRender();
              return (
                <li key={item.id} data-row={item.id} style={{ height: 30 }}>
                  {item.id}
                </li>
              );
            }}
            end={<li data-end="">end</li>}
          />
        </ul>
      ) : null}
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
