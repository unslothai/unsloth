import { WindowTitlebar } from "./src/components/tauri/window-titlebar";
// Backend-free wallpaper integration fixture: production settings and renderer.
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChatBackground } from "./src/features/chat-background/background";
import { BACKGROUND_ON_ALL_PAGES } from "./src/features/chat-background/model";
import { ChatBackgroundControls } from "./src/features/chat-background/controls";
import {
  resetChatBackground,
  patchBackground,
  useBackgroundStore,
} from "./src/features/chat-background/storage";
import { TooltipProvider } from "./src/components/ui/tooltip";
import "./src/index.css";
declare global {
  interface Window {
    __wallpaperSmoke: {
      patch: typeof patchBackground;
      state: typeof useBackgroundStore.getState;
    };
  }
}
window.__wallpaperSmoke = {
  patch: patchBackground,
  state: useBackgroundStore.getState,
};
function App() {
  const [conversation, setConversation] = useState(false),
    [active, setActive] = useState(true),
    [sidebar, setSidebar] = useState(false),
    [otherPage, setOtherPage] = useState(false),
    [dark, setDark] = useState(false);
  return (
    <TooltipProvider>
      <div className="p-4 text-foreground">
        <div className="mb-4 flex flex-wrap gap-4">
          <button onClick={() => setConversation((v) => !v)}>
            Toggle conversation
          </button>
          <button
            onClick={() => {
              setDark(!dark);
              document.documentElement.classList.toggle("dark", !dark);
            }}
          >
            Toggle theme
          </button>
          <button onClick={() => setActive((v) => !v)}>Toggle route</button>
          <button onClick={() => setSidebar(v => !v)}>Toggle sidebar</button>
          <button onClick={() => setOtherPage(v => !v)}>Toggle page</button>
          <button onClick={() => void resetChatBackground()}>Reset all</button>
        </div>
        <div className="grid gap-8 lg:grid-cols-2">
          <div className="settings-page">
            <ChatBackgroundControls />
          </div>
          <div
            data-chat-content-root=""
            data-wallpaper-workspace=""
            data-wallpaper-route={otherPage ? "page" : "chat"}
            className="relative h-[700px] overflow-hidden rounded-xl border bg-background"
          >
            <ChatBackground active={active && (!otherPage || BACKGROUND_ON_ALL_PAGES)} />
            <WindowTitlebar showSidebarSurface />
            {sidebar && <aside data-slot="sidebar" className="absolute inset-y-0 left-0 z-50 w-40 bg-sidebar p-4">Unsloth<br />New chat<br />Model hub</aside>}
            <main data-slot="sidebar-inset" className="relative h-full" style={{marginLeft: sidebar ? 160 : 0}}>
            {otherPage && <div className="absolute inset-0 z-40 p-16"><h1>Model hub</h1><div className="mt-4 rounded-xl bg-card p-6">Models and downloads</div></div>}
            <div style={{visibility: otherPage ? "hidden" : "visible"}} className="h-full">
            <div data-side-panel-inset="" className="absolute inset-x-0 top-[var(--studio-content-top-inset,0px)] z-40 flex h-12 items-center justify-between bg-background px-3">
              <div data-wallpaper-chrome-controls="" className="p-2 text-xs">Select model</div>
              <div data-wallpaper-chrome-controls="" className="p-2 text-xs">Search · Browser</div>
            </div>
            <div data-side-panel-inset="" className="chat-header-fade pointer-events-none absolute inset-x-0 top-[calc(var(--studio-content-top-inset,0px)+48px)] z-20 h-6 bg-gradient-to-b from-background to-transparent" />
            <div className="aui-thread-root relative h-full">
              <div className="aui-thread-viewport h-full p-8">
                <div
                  className={
                    conversation
                      ? "space-y-8 pt-12"
                      : "aui-thread-welcome-center pt-48"
                  }
                >
                  {conversation ? (
                    <>
                      <p className="text-muted-foreground">
                        Tell me about this scene.
                      </p>
                      <p>
                        A clear conversation, with the artwork quietly in the
                        background.
                      </p>
                      <pre className="rounded-xl bg-card p-4">
                        const greeting = "Hello";
                      </pre>
                    </>
                  ) : (
                    <h1 className="unsloth-welcome-greeting mb-6 text-center text-2xl">
                      What would you like to explore?
                    </h1>
                  )}
                  <div className="unsloth-composer-surface mt-8 p-6">
                    <textarea
                      aria-label="Message"
                      placeholder="Ask anything…"
                      className="w-full bg-transparent outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>
            </div>
            </main>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
const root = createRoot(document.getElementById("root")!);
if (import.meta.hot) import.meta.hot.dispose(() => root.unmount());
root.render(
  <StrictMode>
    <App />
  </StrictMode>,
);
