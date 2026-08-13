import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type ProxyOptions } from "vite";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxy: Record<string, ProxyOptions> = {};
  if (env.VITE_RAG_PLATFORM_ENABLED?.trim().toLowerCase() !== "false") {
    proxy["/api/v1"] = {
      // The owned runtime exposes its method-aware Python/Go hybrid map through
      // nginx on port 80. Keeping this default prevents `/api/v1` from falling
      // through to the legacy `/api` proxy when a developer has not copied the
      // example env file yet.
      target:
        env.VITE_RAG_PLATFORM_PROXY_TARGET?.trim() ||
        "http://127.0.0.1",
      changeOrigin: true,
    };
  }
  Object.assign(proxy, {
    "/api": {
      target: "http://127.0.0.1:8888",
      changeOrigin: true,
    },
    "/v1": {
      target: "http://127.0.0.1:8888",
      changeOrigin: true,
    },
    "/seed/inspect": {
      target: "http://127.0.0.1:8004",
      changeOrigin: true,
    },
    "/seed/preview": {
      target: "http://127.0.0.1:8004",
      changeOrigin: true,
    },
    "/preview": {
      target: "http://127.0.0.1:8004",
      changeOrigin: true,
    },
    "/validate": {
      target: "http://127.0.0.1:8004",
      changeOrigin: true,
    },
    "/tools": {
      target: "http://127.0.0.1:8004",
      changeOrigin: true,
    },
  });

  return {
    plugins: [react(), tailwindcss()],
    // Keep an unrelated PostCSS config in an ancestor directory from leaking
    // into Studio installs. Tailwind is provided by its dedicated Vite plugin.
    css: {
      postcss: {
        plugins: [],
      },
    },
    optimizeDeps: {
      include: ["@dagrejs/dagre", "@dagrejs/graphlib"],
    },
    server: {
      host: "0.0.0.0",
      allowedHosts: true,
      proxy,
    },
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
        "@dagrejs/dagre": path.resolve(
          __dirname,
          "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
        ),
      },
    },
    build: {
      commonjsOptions: {
        include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
      },
    },
  };
});
