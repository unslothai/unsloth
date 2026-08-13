#!/usr/bin/env node
/**
 * Rag Platform backend route inventory generator.
 *
 * Re-runnable: it re-reads the pinned backend source ref on every run and rewrites
 * docs/rag-platform/route-inventory.json + route-inventory.md from scratch.
 * Nothing in the outputs is hand-edited.
 *
 * Discovery surfaces (all verified present in the backend tree):
 *   Python API    api/apps/restful_apis/*.py    Quart blueprints, prefix /api/v1   port 9380
 *   Python API    api/apps/backward_compat.py   two blueprints, /api/v1 and /v1    port 9380
 *   Python admin  admin/server/routes.py        Blueprint prefix /api/v1/admin     port 9381
 *   Go API        internal/router/*.go          gin groups                         port 9384
 *   Go admin      internal/admin/router.go      gin groups                         port 9383
 *   MCP           mcp/server/server.py          starlette Route/Mount              port 9382
 *
 * Proxy resolution comes from docker/nginx/ragflow.conf.<scheme> — the same file
 * docker/entrypoint.sh copies into /etc/nginx/conf.d/ragflow.conf based on
 * API_PROXY_SCHEME. The scheme is read from the owned deployment env
 * (infra/rag-platform/.env.rag-platform) with the upstream docker/.env as
 * fallback, so proxy_mode always matches what would actually run.
 *
 * Usage:
 *   node scripts/rag-platform/route-inventory.mjs [--backend <path>]
 *     [--backend-ref <git-ref>|worktree] [--check]
 *
 *   --check   Do not write. Exit 1 if the committed outputs differ from a fresh
 *             scan (CI drift guard).
 */

import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, "..", "..");
const DEFAULT_BACKEND = "/Users/baran/Desktop/rag-backend";
const DEFAULT_BACKEND_REF = "v0.26.4";
const OUT_DIR = join(FRONTEND_ROOT, "docs", "rag-platform");

const args = process.argv.slice(2);
const checkOnly = args.includes("--check");
const backendArgIndex = args.indexOf("--backend");
const BACKEND_REPO_ROOT =
  backendArgIndex >= 0 && args[backendArgIndex + 1]
    ? args[backendArgIndex + 1]
    : process.env.RAG_PLATFORM_BACKEND_PATH || DEFAULT_BACKEND;
const backendRefIndex = args.indexOf("--backend-ref");
const BACKEND_REF =
  backendRefIndex >= 0 && args[backendRefIndex + 1]
    ? args[backendRefIndex + 1]
    : process.env.RAG_PLATFORM_BACKEND_REF || DEFAULT_BACKEND_REF;

if (!existsSync(BACKEND_REPO_ROOT)) {
  console.error(`backend path not found: ${BACKEND_REPO_ROOT}`);
  process.exit(2);
}

let BACKEND_ROOT = BACKEND_REPO_ROOT;
let BACKEND_COMMIT = "worktree";
let disposableBackendRoot;
if (BACKEND_REF !== "worktree") {
  try {
    BACKEND_COMMIT = execFileSync(
      "git",
      ["-C", BACKEND_REPO_ROOT, "rev-parse", `${BACKEND_REF}^{commit}`],
      { encoding: "utf8" },
    ).trim();
    disposableBackendRoot = mkdtempSync(join(tmpdir(), "rag-platform-route-inventory-"));
    const archive = execFileSync(
      "git",
      ["-C", BACKEND_REPO_ROOT, "archive", "--format=tar", BACKEND_REF],
      { maxBuffer: 512 * 1024 * 1024 },
    );
    execFileSync("tar", ["-xf", "-", "-C", disposableBackendRoot], {
      input: archive,
      maxBuffer: 512 * 1024 * 1024,
    });
    BACKEND_ROOT = disposableBackendRoot;
  } catch (error) {
    console.error(`cannot materialize backend ref ${BACKEND_REF}: ${error.message}`);
    process.exit(2);
  }
  process.on("exit", () => {
    rmSync(disposableBackendRoot, { recursive: true, force: true });
  });
}

// ---------------------------------------------------------------------------
// Service topology. Ports are the in-container listen ports, which are also the
// hybrid nginx upstream ports (docker/nginx/ragflow.conf.hybrid).
// ---------------------------------------------------------------------------

const SERVICES = {
  "python-api": { port: 9380, label: "Python API (Quart)" },
  "python-admin": { port: 9381, label: "Python admin server" },
  mcp: { port: 9382, label: "MCP server (starlette)" },
  "go-admin": { port: 9383, label: "Go admin server (gin)" },
  "go-api": { port: 9384, label: "Go API server (gin)" },
};

// ---------------------------------------------------------------------------
// Proxy scheme + nginx location map
// ---------------------------------------------------------------------------

function readEnvValue(file, key) {
  if (!existsSync(file)) return undefined;
  const text = readFileSync(file, "utf8");
  let found;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#")) continue;
    const match = trimmed.match(new RegExp(`^${key}\\s*=\\s*(.+)$`));
    // Upstream docker/.env keeps trailing `# ...` notes on the same line
    // (e.g. `API_PROXY_SCHEME=python # use pure python server deployment`).
    if (match) found = match[1].replace(/\s+#.*$/, "").trim().replace(/^["']|["']$/g, "");
  }
  return found;
}

function resolveProxyScheme() {
  const ownedEnv = join(FRONTEND_ROOT, "infra", "rag-platform", ".env.rag-platform");
  const upstreamEnv = join(BACKEND_ROOT, "docker", ".env");
  const owned = readEnvValue(ownedEnv, "API_PROXY_SCHEME");
  if (owned) return { scheme: owned, source: relative(FRONTEND_ROOT, ownedEnv) };
  const upstream = readEnvValue(upstreamEnv, "API_PROXY_SCHEME");
  if (upstream) return { scheme: upstream, source: `${BACKEND_ROOT}/docker/.env` };
  // entrypoint.sh falls back to the python config when the variable is unset.
  return { scheme: "python", source: "docker/entrypoint.sh default" };
}

const { scheme: PROXY_SCHEME, source: PROXY_SCHEME_SOURCE } = resolveProxyScheme();

const NGINX_FILE_BY_SCHEME = {
  hybrid: "ragflow.conf.hybrid",
  go: "ragflow.conf.golang",
  python: "ragflow.conf.python",
};

/**
 * Parse `location ~ <regex> { ... proxy_pass http://127.0.0.1:<port>; }` blocks
 * in nginx source order. nginx evaluates regex locations top to bottom and takes
 * the first match, so order is significant and is preserved here.
 */
function parseNginxLocations(nginxPath) {
  if (!existsSync(nginxPath)) return [];
  const text = readFileSync(nginxPath, "utf8");
  const owned = [];
  for (const line of text.split("\n")) {
    let marker = line.match(
      /^\s*#\s*rag-platform-(?:route|runtime-override)\s+(\w+)\s+(\S+)\s+(\d+)\s*$/,
    );
    if (marker) {
      owned.push({
        operator: "~",
        pattern: marker[2],
        port: Number(marker[3]),
        isRegex: true,
        method: marker[1],
      });
      continue;
    }
    marker = line.match(/^\s*#\s*rag-platform-default\s+(\S+)\s+(\d+)\s*$/);
    if (marker) {
      owned.push({
        operator: "~",
        pattern: marker[1],
        port: Number(marker[2]),
        isRegex: true,
        method: null,
      });
    }
  }
  if (owned.length > 0) return owned;
  const locations = [];
  const blockPattern = /location\s+(~\*?|=)?\s*([^\s{]+)\s*\{([^}]*)\}/g;
  let match;
  while ((match = blockPattern.exec(text)) !== null) {
    const [, operator, pattern, body] = match;
    const proxy = body.match(/proxy_pass\s+http:\/\/127\.0\.0\.1:(\d+)/);
    if (!proxy) continue;
    locations.push({
      operator: (operator || "prefix").trim(),
      pattern,
      port: Number(proxy[1]),
      isRegex: (operator || "").startsWith("~"),
    });
  }
  return locations;
}

const nginxFile = NGINX_FILE_BY_SCHEME[PROXY_SCHEME] ?? NGINX_FILE_BY_SCHEME.python;
const NGINX_PATH =
  PROXY_SCHEME === "hybrid"
    ? join(FRONTEND_ROOT, "infra", "rag-platform", "rag-platform.hybrid.conf")
    : join(BACKEND_ROOT, "docker", "nginx", nginxFile);
const NGINX_LOCATIONS = parseNginxLocations(NGINX_PATH);

/** Concrete probe path for a route template, so the nginx regexes can be matched. */
function probePath(path) {
  return path
    .replace(/<[^>]*>/g, "x") // Quart <id> / <int:id>
    .replace(/:[A-Za-z_][A-Za-z0-9_]*/g, "x") // gin :id
    .replace(/\*[A-Za-z_]*/g, "x"); // gin wildcard
}

/**
 * Resolve how (and whether) a route can be reached at runtime.
 *
 * Reachability is the conjunction of two independent conditions, and a route
 * needs both:
 *
 *   1. `proxy_reaches_service` — nginx's first matching location for this path
 *      forwards to the port that this route's own service listens on. A path
 *      claimed by an earlier location belonging to a *different* service is
 *      shadowed and never arrives.
 *   2. `service_started` — that service is actually running (see
 *      serviceStarted: scheme gating plus the missing Go binary).
 *
 * Keeping them separate matters because they fail for different reasons and
 * have different fixes: a shadowed route is fixed by changing the proxy scheme,
 * whereas a dead service is fixed only by deploying an image that contains its
 * executable. `runtime_disabled_reason` names which one applies.
 */
function resolveProxy(route) {
  // The MCP server is published on its own host port and is not part of the
  // nginx location map at all.
  if (route.service === "mcp") {
    return {
      proxy_mode: PROXY_SCHEME,
      proxy_destination: `direct:${SERVICES.mcp.port} (not proxied by nginx)`,
      proxy_reaches_service: null,
      runtime_enabled: null,
      runtime_disabled_reason: null,
      proxy_match: null,
    };
  }
  const started = serviceStarted(route.service);
  const probe = probePath(route.path);
  for (const location of NGINX_LOCATIONS) {
    if (location.method && location.method !== route.method) continue;
    let hit = false;
    if (location.isRegex) {
      try {
        hit = new RegExp(location.pattern).test(probe);
      } catch {
        hit = false;
      }
    } else {
      hit = probe.startsWith(location.pattern);
    }
    if (!hit) continue;
    const port = location.port;
    const reaches = port === SERVICES[route.service].port;
    let reason = null;
    if (!reaches) {
      // nginx sends this path somewhere else entirely. Report where it lands
      // and whether that destination is even alive, because a route shadowed
      // onto a dead service returns 502 rather than another service's answer.
      const owner = Object.entries(SERVICES).find(([, meta]) => meta.port === port)?.[0] ?? "unknown";
      const ownerStarted = owner === "unknown" ? null : serviceStarted(owner);
      reason =
        ownerStarted === false
          ? `proxy-shadowed onto ${owner}:${port}, which is not running (502)`
          : `proxy-shadowed onto ${owner}:${port} under scheme "${PROXY_SCHEME}"`;
      // Both halves of the conjunction can fail at once. Saying only where the
      // path lands would imply the route would work if nginx were re-pointed,
      // which is false when its own service is absent from the image too.
      if (started === false) {
        reason += `; ${route.service} is not running either`;
      }
    } else if (started === false) {
      reason = `${route.service} is not running under scheme "${PROXY_SCHEME}" (502)`;
    }
    return {
      proxy_mode: PROXY_SCHEME,
      proxy_destination: `127.0.0.1:${port}`,
      proxy_reaches_service: reaches,
      runtime_enabled: reaches && started !== false,
      runtime_disabled_reason: reason,
      proxy_match: `location ${location.operator} ${location.pattern}`,
    };
  }
  return {
    proxy_mode: PROXY_SCHEME,
    proxy_destination: null,
    proxy_reaches_service: false,
    runtime_enabled: false,
    runtime_disabled_reason: `no nginx location matches under scheme "${PROXY_SCHEME}"`,
    proxy_match: null,
  };
}

/**
 * Whether the deployed image can run the Go servers at all.
 *
 * The Go servers are `bin/ragflow_server`, which the published image does not
 * contain. Two lines of the upstream release prove it:
 *
 *   * `Dockerfile:268` — `COPY bin bin` is the only thing that populates
 *     /ragflow/bin, and no stage of the Dockerfile runs `go build` or
 *     `build.sh` (verified on both v0.26.4 and the checkout's main).
 *   * `.gitignore:237` — `bin/*` keeps every build product out of the tree, so
 *     the directory COPY sees holds nothing but the tracked `.gitkeep`.
 *
 * The owned `Dockerfile.backend-with-go` closes that packaging gap without
 * modifying the backend checkout: `build-backend-image.sh` verifies and archives
 * the exact v0.26.4 commit, builds the executable in the documented pure-Go
 * profile, and copies only that executable plus the runtime key wrapper over the
 * pinned upstream runtime image. Runtime smoke remains the final evidence.
 */
const GO_BINARY_IN_IMAGE = true;

/**
 * Which services are actually listening for the active scheme.
 *
 * Two independent gates: entrypoint.sh only launches a service when the scheme
 * selects it, and the image must contain the executable. Under `hybrid` the
 * entrypoint does try to start all four, but the Go pair dies on exec — so a
 * route proxied to 9383/9384 is unreachable no matter what nginx says.
 */
function serviceStarted(service) {
  if (service === "mcp") return null; // opt-in via --enable-mcpserver
  const isGo = service === "go-api" || service === "go-admin";
  if (isGo && !GO_BINARY_IN_IMAGE) return false;
  if (PROXY_SCHEME === "hybrid") return true;
  if (PROXY_SCHEME === "python") return service === "python-api" || service === "python-admin";
  if (PROXY_SCHEME === "go") return isGo;
  return null;
}

// ---------------------------------------------------------------------------
// Python route sources
// ---------------------------------------------------------------------------

const HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];

function normalizePath(prefix, path) {
  const joined = `${prefix.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
  return joined.replace(/\/{2,}/g, "/").replace(/(.+)\/$/, "$1");
}

function parseMethodsList(raw) {
  if (!raw) return ["GET"];
  const found = [...raw.matchAll(/"([A-Z]+)"|'([A-Z]+)'/g)]
    .map((m) => m[1] || m[2])
    .filter((m) => HTTP_METHODS.includes(m));
  return found.length ? found : ["GET"];
}

/**
 * Nearest auth decorator for a route, read from the lines between the route
 * decorator and the handler's `def`.
 */
function pythonAuthFor(lines, decoratorIndex) {
  for (let i = decoratorIndex; i < Math.min(decoratorIndex + 8, lines.length); i += 1) {
    const line = lines[i];
    if (/@login_required/.test(line)) {
      const types = line.match(/auth_types\s*=\s*\[([^\]]*)\]/);
      return types ? `login_required(${types[1].replace(/["'\s]/g, "")})` : "login_required";
    }
    if (/@token_required/.test(line)) return "token_required";
    if (/^(async\s+)?def\s/.test(line)) break;
  }
  return "public";
}

/**
 * Route paths built from an f-string, e.g.
 *   `@manager.route(f"{prefix}/commits", methods=["POST"])`
 * inside `_register_commit_routes(prefix, ...)`, called once per prefix at
 * module scope. A literal-only scan skips these silently, which is the worst
 * failure mode available to an inventory: the route is live and the matrix
 * never mentions it. Each interpolated name is resolved to the literal
 * arguments the module passes for it, and one route is emitted per prefix.
 *
 * Returns null when a placeholder cannot be resolved, so the caller can fail
 * loudly rather than emit a path containing a literal `{prefix}`.
 */
function resolveInterpolatedPaths(rawPath, lines) {
  const placeholders = [...rawPath.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
  if (placeholders.length === 0) return [rawPath];
  // Only single-placeholder prefixes occur upstream; more would need a product.
  if (placeholders.length > 1) return null;
  const [name] = placeholders;
  const source = lines.join("\n");
  // Find the enclosing registration helper and the literals passed to it.
  const helper = source.match(new RegExp(`def\\s+(\\w+)\\s*\\(\\s*${name}\\b`));
  if (!helper) return null;
  const calls = [...source.matchAll(new RegExp(`^${helper[1]}\\(\\s*["']([^"']+)["']`, "gm"))].map(
    (m) => m[1],
  );
  if (calls.length === 0) return null;
  return calls.map((prefix) => rawPath.replace(`{${name}}`, prefix));
}

function scanPythonRestfulApis() {
  const dir = join(BACKEND_ROOT, "api", "apps", "restful_apis");
  if (!existsSync(dir)) return [];
  const routes = [];
  for (const name of readdirSync(dir).sort()) {
    if (!name.endsWith(".py") || name.startsWith("_")) continue;
    const lines = readFileSync(join(dir, name), "utf8").split("\n");
    // api/apps/__init__.py:register_page -> url_prefix "/api/v1" for restful_apis.
    const prefix = "/api/v1";
    lines.forEach((line, index) => {
      const match = line.match(
        /@manager\.route\(\s*f?["']([^"']+)["']\s*(?:,\s*methods\s*=\s*(\[[^\]]*\]))?/,
      );
      if (!match) return;
      const [, rawPath, methodsRaw] = match;
      const auth = pythonAuthFor(lines, index + 1);
      const source = `api/apps/restful_apis/${name}:${index + 1}`;
      const resolved = resolveInterpolatedPaths(rawPath, lines);
      if (resolved === null) {
        throw new Error(
          `${source}: cannot resolve interpolated route path "${rawPath}". ` +
            "Extend resolveInterpolatedPaths() rather than letting the route go unrecorded.",
        );
      }
      for (const resolvedPath of resolved) {
        for (const method of parseMethodsList(methodsRaw)) {
          routes.push({
            method,
            path: normalizePath(prefix, resolvedPath),
            service: "python-api",
            auth,
            source,
            notes:
              resolved.length > 1
                ? `Registered for ${resolved.length} prefixes by a shared helper`
                : "",
          });
        }
      }
    });
  }
  return routes;
}

function scanPythonBackwardCompat() {
  const file = join(BACKEND_ROOT, "api", "apps", "backward_compat.py");
  if (!existsSync(file)) return [];
  const lines = readFileSync(file, "utf8").split("\n");
  const routes = [];
  lines.forEach((line, index) => {
    const match = line.match(
      /@(manager|legacy_v1_manager)\.route\(\s*["']([^"']+)["']\s*(?:,\s*methods\s*=\s*(\[[^\]]*\]))?/,
    );
    if (!match) return;
    const [, blueprint, rawPath, methodsRaw] = match;
    // register_backward_compat_routes(): manager -> /api/v1, legacy_v1_manager -> /v1
    const prefix = blueprint === "legacy_v1_manager" ? "/v1" : "/api/v1";
    for (const method of parseMethodsList(methodsRaw)) {
      routes.push({
        method,
        path: normalizePath(prefix, rawPath),
        service: "python-api",
        auth: pythonAuthFor(lines, index + 1),
        source: `api/apps/backward_compat.py:${index + 1}`,
        notes: "backward-compat shim retained by upstream for older clients",
      });
    }
  });
  return routes;
}

function scanPythonAdmin() {
  const file = join(BACKEND_ROOT, "admin", "server", "routes.py");
  if (!existsSync(file)) return [];
  const text = readFileSync(file, "utf8");
  const lines = text.split("\n");
  const prefixMatch = text.match(
    /Blueprint\(\s*["']admin["'][^)]*url_prefix\s*=\s*["']([^"']+)["']/,
  );
  const prefix = prefixMatch ? prefixMatch[1] : "/api/v1/admin";
  const routes = [];
  lines.forEach((line, index) => {
    const match = line.match(
      /@admin_bp\.route\(\s*["']([^"']+)["']\s*(?:,\s*methods\s*=\s*(\[[^\]]*\]))?/,
    );
    if (!match) return;
    const [, rawPath, methodsRaw] = match;
    const auth = /^\/?(ping|login)$/.test(rawPath) ? "public" : "admin-session";
    for (const method of parseMethodsList(methodsRaw)) {
      routes.push({
        method,
        path: normalizePath(prefix, rawPath),
        service: "python-admin",
        auth,
        source: `admin/server/routes.py:${index + 1}`,
        notes: "",
      });
    }
  });
  return routes;
}

// ---------------------------------------------------------------------------
// Go: gin routers. Group() assignments are tracked so prefixes compose.
// ---------------------------------------------------------------------------

function scanGinFile(absFile, relFile, service, options = {}) {
  if (!existsSync(absFile)) return [];
  const lines = readFileSync(absFile, "utf8").split("\n");
  const routes = [];
  // varName -> accumulated prefix. Seeded with the engine and, for helper
  // functions, with the prefix the caller passes in (options.seed).
  const groups = new Map([["engine", ""]]);
  for (const [name, prefix] of Object.entries(options.seed || {})) groups.set(name, prefix);
  const authByVar = options.authByVar || {};

  lines.forEach((line, index) => {
    const groupMatch = line.match(
      /(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*([A-Za-z_][A-Za-z0-9_]*)\.Group\(\s*["']([^"']*)["']/,
    );
    if (groupMatch) {
      const [, child, parent, suffix] = groupMatch;
      const parentPrefix = groups.has(parent) ? groups.get(parent) : "";
      groups.set(child, `${parentPrefix}${suffix}`);
      return;
    }
    const routeMatch = line.match(
      /(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)\.(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|Any)\(\s*["']([^"']*)["']/,
    );
    if (routeMatch) {
      const [, varName, verb, rawPath] = routeMatch;
      if (!groups.has(varName)) return;
      const prefix = groups.get(varName);
      const fullPath = normalizePath(prefix || "/", rawPath) || "/";
      const comment = line.includes("//") ? line.slice(line.indexOf("//") + 2).trim() : "";
      const methods = verb === "Any" ? HTTP_METHODS : [verb];
      for (const method of methods) {
        routes.push({
          method,
          path: fullPath,
          service,
          auth: authByVar[varName] || options.defaultAuth || "unknown",
          source: `${relFile}:${index + 1}`,
          notes: comment,
        });
      }
      return;
    }
    // registerAnyMethod(g, "/path", handler) — six verbs share one handler.
    const anyMethod = line.match(
      /registerAnyMethod\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*["']([^"']+)["']/,
    );
    if (anyMethod) {
      const [, varName, rawPath] = anyMethod;
      const prefix = groups.has(varName) ? groups.get(varName) : "";
      const fullPath = normalizePath(prefix || "/", rawPath);
      for (const method of ["POST", "GET", "PUT", "PATCH", "DELETE", "HEAD"]) {
        routes.push({
          method,
          path: fullPath,
          service,
          auth: authByVar[varName] || options.defaultAuth || "unknown",
          source: `${relFile}:${index + 1}`,
          notes: "registerAnyMethod: six verbs share one handler",
        });
      }
    }
  });
  return routes;
}

const GO_API_AUTH_BY_VAR = {
  engine: "public",
  apiNoAuth: "public",
  apiBetaAuth: "beta-token",
  authorized: "session",
  v1: "session",
};

function scanGoApi() {
  const routes = [];
  routes.push(
    ...scanGinFile(
      join(BACKEND_ROOT, "internal", "router", "router.go"),
      "internal/router/router.go",
      "go-api",
      { defaultAuth: "session", authByVar: GO_API_AUTH_BY_VAR },
    ),
  );
  routes.push(
    ...scanGinFile(
      join(BACKEND_ROOT, "internal", "router", "router_ee.go"),
      "internal/router/router_ee.go",
      "go-api",
      { seed: { apiNoAuth: "/api/v1" }, defaultAuth: "public" },
    ),
  );
  // Helper files receive an already-prefixed group from router.go:
  //   :596 RegisterAgentRoutes(v1.Group("/agents"), ...)
  //   :599 RegisterAgentCancelRoutes(v1.Group("/tasks"), ...)
  // Each helper takes its group as `g`, so both are scanned with the matching seed.
  routes.push(
    ...scanGinFile(
      join(BACKEND_ROOT, "internal", "router", "agent_routes.go"),
      "internal/router/agent_routes.go",
      "go-api",
      { seed: { g: "/api/v1/agents" }, defaultAuth: "session" },
    ).filter((route) => !route.path.endsWith("/cancel")),
  );
  routes.push(
    ...scanGinFile(
      join(BACKEND_ROOT, "internal", "router", "agent_routes.go"),
      "internal/router/agent_routes.go",
      "go-api",
      { seed: { g: "/api/v1/tasks" }, defaultAuth: "session" },
    ).filter((route) => route.path.endsWith("/cancel")),
  );
  return routes;
}

function scanGoAdmin() {
  return scanGinFile(
    join(BACKEND_ROOT, "internal", "admin", "router.go"),
    "internal/admin/router.go",
    "go-admin",
    {
      defaultAuth: "admin-session",
      authByVar: { engine: "public", admin: "public", protected: "admin-session" },
    },
  );
}

// ---------------------------------------------------------------------------
// MCP: starlette Route()/Mount() in mcp/server/server.py
// ---------------------------------------------------------------------------

function scanMcp() {
  const file = join(BACKEND_ROOT, "mcp", "server", "server.py");
  if (!existsSync(file)) return [];
  const lines = readFileSync(file, "utf8").split("\n");
  const routes = [];
  lines.forEach((line, index) => {
    const routeMatch = line.match(/Route\(\s*["']([^"']+)["'][^)]*methods\s*=\s*(\[[^\]]*\])/);
    if (routeMatch) {
      const [, path, methodsRaw] = routeMatch;
      for (const method of parseMethodsList(methodsRaw)) {
        routes.push({
          method,
          path,
          service: "mcp",
          auth: "mcp-api-key",
          source: `mcp/server/server.py:${index + 1}`,
          notes: "MCP transport endpoint; own port, opt-in via --enable-mcpserver",
        });
      }
      return;
    }
    const mountMatch = line.match(/Mount\(\s*["']([^"']+)["']/);
    if (mountMatch) {
      routes.push({
        method: "POST",
        path: mountMatch[1],
        service: "mcp",
        auth: "mcp-api-key",
        source: `mcp/server/server.py:${index + 1}`,
        notes: "starlette Mount (sub-application), opt-in via --enable-mcpserver",
      });
    }
  });
  return routes;
}

// ---------------------------------------------------------------------------
// Assemble
// ---------------------------------------------------------------------------

const rawRoutes = [
  ...scanPythonRestfulApis(),
  ...scanPythonBackwardCompat(),
  ...scanPythonAdmin(),
  ...scanGoApi(),
  ...scanGoAdmin(),
  ...scanMcp(),
];

/**
 * The deployed contract is pinned to v0.26.4, but the local backend worktree is
 * also normative for forward source discovery. Enterprise auth declarations
 * added after the pinned ref must not disappear merely because the runtime
 * image cannot contain them. Parse the current worktree's EE router, keep only
 * method+paths absent from the pinned scan, and record them as source-only
 * runtime-disabled. Their handler bodies are verified as not-implemented stubs
 * rather than inferred from route names.
 */
function scanForwardAuthSourceOnly() {
  if (BACKEND_REF === "worktree") return { commit: BACKEND_COMMIT, routes: [] };
  const routerPath = join(BACKEND_REPO_ROOT, "internal", "router", "router_ee.go");
  const handlerPath = join(BACKEND_REPO_ROOT, "internal", "handler", "user_auth_ee.go");
  if (!existsSync(routerPath) || !existsSync(handlerPath)) {
    return { commit: "unavailable", routes: [] };
  }
  const commit = execFileSync(
    "git",
    ["-C", BACKEND_REPO_ROOT, "rev-parse", "HEAD"],
    { encoding: "utf8" },
  ).trim();
  const known = new Set(rawRoutes.map((route) => `${route.method} ${route.path}`));
  const handlerLines = readFileSync(handlerPath, "utf8").split("\n");
  const routes = [];
  readFileSync(routerPath, "utf8")
    .split("\n")
    .forEach((line, index) => {
      const match = line.match(
        /apiNoAuth\.(GET|POST|PUT|PATCH|DELETE)\("([^"]+)",\s*r\.userHandler\.([A-Za-z0-9_]+)\)/,
      );
      if (!match) return;
      const method = match[1];
      const path = `/api/v1${match[2]}`;
      if (known.has(`${method} ${path}`)) return;
      const handler = match[3];
      const handlerStart = handlerLines.findIndex((candidate) =>
        new RegExp(`^func \\(h \\*UserHandler\\) ${handler}\\(`).test(candidate),
      );
      const handlerBody =
        handlerStart < 0 ? "" : handlerLines.slice(handlerStart, handlerStart + 5).join("\n");
      if (!handlerBody.includes("CodeNotImplemented")) {
        throw new Error(
          `forward auth route ${method} ${path} is not a verified CodeNotImplemented stub`,
        );
      }
      const base = {
        method,
        path,
        service: "go-api",
        auth: "public",
        source: `internal/router/router_ee.go:${index + 1}`,
        notes:
          `backend worktree-only at ${commit.slice(0, 12)}; ${handler} is ` +
          `CodeNotImplemented at internal/handler/user_auth_ee.go:${handlerStart + 1}`,
      };
      const proxy = resolveProxy(base);
      routes.push({
        ...base,
        service_port: SERVICES[base.service].port,
        service_started: serviceStarted(base.service),
        ...proxy,
        runtime_enabled: false,
        runtime_disabled_reason:
          `declared only in backend worktree ${commit.slice(0, 12)}; absent from deployed ` +
          `${BACKEND_REF} (${BACKEND_COMMIT.slice(0, 12)}); handler ${handler} returns CodeNotImplemented`,
        source_scope: "backend-worktree-only",
        source_commit: commit,
        alternates: [],
      });
    });
  return { commit, routes };
}

const forwardAuthSource = scanForwardAuthSourceOnly();

/**
 * Pipeline catalog handlers are implemented in the normative backend worktree,
 * but do not exist in the pinned v0.26.4 runtime source. Keep those routes in
 * the inventory as an explicit runtime gap instead of silently omitting them.
 */
function scanForwardPipelineSourceOnly() {
  if (BACKEND_REF === "worktree") return { commit: BACKEND_COMMIT, routes: [] };
  const routerPath = join(BACKEND_REPO_ROOT, "internal", "router", "router.go");
  const handlerPath = join(BACKEND_REPO_ROOT, "internal", "handler", "pipeline.go");
  if (!existsSync(routerPath) || !existsSync(handlerPath)) {
    return { commit: "unavailable", routes: [] };
  }
  const commit = execFileSync(
    "git",
    ["-C", BACKEND_REPO_ROOT, "rev-parse", "HEAD"],
    { encoding: "utf8" },
  ).trim();
  const known = new Set(rawRoutes.map((route) => `${route.method} ${route.path}`));
  const handlerText = readFileSync(handlerPath, "utf8");
  const routes = [];
  readFileSync(routerPath, "utf8")
    .split("\n")
    .forEach((line, index) => {
      const match = line.match(
        /apiNoAuth\.(GET)\("(\/pipelines(?:\/:id)?)",\s*r\.pipelineHandler\.(ListPipelines|GetPipeline)\)/,
      );
      if (!match) return;
      const method = match[1];
      const path = `/api/v1${match[2]}`;
      if (known.has(`${method} ${path}`)) return;
      const handler = match[3];
      if (!new RegExp(`func \\(h \\*PipelineHandler\\) ${handler}\\(`).test(handlerText)) {
        throw new Error(`forward pipeline handler ${handler} was not found`);
      }
      const base = {
        method,
        path,
        service: "go-api",
        auth: "public",
        source: `internal/router/router.go:${index + 1}`,
        notes:
          `backend worktree-only implemented pipeline catalog at ${commit.slice(0, 12)}; ` +
          `${handler} is implemented in internal/handler/pipeline.go`,
      };
      routes.push({
        ...base,
        service_port: SERVICES[base.service].port,
        service_started: serviceStarted(base.service),
        ...resolveProxy(base),
        runtime_enabled: false,
        runtime_disabled_reason:
          `implemented only in backend worktree ${commit.slice(0, 12)}; absent from deployed ` +
          `${BACKEND_REF} (${BACKEND_COMMIT.slice(0, 12)}); live hybrid proxy probe returns HTTP 404`,
        source_scope: "backend-worktree-only",
        source_commit: commit,
        alternates: [],
      });
    });
  return { commit, routes };
}

const forwardPipelineSource = scanForwardPipelineSourceOnly();
const forwardSourceRoutes = [
  ...forwardAuthSource.routes,
  ...forwardPipelineSource.routes,
];

/**
 * Collapse by method+path. Multiple services implementing the same method+path
 * are NOT overwritten: the implementation the active proxy actually reaches wins
 * the primary record, the others are listed in `alternates`.
 */
const byKey = new Map();
for (const route of rawRoutes) {
  const key = `${route.method} ${route.path}`;
  const entry = {
    ...route,
    service_port: SERVICES[route.service].port,
    service_started: serviceStarted(route.service),
    ...resolveProxy(route),
  };
  if (!byKey.has(key)) {
    byKey.set(key, { ...entry, alternates: [] });
    continue;
  }
  const existing = byKey.get(key);
  if (existing.service === entry.service && existing.source === entry.source) continue;
  if (!existing.runtime_enabled && entry.runtime_enabled) {
    const demoted = { ...existing };
    delete demoted.alternates;
    byKey.set(key, { ...entry, alternates: [...existing.alternates, demoted] });
  } else {
    existing.alternates.push({ ...entry });
  }
}

const routes = [...byKey.values(), ...forwardSourceRoutes].sort((a, b) =>
  a.path === b.path ? a.method.localeCompare(b.method) : a.path.localeCompare(b.path),
);

/**
 * Strip parameter syntax so the same route expressed in two frameworks compares
 * equal: gin writes `:canvas_id` and `*filepath`, Quart writes `<canvas_id>`.
 * Without this every Go route looks Go-exclusive.
 */
function canonicalPath(path) {
  const canon = path
    .replace(/<[^>]+>:<[^>]+>/g, "{p}")
    .replace(/<[^>]+>/g, "{p}")
    .replace(/:[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\*[A-Za-z_][A-Za-z0-9_]*/g, "{p}")
    .replace(/\/+$/, "");
  return canon || "/";
}

/**
 * A source-only static route can still be served by a reachable parameterised
 * route. Gin's `/oauth/:channel/callback`, for example, handles the concrete
 * `/oauth/github/callback` request even though the EE worktree also declares a
 * static stub for it. This check models request reachability, not just textual
 * route-shape equality.
 */
function parameterRouteServesConcrete(pattern, concrete) {
  if (/[<:*]/.test(concrete)) return false;
  const patternSegments = pattern.split("/").filter(Boolean);
  const concreteSegments = concrete.split("/").filter(Boolean);
  for (let index = 0; index < patternSegments.length; index += 1) {
    const segment = patternSegments[index];
    if (segment.startsWith("*")) return index < concreteSegments.length;
    if (index >= concreteSegments.length) return false;
    if (segment.startsWith(":") || /^<[^>]+>$/.test(segment)) continue;
    if (segment !== concreteSegments[index]) return false;
  }
  return patternSegments.length === concreteSegments.length;
}

// A runtime-disabled route matters much more when nothing else serves its
// method+path. Record which of the two cases each one is, so the report can
// separate "lost capability" from "same capability, different implementation".
for (const route of routes) {
  if (route.runtime_enabled !== false) continue;
  const equivalent = routes.find(
    (other) =>
      other.runtime_enabled === true &&
      other.method === route.method &&
      (canonicalPath(other.path) === canonicalPath(route.path) ||
        parameterRouteServesConcrete(other.path, route.path)),
  );
  route.equivalent_reachable_route = equivalent
    ? { method: equivalent.method, path: equivalent.path, service: equivalent.service }
    : null;
}

const disabledRoutes = routes.filter((route) => route.runtime_enabled === false);
const goExclusiveRoutes = disabledRoutes.filter((route) => !route.equivalent_reachable_route);

const byService = {};
for (const route of routes) {
  byService[route.service] = (byService[route.service] || 0) + 1;
}

const inventory = {
  generated_by: "scripts/rag-platform/route-inventory.mjs",
  backend_path: BACKEND_REPO_ROOT,
  backend: {
    source_ref: BACKEND_REF,
    source_commit: BACKEND_COMMIT,
    source_image: readEnvValue(join(BACKEND_ROOT, "docker", ".env"), "RAGFLOW_IMAGE") || "unknown",
    api_version: "v1",
    forward_source_commit: forwardAuthSource.commit,
  },
  proxy: {
    scheme: PROXY_SCHEME,
    scheme_source: PROXY_SCHEME_SOURCE,
    nginx_config:
      PROXY_SCHEME === "hybrid"
        ? "infra/rag-platform/rag-platform.hybrid.conf"
        : `docker/nginx/${nginxFile}`,
    locations: NGINX_LOCATIONS.map((l) => ({
      match: `${l.method ? `${l.method} ` : ""}${l.operator} ${l.pattern}`,
      upstream: `127.0.0.1:${l.port}`,
    })),
  },
  services: Object.fromEntries(
    Object.entries(SERVICES).map(([name, meta]) => [
      name,
      { ...meta, started_in_active_scheme: serviceStarted(name) },
    ]),
  ),
  totals: {
    routes: routes.length,
    by_service: byService,
    runtime_enabled: routes.filter((r) => r.runtime_enabled === true).length,
    runtime_disabled: disabledRoutes.length,
    source_only_runtime_disabled: forwardSourceRoutes.length,
    not_proxied: routes.filter((r) => r.runtime_enabled === null).length,
    with_alternates: routes.filter((r) => r.alternates.length > 0).length,
    runtime_disabled_breakdown: {
      with_reachable_equivalent: disabledRoutes.length - goExclusiveRoutes.length,
      no_reachable_equivalent: goExclusiveRoutes.length,
    },
  },
  routes,
};

function renderMarkdown(data) {
  const lines = [];
  lines.push("# Rag Platform — backend route inventory");
  lines.push("");
  lines.push("<!-- GENERATED FILE. Do not edit by hand.");
  lines.push("     Regenerate: node scripts/rag-platform/route-inventory.mjs -->");
  lines.push("");
  lines.push(
    `- Backend source: \`${data.backend_path}\` at \`${data.backend.source_ref}\` ` +
      `(\`${data.backend.source_commit}\`)`,
  );
  lines.push(
    `- Source image: \`${data.backend.source_image}\` (API version \`${data.backend.api_version}\`)`,
  );
  lines.push(
    `- Forward source audit: backend worktree \`${data.backend.forward_source_commit}\`; ` +
      `${data.totals.source_only_runtime_disabled} source-only runtime-disabled route(s)`,
  );
  lines.push(`- Active proxy scheme: \`${data.proxy.scheme}\` (from ${data.proxy.scheme_source})`);
  lines.push(`- Proxy config: \`${data.proxy.nginx_config}\``);
  lines.push("");
  lines.push("## Totals");
  lines.push("");
  lines.push("| Metric | Count |");
  lines.push("| --- | --- |");
  lines.push(`| routes | ${data.totals.routes} |`);
  for (const [service, count] of Object.entries(data.totals.by_service)) {
    lines.push(`| ${service} (port ${data.services[service].port}) | ${count} |`);
  }
  lines.push(`| runtime-enabled | ${data.totals.runtime_enabled} |`);
  lines.push(`| runtime-disabled | ${data.totals.runtime_disabled} |`);
  lines.push(`| — source-only forward declarations | ${data.totals.source_only_runtime_disabled} |`);
  lines.push(`| not proxied by nginx | ${data.totals.not_proxied} |`);
  lines.push(`| method+path with alternate implementations | ${data.totals.with_alternates} |`);
  lines.push("");
  lines.push("## Proxy location map (nginx evaluation order)");
  lines.push("");
  lines.push("| Order | Location | Upstream |");
  lines.push("| --- | --- | --- |");
  data.proxy.locations.forEach((location, index) => {
    lines.push(`| ${index + 1} | \`${location.match}\` | ${location.upstream} |`);
  });
  lines.push("");
  lines.push("## Routes");
  lines.push("");
  lines.push(
    "| Method | Path | Service | Port | Proxy mode | Proxy destination | Auth / role | Runtime | Source | Alternates | Notes |",
  );
  lines.push("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
  for (const route of data.routes) {
    const runtime =
      route.runtime_enabled === true
        ? "enabled"
        : route.runtime_enabled === false
          ? "**runtime-disabled**"
          : "not-proxied";
    const alternates = route.alternates.length
      ? route.alternates.map((a) => `${a.service}@${a.service_port} (\`${a.source}\`)`).join("<br>")
      : "—";
    lines.push(
      `| ${route.method} | \`${route.path}\` | ${route.service} | ${route.service_port} | ${route.proxy_mode} | ${
        route.proxy_destination ?? "—"
      } | ${route.auth} | ${runtime} | \`${route.source}\` | ${alternates} | ${route.notes || "—"} |`,
    );
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

/**
 * The plan forbids silently skipping a route that is closed at runtime: each one
 * has to be recorded with source, proxy and smoke-test evidence. That record is
 * generated from the same inventory rather than maintained by hand, so it can
 * never disagree with it.
 */
function renderRuntimeDisabled(data) {
  const disabled = data.routes.filter((route) => route.runtime_enabled === false);
  const exclusive = disabled.filter((route) => !route.equivalent_reachable_route);
  const shadowed = disabled.filter((route) => route.equivalent_reachable_route);

  const areaOf = (path) => {
    const segments = path.split("/").filter(Boolean);
    const versionIndex = segments.indexOf("v1");
    return (versionIndex >= 0 ? segments[versionIndex + 1] : segments[0]) || "(root)";
  };
  const groupByArea = (list) => {
    const groups = new Map();
    for (const route of list) {
      const area = areaOf(route.path);
      if (!groups.has(area)) groups.set(area, []);
      groups.get(area).push(route);
    }
    return [...groups.entries()].sort(
      (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
    );
  };

  const lines = [];
  lines.push("# Rag Platform — runtime-disabled backend routes");
  lines.push("");
  lines.push("<!-- GENERATED FILE. Do not edit by hand.");
  lines.push("     Regenerate: node scripts/rag-platform/route-inventory.mjs -->");
  lines.push("");
  lines.push(
    "Every backend route that the deployed stack cannot serve, with the reason it",
    "cannot. Nothing here is skipped silently: a route is listed if either nginx",
    "does not forward it to the service that implements it, or that service is not",
    "running in the active scheme.",
  );
  lines.push("");
  lines.push(`- Active proxy scheme: \`${data.proxy.scheme}\` (from ${data.proxy.scheme_source})`);
  lines.push(`- Proxy config: \`${data.proxy.nginx_config}\``);
  lines.push(`- Source image: \`${data.backend.source_image}\``);
  lines.push("- Decision record: `docs/adr/0005-backend-proxy-scheme.md`");
  lines.push("");
  lines.push("## Totals");
  lines.push("");
  lines.push("| Metric | Count |");
  lines.push("| --- | --- |");
  lines.push(`| routes discovered | ${data.totals.routes} |`);
  lines.push(`| reachable | ${data.totals.runtime_enabled} |`);
  lines.push(`| runtime-disabled | ${disabled.length} |`);
  lines.push(`| — no reachable equivalent (capability lost) | ${exclusive.length} |`);
  lines.push(`| — same concrete request served elsewhere (no capability lost) | ${shadowed.length} |`);
  lines.push(`| not proxied by nginx | ${data.totals.not_proxied} |`);
  lines.push("");
  lines.push("## Why these routes are closed");
  lines.push("");
  lines.push(
    "The owned method-aware hybrid map selects one implementation for each",
    "method+path. When both Python and Go register the same contract, the Go",
    "implementation is selected and the duplicate Python registration appears",
    "below as runtime-disabled with a reachable equivalent. This is intentional",
    "deduplication, not a lost capability. The Go executable provenance and the",
    "four direct service smoke probes are recorded in ADR 0005 and the Faz 0",
    "result report.",
  );
  if (data.totals.source_only_runtime_disabled > 0) {
    lines.push("");
    lines.push(
      `${data.totals.source_only_runtime_disabled} route(s) are separate forward-source cases: ` +
        `they are declared only at backend worktree \`${data.backend.forward_source_commit}\`, ` +
        `and are absent from deployed \`${data.backend.source_ref}\`. Nine auth handlers return ` +
        "`CodeNotImplemented`; the two pipeline catalog handlers are implemented but absent from the pinned runtime. " +
        "Live hybrid smoke returns HTTP 404 for the pipeline list/detail and seven auth paths; " +
        "GitHub and Lark callback URLs return 302 through the active parameterised callback. " +
        "The auth UI uses live channels without a false captcha/OTP step, while the pipeline selector shows an explicit runtime-disabled reason.",
    );
  }
  lines.push("");
  lines.push("## Capability lost — no reachable route serves this method and path");
  lines.push("");
  lines.push(
    "Compared after canonicalising parameter syntax (`<id>`, `:id`, `*path` all",
    "normalise to `{p}`), so a Go route is only listed here when no Python route",
    "provides the same method and path shape.",
  );
  lines.push("");
  for (const [area, group] of groupByArea(exclusive)) {
    lines.push(`### \`${area}\` (${group.length})`);
    lines.push("");
    lines.push("| Method | Path | Service | Source | Proxy result |");
    lines.push("| --- | --- | --- | --- | --- |");
    for (const route of group) {
      lines.push(
        `| ${route.method} | \`${route.path}\` | ${route.service}@${route.service_port} | ` +
          `\`${route.source}\` | ${route.runtime_disabled_reason} |`,
      );
    }
    lines.push("");
  }
  lines.push("## No capability lost — same method and path is served by a reachable route");
  lines.push("");
  lines.push(
    "Duplicate implementations of one contract, or a static source-only route",
    "whose concrete request is handled by a reachable parameterised route. The",
    "serving implementation shown below keeps the surface available.",
  );
  lines.push("");
  lines.push("| Method | Path | Unreachable | Serving instead | Source |");
  lines.push("| --- | --- | --- | --- | --- |");
  for (const route of shadowed) {
    const equivalent = route.equivalent_reachable_route;
    lines.push(
      `| ${route.method} | \`${route.path}\` | ${route.service}@${route.service_port} | ` +
        `${equivalent.service} (\`${equivalent.path}\`) | \`${route.source}\` |`,
    );
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

const jsonPath = join(OUT_DIR, "route-inventory.json");
const mdPath = join(OUT_DIR, "route-inventory.md");
const disabledPath = join(OUT_DIR, "runtime-disabled.md");
const jsonText = `${JSON.stringify(inventory, null, 2)}\n`;
const mdText = renderMarkdown(inventory);
const disabledText = renderRuntimeDisabled(inventory);

if (checkOnly) {
  let drift = false;
  for (const [path, expected] of [
    [jsonPath, jsonText],
    [mdPath, mdText],
    [disabledPath, disabledText],
  ]) {
    if (!existsSync(path)) {
      console.error(`missing generated file: ${relative(FRONTEND_ROOT, path)}`);
      drift = true;
      continue;
    }
    if (readFileSync(path, "utf8") !== expected) {
      console.error(`stale generated file: ${relative(FRONTEND_ROOT, path)}`);
      drift = true;
    }
  }
  if (drift) {
    console.error(
      "route inventory is out of date — run: node scripts/rag-platform/route-inventory.mjs",
    );
    process.exit(1);
  }
  console.log(`route inventory up to date (${inventory.totals.routes} routes)`);
  process.exit(0);
}

if (!existsSync(OUT_DIR) || !statSync(OUT_DIR).isDirectory()) {
  console.error(`output dir missing: ${OUT_DIR}`);
  process.exit(2);
}

writeFileSync(jsonPath, jsonText);
writeFileSync(mdPath, mdText);
writeFileSync(disabledPath, disabledText);
console.log(
  `wrote ${relative(FRONTEND_ROOT, jsonPath)}, ${relative(FRONTEND_ROOT, mdPath)} and ` +
    `${relative(FRONTEND_ROOT, disabledPath)}: ${inventory.totals.routes} routes, ` +
    `scheme=${PROXY_SCHEME}, runtime-disabled=${inventory.totals.runtime_disabled} ` +
    `(${inventory.totals.runtime_disabled_breakdown.no_reachable_equivalent} with no reachable equivalent)`,
);
