// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";
import * as transition from "../src/lib/account-transition.ts";
import * as deadline from "../src/features/auth/bootstrap-deadline.ts";
import type * as LoginClient from "../src/features/auth/login-client.ts";
import type * as AccountSession from "../src/features/auth/account-session.ts";

const formUrl = new URL(
  "../src/features/auth/components/auth-form.tsx",
  import.meta.url,
);
const clientUrl = new URL(
  "../src/features/auth/login-client.ts",
  import.meta.url,
);
const accountUrl = new URL(
  "../src/features/auth/account-session.ts",
  import.meta.url,
);
const base = { apiUrl: (path: string) => path };
const token = {
  access_token: "access",
  refresh_token: "refresh",
  must_change_password: false,
};

function elements(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as StubElement;
  return [element, ...elements(element.props.children)];
}
function textContent(node: unknown): string {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(textContent).join(" ");
  if (node && typeof node === "object" && "props" in node)
    return textContent((node as StubElement).props.children);
  return "";
}
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

function mountForm(
  client: typeof LoginClient,
  session: { access: string | null; change: boolean },
  switched = false,
  failTransitionOnce = false,
) {
  const state: unknown[] = [];
  let cursor = 0;
  const effects: (() => void)[] = [];
  const routes: string[] = [];
  const transitions: string[] = [];
  const api = loadWithStubs<{
    AuthForm: (props: { mode: string }) => StubElement | null;
  }>(formUrl, {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {
      useState: (initial: unknown) => {
        const index = cursor++;
        if (!(index in state))
          state[index] = typeof initial === "function" ? initial() : initial;
        return [
          state[index],
          (value: unknown) => {
            state[index] =
              typeof value === "function" ? value(state[index]) : value;
          },
        ];
      },
      useRef: (current: unknown) => {
        const index = cursor++;
        if (!(index in state)) state[index] = { current };
        return state[index];
      },
      useEffect: (effect: () => void) => {
        effects.push(effect);
      },
    },
    "@/lib/api-base": base,
    "@/lib/account-transition": {
      normalizeAccountUsername: transition.normalizeAccountUsername,
      transitionBrowserAccount: async (
        username: string,
        route: string,
        commit: () => void,
      ) => {
        transitions.push(`${username}:${route}`);
        if (failTransitionOnce && transitions.length === 1)
          throw new Error("Close other Unsloth tabs and retry");
        commit();
        return switched;
      },
    },
    "../account-session": {
      useLoginMode: client.getLoginMode,
      sessionAccount: () => (session.access ? { username: "alice" } : null),
    },
    "../login-client": client,
    "@/components/ui/button": { Button: "Button" },
    "@/components/mascot-img": { MascotImg: "MascotImg" },
    "@/components/ui/input": { Input: "Input" },
    "@/components/ui/label": { Label: "Label" },
    "@tanstack/react-router": {
      Link: "Link",
      useNavigate:
        () =>
        ({ to }: { to: string }) => {
          routes.push(to);
        },
    },
    "lucide-react": { Eye: "Eye", EyeOff: "EyeOff" },
    "../api": { refreshSession: async () => true },
    "../bootstrap-deadline": deadline,
    "../session": {
      clearAuthTokens: () => {
        session.access = null;
      },
      getAuthToken: () => session.access,
      hasAuthToken: () => Boolean(session.access),
      hasRefreshToken: () => false,
      getPostAuthRoute: () => (session.change ? "/change-password" : "/chat"),
      mustChangePassword: () => session.change,
      setMustChangePassword: (change: boolean) => {
        session.change = change;
      },
      storeAuthTokens: (access: string) => {
        session.access = access;
      },
    },
  });
  const render = (mode = "login") => {
    cursor = 0;
    return api.AuthForm({ mode });
  };
  return {
    render,
    routes,
    transitions,
    initialize: async (mode = "login") => {
      render(mode);
      for (const effect of effects.splice(0)) effect();
      await tick();
      return render(mode);
    },
  };
}

function client() {
  return loadWithStubs<typeof LoginClient>(clientUrl, {
    "@/lib/api-base": base,
    "@/lib/account-transition": transition,
  });
}
function environment(t: TestContext) {
  const previousFetch = globalThis.fetch;
  const previousWindow = globalThis.window;
  const values = new Map<string, string>();
  globalThis.window = {
    dispatchEvent: () => true,
    localStorage: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => {
        values.set(key, value);
      },
    },
  } as unknown as Window & typeof globalThis;
  t.after(() => {
    globalThis.fetch = previousFetch;
    globalThis.window = previousWindow;
  });
}

for (const mode of ["single", "multi"] as const) {
  test(`${mode} form renders the correct username fields and setup hint`, async (t) => {
    environment(t);
    globalThis.fetch = async () =>
      Response.json({
        initialized: true,
        requires_password_change: false,
        login_mode: mode,
      });
    const form = mountForm(client(), { access: null, change: false });
    const tree = await form.initialize();
    const username = elements(tree).find(
      (element) => element.props.id === "username",
    );
    assert.equal(Boolean(username), mode === "multi");
    if (username) assert.equal(username.props.autoComplete, "username");
    const password = elements(tree).find(
      (element) => element.props.id === "password",
    );
    assert.equal(password?.props.autoComplete, "current-password");
    assert.equal(
      textContent(tree).includes("setup code your administrator"),
      mode === "multi",
    );
    assert.deepEqual(form.routes, []);
  });
}

test("single mode still posts unsloth and does not refetch on success", async (t) => {
  environment(t);
  const requests: unknown[] = [];
  globalThis.fetch = async (path, init) => {
    requests.push([path, JSON.parse(String(init?.body))]);
    return Response.json(token);
  };
  assert.deepEqual(
    await client().loginFromForm("single", "ignored", "owner-password"),
    token,
  );
  assert.deepEqual(requests, [
    ["/api/auth/login", { username: "unsloth", password: "owner-password" }],
  ]);
});

test("multi mode case folds and trims username without changing password", async (t) => {
  environment(t);
  let sent: unknown;
  globalThis.fetch = async (_path, init) => {
    sent = JSON.parse(String(init?.body));
    return Response.json(token);
  };
  await client().loginFromForm("multi", " ALIce ", "Keep-CaSe ");
  assert.deepEqual(sent, { username: "alice", password: "Keep-CaSe " });
});

test("401 after a second account is created rerenders username entry without showing the error", async (t) => {
  environment(t);
  let statusReads = 0;
  globalThis.fetch = async (path) =>
    String(path).endsWith("/status")
      ? Response.json({
          initialized: true,
          requires_password_change: false,
          login_mode: ++statusReads === 1 ? "single" : "multi",
        })
      : Response.json({ detail: "bad password" }, { status: 401 });
  const form = mountForm(client(), { access: null, change: false });
  let tree = await form.initialize();
  const submit = elements(tree).find((element) => element.type === "form")
    ?.props.onSubmit as (event: unknown) => Promise<void>;
  await submit({ preventDefault() {} });
  tree = form.render();
  assert.ok(elements(tree).find((element) => element.props.id === "username"));
  assert.ok(!textContent(tree).includes("bad password"));
  assert.equal(statusReads, 2);
});

test("other login failures preserve the server error and only single-mode 401 probes status", async (t) => {
  environment(t);
  for (const [mode, status, reads] of [
    ["multi", 401, 1],
    ["single", 429, 1],
    ["single", 401, 2],
  ] as const) {
    let count = 0;
    globalThis.fetch = async (path) => {
      count++;
      return String(path).endsWith("/status")
        ? Response.json({ login_mode: "single" })
        : Response.json({ detail: "Server detail" }, { status });
    };
    await assert.rejects(
      client().loginFromForm(mode, "alice", "password"),
      /Server detail/,
    );
    assert.equal(count, reads);
  }
});

test("managed setup follows the token requirement despite public owner status", async (t) => {
  environment(t);
  globalThis.fetch = async () =>
    Response.json({
      initialized: true,
      requires_password_change: false,
      login_mode: "multi",
    });
  const session = { access: "setup-session", change: true };
  const form = mountForm(client(), session);
  const tree = await form.initialize("change-password");
  assert.ok(
    elements(tree).find((element) => element.props.id === "new-password"),
  );
  assert.equal(session.change, true);
  assert.deepEqual(form.routes, []);
});

test("owner bootstrap still redirects single-mode login to password setup", async (t) => {
  environment(t);
  globalThis.fetch = async () =>
    Response.json({
      initialized: true,
      requires_password_change: true,
      login_mode: "single",
    });
  const form = mountForm(client(), { access: null, change: false });
  await form.initialize();
  assert.deepEqual(form.routes, ["/change-password"]);
});

test("successful login commits through transition and avoids SPA navigation when replacing", async (t) => {
  environment(t);
  globalThis.fetch = async (path) =>
    String(path).endsWith("/status")
      ? Response.json({
          initialized: true,
          requires_password_change: false,
          login_mode: "multi",
        })
      : Response.json({ ...token, must_change_password: true });
  const session = { access: null as string | null, change: false };
  const form = mountForm(client(), session, true);
  let tree = await form.initialize();
  const change = elements(tree).find(
    (element) => element.props.id === "username",
  )?.props.onChange as (event: unknown) => void;
  change({ target: { value: "alice" } });
  tree = form.render();
  const submit = elements(tree).find((element) => element.type === "form")
    ?.props.onSubmit as (event: unknown) => Promise<void>;
  await submit({ preventDefault() {} });
  assert.deepEqual(form.transitions, ["alice:/change-password"]);
  assert.equal(session.change, true);
  assert.equal(session.access, "access");
  assert.deepEqual(form.routes, []);
});

test("owner visibility uses the authenticated subject or role and fails closed", () => {
  const session = loadWithStubs<typeof AccountSession>(accountUrl, {
    react: {},
    "./login-client": {},
    "./session": {},
  });
  const jwt = (payload: unknown) =>
    `e30.${Buffer.from(JSON.stringify(payload)).toString("base64url")}.sig`;
  assert.equal(session.sessionAccount(jwt({ sub: "unsloth" }))?.isOwner, true);
  assert.equal(
    session.sessionAccount(jwt({ sub: "alice", role: "user" }))?.isOwner,
    false,
  );
  assert.equal(
    session.sessionAccount(jwt({ sub: "unsloth", role: "user" }))?.isOwner,
    false,
  );
  assert.equal(session.sessionAccount("malformed"), null);
  assert.equal(session.sessionAccount(null), null);
});

test("cleanup retries reuse an issued setup session without consuming the code again", async (t) => {
  environment(t);
  let logins = 0;
  globalThis.fetch = async (path) => {
    if (String(path).endsWith("/status"))
      return Response.json({
        initialized: true,
        requires_password_change: false,
        login_mode: "multi",
      });
    logins++;
    return Response.json({ ...token, must_change_password: true });
  };
  const session = { access: null as string | null, change: false };
  const form = mountForm(client(), session, true, true);
  let tree = await form.initialize();
  const username = elements(tree).find(
    (element) => element.props.id === "username",
  );
  (username?.props.onChange as (event: unknown) => void)({
    target: { value: "alice" },
  });
  for (let attempt = 0; attempt < 2; attempt++) {
    tree = form.render();
    const submit = elements(tree).find((element) => element.type === "form")
      ?.props.onSubmit as (event: unknown) => Promise<void>;
    await submit({ preventDefault() {} });
    if (attempt === 0) {
      assert.equal(session.access, null);
      assert.match(textContent(form.render()), /Close other Unsloth tabs/);
    }
  }
  assert.equal(logins, 1);
  assert.equal(session.access, "access");
  assert.equal(form.transitions.length, 2);
});

test("an owner-only browser never adds a startup status probe or changes stored preferences", async (t) => {
  environment(t);
  globalThis.fetch = async () => {
    assert.fail("single-user startup must not probe status");
  };
  const api = client();
  assert.equal(api.getLoginMode(), "single");
  api.ensureLoginMode();
  await tick();
  assert.equal(window.localStorage.getItem(api.LOGIN_MODE_HINT_KEY), null);
});

test("multi-user policy survives document reloads and revalidates only multi-user hints", async (t) => {
  environment(t);
  const first = client();
  first.setLoginMode("multi");
  const reloaded = client();
  assert.equal(reloaded.getLoginMode(), "multi");
  let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    return Response.json({
      initialized: true,
      requires_password_change: false,
      login_mode: "multi",
    });
  };
  reloaded.ensureLoginMode();
  await tick();
  assert.equal(requests, 1);
});

test("creating an account hides full access in peer tabs without a status request", (t) => {
  environment(t);
  globalThis.fetch = async () => {
    assert.fail("storage synchronization must not fetch status");
  };
  const handlers = new Set<(event: Partial<StorageEvent>) => void>();
  window.addEventListener = ((
    _name: string,
    handler: (event: Partial<StorageEvent>) => void,
  ) => {
    handlers.add(handler);
  }) as typeof window.addEventListener;
  window.removeEventListener = ((
    _name: string,
    handler: (event: Partial<StorageEvent>) => void,
  ) => {
    handlers.delete(handler);
  }) as typeof window.removeEventListener;
  const api = client();
  let notified = 0;
  const unsubscribe = api.subscribeLoginMode(() => {
    notified++;
  });
  for (const handler of handlers)
    handler({
      key: api.LOGIN_MODE_HINT_KEY,
      oldValue: null,
      newValue: "multi",
    });
  assert.equal(api.getLoginMode(), "multi");
  assert.equal(notified, 1);
  for (const handler of handlers)
    handler({
      key: api.LOGIN_MODE_HINT_KEY,
      oldValue: "multi",
      newValue: null,
    });
  assert.equal(api.getLoginMode(), "multi");
  unsubscribe();
  assert.equal(handlers.size, 0);
});
