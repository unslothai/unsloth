# Local HTTPS reverse proxies

Studio's packaged desktop frontend can be served through an operator-managed
HTTPS reverse proxy, such as Tailscale Serve. This is disabled by default.
The proxy must run on the same machine and connect to Studio over loopback.
Studio login and API authentication are still required; proxy headers never
authenticate a user.

## Configure Studio

Set `UNSLOTH_STUDIO_PROXY_ORIGIN` in the backend's environment before starting
Studio. Use the single HTTPS origin visible in the browser, with no trailing
slash, path, query, fragment, wildcard or credentials:

```sh
export UNSLOTH_STUDIO_PROXY_ORIGIN=https://studio.example.ts.net
# Start Studio normally from this shell.
```

A non-default HTTPS port is allowed, for example `https://studio.example:8443`.
Malformed values prevent frontend mounting and do not restore local trust
shortcuts. An unset or empty value leaves the default frontend policy in place.

For macOS Desktop, quit Unsloth, then set the environment for applications
launched in your current login session and reopen the app normally:

```sh
launchctl setenv UNSLOTH_STUDIO_PROXY_ORIGIN https://studio.example.ts.net
open -a Unsloth
```

Use LaunchServices (`open -a`) or Finder to launch the desktop app so macOS
attributes external-volume access to the app. This `launchctl` setting is for
the current login session; it is not persistent installation configuration.
To remove it, quit the app, run
`launchctl unsetenv UNSLOTH_STUDIO_PROXY_ORIGIN`, then reopen it.

## Configure the proxy

The proxy must terminate HTTPS and forward requests to the backend's loopback
address and actual port. It must preserve the external `Host` and overwrite
the following headers:

```text
Host: studio.example.ts.net
X-Forwarded-Host: studio.example.ts.net
X-Forwarded-Proto: https
```

Both authorities must match the configured origin, including any non-default
port. Duplicate headers, comma-separated values and a mismatched browser
`Origin` are rejected for frontend requests. A missing `Origin` is allowed,
as it is normal for top-level navigation. These checks do not expand CORS.

One optional DNS root dot is accepted in `Host` and `X-Forwarded-Host`, so a
proxy forwarding `studio.example.ts.net.` still reaches the configured host.
For the configured URL and the browser's `Origin`, the dotted and undotted
spellings remain distinct: configure the exact spelling used in the browser.
Empty labels, repeated trailing dots and labels starting or ending with a
hyphen are rejected.

For Tailscale Serve, replace the example origin with your device's HTTPS name
and replace `8888` below if Studio uses a different port:

```sh
tailscale serve --bg --https=443 http://127.0.0.1:8888
tailscale serve status
```

Serve controls which tailnet devices may reach the proxy using your Tailscale
access rules. Do not use Funnel if you intend tailnet-only access. Other proxies
must enforce their own network access policy. Studio does not manage the proxy,
its TLS certificates or its access rules.

## Authentication and local trust

While the setting is nonempty, Studio treats its loopback backend as reachable
through a connector. This disables keyless loopback API access and automatic
loopback-only stdio MCP enablement. Existing explicit stdio MCP configuration
continues to follow its normal policy.

Bootstrap credentials are never embedded in served HTML while proxy mode is
configured, including requests that appear to come directly from localhost.
This also covers a proxy accidentally stripping all forwarding headers. Use
your existing Studio login credentials; this setting does not create or reset
an account. Desktop authentication through its existing secret remains
unchanged.

## Check the setup

Open the configured HTTPS URL from an allowed device. The frontend and its
assets should load, and Studio should require login before accessing protected
API routes. If the page returns 404, check the backend port, loopback target,
the setting inherited by the backend, and the three proxy headers above.
An API request without a session or API key must still be rejected.
