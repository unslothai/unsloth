# unsloth-link

A small helper that carries Studio cluster traffic over [tailcat](https://github.com/tailscale/tailcat)
tunnels (userspace WireGuard with NAT traversal; no Tailscale account, no admin rights), so a
worker on another network can be reached. The cluster's mutual TLS still runs inside the tunnel.
`studio/cluster_link/tailnet.py` drives it.

- `unsloth-link stdio [--derpmap-url URL] [--parent-pid PID]`: JSON lines on stdin/stdout (`serve`,
  `allow`, `revoke`, `forward`, `unforward`, `probe`, `pubkey`, `shutdown`). Addresses and keys travel
  only on stdin and stdout. The helper exits when stdin closes, or once PID is no longer its parent.
- `unsloth-link genkey --kind server|client --path FILE [--fixed-region]`: atomically writes a 0600
  key file in tailcat's key format and prints its public key; an existing key is kept and its public
  key printed again.

Build: `./build.sh [OUT_DIR]` cross-compiles linux, macOS and Windows (amd64, arm64) with
`CGO_ENABLED=0` and writes `SHA256SUMS`. It needs the Go version in `go.mod`. tailcat is pinned
in `go.mod`. `build-tags.txt` is tailcat's release tag list for that commit, and
`LICENSE.tailcat` is its BSD-3-Clause licence, shipped next to each binary.

Release: `package.py DIST OUT` packs byte-reproducible archives and `SHA256SUMS`;
`.github/workflows/unsloth-link.yml` builds them, fails unless they match
`studio/unsloth_link_pins.json`, smokes each platform natively (`smoke.py`) and publishes them on
the pinned tag. `update_pins.py` rewrites the pins from a release. `studio/install_unsloth_link.py`
downloads the archive for this host on first need and refuses anything that differs from its pin;
the pins also carry each executable's sha256, checked before an installed helper is run.

Tests: set `TS_DEBUG_TAILCAT_LOCAL_DERP=1` for a loopback relay. See
`tests/studio/test_cluster_link_tailnet.py`, which runs against real helpers when
`UNSLOTH_LINK_BIN` is set.
