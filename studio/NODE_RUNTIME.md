# Managed Node updates

Studio pins one exact release of a supported Node LTS major in
`node_prebuilt_pins.json`. Setup verifies downloaded archives against the six
committed SHA-256 digests. The default does not query Node's release index.

Review the pin monthly. Take security releases promptly, and take an earlier
update when a confirmed compatibility or reliability fix needs it. Routine
patches and minor releases can be grouped into the monthly update. Move to a new
LTS major after compatibility testing and before the current major reaches
end of life. Freezing an LTS version until the next major would miss security
fixes within that line.

This is a maintainer release policy, not a timer on the user's machine. A new
pin reaches managed installations through Studio setup or update. An unchanged,
verified installation downloads nothing. Setup checks the new pin even when
its managed Node is already on PATH, and reruns once if the package update
changes only the pin. A usable old installation is retained with a warning if
replacement fails; a successful Studio update alone does not prove Node was
replaced.

A compatible external Node/npm remains user-managed and is used read-only.
`UNSLOTH_SKIP_NODE_INSTALL=1` disables automatic Node installation and updates.
The Docker `unsloth-studio-update` helper also moves the image's managed Node
(`$STUDIO_HOME/node`) to the new package's pin once the updated Studio passes its
checks; an unchanged pin downloads nothing, and a failed download keeps the
current Node without failing the update.

## Updating the pin

Dependabot cannot update this file: it is not a package manifest, and the digests have to be
checked against Node's signed checksums. Instead `.github/workflows/node-pins-update.yml` runs
`scripts/update_node_pins.py` daily. When the pinned LTS line has a newer release (security
releases included), it verifies `SHASUMS256.txt` against the Node release keys, downloads and
hashes all six archives, and opens or refreshes one PR on `ci/node-pins`. It never moves to a
new LTS major; that stays the manual review below. Run it locally with
`python scripts/update_node_pins.py --check` (exit 1 when a newer release exists) or `--write`.

For a manual update or a new major:

1. Check the [Node release schedule](https://github.com/nodejs/Release#release-schedule)
   and the selected release's notes, including security fixes.
2. Fetch its `SHASUMS256.txt` and signature from `https://nodejs.org/dist/v<version>/`.
   Verify the signature with the [Node release keys](https://github.com/nodejs/release-keys).
3. Update `default_version` and all six archive digests together: Linux and macOS
   x64/ARM64 `.tar.gz` files and Windows x64/ARM64 `.zip` files. Download the
   archives and verify their digests before committing the pins.
4. Run the Node installer/runtime tests, both platforms' Node selection tests,
   and setup rerun tests. Check an upgrade from the previous pin, a second run
   with no download, and frontend/OXC compatibility with the new runtime.

## Windows reputation warnings

Keeping Node patched addresses known Node vulnerabilities. It does not guarantee
that Windows accepts every executable. [Microsoft Defender SmartScreen](https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/)
checks reputation and signatures. Diagnose the actual flagged file and warning
before attributing it to Node. Do not disable Windows protections or remove
download security metadata as part of a Node update.
