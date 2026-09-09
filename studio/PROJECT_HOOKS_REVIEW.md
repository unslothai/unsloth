# Reviewed project tool hooks

This split adds a Hooks tab, exact-content trust and synchronous pre/post tool
hooks. It is independent of the verification engine source diff; review/execution
use that engine (#10585), lifecycle (#10633), edits (#10577) and commands (#10636)
as prerequisites. Missing verification support leaves unconfigured tool calls usable
and makes configured hook review/execution unavailable without a shell fallback.

Only trusted, matching synchronous PreToolUse/PostToolUse handlers run. They cannot
change tool arguments or permissions. Trust, workspace identity and source bytes
are revalidated before native dispatch. A conflicting saved conversation cannot
bypass hooks on an explicitly bound project request. Failed capability probing
after trust is saved returns the saved trust with execution unavailable.

Post-hook failure appears before bounded tool output, so truncation cannot turn a
completed mutation plus failed validation into apparent success. With no stored
hook trust, project calls perform a bounded trust-state lookup, then skip hook file
discovery, extra workspace leases and process probes. No-output hooks return the
original result without another truncation pass. This is not a claim of literally
zero routing overhead: the project/trust lookup is additional work.

Hook commands share the verification process boundary. Other lifecycle and async
hook declarations are reviewable but remain inactive. Native fixtures execute
reviewed hooks without models; live provider and packaged desktop behavior are
separate gates.
