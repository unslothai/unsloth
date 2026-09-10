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
hook trust, project calls perform one bounded trust-state lookup, then call the
original executor with the original arguments. They skip extra project/thread
routing, archived-project checks, hook file discovery, workspace leases and
process probes. Non-project and non-hooked calls do not query hook trust. No-output hooks return the
original result without another truncation pass. This is not a claim of literally
zero routing overhead: argument binding and the project trust lookup add work.
The existing tool executor remains authoritative for unconfigured project routing
and errors. Review #10577 edits and #10585 verification before this integration;
#10633 lifecycle and #10636 commands complete its execution prerequisites.

Hook commands share the verification process boundary. Other lifecycle and async
hook declarations are reviewable but remain inactive. Native fixtures execute
reviewed hooks without models; live provider and packaged desktop behavior are
separate gates.

A local warmed SQLite microbenchmark (Python 3.11, seven batches of 100 stub
executor calls, no model or command execution) measured median 0.03 microseconds
for the direct stub, 1.13 microseconds through the non-project wrapper, and
392.91 microseconds for an unconfigured project call. This quantifies wrapper
and trust lookup cost on that machine; it is not a production latency guarantee.
