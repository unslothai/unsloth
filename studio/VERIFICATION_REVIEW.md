# Project verification engine

This split owns reviewed verification profiles, durable asynchronous execution,
bounded progress, cancellation and historical evidence. The Verification tab and
/verify command operate on the stored project workspace. Ordinary tool calls have
no hook wrapper in this PR. Hook discovery, trust and tool interception are in a
separate follow-up.

Execution requires confined edits (#10577), supervised commands (#10636), and
project lifecycle retirement (#10633). Read-only profile/history routes work
without the native runner. Missing lifecycle support refuses execution. The shared
archive/delete changes are exclusively in #10633; its owner retires verification
before the project record or workspace changes.

The profile editor enforces both per-field limits and the 128 KiB normalized UTF-8
JSON limit, including escaped characters. Heartbeat or progress-storage failure
stops execution and records a failed run, rather than claiming user cancellation.
Workspace resolution rechecks deletion/archive and carries the stored revision.

The dedicated native workflow composes lifecycle support in both variants and
pinned edit/command support in the native variant. Linux executes the confined
runner; macOS/Windows verify refusal. Local focused review regressions passed;
each new source head still requires the full workflow and frontend gates.

Run the four test_project_verification_*.py files from studio/backend. Frontend
validation uses npm test, npm run typecheck, npm run build and npm run bundle:check.
Packaged desktop and live model/provider behavior remain separate qualification.
