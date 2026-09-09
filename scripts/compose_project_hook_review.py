# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve only the known additive hook/verification review composition conflicts."""
import re
import subprocess
from pathlib import Path


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("Review composition context changed")
    return text.replace(before, after, 1)


def main():
    conflicts = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=U"], text=True).splitlines()
    allowed = {"studio/backend/main.py", "studio/frontend/src/features/chat/chat-page.tsx"}
    if not conflicts or set(conflicts) - allowed:
        raise RuntimeError("Unexpected review composition conflicts")
    for name in conflicts:
        path = Path(name)
        if name.endswith("main.py"):
            def combine(match):
                left, right = match.group(1), match.group(2)
                for line in (left + right).splitlines():
                    if not any(router in line for router in ("project_hooks_router", "project_verification_router")):
                        raise RuntimeError("Unexpected route-registration conflict")
                return left + right
            resolved = re.sub(r"^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n", combine, path.read_text(), flags=re.M | re.S)
        else:
            resolved = subprocess.check_output(["git", "show", ":2:" + name], text=True)
            engine = subprocess.check_output(["git", "show", ":3:" + name], text=True)
            a = engine.index("const ProjectChecksPanel = lazy(")
            b = engine.index("const ProjectSourcesPanel = lazy(", a)
            resolved = replace_once(resolved, "const ProjectHooksLandingPanel = lazy(", engine[a:b] + "const ProjectHooksLandingPanel = lazy(")
            resolved = replace_once(resolved, '"chats" | "sources" | "hooks"', '"chats" | "sources" | "hooks" | "checks"')
            start = '              <button\n                type="button"\n                onClick={() => setProjectTab("checks")}'
            a = engine.index(start)
            b = engine.index("              </button>", a) + len("              </button>\n")
            hook_start = start.replace('"checks"', '"hooks"')
            resolved = replace_once(resolved, hook_start, engine[a:b] + hook_start)
            before = '            {projectTab === "hooks" ? ('
            after = '''            {projectTab === "checks" ? (
              <Suspense fallback={<p className="mt-8 text-sm text-muted-foreground">Loading verification…</p>}>
                <ProjectChecksPanel key={projectId} projectId={projectId} />
              </Suspense>
            ) : projectTab === "hooks" ? ('''
            resolved = replace_once(resolved, before, after)
        if "<<<<<<<" in resolved or ">>>>>>>" in resolved:
            raise RuntimeError("Unresolved review composition")
        path.write_text(resolved)
        subprocess.run(["git", "add", name], check=True)


if __name__ == "__main__":
    main()
