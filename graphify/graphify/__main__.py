"""graphify CLI - `graphify install` sets up the Claude Code skill."""

from __future__ import annotations
import json
import re
import shutil
import sys
from pathlib import Path

_SETTINGS_HOOK = {
    "matcher": "Glob|Grep",
    "hooks": [
        {
            "type": "command",
            "command": (
                "[ -f graphify-out/graph.json ] && "
                "echo 'graphify: Knowledge graph exists. Read graphify-out/GRAPH_REPORT.md "
                "for god nodes and community structure before searching raw files.' || true"
            ),
        }
    ],
}

_SKILL_REGISTRATION = (
    "\n# graphify\n"
    "- **graphify** (`~/.claude/skills/graphify/SKILL.md`) "
    "- any input to knowledge graph. Trigger: `/graphify`\n"
    "When the user types `/graphify`, invoke the Skill tool "
    'with `skill: "graphify"` before doing anything else.\n'
)


_PLATFORM_CONFIG: dict[str, dict] = {
    "claude": {
        "skill_file": "skill.md",
        "skill_dst": Path(".claude") / "skills" / "graphify" / "SKILL.md",
        "claude_md": True,
    },
    "codex": {
        "skill_file": "skill-codex.md",
        "skill_dst": Path(".agents") / "skills" / "graphify" / "SKILL.md",
        "claude_md": False,
    },
    "opencode": {
        "skill_file": "skill-opencode.md",
        "skill_dst": Path(".config") / "opencode" / "skills" / "graphify" / "SKILL.md",
        "claude_md": False,
    },
    "claw": {
        "skill_file": "skill-claw.md",
        "skill_dst": Path(".claw") / "skills" / "graphify" / "SKILL.md",
        "claude_md": False,
    },
}


def install(platform: str = "claude") -> None:
    if platform not in _PLATFORM_CONFIG:
        print(
            f"error: unknown platform '{platform}'. Choose from: {', '.join(_PLATFORM_CONFIG)}",
            file = sys.stderr,
        )
        sys.exit(1)

    cfg = _PLATFORM_CONFIG[platform]
    skill_src = Path(__file__).parent / cfg["skill_file"]
    if not skill_src.exists():
        print(
            f"error: {cfg['skill_file']} not found in package - reinstall graphify",
            file = sys.stderr,
        )
        sys.exit(1)

    skill_dst = Path.home() / cfg["skill_dst"]
    skill_dst.parent.mkdir(parents = True, exist_ok = True)
    shutil.copy(skill_src, skill_dst)
    print(f"  skill installed  →  {skill_dst}")

    if cfg["claude_md"]:
        # Register in ~/.claude/CLAUDE.md (Claude Code only)
        claude_md = Path.home() / ".claude" / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()
            if "graphify" in content:
                print(f"  CLAUDE.md        →  already registered (no change)")
            else:
                claude_md.write_text(content.rstrip() + _SKILL_REGISTRATION)
                print(f"  CLAUDE.md        →  skill registered in {claude_md}")
        else:
            claude_md.parent.mkdir(parents = True, exist_ok = True)
            claude_md.write_text(_SKILL_REGISTRATION.lstrip())
            print(f"  CLAUDE.md        →  created at {claude_md}")

    print()
    print("Done. Open your AI coding assistant and type:")
    print()
    print("  /graphify .")
    print()


_CLAUDE_MD_SECTION = """\
## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
"""

_CLAUDE_MD_MARKER = "## graphify"

# AGENTS.md section for Codex, OpenCode, and OpenClaw.
# All three platforms read AGENTS.md in the project root for persistent instructions.
_AGENTS_MD_SECTION = """\
## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
"""

_AGENTS_MD_MARKER = "## graphify"


def _agents_install(project_dir: Path, platform: str) -> None:
    """Write the graphify section to the local AGENTS.md (Codex/OpenCode/OpenClaw)."""
    target = (project_dir or Path(".")) / "AGENTS.md"

    if target.exists():
        content = target.read_text()
        if _AGENTS_MD_MARKER in content:
            print(f"graphify already configured in AGENTS.md")
            return
        new_content = content.rstrip() + "\n\n" + _AGENTS_MD_SECTION
    else:
        new_content = _AGENTS_MD_SECTION

    target.write_text(new_content)
    print(f"graphify section written to {target.resolve()}")
    print()
    print(
        f"{platform.capitalize()} will now check the knowledge graph before answering"
    )
    print("codebase questions and rebuild it after code changes.")
    print()
    print("Note: unlike Claude Code, there is no PreToolUse hook equivalent for")
    print(f"{platform.capitalize()} — the AGENTS.md rules are the always-on mechanism.")


def _agents_uninstall(project_dir: Path) -> None:
    """Remove the graphify section from the local AGENTS.md."""
    target = (project_dir or Path(".")) / "AGENTS.md"

    if not target.exists():
        print("No AGENTS.md found in current directory - nothing to do")
        return

    content = target.read_text()
    if _AGENTS_MD_MARKER not in content:
        print("graphify section not found in AGENTS.md - nothing to do")
        return

    cleaned = re.sub(
        r"\n*## graphify\n.*?(?=\n## |\Z)",
        "",
        content,
        flags = re.DOTALL,
    ).rstrip()
    if cleaned:
        target.write_text(cleaned + "\n")
        print(f"graphify section removed from {target.resolve()}")
    else:
        target.unlink()
        print(f"AGENTS.md was empty after removal - deleted {target.resolve()}")


def claude_install(project_dir: Path | None = None) -> None:
    """Write the graphify section to the local CLAUDE.md."""
    target = (project_dir or Path(".")) / "CLAUDE.md"

    if target.exists():
        content = target.read_text()
        if _CLAUDE_MD_MARKER in content:
            print("graphify already configured in CLAUDE.md")
            return
        new_content = content.rstrip() + "\n\n" + _CLAUDE_MD_SECTION
    else:
        new_content = _CLAUDE_MD_SECTION

    target.write_text(new_content)
    print(f"graphify section written to {target.resolve()}")

    # Also write Claude Code PreToolUse hook to .claude/settings.json
    _install_claude_hook(project_dir or Path("."))

    print()
    print("Claude Code will now check the knowledge graph before answering")
    print("codebase questions and rebuild it after code changes.")


def _install_claude_hook(project_dir: Path) -> None:
    """Add graphify PreToolUse hook to .claude/settings.json."""
    settings_path = project_dir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents = True, exist_ok = True)

    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    else:
        settings = {}

    hooks = settings.setdefault("hooks", {})
    pre_tool = hooks.setdefault("PreToolUse", [])

    # Check if already installed
    if any(h.get("matcher") == "Glob|Grep" and "graphify" in str(h) for h in pre_tool):
        print(f"  .claude/settings.json  →  hook already registered (no change)")
        return

    pre_tool.append(_SETTINGS_HOOK)
    settings_path.write_text(json.dumps(settings, indent = 2))
    print(f"  .claude/settings.json  →  PreToolUse hook registered")


def _uninstall_claude_hook(project_dir: Path) -> None:
    """Remove graphify PreToolUse hook from .claude/settings.json."""
    settings_path = project_dir / ".claude" / "settings.json"
    if not settings_path.exists():
        return
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        return
    pre_tool = settings.get("hooks", {}).get("PreToolUse", [])
    filtered = [
        h
        for h in pre_tool
        if not (h.get("matcher") == "Glob|Grep" and "graphify" in str(h))
    ]
    if len(filtered) == len(pre_tool):
        return
    settings["hooks"]["PreToolUse"] = filtered
    settings_path.write_text(json.dumps(settings, indent = 2))
    print(f"  .claude/settings.json  →  PreToolUse hook removed")


def claude_uninstall(project_dir: Path | None = None) -> None:
    """Remove the graphify section from the local CLAUDE.md."""
    target = (project_dir or Path(".")) / "CLAUDE.md"

    if not target.exists():
        print("No CLAUDE.md found in current directory - nothing to do")
        return

    content = target.read_text()
    if _CLAUDE_MD_MARKER not in content:
        print("graphify section not found in CLAUDE.md - nothing to do")
        return

    # Remove the ## graphify section: from the marker to the next ## heading or EOF
    cleaned = re.sub(
        r"\n*## graphify\n.*?(?=\n## |\Z)",
        "",
        content,
        flags = re.DOTALL,
    ).rstrip()
    if cleaned:
        target.write_text(cleaned + "\n")
        print(f"graphify section removed from {target.resolve()}")
    else:
        target.unlink()
        print(f"CLAUDE.md was empty after removal - deleted {target.resolve()}")

    _uninstall_claude_hook(project_dir or Path("."))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: graphify <command>")
        print()
        print("Commands:")
        print(
            "  install [--platform P]  copy skill to platform config dir (claude|codex|opencode|claw)"
        )
        print(
            "  benchmark [graph.json]  measure token reduction vs naive full-corpus approach"
        )
        print(
            "  hook install            install post-commit/post-checkout git hooks (all platforms)"
        )
        print("  hook uninstall          remove git hooks")
        print("  hook status             check if git hooks are installed")
        print(
            "  claude install          write graphify section to CLAUDE.md + PreToolUse hook (Claude Code)"
        )
        print(
            "  claude uninstall        remove graphify section from CLAUDE.md + PreToolUse hook"
        )
        print("  codex install           write graphify section to AGENTS.md (Codex)")
        print("  codex uninstall         remove graphify section from AGENTS.md")
        print(
            "  opencode install        write graphify section to AGENTS.md (OpenCode)"
        )
        print("  opencode uninstall      remove graphify section from AGENTS.md")
        print(
            "  claw install            write graphify section to AGENTS.md (OpenClaw)"
        )
        print("  claw uninstall          remove graphify section from AGENTS.md")
        print()
        return

    cmd = sys.argv[1]
    if cmd == "install":
        platform = "claude"
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i].startswith("--platform="):
                platform = args[i].split("=", 1)[1]
                i += 1
            elif args[i] == "--platform" and i + 1 < len(args):
                platform = args[i + 1]
                i += 2
            else:
                i += 1
        install(platform = platform)
    elif cmd == "claude":
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd == "install":
            claude_install()
        elif subcmd == "uninstall":
            claude_uninstall()
        else:
            print("Usage: graphify claude [install|uninstall]", file = sys.stderr)
            sys.exit(1)
    elif cmd in ("codex", "opencode", "claw"):
        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd == "install":
            _agents_install(Path("."), cmd)
        elif subcmd == "uninstall":
            _agents_uninstall(Path("."))
        else:
            print(f"Usage: graphify {cmd} [install|uninstall]", file = sys.stderr)
            sys.exit(1)
    elif cmd == "hook":
        from graphify.hooks import (
            install as hook_install,
            uninstall as hook_uninstall,
            status as hook_status,
        )

        subcmd = sys.argv[2] if len(sys.argv) > 2 else ""
        if subcmd == "install":
            print(hook_install(Path(".")))
        elif subcmd == "uninstall":
            print(hook_uninstall(Path(".")))
        elif subcmd == "status":
            print(hook_status(Path(".")))
        else:
            print("Usage: graphify hook [install|uninstall|status]", file = sys.stderr)
            sys.exit(1)
    elif cmd == "benchmark":
        from graphify.benchmark import run_benchmark, print_benchmark

        graph_path = sys.argv[2] if len(sys.argv) > 2 else "graphify-out/graph.json"
        # Try to load corpus_words from detect output
        corpus_words = None
        detect_path = Path(".graphify_detect.json")
        if detect_path.exists():
            try:
                detect_data = json.loads(detect_path.read_text())
                corpus_words = detect_data.get("total_words")
            except Exception:
                pass
        result = run_benchmark(graph_path, corpus_words = corpus_words)
        print_benchmark(result)
    else:
        print(f"error: unknown command '{cmd}'", file = sys.stderr)
        print("Run 'graphify --help' for usage.", file = sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
