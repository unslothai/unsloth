#!/usr/bin/env python3
"""
Run skill evals: with-skill vs without-skill using claude -p.

Usage:
    python3 skills/unsloth/evals/run_evals.py
    python3 skills/unsloth/evals/run_evals.py --evals-file skills/unsloth/evals/evals.json
    python3 skills/unsloth/evals/run_evals.py --only setup         # run single eval by id
    python3 skills/unsloth/evals/run_evals.py --skip-baseline      # only run with-skill
    python3 skills/unsloth/evals/run_evals.py --model sonnet       # cheaper model for testing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
SKILL_FILE = SKILL_DIR / "SKILL.md"
DEFAULT_EVALS = SCRIPT_DIR / "evals.json"
WORKSPACE_DIR = SKILL_DIR.parent / "unsloth-workspace"
REPO_ROOT = SKILL_DIR.parent.parent  # skills/unsloth -> skills -> repo root

# Context prepended to every eval prompt so the agent knows the environment
ENV_CONTEXT = """
CONTEXT about the environment:
- Working directory: {repo_root}
- Mac M4, no NVIDIA GPU. Full Unsloth training does not work here, but dry-run flows do. GGUF inference does work on this machine.
- unsloth CLI is installed at /opt/homebrew/bin/unsloth (version 2026.3.5)
- Studio venv exists at ~/.unsloth/studio/.venv (Python 3.11)
- Use pip3, not pip.
- `unsloth studio setup` is already done in this environment. Only run it if the task explicitly requires verifying setup behavior.
- If a command fails, note it and keep going.
- Save all output/files you create into: {output_dir}
""".strip()


def run_claude(
    prompt: str, with_skill: bool, output_dir: Path, model: str = "opus"
) -> dict:
    """Run a single eval via claude -p and return the JSON result."""
    output_dir.mkdir(parents = True, exist_ok = True)

    full_prompt = f"{prompt}\n\n{ENV_CONTEXT.format(repo_root = REPO_ROOT, output_dir = output_dir)}"

    cmd = [
        "claude",
        "-p",
        full_prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--allowedTools",
        "Bash,Read,Write,Glob,Grep,Edit",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]

    if with_skill:
        # Inject skill content as appended system prompt
        skill_content = SKILL_FILE.read_text()
        cmd.extend(
            [
                "--append-system-prompt",
                f"You have the following skill loaded. Use it to complete the task:\n\n{skill_content}",
            ]
        )
    else:
        # Without skill: tell the agent to explore the codebase, but not read skills/
        cmd.extend(
            [
                "--append-system-prompt",
                "You do NOT have any skill documentation. Figure out how to complete the task by exploring the codebase. Do NOT read any files under the skills/ directory.",
            ]
        )

    print(
        f"  Running claude -p ({('with_skill' if with_skill else 'without_skill')})..."
    )
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output = True,
            text = True,
            timeout = 300,  # 5 min max per eval
            cwd = str(REPO_ROOT),
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
        return {"error": "timeout", "duration_ms": 300000}

    # Parse JSON output
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Save raw output for debugging
        (output_dir / "raw_stdout.txt").write_text(result.stdout)
        (output_dir / "raw_stderr.txt").write_text(result.stderr)
        print(f"  Failed to parse JSON output (saved raw to {output_dir})")
        return {"error": "json_parse_failed", "duration_ms": int(elapsed * 1000)}

    # Save results
    (output_dir / "result.json").write_text(json.dumps(data, indent = 2))
    (output_dir / "response.md").write_text(data.get("result", ""))

    # Extract timing
    timing = {
        "total_tokens": sum(
            v.get("inputTokens", 0)
            + v.get("outputTokens", 0)
            + v.get("cacheReadInputTokens", 0)
            + v.get("cacheCreationInputTokens", 0)
            for v in data.get("modelUsage", {}).values()
        ),
        "duration_ms": data.get("duration_ms", int(elapsed * 1000)),
        "total_duration_seconds": round(elapsed, 1),
        "cost_usd": data.get("total_cost_usd", 0),
        "num_turns": data.get("num_turns", 0),
    }
    (output_dir / "timing.json").write_text(json.dumps(timing, indent = 2))

    print(
        f"  Done: {timing['total_tokens']} tokens, {timing['total_duration_seconds']}s, ${timing['cost_usd']:.4f}"
    )
    return {**timing, "data": data}


def run_grader(
    eval_entry: dict,
    with_result: dict,
    without_result: dict,
    iteration_dir: Path,
    model: str = "opus",
) -> dict:
    """Run grader via claude -p to evaluate both outputs."""
    eval_id = eval_entry["id"]
    eval_dir = iteration_dir / eval_id

    with_response_path = eval_dir / "with_skill" / "outputs" / "response.md"
    without_response_path = eval_dir / "without_skill" / "outputs" / "response.md"

    with_response = with_response_path.read_text() if with_response_path.exists() else "(run failed — no output)"
    without_response = (
        without_response_path.read_text()
        if without_result and without_response_path.exists()
        else "N/A"
    )

    expectations = eval_entry.get("expectations", [])
    expectations_text = "\n".join(f"- {e}" for e in expectations)

    grader_prompt = f"""You are grading two agent runs on this task:

**Task prompt:** {eval_entry['prompt']}

**Expectations:**
{expectations_text}

**WITH SKILL output:**
{with_response[:3000]}

**WITHOUT SKILL output:**
{without_response[:3000]}

**WITH SKILL timing:** {with_result.get('total_tokens', '?')} tokens, {with_result.get('total_duration_seconds', '?')}s
**WITHOUT SKILL timing:** {without_result.get('total_tokens', '?') if without_result else 'N/A'} tokens, {without_result.get('total_duration_seconds', '?') if without_result else 'N/A'}s

Grade each expectation as passed/failed for BOTH runs. Use this exact JSON format:
```json
{{
  "eval_id": "{eval_id}",
  "with_skill": {{
    "expectations": [
      {{"text": "expectation text", "passed": true, "evidence": "why"}}
    ],
    "pass_rate": 0.0
  }},
  "without_skill": {{
    "expectations": [
      {{"text": "expectation text", "passed": true, "evidence": "why"}}
    ],
    "pass_rate": 0.0
  }},
  "summary": "one-line comparison"
}}
```

Output ONLY the JSON, nothing else."""

    cmd = [
        "claude",
        "-p",
        grader_prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--tools",
        "",  # no tools needed for grading
    ]

    print(f"  Grading {eval_id}...")
    result = subprocess.run(
        cmd, capture_output = True, text = True, timeout = 120, cwd = str(REPO_ROOT)
    )

    try:
        data = json.loads(result.stdout)
        grading_text = data.get("result", "")
        # Extract JSON from the response
        json_start = grading_text.find("{")
        json_end = grading_text.rfind("}") + 1
        if json_start >= 0:
            grading = json.loads(grading_text[json_start:json_end])
            (eval_dir / "grading.json").write_text(json.dumps(grading, indent = 2))
            print(
                f"  Graded: with_skill={grading.get('with_skill', {}).get('pass_rate', '?')}, "
                f"without_skill={grading.get('without_skill', {}).get('pass_rate', '?')}"
            )
            return grading
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Grading parse error: {e}")
        (eval_dir / "grading_raw.txt").write_text(result.stdout)

    return {}


def build_benchmark(iteration_dir: Path, results: list, evals: list) -> dict:
    """Build benchmark.json from all results."""
    runs = []
    for eval_entry, with_res, without_res in results:
        eval_id = eval_entry["id"]
        grading_file = iteration_dir / eval_id / "grading.json"
        grading = {}
        if grading_file.exists():
            grading = json.loads(grading_file.read_text())

        for config, res in [("with_skill", with_res), ("without_skill", without_res)]:
            if not res or "error" in res:
                continue
            g = grading.get(config, {})
            expectations = g.get("expectations", [])
            passed = sum(1 for e in expectations if e.get("passed"))
            total = len(expectations)
            runs.append(
                {
                    "eval_id": eval_id,
                    "eval_name": eval_id,
                    "configuration": config,
                    "run_number": 1,
                    "result": {
                        "pass_rate": passed / total if total else 0,
                        "passed": passed,
                        "failed": total - passed,
                        "total": total,
                        "time_seconds": res.get("total_duration_seconds", 0),
                        "tokens": res.get("total_tokens", 0),
                        "cost_usd": res.get("cost_usd", 0),
                        "errors": 0,
                    },
                    "expectations": expectations,
                }
            )

    # Compute summaries
    with_runs = [r for r in runs if r["configuration"] == "with_skill"]
    without_runs = [r for r in runs if r["configuration"] == "without_skill"]

    def stats(run_list, key):
        vals = [r["result"][key] for r in run_list]
        if not vals:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": round(sum(vals) / len(vals), 3),
            "min": min(vals),
            "max": max(vals),
        }

    benchmark = {
        "metadata": {
            "skill_name": "unsloth",
            "skill_path": str(SKILL_DIR),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evals_run": [e["id"] for e in evals],
            "runs_per_configuration": 1,
        },
        "runs": runs,
        "run_summary": {
            "with_skill": {
                "pass_rate": stats(with_runs, "pass_rate"),
                "time_seconds": stats(with_runs, "time_seconds"),
                "tokens": stats(with_runs, "tokens"),
            },
            "without_skill": {
                "pass_rate": stats(without_runs, "pass_rate"),
                "time_seconds": stats(without_runs, "time_seconds"),
                "tokens": stats(without_runs, "tokens"),
            },
        },
    }

    # Delta
    w_pr = benchmark["run_summary"]["with_skill"]["pass_rate"]["mean"]
    wo_pr = benchmark["run_summary"]["without_skill"]["pass_rate"]["mean"]
    w_tok = benchmark["run_summary"]["with_skill"]["tokens"]["mean"]
    wo_tok = benchmark["run_summary"]["without_skill"]["tokens"]["mean"]
    w_time = benchmark["run_summary"]["with_skill"]["time_seconds"]["mean"]
    wo_time = benchmark["run_summary"]["without_skill"]["time_seconds"]["mean"]
    benchmark["run_summary"]["delta"] = {
        "pass_rate": f"{w_pr - wo_pr:+.3f}",
        "time_seconds": f"{w_time - wo_time:+.1f}",
        "tokens": f"{int(w_tok - wo_tok):+d}",
    }

    return benchmark


def main():
    parser = argparse.ArgumentParser(description = "Run unsloth skill evals")
    parser.add_argument("--evals-file", type = Path, default = DEFAULT_EVALS)
    parser.add_argument("--only", type = str, help = "Run only this eval id")
    parser.add_argument(
        "--skip-baseline", action = "store_true", help = "Skip without-skill runs"
    )
    parser.add_argument("--skip-grading", action = "store_true", help = "Skip grading step")
    parser.add_argument(
        "--model", type = str, default = "opus", help = "Model to use (opus, sonnet, haiku)"
    )
    parser.add_argument(
        "--iteration",
        type = int,
        default = None,
        help = "Iteration number (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # Load evals
    with open(args.evals_file) as f:
        evals_data = json.load(f)
    evals = evals_data.get("evals", [])

    if args.only:
        evals = [e for e in evals if e["id"] == args.only]
        if not evals:
            print(f"No eval found with id '{args.only}'")
            sys.exit(1)

    # Determine iteration number
    if args.iteration is not None:
        iteration = args.iteration
    else:
        existing = (
            sorted(WORKSPACE_DIR.glob("iteration-*")) if WORKSPACE_DIR.exists() else []
        )
        iteration = len(existing) + 1
    iteration_dir = WORKSPACE_DIR / f"iteration-{iteration}"
    iteration_dir.mkdir(parents = True, exist_ok = True)

    print(f"=== Unsloth Skill Evals — Iteration {iteration} ===")
    print(f"Evals: {[e['id'] for e in evals]}")
    print(f"Model: {args.model}")
    print(f"Output: {iteration_dir}\n")

    # Run evals
    results = []
    for eval_entry in evals:
        eval_id = eval_entry["id"]
        prompt = eval_entry["prompt"]
        print(f"\n--- Eval: {eval_id} ---")
        print(f"Prompt: {prompt}\n")

        # With skill
        with_dir = iteration_dir / eval_id / "with_skill" / "outputs"
        with_result = run_claude(
            prompt, with_skill = True, output_dir = with_dir, model = args.model
        )

        # Without skill (baseline)
        without_result = None
        if not args.skip_baseline:
            without_dir = iteration_dir / eval_id / "without_skill" / "outputs"
            without_result = run_claude(
                prompt, with_skill = False, output_dir = without_dir, model = args.model
            )

        results.append((eval_entry, with_result, without_result))

    # Grade
    if not args.skip_grading:
        print(f"\n=== Grading ===")
        for eval_entry, with_res, without_res in results:
            run_grader(
                eval_entry, with_res, without_res, iteration_dir, model = args.model
            )

    # Build benchmark
    print(f"\n=== Building Benchmark ===")
    benchmark = build_benchmark(iteration_dir, results, evals)
    benchmark_path = iteration_dir / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, indent = 2))
    print(f"Saved: {benchmark_path}")

    # Print summary
    summary = benchmark["run_summary"]
    delta = summary.get("delta", {})
    print(f"\n{'='*60}")
    print(f"RESULTS — Iteration {iteration}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'With Skill':<15} {'Without Skill':<15} {'Delta':<10}")
    print(f"{'-'*60}")
    print(
        f"{'Pass rate':<20} {summary['with_skill']['pass_rate']['mean']:<15.1%} "
        f"{summary['without_skill']['pass_rate']['mean']:<15.1%} {delta.get('pass_rate', 'N/A')}"
    )
    print(
        f"{'Tokens':<20} {summary['with_skill']['tokens']['mean']:<15.0f} "
        f"{summary['without_skill']['tokens']['mean']:<15.0f} {delta.get('tokens', 'N/A')}"
    )
    print(
        f"{'Time (s)':<20} {summary['with_skill']['time_seconds']['mean']:<15.1f} "
        f"{summary['without_skill']['time_seconds']['mean']:<15.1f} {delta.get('time_seconds', 'N/A')}"
    )

    # Try to launch viewer
    viewer_script = (
        Path.home()
        / ".claude"
        / "skills"
        / "skill-creator"
        / "eval-viewer"
        / "generate_review.py"
    )
    if viewer_script.exists():
        print(f"\nLaunching eval viewer...")
        subprocess.Popen(
            [
                "python3",
                str(viewer_script),
                str(iteration_dir),
                "--skill-name",
                "unsloth",
                "--benchmark",
                str(benchmark_path),
            ],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
    else:
        print(f"\nViewer not found at {viewer_script}")

    print(f"\nDone! Results in {iteration_dir}")


if __name__ == "__main__":
    main()
