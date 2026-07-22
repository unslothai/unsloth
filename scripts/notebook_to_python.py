#!/usr/bin/env python
# coding: utf-8
"""
Convert Jupyter notebooks (.ipynb) to executable Python scripts (.py).

Converts IPython magics to plain Python:
    !command          -> subprocess.run('command', shell=True)
    %cd path          -> os.chdir('path')
    %env VAR=value    -> os.environ['VAR'] = 'value'
    %%file filename   -> with open('filename', 'w') as f: f.write(...)
    %%capture         -> (skipped)
    /content/...      -> _WORKING_DIR + /...
"""

import nbformat
import re
import shlex
import sys
import os
import urllib.request
import urllib.parse
from pathlib import Path


# Allowlist of hosts for raw notebook fetches; anything else rejected before urlopen.
_ALLOWED_NOTEBOOK_HOSTS = {
    "raw.githubusercontent.com",
    "gist.githubusercontent.com",
}


# Metacharacters that mean a `!cmd` line can't be a flat argv -> keep shell=True + review marker.
_SHELL_METACHARS_RE = re.compile(r"\$\(|`|\|\||\||&&|>>?|<<?|\*|\?|;")


def needs_fstring(cmd: str) -> bool:
    """Check if command has Python variable interpolation like {var_name}."""
    pattern = r"(?<!\$)\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return bool(re.search(pattern, cmd))


def github_blob_to_raw(url: str) -> str:
    """Convert GitHub blob URL to raw URL."""
    # github.com/user/repo/blob/branch/path -> raw.githubusercontent.com/user/repo/branch/path
    # Exact host match (not substring) so attacker.example.com/github.com/blob/... is not rewritten.
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "github.com" or "/blob/" not in parsed.path:
        return url
    new_path = parsed.path.replace("/blob/", "/", 1)
    return urllib.parse.urlunparse(
        parsed._replace(netloc = "raw.githubusercontent.com", path = new_path)
    )


def download_notebook(url: str) -> tuple[str, str]:
    """Download notebook from URL. Returns (content, filename)."""
    raw_url = github_blob_to_raw(url)

    parsed = urllib.parse.urlparse(raw_url)
    filename = os.path.basename(urllib.parse.unquote(parsed.path))

    # Host allowlist: refuse to fetch from anything we don't recognise.
    host = parsed.hostname
    if host not in _ALLOWED_NOTEBOOK_HOSTS:
        raise ValueError(
            f"Refused notebook fetch from {host!r}: not in allowlist "
            f"{sorted(_ALLOWED_NOTEBOOK_HOSTS)}"
        )

    print(f"Downloading {url}...")
    with urllib.request.urlopen(raw_url, timeout = 60) as response:
        content = response.read().decode("utf-8")

    return content, filename


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def replace_colab_paths(source: str) -> str:
    """Replace Colab-specific /content/ paths with current working directory."""
    source = source.replace('"/content/', 'f"{_WORKING_DIR}/')
    source = source.replace("'/content/", "f'{_WORKING_DIR}/")
    return source


def _emit_shell_command(indent: str, full_cmd: str, *, allow_shell: bool) -> list[str]:
    """Render a `!cmd` notebook line as Python statements.

    f-string interpolation, shell metacharacters, or multiline force
    shell=True (shlex.split would drop operators), flagged with a
    WARNING comment. Otherwise emit shell=False argv form. allow_shell
    False makes shell=True emission a hard error.
    """
    needs_f = needs_fstring(full_cmd)
    has_meta = bool(_SHELL_METACHARS_RE.search(full_cmd))
    multiline = "\n" in full_cmd

    must_use_shell = needs_f or has_meta or multiline

    if must_use_shell:
        if not allow_shell:
            raise ValueError(
                "Cell uses shell metacharacters / interpolation but "
                "--no-allow-shell was set; refusing to emit shell=True"
            )
        warn = f"{indent}# WARNING: shell=True; reviewed for hostile input"
        f_prefix = "f" if needs_f else ""
        if multiline:
            escaped_cmd = full_cmd.replace('"""', r"\"\"\"")
            if escaped_cmd.rstrip().endswith('"'):
                escaped_cmd = escaped_cmd.rstrip() + " "
            stmt = f'{indent}subprocess.run({f_prefix}"""{escaped_cmd}""", shell=True)'
        else:
            stmt = f"{indent}subprocess.run({f_prefix}{full_cmd!r}, shell=True)"
        return [warn, stmt]

    return [f"{indent}subprocess.run(shlex.split({full_cmd!r}), shell=False)"]


def convert_cell_to_python(source: str, *, allow_shell: bool = True) -> str:
    """Convert a cell's IPython magics to plain Python."""
    lines = source.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped.startswith("%%capture"):
            i += 1
            continue

        if stripped.startswith("%%file "):
            filename = stripped[7:].strip()
            file_lines = []
            i += 1
            while i < len(lines):
                file_lines.append(lines[i])
                i += 1
            file_content = "\n".join(file_lines)
            file_content = file_content.replace('"""', r"\"\"\"")
            result.append(f'{indent}with open({filename!r}, "w") as _f:')
            result.append(f'{indent}    _f.write("""{file_content}""")')
            continue

        if stripped.startswith("!"):
            cmd_lines = [stripped[1:]]
            while cmd_lines[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                cmd_lines.append(lines[i].strip())
            full_cmd = "\n".join(cmd_lines)

            result.extend(_emit_shell_command(indent, full_cmd, allow_shell = allow_shell))

        # %cd path -> os.chdir(path)
        elif stripped.startswith("%cd "):
            path = stripped[4:].strip()
            result.append(f"{indent}os.chdir({path!r})")

        # %env VAR=value
        elif stripped.startswith("%env ") and "=" in stripped:
            match = re.match(r"%env\s+(\w+)=(.+)", stripped)
            if match:
                var, val = match.groups()
                result.append(f"{indent}os.environ[{var!r}] = {val!r}")

        # %env VAR
        elif stripped.startswith("%env "):
            var = stripped[5:].strip()
            result.append(f"{indent}os.environ.get({var!r})")

        # %pwd
        elif stripped == "%pwd":
            result.append(f"{indent}os.getcwd()")

        else:
            result.append(line)

        i += 1

    return "\n".join(result)


def convert_notebook(
    notebook_content: str,
    source_name: str = "notebook",
    *,
    allow_shell: bool = True,
) -> str:
    """Convert notebook JSON content to Python script."""
    # Parse notebook
    if isinstance(notebook_content, str):
        notebook = nbformat.reads(notebook_content, as_version = 4)
    else:
        notebook = notebook_content

    lines = [
        "#!/usr/bin/env python",
        "# coding: utf-8",
        f"# Converted from: {source_name}",
        "",
        "import shlex",
        "import subprocess",
        "import os",
        "import sys",
        "import re",
        "",
        "# Capture original packages before any installs",
        "_original_packages = subprocess.run(",
        "    [sys.executable, '-m', 'pip', 'freeze'],",
        "    capture_output=True, text=True",
        ").stdout",
        "",
        "# Working directory (replaces Colab's /content/)",
        "_WORKING_DIR = os.getcwd()",
        "",
    ]

    for cell in notebook.cells:
        source = cell.source.strip()
        if not source:
            continue

        if cell.cell_type == "code":
            converted = convert_cell_to_python(source, allow_shell = allow_shell)
            converted = replace_colab_paths(converted)
            lines.append(converted)
            lines.append("")

        elif cell.cell_type == "markdown":
            for line in source.split("\n"):
                lines.append(f"# {line}")
            lines.append("")

    # Add package restoration at the end
    lines.extend(
        [
            "",
            "# Restore original packages (install one by one, skip failures)",
            "for _pkg in _original_packages.strip().split('\\n'):",
            "    if _pkg:",
            "        subprocess.run([sys.executable, '-m', 'pip', 'install', _pkg, '-q'],",
            "                       stderr=subprocess.DEVNULL)",
            "",
        ]
    )

    return "\n".join(lines)


def convert_notebook_to_script(
    source: str,
    output_dir: str | None = None,
    *,
    allow_shell: bool = True,
):
    """
    Convert a notebook to Python script.

    Args:
        source: Local file path or URL to notebook
        output_dir: Output directory (optional, defaults to current directory)
        allow_shell: When False, refuse to emit `shell=True` for any
            `!cmd` cell that uses metacharacters / interpolation.
    """
    if is_url(source):
        content, filename = download_notebook(source)
        source_name = source
    else:
        filename = os.path.basename(source)
        with open(source, "r", encoding = "utf-8") as f:
            content = f.read()
        source_name = source

    output_filename = filename.replace(".ipynb", ".py")
    output_filename = output_filename.replace("(", "").replace(")", "").replace("-", "_")

    if output_dir:
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = output_filename

    script = convert_notebook(content, source_name, allow_shell = allow_shell)

    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(script)

    print(f"Converted {source} -> {output_path}")
    return output_path


def main():
    import argparse

    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = Formatter,
        epilog = """
Examples:
  python notebook_to_python.py notebook.ipynb
  python notebook_to_python.py -o scripts/ notebook1.ipynb notebook2.ipynb
  python notebook_to_python.py --output ./converted https://github.com/user/repo/blob/main/notebook.ipynb
  python notebook_to_python.py https://github.com/unslothai/notebooks/blob/main/nb/Oute_TTS_(1B).ipynb
""",
    )
    parser.add_argument("notebooks", nargs = "+", help = "Notebook files or URLs to convert.")
    parser.add_argument("-o", "--output", dest = "output_dir", default = ".", help = "Output directory.")
    # Default True for backwards compat; pass --no-allow-shell for untrusted notebooks.
    parser.add_argument(
        "--allow-shell",
        dest = "allow_shell",
        action = "store_true",
        default = True,
        help = "Allow emitting subprocess.run(..., shell=True) for cells "
        "that use shell metacharacters or interpolation (default).",
    )
    parser.add_argument(
        "--no-allow-shell",
        dest = "allow_shell",
        action = "store_false",
        help = "Refuse to emit shell=True; cells with metacharacters error out.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)

    # Track per-notebook failures; continue the loop and exit 1 if any failed.
    failures: list[tuple[str, str]] = []
    ok = 0
    total = len(args.notebooks)
    for source in args.notebooks:
        try:
            convert_notebook_to_script(
                source,
                output_dir = args.output_dir if args.output_dir != "." else None,
                allow_shell = args.allow_shell,
            )
            ok += 1
        except Exception as e:
            print(f"ERROR converting {source}: {e}")
            failures.append((source, f"{type(e).__name__}: {e}"))

    print(
        f"converted {ok}/{total}, {len(failures)} failed",
        file = sys.stderr if failures else sys.stdout,
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
