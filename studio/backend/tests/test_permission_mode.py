# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for permission_mode ("Ask for approval" / "Approve for me" /
"Off" / "Full access") permission levels.

Covers the auto-mode safety classifier in tools.py and the loop-level
behavior of run_safetensors_tool_loop: in "auto" mode only calls detected
as potentially unsafe pause for confirmation, in "full" mode nothing
pauses and the sandbox is dropped, and unset/unknown modes behave as
"ask" (every call pauses when confirm_tool_calls is on).
"""

import pytest

from core.inference.mcp_client import MCP_TOOL_PREFIX
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tools import is_potentially_unsafe_tool_call
from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from state import tool_approvals
from state.tool_approvals import resolve_tool_decision

_SESSION = "perm-mode-session"


@pytest.fixture(autouse = True)
def _clear_pending():
    with tool_approvals._lock:
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()


# ── classifier ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "unsafe"),
    [
        ("ls -la", False),
        ("cat foo.txt | grep hello", False),
        ("find . -name '*.py' | head -5", False),
        ("env FOO=1 grep -r pattern .", False),
        ("echo hi > out.txt", True),  # write redirection
        ("rm -rf /", True),
        ("ls; rm x", True),  # unsafe after separator
        ("xargs rm", True),  # wrapper forwards to unsafe target
        ("sudo ls", True),
        ("git push origin main", True),
        ("pip install requests", True),
        ("echo `whoami`", True),  # substitution fails closed
        ("python -c 'print(1)'", True),  # arbitrary code
        ("find . -exec rm {} ;", True),  # find can execute
        ("find . -delete", True),  # find can delete
        ("fd -x rm", True),  # fd runs a command per result
        ("fd --exec-batch rm", True),
        ("fd -e py pattern", False),  # plain fd search stays read only
        ("sort -o out.txt in.txt", True),  # -o writes a file
        ("sort --output=out in", True),
        ("sort --compress-program=sh big.txt", True),  # runs an external program
        ("sort in.txt", False),  # plain sort stays read only
        ("rg --pre sh needle f.sh", True),  # rg preprocessor runs a command
        ("rg --pre=/tmp/x needle .", True),
        ("rg --hostname-bin /tmp/x foo .", True),
        ("rg --pre-glob '*.txt' needle .", False),  # glob filter stays read only
        ("rg needle .", False),  # plain rg stays read only
        ("/tmp/cat secrets", True),  # path-qualified command is an arbitrary binary
        ("./ls -la", True),
        ("env /tmp/cat x", True),  # path-qualified target after a wrapper
        ("tree -o out.txt", True),  # -o writes a file
        ("xxd -r dump.hex out.bin", True),  # -r can write
        ("awk '{print}' file", True),  # awk can system()/write
        ("grep -o x file", False),  # grep -o is stdout only
        ("ls\nrm -rf x", True),  # newline separates commands
        ("ls\r\nrm x", True),  # CRLF separates commands
        ("ls\n\n\nrm x", True),  # blank lines collapse to one separator
        ("ls\npwd", False),  # multi-line stays safe when every line is
        ("ls\n", False),
        ("sort -o/tmp/out /tmp/in", True),  # attached short output flag
        ("sort -uo out.txt in.txt", True),  # -o bundled in a short cluster
        ("sort -bo out in", True),
        ("sort -u in.txt", False),  # cluster without a write flag stays safe
        ("find . \\( -name x -delete \\)", True),  # -delete inside a group
        ("cat ../../.ssh/id_rsa", True),  # parent traversal read
        ("cat ~/.aws/credentials", True),  # credential path
        ("cat /home/a/.azure/msal_token_cache.json", True),  # azure token store
        ("cat ~/.config/gh/hosts.yml", True),  # gh cli credentials
        ("cat ~/.config/app/settings.json", False),  # ordinary config stays safe
        ("cat /run/secrets/hf_token", True),  # docker secret mount
        ("cat /var/run/secrets/kubernetes.io/serviceaccount/token", True),  # k8s mount
        ("cat /run/app.pid", False),  # ordinary /run file stays safe
        ("cat /etc/passwd", True),  # sensitive system file
        ("cat /proc/self/environ", True),  # procfs env dump
        ("cat /proc/1/cmdline", True),
        ("head /proc/self/maps", True),
        ("LD_PRELOAD=/tmp/hook.so ls", True),  # code-loading env prefix
        ("PATH=. ls", True),  # command-lookup env prefix
        ("IFS=x ls", True),
        ("FOO=1 grep -r x .", False),  # benign env prefix stays safe
        ("ps auxe", True),  # ps can dump process env; not on the safe list
        ("ps aux", True),
        ("cd /; cat etc/passwd", True),  # cd escapes the workdir
        ("cd subdir; ls", True),  # cd is no longer auto-approved
        ("env --chdir=/ cat etc/passwd", True),  # env -C escapes the workdir
        ("env -S 'sh -c id' true", True),  # env --split-string builds a command
        ("env FOO=1 grep -r x .", False),  # benign env wrapper stays safe
        ("cat /etc//passwd", True),  # redundant slashes resolve to /etc/passwd
        ("cat /etc/./passwd", True),
        ("p=/etc; cat $p/passwd", True),  # path split across an assignment
        ("d=/etc; cat ${d}/shadow", True),
        ("FOO=1 echo $FOO", False),  # benign variable expansion stays safe
        ("cat /proc/$PPID/enviro''n", True),  # quote-split procfs read
        ("cat /proc/self/'environ'", True),
        ('p="/proc/$PPID"; cat $p/environ', True),  # quoted+nested var procfs
        ("LESSOPEN='|touch x; cat %s' less f.txt", True),  # less input preprocessor
        ("less file.txt", True),  # less pager escapes (+cmd, !shell, -o) so it asks
        ("less '+!touch pwned' notes.txt", True),  # less +command runs a shell command
        ("more file.txt", True),  # more shares the !shell pager escape
        ("cat /proc/cpuinfo", False),  # non-sensitive procfs read stays safe
        ("cat /e??/passwd", True),  # glob expands to /etc/passwd
        ("cat /e[t]c/passwd", True),  # bracket class hides etc
        ("head /etc/shado?", True),
        ("cat /et\\c/passwd", True),  # backslash escape hides /etc/passwd
        ("cat /etc/pass\\wd", True),
        ("ls *.py", False),  # benign glob stays safe
        ("head data?.txt", False),
        ("grep -R TOKEN /home", True),  # recursive search escapes the workdir
        ("rg TOKEN /", True),
        ("fd pattern /etc", True),
        ("grep -r foo src/", False),  # sandbox-relative search stays safe
        ("rg TOKEN .", False),
        ("cat logs/app.log", False),  # ordinary relative read
        ("cat /r?n/secrets/hf_token", True),  # glob into a secret mount
        ("cat /var/r?n/secrets/db", True),
        ("cat /root/.s??/id_rsa", True),  # glob into a credential dir
        ("ls /home/*/projects", False),  # benign glob not into a cred dir
        ("grep -R TOKEN ~root", True),  # tilde-user recursive root escapes
        ("grep -R TOKEN ~/logs", True),  # tilde-home recursive root escapes
        ("cat /etc/pass{w,}d", True),  # brace expansion builds /etc/passwd
        ("cat report{1,2}.txt", False),  # benign brace stays safe
        ("cat /etc/pass${x:-wd}", True),  # default param expansion builds path
        ("cat /etc/pass${x:=wd}", True),
        ("echo ${x:-hello}", False),  # benign default param stays safe
        ("cat </e??/passwd", True),  # redirection prefix hides the glob
        ("cat <../../notes", True),  # redirection with no space escapes workdir
        ("cat notes.txt", False),  # ordinary read stays safe
        ("p=/; grep -R TOKEN $p", True),  # recursive root hidden in an assignment
        ("p=/home; grep -R TOKEN $p", True),
        ("p=src; grep -R TOKEN $p", False),  # relative assigned root stays safe
        ("cat /etc/pass{w..w}d", True),  # sequence brace builds /etc/passwd
        ("cat /etc/pass{v..x}d", True),  # sequence brace range spans passwd
        ("cat file{1..3}.txt", False),  # benign sequence brace stays safe
        ("p=passwd; cat /etc/${p:0:6}", True),  # substring expansion builds path
        ("p=hello; cat notes/${p:0:3}", False),  # benign substring stays safe
        ("cat $'/etc/pass\\x77d'", True),  # ANSI-C escape hides /etc/passwd
        ("cat $'notes.txt'", False),  # benign ANSI-C quote stays safe
        ("cat /home/*/.az?re/msal_token_cache.json", True),  # azure token glob
        ("cat /home/*/.config/g?/hosts.yml", True),  # gh config glob
        ("cat /home/*/projects/readme", False),  # benign home glob stays safe
        ("cat /proc/$PPID/task/$PPID/environ", True),  # per-thread proc env alias
        ("cat /proc/cpuinfo", False),  # non-sensitive proc read stays safe
        ("grep -R TOKEN ${root:-/home}", True),  # default-param recursive root
        ("grep -R TOKEN ${root:-src}", False),  # relative default root stays safe
        ("p=passXd; cat /etc/${p/X/w}", True),  # pattern replacement builds path
        ("p=passXd; cat /etc/${p//X/w}", True),  # global pattern replacement
        ("p=hello; cat notes/${p/l/L}", False),  # benign replacement stays safe
        ("p=PASSWD; cat /etc/${p,,}", True),  # case-lower expansion builds path
        ("p=hello; cat notes/${p,,}", False),  # benign case expansion stays safe
        ("f=-delete; find . $f", True),  # find action hidden behind an assignment
        ("g=e??; cat /$g/passwd", True),  # glob assembled through an assignment
        ("g=abc; cat /$g/readme", False),  # benign assigned path stays safe
        ("cat /etc/pass[[:lower:]]d", True),  # POSIX class glob builds /etc/passwd
        ("x=passwd; p=x; cat /etc/${!p}", True),  # indirect expansion builds path
        ("x=notes; p=x; cat /home/${!p}", False),  # benign indirect expansion stays safe
        ("cat </dev/tcp/example.com/80", True),  # bash /dev/tcp opens a socket
        ("cat < /dev/udp/1.2.3.4/53", True),  # bash /dev/udp opens a socket
        ("cat /dev/null", False),  # ordinary /dev file stays safe
        ("cat /etc/ssh/ssh_host_ed25519_key", True),  # ssh host private key read
        ("cat /etc/ssh/sshd_config", True),  # whole /etc/ssh dir is sensitive
        ("cat /etc/hostname", False),  # non-key /etc read stays safe
        ("sort --out=/tmp/o in", True),  # abbreviated --output writes a file
        ("env --ch=/ cat etc/passwd", True),  # abbreviated --chdir escapes workdir
        ("sort --check in", False),  # benign abbreviation-free long flag stays safe
        ("printf -v PATH %s .; ls", True),  # printf -v rewrites PATH then runs ./ls
        ("printf 'hello %s' world", False),  # ordinary printf stays safe
        ("fd --base-directory=/ passwd etc", True),  # fd root move escapes workdir
        ("fd --search-path=/etc passwd", True),  # fd search-path escapes workdir
        ("fd --base-dir=/ passwd etc", True),  # abbreviated fd root flag too
        ("fd passwd", False),  # in-workdir fd search stays safe
        ("uniq input.txt output.txt", True),  # second positional is a written OUTPUT
        ("uniq -f 2 in out", True),  # numeric flag value skipped, two file positionals
        ("uniq input.txt", False),  # single positional reads to stdout, stays safe
        ("sort a.txt | uniq -c", False),  # piped uniq with no output file stays safe
    ],
)
def test_terminal_classifier(command, unsafe):
    assert is_potentially_unsafe_tool_call("terminal", {"command": command}) is unsafe


@pytest.mark.parametrize(
    ("code", "unsafe"),
    [
        ("print(1+1)", False),
        ("import math\nprint(math.pi)", False),
        ("print(open('x.txt').read())", False),  # read-mode open
        ("open('x.txt', 'w').write('hi')", True),
        ("import shutil; shutil.rmtree('x')", True),
        ("import os; os.remove('x')", True),
        ("import requests", True),  # network module
        ("exec('print(1)')", True),
        ("from os import remove\nremove('x')", True),  # from-import binding
        ("from os import remove as rm\nrm('x')", True),
        ("from os import *", True),  # star import hides anything
        ("import os\nprint(os.getcwd())", False),  # read-only os use
        ("f = os.remove\nf('x')", True),  # indirect reference
        ("import os\nrm = os.remove\nrm('x')", True),  # alias assignment
        ("from pathlib import Path\nPath('x').open('w')", True),  # Path.open mode
        ("from pathlib import Path\nprint(Path('x').open().read())", False),
        ("import zipfile\nprint(zipfile.ZipFile('a').open('n.txt'))", False),
        ("print(open('../../.ssh/id_rsa').read())", True),  # traversal read
        ("print(open('creds.env').read())", True),  # credential file
        ("import os\nos.open('data.txt', os.O_CREAT)", True),  # os.open writes fd
        ("import tempfile\ntempfile.mkstemp()", True),  # tempfile side effects
        ("getattr(os, 'remove')('x')", True),  # dynamic call target
        ("import os as o\no.open('out.txt', o.O_CREAT)", True),  # os.open via alias
        ("from os import open as o, O_CREAT\no('out', O_CREAT)", True),  # os.open bare name
        ("from pathlib import Path\nPath('l').symlink_to('t')", True),  # pathlib link
        ("import importlib\nimportlib.import_module('subprocess')", True),  # dynamic import
        ("import os\nos.mkfifo('p')", True),  # node creation
        ("import os\nos.utime('x', None)", True),  # metadata mutation
        ("f = open\nf('x', 'w')", True),  # builtin open aliased to a name
        ("from builtins import open as w\nw('x', 'w')", True),
        ("globals()['open']('x', 'w')", True),  # dynamic open lookup
        ("import pickle\npickle.loads(b'')", True),  # code exec on load
        ("import io\nio.FileIO('out', 'w')", True),  # raw write handle
        (
            "import zipfile\nprint(zipfile.ZipFile('a').open('n.txt', 'r'))",
            False,
        ),  # explicit read mode
        ("f, _ = (open, print)\nf('out', 'w')", True),  # destructured open alias
        ("import builtins\nbuiltins.exec('x=1')", True),  # attribute exec
        ("import builtins as b\nb.eval('1')", True),
        ("import re\nre.compile('x')", False),  # re.compile is not eval/exec
        ("import os\nopen(os.path.join('/etc', 'passwd')).read()", True),  # composed path
        ("open('/etc' + '/passwd').read()", True),  # concatenated path
        ("import zipfile\nzipfile.ZipFile('o.zip', 'w').writestr('x', 'y')", True),  # zip write
        ("import zipfile\nzipfile.ZipFile('o.zip', mode='a')", True),
        ("import zipfile\nzipfile.ZipFile('a.zip').read('n')", False),  # zip read stays safe
        ("import os\nopen(f'/proc/{os.getppid()}/environ').read()", True),  # f-string procfs
        ("import os\nos.chdir('/')\nprint(open('etc/passwd').read())", True),  # chdir escape
        (
            "from pathlib import Path\nprint((Path('/etc') / 'passwd').read_text())",
            True,
        ),  # pathlib /
        (
            "from pathlib import Path\nprint((Path('a') / 'b.txt').read_text())",
            False,
        ),  # relative stays safe
        ("import runpy\nrunpy.run_path('s.py')", True),  # runpy runs code
        ("from runpy import run_module\nrun_module('m')", True),
        ("import os\nrm = getattr(os, 'remove')\nrm('f')", True),  # getattr alias call
        ("x = getattr(obj, 'name')\nprint(x)", False),  # getattr result not called
        ("__builtins__.exec('x=1')", True),  # __builtins__ dynamic exec
        ("f = globals()['open']\nf('out', 'w')", True),  # subscript alias write
        ("import builtins\nf = builtins.open\nf('out', 'w')", True),  # attribute alias write
        ("open('out', **{'mode': 'w'}).write('x')", True),  # kwargs splat mode
        ("name = 'passwd'\nopen(f'/etc/{name}').read()", True),  # dynamic /etc segment
        ("import os\nopen(os.path.join('/etc', name)).read()", True),  # composed dynamic seg
        ("open(f'/tmp/{name}.txt').read()", False),  # dynamic seg under /tmp stays safe
        ("import pathlib\n(pathlib.Path('/etc') / name).read_text()", True),  # qualified pathlib
        ("import pathlib\n(pathlib.Path('data') / name).read_text()", False),  # relative stays safe
        ("f: object = open\nf('out', 'w').write('x')", True),  # annotated open alias
        ("import urllib3\nurllib3.PoolManager().request('GET', 'http://x')", True),  # network
        ("import dbm\ndbm.open('cache', 'c')", True),  # dbm create flag writes
        ("import dbm\ndbm.open('cache')", True),  # dbm import itself signals writes
        (
            "import sqlite3\nsqlite3.connect('results.db').execute('create table t(x)')",
            True,
        ),  # sqlite3 db write
        ("import sqlite3\nsqlite3.connect('data.db')", True),  # sqlite3 connect creates the file
        ("import posix as p\np.open('out', 64)", True),  # posix.open via module alias
        ("import os as o\nprint(o.getcwd())", False),  # read-only os-alias use stays safe
        ("model.save_pretrained('out')", True),  # transformers/peft persistence helper
        (
            "from safetensors.torch import save_file\nsave_file(sd, 'o.safetensors')",
            True,
        ),  # bare imported save_file writer
        ("st.save_file(sd, 'o.safetensors')", True),  # safetensors save_file method
        ("print(model.state_dict())", False),  # non-persisting call stays safe
        (
            "from pathlib import Path\nopen(next(Path('/etc').glob('passw?'))).read()",
            True,
        ),  # pathlib glob receiver+pattern resolves to /etc/passwd
        (
            "from pathlib import Path\nfor f in Path('data').glob('*.py'):\n    print(f)",
            False,
        ),  # benign pathlib glob stays safe
        (
            "import os\nbase = os.path.abspath('/etc')\nopen(base + '/passwd').read()",
            True,
        ),  # abspath keeps the sensitive root
        (
            "from pathlib import Path\n(Path('/etc').resolve() / 'passwd').read_text()",
            True,
        ),  # Path.resolve keeps the sensitive root
        (
            "import os\nbase = os.path.abspath('data')\nopen(base + '/x.txt').read()",
            False,
        ),  # benign normalizer stays safe
        ("import torch\ntorch.load('model.pt')", True),  # pickle-backed loader
        ("import joblib\njoblib.load('x.pkl')", True),  # joblib loader
        ("import pandas as pd\npd.read_pickle('x.pkl')", True),  # pandas pickle reader
        ("import json\nprint(json.load(open('x.json')))", False),  # json.load stays safe
        (
            "import types\nc = compile('x=1', '', 'exec')\nf = types.FunctionType(c, globals())\nf()",
            True,
        ),  # compiled code wrapped into a callable
        ("cfg = d['k']\nprint(cfg)", False),  # subscript result not called stays safe
        ("open('/etc/{}'.format('passwd')).read()", True),  # str.format sensitive path
        ("open('/etc/{}'.format(name)).read()", True),  # format dynamic /etc segment
        ("print('/tmp/{}'.format('a'))", False),  # format under /tmp stays safe
        ("import numpy\nnumpy.save('x.npy', a)", True),  # numpy writer method
        ("plt.savefig('f.png')", True),  # matplotlib writer method
        ("df.to_csv('out.csv')", True),  # pandas writer method
        ("img.save('o.png')", True),  # PIL writer method
        ("import json\njson.dump(obj, f)", True),  # serialization writer
        ("df.to_string()", False),  # non-persisting render stays safe
        ("model.forward(x)", False),  # ordinary method call stays safe
        ("open(''.join(['/etc', '/passwd'])).read()", True),  # str.join sensitive path
        ("open('/'.join(['/etc', 'passwd'])).read()", True),  # separator join
        ("print(''.join(['a', 'b']))", False),  # benign join stays safe
        ("from builtins import eval as e\ne('1')", True),  # aliased builtin eval
        ("import builtins\nx = builtins.exec\nx('a=1')", True),  # attr-aliased exec
        ("from builtins import __import__ as imp\nimp('os')", True),  # aliased __import__
        ("from mymod import evaluate as e\ne(1)", False),  # unrelated alias stays safe
        ("base = '/etc'\nopen(base + '/passwd').read()", True),  # literal-var path
        ("d = '/etc'\nopen(f'{d}/passwd').read()", True),  # literal var in f-string
        ("base = 'data'\nopen(base + '/x.txt').read()", False),  # benign literal var
        ("import numpy as np\nnp.array([1]).tofile('out.bin')", True),  # numpy tofile
        ("arr.tolist()", False),  # non-persisting numpy call stays safe
        (
            "from pathlib import Path\np = Path('/etc')\n(p / 'passwd').read_text()",
            True,
        ),  # pathlib path alias reused
        (
            "from pathlib import Path\np = Path('data')\n(p / 'x.txt').read_text()",
            False,
        ),  # relative path alias stays safe
        ("open('%s/%s' % ('/etc', 'passwd')).read()", True),  # percent-format path
        ("open('/etc/%s' % name).read()", True),  # percent-format dynamic segment
        ("open('%s/%s' % ('data', 'x.txt')).read()", False),  # benign percent-format
        (
            "from pathlib import Path\nPath('/etc').joinpath('passwd').read_text()",
            True,
        ),  # pathlib joinpath
        (
            "from pathlib import Path\nPath('data').joinpath('x.txt').read_text()",
            False,
        ),  # relative joinpath stays safe
        ("open(b'/etc/passwd').read()", True),  # bytes path literal
        ("open(b'data.txt').read()", False),  # benign bytes literal stays safe
        (
            "from pathlib import Path\n(Path.cwd().parent / 'other' / 'notes').read_text()",
            True,
        ),  # pathlib parent escapes the sandbox
        (
            "from pathlib import Path\n(Path('data') / 'notes').read_text()",
            False,
        ),  # in-sandbox pathlib read stays safe
        ("import glob\nopen(glob.glob('/e??/passwd')[0]).read()", True),  # python glob to secret
        ("import glob\nfor f in glob.glob('*.py'):\n    print(f)", False),  # benign glob stays safe
        (
            "import glob\nbase = '/e??'\nopen(glob.glob(base + '/passwd')[0]).read()",
            True,
        ),  # glob pattern folded from a literal variable
        ("from os.path import join\nopen(join('/etc', 'passwd')).read()", True),  # bare join alias
        ("from os.path import join\nopen(join('data', 'x.txt')).read()", False),  # benign bare join
        ("from numpy import save\nsave('out.npy', arr)", True),  # writer imported as a bare name
        ("from numpy import mean\nmean(arr)", False),  # benign bare import stays safe
        (
            "from pathlib import Path as P\n(P('/etc') / 'passwd').read_text()",
            True,
        ),  # aliased pathlib constructor
        (
            "from pathlib import Path as P\n(P('data') / 'x').read_text()",
            False,
        ),  # aliased ctor with a relative path stays safe
        (
            "from pathlib import PosixPath\n(PosixPath('/etc') / 'passwd').read_text()",
            True,
        ),  # concrete PosixPath constructor is folded too
        (
            "import pathlib\n(pathlib.PosixPath('/etc') / 'passwd').read_text()",
            True,
        ),  # qualified concrete constructor
        (
            "from pathlib import WindowsPath as W\n(W('/etc') / 'passwd').read_text()",
            True,
        ),  # aliased concrete Windows constructor
        (
            "from pathlib import PosixPath\n(PosixPath('data') / 'x').read_text()",
            False,
        ),  # concrete ctor with a relative path stays safe
        (
            "base = '/etc'\nopen(base + '/passwd').read()\nbase = 'data'",
            True,
        ),  # a later reassignment must not mask the earlier sensitive read
        (
            "base = 'data'\nopen(base + '/x').read()\nbase = '/etc'",
            True,
        ),  # any reassignment of a path var fails closed
        (
            "base = 'data'\nopen(base + '/x').read()",
            False,
        ),  # a single benign literal path var stays safe
        (
            "from zipfile import ZipFile\nZipFile('out.zip', 'w')",
            True,
        ),  # bare archive constructor with write mode
        (
            "from tarfile import TarFile as T\nT('a.tar', 'w')",
            True,
        ),  # aliased bare archive constructor
        (
            "from zipfile import ZipFile\nZipFile('in.zip')",
            False,
        ),  # bare archive constructor reading stays safe
        (
            "import os\ng = getattr\nrm = g(os, 'remove')\nrm('file')",
            True,
        ),  # dynamic lookup aliased through a getattr alias
        (
            "import os\ng = getattr\nn = g(os, 'name')\nprint(n)",
            False,
        ),  # resolving (not calling) through a getattr alias stays safe
        (
            "from functools import partial\nw = partial(open, mode='w')\nw('out.txt')",
            True,
        ),  # partial wrapping open hides the write mode
        (
            "import os\nfrom functools import partial\nw = partial(os.remove)\nw('f')",
            True,
        ),  # partial wrapping a mutating callable
        (
            "from functools import partial\np = partial(print, end='')\np('hi')",
            False,
        ),  # partial wrapping a safe callable stays safe
    ],
)
def test_python_classifier(code, unsafe):
    assert is_potentially_unsafe_tool_call("python", {"code": code}) is unsafe


def test_builtin_readonly_tools_are_safe():
    assert is_potentially_unsafe_tool_call("web_search", {"query": "hi"}) is False
    assert is_potentially_unsafe_tool_call("search_knowledge_base", {}) is False
    assert is_potentially_unsafe_tool_call("render_html", {}) is False


def test_unknown_tools_fail_closed():
    assert is_potentially_unsafe_tool_call("mystery_tool", {}) is True


def test_is_always_safe_tool():
    from core.inference.tools import is_always_safe_tool
    for name in ("web_search", "search_knowledge_base", "render_html"):
        assert is_always_safe_tool(name) is True
    for name in ("python", "terminal", "mystery_tool", "mcp__srv__read"):
        assert is_always_safe_tool(name) is False


@pytest.mark.parametrize(
    ("tool", "unsafe"),
    [
        ("get_weather", False),
        ("list_files", False),
        ("search", False),
        ("send_email", True),
        ("create_issue", True),
        ("delete_row", True),
        ("get_or_create_issue", True),  # mutating verb overrides read prefix
        ("read_and_delete_file", True),
        ("find_and_update_row", True),
        ("get_and_commit_changes", True),  # commit/save/archive are mutating
        ("read_and_save_file", True),
        ("list_and_archive", True),
        ("list_and_clone_repo", True),  # clone/checkout/comment are mutating
        ("fetch_and_comment_issue", True),
        ("get_and_checkout_branch", True),
        ("read_and_append_file", True),  # append/prepend are mutating
        ("prepend_line", True),
        ("get_and_upsert_row", True),  # upsert/assign are mutating
        ("list_and_assign_issue", True),
        ("read_and_copy_file", True),  # copy-style verbs create/overwrite state
        ("get_and_copy_resource", True),
        ("read_and_duplicate_entry", True),
        ("fetch_and_download_asset", True),  # download writes local state
        ("list_and_export_data", True),  # import/export/backup/restore/snapshot
        ("get_and_snapshot_volume", True),
        ("read_report", False),  # plain read stays safe
    ],
)
def test_mcp_classifier(tool, unsafe):
    name = f"{MCP_TOOL_PREFIX}srv1__{tool}"
    assert is_potentially_unsafe_tool_call(name, {}) is unsafe


@pytest.mark.parametrize(
    ("args", "unsafe"),
    [
        ({"path": "/etc/passwd"}, True),  # read-named tool at a credential path
        ({"path": "../../.ssh/id_rsa"}, True),
        ({"nested": {"file": "~/.aws/credentials"}}, True),
        ({"path": "notes.txt"}, False),  # ordinary path stays safe
        ({"path": "data/report.csv"}, False),
    ],
)
def test_mcp_sensitive_arguments(args, unsafe):
    name = f"{MCP_TOOL_PREFIX}fs__read_file"
    assert is_potentially_unsafe_tool_call(name, args) is unsafe


@pytest.mark.parametrize(
    ("args", "unsafe"),
    [
        ({"query": "DELETE FROM runs"}, True),  # read-named tool, mutating query
        ({"sql": "DROP TABLE users"}, True),
        ({"query": "UPDATE t SET x=1"}, True),
        ({"query": "INSERT INTO t VALUES (1)"}, True),
        ({"query": "SELECT * FROM runs"}, False),  # read query stays safe
        ({"query": "how to delete old files"}, False),  # NL text with 'delete' stays safe
        ({"query": "find the created_at column"}, False),  # 'created' substring stays safe
        ({"query": "DELETE/**/FROM runs"}, True),  # inline SQL comment as whitespace
        ({"query": "UPDATE/**/t SET x=1"}, True),
        ({"query": "DROP/**/TABLE users"}, True),
        ({"query": "SELECT * FROM runs -- delete later"}, False),  # trailing comment stays safe
    ],
)
def test_mcp_mutating_arguments(args, unsafe):
    name = f"{MCP_TOOL_PREFIX}db__query_database"
    assert is_potentially_unsafe_tool_call(name, args) is unsafe


# ── loop behavior ───────────────────────────────────────────────────

_DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "python"}},
    {"type": "function", "function": {"name": "web_search"}},
]


class _FakeExecuteTool:
    def __init__(self):
        self.calls = []
        self.disable_sandbox_seen = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        self.disable_sandbox_seen.append(disable_sandbox)
        return f"RESULT[{name}]"


def _tool_call(name, args_json):
    return f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'


def _multi_turn(turns):
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            yield next(turn_iter)
        except StopIteration:
            return

    return _gen


def _drive(turns, decisions, **loop_kwargs):
    """Run the loop, resolving each gated tool_start with the next decision."""
    decision_iter = iter(decisions)
    exec_fn = _FakeExecuteTool()
    gen = run_safetensors_tool_loop(
        single_turn = _multi_turn(turns),
        messages = [{"role": "user", "content": "hi"}],
        tools = _DEFAULT_TOOLS,
        execute_tool = exec_fn,
        session_id = _SESSION,
        **loop_kwargs,
    )
    events = []
    for ev in gen:
        events.append(ev)
        if ev["type"] == "tool_start" and ev.get("awaiting_confirmation"):
            resolve_tool_decision(ev["approval_id"], next(decision_iter), session_id = _SESSION)
    return events, exec_fn


def _tool_starts(events):
    return [e for e in events if e["type"] == "tool_start"]


def test_auto_mode_does_not_gate_safe_calls():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        [],
        confirm_tool_calls = True,
        permission_mode = "auto",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""
    assert exec_fn.calls == [("python", {"code": "print(1)"})]
    assert exec_fn.disable_sandbox_seen == [False]  # sandbox stays on in auto


def test_auto_mode_gates_unsafe_calls():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
        permission_mode = "auto",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True
    assert starts[0]["approval_id"]
    assert len(exec_fn.calls) == 1
    assert exec_fn.disable_sandbox_seen == [False]


def test_ask_mode_gates_even_safe_calls():
    events, _ = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
        permission_mode = "ask",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True


def test_unset_mode_behaves_as_ask():
    events, _ = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True


def test_off_mode_never_gates_and_keeps_sandbox():
    # "Off": no prompts even for unsafe calls, but the sandbox stays on.
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        [],
        confirm_tool_calls = True,  # off must win over a stray confirm flag
        permission_mode = "off",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""
    assert exec_fn.disable_sandbox_seen == [False]


def test_full_mode_never_gates_and_drops_sandbox():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        [],
        confirm_tool_calls = True,  # full must win over the confirm gate
        permission_mode = "full",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert exec_fn.disable_sandbox_seen == [True]


def test_bypass_flag_implies_full_mode():
    # Legacy callers that only set bypass_permissions keep the same behavior.
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        [],
        confirm_tool_calls = True,
        bypass_permissions = True,
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert exec_fn.disable_sandbox_seen == [True]


def test_bypass_permissions_folds_to_full_on_request_models():
    # A legacy bypass caller that also sends a stale ask/auto mode normalizes to
    # full, so the route guards (which reject ask/auto) don't 400 the request.
    for cls in (ChatCompletionRequest, AnthropicMessagesRequest):
        req = cls(
            messages = [{"role": "user", "content": "hi"}],
            bypass_permissions = True,
            permission_mode = "auto",
        )
        assert req.permission_mode == "full"
        assert req.bypass_permissions is True


def test_ask_auto_self_enable_confirm_on_chat_request():
    # A direct /chat/completions caller that requests ask/auto but omits the
    # legacy confirm flag must still hit the confirmation gate when Studio's own
    # tool loop is requested. Only the router's own loop-entry signals count
    # (enable_tools / mcp_enabled); enabled_tools alone never starts the loop.
    for mode in ("ask", "auto"):
        for loop in (
            {"enable_tools": True},
            {"mcp_enabled": True},
        ):
            req = ChatCompletionRequest(
                messages = [{"role": "user", "content": "hi"}],
                permission_mode = mode,
                **loop,
            )
            assert req.confirm_tool_calls is True
    # enabled_tools by itself is a passthrough filter, not a loop-entry signal:
    # a client-tool passthrough that also lists enabled_tools must route verbatim
    # (confirm stays unset), else the confirm-without-stream guard 400s it.
    for mode in ("ask", "auto"):
        req = ChatCompletionRequest(
            messages = [{"role": "user", "content": "hi"}],
            permission_mode = mode,
            enabled_tools = ["terminal"],
            tools = [{"type": "function", "function": {"name": "f"}}],
        )
        assert req.confirm_tool_calls is None
    # A contradictory confirm=False is overridden by the explicit mode.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        permission_mode = "ask",
        enable_tools = True,
        confirm_tool_calls = False,
    )
    assert req.confirm_tool_calls is True
    # A plain client-tool passthrough (client-supplied tools that Studio does not
    # execute) must NOT self-enable confirm, or the route rejects the passthrough.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        permission_mode = "ask",
        tools = [{"type": "function", "function": {"name": "f"}}],
    )
    assert req.confirm_tool_calls is None
    # ask/auto without any tool request has nothing to gate; confirm stays unset.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        permission_mode = "ask",
    )
    assert req.confirm_tool_calls is None
    # Legacy callers with no permission_mode keep their confirm flag untouched.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        confirm_tool_calls = False,
    )
    assert req.confirm_tool_calls is False
    # External-provider requests are not folded (the provider branch rejects
    # confirm_tool_calls with tools, and permission_mode is a local concept).
    for extra in ({"provider_id": "p1"}, {"provider_type": "openai"}):
        req = ChatCompletionRequest(
            messages = [{"role": "user", "content": "hi"}],
            permission_mode = "ask",
            enable_tools = True,
            **extra,
        )
        assert req.confirm_tool_calls is None


def test_permission_mode_confirm_derivation():
    # The route derives the effective confirm gate from permission_mode so that a
    # tool loop forced on by CLI policy (no request-level tool flag) still honors
    # the documented "unset behaves as ask" default.
    from routes.inference import _permission_mode_confirm

    def req(**kw):
        return ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}], **kw)

    # An explicit confirm flag always wins (True gates, False opts out).
    assert _permission_mode_confirm(req(confirm_tool_calls = True, stream = False)) is True
    assert _permission_mode_confirm(req(confirm_tool_calls = False, permission_mode = "ask")) is False
    # Explicit ask/auto always engage the gate (a non-streaming one is rejected
    # by the guard that reads this).
    assert _permission_mode_confirm(req(permission_mode = "ask", stream = False)) is True
    assert _permission_mode_confirm(req(permission_mode = "auto", stream = False)) is True
    # off/full never prompt.
    assert _permission_mode_confirm(req(permission_mode = "off")) is False
    assert _permission_mode_confirm(req(permission_mode = "full")) is False
    # An unset mode defaults to ask, but only realizably on a streaming request;
    # a non-streaming unset request keeps the legacy run-without-gate behavior.
    assert _permission_mode_confirm(req(stream = True)) is True
    assert _permission_mode_confirm(req(stream = False)) is False
