import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
env = dict(os.environ)
for key, folder in {
    'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    'UV_CACHE_DIR': 'uv-cache', 'HF_HOME': 'hf',
    'XDG_CACHE_HOME': 'cache', 'UNSLOTH_STUDIO_HOME': 'studio-home',
    'PLAYWRIGHT_BROWSERS_PATH': 'browsers', 'SE_CACHE_PATH': 'selenium',
    'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
    'HYPOTHESIS_STORAGE_DIRECTORY': 'hypothesis',
    'npm_config_cache': 'npm-cache',
}.items():
    path = root / folder
    path.mkdir(exist_ok=True)
    env[key] = str(path)
env.update(UNSLOTH_ALLOW_CPU='1', UNSLOTH_IS_PRESENT='1',
           UNSLOTH_DIFFUSION_ATTENTION_INSTALL='0', UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE='1',
           UNSLOTH_SETTLE_DELAY_S='0', PYTHONDONTWRITEBYTECODE='1')
sys.exit(subprocess.call(sys.argv[1:], cwd=root.parent, env=env))
