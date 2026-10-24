"""Download the CP validation checkpoints from ModelScope (32 parallel HTTP range requests per file).

usage: MODELS_DIR=<dir> python download_models.py [repo-name-fragment ...]
With no arguments every repo in REPOS is fetched. Files already present with the right size are skipped,
so an interrupted run can be restarted.
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS_DIR = os.environ.get('MODELS_DIR')
if not MODELS_DIR:
    sys.exit('set MODELS_DIR to the directory the checkpoints should be downloaded into')
# (ModelScope repo id, local directory name). These are the checkpoints the CP results were measured on.
REPOS = [
    (repo, os.path.join(MODELS_DIR, name))
    for repo, name in [
        ('unsloth/gemma-4-E2B-it', 'gemma-4-E2B-it'),
        ('unsloth/gemma-4-E4B-it', 'gemma-4-E4B-it'),
        ('google/gemma-4-26B-A4B-it', 'gemma-4-26B-A4B-it'),
        ('google/gemma-4-31B-it', 'gemma-4-31B-it'),
        ('deepseek-ai/DeepSeek-V2-Lite', 'DeepSeek-V2-Lite'),
        ('moonshotai/Moonlight-16B-A3B', 'Moonlight-16B-A3B'),
        ('moonshotai/Kimi-Linear-48B-A3B-Instruct', 'Kimi-Linear-48B-A3B-Instruct'),
        ('inference-optimization/Kimi-K3-0.40B', 'Kimi-K3-0.40B'),
    ]
]

CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB per range task
WORKERS = 32

def fetch_and_pwrite(url, fd, start, end):
    cur = start
    while cur <= end:
        req = urllib.request.Request(url, headers={'Range': f'bytes={cur}-{end}'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    os.pwrite(fd, buf, cur)
                    cur += len(buf)
        except Exception:
            time.sleep(0.5)
    return end - start + 1

def download_repo(repo_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    api_url = f'https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files'
    data = json.loads(urllib.request.urlopen(api_url, timeout=20).read().decode())
    files = [
        f for f in data['Data']['Files']
        if f['Type'] == 'blob' and not f['Path'].startswith(('figures/', '.git'))
    ]
    total_bytes = sum(f.get('Size', 0) for f in files)
    print(f'=== Downloading {repo_id} -> {out_dir} ({len(files)} files, {total_bytes/1024**3:.2f} GiB) ===', flush=True)
    t_repo = time.time()
    done_bytes = 0

    for f in files:
        rel_path = f['Path']
        size = f.get('Size', 0)
        dst = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == size and size > 0:
            done_bytes += size
            print(f'  [skip] {rel_path} ({size/1024**2:.1f} MiB)', flush=True)
            continue

        url = f'https://www.modelscope.cn/models/{repo_id}/resolve/master/{rel_path}'
        t0 = time.time()
        if size <= CHUNK_SIZE:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, 'wb') as out_f:
                out_f.write(resp.read())
            done_bytes += size
            print(f'  [ok] {rel_path} ({size/1024**2:.2f} MiB in {time.time()-t0:.1f}s)', flush=True)
        else:
            tmp_dst = dst + '.part'
            fd = os.open(tmp_dst, os.O_CREAT | os.O_WRONLY, 0o644)
            os.ftruncate(fd, size)
            spans = []
            for s in range(0, size, CHUNK_SIZE):
                e = min(s + CHUNK_SIZE - 1, size - 1)
                spans.append((s, e))
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(fetch_and_pwrite, url, fd, s, e) for s, e in spans]
                for fut in as_completed(futs):
                    fut.result()
            os.close(fd)
            os.replace(tmp_dst, dst)
            done_bytes += size
            dt = max(time.time() - t0, 1e-3)
            speed = (size / 1024**2) / dt
            print(f'  [ok] {rel_path} ({size/1024**2:.1f} MiB in {dt:.1f}s, {speed:.1f} MiB/s) | repo {done_bytes/1024**3:.2f}/{total_bytes/1024**3:.2f} GiB', flush=True)

    dt_repo = max(time.time() - t_repo, 1e-3)
    print(f'=== Finished {repo_id} in {dt_repo:.1f}s ({(total_bytes/1024**2)/dt_repo:.1f} MiB/s avg) ===\n', flush=True)

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else [r[0] for r in REPOS]
    for repo_id, out_dir in REPOS:
        if any(t in repo_id for t in targets):
            download_repo(repo_id, out_dir)
