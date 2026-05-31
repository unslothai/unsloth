import subprocess
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("curl")
    .run_commands(
        "curl -fsSL https://unsloth.ai/install.sh | sh",
        "echo 'export PATH=$PATH:$HOME/.local/bin:/root/.local/bin' >> /etc/environment",
    )
)

app = modal.App()


@app.function(
    gpu="H100",
    image=image,
    timeout=3600,
)
def run_jupyter():
    import os
    os.environ["PATH"] += ":/root/.local/bin:/usr/local/bin"

    subprocess.run(
        "/root/.unsloth/studio/unsloth_studio/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121",
        shell=True,
        env=os.environ,
    )

    with modal.forward(8888) as tunnel:
        print(f"Jupyter URL: {tunnel.url}")
        subprocess.run(
            "unsloth studio -p 8888 -H 0.0.0.0",
            shell=True,
            env=os.environ,
        )

@app.local_entrypoint()
def main():
    run_jupyter.remote()