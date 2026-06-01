import subprocess
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("curl", "bubblewrap", "git")
    .run_commands(
        "git clone https://github.com/NilayYadav/unsloth.git /root/unsloth",
        # Build on a GPU so install.sh detects CUDA and installs the matching
        # GPU torch (the version studio is built for) instead of CPU wheels.
        # Let install.sh install the torch Unsloth expects (torch>=2.6, which the
        # studio trainer needs). Building on a GPU lets it resolve CUDA wheels.
        "cd /root/unsloth && git checkout cloud-deploy && sh install.sh --local --python 3.11",
        "/root/.unsloth/studio/unsloth_studio/bin/pip install --upgrade transformers",                                                    
        "echo 'export PATH=$PATH:$HOME/.local/bin:/root/.local/bin' >> /etc/environment",
        gpu="H100",
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
    # install.sh's editable overlay leaves `import unsloth` resolving to an empty
    # namespace ("unknown location"). Point PYTHONPATH at the cloned repo so the
    # real unsloth package (with FastLanguageModel) is imported instead.
    os.environ["PYTHONPATH"] = "/root/unsloth" + (
        ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
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