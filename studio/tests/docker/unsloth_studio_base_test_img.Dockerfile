# tests/docker/base.Dockerfile
FROM ubuntu:26.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools required by the script's logic
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    coreutils \
	python-is-python3 \
	cmake \
	libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for testing (to mimic real user environments)
RUN useradd -m tester
USER tester
WORKDIR /home/tester

# Copy the setup script and the repo into the container
# (Assuming the test runner is executed from the repo root)
COPY --chown=tester:tester . /home/tester/

# Make the script executable
RUN chmod +x /home/tester/studio/setup.sh
RUN bash /home/tester/install.sh --local

CMD ["/bin/bash"]
