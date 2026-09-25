# Building Unsloth/llama.cpp for Intel Arc GPUs

This guide explains how to properly build the SYCL backend for `llama.cpp` to run Unsloth Studio natively on Intel Arc graphics cards (e.g., Arc A-Series, B60, B580, etc.).

## Prerequisites

1. **Linux Kernel 7.0+**: Essential for the `xe` Intel graphics driver.
2. **Intel Compute Runtime**: 
   Install the latest `intel-compute-runtime` and `level-zero-loader`.
3. **Level Zero Development Headers**:
   Required for maximum performance and hardware features.
   ```bash
   sudo pacman -S level-zero-headers  # On Arch
   # OR build from source:
   git clone https://github.com/oneapi-src/level-zero.git
   cd level-zero
   cmake -B build -DCMAKE_INSTALL_PREFIX=~/.local
   cmake --build build --target install
   ```
4. **Intel oneAPI Base Toolkit (2025.3+)**:
   Download the offline installer (`.sh`) from Intel's website and install it.
   ```bash
   sh ./intel-oneapi-base-toolkit-2025.3.x_offline.sh -a --silent --eula accept
   ```

## Build Instructions

### 1. Load the Environment
Before compiling, you **must** load the Intel DPC++/C++ compiler (`icx`/`icpx`) into your shell environment:
```bash
source /opt/intel/oneapi/setvars.sh
# Or if installed locally:
# source ~/intel/oneapi/setvars.sh
```

### 2. Configure CMake
Run the following inside the `llama.cpp` directory to configure the build. 
*Note: If you installed Level Zero headers locally (like in `~/.local`), prepend `export CPLUS_INCLUDE_PATH=~/.local/include:$CPLUS_INCLUDE_PATH` and add `-DCMAKE_PREFIX_PATH=~/.local`.*

```bash
cmake -B build \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
```
* **`-DGGML_SYCL=ON`**: Enables the SYCL backend for Intel GPUs.
* **`-DGGML_SYCL_F16=ON`**: Highly recommended for Intel Arc! Offloads 16-bit float math directly to the GPU's XMX engines, dramatically increasing generation speed.

### 3. Compile
Build the project using parallel jobs:
```bash
cmake --build build -j4
```
* **`-j4`**: Tells the compiler to use 4 parallel threads/jobs. This speeds up compilation substantially. You can change `4` to the number of CPU cores you have (e.g., `-j8`).

### 4. Install for Unsloth Studio (Wrapper Script)
Unsloth Studio launches `llama-server` in a clean environment, meaning it won't be able to find Intel's math libraries (like `libsvml.so`) automatically. To fix this, we create a wrapper script.

First, copy your compiled binary as `.bin`:
```bash
mkdir -p ~/.local/bin
cp build/bin/llama-server ~/.local/bin/llama-server.bin
```

Then, create a wrapper script at `~/.local/bin/llama-server`:
```bash
cat << 'EOF' > ~/.local/bin/llama-server
#!/bin/bash
# Source Intel oneAPI environment dynamically
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
# (Use source ~/intel/oneapi/setvars.sh if installed locally)

exec ~/.local/bin/llama-server.bin "$@"
EOF
chmod +x ~/.local/bin/llama-server
```

Because `~/.local/bin` takes precedence in your `PATH`, Unsloth Studio will automatically use this wrapper script to load the correct environment and execute your optimized SYCL build!
