#!/usr/bin/env sh
# Build and install Mimir with GPU acceleration from a source checkout.
# Handles the two GPU-build chores automatically: the linker workaround
# (via .cargo/config.toml) and placing libwebgpu_dawn.so next to the
# binary (found via the $ORIGIN rpath).
#
# Usage, from the repo root:
#   ./scripts/setup-gpu.sh            # WebGPU: Vulkan / D3D12 / Metal (AMD, Intel, Apple)
#   ./scripts/setup-gpu.sh cuda      # NVIDIA CUDA
set -eu

backend="${1:-webgpu}"
case "$backend" in
  webgpu | cuda) ;;
  *) echo "usage: $0 [webgpu|cuda]" >&2; exit 1 ;;
esac

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust is required for GPU builds: https://rustup.rs" >&2
  exit 1
fi

echo "Building mimir with --features gpu-$backend (first build downloads ~200-700 MB of onnxruntime)..."
cargo install --path crates/mimir-cli --features "gpu-$backend" --force

bin_dir="$(dirname "$(command -v mimir || echo "$HOME/.cargo/bin/mimir")")"

if [ "$backend" = "webgpu" ]; then
  # Dawn is dynamically linked; the binary's rpath looks next to itself.
  dawn="$(find "${ORT_CACHE:-$HOME/.cache/ort.pyke.io}" -name 'libwebgpu_dawn.*' 2>/dev/null | head -1 || true)"
  if [ -n "$dawn" ]; then
    cp "$dawn" "$bin_dir/"
    echo "Copied $(basename "$dawn") next to the binary."
  else
    echo "WARNING: libwebgpu_dawn not found in the ort cache; if mimir fails to start, see README." >&2
  fi
fi

echo
mimir doctor || true
echo
echo "Done. Verify GPU is active with: RUST_LOG=mimir=info mimir embed"
