#!/usr/bin/env sh
# Mimir installer: fetches the latest prebuilt binary from GitHub releases.
#
#   curl -fsSL https://raw.githubusercontent.com/MakerViking/mimir/main/install.sh | sh
#
# Overrides:
#   MIMIR_REPO=owner/name   release repository
#   MIMIR_BIN_DIR=~/bin     install location (default ~/.local/bin)
#   MIMIR_VERSION=v0.1.0    pin a version (default: latest)
set -eu

REPO="${MIMIR_REPO:-MakerViking/mimir}"
BIN_DIR="${MIMIR_BIN_DIR:-$HOME/.local/bin}"
VERSION="${MIMIR_VERSION:-latest}"

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  *)
    echo "Unsupported OS: $(uname -s)." >&2
    echo "On Windows, download the zip from https://github.com/$REPO/releases" >&2
    exit 1
    ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch=x86_64 ;;
  aarch64 | arm64) arch=aarch64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
if [ "$os" = "macos" ] && [ "$arch" = "x86_64" ]; then
  echo "Intel Macs are not supported: onnxruntime no longer ships x86_64 macOS binaries." >&2
  exit 1
fi

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/mimir-$os-$arch.tar.gz"
else
  url="https://github.com/$REPO/releases/download/$VERSION/mimir-$os-$arch.tar.gz"
fi

echo "Installing mimir ($os-$arch) to $BIN_DIR ..."
mkdir -p "$BIN_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/mimir.tar.gz"
tar -xzf "$tmp/mimir.tar.gz" -C "$tmp"
install -m 755 "$tmp/mimir" "$BIN_DIR/mimir"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "NOTE: $BIN_DIR is not on your PATH — add it to your shell profile." ;;
esac

echo
echo "Installed: $("$BIN_DIR/mimir" --version)"
echo
echo "Get started:"
echo "  mimir init                                        # config + db + embedding model (~34 MB)"
echo "  mimir remember \"my first memory\" -t note"
echo "  mimir recall first"
echo
echo "Using Claude Code? Register the MCP server once, globally:"
echo "  claude mcp add --scope user mimir -- mimir mcp"
