#!/usr/bin/env sh
# Mimir installer: fetches the latest prebuilt binary from GitHub releases.
#
#   curl -fsSL https://raw.githubusercontent.com/MakerViking/mimir/main/install.sh | sh
#
# Overrides:
#   MIMIR_REPO=owner/name   release repository
#   MIMIR_BIN_DIR=~/bin     install location (default ~/.local/bin)
#   MIMIR_VERSION=v0.1.0    pin a version (default: latest)
#   MIMIR_BASE_URL=...      release base URL (testing; see .github/workflows/ci.yml)
set -eu

REPO="${MIMIR_REPO:-MakerViking/mimir}"
BIN_DIR="${MIMIR_BIN_DIR:-$HOME/.local/bin}"
VERSION="${MIMIR_VERSION:-latest}"
RELEASES="${MIMIR_BASE_URL:-https://github.com/$REPO/releases}"

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

# SHA-256 of a file, on stdout, using whatever the platform actually has.
#
# `sha256sum` is GNU coreutils and is NOT present on a stock macOS, which ships
# `shasum` (Perl) and `openssl`. An installer that assumes coreutils verifies
# nothing on a Mac — it aborts under `set -e` before it can, which is a broken
# install rather than a refused one. Order is preference, not capability: all
# three compute the same digest.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$1" | awk '{print $NF}'
  else
    echo "No SHA-256 tool found (looked for sha256sum, shasum, openssl)." >&2
    echo "Install one, or build from source: cargo install mimir-mem" >&2
    return 1
  fi
}

if [ "$VERSION" = "latest" ]; then
  base="$RELEASES/latest/download"
else
  base="$RELEASES/download/$VERSION"
fi
tarball="mimir-$os-$arch.tar.gz"
url="$base/$tarball"

echo "Installing mimir ($os-$arch) to $BIN_DIR ..."
mkdir -p "$BIN_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/$tarball"

# Verify against the release's SHA256SUMS, fail-closed.
#
# Deliberately an error and not a warning when the file is missing: an attacker
# who can strip one response can strip this one, and an installer that shrugs
# and continues is an installer with no integrity check at all. Releases
# published before checksums existed genuinely do not have it — that is a
# release to re-cut, not a check to skip, so the message says so rather than
# leaving a bare `curl: (22)` behind.
if ! curl -fsL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS" 2>/dev/null; then
  echo >&2
  echo "No SHA256SUMS published for this release, so the download cannot be verified." >&2
  echo "Refusing to install unverified binaries." >&2
  echo >&2
  echo "Install from source instead:" >&2
  echo "  cargo install mimir-mem" >&2
  exit 1
fi

# The workflow builds this with `find . | xargs sha256sum`, so names carry a
# leading `./`; match the basename at a word boundary rather than anchoring.
expected="$(grep "[ /]$tarball\$" "$tmp/SHA256SUMS" | cut -d' ' -f1 | head -1)"
if [ -z "$expected" ]; then
  echo "SHA256SUMS has no entry for $tarball — refusing to install." >&2
  exit 1
fi
actual="$(sha256_of "$tmp/$tarball")"
if [ "$expected" != "$actual" ]; then
  echo >&2
  echo "CHECKSUM MISMATCH for $tarball — refusing to install." >&2
  echo "  expected: $expected" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi

tar -xzf "$tmp/$tarball" -C "$tmp"
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
