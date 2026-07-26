#!/bin/sh
# anvil — Forge anything.
# install:  curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh
set -e

VERSION="0.2.0"
REPO="https://github.com/anvil-llm/anvil"
INSTALL_DIR="/usr/local/bin"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[0;33m'; C='\033[0;36m'; A='\033[0;33m'; N='\033[0m'
info(){ printf "${C}▸${N} %s\n" "$1"; }
ok(){   printf "${G}✓${N} %s\n" "$1"; }
warn(){ printf "${Y}⚠${N} %s\n" "$1"; }
die(){  printf "${R}✗ %s${N}\n" "$1"; exit 1; }

printf "${A}"
cat <<'EOF'
   ░███                          ░██░██ 
  ░██░██                            ░██ 
 ░██  ░██  ░████████  ░██    ░██ ░██░██ 
░█████████ ░██    ░██ ░██    ░██ ░██░██ 
░██    ░██ ░██    ░██  ░██  ░██  ░██░██ 
░██    ░██ ░██    ░██   ░██░██   ░██░██ 
░██    ░██ ░██    ░██    ░███    ░██░██
EOF
printf "${N}\n  anvil installer v%s\n\n" "$VERSION"

# ── detect platform ──
OS=$(uname -s); ARCH=$(uname -m)
case "$OS" in
  Linux*)  OSN=linux ;;
  Darwin*) OSN=macos ;;
  *) die "unsupported OS: $OS" ;;
esac
case "$ARCH" in
  x86_64|amd64)   ARN=x86_64 ;;
  aarch64|arm64)  ARN=aarch64 ;;
  *) die "unsupported arch: $ARCH" ;;
esac
info "detected: $OSN / $ARN"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── 1. download the prebuilt binary (built by GitHub Actions) ──
try_binary(){
  URL="$REPO/releases/download/v$VERSION/anvil-$OSN-$ARN"
  info "downloading prebuilt binary…"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$TMP/anvil" "$URL" 2>/dev/null || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$TMP/anvil" "$URL" 2>/dev/null || return 1
  else return 1; fi
  chmod +x "$TMP/anvil"
  # Test-run it. If a required lib is missing, the loader aborts with a
  # non-zero exit BEFORE main() — we catch it here and fall back to building.
  if "$TMP/anvil" --version >/dev/null 2>&1; then
    return 0
  fi
  warn "prebuilt binary wouldn't run on this machine"
  return 1
}

# ── 2. fallback: build from source (rare) ──
build_source(){
  info "building from source (few minutes)…"
  command -v git   >/dev/null 2>&1 || die "git is required"
  command -v cmake >/dev/null 2>&1 || die "cmake is required"
  if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    die "g++ or clang++ is required"
  fi
  info "cloning anvil (with submodules)…"
  git clone --recursive --depth 1 --shallow-submodules "$REPO" "$TMP/anvil-src" \
    || die "failed to clone anvil"
  info "compiling…"
  cmake -B "$TMP/anvil-src/build" -S "$TMP/anvil-src" \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$TMP/anvil-src/build" -j"$NPROC" >/dev/null || die "build failed"
  cp "$TMP/anvil-src/build/anvil" "$TMP/anvil"
  chmod +x "$TMP/anvil"
  ok "build complete"
}

if try_binary; then ok "got prebuilt binary"; else build_source; fi

# ── install ──
info "installing…"
if [ -w "$INSTALL_DIR" ]; then
  mv "$TMP/anvil" "$INSTALL_DIR/anvil"
elif command -v sudo >/dev/null 2>&1; then
  sudo mv "$TMP/anvil" "$INSTALL_DIR/anvil"
else
  INSTALL_DIR="$HOME/.local/bin"; mkdir -p "$INSTALL_DIR"
  mv "$TMP/anvil" "$INSTALL_DIR/anvil"
fi
ok "installed to $INSTALL_DIR/anvil"

mkdir -p "$HOME/.anvil"
ok "config dir: $HOME/.anvil"

if command -v anvil >/dev/null 2>&1; then
  printf "\n"; ok "anvil is ready. go forge something."
  printf "\n    anvil run model.gguf\n    anvil --help\n\n"
else
  warn "$INSTALL_DIR isn't in your PATH. add this to your shell profile:"
  printf "    export PATH=\"%s:\$PATH\"\n" "$INSTALL_DIR"
fi
