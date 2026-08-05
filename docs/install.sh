#!/bin/sh
set -e
REPO="anvil-llm/anvil"
REPO_URL="https://github.com/${REPO}"
API_URL="https://api.github.com/repos/${REPO}/releases"
DEFAULT_DIR="/usr/local/bin"
log() { printf "%s\n" "$1"; }
err() { printf "error: %s\n" "$1" >&2; }
die() { err "$1"; exit 1; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
is_tty() { [ -t 0 ] && [ -t 1 ]; }
cleanup() { rm -rf "${TMP_BASE:-}"; exit 1; }
ensure_target_writable() {
_dir="$1"
if ! mkdir -p "$_dir" 2>/dev/null && has_cmd sudo; then sudo -n mkdir -p "$_dir" 2>/dev/null || true; fi
if [ -w "$_dir" ]; then PRIV=""; return 0; fi
if has_cmd sudo && sudo -n true 2>/dev/null; then PRIV="sudo "; return 0; fi
return 1
}
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
Linux*) OSN=linux ;;
Darwin*) OSN=macos ;;
*) die "unsupported OS: $OS" ;;
esac
case "$ARCH" in
x86_64|amd64) ARN=x86_64 ;;
aarch64|arm64) ARN=aarch64 ;;
*) die "unsupported architecture: $ARCH" ;;
esac
ASSET="anvil-${OSN}-${ARN}"

# NVIDIA CUDA prebuilt (Linux x86_64 only). Detection is additive: the plain
# CPU/Vulkan asset always remains the fallback if the CUDA one is missing.
NVIDIA=0
CUDA_ASSET=""
detect_nvidia_gpu() {
    if has_cmd lspci; then
        if lspci -nn 2>/dev/null | grep -qi '\[10de:'; then NVIDIA=1; return; fi
        if lspci 2>/dev/null | grep -qi nvidia; then NVIDIA=1; return; fi
    fi
    for _v in /sys/bus/pci/devices/*/vendor; do
        [ -r "$_v" ] || continue
        if [ "$(cat "$_v" 2>/dev/null)" = "0x10de" ]; then NVIDIA=1; return; fi
    done
}
if [ "$OSN" = "linux" ] && [ "$ARN" = "x86_64" ]; then
    detect_nvidia_gpu || true
fi
TMPDIR=${TMPDIR:-/tmp}
TMP_BASE="${TMPDIR}/anvil-install-$$"
mkdir -p "$TMP_BASE"
trap cleanup INT TERM
trap 'rm -rf "$TMP_BASE"' EXIT
http_get_stdout() { if has_cmd curl; then curl -fsSL "$1"; elif has_cmd wget; then wget -qO- "$1"; else die "curl or wget is required"; fi; }
http_get_file() { _url=$1; _out=$2; if has_cmd curl; then curl -fsSL -o "$_out" "$_url"; elif has_cmd wget; then wget -q -O "$_out" "$_url"; else die "curl or wget is required"; fi; }
resolve_tag() {
if [ -n "$ANVIL_VERSION" ]; then TAG=$ANVIL_VERSION; log "Using requested version: $TAG"; return; fi
log "Detecting latest release..."
if has_cmd curl; then TAG=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "${REPO_URL}/releases/latest"); TAG=${TAG##*/}; fi
if [ -z "$TAG" ]; then TAG=$(http_get_stdout "${API_URL}/latest" | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p' | head -1); fi
if [ -z "$TAG" ]; then die "could not determine latest release (network issue or GitHub API rate limit)."; fi
log "Latest release: $TAG"
}
verify_checksum() {
_file=$1
_asset=${2:-$ASSET}
[ -n "$ANVIL_SKIP_CHECKSUM" ] && { log "Checksum verification skipped (ANVIL_SKIP_CHECKSUM set)."; return 0; }
log "Verifying checksum..."
# The tag endpoint lists every asset with its sha256 digest. Collapse the JSON
# to one line, then pull the digest belonging to our asset name.
_meta=$(http_get_stdout "${API_URL}/tags/${TAG}" 2>/dev/null | tr -d '\n')
_digest=$(printf '%s' "$_meta" | sed -n "s/.*\"name\"[[:space:]]*:[[:space:]]*\"${_asset}\"[^}]*\"digest\"[[:space:]]*:[[:space:]]*\"sha256:\([^\"]*\)\".*/\1/p" | head -1)
if [ -z "$_digest" ]; then log "warning: no checksum available for ${_asset}; skipping verification"; return 0; fi
if has_cmd sha256sum; then _actual=$(sha256sum "$_file" | awk '{print $1}')
elif has_cmd shasum; then _actual=$(shasum -a 256 "$_file" | awk '{print $1}')
else log "warning: no sha256 tool available; skipping verification"; return 0; fi
if [ "$_actual" != "$_digest" ]; then err "checksum mismatch for ${_asset} (got $_actual, expected $_digest)"; return 1; fi
log "Checksum OK (sha256:${_digest})"
}

install_binary() {
resolve_tag
# Prefer the CUDA prebuilt on NVIDIA Linux x86_64; the plain CPU/Vulkan asset
# is the automatic fallback if the CUDA one is unavailable or won't run.
for _asset in ${CUDA_ASSET:-} "$ASSET"; do
    [ -n "$_asset" ] || continue
    URL="${REPO_URL}/releases/download/${TAG}/${_asset}"
    TMP_BIN="${TMP_BASE}/anvil"
    log "Downloading ${_asset}..."
    if ! http_get_file "$URL" "$TMP_BIN"; then
        log "  ${_asset} unavailable at ${TAG}; trying fallback"
        continue
    fi
    if ! verify_checksum "$TMP_BIN" "$_asset"; then
        log "  checksum failed for ${_asset}; trying fallback"
        continue
    fi
    if ! "$TMP_BIN" --version >/dev/null 2>&1; then
        log "  ${_asset} does not run on this system (missing libraries?); trying fallback"
        continue
    fi
    # Explicit failure guards: install_binary may be called in a context that
    # suppresses set -e (e.g. "install_binary || exit 1"), so check each step.
    if ! ${PRIV}mv "$TMP_BIN" "$TARGET/anvil"; then
        err "could not write ${TARGET}/anvil (run with sudo, or pick a writable target)"
        return 1
    fi
    ${PRIV}chmod +x "$TARGET/anvil"
    log "Installed anvil ${TAG} (${_asset}) -> ${TARGET}/anvil"
    return 0
done
err "no prebuilt binary worked for ${OSN}/${ARN} (tried: ${CUDA_ASSET:-none} ${ASSET})"
return 1
}

# NVIDIA driver helper (downloads the installer, never runs it for you — the
# privileged step stays one explicit sudo command the user types themselves).
DRIVER_URL="https://raw.githubusercontent.com/${REPO}/main/docs/anvil-nvidia-install.sh"
offer_nvidia_driver() {
    [ "$OSN" = "linux" ] || return 0
    [ "$NVIDIA" = 1 ] || return 0
    if has_cmd nvidia-smi; then return 0; fi
    log ""
    log "NVIDIA GPU detected, but the NVIDIA driver is not installed."
    log "anvil runs on CPU/Vulkan regardless, but the CUDA build needs the driver."
    log "Install it with the anvil driver helper (distro packages, never .run files):"
    log ""
    log "  curl -fsSL -O ${DRIVER_URL}"
    log "  chmod +x anvil-nvidia-install.sh"
    log "  ./anvil-nvidia-install.sh --check      # report only, changes nothing"
    log "  sudo ./anvil-nvidia-install.sh         # installs the driver"
    log ""
    if is_tty; then
        printf "Download the driver installer into the current directory? [y/N]: "
        read -r _yn || true
        case "$_yn" in
            [yY]*)
                if http_get_file "$DRIVER_URL" ./anvil-nvidia-install.sh; then
                    chmod +x ./anvil-nvidia-install.sh 2>/dev/null || true
                    log "Saved ./anvil-nvidia-install.sh - inspect it, then run with sudo."
                else
                    log "Download failed - grab it manually from ${DRIVER_URL}"
                fi
                ;;
        esac
    fi
}
build_from_source() {
has_cmd git || die "git is required to build from source"
has_cmd cmake || die "cmake is required to build from source"
if ! has_cmd g++ && ! has_cmd clang++; then die "g++ or clang++ is required to build from source"; fi
log "Cloning ${REPO}..."
SRC_DIR="${TMP_BASE}/anvil-src"
git clone --recursive "$REPO_URL" "$SRC_DIR"
log "Building anvil (this may take a few minutes)..."
cmake -B "$SRC_DIR/build" -S "$SRC_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$SRC_DIR/build" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)" >/dev/null || die "build failed"
${PRIV}cp "$SRC_DIR/build/anvil" "$TARGET/anvil" || die "could not copy binary to ${TARGET}"
${PRIV}chmod +x "$TARGET/anvil"
log "Built and installed anvil -> ${TARGET}/anvil"
}
choose_target() {
if [ -n "$INSTALL_DIR" ]; then
    TARGET=$INSTALL_DIR
    # Create/validate the target up front so the install step never hits a
    # missing-directory mv failure deep in the flow.
    ensure_target_writable "$TARGET" || die "cannot write to install target ${TARGET}"
    return
fi
if is_tty; then
log ""; log "Where should anvil be installed?"
printf "  1) %s (needs sudo if not writable)\n" "$DEFAULT_DIR"
printf "  2) %s\n" "$HOME/.local/bin"
printf "  3) custom path\n"
printf "Choose [1]: "; read -r choice || true
case "$choice" in 2) TARGET="$HOME/.local/bin" ;; 3) printf "Path: "; read -r TARGET || true ;; *) TARGET="$DEFAULT_DIR" ;; esac
else TARGET="$DEFAULT_DIR"; fi
if ! ensure_target_writable "$TARGET"; then
TARGET="$HOME/.local/bin"
log "${DEFAULT_DIR} is not writable; falling back to ${TARGET}"
if ! mkdir -p "$TARGET" 2>/dev/null; then die "cannot create ${TARGET}"; fi
PRIV=""
fi
}
ensure_path() {
[ "$TARGET" = "$DEFAULT_DIR" ] && return
case ":$PATH:" in *":$TARGET:"*) ;; *)
log ""; log "Add the following to your shell profile so 'anvil' is in your PATH:"
log "  export PATH=\"${TARGET}:\$PATH\"" ;;
esac
}
main() {
BUILD=0
FORCE_NVIDIA=0
# Preserve the externally-set ANVIL_BUILD env var (e.g. ANVIL_BUILD=1 curl ... | sh).
[ -n "${ANVIL_BUILD:-}" ] && BUILD=1
for _arg in "$@"; do
    case "$_arg" in
        --build|--source) BUILD=1 ;;
        --nvidia) FORCE_NVIDIA=1 ;;
    esac
done
if { [ "$NVIDIA" = 1 ] || [ "$FORCE_NVIDIA" = 1 ]; } && [ "$OSN" = "linux" ] && [ "$ARN" = "x86_64" ]; then
    CUDA_ASSET="anvil-${OSN}-${ARN}-cuda"
fi
log ""; log "anvil installer"; log "Detected: ${OSN} / ${ARN}"
if [ "$NVIDIA" = 1 ]; then log "  NVIDIA GPU detected - will prefer the CUDA build"; fi
if [ "$FORCE_NVIDIA" = 1 ]; then log "  --nvidia: forcing the CUDA build"; fi
choose_target; log "Install target: ${TARGET}"
if [ "$BUILD" = 1 ]; then build_from_source
elif ! is_tty; then log "Non-interactive mode: trying prebuilt binary..."; install_binary || exit 1
else
log ""; log "What would you like to do?"
printf "  1) Install prebuilt binary (fast, default)\n"
printf "  2) Build from source (slow, but always works)\n"
printf "  3) Cancel\n"
printf "Choose [1]: "; read -r choice || true
case "$choice" in
3) log "Cancelled."; exit 0 ;;
2) build_from_source ;;
*) if ! install_binary; then printf "Prebuilt binary unavailable. Build from source instead? [y/N]: "; read -r yn || true
case "$yn" in [yY]*) build_from_source ;; *) log "Cancelled."; exit 0 ;; esac; fi ;;
esac
fi
mkdir -p "$HOME/.anvil/models"
log "Config directory: ${HOME}/.anvil"
log "Models directory: ${HOME}/.anvil/models"
if "$TARGET/anvil" --version >/dev/null 2>&1; then log ""; log "anvil is ready. Go forge something."; log "  anvil --help"; log "  anvil run model.gguf"
else die "installation completed, but ${TARGET}/anvil could not be executed."; fi
offer_nvidia_driver
ensure_path
}
main "$@"
