#!/bin/sh
set -e

# ─────────────────────────────────────────────────────────────────────────────
#  anvil-nvidia-install.sh — NVIDIA driver installer for ANY Linux distro
#  Part of anvil (https://github.com/Anvil-LLM/anvil)
#
#  Universal by design:
#    * Uses NVIDIA's official distribution-independent .run installer with
#      --dkms, so the kernel module survives every kernel update on any distro.
#      (NVIDIA calls this method "distribution-independent".)
#    * Reads your actual hardware: detects GPU generation and automatically
#      picks the correct driver branch:
#        - Turing and newer (RTX, GTX 16xx, A/H/B-series) -> latest driver
#        - Pascal / Maxwell (GTX 10xx, GTX 9xx)           -> 580 series
#        - Kepler (GTX 6xx/7xx)                           -> 470 series
#      (Since 590, NVIDIA ships open kernel modules only; pre-Turing cards
#      need the last proprietary branches, 580 and 470.)
#    * The only distro-specific step is the tiny build toolchain (kernel
#      headers + gcc + make + dkms) — there is no way around distro packages
#      for those, but that is all they are used for.
#
#  Design rules (from the anvil CUDA-prebuilt project):
#    * The driver is the ONLY thing this installs. The anvil CUDA prebuilt
#      ships its CUDA runtime (libcudart/libcublas), so users never touch the
#      CUDA toolkit.
#    * Nothing is ever auto-sudo'd from a curl | sh pipeline: non-interactive
#      runs require an explicit --yes, and the privileged step stays one
#      explicit sudo command the user types.
#    * Every command is previewed before it runs (--dry-run prints exactly
#      what will execute).
#
#  Usage:
#    sudo ./anvil-nvidia-install.sh                universal install (default)
#    sudo ./anvil-nvidia-install.sh --check        report only, changes nothing
#    sudo ./anvil-nvidia-install.sh --dry-run      show every command, run none
#    sudo ./anvil-nvidia-install.sh --yes          skip confirmations
#    sudo ./anvil-nvidia-install.sh --driver 580.178.04   pin a driver version
#    sudo ./anvil-nvidia-install.sh --distro       distro packages instead (opt-in)
#    sudo ./anvil-nvidia-install.sh --nvidia-all   Frogging-Family nvidia-all (Arch)
# ─────────────────────────────────────────────────────────────────────────────

VERSION="0.2.0"
SCRIPT_NAME=$(basename "$0")
NVIDIA_BASE_X86="https://download.nvidia.com/XFree86/Linux-x86_64"
NVIDIA_BASE_ARM="https://download.nvidia.com/XFree86/Linux-aarch64"

# ─── Helpers (same conventions as install.sh) ──────────────────────────────
log()   { printf "%s\n" "$1"; }
err()   { printf "error: %s\n" "$1" >&2; }
die()   { err "$1"; exit 1; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
is_tty()  { [ -t 0 ] && [ -t 1 ]; }

# ─── Flags ─────────────────────────────────────────────────────────────────
HELP=0
SHOW_VERSION=0
CHECK_ONLY=0
DRY_RUN=0
YES=0
MODE="universal"          # universal | distro | nvidia-all
DRIVER_OVERRIDE=""
TMP_DIR=""
trap '[ -n "$TMP_DIR" ] && rm -rf "$TMP_DIR"' EXIT

usage() {
    cat <<EOF
anvil-nvidia-install v${VERSION}

Universal NVIDIA driver installer for any Linux distro and any NVIDIA GPU.
Uses NVIDIA's official .run installer with DKMS (auto-rebuilds on kernel
updates). GPU generation is auto-detected and maps to the right driver
branch (Turing+ = latest, Pascal/Maxwell = 580, Kepler = 470).

Usage:
  ${SCRIPT_NAME} [options]

Options:
  -h, --help            Show this help
  -V, --version         Show version
  -c, --check           Report GPU/driver/secure-boot status, change nothing
  -d, --dry-run         Print every command that would run, run nothing
  -y, --yes             Skip all confirmations (for non-interactive use)
      --driver <ver>    Pin an exact driver version, e.g. 580.178.04
                        (universal mode: exact version; --distro mode: package
                        suffix, e.g. --driver 570 for nvidia-driver-570)
      --distro          Use distro packages (apt/dnf/pacman/zypper) instead
      --nvidia-all      Use Frogging-Family/nvidia-all (mature on Arch)

Examples:
  sudo ${SCRIPT_NAME} --check
  sudo ${SCRIPT_NAME} --dry-run
  sudo ${SCRIPT_NAME}
  sudo ${SCRIPT_NAME} --driver 580.178.04

Notes:
  * Requires root for installation (run with sudo). --check does not.
  * Installs a build toolchain (kernel headers, gcc, make, dkms) via your
    distro — unavoidable, since kernel modules must match your exact kernel.
  * If Secure Boot is enabled you will be asked to enroll a key (MOK) on
    the next reboot — this is normal and required for the module to load.
  * A reboot is required after installation.
EOF
}

parse_flags() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)          HELP=1 ;;
            -V|--version)       SHOW_VERSION=1 ;;
            -c|--check)         CHECK_ONLY=1 ;;
            -d|--dry-run)       DRY_RUN=1 ;;
            -y|--yes)           YES=1 ;;
            --distro)           MODE="distro" ;;
            --nvidia-all)       MODE="nvidia-all" ;;
            --driver)
                [ $# -ge 2 ] || die "--driver requires a version, e.g. 580.178.04"
                DRIVER_OVERRIDE=$2
                shift ;;
            *)
                err "unknown option: $1"
                usage
                exit 1 ;;
        esac
        shift
    done
}

# Execute a command, or print it under --dry-run.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        log "  would run: $*"
        return 0
    fi
    "$@"
}

# ─── Detection ─────────────────────────────────────────────────────────────
FAMILY="other"
DISTRO_ID=""
GPU_LINE=""
NVIDIA_PRESENT=0
GPU_GEN="unknown"          # modern | pascal | kepler | unknown
DRIVER_VERSION=""
DRIVER_INSTALLED=0
SB_ENABLED=0
NOUVEAU_LOADED=0
SESSION_TYPE=""
KVER=""
ARN="x86_64"

detect_distro() {
    if [ ! -r /etc/os-release ]; then
        FAMILY="other"
        return 0
    fi
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_ID=${ID:-unknown}
    DISTRO_LIKE=${ID_LIKE:-}
    case "$DISTRO_ID" in
        ubuntu|debian|linuxmint|pop|elementary|zorin|kali) FAMILY=debian ;;
        fedora|rhel|centos|rocky|almalinux|nobara)         FAMILY=fedora ;;
        arch|manjaro|endeavouros|garuda|arcolinux|cachyos) FAMILY=arch ;;
        opensuse*|suse|sles)                               FAMILY=suse ;;
        alpine)                                            FAMILY=alpine ;;
        void)                                              FAMILY=void ;;
        gentoo)                                            FAMILY=gentoo ;;
        *)
            case "$DISTRO_LIKE" in
                *debian*) FAMILY=debian ;;
                *fedora*) FAMILY=fedora ;;
                *arch*)   FAMILY=arch ;;
                *)        FAMILY=other ;;
            esac
            ;;
    esac
}

detect_gpu() {
    NVIDIA_PRESENT=0
    if has_cmd lspci; then
        GPU_LINE=$(lspci 2>/dev/null | grep -i nvidia | head -1 || true)
        if lspci -nn 2>/dev/null | grep -qi '\[10de:'; then NVIDIA_PRESENT=1; return; fi
        if [ -n "$GPU_LINE" ]; then NVIDIA_PRESENT=1; return; fi
    fi
    for _v in /sys/bus/pci/devices/*/vendor; do
        [ -r "$_v" ] || continue
        if [ "$(cat "$_v" 2>/dev/null)" = "0x10de" ]; then NVIDIA_PRESENT=1; return; fi
    done
    [ -z "$GPU_LINE" ] && GPU_LINE="NVIDIA GPU (vendor 10de)"
}

# Classify GPU generation from the lspci name so the correct driver branch is
# selected. Order matters: modern first, then Pascal/Maxwell, then Kepler.
detect_gpu_gen() {
    GPU_GEN="unknown"
    [ "$NVIDIA_PRESENT" = 1 ] || return 0
    case "$GPU_LINE" in
        *RTX*|*"GTX 16"*|*Turing*|*Ampere*|*Ada*|*Hopper*|*Blackwell*|*"RTX A"*|*"RTX PRO"*|*"Quadro RTX"*|*"Tesla A"*|*"Tesla T"*|*"Tesla H"*|*"Tesla L"*|*"Tesla B"*|*A100*|*H100*|*H200*|*B100*|*B200*|*L4*|*L40*|*L20*|*A10*|*A30*|*A40*|*A6000*|*TU10*|*TU11*|*T1000*|*T2000*)
            GPU_GEN="modern" ;;
        *"GTX 10"*|*"GTX 9"*|*"GT 10"*|*Pascal*|*Maxwell*|*"Quadro P"*|*"Quadro M"*|*"Tesla P"*|*"Tesla V"*|*P100*|*V100*|*P40*|*P4*)
            GPU_GEN="pascal" ;;
        *"GTX 6"*|*"GTX 7"*|*"GT 6"*|*"GT 7"*|*Kepler*|*"Quadro K"*|*"Tesla K"*|*K80*|*K40*)
            GPU_GEN="kepler" ;;
    esac
}

detect_driver_state() {
    if has_cmd nvidia-smi; then
        DRIVER_INSTALLED=1
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    fi
}

detect_secure_boot() {
    if has_cmd mokutil; then
        if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then SB_ENABLED=1; fi
    fi
}

detect_nouveau() {
    if grep -q nouveau /proc/modules 2>/dev/null; then NOUVEAU_LOADED=1; fi
}

detect_session() {
    SESSION_TYPE=${XDG_SESSION_TYPE:-unknown}
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then SESSION_TYPE="graphical"; fi
}

# ─── Driver version resolution (needs network) ─────────────────────────────
nvidia_base() {
    case "$ARN" in
        x86_64)  printf '%s' "$NVIDIA_BASE_X86" ;;
        aarch64) printf '%s' "$NVIDIA_BASE_ARM" ;;
        *) die "unsupported architecture: $ARN" ;;
    esac
}

# Latest 580.x / 470.x from the browsable version index.
latest_branch_version() {
    _branch=$1
    curl -fsSL "$(nvidia_base)/" 2>/dev/null \
        | grep -oE "${_branch}\.[0-9]+(\.[0-9]+)?/" \
        | tr -d '/' \
        | sort -t. -k1,1n -k2,2n -k3,3n \
        | tail -1 || true
}

resolve_driver() {
    # Resolves the version to INSTALL. DRIVER_VERSION is left holding the
    # already-installed driver (if any) for reporting.
    INSTALL_VERSION=""
    if [ -n "$DRIVER_OVERRIDE" ]; then
        INSTALL_VERSION=$DRIVER_OVERRIDE
    else
        case "$GPU_GEN" in
            modern) INSTALL_VERSION=$(curl -fsSL "$(nvidia_base)/latest.txt" 2>/dev/null | awk '{print $1}' || true) ;;
            pascal) INSTALL_VERSION=$(latest_branch_version 580) ;;
            kepler) INSTALL_VERSION=$(latest_branch_version 470) ;;
            *)
                # Unknown generation: default to the latest driver and say so.
                INSTALL_VERSION=$(curl -fsSL "$(nvidia_base)/latest.txt" 2>/dev/null | awk '{print $1}' || true)
                ;;
        esac
    fi
    if [ -z "$INSTALL_VERSION" ]; then
        die "could not resolve an NVIDIA driver version (network issue?). Pin one with --driver."
    fi
    DRIVER_URL="$(nvidia_base)/${INSTALL_VERSION}/NVIDIA-Linux-${ARN}-${INSTALL_VERSION}.run"
}

# ─── Report ────────────────────────────────────────────────────────────────
print_report() {
    _branch_note=""
    case "$GPU_GEN" in
        modern) _branch_note="latest (Turing+ -> open kernel modules)" ;;
        pascal) _branch_note="580 series (Pascal/Maxwell, proprietary)" ;;
        kepler) _branch_note="470 series (Kepler, proprietary)" ;;
        *)      _branch_note="latest (generation undetermined)" ;;
    esac
    log ""
    log "anvil-nvidia-install ${VERSION} — status report"
    log "  Distro        : ${DISTRO_ID:-unknown} (family: ${FAMILY})"
    log "  Arch          : ${ARN}"
    log "  GPU           : $([ "$NVIDIA_PRESENT" = 1 ] && echo "${GPU_LINE}" || echo "none detected")"
    if [ "$NVIDIA_PRESENT" = 1 ]; then
        log "  GPU generation: ${GPU_GEN} -> ${_branch_note}"
        if [ "$DRIVER_INSTALLED" = 1 ]; then
            log "  Driver        : installed (${DRIVER_VERSION:-unknown})"
        else
            log "  Driver        : NOT installed"
        fi
        log "  Secure Boot   : $([ "$SB_ENABLED" = 1 ] && echo "enabled (MOK enrollment needed)" || echo "disabled/unknown")"
        log "  Nouveau       : $([ "$NOUVEAU_LOADED" = 1 ] && echo "loaded (will be replaced)" || echo "not loaded")"
        log "  Kernel headers: $([ -d "/usr/src/linux-headers-${KVER}" ] || [ -d "/usr/src/kernels/${KVER}" ] || [ -d "/usr/lib/modules/${KVER}/build" ] && echo "present" || echo "missing - will be installed")"
        log "  Session       : ${SESSION_TYPE}"
    else
        log "  No NVIDIA GPU detected — nothing to do."
    fi
    log ""
}

# ─── Toolchain (the only distro-specific step) ─────────────────────────────
install_toolchain() {
    _kver=$(uname -r)
    case "$FAMILY" in
        debian)
            run apt-get update
            run apt-get install -y "linux-headers-${_kver}" dkms build-essential
            ;;
        fedora)
            run dnf install -y "kernel-devel-${_kver}" dkms gcc make
            ;;
        arch)
            if [ ! -d "/usr/lib/modules/${_kver}/build" ]; then
                log "  custom kernel detected: install matching headers manually"
            else
                run pacman -S --noconfirm --needed dkms gcc make
            fi
            ;;
        suse)
            run zypper --non-interactive install kernel-devel kernel-source dkms gcc make
            ;;
        alpine)
            run apk add linux-headers build-base dkms
            ;;
        void)
            run xbps-install -y linux-headers dkms gcc make
            ;;
        gentoo)
            run emerge --ask=n sys-kernel/linux-headers sys-devel/gcc sys-devel/make dkms
            ;;
        *)
            if [ "$DRY_RUN" = 1 ]; then
                log "  would run: install kernel headers for $(uname -r), gcc, make and dkms (manual)"
            else
                die "unsupported distro (${DISTRO_ID}): install kernel headers for $(uname -r), gcc, make and dkms manually, then re-run"
            fi
            ;;
    esac
}

# ─── Universal install: official NVIDIA .run + DKMS ────────────────────────
install_universal() {
    install_toolchain

    if [ "$DRIVER_INSTALLED" = 1 ]; then
        log "  NOTE: an NVIDIA driver (${DRIVER_VERSION}) is already installed; installing over it."
    fi

    TMP_DIR="/tmp/anvil-nvidia-$$"
    run mkdir -p "$TMP_DIR"
    log "  Downloading NVIDIA-Linux-${ARN}-${INSTALL_VERSION}.run (~200 MB)..."
    run curl -fSL -o "$TMP_DIR/installer.run" "$DRIVER_URL"

    if [ "$DRY_RUN" != 1 ]; then
        # Sanity check: the .run is a 100+ MB self-extracting archive; a tiny
        # file means a captive-portal or error page slipped past curl -f.
        _size=$(wc -c < "$TMP_DIR/installer.run" 2>/dev/null || echo 0)
        if [ "$_size" -lt 10485760 ]; then
            die "downloaded installer looks wrong (${_size} bytes) - network issue?"
        fi
    fi

    # The official installer: --dkms keeps the module rebuilt on every kernel
    # update; open kernel modules are the default on modern (590+) versions.
    # shellcheck disable=SC2086
    _flags="--silent --accept-license --dkms --no-x-check --no-cc-version-check"
    if [ "$NOUVEAU_LOADED" = 1 ]; then
        _flags="$_flags --no-nouveau-check"
    fi
    log "  Running the NVIDIA installer (DKMS)..."
    run sh "$TMP_DIR/installer.run" $_flags
    run rm -rf "$TMP_DIR"
    TMP_DIR=""
}

# ─── Opt-in: distro packages ───────────────────────────────────────────────
install_distro() {
    case "$FAMILY" in
        debian)
            run apt-get update
            run apt-get install -y "linux-headers-$(uname -r)" dkms
            if [ -n "$DRIVER_OVERRIDE" ]; then
                run apt-get install -y "nvidia-driver-${DRIVER_OVERRIDE}"
            elif has_cmd ubuntu-drivers; then
                _rec=$(ubuntu-drivers devices 2>/dev/null | grep -i recommended | sed -n 's/.*\(nvidia-driver-[0-9][0-9]*[^ ]*\).*/\1/p' | head -1 || true)
                if [ -n "$_rec" ]; then run apt-get install -y "$_rec"; else run ubuntu-drivers autoinstall; fi
            else
                run apt-get install -y nvidia-driver
            fi
            ;;
        fedora)
            run dnf install -y \
                "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
                "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm"
            run dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
            has_cmd akmods && run akmods --force
            ;;
        arch)
            run pacman -S --noconfirm --needed nvidia nvidia-utils
            ;;
        suse)
            if printf '%s' "${VERSION_ID:-} ${PRETTY_NAME:-}" | grep -qi tumbleweed; then
                run zypper --non-interactive addrepo --refresh https://download.nvidia.com/opensuse/tumbleweed/ nvidia
            else
                die "openSUSE Leap: add the NVIDIA repo for your version manually, then run zypper --non-interactive install nvidia-driver-G06"
            fi
            if zypper --non-interactive search -s nvidia-open-driver-G07-signed-kmp-default 2>/dev/null | grep -q nvidia-open-driver-G07; then
                run zypper --non-interactive install nvidia-open-driver-G07-signed-kmp-default
            else
                run zypper --non-interactive install x11-video-nvidiaG06 nvidia-glG06
            fi
            ;;
        *)
            die "no distro-package path for ${DISTRO_ID}. Use the default universal mode."
            ;;
    esac
}

# ─── Opt-in: Frogging-Family/nvidia-all (mature on Arch) ───────────────────
install_nvidia_all() {
    has_cmd git || die "git is required for --nvidia-all"
    run git clone --depth 1 https://github.com/Frogging-Family/nvidia-all.git /tmp/anvil-nvidia-all
    log "  Running nvidia-all installer (interactive prompts)..."
    run sh /tmp/anvil-nvidia-all/install.sh
}

do_install() {
    case "$MODE" in
        universal)  install_universal ;;
        distro)     install_distro ;;
        nvidia-all) install_nvidia_all ;;
    esac
}

# ─── Plan / confirm ────────────────────────────────────────────────────────
print_plan() {
    _branch_note=""
    case "$GPU_GEN" in
        modern) _branch_note="latest driver (Turing+, open kernel modules)" ;;
        pascal) _branch_note="580 series (Pascal/Maxwell)" ;;
        kepler) _branch_note="470 series (Kepler)" ;;
        *)      _branch_note="latest driver (generation undetermined)" ;;
    esac
    log ""
    log "Plan:"
    log "  Mode        : ${MODE}"
    log "  Distro      : ${DISTRO_ID} (${FAMILY})"
    log "  GPU         : ${GPU_LINE}"
    log "  Driver      : ${INSTALL_VERSION:-to be resolved} (${_branch_note})"
    if [ "$DRIVER_INSTALLED" = 1 ]; then log "  Already     : driver ${DRIVER_VERSION} installed - will be replaced"; fi
    [ "$SB_ENABLED" = 1 ] && log "  Secure Boot : enabled - expect a MOK enrollment prompt on reboot"
    [ "$NOUVEAU_LOADED" = 1 ] && log "  Nouveau     : loaded - will be blacklisted and replaced"
    log ""
    log "Commands that will be run:"
    ( DRY_RUN=1 do_install )
    log ""
}

confirm_install() {
    if [ "$DRY_RUN" = 1 ] || [ "$YES" = 1 ]; then return 0; fi
    printf "%s" "Proceed with installation? [y/N] "
    read -r _ans || true
    case "$_ans" in
        [yY]*) return 0 ;;
        *) log "Cancelled."; exit 0 ;;
    esac
}

# ─── Post-install ──────────────────────────────────────────────────────────
secure_boot_advice() {
    [ "$SB_ENABLED" = 1 ] || return 0
    log ""
    log "Secure Boot is enabled. The DKMS-built module must be signed to load."
    if [ -r /var/lib/dkms/mok.pub ] && has_cmd mokutil; then
        log "  Enroll the DKMS key now with:"
        log "    sudo mokutil --import /var/lib/dkms/mok.pub"
        log "  (You will set a one-time password, then enroll it at the blue"
        log "  'MOK management' screen on reboot.)"
    else
        log "  Enroll a MOK key, or check your distro's DKMS signing setup,"
        log "  before rebooting - otherwise the module will not load."
    fi
}

post_install() {
    log ""
    log "Installation finished."
    if has_cmd nvidia-smi; then
        log "  nvidia-smi available at $(command -v nvidia-smi)"
    else
        log "  nvidia-smi will appear after reboot."
    fi
    secure_boot_advice
    log "  A reboot is required for the driver to load."
    if is_tty && [ "$DRY_RUN" != 1 ]; then
        printf "%s" "Reboot now? [y/N] "
        read -r _ans || true
        case "$_ans" in
            [yY]*)
                if has_cmd systemctl; then systemctl reboot || true
                else log "Run 'reboot' manually."; fi ;;
        esac
    fi
}

# ─── Main ──────────────────────────────────────────────────────────────────
main() {
    parse_flags "$@"
    [ "$HELP" = 1 ] && { usage; exit 0; }
    [ "$SHOW_VERSION" = 1 ] && { log "anvil-nvidia-install ${VERSION}"; exit 0; }

    case "$(uname -m 2>/dev/null)" in
        x86_64|amd64) ARN="x86_64" ;;
        aarch64|arm64) ARN="aarch64" ;;
        *) die "unsupported architecture: $(uname -m)" ;;
    esac
    KVER=$(uname -r 2>/dev/null || echo unknown)

    log "anvil-nvidia-install ${VERSION} — universal NVIDIA driver installer"
    log ""

    detect_distro
    detect_gpu
    detect_gpu_gen
    detect_driver_state
    detect_secure_boot
    detect_nouveau
    detect_session

    if [ "$CHECK_ONLY" = 1 ]; then
        print_report
        exit 0
    fi

    # The "no auto-sudo from curl | sh" rule: non-interactive without an
    # explicit --yes never installs anything.
    if ! is_tty && [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
        die "non-interactive shell. Re-run with --yes to confirm, or --dry-run to preview."
    fi

    if [ "$FAMILY" = "other" ] && [ "$MODE" != "universal" ]; then
        die "unsupported distro '${DISTRO_ID}' for ${MODE} mode. The default universal mode works on any distro."
    fi
    if [ "$NVIDIA_PRESENT" != 1 ]; then
        die "no NVIDIA GPU detected. This installer is for NVIDIA hardware only."
    fi

    # Root is only required to actually install (never for --check/--dry-run).
    if [ "$DRY_RUN" != 1 ] && [ "$(id -u)" != 0 ]; then
        die "must run as root. Use: sudo $0 $*"
    fi

    if [ "$MODE" = "universal" ]; then
        resolve_driver
    fi

    if [ "$DRY_RUN" = 1 ]; then
        print_plan
        log "(dry run — nothing was changed)"
        exit 0
    fi

    print_plan
    confirm_install

    log "Installing..."
    do_install

    post_install
    log ""
    log "Done. Reboot, then verify with: nvidia-smi"
}

main "$@"
