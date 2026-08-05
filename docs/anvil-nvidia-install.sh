#!/bin/sh
set -e

# ─────────────────────────────────────────────────────────────────────────────
#  anvil-nvidia-install.sh — NVIDIA driver autoinstaller for Linux
#  Part of anvil (https://github.com/Anvil-LLM/anvil)
#
#  Safely installs the NVIDIA proprietary driver via the distro's own package
#  manager (never the .run file), with preflight checks, a full command
#  preview before anything runs, and a rollback-friendly package path.
#
#  Design rules (from the anvil CUDA-prebuilt project):
#    * The driver is the ONLY thing this installs. The anvil CUDA prebuilt
#      ships its CUDA runtime (libcudart/libcublas) alongside the binary, so
#      users never touch the CUDA toolkit.
#    * Nothing is ever auto-sudo'd from a curl | sh pipeline: the installer
#      refuses to run non-interactively without an explicit --yes, and the
#      privileged step is exactly one command the user runs themselves.
#    * Distro packages over .run files, always. Rollback = apt/dnf/pacman
#      purge. No manual kernel-module tarballs.
#
#  Usage:
#    sudo ./anvil-nvidia-install.sh            interactive (recommended)
#    sudo ./anvil-nvidia-install.sh --check    report only, changes nothing
#    sudo ./anvil-nvidia-install.sh --dry-run  show every command, run nothing
#    sudo ./anvil-nvidia-install.sh --yes      skip confirmations (non-interactive)
#    sudo ./anvil-nvidia-install.sh --driver nvidia-driver-570   pin a version
# ─────────────────────────────────────────────────────────────────────────────

VERSION="0.1.0"
SCRIPT_NAME=$(basename "$0")

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
DRIVER_OVERRIDE=""

usage() {
    cat <<EOF
anvil-nvidia-install v${VERSION}

Safely install the NVIDIA proprietary driver on Linux using your distro's
package manager. Supported: Ubuntu/Debian, Fedora/RHEL, Arch, openSUSE.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  -h, --help            Show this help
  -V, --version         Show version
  -c, --check           Report driver/GPU/secure-boot status, change nothing
  -d, --dry-run         Print every command that would run, run nothing
  -y, --yes             Skip all confirmations (for non-interactive use)
      --driver <pkg>    Pin a driver package (Ubuntu/Debian only), e.g.
                        nvidia-driver-570 or nvidia-driver-570-open

Examples:
  sudo ${SCRIPT_NAME} --check
  sudo ${SCRIPT_NAME} --dry-run
  sudo ${SCRIPT_NAME}
  sudo ${SCRIPT_NAME} --driver nvidia-driver-570

Notes:
  * Requires root for installation (run with sudo). --check does not.
  * Uses distro packages only (dkms/akmod); never the NVIDIA .run file.
  * If Secure Boot is enabled you may be asked to enroll a key (MOK)
    on the next reboot — this is normal and required for the module.
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
            --driver)
                [ $# -ge 2 ] || die "--driver requires a package name"
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
DISTRO_ID=""
DISTRO_LIKE=""
FAMILY="other"
GPU_LINE=""
NVIDIA_PRESENT=0
DRIVER_VERSION=""
DRIVER_INSTALLED=0
SB_ENABLED=0
NOUVEAU_LOADED=0
HEADERS_OK=0
HEADERS_CMD=""
HEADERS_NOTE=""
SESSION_TYPE=""

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
    # Fallback: sysfs PCI vendor id (works without lspci)
    for v in /sys/bus/pci/devices/*/vendor; do
        [ -r "$v" ] || continue
        if [ "$(cat "$v" 2>/dev/null)" = "0x10de" ]; then NVIDIA_PRESENT=1; return; fi
    done
    [ -z "$GPU_LINE" ] && GPU_LINE="NVIDIA GPU (vendor 10de)"
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
    elif [ -d /sys/firmware/efi ]; then
        # No mokutil: treat unknown-but-EFI as "possibly enabled" and warn later.
        SB_ENABLED=0
    fi
}

detect_nouveau() {
    if grep -q nouveau /proc/modules 2>/dev/null; then NOUVEAU_LOADED=1; fi
}

detect_headers() {
    kver=$(uname -r)
    case "$FAMILY" in
        debian)
            if dpkg -s "linux-headers-${kver}" 2>/dev/null | grep -q "^Status: install ok"; then
                HEADERS_OK=1
            else
                HEADERS_CMD="apt-get install -y linux-headers-${kver}"
                HEADERS_NOTE="linux-headers-${kver} will be installed (needed for DKMS)"
            fi
            ;;
        fedora)
            if [ -d "/usr/src/kernels/${kver}" ]; then
                HEADERS_OK=1
            else
                HEADERS_CMD="dnf install -y kernel-devel-${kver}"
                HEADERS_NOTE="kernel-devel-${kver} will be installed (needed by akmod)"
            fi
            ;;
        arch)
            if [ -d "/usr/lib/modules/${kver}/build" ]; then
                HEADERS_OK=1
            else
                HEADERS_NOTE="custom kernel detected: install the matching headers package manually"
            fi
            ;;
        suse)
            if [ -d "/usr/src/linux-obj/${kver}" ] || [ -d "/lib/modules/${kver}/build" ]; then
                HEADERS_OK=1
            else
                HEADERS_CMD="zypper --non-interactive install kernel-devel kernel-source"
                HEADERS_NOTE="kernel-devel/kernel-source will be installed (needed for DKMS)"
            fi
            ;;
    esac
}

detect_session() {
    SESSION_TYPE=${XDG_SESSION_TYPE:-unknown}
}

# Recommended driver branch (Ubuntu/Debian).
detect_recommended_driver() {
    if [ -n "$DRIVER_OVERRIDE" ]; then
        DRIVER_BRANCH=$DRIVER_OVERRIDE
        return
    fi
    DRIVER_BRANCH=""
    if [ "$FAMILY" = "debian" ] && has_cmd ubuntu-drivers; then
        DRIVER_BRANCH=$(ubuntu-drivers devices 2>/dev/null \
            | grep -i recommended \
            | sed -n 's/.*\(nvidia-driver-[0-9][0-9]*[^ ]*\).*/\1/p' \
            | head -1 || true)
    fi
}

# ─── Report ────────────────────────────────────────────────────────────────
print_report() {
    log ""
    log "anvil-nvidia-install ${VERSION} — status report"
    log "  Distro        : ${DISTRO_ID:-unknown} (family: ${FAMILY})"
    log "  GPU           : $([ "$NVIDIA_PRESENT" = 1 ] && echo "${GPU_LINE}" || echo "none detected")"
    if [ "$NVIDIA_PRESENT" = 1 ]; then
        if [ "$DRIVER_INSTALLED" = 1 ]; then
            log "  Driver        : installed (${DRIVER_VERSION:-unknown})"
        else
            log "  Driver        : NOT installed"
        fi
        log "  Secure Boot   : $([ "$SB_ENABLED" = 1 ] && echo "enabled" || echo "disabled/unknown")"
        log "  Nouveau       : $([ "$NOUVEAU_LOADED" = 1 ] && echo "loaded (will be replaced)" || echo "not loaded")"
        log "  Kernel headers: $([ "$HEADERS_OK" = 1 ] && echo "present" || echo "missing")"
        [ -n "$HEADERS_NOTE" ] && log "                  ${HEADERS_NOTE}"
        log "  Session       : ${SESSION_TYPE}"
    else
        log "  No NVIDIA GPU detected — nothing to do."
    fi
    log ""
}

# ─── Install paths (distro package manager only) ───────────────────────────
install_debian() {
    run apt-get update
    if [ -n "$HEADERS_CMD" ]; then
        # shellcheck disable=SC2086
        run $HEADERS_CMD
    fi
    if [ -n "$DRIVER_BRANCH" ]; then
        run apt-get install -y "$DRIVER_BRANCH"
    elif has_cmd ubuntu-drivers; then
        run ubuntu-drivers autoinstall
    else
        run apt-get install -y nvidia-driver
    fi
}

install_fedora() {
    # Enable RPM Fusion free + nonfree (required for akmod-nvidia).
    run dnf install -y \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm"
    if [ -n "$HEADERS_CMD" ]; then
        # shellcheck disable=SC2086
        run $HEADERS_CMD
    fi
    run dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
    if has_cmd akmods; then
        run akmods --force
    fi
}

install_arch() {
    # nvidia = proprietary module for the stock linux kernel.
    # Turing+ users may prefer: pacman -S --noconfirm --needed nvidia-open nvidia-utils
    run pacman -S --noconfirm --needed nvidia nvidia-utils
}

install_suse() {
    if printf '%s' "${VERSION_ID:-} ${PRETTY_NAME:-}" | grep -qi tumbleweed; then
        suse_stream="tumbleweed"
    else
        suse_stream="leap"
    fi
    if [ "$suse_stream" = "tumbleweed" ]; then
        run zypper --non-interactive addrepo --refresh \
            https://download.nvidia.com/opensuse/tumbleweed/ nvidia
    else
        die "openSUSE Leap: add the NVIDIA repo for your version manually, then run zypper --non-interactive install nvidia-driver-G06"
    fi
    if [ -n "$HEADERS_CMD" ]; then
        # shellcheck disable=SC2086
        run $HEADERS_CMD
    fi
    # Modern open-kernel driver (Turing+) with a fallback to the G06 line.
    if zypper --non-interactive search -s nvidia-open-driver-G07-signed-kmp-default 2>/dev/null | grep -q nvidia-open-driver-G07; then
        run zypper --non-interactive install nvidia-open-driver-G07-signed-kmp-default
    else
        run zypper --non-interactive install x11-video-nvidiaG06 nvidia-glG06
    fi
}

do_install() {
    case "$FAMILY" in
        debian) install_debian ;;
        fedora) install_fedora ;;
        arch)   install_arch ;;
        suse)   install_suse ;;
        *)      die "unsupported distro family: ${FAMILY}" ;;
    esac
}

# ─── Plan / confirm ────────────────────────────────────────────────────────
print_plan() {
    log ""
    log "Plan:"
    log "  Distro      : ${DISTRO_ID} (${FAMILY})"
    [ -n "$DRIVER_BRANCH" ] && log "  Driver      : ${DRIVER_BRANCH}"
    [ "$SB_ENABLED" = 1 ] && log "  Secure Boot : enabled — expect a MOK enrollment prompt on reboot"
    [ "$NOUVEAU_LOADED" = 1 ] && log "  Nouveau     : loaded — will be blacklisted and replaced"
    [ -n "$HEADERS_NOTE" ] && log "  Headers     : ${HEADERS_NOTE}"
    log ""
    log "Commands that will be run:"
    # Subshell is required: a variable assignment prefixing a function call
    # persists in the current shell after the call returns (POSIX), which
    # would otherwise leave DRY_RUN=1 set for the real install below.
    ( DRY_RUN=1 do_install )
    log ""
}

confirm_install() {
    if [ "$DRY_RUN" = 1 ] || [ "$YES" = 1 ]; then return 0; fi
    printf "%s" "Proceed with installation? [y/N] "
    read -r ans || true
    case "$ans" in
        [yY]*) return 0 ;;
        *) log "Cancelled."; exit 0 ;;
    esac
}

# ─── Post-install ──────────────────────────────────────────────────────────
post_install() {
    log ""
    log "Installation finished."
    if has_cmd nvidia-smi; then
        log "  nvidia-smi available at $(command -v nvidia-smi)"
    else
        log "  nvidia-smi not found yet — it appears after reboot."
    fi
    if [ "$SB_ENABLED" = 1 ]; then
        log "  Secure Boot is enabled: on reboot you may get a blue 'MOK management' prompt."
        log "  Enroll the key to let the NVIDIA module load."
    fi
    log "  A reboot is required for the driver to load."
    if is_tty && [ "$DRY_RUN" != 1 ]; then
        printf "%s" "Reboot now? [y/N] "
        read -r ans || true
        case "$ans" in
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

    log "anvil-nvidia-install ${VERSION} — NVIDIA driver installer"
    log ""

    detect_distro
    detect_gpu
    detect_driver_state
    detect_secure_boot
    detect_nouveau
    detect_headers
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

    if [ "$FAMILY" = "other" ]; then
        die "unsupported distro '${DISTRO_ID}'. See https://www.nvidia.com/en-us/drivers/unix/ for manual instructions."
    fi
    if [ "$NVIDIA_PRESENT" != 1 ]; then
        die "no NVIDIA GPU detected. This installer is for NVIDIA hardware only."
    fi

    # Root is only required to actually install (never for --check/--dry-run).
    if [ "$DRY_RUN" != 1 ] && [ "$(id -u)" != 0 ]; then
        die "must run as root. Use: sudo $0 $*"
    fi

    detect_recommended_driver

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
