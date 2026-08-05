#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  colab-nvidia-test.sh — test anvil-nvidia-install.sh on Google Colab
#
#  Colab gives a real Ubuntu VM with a real NVIDIA GPU (free tier: T4/Turing),
#  already running the NVIDIA driver. That makes it perfect for testing the
#  installer's detection, branch resolution, toolchain and download paths for
#  real — but NOT the final module swap (Colab's driver is already loaded and
#  you cannot reboot a session).
#
#  What this tests (safe, non-destructive):
#    1. --check on real hardware (distro + GPU + generation classification)
#    2. --dry-run full plan preview
#    3. Branch consistency: resolved driver major matches detected generation
#    4. Real toolchain install (kernel headers + dkms + build tools)
#    5. Real .run download + validity check (sh installer.run --version)
#
#  What it does NOT run (unless you pass --install): the actual driver swap.
#  On Colab that will conflict with the loaded driver and likely break the
#  GPU for the rest of the session (no reboot). Test that on a real machine.
#
#  Usage in a Colab cell:
#    !bash <(curl -fsSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/tests/colab-nvidia-test.sh)
#  Destructive variant (breaks the session GPU — last cell before giving up):
#    !bash <(curl -fsSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/tests/colab-nvidia-test.sh) --install
# ─────────────────────────────────────────────────────────────────────────────

REPO="gondaliyashreyan1/Anvil"
BRANCH="main"
INSTALLER_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/docs/anvil-nvidia-install.sh"
DRY=""

PASS=0
FAIL=0
WARN=0

report() { # pass|fail|warn desc
    case "$1" in
        pass) echo "  [PASS] $2"; PASS=$((PASS + 1)) ;;
        fail) echo "  [FAIL] $2"; FAIL=$((FAIL + 1)) ;;
        warn) echo "  [WARN] $2"; WARN=$((WARN + 1)) ;;
    esac
}

echo "== anvil-nvidia-install Colab test =="
echo "== repo: ${REPO}@${BRANCH}"
echo ""

# 0. Fetch the installer -----------------------------------------------------
echo "── 0. Fetch installer"
cd /tmp || exit 1
if curl -fsSL "$INSTALLER_URL" -o anvil-nvidia-install.sh; then
    report pass "downloaded installer (${INSTALLER_URL})"
    chmod +x anvil-nvidia-install.sh
else
    report fail "could not download installer — check REPO/BRANCH"
    echo "TOTAL: ${PASS} pass, ${FAIL} fail, ${WARN} warn"
    exit 1
fi

# 1. --check on real hardware ------------------------------------------------
echo ""
echo "── 1. Installer --check (real hardware)"
CHECK_OUT=$(./anvil-nvidia-install.sh --check 2>&1)
echo "$CHECK_OUT"
echo ""
echo "$CHECK_OUT" | grep -q "GPU           :" && report pass "GPU line present"
echo "$CHECK_OUT" | grep -q "GPU generation:" && report pass "generation classified" || report fail "generation not classified"
echo "$CHECK_OUT" | grep -q "Distro        :" && report pass "distro detected" || report fail "distro missing"

GEN=$(echo "$CHECK_OUT" | sed -n 's/.*GPU generation: \([a-z]*\).*/\1/p')
echo "  → detected generation: ${GEN:-unknown}"

# 2. --dry-run ---------------------------------------------------------------
echo ""
echo "── 2. Installer --dry-run (full plan preview)"
DRY_OUT=$(./anvil-nvidia-install.sh --dry-run 2>&1)
echo "$DRY_OUT"
echo ""
INSTALL_VER=$(echo "$DRY_OUT" | sed -n 's/.*Driver      : \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -n "$INSTALL_VER" ]; then
    report pass "driver version resolved: ${INSTALL_VER}"
else
    report fail "no driver version in dry-run (network blocked?)"
fi

# 3. Branch consistency ------------------------------------------------------
echo ""
echo "── 3. Branch consistency (generation -> driver major)"
MAJOR=$(printf '%s' "$INSTALL_VER" | cut -d. -f1)
case "$GEN" in
    modern)
        if [ "${MAJOR:-0}" -ge 590 ]; then
            report pass "Turing+ -> latest (${INSTALL_VER}), open-kernel era"
        else
            report fail "modern GPU got major ${MAJOR}, expected >= 590"
        fi
        ;;
    pascal)
        [ "$MAJOR" = "580" ] && report pass "Pascal -> 580 (${INSTALL_VER})" || report fail "Pascal got ${INSTALL_VER}, expected 580.x"
        ;;
    kepler)
        [ "$MAJOR" = "470" ] && report pass "Kepler -> 470 (${INSTALL_VER})" || report fail "Kepler got ${INSTALL_VER}, expected 470.x"
        ;;
    unknown)
        report warn "generation unknown; branch check skipped"
        ;;
esac

# 4. Real toolchain install --------------------------------------------------
echo ""
echo "── 4. Toolchain install (real, non-destructive)"
echo "  Running apt-get update + linux-headers + dkms + build-essential..."
if ./anvil-nvidia-install.sh --yes --dry-run 2>&1 | grep -q 'apt-get install -y linux-headers'; then
    report pass "toolchain command present in plan"
else
    report warn "toolchain command not found in plan"
fi
if apt-get update >/tmp/anvil-colab-apt.log 2>&1; then
    report pass "apt-get update OK"
else
    report warn "apt-get update failed (see /tmp/anvil-colab-apt.log) — Colab kernels are custom"
fi
KPKG="linux-headers-$(uname -r)"
if apt-get install -y "$KPKG" dkms build-essential >/tmp/anvil-colab-toolchain.log 2>&1; then
    report pass "toolchain installed (${KPKG}, dkms, build-essential)"
else
    report warn "toolchain install had issues (Colab custom kernel) — see /tmp/anvil-colab-toolchain.log"
fi
has_cmd() { command -v "$1" >/dev/null 2>&1; }
has_cmd gcc && report pass "gcc present" || report fail "gcc missing"
has_cmd dkms && report pass "dkms present" || report fail "dkms missing"
[ -d "/usr/src/linux-headers-$(uname -r)" ] || [ -d "/usr/lib/modules/$(uname -r)/build" ] && report pass "kernel headers present" || report warn "kernel headers not found for $(uname -r)"

# 5. Real .run download + validity -------------------------------------------
echo ""
echo "── 5. Real driver .run download + validity"
if [ -n "$INSTALL_VER" ]; then
    RUN_URL="https://download.nvidia.com/XFree86/Linux-x86_64/${INSTALL_VER}/NVIDIA-Linux-x86_64-${INSTALL_VER}.run"
    echo "  Downloading ${RUN_URL} (~200 MB, one time)..."
    if curl -fSL -o /tmp/anvil-driver.run "$RUN_URL"; then
        report pass "downloaded ${INSTALL_VER}.run"
        SIZE=$(wc -c < /tmp/anvil-driver.run)
        echo "  size: $SIZE bytes"
        [ "$SIZE" -gt 10485760 ] && report pass "size sanity OK (>10 MB)" || report fail "suspiciously small download"
        if sh /tmp/anvil-driver.run --version >/tmp/anvil-run-version.txt 2>&1; then
            report pass "installer is valid (--version worked)"
            head -3 /tmp/anvil-run-version.txt
        else
            report fail "installer --version failed"
            head -5 /tmp/anvil-run-version.txt
        fi
    else
        report fail "download failed for ${INSTALL_VER}"
    fi
fi

# 6. Optional destructive real install ---------------------------------------
if [ "$1" = "--install" ]; then
    echo ""
    echo "── 6. REAL INSTALL (destructive on Colab — driver swap, no reboot)"
    echo "  This will conflict with the loaded driver and likely kill the"
    echo "  session's GPU. Proceeding in 5s (Ctrl-C to abort)..."
    sleep 5
    ./anvil-nvidia-install.sh --yes
    echo "  exit code: $?"
    report warn "real install attempted — check nvidia-smi for outcome"
fi

echo ""
echo "══════════════════════════════════════"
echo "  TOTAL: ${PASS} pass, ${FAIL} fail, ${WARN} warn"
echo "══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
