#!/bin/bash
set -e

BIN="${1:-./build/anvil}"

if [ ! -x "$BIN" ]; then
    echo "missing executable: $BIN"
    exit 1
fi

echo "version check"
"$BIN" --version

echo "help check"
"$BIN" --help

echo "invalid model check"
set +e
"$BIN" run /tmp/not-a-model.gguf 2>/dev/null
RC=$?
set -e
if [ "$RC" -eq 0 ]; then
    echo "expected non-zero exit for invalid model, got $RC"
    exit 1
fi

echo "invalid numeric argument check"
set +e
"$BIN" run /tmp/not-a-model.gguf --ctx notanumber 2>/dev/null
RC=$?
set -e
if [ "$RC" -eq 0 ]; then
    echo "expected non-zero exit for invalid --ctx, got $RC"
    exit 1
fi

echo "unknown option check"
set +e
"$BIN" run /tmp/not-a-model.gguf --notarealoption 2>/dev/null
RC=$?
set -e
if [ "$RC" -eq 0 ]; then
    echo "expected non-zero exit for unknown option, got $RC"
    exit 1
fi

echo "smoke tests passed"
