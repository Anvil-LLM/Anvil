#!/bin/bash
set -e
BIN="${1:-./build/anvil}"
if [ ! -x "$BIN" ]; then echo "missing executable: $BIN"; exit 1; fi

echo "version check"
"$BIN" --version || { echo "version check failed"; exit 1; }

echo "help check"
"$BIN" --help >/dev/null || { echo "help check failed"; exit 1; }

expect_nonzero() {
    desc="$1"; shift
    set +e
    "$@" >/dev/null 2>&1
    RC=$?
    set -e
    if [ "$RC" -eq 0 ]; then echo "expected non-zero exit for $desc"; exit 1; fi
}

echo "invalid model check"
expect_nonzero "invalid model" "$BIN" run /tmp/not-a-model.gguf

echo "invalid numeric argument check"
expect_nonzero "invalid --ctx" "$BIN" run /tmp/not-a-model.gguf --ctx notanumber
expect_nonzero "negative --ctx" "$BIN" run /tmp/not-a-model.gguf --ctx -5
expect_nonzero "oversized --ctx" "$BIN" run /tmp/not-a-model.gguf --ctx 999999999999
expect_nonzero "invalid --temp" "$BIN" run /tmp/not-a-model.gguf --temp notanumber
expect_nonzero "oversized --temp" "$BIN" run /tmp/not-a-model.gguf --temp 99
expect_nonzero "invalid --top-k" "$BIN" run /tmp/not-a-model.gguf --top-k -1
expect_nonzero "invalid --top-p" "$BIN" run /tmp/not-a-model.gguf --top-p 1.5

echo "unknown option check"
expect_nonzero "unknown option" "$BIN" run /tmp/not-a-model.gguf --notarealoption

echo "removed dead flag check"
expect_nonzero "removed --triattn" "$BIN" run /tmp/not-a-model.gguf --triattn

echo "no model check"
expect_nonzero "no model" "$BIN"

echo "registry + profile checks (isolated HOME)"
OLD_HOME="$HOME"
export HOME="$(mktemp -d)"
printf 'GGUF' > "$HOME/fake.gguf"

"$BIN" models | grep -q 'No models' || { echo "registry not empty initially"; exit 1; }

"$BIN" models import "$HOME/fake.gguf" --name test-model >/dev/null 2>&1 \
    || { echo "import failed"; exit 1; }
"$BIN" models | grep -q 'test-model' || { echo "import not listed"; exit 1; }

expect_nonzero "duplicate import" "$BIN" models import "$HOME/fake.gguf" --name test-model
expect_nonzero "non-gguf import" "$BIN" models import /etc/hosts --name nope

"$BIN" profile test-model set temp=0.9 type_k=turbo4 n_ctx=8192 >/dev/null 2>&1 \
    || { echo "profile set failed"; exit 1; }
"$BIN" profile test-model | grep -q 'temp\s*= 0.9' || { echo "profile temp missing"; exit 1; }
"$BIN" profile test-model | grep -q 'turbo4' || { echo "profile type_k missing"; exit 1; }
expect_nonzero "profile bad key" "$BIN" profile test-model set bogus=1
expect_nonzero "profile bad value" "$BIN" profile test-model set temp=abc

"$BIN" rm test-model >/dev/null 2>&1 || { echo "rm failed"; exit 1; }
"$BIN" models | grep -q 'No models' || { echo "model not removed"; exit 1; }

expect_nonzero "unknown registry name" "$BIN" run not-a-real-model

rm -rf "$HOME"
export HOME="$OLD_HOME"

echo "smoke tests passed"
