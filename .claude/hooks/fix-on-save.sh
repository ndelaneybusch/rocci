#!/usr/bin/env bash
# Stop-hook fix-on-finish for Python files.
#
# Wired in .claude/settings.json under hooks.Stop (no matcher — Stop has no tool).
# Fires when Claude finishes a turn. Unlike a PostToolUse hook there is no
# tool_input.file_path, so we recover the .py files Claude edited from the session
# transcript, then run, inside a SINGLE `uv run` (one env resolution):
#   ruff check --fix   autofix lint (incl. import sorting via the I rules)
#   ruff format        formatting
#   ty check --fix     autofix the subset of type diagnostics ty can fix
#
# If anything remains broken after autofix we return {"decision":"block"} on stdout
# (exit 0 — JSON is ignored on exit 2), which keeps Claude working instead of
# yielding to the user, feeding the diagnostics back as the reason.
#
# stop_hook_active guards the loop: when true, Claude is ALREADY continuing because
# of a previous block, so we never block again — we still tidy the files, then let
# it stop. Net effect: exactly one push-back per stretch of work.
#
# NOTE: ty is in preview; `ty check --fix` covers a narrow set of diagnostics and
# the flag may not exist in older builds. If your ty rejects --fix, drop it.

set -u

payload=$(cat)
transcript=$(printf '%s' "$payload" | jq -r '.transcript_path // empty')
active=$(printf '%s' "$payload" | jq -r '.stop_hook_active // false')

[ -f "$transcript" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

# Every .py path Claude edited this session (deduped). Whole-session scope is
# deliberate: the fixers are idempotent, so re-touching already-clean files is a
# cheap no-op, and it catches anything that regressed across earlier turns.
mapfile -t edited < <(
    jq -r '
        select(.type=="assistant")
        | .message.content[]?
        | select(.type=="tool_use"
                 and (.name=="Edit" or .name=="Write" or .name=="MultiEdit"))
        | .input.file_path? // empty
        | select(endswith(".py"))
    ' "$transcript" 2>/dev/null | sort -u
)
[ "${#edited[@]}" -eq 0 ] && exit 0

# Keep only files that still exist (an edited-then-deleted file shouldn't fix).
files=()
for p in "${edited[@]}"; do
    [ -f "$p" ] && files+=("$p")
done
[ "${#files[@]}" -eq 0 ] && exit 0

# One env resolution; bare ruff/ty resolve via the activated venv on PATH.
# status flips to 1 on ANY non-zero tool — explicit, unlike a bitwise OR of codes.
out=$(uv run --quiet -- bash -c '
    status=0
    ruff check --fix "$@" || status=1
    ruff format "$@"       || status=1
    ty check --fix "$@"    || status=1
    exit "$status"
' _ "${files[@]}" 2>&1)
code=$?

# Issues remain AND this is our first pass: block so Claude resolves them before
# finishing. On a continuation pass (active=true) we've already tidied above and
# just let Claude stop.
if [ "$code" -ne 0 ] && [ "$active" != "true" ]; then
    jq -n --arg r "$out" \
        '{decision:"block",
          reason:("Lint/type issues remain after autofix. Resolve these before finishing:\n\n" + $r)}'
    exit 0
fi

exit 0