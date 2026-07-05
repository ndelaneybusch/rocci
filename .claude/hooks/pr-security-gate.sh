#!/usr/bin/env bash
# PreToolUse gate: security-review the branch before `gh pr create` is allowed.
#
# Wired in .claude/settings.json under hooks.PreToolUse (matcher Bash, narrowed
# with `if: "Bash(gh pr create:*)"` so it fires on PR creation, not every shell
# command). Runs the /security-review skill in a fresh headless `claude -p`
# session — one dedicated reviewer process — and converts its verdict into
# PreToolUse semantics:
#   exit 0            review clean -> gh pr create proceeds
#   exit 2 + stderr   findings (or no verdict) -> the tool call is blocked and
#                     stderr is fed back to the calling Claude to act on
#
# The reviewer is read-only (--allowedTools below): a gate must not be able to
# edit the branch it is judging. A VERDICT sentinel is appended to its system
# prompt so pass/fail is parsed deterministically instead of grepping prose.
# Anything ambiguous — missing sentinel, CLI failure, timeout — fails CLOSED;
# SKIP_SECURITY_GATE=1 is the escape hatch for false positives or offline work.

set -u

[ "${SKIP_SECURITY_GATE:-}" = "1" ] && exit 0

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty')

# Defense in depth: the `if` filter already narrows to PR creation, but never
# start a multi-minute review for anything else.
case "$cmd" in
    *"gh pr create"*) ;;
    *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

if ! command -v claude >/dev/null 2>&1; then
    echo "pr-security-gate: 'claude' CLI not found; blocking PR creation." \
        "Set SKIP_SECURITY_GATE=1 to bypass." >&2
    exit 2
fi

# Inner timeout (540s) sits below the hook timeout (600s) so timeouts surface
# as our own fail-closed block rather than the harness's non-blocking treatment
# of a timed-out hook. Model pinned so gate latency doesn't drift with whatever
# heavyweight model the interactive session happens to run. stdin is redirected
# from /dev/null so the headless session cannot swallow hook input.
out=$(timeout 540 claude -p "/security-review" \
    --model claude-sonnet-5 \
    --append-system-prompt "End your final message with exactly one line: 'VERDICT: CLEAN' if there are no high-confidence, exploitable findings, otherwise 'VERDICT: VULNERABLE'. Before the verdict line, list each finding as file:line — severity — one-line fix." \
    --allowedTools "Read,Grep,Glob,Bash(git diff:*),Bash(git log:*),Bash(git status:*),Bash(git show:*),Bash(git merge-base:*),Bash(git branch:*)" \
    </dev/null 2>&1)
code=$?

verdict=$(printf '%s' "$out" | grep -oE 'VERDICT: (CLEAN|VULNERABLE)' | tail -1)

if [ "$code" -eq 0 ] && [ "$verdict" = "VERDICT: CLEAN" ]; then
    exit 0
fi

{
    if [ "$verdict" = "VERDICT: VULNERABLE" ]; then
        echo "Security review blocked this PR. Fix the findings below, then re-run gh pr create:"
    else
        echo "Security review did not complete (exit $code, no verdict); failing closed."
        echo "Re-run gh pr create to retry, or set SKIP_SECURITY_GATE=1 to bypass."
    fi
    echo
    printf '%s\n' "$out"
} >&2
exit 2
