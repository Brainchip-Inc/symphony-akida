#!/usr/bin/env bash
#
# scripts/commit_msg_hook.sh
# Enforces commit messages of the form:
#   type(scope): message with at least three words
# or
#   type: message with at least three words
#

set -euo pipefail

commit_msg_file="$1"
subject_line="$(head -n 1 "$commit_msg_file" | tr -d '\r')"

# If you ever need to bypass for a one-off:
#   SKIP_COMMIT_MSG_CHECK=1 git commit ...
if [[ "${SKIP_COMMIT_MSG_CHECK:-0}" == "1" ]]; then
  echo "⚠️  SKIP_COMMIT_MSG_CHECK=1 set, skipping commit message validation."
  exit 0
fi

# Ignore merge commits created by Git
if [[ "$subject_line" =~ ^Merge\  ]]; then
  echo "ℹ️  Merge commit detected, skipping commit message validation."
  exit 0
fi

allowed_types="feat fix docs style refactor perf test build ci chore revert"

echo "Checking commit message: '$subject_line'"

# 1) Require ": " to split prefix and subject
if [[ "$subject_line" != *": "* ]]; then
  echo "❌ ERROR: Commit message must start with 'type(scope): message' or 'type: message'"
  echo "   Examples:"
  echo "     feat(api): add drift compensation"
  echo "     fix(kws): handle zero-length audio input"
  exit 1
fi

prefix="${subject_line%%:*}"   # text before the first colon
subject="${subject_line#*: }"  # text after ": "

# 2) Extract type from prefix (before optional (scope))
type="${prefix%%(*}"

# 3) Check type is allowed
if ! echo "$allowed_types" | grep -qw "$type"; then
  echo "❌ ERROR: Invalid type '$type'."
  echo "   Allowed types: $allowed_types"
  exit 1
fi

# 4) Ensure subject (message) has at least 3 words
word_count="$(echo "$subject" | wc -w | tr -d ' ')"

if [[ "$word_count" -lt 3 ]]; then
  echo "❌ ERROR: Commit message subject must be at least 3 words."
  echo "   Current subject: '$subject'"
  echo "   Example: $type(api): add drift compensation"
  exit 1
fi

echo "✅ Commit message OK."
exit 0
