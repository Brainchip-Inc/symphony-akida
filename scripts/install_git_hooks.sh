#!/usr/bin/env bash
#
# scripts/install_git_hooks.sh
# One-time script to install the local commit-msg hook for this repo.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
SOURCE_HOOK="$REPO_ROOT/scripts/commit_msg_hook.sh"
TARGET_HOOK="$HOOKS_DIR/commit-msg"

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  echo "❌ This does not look like a git repository (no .git directory)."
  exit 1
fi

mkdir -p "$HOOKS_DIR"

# Copy instead of symlink to keep it simple and cross-platform
cp "$SOURCE_HOOK" "$TARGET_HOOK"
chmod +x "$TARGET_HOOK"

echo "✅ Installed commit-msg hook at:"
echo "   $TARGET_HOOK"
echo ""
echo "From now on, commits in this repo must look like:"
echo "   type(scope): message with at least three words"
echo ""
echo "Examples:"
echo "   feat(api): add fusion pipeline"
echo "   fix(kws): handle zero-length audio input"
