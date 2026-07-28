#!/usr/bin/env bash
# Reclaim host ownership of .cluster/shared before wiping it.
#
# Every container service (EGO, SOAM instances, egoconfig) runs as the image's
# baked-in egoadmin (uid 1000), so anything written to the bind-mounted
# $SHARED ends up owned by whatever host account happens to have uid 1000 --
# not necessarily the user running up.sh/down.sh. A one-off container (root by
# default, entrypoint bypassed) chowns it back to the current host user so
# neither script ever needs sudo, regardless of which containers are running.
set -euo pipefail

SHARED="$1"
IMAGE="${IMAGE:-symphony-akida-demo:local}"
[ -d "$SHARED" ] || exit 0

docker run --rm --entrypoint /usr/bin/chown -v "$SHARED:/shared" "$IMAGE" \
    -R "$(id -u):$(id -g)" /shared >/dev/null 2>&1 || true
