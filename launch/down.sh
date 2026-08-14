#!/usr/bin/env bash
# Tear down the Symphony + Akida cluster and wipe repo-local state.
#
# Run `launch/down.sh --help` for the details; usage() below is the single source.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: launch/down.sh
       launch/down.sh -h | --help

Tear down the Symphony + Akida cluster:
  1. remove the symphony-master and symphony-compute-* containers
  2. remove the docker network
  3. reclaim host ownership of .cluster/shared
  4. delete .cluster/

Safe to run when nothing is up: it reports what it removed either way, so it is
also the way to clear a half-started cluster before trying again.

Never needs sudo. Everything the cluster writes to .cluster/shared is owned by
the container's egoadmin (uid 1000), so a throwaway container chowns it back to
you first -- see launch/reclaim_shared.sh.

Options
  -h, --help   show this and exit.

Environment overrides
  NETWORK   docker network name                        (default symcluster)
  IMAGE     image used for the ownership reclaim step  (default symphony-akida)

Bring a cluster up with launch/up.sh (run it with --help for the app list).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0;;
        *) echo "unexpected argument: '$1' (see --help)" >&2; exit 1;;
    esac
done

NETWORK="${NETWORK:-symcluster}"

removed=0
for c in $(docker ps -aq --filter "name=symphony-master" --filter "name=symphony-compute-"); do
    docker rm -f "$c" >/dev/null && removed=$((removed+1))
done
docker network rm "$NETWORK" >/dev/null 2>&1 || true
"$HERE/launch/reclaim_shared.sh" "$HERE/.cluster/shared"
rm -rf "$HERE/.cluster"
echo "[down] removed $removed container(s), network '$NETWORK', and .cluster/"
