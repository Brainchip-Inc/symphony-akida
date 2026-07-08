#!/bin/bash
# Print the indices of Akida chips that enumerate healthily, one per line.
# Runs as the entrypoint of a privileged throwaway container: it remaps each
# host chip to /dev/akida0 in turn (private tmpfs /dev) and checks whether
# akida sees it. launch/up.sh uses this to place nodes only on good chips and
# skip any with a stuck DMA. akida enumerates /dev/akida0.. contiguously, so we
# expose exactly one chip at a time as akida0.
export PYTHONPATH=/opt/akida-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/opt/akida-venv/lib/python3.12/site-packages/akida.libs
PY=/opt/python3.12/bin/python3.12

declare -A MAJ MIN
present=""
for d in /dev/akida[0-9]*; do
    [ -e "$d" ] || continue
    i="${d#/dev/akida}"
    MAJ[$i]=$((16#$(stat -c '%t' "$d")))
    MIN[$i]=$((16#$(stat -c '%T' "$d")))
    present="$present $i"
done

for i in $present; do
    rm -f /dev/akida[0-9]*
    mknod /dev/akida0 c "${MAJ[$i]}" "${MIN[$i]}" 2>/dev/null && chmod 666 /dev/akida0
    if timeout 20 "$PY" -c "import akida,sys; sys.exit(0 if akida.devices() else 1)" >/dev/null 2>&1; then
        echo "$i"
    fi
done
