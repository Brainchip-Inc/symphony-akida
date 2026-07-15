#!/bin/bash
# Print the /dev node names of Akida chips that enumerate healthily, one per line,
# AKD1500 first (preferred) then AKD1000 (NSoC_v2). Runs as the entrypoint of a
# privileged throwaway container: it exposes each host chip in isolation (as slot 0
# of its family in the container's private tmpfs /dev) and checks whether akida sees
# it. launch/up.sh uses this to place nodes only on good chips and skip any with a
# stuck DMA.
#
# Two chip families with different /dev nodes:
#   AKD1500 (PCI 1e7c:a500)  -> /dev/akd1500_<N>   (preferred: newer, more of them)
#   AKD1000 (PCI 1e7c:bca1)  -> /dev/akida<N>      (NSoC_v2)
# akida enumerates each family contiguously from slot 0, and enumerating ALL chips at
# once triggers DMA contention (multi-second stalls, dropped devices), so we probe one
# chip at a time -- recreate exactly one node at slot 0 and enumerate just that chip.
export PYTHONPATH=/opt/akida-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/opt/akida-venv/lib/python3.12/site-packages/akida.libs
PY=/opt/python3.12/bin/python3.12

declare -A MAJ MIN FAM
order=()
add() {  # $1 = node basename, $2 = family (akd1500|akida)
    local b="$1"
    MAJ[$b]=$((16#$(stat -c '%t' "/dev/$b")))
    MIN[$b]=$((16#$(stat -c '%T' "/dev/$b")))
    FAM[$b]="$2"
    order+=("$b")
}
# AKD1500 first (preferred), then AKD1000 -- launch/up.sh takes the first N in this order.
for d in /dev/akd1500_*;   do [ -e "$d" ] && add "${d#/dev/}" akd1500; done
for d in /dev/akida[0-9]*; do [ -e "$d" ] && add "${d#/dev/}" akida;   done

for b in "${order[@]}"; do
    find /dev -maxdepth 1 -name 'akida[0-9]*' -delete 2>/dev/null
    find /dev -maxdepth 1 -name 'akd1500_*'   -delete 2>/dev/null
    if [ "${FAM[$b]}" = akd1500 ]; then slot=/dev/akd1500_0; else slot=/dev/akida0; fi
    mknod "$slot" c "${MAJ[$b]}" "${MIN[$b]}" 2>/dev/null && chmod 666 "$slot"
    if timeout 30 "$PY" -c "import akida,sys; sys.exit(0 if akida.devices() else 1)" >/dev/null 2>&1; then
        echo "$b"
    fi
done
