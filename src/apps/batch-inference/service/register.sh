#!/bin/bash
# Register + enable AkidaGenericService, then apply one-per-host spreading.
# Run inside the master after all compute nodes are ok:
#   docker exec symphony-master /opt/akida-service/register.sh <num_nodes>
set +e
EGO_TOP=/opt/ibm/spectrumcomputing
source "$EGO_TOP/profile.platform" >/dev/null 2>&1

N="${1:-1}"
DIR=/opt/akida-service
CT=/shared/kernel/conf/ConsumerTrees.xml

egosh user logon -u Admin -x "${ADMIN_PASSWORD:-Admin}" >/dev/null 2>&1

# Wait for the Session Director to accept soam commands.
for i in $(seq 1 40); do soamview app >/dev/null 2>&1 && break; sleep 3; done

# Consumer hierarchy. ComputeHosts runs the instances; ManagementHosts is
# required so the app's SSM (session manager) can get a slot and start.
echo y | egosh consumer add /AkidaServices                  -a Admin -e egoadmin 2>/dev/null
echo y | egosh consumer add /AkidaServices/GenericInference -a Admin -e egoadmin 2>/dev/null
for c in /AkidaServices /AkidaServices/GenericInference; do
    egosh consumer addrg "$c" -g ComputeHosts    2>/dev/null
    egosh consumer addrg "$c" -g ManagementHosts 2>/dev/null
done

# Deploy package + register the profile (patched to preload N instances) + enable.
soamdeploy add AkidaGenericServicePackage.v1 -p "$DIR/AkidaGenericServicePackage.v1.tar.gz" -c /AkidaServices/GenericInference -f
sed -E "s/numOfSlotsForPreloadedServices=\"[0-9]+\"/numOfSlotsForPreloadedServices=\"$N\"/" \
    "$DIR/AkidaGenericService.xml" > /tmp/reg.xml
soamreg /tmp/reg.xml -f
echo Y | soamcontrol app enable AkidaGenericService 2>/dev/null

# Spread one instance per host: the shipped ComputeHosts tree leaves DistributeBy
# empty (packs one host), so set EqualFreeSlot, apply it, then bounce the app so
# EGO re-places the instances cleanly under the new policy.
sed -i '/<ResourceGroupName>ComputeHosts<\/ResourceGroupName>/,+2 s#<PolicyParameter ParameterName="DistributeBy"/>#<PolicyParameter ParameterName="DistributeBy">EqualFreeSlot</PolicyParameter>#' "$CT"
egosh consumer applyresplan -f "$CT" 2>/dev/null
echo Y | soamcontrol app disable AkidaGenericService 2>/dev/null
sleep 10
echo Y | soamcontrol app enable AkidaGenericService 2>/dev/null

echo "[register] done (nodes=$N)"
soamview app AkidaGenericService 2>&1 | head -8
