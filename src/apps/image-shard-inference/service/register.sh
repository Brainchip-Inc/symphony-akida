#!/bin/bash
# Register + enable the three shard-pipeline SOAM services, then place them:
#   ShardInferenceService -> one instance per Akida chip (ComputeHosts, EqualFreeSlot)
#   ShardSegmentService / ShardStitchService -> CPU instances on the management host
# Run inside the master after all compute nodes are ok:
#   docker exec symphony-master /opt/akida-shard-service/register.sh <num_nodes>
set +e
EGO_TOP=/opt/ibm/spectrumcomputing
source "$EGO_TOP/profile.platform" >/dev/null 2>&1

N="${1:-1}"                                   # inference instances = chip count
SEG_SIS="${SHARD_SEG_SIS:-1}"                 # segment instances on the mgmt host
STITCH_SIS="${SHARD_STITCH_SIS:-1}"           # stitch instances on the mgmt host
DIR=/opt/akida-shard-service
CT=/shared/kernel/conf/ConsumerTrees.xml

egosh user logon -u Admin -x "${ADMIN_PASSWORD:-Admin}" >/dev/null 2>&1

# Wait for the Session Director to accept soam commands.
for i in $(seq 1 40); do soamview app >/dev/null 2>&1 && break; sleep 3; done

# Consumer hierarchy. ComputeHosts runs the inference instances; ManagementHosts runs the
# segment/stitch instances AND every app's SSM (session manager) needs a mgmt slot to start.
echo y | egosh consumer add /AkidaShard          -a Admin -e egoadmin 2>/dev/null
for leaf in Segment Inference Stitch; do
    echo y | egosh consumer add "/AkidaShard/$leaf" -a Admin -e egoadmin 2>/dev/null
done
for c in /AkidaShard /AkidaShard/Segment /AkidaShard/Inference /AkidaShard/Stitch; do
    egosh consumer addrg "$c" -g ComputeHosts    2>/dev/null
    egosh consumer addrg "$c" -g ManagementHosts 2>/dev/null
done

# Deploy package + register profile (preload count patched) + enable, for each service.
reg_one() {  # <pkg> <consumer> <xml> <app> <preload>
    local pkg="$1" cons="$2" xml="$3" app="$4" preload="$5"
    soamdeploy add "$pkg" -p "$DIR/$pkg.tar.gz" -c "$cons" -f
    sed -E "s/numOfSlotsForPreloadedServices=\"[0-9]+\"/numOfSlotsForPreloadedServices=\"$preload\"/" \
        "$xml" > "/tmp/reg-$app.xml"
    soamreg "/tmp/reg-$app.xml" -f
    echo Y | soamcontrol app enable "$app" 2>/dev/null
}

reg_one ShardSegmentServicePackage.v1   /AkidaShard/Segment   "$DIR/segment/ShardSegmentService.xml"     ShardSegmentService   "$SEG_SIS"
reg_one ShardInferenceServicePackage.v1 /AkidaShard/Inference "$DIR/inference/ShardInferenceService.xml" ShardInferenceService "$N"
reg_one ShardStitchServicePackage.v1    /AkidaShard/Stitch    "$DIR/stitch/ShardStitchService.xml"       ShardStitchService    "$STITCH_SIS"

# Spread the inference instances one-per-chip: the shipped ComputeHosts tree leaves DistributeBy
# empty (packs one host), so set EqualFreeSlot, apply it, then bounce the inference app so EGO
# re-places its instances under the new policy. (No-op if a prior app already set it.)
sed -i '/<ResourceGroupName>ComputeHosts<\/ResourceGroupName>/,+2 s#<PolicyParameter ParameterName="DistributeBy"/>#<PolicyParameter ParameterName="DistributeBy">EqualFreeSlot</PolicyParameter>#' "$CT"
egosh consumer applyresplan -f "$CT" 2>/dev/null
echo Y | soamcontrol app disable ShardInferenceService 2>/dev/null
sleep 10
echo Y | soamcontrol app enable ShardInferenceService 2>/dev/null

echo "[register] done (inference=$N seg=$SEG_SIS stitch=$STITCH_SIS)"
for app in ShardSegmentService ShardInferenceService ShardStitchService; do
    echo "--- $app ---"
    soamview app "$app" 2>&1 | head -5
done
