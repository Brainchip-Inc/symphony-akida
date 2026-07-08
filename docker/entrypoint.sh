#!/bin/bash
# Entrypoint for the Symphony + Akida demo image.
#
#   HOST_ROLE=MANAGEMENT (default)
#     First boot (no shared ego.conf): init the cluster (join/setpassword/
#     setentitlement/mghost /shared), start EGO, and register the Akida SOAM
#     service if one is baked at /opt/akida-service.
#     Subsequent boots: just start EGO against the existing shared state.
#
#   HOST_ROLE=COMPUTE
#     Wait for the master's shared ego.conf, join, start EGO. The SOAM SIM
#     pulls the preloaded service package from the master.
#
# Everything shared lives under /shared (bind-mounted from the repo, owned by
# uid 1000 = egoadmin, so no host sudo is needed). Optional EGO_DEFINE_NCPUS
# (procs|cores|threads) lets the launcher shrink the license cores each host
# reports so the cluster stays under the CE 64-core cap.

set +e   # Symphony CLIs often return non-zero on idempotent re-runs

SHARED=/shared
: "${EGO_TOP:=/opt/ibm/spectrumcomputing}"
# profile.platform references unset vars internally; source it before `set -u`.
source "$EGO_TOP/profile.platform" >/dev/null 2>&1
set -o pipefail

role="${HOST_ROLE:-MANAGEMENT}"
admin_pw="${ADMIN_PASSWORD:-Admin}"
hostname_fq="$(hostname)"
akida_dir="/opt/akida-service"

log() { printf '[entrypoint %s] %s\n' "$hostname_fq" "$*"; }

# Optional per-host ncpus definition (procs|cores|threads). Combined with a
# container --cpuset-cpus, EGO_DEFINE_NCPUS=procs lets each host advertise
# fewer license cores; verified via `egosh resource list`.
if [ -n "${EGO_DEFINE_NCPUS:-}" ]; then
    log "EGO_DEFINE_NCPUS=$EGO_DEFINE_NCPUS"
fi

# SSH access (optional): pass -e SSH_PUBLIC_KEY to enable key login as egoadmin.
setup_ssh() {
    command -v sshd >/dev/null 2>&1 || { log "sshd not installed; skipping SSH"; return; }
    rm -f /run/nologin /etc/nologin 2>/dev/null
    [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A >/dev/null 2>&1
    if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
        install -d -m 700 -o egoadmin -g egoadmin /home/egoadmin/.ssh
        printf '%s\n' "$SSH_PUBLIC_KEY" > /home/egoadmin/.ssh/authorized_keys
        chmod 600 /home/egoadmin/.ssh/authorized_keys
        chown egoadmin:egoadmin /home/egoadmin/.ssh/authorized_keys
        log "installed egoadmin authorized_keys"
    fi
    pgrep -x sshd >/dev/null 2>&1 || { /usr/sbin/sshd && log "sshd started on :22"; }
}
setup_ssh

# egoconfig insists on running as the cluster admin; the container runs as root
# (LIM needs it), so bounce those calls through su with the profile re-sourced.
as_egoadmin() {
    su - egoadmin -c "source $EGO_TOP/profile.platform >/dev/null 2>&1; $*"
}

register_akida_service() {
    local xml="$akida_dir/AkidaGenericService.xml"
    local pkg="$akida_dir/AkidaGenericServicePackage.v1.tar.gz"
    if [ ! -f "$xml" ] || [ ! -f "$pkg" ]; then
        log "no Akida service baked at $akida_dir; skipping registration"
        return
    fi
    if soamview app AkidaGenericService >/dev/null 2>&1; then
        log "AkidaGenericService already registered; skipping"
        return
    fi
    log "registering AkidaGenericService"
    egosh user logon -u Admin -x "$admin_pw" >/dev/null

    echo y | egosh consumer add /AkidaServices                    -a Admin -e egoadmin 2>/dev/null || true
    echo y | egosh consumer add /AkidaServices/GenericInference   -a Admin -e egoadmin 2>/dev/null || true
    egosh consumer addrg /AkidaServices                   -g ComputeHosts    2>/dev/null || true
    egosh consumer addrg /AkidaServices/GenericInference  -g ComputeHosts    2>/dev/null || true

    soamdeploy add AkidaGenericServicePackage.v1 -p "$pkg" -c /AkidaServices/GenericInference -f
    soamreg "$xml" -f
    echo Y | soamcontrol app enable AkidaGenericService 2>/dev/null || true
    log "AkidaGenericService registered + enabled"
}

# Seed the shared models dir from a baked/mounted source, if present.
seed_models() {
    local src="${AKIDA_MODELS_SRC:-$akida_dir/demo-models}"
    [ -d "$src" ] || { log "no model source at $src; skipping seed"; return; }
    install -d -o egoadmin -g egoadmin "$SHARED/models"
    cp -n "$src"/*.fbz "$src"/*.json "$SHARED/models/" 2>/dev/null || true
    chown -R egoadmin:egoadmin "$SHARED/models" 2>/dev/null || true
    log "seeded models into $SHARED/models"
}

case "$role" in
MANAGEMENT)
    if [ ! -f "$SHARED/kernel/conf/ego.conf" ]; then
        log "first boot: initializing cluster as primary master"
        chown egoadmin:egoadmin "$SHARED" 2>/dev/null || true
        as_egoadmin "egoconfig join $hostname_fq -f"
        as_egoadmin "egoconfig setpassword -x $admin_pw -f"
        ent="$EGO_TOP/kernel/conf/sym_com_entitlement.dat"
        [ -f "$ent" ] && as_egoadmin "egoconfig setentitlement $ent" || log "WARN: no entitlement at $ent"
        as_egoadmin "egoconfig mghost $SHARED -f"
        source "$EGO_TOP/profile.platform" >/dev/null 2>&1
        seed_models
        first_boot=1
    else
        log "master recovery: shared state present"
        source "$EGO_TOP/profile.platform" >/dev/null 2>&1
        first_boot=0
    fi

    log "starting EGO (as root)"
    egosh ego start

    for i in {1..30}; do
        egosh ego info 2>/dev/null | grep -q "primary host name" && break
        sleep 2
    done

    [ "$first_boot" = "1" ] && register_akida_service
    ;;
COMPUTE)
    log "waiting for master shared ego.conf..."
    until [ -f "$SHARED/kernel/conf/ego.conf" ]; do sleep 3; done
    master="$(awk -F= '/EGO_MASTER_LIST/{print $2}' "$SHARED/kernel/conf/ego.conf" | tr -d '"')"
    [ -n "$master" ] || { log "ERROR: EGO_MASTER_LIST empty in shared ego.conf" >&2; exit 1; }
    log "joining cluster, master=$master"
    as_egoadmin "egoconfig join $master -f"
    source "$EGO_TOP/profile.platform" >/dev/null 2>&1
    log "starting EGO (as root)"
    egosh ego start
    ;;
*)
    log "HOST_ROLE=$role (unknown); dropping to shell"
    exec bash
    ;;
esac

log "ready"
if [ "${1:-}" = "--hold" ]; then
    tail -f /dev/null
else
    exec "$@"
fi
