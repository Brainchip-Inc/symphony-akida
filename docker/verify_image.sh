#!/usr/bin/env bash
# The image's own acceptance test. Runs twice during the build and is installed as
# /usr/local/bin/verify-image so it can be re-run against any built image or any
# live container:
#
#     docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full
#     docker exec symphony-master verify-image --full
#
# Every check here corresponds to something this repo depends on silently. Because
# the platform half of the image now comes out of a different base OS than the
# code was written against, these need to be build failures rather than runtime
# mysteries three layers down in vemkd.log.
#
#     --platform  the base-swap-critical half; run before the app layers exist
#     --full      everything, including the app layers
set -uo pipefail

MODE=platform
EGO_TOP="${EGO_TOP:-/opt/ibm/spectrumcomputing}"
SYM_VERSION="${SYM_VERSION:-7.3.2}"
EGO_VERSION="${EGO_VERSION:-4.0}"
PY_VERSION="${PY_VERSION:-}"
AKIDA_VERSION="${AKIDA_VERSION:-}"
NUMPY_VERSION="${NUMPY_VERSION:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --platform) MODE=platform; shift ;;
        --full)     MODE=full; shift ;;
        --sym-version)   SYM_VERSION="$2"; shift 2 ;;
        --ego-version)   EGO_VERSION="$2"; shift 2 ;;
        --py-version)    PY_VERSION="$2"; shift 2 ;;
        --akida-version) AKIDA_VERSION="$2"; shift 2 ;;
        --numpy-version) NUMPY_VERSION="$2"; shift 2 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

PY312=/opt/python3.12/bin/python3.12
VENV_SP=/opt/akida-venv/lib/python3.12/site-packages
SOAM_LIB="$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/lib64"
FAILED=0

ok()   { printf '[verify]  ok   %s\n' "$*"; }
bad()  { printf '[verify] FAIL  %s\n' "$*"; FAILED=$((FAILED + 1)); }
warn() { printf '[verify] warn  %s\n' "$*"; }
chk()  { local d="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$d"; else bad "$d"; fi; }

printf '[verify] mode=%s EGO_TOP=%s SYM_VERSION=%s\n' "$MODE" "$EGO_TOP" "$SYM_VERSION"

# --- 1. platform assumptions the launcher relies on ------------------------
# akida ships x86_64 wheels only and the harvested tree is linux-x86_64.
chk "arch is x86_64" test "$(uname -m)" = x86_64
# The image must run as root: LIM needs it, and launch/reclaim_shared.sh does
# `docker run --entrypoint /usr/bin/chown` to hand /shared back to the host user
# without sudo. A USER line in the Dockerfile would break launch/down.sh.
chk "runs as uid 0 (no USER line)" test "$(id -u)" = 0
chk "/usr/bin/chown exists (reclaim_shared.sh entrypoint)" test -x /usr/bin/chown
chk "egoadmin is 1000:1000" test "$(id -u egoadmin):$(id -g egoadmin)" = "1000:1000"
for t in su install mknod stat find pgrep awk hostname nc bc ed diff tar gzip openssl; do
    chk "tool: $t" command -v "$t"
done
chk "tool: sshd" test -x /usr/sbin/sshd
chk "tool: ssh-keygen" command -v ssh-keygen

# --- 2. the two interpreters ----------------------------------------------
# python3.6 is not a nicety: pythonapi_3.6.7 is a sourceless soamapi.pyc plus
# SoamFactory.so, frozen to the CPython 3.6 ABI.
chk "/usr/bin/python3.6 exists" test -x /usr/bin/python3.6
v36="$(/usr/bin/python3.6 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
[ "$v36" = "3.6" ] && ok "python3.6 is $(/usr/bin/python3.6 -V 2>&1)" \
                   || bad "python3.6 reports '$v36', the soamapi .pyc needs 3.6"

chk "$PY312 exists" test -x "$PY312"
got312="$("$PY312" -c 'import platform;print(platform.python_version())' 2>/dev/null)"
if [ -n "$PY_VERSION" ]; then
    [ "$got312" = "$PY_VERSION" ] && ok "python3.12 is $got312" \
                                  || bad "python3.12 is '$got312', expected $PY_VERSION"
else
    ok "python3.12 is $got312"
fi
chk "python3.12 shared libs resolve" bash -c "! ldd $PY312 | grep -q 'not found'"
chk "python3.12 has the stdlib the demo uses" "$PY312" -c \
    'import ssl,zlib,bz2,lzma,ctypes,mmap,hashlib,select,fcntl,json,array,struct,zipfile'
chk "/usr/local/bin/python3.12 symlink" test -x /usr/local/bin/python3.12

# --- 3. soamapi under python3.6 -- the single most important check ---------
# If this passes, the SOAM half of all three apps can work. It is also the check
# most likely to break on a base change: SoamFactory.so is a RHEL7-era shared
# object resolving libsoam* out of lib64, which IBM's configure_image.sh turned
# into symlinks into $EGO_VERSION/linux-x86_64/lib.
chk "pythonapi_3.6.7 present" test -d "$SOAM_LIB/pythonapi_3.6.7"
if PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7" \
   LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/lib" \
   /usr/bin/python3.6 -c 'import soamapi; print(soamapi.__file__)' >/tmp/.v_soam 2>&1; then
    ok "import soamapi -> $(cat /tmp/.v_soam)"
else
    bad "import soamapi FAILED:"; sed 's/^/[verify]        /' /tmp/.v_soam
fi
rm -f /tmp/.v_soam

# --- 4. akida under python3.12 --------------------------------------------
# No akida.devices() call: the builder has no /dev/akida*, and a build must never
# depend on hardware. probe_chips.sh does the hardware probe at launch time.
if PYTHONPATH="$VENV_SP" LD_LIBRARY_PATH="$VENV_SP/akida.libs" \
   "$PY312" -c 'import numpy,akida;print(akida.__version__,numpy.__version__)' \
   >/tmp/.v_akida 2>&1; then
    read -r gotak gotnp < /tmp/.v_akida
    ok "import akida,numpy -> akida $gotak, numpy $gotnp"
    [ -z "$AKIDA_VERSION" ] || [ "$gotak" = "$AKIDA_VERSION" ] \
        || bad "akida is $gotak, expected $AKIDA_VERSION"
    [ -z "$NUMPY_VERSION" ] || [ "$gotnp" = "$NUMPY_VERSION" ] \
        || bad "numpy is $gotnp, expected $NUMPY_VERSION"
else
    bad "import akida FAILED (glibc / GLIBCXX too old?):"
    sed 's/^/[verify]        /' /tmp/.v_akida
fi
rm -f /tmp/.v_akida
chk "akida.libs is in the ldconfig cache" bash -c "ldconfig -p | grep -q akida"
chk "libakida.so.2 deps resolve" bash -c \
    "! LD_LIBRARY_PATH=$VENV_SP/akida.libs ldd $VENV_SP/akida/libakida.so.2 | grep -q 'not found'"

# --- 5. ldd closure across the base change --------------------------------
# The biggest ABI risk: these binaries were built for RHEL 7 (glibc 2.17) and now
# run on glibc 2.28. Targeted at the binaries the demo actually runs, so no
# allowlist file is needed.
LDPATH="$EGO_TOP/$EGO_VERSION/linux-x86_64/lib:$SOAM_LIB:$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/lib"
for b in "$EGO_TOP/$EGO_VERSION/linux-x86_64/bin/egosh" \
         "$EGO_TOP/$EGO_VERSION/linux-x86_64/bin/egoconfig" \
         "$EGO_TOP/$EGO_VERSION/linux-x86_64/etc/lim" \
         "$EGO_TOP/$EGO_VERSION/linux-x86_64/etc/pem" \
         "$EGO_TOP/$EGO_VERSION/linux-x86_64/etc/vemkd" \
         "$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/etc/sd" \
         "$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/etc/ssm" \
         "$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/bin/soamview" \
         "$EGO_TOP/soam/$SYM_VERSION/linux-x86_64/bin/soamdeploy" \
         "$SOAM_LIB/pythonapi_3.6.7/SoamFactory.so"; do
    [ -e "$b" ] || { warn "absent, not fatal: $b"; continue; }
    miss="$(LD_LIBRARY_PATH="$LDPATH" ldd "$b" 2>/dev/null | grep 'not found' || true)"
    if [ -z "$miss" ]; then
        ok "ldd $(basename "$b")"
    else
        bad "ldd $b has unresolved deps:"
        printf '%s\n' "$miss" | sed 's/^/[verify]        /'
    fi
done
# The bundled JRE is a RHEL7-era build; Liberty (the :8443 console) and the
# keytool that built our keystores both need it working on glibc 2.28.
JAVA="$(find "$EGO_TOP/jre" -type f -name java -perm -u+x 2>/dev/null | head -1)"
if [ -n "$JAVA" ]; then
    chk "bundled JRE runs" "$JAVA" -version
else
    bad "no java under $EGO_TOP/jre"
fi

# --- 6. the profiles resolve BINARY_TYPE on this kernel -------------------
# profile.soam and profile.perf historically closed their kernel-major list at 5,
# which silently produced .../soam/<v>/fail/... paths. docker/patch_symphony.sh
# extends the list; this proves the result rather than the edit.
probe="$(bash -c ". $EGO_TOP/profile.platform >/dev/null 2>&1;
                  printf '%s|%s|%s|%s' \"\${BINARY_TYPE:-}\" \"\${SOAM_HOME:-}\" \
                         \"\$(command -v egosh)\" \
                         \"\${PATH:-}:\${LD_LIBRARY_PATH:-}:\${PYTHONPATH:-}\"" 2>/dev/null)"
IFS='|' read -r p_bt p_sh p_egosh p_paths <<<"$probe"
[ "$p_bt" = "linux-x86_64" ] && ok "BINARY_TYPE=$p_bt on kernel $(uname -r)" \
    || bad "BINARY_TYPE='$p_bt' on kernel $(uname -r), expected linux-x86_64 -- the kernel-major check in profile.soam/profile.perf did not take"
[ -d "${p_sh:-/nonexistent}" ] && ok "SOAM_HOME=$p_sh" || bad "SOAM_HOME='$p_sh' is not a directory"
[ -n "$p_egosh" ] && ok "egosh on PATH at $p_egosh" || bad "egosh not on PATH after profile.platform"
# The consequence of an unresolved BINARY_TYPE is .../soam/<v>/fail/... on the
# search paths, which is the actual failure the kernel patch prevents.
case "$p_paths" in
    */fail/*) bad "profile.platform put a /fail/ path on PATH/LD_LIBRARY_PATH/PYTHONPATH" ;;
    *)        ok "no /fail/ paths after profile.platform" ;;
esac
chk "soam CLIs on PATH after profile.platform" bash -c \
    ". $EGO_TOP/profile.platform >/dev/null 2>&1; \
     command -v soamview && command -v soamreg && command -v soamdeploy \
     && command -v soamcontrol && command -v egoconfig"

# --- 7. tree hygiene ------------------------------------------------------
chk "entitlement at kernel/conf/sym_com_entitlement.dat" \
    test -s "$EGO_TOP/kernel/conf/sym_com_entitlement.dat"
chk "cluster is named symphony" test -f "$EGO_TOP/kernel/conf/ego.cluster.symphony"
stray="$(find "$EGO_TOP" \( ! -user egoadmin -o ! -group egoadmin \) -print -quit 2>/dev/null)"
[ -z "$stray" ] && ok "whole tree is egoadmin:egoadmin" \
                || bad "not egoadmin-owned, e.g. $stray"
chk "kernel/log writable by egoadmin" su egoadmin -c "test -w $EGO_TOP/kernel/log"
chk "soam/ writable by egoadmin (SIM extracts deploy packages there)" \
    su egoadmin -c "test -w $EGO_TOP/soam"
for f in python3.12 akida ego soam-lib soam-lib64; do
    chk "/etc/ld.so.conf.d/$f.conf" test -s "/etc/ld.so.conf.d/$f.conf"
done
# The demo runs the EGO base transport in plaintext; see docker/patch_symphony.sh.
chk "ego.conf does not enable EGO_TRANSPORT_SECURITY" bash -c \
    "! grep -qE '^[[:space:]]*EGO_TRANSPORT_SECURITY[[:space:]]*=' $EGO_TOP/kernel/conf/ego.conf"
chk "sd.xml has no TLS transport entries" bash -c \
    "! grep -q 'TCPIPv4SSL' $EGO_TOP/soam/$SYM_VERSION/eservice/sd.xml"
chk "rs.xml has no TLS transport entries" bash -c \
    "! grep -q 'TCPIPv4SSL' $EGO_TOP/eservice/esc/conf/services/rs.xml"

# --- 8. PKI ---------------------------------------------------------------
SEC="$EGO_TOP/wlp/usr/shared/resources/security"
NINE_YEARS=283824000
for c in "$SEC/cacert.pem" "$SEC/user.pem" "$EGO_TOP/kernel/conf/cacert.pem" \
         "$EGO_TOP/kernel/conf/server.pem"; do
    label="$(printf '%s' "$c" | sed "s|^$EGO_TOP/||")"
    if openssl x509 -in "$c" -noout -checkend "$NINE_YEARS" >/dev/null 2>&1; then
        ok "$label valid until $(openssl x509 -in "$c" -noout -enddate | cut -d= -f2)"
    else
        bad "$label is missing or expires within 9 years (IBM's originals expired in 2005/2022/2023)"
    fi
done
chk "the leaf verifies against the CA" openssl verify -CAfile "$SEC/cacert.pem" "$SEC/user.pem"
chk "kernel/conf/server.pem holds cert + key" bash -c \
    "grep -q 'BEGIN CERTIFICATE' $EGO_TOP/kernel/conf/server.pem \
     && grep -q 'BEGIN PRIVATE KEY' $EGO_TOP/kernel/conf/server.pem"
chk "ego.conf TS params point at the regenerated PEMs" bash -c \
    "grep -q '$SEC/user.pem' $EGO_TOP/kernel/conf/ego.conf \
     && grep -q '$SEC/cacert.pem' $EGO_TOP/kernel/conf/ego.conf"

if [ "$MODE" = platform ]; then
    printf '[verify] platform checks complete: %d failure(s)\n' "$FAILED"
    [ "$FAILED" -eq 0 ] || exit 1
    exit 0
fi

# ======================== --full only ====================================
# --- 9. the SYM_VERSION literal spelled out in the repo -------------------
# Five wrapper scripts and four XML profiles hardcode the Symphony version. Rather
# than templating the files developers iterate on fastest, assert here that every
# literal agrees with the tree we shipped, so a version bump fails the build with
# the exact file list to edit.
for f in /opt/akida-service/run_akida_service.sh \
         /opt/akida-client/run_client.sh \
         /opt/akida-shard-service/run_akida_service.sh \
         /opt/akida-shard-service/run_cpu_service.sh \
         /opt/akida-shard-client/run_client.sh; do
    if [ ! -f "$f" ]; then bad "missing wrapper $f"; continue; fi
    grep -q "soam/$SYM_VERSION/linux-x86_64" "$f" \
        && ok "$(basename "$f") references soam/$SYM_VERSION" \
        || bad "$f does not reference soam/$SYM_VERSION"
    grep -q 'pythonapi_3.6.7' "$f" \
        && ok "$(basename "$f") references pythonapi_3.6.7" \
        || bad "$f does not reference pythonapi_3.6.7"
done
for f in /opt/akida-service/AkidaGenericService.xml \
         /opt/akida-shard-service/segment/ShardSegmentService.xml \
         /opt/akida-shard-service/inference/ShardInferenceService.xml \
         /opt/akida-shard-service/stitch/ShardStitchService.xml; do
    if [ ! -f "$f" ]; then bad "missing profile $f"; continue; fi
    grep -q "version=\"$SYM_VERSION\"" "$f" \
        && ok "$(basename "$f") declares $SYM_VERSION" \
        || bad "$f does not declare version=\"$SYM_VERSION\""
    chk "$(basename "$f") is well-formed XML" \
        /usr/bin/python3.6 -c 'import sys,xml.etree.ElementTree as E;E.parse(sys.argv[1])' "$f"
done

# --- 10. app layout ------------------------------------------------------
chk "/entrypoint.sh executable" test -x /entrypoint.sh
chk "/opt/akida-common/akida_chip.py" test -f /opt/akida-common/akida_chip.py
chk "probe_chips.sh executable (up.sh runs it as an entrypoint)" \
    test -x /opt/akida-service/probe_chips.sh
chk "run_http_server.sh executable" test -x /opt/akida-http/run_http_server.sh
chk "shard_wire.py sits next to the shard client" test -f /opt/akida-shard-client/shard_wire.py
for p in /opt/akida-service/AkidaGenericServicePackage.v1.tar.gz \
         /opt/akida-shard-service/ShardSegmentServicePackage.v1.tar.gz \
         /opt/akida-shard-service/ShardInferenceServicePackage.v1.tar.gz \
         /opt/akida-shard-service/ShardStitchServicePackage.v1.tar.gz; do
    [ -s "$p" ] && ok "deploy package $(basename "$p"): $(tar -tzf "$p" | tr '\n' ' ')" \
                || bad "missing deploy package $p"
done

# Every SOAM ServiceContainer and client runs under python3.6 -- catch 3.7+
# syntax now, not when a service instance silently fails to start.
for f in /opt/akida-service/AkidaServiceContainer.py \
         /opt/akida-shard-service/segment/SegmentServiceContainer.py \
         /opt/akida-shard-service/inference/InferenceServiceContainer.py \
         /opt/akida-shard-service/stitch/StitchServiceContainer.py \
         /opt/akida-shard-service/shard_wire.py \
         /opt/akida-client/soam_client.py \
         /opt/akida-shard-client/shard_client.py; do
    [ -f "$f" ] || { warn "absent: $f"; continue; }
    chk "python3.6 parses $(basename "$f")" /usr/bin/python3.6 -m py_compile "$f"
done
# And every worker under python3.12.
for f in /opt/akida-service/akida_worker.py \
         /opt/akida-shard-service/segment/segment_worker.py \
         /opt/akida-shard-service/inference/inference_worker.py \
         /opt/akida-shard-service/stitch/stitch_worker.py \
         /opt/akida-http/http_server.py; do
    [ -f "$f" ] || { warn "absent: $f"; continue; }
    chk "python3.12 parses $(basename "$f")" "$PY312" -m py_compile "$f"
done
chk "/opt/akida-common imports under python3.12" bash -c \
    "PYTHONPATH=$VENV_SP:/opt/akida-common $PY312 -c 'import akida_chip, models, worker_io, tiled_shard'"

printf '[verify] complete: %d failure(s)\n' "$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
exit 0
