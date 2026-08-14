#!/usr/bin/env bash
# Normalise the Symphony CE tree harvested out of ibmcom/spectrum-symphony.
#
# IBM installs Symphony with the same sym-7.3.2.0_x86_64.bin we would, then runs
# their own scripts/configure_image.sh over it. That script fixes several things
# we would otherwise have to fix ourselves, but it also (a) leaves two profiles
# unable to parse a modern kernel version and (b) makes container-tuning choices
# this demo does not run on. This script closes both gaps.
#
# Every rule asserts its own result. A sed that silently matches nothing is the
# one failure mode that yields an image which builds clean and then breaks deep
# inside vemkd.log, so anything unexpected fails the build here instead.
#
# To go back to IBM's TLS posture (not recommended without a full three-app
# validation run -- see docker/gen_certs.sh and the plan for why): drop the
# ego.conf keys EGO_TRANSPORT_SECURITY/EGO_KD_TS_PORT from IBM_TUNING_KEYS
# below, and delete the two strip_env_vars calls at the end.
set -euo pipefail

EGO_TOP="${EGO_TOP:-/opt/ibm/spectrumcomputing}"
SYM_VERSION="${SYM_VERSION:-7.3.2}"
EGO_VERSION="${EGO_VERSION:-4.0}"

FAILURES=0
note() { printf '[patch] %s\n' "$*"; }
fail() { printf '[patch] FAIL  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

# ---------------------------------------------------------------------------
# 0. The tree must look like what we expect before we start editing it.
# ---------------------------------------------------------------------------
for rel in \
    "profile.platform" \
    "kernel/conf/ego.conf" \
    "kernel/conf/ego.cluster.symphony" \
    "kernel/conf/profile.ego" \
    "soam/conf/profile.soam" \
    "perf/conf/profile.perf" \
    "scripts/sym_com_entitlement.dat" \
    "soam/${SYM_VERSION}/linux-x86_64/lib64/pythonapi_3.6.7/soamapi.pyc" \
    "${EGO_VERSION}/linux-x86_64/bin/egosh"
do
    [ -e "$EGO_TOP/$rel" ] || fail "harvested tree is missing $rel"
done
[ "$FAILURES" -eq 0 ] || { note "aborting: the harvested tree is not what we expect"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Kernel major version.
#
# profile.soam and profile.perf close their accepted-kernel list at 5 with no
# fallback:
#     elif [ $version = "3" -o $version = "4"  -o $version = "5" ]; then
# On anything newer BINARY_TYPE stays "fail", so SOAM_BINDIR/SOAM_LIBDIR resolve
# under soam/<v>/fail/ and every SOAM wrapper breaks. Our hosts run 6.x.
#
# kernel/conf/profile.ego deliberately gets no rule: it already ends its version
# ladder with a catch-all `else` that derives the type from `uname -m`.
#
# We extend to 15 rather than 6 so this patch does not need revisiting for the
# life of Symphony 7.3.2.
# ---------------------------------------------------------------------------
kernel_list() {
    local out="" v
    for v in 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        out="${out:+$out -o }\$version = \"$v\""
    done
    printf 'elif [ %s ]; then' "$out"
}
NEW_TEST="$(kernel_list)"

for rel in soam/conf/profile.soam perf/conf/profile.perf; do
    f="$EGO_TOP/$rel"
    if grep -q '\$version = "15"' "$f"; then
        note "$rel: kernel list already extended"
        continue
    fi
    # The vendor text has two spaces before the third clause; match loosely.
    before="$(grep -c 'elif \[ \$version = "3" *-o \$version = "4" *-o \$version = "5" \]; then' "$f" || true)"
    if [ "$before" -ne 1 ]; then
        fail "$rel: expected exactly 1 closed kernel test, found $before"
        continue
    fi
    python3.6 - "$f" "$NEW_TEST" <<'PY'
import re, sys, os, stat
path, new = sys.argv[1], sys.argv[2]
src = open(path).read()
pat = re.compile(r'elif \[ \$version = "3" *-o \$version = "4" *-o \$version = "5" \]; then')
out, n = pat.subn(lambda m: new, src)
assert n == 1, n
st = os.stat(path)
open(path, 'w').write(out)
os.chmod(path, stat.S_IMODE(st.st_mode))
os.chown(path, st.st_uid, st.st_gid)
PY
    grep -q '\$version = "15"' "$f" || fail "$rel: kernel list not extended"
    note "$rel: kernel list extended to 3..15"
done

# The installer emits `SOAM_HOME=$SOAM_HOME` in some builds, which resolves to
# empty for any process that did not already export it -- and SOAM launches
# service wrappers with an empty environment. IBM's tree already carries the
# literal path, so this is an assertion rather than an edit.
for rel in soam/conf/profile.soam soam/conf/cshrc.soam; do
    f="$EGO_TOP/$rel"
    [ -f "$f" ] || continue
    if grep -qE 'SOAM_HOME[[:space:]=]+\$SOAM_HOME' "$f"; then
        fail "$rel assigns SOAM_HOME from itself; it must be the literal path"
    fi
done
grep -q "^SOAM_HOME=$EGO_TOP/soam\$" "$EGO_TOP/soam/conf/profile.soam" \
    || fail "profile.soam does not set SOAM_HOME to $EGO_TOP/soam"
note "SOAM_HOME is literal"

# ---------------------------------------------------------------------------
# 2. BINARY_TYPE in webserverstart.sh -- the script behind the :8443 console.
#
# It carries its own copy of the platform-detection logic and falls back to
# "fail", which leaves WEBGUI and REST in ERROR in `egosh service list`. IBM's
# configure_image.sh applies exactly this substitution to the profiles; we apply
# it here too, which turns "could not detect" into "assume linux-x86_64" (the
# only platform the akida wheels support anyway).
# ---------------------------------------------------------------------------
WSS="$EGO_TOP/${EGO_VERSION}/linux-x86_64/etc/webserverstart.sh"
if grep -qE 'BINARY_TYPE(_PMC)?="fail"' "$WSS"; then
    n="$(grep -cE 'BINARY_TYPE(_PMC)?="fail"' "$WSS")"
    sed -i -e 's|BINARY_TYPE="fail"|BINARY_TYPE="linux-x86_64"|g' \
           -e 's|BINARY_TYPE_PMC="fail"|BINARY_TYPE_PMC="linux-x86_64"|g' "$WSS"
    grep -qE 'BINARY_TYPE(_PMC)?="fail"' "$WSS" \
        && fail "webserverstart.sh still has a BINARY_TYPE fallback to fail"
    note "webserverstart.sh: $n BINARY_TYPE fallback(s) -> linux-x86_64"
else
    note "webserverstart.sh: no BINARY_TYPE=fail fallback to patch"
fi

# ---------------------------------------------------------------------------
# 3. Restore the stock kernel/conf/ego.conf.
#
# configure_image.sh appends a container-tuning block. The demo has always run
# the stock installer values -- verified against the live .cluster/shared config
# the previous image produced, which carries none of these keys -- and three of
# them are individually risky here:
#
#   EGO_LIM_IS_IN_CONTAINER=Y  changes how LIM counts cores. This cluster sits
#       exactly on the Community Edition 64-core cap, so a shift in core
#       accounting does not degrade the demo, it stops hosts being admitted.
#   EGO_TRANSPORT_SECURITY=SSL puts TLS under the 2017-vintage python3.6 soamapi
#       binding (whose failure mode is an opaque hang) and presents one
#       certificate from every daemon on every host, which a single CN cannot
#       satisfy. It protects nothing on a single-host docker bridge network
#       where /shared is a bind mount anyway.
#   EGO_DISABLE_ROOT_REX=Y     the containers run as root because LIM needs it.
#
# The rest are dropped for the same reason: the demo is proven on the stock
# values and on nothing else.
# ---------------------------------------------------------------------------
EGO_CONF="$EGO_TOP/kernel/conf/ego.conf"
IBM_TUNING_KEYS=(
    EGO_SIMPLIFIED_WEM
    EGO_TRANSPORT_SECURITY
    EGO_KD_TS_PORT
    EGO_DYNAMIC_HOST_TIMEOUT
    EGO_RESOURCE_UPDATE_INTERVAL
    EGO_ENABLE_RG_UPDATE_MEMBERSHIP
    EGO_RG_UPDATE_MEMBERSHIP_INTERVAL
    EGO_DISABLE_ROOT_REX
    EGO_ELIM_RUNAS_CLUSTER_ADMIN
    EGO_LIM_IS_IN_CONTAINER
)
for k in "${IBM_TUNING_KEYS[@]}"; do
    if grep -qE "^[[:space:]]*${k}[[:space:]]*=" "$EGO_CONF"; then
        note "ego.conf: dropping $(grep -E "^[[:space:]]*${k}[[:space:]]*=" "$EGO_CONF" | tr -d '\n')"
        sed -i -E "/^[[:space:]]*${k}[[:space:]]*=/d" "$EGO_CONF"
    fi
done
# The commented-out SSL variants IBM appends alongside them.
sed -i -E '/^#EGO_(PEM_TRANSPORT_SECURITY|KD_PEM_TS_PORT|PEM_TS_PORT)=/d' "$EGO_CONF"

# IBM overrides EGO_DYNAMIC_HOST_WAIT_TIME by appending a second assignment
# (=1). Keep the stock one (=60), which appears first.
dups="$(grep -cE '^[[:space:]]*EGO_DYNAMIC_HOST_WAIT_TIME[[:space:]]*=' "$EGO_CONF" || true)"
if [ "$dups" -gt 1 ]; then
    python3.6 - "$EGO_CONF" <<'PY'
import os, stat, sys
path = sys.argv[1]
seen, out = False, []
for line in open(path):
    if line.lstrip().startswith('EGO_DYNAMIC_HOST_WAIT_TIME'):
        if seen:
            print('[patch] ego.conf: dropping duplicate %s' % line.strip())
            continue
        seen = True
    out.append(line)
st = os.stat(path)
open(path, 'w').writelines(out)
os.chmod(path, stat.S_IMODE(st.st_mode))
os.chown(path, st.st_uid, st.st_gid)
PY
fi

for k in "${IBM_TUNING_KEYS[@]}"; do
    grep -qE "^[[:space:]]*${k}[[:space:]]*=" "$EGO_CONF" \
        && fail "ego.conf still sets $k"
done
for k in EGO_MASTER_LIST EGO_LIM_PORT EGO_KD_PORT EGO_PEM_PORT EGO_ESRVDIR \
         EGO_SEC_CONF EGO_ENTITLEMENT_FILE EGO_DEFAULT_TS_PARAMS \
         EGO_CLIENT_TS_PARAMS EGO_VERSION
do
    grep -qE "^[[:space:]]*${k}[[:space:]]*=" "$EGO_CONF" \
        || fail "ego.conf lost the stock key $k"
done
note "--- resulting ego.conf ---"
grep -vE '^[[:space:]]*(#|$)' "$EGO_CONF" | sed 's/^/[patch]     /'

# ---------------------------------------------------------------------------
# 4. Revert IBM's TLS wiring in the Session Director and Repository Service.
#
# configure_image.sh inserts these EnvironmentVariable entries to switch the SD
# and RS SOAP/SDK transports to TCPIPv4SSL. An SSL-enabled Session Director on
# top of a plaintext base transport is exactly the mismatch that hangs at
# `soamview app`, so with EGO_TRANSPORT_SECURITY gone these have to go too.
#
# The name lists below are not guesses: they are the complete diff of
# EnvironmentVariable names between IBM's tree and the live .cluster/shared tree
# the previous image produced. Nothing else differs.
# ---------------------------------------------------------------------------
strip_env_vars() {
    local f="$1"; shift
    local n removed=0 hits
    [ -f "$f" ] || { fail "missing $f"; return; }
    for n in "$@"; do
        hits="$(grep -c "<ego:EnvironmentVariable name=\"$n\">" "$f" || true)"
        [ "$hits" -eq 0 ] && continue
        sed -i "/<ego:EnvironmentVariable name=\"$n\">/d" "$f"
        removed=$((removed + hits))
    done
    for n in "$@"; do
        grep -q "<ego:EnvironmentVariable name=\"$n\">" "$f" \
            && fail "$(basename "$f") still declares $n"
    done
    python3.6 -c 'import sys,xml.etree.ElementTree as E; E.parse(sys.argv[1])' "$f" \
        || fail "$(basename "$f") is no longer well-formed XML"
    note "$(basename "$f"): removed $removed TLS entr$( [ "$removed" = 1 ] && echo y || echo ies)"
}

strip_env_vars "$EGO_TOP/soam/${SYM_VERSION}/eservice/sd.xml" \
    SDK_TRANSPORT SDK_TRANSPORT_ARG \
    SD_SDK_TRANSPORT SD_SDK_TRANSPORT_ARG \
    SDSOAPCLIENT_ARG SD_SOAP_TRANSPORT SD_SOAP_TRANSPORT_ARG \
    SSM_SDK_ADDR SSM_SDK_TRANSPORT SSM_SDK_TRANSPORT_ARG

strip_env_vars "$EGO_TOP/eservice/esc/conf/services/rs.xml" \
    RS_RSSDK_TRANSPORT RS_RSSDK_TRANSPORT_ARG RSSDK_TRANSPORT_ARG

# SD_SDK_PORT and REPOSITORY_SERVICE_PORT sit next to the removed entries and
# are stock -- losing them would silently move the SD/RS off their known ports.
grep -q 'name="SD_SDK_PORT"' "$EGO_TOP/soam/${SYM_VERSION}/eservice/sd.xml" \
    || fail "sd.xml lost SD_SDK_PORT"
grep -q 'name="REPOSITORY_SERVICE_PORT"' "$EGO_TOP/eservice/esc/conf/services/rs.xml" \
    || fail "rs.xml lost REPOSITORY_SERVICE_PORT"

# ---------------------------------------------------------------------------
if [ "$FAILURES" -gt 0 ]; then
    note "$FAILURES failure(s)"
    exit 1
fi
note "OK"
