#!/usr/bin/env bash
# Mint a fresh self-signed PKI for the harvested Symphony CE tree.
#
# Every certificate in ibmcom/spectrum-symphony:7.3.2.0 is expired:
#     wlp/usr/shared/resources/security/user.pem    notAfter 2023-01-10
#     wlp/usr/shared/resources/security/cacert.pem  notAfter 2022-09-29
#     kernel/conf/server.pem                        the gSOAP sample certificate
#                                                   (CN=localhost, O=Genivia),
#                                                   expired 2005
# and $EGO_TOP/scripts/generate_ssl.sh cannot fix it -- its openssl half is
# commented out, so it only rebuilds the Java keystores, re-signing with the
# expired CA in caKeyStore.jks.
#
# These files matter even though this image runs the EGO base transport in
# plaintext (see docker/patch_symphony.sh): ego.conf's stock EGO_DEFAULT_TS_PARAMS
# and EGO_CLIENT_TS_PARAMS point at the wlp PEMs, WebSphere Liberty serves the
# :8443 console out of serverKeyStore.jks, and kernel/conf/server.pem backs the
# gSOAP endpoints.
#
# This script is generate_ssl.sh's missing half: one CA, one leaf, then the same
# three keystores with the same aliases and the same store password, so Liberty's
# server.xml -- which we never touch -- can still open them. The password and
# aliases are read out of IBM's own keystores rather than assumed; guessing here
# would produce an image whose console silently never starts.
#
# Also runnable against a live container, e.g. to change the CN or push the
# expiry out, without rebuilding:
#     docker exec symphony-master /usr/local/bin/gen-certs --cn my-host.local
# followed by a cluster restart.
set -euo pipefail

EGO_TOP="${EGO_TOP:-/opt/ibm/spectrumcomputing}"
DAYS=3653                       # 10 years
ORG="BrainChip"
CN="symphony-master.local"
PASS_CANDIDATES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --ego-top)   EGO_TOP="$2"; shift 2 ;;
        --days)      DAYS="$2"; shift 2 ;;
        --org)       ORG="$2"; shift 2 ;;
        --cn)        CN="$2"; shift 2 ;;
        --storepass) PASS_CANDIDATES+=("$2"); shift 2 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

SEC="$EGO_TOP/wlp/usr/shared/resources/security"
KCONF="$EGO_TOP/kernel/conf"
log()  { printf '[pki] %s\n' "$*"; }
die()  { printf '[pki] FATAL: %s\n' "$*" >&2; exit 1; }

# --- tools -----------------------------------------------------------------
# Use the JRE bundled in the harvested tree: it is the same JVM Liberty runs, so
# a keystore it can read is a keystore Liberty can read. Discovered rather than
# hardcoded because the jre/<version>/ directory name is vendor-specific.
KEYTOOL="$(find "$EGO_TOP/jre" -type f -name keytool -perm -u+x 2>/dev/null | head -1 || true)"
[ -n "$KEYTOOL" ] || die "no keytool under $EGO_TOP/jre"
"$KEYTOOL" -help >/dev/null 2>&1 || die "$KEYTOOL will not run (JRE/glibc mismatch?)"
command -v openssl >/dev/null || die "openssl is not installed"
log "keytool: $KEYTOOL"

for f in caKeyStore.jks serverKeyStore.jks serverTrustStore.jks; do
    [ -f "$SEC/$f" ] || die "expected harvested keystore missing: $SEC/$f"
done

# --- discover and prove the store password ---------------------------------
GEN="$EGO_TOP/scripts/generate_ssl.sh"
if [ -f "$GEN" ]; then
    while read -r p; do
        [ -n "$p" ] && PASS_CANDIDATES+=("$p")
    done < <(grep -oE '\-(src|dest)?store?pass[[:space:]]+[^[:space:]"$]+' "$GEN" \
             | awk '{print $NF}' | sort -u)
fi
PASS_CANDIDATES+=(Liberty changeit symphony)

STOREPASS=""
for p in "${PASS_CANDIDATES[@]}"; do
    [ -n "$p" ] || continue
    if "$KEYTOOL" -list -keystore "$SEC/serverKeyStore.jks" -storepass "$p" >/dev/null 2>&1; then
        STOREPASS="$p"; break
    fi
done
[ -n "$STOREPASS" ] || die "could not open $SEC/serverKeyStore.jks with any candidate password.
Pass the right one with --storepass. Do not invent a new one: Liberty's server.xml
already encodes it and this script does not modify server.xml."
log "store password verified against the harvested keystore"

# Reuse IBM's aliases exactly -- server.xml and the EGO services reference them.
alias_of() {  # <keystore> <entry-type>
    "$KEYTOOL" -list -keystore "$1" -storepass "$STOREPASS" 2>/dev/null \
        | awk -F, -v t="$2" '$0 ~ t {gsub(/^[ \t]+/,"",$1); print $1; exit}'
}
CA_ALIAS="$(alias_of "$SEC/caKeyStore.jks" 'keyEntry|PrivateKeyEntry')"
SRV_ALIAS="$(alias_of "$SEC/serverKeyStore.jks" 'keyEntry|PrivateKeyEntry')"
TRUST_ALIAS="$(alias_of "$SEC/serverTrustStore.jks" 'trustedCertEntry')"
: "${CA_ALIAS:=caalias}"; : "${SRV_ALIAS:=srvalias}"; : "${TRUST_ALIAS:=srvalias}"
log "aliases: ca=$CA_ALIAS server=$SRV_ALIAS trust=$TRUST_ALIAS"

# --- generate --------------------------------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

# The SANs cover every hostname scripts/launch/up.sh assigns (--hostname /
# --network-alias symphony-master.local and symphony-compute-<j>.local) plus
# localhost, for https://localhost:8443/platform.
cat > openssl.cnf <<EOF
[ req ]
distinguished_name = dn
prompt             = no
[ dn ]
C  = US
O  = $ORG
CN = $CN
[ v3_ca ]
basicConstraints       = critical,CA:TRUE
keyUsage               = critical,keyCertSign,cRLSign
subjectKeyIdentifier   = hash
[ v3_leaf ]
basicConstraints       = critical,CA:FALSE
keyUsage               = critical,digitalSignature,keyEncipherment
extendedKeyUsage       = serverAuth,clientAuth
subjectKeyIdentifier   = hash
authorityKeyIdentifier = keyid,issuer
subjectAltName         = @san
[ san ]
DNS.1 = $CN
DNS.2 = symphony-master.local
DNS.3 = symphony-master
DNS.4 = *.local
DNS.5 = localhost
IP.1  = 127.0.0.1
EOF

log "CA, ${DAYS} days"
openssl genrsa -out ca.key 4096 2>/dev/null
openssl req -x509 -new -key ca.key -sha256 -days "$DAYS" \
    -subj "/C=US/O=$ORG/CN=$ORG Symphony CE Root CA" \
    -config openssl.cnf -extensions v3_ca -out cacert.pem

log "leaf CN=$CN, ${DAYS} days"
openssl genrsa -out leaf.rsa 2048 2>/dev/null
# PKCS#8, unencrypted -- the shape `openssl pkcs12 -nodes` produced before, and
# what ego.conf's PRIVATE_KEY= expects.
openssl pkcs8 -topk8 -nocrypt -in leaf.rsa -out user.key
openssl req -new -key user.key -subj "/C=US/O=$ORG/CN=$CN" \
    -config openssl.cnf -out leaf.csr
openssl x509 -req -in leaf.csr -CA cacert.pem -CAkey ca.key -CAcreateserial \
    -sha256 -days "$DAYS" -extfile openssl.cnf -extensions v3_leaf -out user.pem 2>/dev/null
openssl verify -CAfile cacert.pem user.pem >/dev/null \
    || die "the generated leaf does not verify against the generated CA"

# PKCS#12 bundles for the keystore imports, and for parity with the installer's
# user.p12. EL8's openssl 1.1.1 writes a PKCS#12 that Java 8 keytool reads
# directly -- one more reason to generate this inside the image rather than on a
# developer's openssl-3 host.
openssl pkcs12 -export -name "$SRV_ALIAS" -inkey user.key -in user.pem \
    -certfile cacert.pem -passout pass:"$STOREPASS" -out server.p12
openssl pkcs12 -export -name "$CA_ALIAS" -inkey ca.key -in cacert.pem \
    -passout pass:"$STOREPASS" -out ca.p12
openssl pkcs12 -export -name "$SRV_ALIAS" -inkey user.key -in user.pem \
    -certfile cacert.pem -passout pass:"$STOREPASS" -out user.p12

kt() { "$KEYTOOL" "$@" 2>&1 | grep -vE '^Warning:|proprietary format|Re-import' || true; }

jks_from_p12() {  # <p12> <out.jks> <alias>
    rm -f "$2"
    kt -importkeystore -noprompt \
        -srckeystore "$1" -srcstoretype PKCS12 -srcstorepass "$STOREPASS" \
        -destkeystore "$2" -deststoretype JKS -deststorepass "$STOREPASS" \
        -destkeypass "$STOREPASS" -srcalias "$3" -destalias "$3"
    "$KEYTOOL" -list -keystore "$2" -storepass "$STOREPASS" >/dev/null 2>&1 \
        || die "failed to build $2"
}

# caKeyStore.jks  : <ca alias> key entry, so generate_ssl.sh can still re-sign.
# serverKeyStore  : <srv alias> key entry (leaf + chain) plus the CA as trusted.
# serverTrustStore: the CA, under the alias IBM used there.
log "keystores"
jks_from_p12 ca.p12     caKeyStore.jks     "$CA_ALIAS"
jks_from_p12 server.p12 serverKeyStore.jks "$SRV_ALIAS"
kt -importcert -noprompt -alias "$CA_ALIAS" -file cacert.pem \
    -keystore serverKeyStore.jks -storepass "$STOREPASS"
rm -f serverTrustStore.jks
kt -importcert -noprompt -alias "$TRUST_ALIAS" -file cacert.pem \
    -keystore serverTrustStore.jks -storepass "$STOREPASS"

# kernel/conf/server.pem follows the gSOAP convention: the leaf certificate
# followed by its unencrypted private key in one file. kernel/conf/dh512.pem is
# left as harvested -- ego.conf negotiates ECDHE, so it is unused.
cat user.pem user.key > server.pem

# --- install ---------------------------------------------------------------
# Modes and ownership match the image this replaces.
inst() { install -o egoadmin -g egoadmin -m "$1" "$2" "$3"; }
inst 0644 cacert.pem            "$SEC/cacert.pem"
inst 0644 user.pem              "$SEC/user.pem"
inst 0600 user.key              "$SEC/user.key"
inst 0640 user.p12              "$SEC/user.p12"
inst 0640 caKeyStore.jks        "$SEC/caKeyStore.jks"
inst 0640 serverKeyStore.jks    "$SEC/serverKeyStore.jks"
inst 0640 serverTrustStore.jks  "$SEC/serverTrustStore.jks"
inst 0644 server.pem            "$KCONF/server.pem"
inst 0644 cacert.pem            "$KCONF/cacert.pem"

# --- self test -------------------------------------------------------------
# 9 years, so a 10-year certificate that silently came out short fails here.
NINE_YEARS=283824000
for c in "$SEC/cacert.pem" "$SEC/user.pem" "$KCONF/cacert.pem"; do
    openssl x509 -in "$c" -noout -checkend "$NINE_YEARS" >/dev/null \
        || die "$c is not valid for at least 9 more years"
done
openssl x509 -in "$KCONF/server.pem" -noout -checkend "$NINE_YEARS" >/dev/null \
    || die "$KCONF/server.pem is not valid for at least 9 more years"
grep -q 'BEGIN CERTIFICATE' "$KCONF/server.pem" \
    && grep -q 'BEGIN PRIVATE KEY' "$KCONF/server.pem" \
    || die "$KCONF/server.pem must hold the certificate AND the private key"
for ks in caKeyStore serverKeyStore serverTrustStore; do
    "$KEYTOOL" -list -keystore "$SEC/$ks.jks" -storepass "$STOREPASS" >/dev/null 2>&1 \
        || die "$ks.jks is unreadable with the store password Liberty expects"
done
log "$(openssl x509 -in "$SEC/user.pem" -noout -subject -enddate | tr '\n' ' ')"
log "OK"
