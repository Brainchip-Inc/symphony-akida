# Symphony + Akida demo image, built end to end from public sources.
#
# Stage 1 exists only to be copied out of: IBM publishes Spectrum Symphony
# Community Edition as a container image, and the sym-7.3.2.0_x86_64.bin
# installer is a 3 GB download behind an IBMid, so a clone-and-build flow cannot
# run the installer. We harvest the already-installed tree instead -- Symphony
# 7.3.2 + EGO 4.0, the bundled JRE, the CE entitlement, and the python3.6
# soamapi binding at soam/7.3.2/linux-x86_64/lib64/pythonapi_3.6.7 that all
# three apps import.
#
# Stage 2 is the image we ship, and it cannot be IBM's base. That base is RHEL
# UBI 7.9: glibc 2.17 and a GCC-4.8 libstdc++ (GLIBCXX_3.4.19). The akida cp312
# wheel is manylinux_2_28 and its libakida.so.2 needs GLIBC_2.26 plus
# GLIBCXX_3.4.22, and akida needs python >= 3.10. Meanwhile pythonapi_3.6.7 is a
# sourceless soamapi.pyc plus SoamFactory.so frozen to the CPython 3.6 ABI, so
# the runtime has to supply python3.6 *and* python3.12 on glibc >= 2.28. EL8 is
# the only line that does both: EL9 dropped python3.6, EL7 has neither glibc
# 2.28 nor python3.12. EL8 goes EOL 2029-05-31.
#
# Everything the rest of the repo hardcodes is preserved by construction:
#   /opt/ibm/spectrumcomputing                     EGO_TOP
#   .../soam/7.3.2/linux-x86_64/lib64/pythonapi_3.6.7   soamapi, python3.6 only
#   /usr/bin/python3.6                             the SOAM ServiceContainers
#   /opt/python3.12/bin/python3.12                 the akida workers
#   /opt/akida-venv/lib/python3.12/site-packages{,/akida.libs}
#   egoadmin = uid 1000 : gid 1000                 so /shared needs no host sudo
#   runs as root                                   LIM needs it, and
#                                                   scripts/launch/reclaim_shared.sh
#                                                   invokes /usr/bin/chown

# The IBM image is pinned by digest: it has not been rebuilt since 2021-12-08,
# and the ibmcom namespace is deprecated in favour of icr.io, so a retag there
# must not silently change what we build on.
ARG SYM_IMAGE=ibmcom/spectrum-symphony:7.3.2.0
ARG SYM_DIGEST=sha256:d6860f11eea6024e51134e58bc1968c41d797bd9555c4e0283224c52ccf17774
# Rocky 8.10, not 8.9 (what the previous base used): 8.9's mirrorlist now returns
# "invalid repo or arch" and dl.rockylinux.org/pub/rocky/8.9 is 404, so dnf on an
# 8.9 base fails. Functionally identical -- glibc 2.28, python36 3.6.8,
# GLIBCXX_3.4.25 -- and the previous image already installed el8_10 packages onto
# its 8.9 rootfs. almalinux:8.10 is a drop-in alternative.
ARG RUNTIME_IMAGE=rockylinux/rockylinux:8.10

# Stage 0 -- IBM licence gate.
#
# This deliberately comes FIRST, on a base we would pull anyway, so that a build
# without explicit acceptance stops here and never downloads IBM's image. We do
# not accept IBM's terms on a developer's behalf: they say yes themselves, or the
# build does not proceed. The final stage copies the marker this writes, so the
# gate is load-bearing rather than a stage a builder could skip.
FROM ${RUNTIME_IMAGE} AS licence-gate
ARG ACCEPT_IBM_LICENSE=no
ARG SYM_IMAGE
RUN set -eu; \
    if [ "$ACCEPT_IBM_LICENSE" != "yes" ]; then \
        { \
        echo ''; \
        echo '=============================================================================='; \
        echo ' IBM Spectrum Symphony Community Edition licence acceptance is required'; \
        echo '=============================================================================='; \
        echo ''; \
        echo ' This image is built from IBM Spectrum Symphony Community Edition 7.3.2,'; \
        echo " which the build downloads from IBM's public image:"; \
        echo ''; \
        echo "     ${SYM_IMAGE}"; \
        echo '     https://hub.docker.com/r/ibmcom/spectrum-symphony'; \
        echo ''; \
        echo ' That software is licensed by IBM under IBM terms, separately from this'; \
        echo ' repository. The full licence text (LA_*, LI_*, Lic_*, non_ibm_license.txt,'; \
        echo ' notices.txt) ships inside the image at /licenses once it is built.'; \
        echo ''; \
        echo ' Nothing has been downloaded yet. To accept and continue:'; \
        echo ''; \
        echo '     docker build --build-arg ACCEPT_IBM_LICENSE=yes \'; \
        echo '         -f docker/Dockerfile -t symphony-akida .'; \
        echo ''; \
        echo '=============================================================================='; \
        echo ''; \
        } >&2; \
        exit 1; \
    fi; \
    printf 'ACCEPT_IBM_LICENSE=yes was passed to docker build for %s\n' "${SYM_IMAGE}" \
        > /ibm-license-accepted

# SYM_DIGEST is the amd64 manifest digest, not the multi-arch list, so it also
# pins the architecture -- IBM publishes ppc64le too and the akida wheels are
# x86_64-only. No RUN in this stage: UBI-minimal's toolset is not ours to rely on.
FROM ${SYM_IMAGE}@${SYM_DIGEST} AS symphony

FROM ${RUNTIME_IMAGE}

ARG SYM_VERSION=7.3.2
ARG EGO_VERSION=4.0
ARG AKIDA_VERSION=2.19.2
# Pinned to the numpy the demos are proven against, not just "whatever akida
# resolves to". akida declares numpy as its only dependency.
ARG NUMPY_VERSION=2.1.3
# The same python-build-standalone CPython the previous base carried, pinned by
# release tag and verified by checksum.
ARG PY312_VERSION=3.12.8
ARG PY312_PBS_TAG=20241219
ARG PY312_SHA256=e172c063cdad4ab016f78834b1ef9fedfac80aae08918188fe6ebb3b325ce347
ARG CERT_CN=symphony-master.local

# The install parameters IBM built their image with, kept as in-image documentation
# of what the harvested tree was configured for.
#
# The previous base image also carried IBM_SPECTRUM_SYMPHONY_LICENSE_ACCEPT=Y here.
# That is deliberately gone: nothing in this image's runtime reads it (IBM's own
# gate is LICENSE=accept, consumed by their scripts/bootstrap.sh, which our
# entrypoint replaces), so it declared acceptance on the developer's behalf while
# enforcing nothing. Acceptance is now the explicit ACCEPT_IBM_LICENSE build arg
# checked in the licence-gate stage above.
#
# LANG is deliberately not set, matching the previous base. IBM's image sets
# en_US.UTF-8; changing the collation under EGO's own shell scripts buys the demo
# nothing and is not something this image has ever been validated with.
ENV EGO_TOP=/opt/ibm/spectrumcomputing \
    CLUSTERADMIN=egoadmin \
    CLUSTERNAME=symphony \
    BASEPORT=17869

# Symphony's prerequisites, as the previous base installed them. The load-bearing
# ones: libnsl supplies the libnsl.so.1 that glibc 2.28 dropped out of libc and
# that SoamFactory.so still needs; gawk because profile.ego aborts without a
# suitable awk; python36 is the 3.6 ABI pythonapi_3.6.7 is compiled against;
# openssh-server backs the optional SSH_PUBLIC_KEY path in entrypoint.sh.
RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \
        gawk tar gettext which hostname iproute procps-ng net-tools \
        python36 python38 openssl ed bc nmap-ncat libnsl \
        findutils diffutils openssh-server curl \
    && dnf clean all && rm -rf /var/cache/dnf

# uid AND gid 1000. IBM's own image uses 1000:0; we do not, because everything the
# cluster writes into the bind-mounted /shared lands as 1000:1000 and
# scripts/launch/reclaim_shared.sh chowns it back to the host user without sudo.
RUN groupadd -g 1000 egoadmin \
    && useradd -m -u 1000 -g 1000 -s /bin/bash egoadmin

# The installed CE tree, in one layer with the right ownership. Doing the chown
# in the COPY avoids a second multi-GB layer for `chown -R /opt/ibm`, which is
# what the previous base spent. COPY preserves the symlinks IBM's
# configure_image.sh created from soam/*/linux-x86_64/lib64/*.so into
# 4.0/linux-x86_64/lib, which is how SoamFactory.so resolves its dependencies.
COPY --from=symphony --chown=egoadmin:egoadmin \
     /opt/ibm/spectrumcomputing /opt/ibm/spectrumcomputing
# IBM's licence text, kept alongside the binaries it covers.
COPY --from=symphony /licenses /licenses
# Makes the licence gate load-bearing: this image cannot be produced without the
# stage that checks ACCEPT_IBM_LICENSE having run and succeeded.
COPY --from=licence-gate /ibm-license-accepted /etc/ibm-license-accepted

# IBM ships the CE entitlement under scripts/; entrypoint.sh points
# `egoconfig setentitlement` at kernel/conf/ on the master's first boot.
RUN install -m 0644 -o egoadmin -g egoadmin \
        "$EGO_TOP/scripts/sym_com_entitlement.dat" \
        "$EGO_TOP/kernel/conf/sym_com_entitlement.dat"

# The parts of IBM's configure_image.sh that live outside /opt/ibm, and so are
# not carried by the COPY above. EGO's SD/SSM fan-out forks and opens plenty of
# fds per task; /var/run/utmp must exist or LIM logs an error every cycle while
# computing its interactive-user index.
RUN set -eux; \
    printf '%s\n' \
        'egoadmin    soft    nproc     65536' \
        'egoadmin    hard    nproc     65536' \
        'egoadmin    soft    nofile    65536' \
        'egoadmin    hard    nofile    65536' >> /etc/security/limits.conf; \
    if [ -f /etc/security/limits.d/20-nproc.conf ]; then \
        sed -i -E 's/(^\*[[:space:]]+soft[[:space:]]+nproc[[:space:]]+)4096/\1unlimited/' \
            /etc/security/limits.d/20-nproc.conf; \
    fi; \
    touch /var/run/utmp && chmod 664 /var/run/utmp && chown root:utmp /var/run/utmp; \
    echo ". $EGO_TOP/profile.platform" >> /home/egoadmin/.bashrc

# Fix the two things IBM's tree still gets wrong for us (the kernel-major check in
# profile.soam/profile.perf, and BINARY_TYPE in webserverstart.sh) and undo the
# container-tuning choices this demo does not run on. Every edit asserts its own
# result, and the script prints the resulting ego.conf into the build log.
COPY docker/patch_symphony.sh /tmp/patch_symphony.sh
RUN bash /tmp/patch_symphony.sh && rm -f /tmp/patch_symphony.sh

# A fresh 10-year PKI. IBM's baked certificates are all expired -- the wlp leaf in
# January 2023, and kernel/conf/server.pem is the gSOAP sample certificate that
# expired in 2005 -- and their generate_ssl.sh cannot regenerate them. Installed
# as gen-certs too, so the CN or expiry can be changed in a live container.
COPY docker/gen_certs.sh /usr/local/bin/gen-certs
RUN chmod +x /usr/local/bin/gen-certs && /usr/local/bin/gen-certs --cn "${CERT_CN}"

# CPython 3.12 for the akida side: the same python-build-standalone build the
# previous base carried, so /opt/python3.12/{bin,lib} and the libpython3.12.so
# that ld.so.conf.d/python3.12.conf points at all land where they did before.
# Ships its own pip.
RUN set -eux; \
    url="https://github.com/astral-sh/python-build-standalone/releases/download/${PY312_PBS_TAG}/cpython-${PY312_VERSION}+${PY312_PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"; \
    curl -fsSLo /tmp/py312.tar.gz "$url"; \
    echo "${PY312_SHA256}  /tmp/py312.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/py312.tar.gz -C /opt; \
    mv /opt/python /opt/python3.12; \
    rm -f /tmp/py312.tar.gz; \
    ln -sf /opt/python3.12/bin/python3.12 /usr/local/bin/python3.12; \
    /opt/python3.12/bin/python3.12 -c 'import ssl, zlib, lzma, ctypes, mmap, sqlite3'

# The akida runtime. Not a real venv -- a --target site-packages tree, which is
# all the wrappers ever reference (they put it on PYTHONPATH and its akida.libs on
# LD_LIBRARY_PATH), and which avoids the dangling console-script shebangs the
# previous base's copied-in venv had. akida + numpy is the complete set: every
# in-container import across all five SOAM containers, all the workers, both
# clients and scripts/verify_reference.py is stdlib, numpy, akida, soamapi, or a
# module from /opt/akida-common.
# pip byte-compiles on install, which matters here: site-packages is root-owned and
# the workers run as egoadmin, so anything left uncompiled would be re-parsed on
# every service-instance start and never cached.
RUN /opt/python3.12/bin/python3.12 -m pip install --no-cache-dir \
        --target=/opt/akida-venv/lib/python3.12/site-packages \
        "akida==${AKIDA_VERSION}" "numpy==${NUMPY_VERSION}"

# SOAM launches service wrappers with an empty environment, so the wrappers set
# LD_LIBRARY_PATH themselves -- but egosh, egoconfig, lim, vemkd, sd and anything
# PEM starts get nothing, hence ldconfig. One file per concern, so a future change
# is a one-line diff.
RUN set -eux; \
    echo "/opt/python3.12/lib" > /etc/ld.so.conf.d/python3.12.conf; \
    echo "/opt/akida-venv/lib/python3.12/site-packages/akida.libs" > /etc/ld.so.conf.d/akida.conf; \
    echo "$EGO_TOP/${EGO_VERSION}/linux-x86_64/lib" > /etc/ld.so.conf.d/ego.conf; \
    echo "$EGO_TOP/soam/${SYM_VERSION}/linux-x86_64/lib" > /etc/ld.so.conf.d/soam-lib.conf; \
    echo "$EGO_TOP/soam/${SYM_VERSION}/linux-x86_64/lib64" > /etc/ld.so.conf.d/soam-lib64.conf; \
    ldconfig

# Gate the platform half before layering any app code on top: soamapi under 3.6,
# akida under 3.12, the ldd closure of IBM's RHEL7-era binaries on glibc 2.28, the
# bundled JRE, BINARY_TYPE on this kernel, the PKI, and tree ownership.
COPY docker/verify_image.sh /usr/local/bin/verify-image
RUN chmod +x /usr/local/bin/verify-image \
    && SYM_VERSION="${SYM_VERSION}" EGO_VERSION="${EGO_VERSION}" \
       PY_VERSION="${PY312_VERSION}" AKIDA_VERSION="${AKIDA_VERSION}" \
       NUMPY_VERSION="${NUMPY_VERSION}" \
       /usr/local/bin/verify-image --platform

# Our entrypoint: repo-local /shared, guarded registration, optional ncpus.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Shared on-chip core (akida_chip.py) + model allowlist, imported by ALL apps'
# workers via PYTHONPATH=/opt/akida-common. Single source, no duplication.
COPY src/common/ /opt/akida-common/

# App 1 (batch-inference) SOAM service: python3.6 ServiceContainer + python3.12
# on-chip worker + wrapper + profile, and a prebuilt deploy package the entrypoint
# registers on the master's first boot.
COPY src/apps/batch-inference/service/ /opt/akida-service/
RUN chmod +x /opt/akida-service/*.sh && /opt/akida-service/build_package.sh

# App 1 SOAM batch client runs inside the master (needs the 3.6 soamapi binding).
COPY src/apps/batch-inference/client/ /opt/akida-client/
RUN chmod +x /opt/akida-client/*.sh

# App 2 (serial-http-round-robin) per-node HTTP inference server (python3.12). The
# entrypoint starts it on each compute when START_HTTP=1; it reuses /opt/akida-common.
COPY src/apps/serial-http-round-robin/service/ /opt/akida-http/
RUN chmod +x /opt/akida-http/*.sh

# App 3 (image-shard-inference) three SOAM services -- segment (CPU) + inference (python3.6
# container + python3.12 akida worker, reusing /opt/akida-common) + stitch (CPU) -- prebuilt
# into three deploy packages that scripts/launch/up.sh registers, plus the thin orchestrator client
# run from the master.
COPY src/apps/image-shard-inference/service/ /opt/akida-shard-service/
RUN chmod +x /opt/akida-shard-service/*.sh && /opt/akida-shard-service/build_packages.sh
COPY src/apps/image-shard-inference/client/ /opt/akida-shard-client/
# The client speaks the same task wire format as the services, so it gets the same module.
RUN chmod +x /opt/akida-shard-client/*.sh \
    && cp /opt/akida-shard-service/shard_wire.py /opt/akida-shard-client/

# Full gate: everything above plus the app layers -- the hardcoded Symphony version
# in the five wrappers and four XML profiles agreeing with the harvested tree, the
# three deploy packages, and every ServiceContainer parsing under python3.6 and
# every worker under python3.12.
RUN SYM_VERSION="${SYM_VERSION}" EGO_VERSION="${EGO_VERSION}" \
    PY_VERSION="${PY312_VERSION}" AKIDA_VERSION="${AKIDA_VERSION}" \
    NUMPY_VERSION="${NUMPY_VERSION}" \
    /usr/local/bin/verify-image --full

EXPOSE 17869 17870 17871 17872 17873 17874 17875 2222 8050 8443 8543 8790
WORKDIR /root
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--hold"]
