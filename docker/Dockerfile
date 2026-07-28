# Symphony + Akida demo image.
#
# The base image provides the platform runtime — Rocky Linux 8.9 (glibc 2.28),
# IBM Spectrum Symphony CE 7.3.2, the python3.6 soamapi binding, python3.12,
# and akida 2.19.1 in /opt/akida-venv. We strip the baked demo service and
# layer our own entrypoint and (from Section 3) our SOAM service.
FROM ghcr.io/brainchip-inc/symphony-akida:7.3.4

# models/tiled_yolov2_voc.fbz was serialized by akida 2.19.2, and 2.19.1 refuses to
# deserialize it outright ("A model created with Akida 2.19.2 is not supported by
# Akida 2.19.1"), so upgrade the runtime in place. The venv was built elsewhere and its
# bin/ shebangs point at a path that does not exist here, hence pip via /opt/python3.12
# with an explicit --target. Drop this layer once the base image ships 2.19.2.
RUN /opt/python3.12/bin/python3.12 -m pip install --no-cache-dir --upgrade \
        --target=/opt/akida-venv/lib/python3.12/site-packages akida==2.19.2

# Remove the baked-in demo service; ours is owned by this repo.
RUN rm -rf /opt/akida-service

# Our entrypoint: repo-local /shared, guarded registration, optional ncpus.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Shared on-chip core (akida_chip.py) + model allowlist, imported by BOTH apps'
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
# into three deploy packages that launch/up.sh registers, plus the thin orchestrator client
# run from the master.
COPY src/apps/image-shard-inference/service/ /opt/akida-shard-service/
RUN chmod +x /opt/akida-shard-service/*.sh && /opt/akida-shard-service/build_packages.sh
COPY src/apps/image-shard-inference/client/ /opt/akida-shard-client/
# The client speaks the same task wire format as the services, so it gets the same module.
RUN chmod +x /opt/akida-shard-client/*.sh \
    && cp /opt/akida-shard-service/shard_wire.py /opt/akida-shard-client/

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--hold"]
