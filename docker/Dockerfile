# Symphony + Akida demo image.
#
# The base image provides the platform runtime — Rocky Linux 8.9 (glibc 2.28),
# IBM Spectrum Symphony CE 7.3.2, the python3.6 soamapi binding, python3.12,
# and akida 2.19.1 in /opt/akida-venv. We strip the baked demo service and
# layer our own entrypoint and (from Section 3) our SOAM service.
FROM ghcr.io/brainchip-inc/symphony-akida:7.3.4

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

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--hold"]
