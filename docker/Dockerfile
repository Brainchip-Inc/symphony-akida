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

# Our SOAM service (python3.6 ServiceContainer + python3.12 on-chip worker +
# wrapper + profile) and a prebuilt deploy package for the entrypoint to
# register on the master's first boot.
COPY src/apps/batch-inference/service/ /opt/akida-service/
RUN chmod +x /opt/akida-service/*.sh && /opt/akida-service/build_package.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--hold"]
