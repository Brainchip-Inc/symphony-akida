# How the image is built

`docker/Dockerfile` builds `symphony-akida` from public sources only: clone and
`docker build`, nothing from a private registry and no IBM credentials.

| stage | from | does |
|---|---|---|
| `licence-gate` | `rockylinux/rockylinux:8.10` | fails the build unless `ACCEPT_IBM_LICENSE=yes`, before anything IBM is downloaded, and writes the marker the runtime stage copies |
| `symphony` | `ibmcom/spectrum-symphony:7.3.2.0`, digest-pinned | nothing; it exists only so the runtime stage can `COPY --from` the installed `/opt/ibm/spectrumcomputing` tree, including the `pythonapi_3.6.7` SOAM binding and the Community Edition entitlement |
| runtime | `rockylinux/rockylinux:8.10` | OS prerequisites, `egoadmin` 1000:1000, the harvested tree, a fresh 10-year PKI, CPython 3.12 in `/opt/python3.12`, akida + numpy in `/opt/akida-venv`, then this repo's entrypoint and three app backends |

IBM publishes Community Edition only as a container image, and the 3 GB installer is
behind an IBMid, so harvesting the tree out of IBM's own image is the only way to build
this from a fresh clone.

## The three build-time scripts

Each one fails the build rather than shipping something subtly wrong.

**`docker/patch_symphony.sh`** extends the kernel-major check in `profile.soam` and
`profile.perf`. IBM's copies accept only 3, 4 and 5, so on a 6.x host `BINARY_TYPE` stays
`fail` and every SOAM path resolves under `soam/7.3.2/fail/`. It also fixes `BINARY_TYPE`
in `webserverstart.sh` behind the `:8443` console, and restores the stock `ego.conf` and
SD/RS service definitions. That last part is deliberate: IBM's image turns on
`EGO_TRANSPORT_SECURITY=SSL` and sets `EGO_LIM_IS_IN_CONTAINER=Y`, neither of which this
demo has ever run on, and the latter changes how LIM counts cores, which matters because
the cluster sits exactly on the Community Edition 64-core cap.

**`docker/gen_certs.sh`** mints a fresh 10-year PKI, because IBM's baked certificates are
all expired: the `wlp` leaf in January 2023, and `kernel/conf/server.pem` is the gSOAP
sample certificate that expired in **2005**. Their `generate_ssl.sh` cannot help, since
its openssl half is commented out and it re-signs with the expired CA. The script reads
the keystore password and aliases out of IBM's own keystores rather than assuming them,
so Liberty's `server.xml` keeps working untouched. It is also installed as
`/usr/local/bin/gen-certs` and can be re-run against a live container:

```bash
docker exec symphony-master /usr/local/bin/gen-certs --cn my-host.local
```

**`docker/verify_image.sh`** is the acceptance test, also installed as
`/usr/local/bin/verify-image`. It asserts `import soamapi` under python3.6, `import akida`
under python3.12, the `ldd` closure of IBM's RHEL7-era binaries now running on glibc 2.28,
that the bundled JRE runs, that `profile.platform` resolves `BINARY_TYPE` with no `/fail/`
paths, certificate validity, and that the `7.3.2` literal in the five wrapper scripts and
four XML profiles still matches the harvested tree.

```bash
docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full
```

## Why Rocky Linux 8

EL8 is forced, not chosen. The akida wheel is `manylinux_2_28` and needs glibc >= 2.26
with GLIBCXX >= 3.4.22, so it cannot run on IBM's UBI 7.9 base. Symphony's only Python
SOAM binding is a sourceless `.pyc` frozen to the CPython **3.6** ABI. EL8 is the only
line that ships python3.6 *and* supports python3.12; EL9 dropped python3.6.

That is why the image carries two interpreters: `/usr/bin/python3.6` runs the SOAM service
containers, `/opt/python3.12` runs the akida workers, and they talk over framed stdio.

EL8 reaches end of life on 2029-05-31. Past that, the options are a Symphony release with
a modern Python binding, or the C++ SOAM API behind a thin extension built for whatever
Python is current.

## Pinned versions

| what | version | why pinned |
|---|---|---|
| `ibmcom/spectrum-symphony` | `7.3.2.0`, by digest | the digest also pins the architecture |
| `rockylinux/rockylinux` | `8.10` | see above |
| CPython | `3.12.8` (python-build-standalone `20241219`, SHA-256 verified) | reproducible, independent of the distro |
| `akida` | `2.19.2` | `models/tiled_yolov2_voc.fbz` is serialized by 2.19.2 and 2.19.1 refuses to deserialize it |
| `numpy` | `2.1.3` | matches the akida wheel |

Rebuild the image after pulling a change that touches `docker/`.

## Security posture

This is a single-host demo, and its defaults say so. Containers run `--privileged`, the
Symphony console uses the stock `Admin` / `Admin` credentials behind a self-signed
certificate, and the EGO base transport runs in plaintext, as this demo always has. None
of that is a production posture. The regenerated PKI still serves the console on `:8443`
and satisfies `ego.conf`'s `EGO_*_TS_PARAMS`.
