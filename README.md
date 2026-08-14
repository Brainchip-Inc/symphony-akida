# Symphony + Akida — on-chip fleet inference

Distribute AI inference across a fleet of **BrainChip Akida** AKD1000 devices on an **IBM
Spectrum Symphony** (Community Edition) cluster. One master + one compute node per chip
(capped at 7 — the CE 64-core limit); each node maps the model onto its chip
(`hw_only=True`) and runs inference **on-silicon**.

## Three demos, one image

All apps build from the same image (`symphony-akida`) and the same cluster.
The launcher activates exactly one at a time — they never run in parallel. Run them
back-to-back to show the contrast:

| App | Transport | Dispatch | Effect | Guide |
|---|---|---|---|---|
| **batch-inference** | Symphony SOAM | concurrent fan-out | every chip busy at once | [guide →](src/apps/batch-inference/README.md) |
| **serial-http-round-robin** | plain HTTP | round-robin, one at a time | ~one chip busy at a moment | [guide →](src/apps/serial-http-round-robin/README.md) |
| **image-shard-inference** | Symphony SOAM (3-stage) | split → fan-out → merge | one 448 frame across 6 chips in parallel, with a real mAP | [guide →](src/apps/image-shard-inference/README.md) |

Each app's README is the full clone → build → launch walkthrough.

<details>
<summary><b>Setup (once — shared by all three apps)</b></summary>

Run on the host with the Akida cards (`/dev/akida*` + the `akida_pcie` driver) and Docker.
The build needs network access to Docker Hub, PyPI and GitHub releases; it needs **no**
private registry and no IBM credentials.

```bash
git clone <repo-url> symphony-akida && cd symphony-akida
git lfs install && git lfs pull                    # model .fbz + anchors + sample .npz
curl -LsSf https://astral.sh/uv/install.sh | sh    # host tooling for the dashboards
uv sync
docker build --build-arg ACCEPT_IBM_LICENSE=yes \
    -f docker/Dockerfile -t symphony-akida .       # bakes ALL app backends
```

`ACCEPT_IBM_LICENSE=yes` is required and has no default. The image is built from IBM
Spectrum Symphony Community Edition, so you accept IBM's licence yourself rather than this
repository doing it for you. Without the flag the build stops before anything is downloaded
and prints what it is asking you to agree to. See [Licensing](#licensing).

The first build pulls ~1.5 GB (IBM's Symphony CE image) plus ~67 MB (CPython 3.12), so it
takes a while; rebuilds are cached.

Sanity-check a fresh build before launching anything — it asserts every invariant the apps
depend on and names the exact one that broke:

```bash
docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full
```

The image pins **akida 2.19.2**; the tiled YOLOv2 checkpoint is serialized by it and 2.19.1
refuses to deserialize it, so rebuild the image after pulling a change that touches
`docker/`.

Then open the app you want to run and follow its README. To switch demos, tear down and
bring the other up:

```bash
./launch/down.sh && ./launch/up.sh <batch-inference|serial-http-round-robin|image-shard-inference>
```

`up.sh` takes `--nodes N|all` to choose how many chips to use — it defaults to 6 for
`image-shard-inference`, one per tile of a 448 frame.
</details>

<details>
<summary><b>Repository layout</b></summary>

```
docker/     Dockerfile (public sources only) + entrypoint + the patch / PKI / verify
            scripts it runs at build time; bakes all three app backends
launch/     up.sh <app> [--nodes N|all] [--dataset <npz>] / down.sh
models/     on-chip .fbz models + anchors (Git LFS)
data/       samples/ committed .npz sets (Git LFS); voc/ test kits symlinked in, never committed
scripts/    sample generation, reference verification, mAP scoring
src/
  common/   shared code: akida_chip (on-chip core), tiled_shard (tile geometry, decode and
            merge), detection_map (mAP), testkit (VOC test kit reader), draw_detections,
            worker_io, models allowlist, sample prep
  apps/
    batch-inference/          SOAM service + client + dashboard (concurrent)
    serial-http-round-robin/  per-node HTTP server + client + dashboard (serial)
    image-shard-inference/    3 SOAM services (segment/inference/stitch) + client + dashboard
```
</details>

<details>
<summary><b>Design constraints</b></summary>

- **Community Edition ≤ 64 cores** → master + 7 compute; an 8th chip idles.
- **On-chip only** — a node with no mappable Akida device is not used for work.
- **Six chips for the shard demo** — one per tile of a 448 frame; the sixth tile is the whole
  frame downscaled, and dropping it costs more accuracy than dropping the other five.
- **Repo-local** — everything under `.cluster/` (bind-mounted to `/shared`); no `/opt`, no host `sudo`.
- **The image builds from public sources only** — clone and `docker build`, nothing from a
  private registry. The Symphony CE tree is harvested out of IBM's own
  `ibmcom/spectrum-symphony:7.3.2.0` (pinned by digest), CPython 3.12 from a pinned
  python-build-standalone release, and `akida` from PyPI. See *How the image is built*.
- **EL8 is forced, not chosen** — the akida wheel is `manylinux_2_28` and needs glibc ≥ 2.26
  with GLIBCXX ≥ 3.4.22, so it cannot run on IBM's UBI 7.9 base; and Symphony's only Python
  SOAM binding is a sourceless `.pyc` frozen to the CPython **3.6** ABI. EL8 is the only line
  that ships python3.6 *and* supports python3.12. EL9 dropped python3.6. EL8 goes EOL
  2029-05-31; past that, the options are a Symphony release with a modern Python binding, or
  the C++ SOAM API behind a thin extension built for whatever Python is current.
</details>

<details>
<summary><b>How the image is built</b></summary>

`docker/Dockerfile` is a two-stage build:

| stage | from | does |
|---|---|---|
| `symphony` | `ibmcom/spectrum-symphony:7.3.2.0`, digest-pinned | nothing — it exists only so the runtime stage can `COPY --from` the installed `/opt/ibm/spectrumcomputing` tree, including the `pythonapi_3.6.7` SOAM binding and the CE entitlement. IBM publishes CE only as an image, and the 3 GB installer is behind an IBMid, so harvesting is the only way to build this from a fresh clone |
| runtime | `rockylinux/rockylinux:8.10` | OS prerequisites, `egoadmin` 1000:1000, the harvested tree, a fresh 10-year PKI, CPython 3.12 in `/opt/python3.12`, akida + numpy in `/opt/akida-venv`, then this repo's entrypoint and three app backends |

Three build-time scripts do the work that makes the harvested tree usable, and each one fails
the build rather than shipping something subtly wrong:

- **`docker/patch_symphony.sh`** — extends the kernel-major check in `profile.soam` and
  `profile.perf` (IBM's copies accept only 3/4/5, so on a 6.x host `BINARY_TYPE` stays `fail`
  and every SOAM path resolves under `soam/7.3.2/fail/`), fixes `BINARY_TYPE` in
  `webserverstart.sh` behind the `:8443` console, and restores the stock `ego.conf` and
  SD/RS service definitions. The last part is deliberate: IBM's image turns on
  `EGO_TRANSPORT_SECURITY=SSL` and sets `EGO_LIM_IS_IN_CONTAINER=Y`, neither of which this
  demo has ever run on — and the latter changes how LIM counts cores, which matters because
  the cluster sits exactly on the CE 64-core cap.
- **`docker/gen_certs.sh`** — mints a fresh 10-year PKI. IBM's baked certificates are all
  expired: the `wlp` leaf in January 2023, and `kernel/conf/server.pem` is the gSOAP sample
  certificate that expired in **2005**. Their `generate_ssl.sh` cannot help, since its
  openssl half is commented out and it re-signs with the expired CA. The script reads the
  keystore password and aliases out of IBM's own keystores rather than assuming them, so
  Liberty's `server.xml` keeps working untouched.
- **`docker/verify_image.sh`** — the acceptance test, also installed as
  `/usr/local/bin/verify-image`. It asserts `import soamapi` under python3.6, `import akida`
  under python3.12, the `ldd` closure of IBM's RHEL7-era binaries now running on glibc 2.28,
  that the bundled JRE runs, that `profile.platform` resolves `BINARY_TYPE` with no `/fail/`
  paths, certificate validity, and that the `7.3.2` literal in the five wrapper scripts and
  four XML profiles still matches the harvested tree.

The EGO base transport runs in plaintext, as this demo always has. The regenerated PKI still
serves the console on `:8443` and satisfies `ego.conf`'s `EGO_*_TS_PARAMS`.
</details>

## Licensing

This repository contains no IBM or BrainChip binaries. `docker build` fetches them, and the
terms below are the ones that apply to what it fetches.

**IBM Spectrum Symphony Community Edition 7.3.2.** The image is built from IBM's public
container image [`ibmcom/spectrum-symphony:7.3.2.0`](https://hub.docker.com/r/ibmcom/spectrum-symphony),
pinned by digest in `docker/Dockerfile`, and the build copies the installed Symphony tree out
of it. That software is licensed by IBM under IBM's terms, separately from this repository's
licence. `docker build` therefore requires `--build-arg ACCEPT_IBM_LICENSE=yes`; there is no
default, and without it the build stops before contacting IBM's registry. The full licence
text ships inside the built image at `/licenses` (`LA_*` license agreement, `LI_*` license
information, `Lic_*`, `non_ibm_license.txt`, `notices.txt`), and `LI_en.txt` names the program
as *IBM Spectrum Symphony Community Edition, 7.3 (Community)*. The Community Edition
entitlement file the cluster uses comes from IBM's own image; this repository supplies no
licence key.

**BrainChip akida (MetaTF).** Installed from PyPI at build time, pinned in
`docker/Dockerfile`, under BrainChip's terms for that package.

A consolidated third-party notice covering IBM, MetaTF and this repository's own licence is
still to be written.

## Contributing

Commit convention and hook install: [CONTRIBUTING.md](CONTRIBUTING.md).
