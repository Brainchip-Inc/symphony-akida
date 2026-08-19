# Contributing

Thank you for your interest in this project.

## Pull requests

This repository is maintained by BrainChip, and changes are made by the maintainer team.
Pull requests opened from outside that team will be closed.

Every result published here is tied to one hardware configuration, so a change is validated
by rebuilding the image, bringing the Akida fleet up and re-measuring. Keeping that
validation with the maintainers is what lets the numbers in these guides stay accurate.

## Reporting a problem

If a demo does not work for you, please open an issue. That is the most useful thing you
can send us, and we read every one.

Please search the existing issues first, then include:

- what you ran, and what happened instead of what you expected
- your host OS and kernel version
- how many Akida devices you have and which family (`ls /dev/akd1500_* /dev/akida*`)
- the `akida` version in the image:
  `docker run --rm -e PYTHONPATH=/opt/akida-venv/lib/python3.12/site-packages --entrypoint /opt/python3.12/bin/python3.12 symphony-akida -c 'import akida; print(akida.__version__)'`
- the output of `docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full`
- the smallest log excerpt that shows the failure

You do not need to diagnose the cause or propose a fix. A clear description of what broke
is enough for us to work from.

## Questions and ideas

The [BrainChip Developer Hub](https://developer.brainchip.com/signup/) has the tools, model
zoo and documentation for the wider Akida platform, and the
[BrainChip Discord](https://discord.com/invite/9bmd9g52vn) is the place for questions,
ideas, and showing us what you have built.

## Building on this work

Please do. This repository is Apache 2.0 licensed precisely so you can fork it and take it
in your own direction without asking us first. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for the terms, including those of the IBM and BrainChip components the build fetches.

---

## For maintainers

Commit subjects are checked by a local hook, so history stays clean and readable.

```bash
./scripts/install_git_hooks.sh   # installs .git/hooks/commit-msg, once after clone
```

Each commit subject must be `type(scope): message with at least three words`. The scope is
optional but recommended (`service`, `launch`, `kws`). Invalid messages are rejected; to
bypass for a one-off, `SKIP_COMMIT_MSG_CHECK=1 git commit ...`.

<details>
<summary><b>Allowed types</b></summary>

| Type | When to use |
|------|-------------|
| **feat** | A new feature or enhancement. |
| **fix** | A bug fix. |
| **docs** | Documentation-only changes. |
| **style** | Formatting only; no behavior change. |
| **refactor** | Restructuring without behavior change; use for most renames. |
| **perf** | Performance improvements. |
| **test** | Adding or updating tests. |
| **build** | Build system / tooling (Dockerfile, uv, scripts). |
| **ci** | CI/CD configuration. |
| **chore** | Routine maintenance (deps, cleanup). |
| **revert** | Reverting a previous commit. |

</details>

<details>
<summary><b>Examples</b></summary>

```
feat(service): map model on akida hardware
fix(kws): handle zero-length audio input
build(image): add repo dockerfile and entrypoint
docs(guides): add app startup guides
```

</details>
