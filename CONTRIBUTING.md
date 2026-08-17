# Contributing

This project uses a small commit-message convention enforced by a local Git hook, so
history stays clean and readable.

## Install the Git hook (run once after clone)

```bash
./scripts/install_git_hooks.sh
```

This installs `.git/hooks/commit-msg` (the existing Git LFS hooks are left untouched).

## Commit message format

Each commit subject must be:

> `type(scope): message with at least three words`

- **type**: required; one of the allowed types below.
- **scope**: optional but recommended (e.g. `service`, `launch`, `kws`).
- **message**: at least three words.

Invalid messages are rejected by the hook. To bypass for a one-off:
`SKIP_COMMIT_MSG_CHECK=1 git commit ...`.

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
