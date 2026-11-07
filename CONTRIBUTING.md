# Contributing

## Reporting bugs

Open a [GitHub issue](https://github.com/nat-e/frigate-decisions/issues), with the logs around the problem and the part of your config that matters. For security problems, follow [SECURITY.md](SECURITY.md) instead.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync                                       # install, with the dev tools
uv run frigate-decisions --config config.yml  # run it outside Docker
```

`config.yml` at the root of the repo is ignored by git, so your camera names and settings stay out of commits.

## Checks

CI runs these, so run them before opening a pull request:

```sh
uv run ruff format   # formatting
uv run ruff check    # lint
uv run basedpyright  # types
uv run pytest        # tests
```

CI also builds the Docker image and checks that the app starts in it.

## Adding a decision model

Write a class with the `Provider` attributes (`image_sides`, `max_questions`) and an `ask(image, context, questions)` method in [`providers.py`](src/frigate_decisions/providers.py), add it to `PROVIDERS`, and list it in the README's "Decision models" table.

## Pull request titles

Pull requests are squash merged, so the title becomes the commit message on `main`. [release-please](https://github.com/googleapis/release-please) reads these messages to pick the next version and write `CHANGELOG.md`, so titles must follow [Conventional Commits](https://www.conventionalcommits.org):

```
<type>(<optional scope>): <what changes>
```

| Type | Use for | Release |
| -- | -- | -- |
| `feat` | A new feature | Minor version (patch before 1.0) |
| `fix` | A bug fix | Patch version |
| `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`, `style`, `revert` | Everything else | None on its own |

Add `!` after the type for a breaking change, such as a renamed config key: `feat!: …`. Before 1.0, breaking changes bump the minor version.

Examples:

- `fix: events never end when Frigate restarts`
- `feat(providers): add OpenAI Decisions`
- `chore(deps): update pillow to 13.1`

A CI check rejects titles that don't follow this format.

## Releases

release-please keeps a release pull request open, which bumps the version in `pyproject.toml` and `uv.lock` and updates `CHANGELOG.md`. Merging it tags the release, and CI publishes the image. Before 1.0, GitHub releases are marked as pre-releases.

To pick the next version yourself, add `Release-As: x.y.z` to the description of a pull request. It ends up in the squash commit's body, where release-please reads it.

Images are published to `ghcr.io/nat-e/frigate-decisions`, for amd64 and arm64, with these tags:

| Tag | Points to |
| -- | -- |
| `latest` | The newest release |
| `1.2.3`, `1.2` | A release |
| `edge` | The latest commit on `main` |
| Commit SHA | That commit on `main` |

## Dependency updates

[Renovate](https://docs.renovatebot.com) (`renovate.json`) opens update PRs on Mondays, only for releases at least 7 days old:

- Patch updates and GitHub Actions updates: one PR each, which Renovate merges itself once CI passes.
- Minor updates, Python, and uv: one PR each, merged by hand.
- Major updates: one PR per dependency, merged by hand.
- `uv.lock` maintenance: one PR, merged by hand.

Updates to the app's dependencies and to the `Dockerfile` are titled `fix(deps): …`, so they appear in the changelog. The "Dependency Dashboard" issue lists pending updates.
