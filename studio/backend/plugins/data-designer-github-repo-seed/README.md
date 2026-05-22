# data-designer-github-repo-seed

A Data Designer seed-reader plugin for **Unsloth Studio** that scrapes real
GitHub data (issues, pull requests, commits) from one or more repositories
and hands it to the recipe pipeline as a seed dataset.

Designed to ship with Studio as a default seed source so any user with a
GitHub token can build training datasets straight from live repos.

## What it does

Given a list of `owner/name` repos, a GitHub token, and a per-resource
`limit`, the plugin uses GitHub's GraphQL API to fetch issues, pull
requests, and/or commits, with labels, state, authors, and the first N
comments of each item, and materialises a single JSONL with uniform
columns so the rest of the recipe (LLM text / LLM structured / processors)
can treat it like any other seed table.

| Column        | Description                                    |
|---------------|------------------------------------------------|
| `item_type`   | `issue` / `pull` / `commit`                    |
| `repo`        | `owner/name`                                   |
| `number`      | Issue/PR number, or commit SHA                 |
| `title`       | Title (or commit message headline)             |
| `body`        | Issue/PR body (or full commit message)         |
| `state`       | `OPEN` / `CLOSED` / `MERGED` (empty for commit)|
| `author`      | GitHub login of the author                     |
| `created_at`  | ISO8601                                        |
| `closed_at`   | ISO8601 (empty for commits)                    |
| `url`         | Permalink                                      |
| `labels`      | List of label names                            |
| `comments`    | First N comments concatenated                  |

## Usage in a recipe

```json
{
  "seed_config": {
    "source": {
      "seed_type": "github_repo",
      "repos": ["unslothai/unsloth", "unslothai/unsloth-zoo"],
      "token": "",
      "item_types": ["issues", "pulls"],
      "limit": 100,
      "include_comments": true,
      "max_comments_per_item": 30
    },
    "sampling_strategy": "shuffle",
    "selection_strategy": null
  }
}
```

Leave `token` empty to fall back to the server's `GH_TOKEN` / `GITHUB_TOKEN`
environment variable, useful when the recipe is published and shouldn't
carry a secret.

## Auth

A GitHub personal access token with `public_repo` scope is enough for public
repositories; `repo` scope is required for private ones. GraphQL requests
are rate-limit aware: the client inspects `x-ratelimit-*` headers and
sleeps until reset when the budget drops below a safety threshold.

## Install

Shipped as a default Studio plugin. For development:

```bash
pip install -e .
```

Registered automatically via the `data_designer.plugins` entry point.
