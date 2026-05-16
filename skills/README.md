# Project Skills

This directory stores external skill bundles vendored for this project.

## nature-skills

- Source: https://github.com/Yuan1z0825/nature-skills
- Vendored at: `nature-skills/`
- Upstream commit: `31bc3bd9aa81ccdbc2b6c8239b92657385c17f08`
- License: MIT, preserved in `nature-skills/LICENSE`

The upstream repository keeps installable skills under `nature-skills/skills/`.
Current bundled skills include:

- `nature-academic-search`
- `nature-citation`
- `nature-data`
- `nature-figure`
- `nature-paper2ppt`
- `nature-polishing`
- `nature-reader`
- `nature-response`
- `nature-writing`

## Local Tooling

Common plotting, PDF, PPT, and citation helper dependencies were installed into
the project `.venv`. The reproducible dependency list is:

- `requirements-nature-skills.txt`

The `nature-academic-search` MCP server needs the `mcp` package, which is not
available for the project's Python 3.9 environment. A dedicated Python 3.13
environment was created at:

- `.venv-nature-skills/`

Its reproducible dependency list is:

- `requirements-academic-search-mcp.txt`

