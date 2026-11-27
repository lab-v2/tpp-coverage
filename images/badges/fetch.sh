#!/usr/bin/env bash
# Renders the README's badges from shields.io into rendered/. The README links the files,
# not shields.io: GitHub's image proxy timed out (HTTP 504) on every shields.io badge with a
# logo (2026-11-26). neurips_mark.png is NeurIPS's dots mark, cut from neurips.cc's favicon
# and made white, since Simple Icons has no NeurIPS logo. Rerun after changing a badge.
set -euo pipefail
cd "$(dirname "$0")"
mark=$(base64 -w0 neurips_mark.png)
S=https://img.shields.io
while read -r name url; do
  curl -fsS --retry 3 -o "rendered/$name.svg" "$url"
done <<LIST
arxiv $S/badge/arXiv-2605.08541-b31b1b.svg?logo=arxiv&logoColor=white
project-page $S/badge/project-page-F76900.svg?logo=githubpages&logoColor=white
neurips $S/badge/NeurIPS-2026-6f42c1.svg?logo=data:image/png;base64,$mark
huggingface $S/badge/data-HuggingFace-ffcc4d.svg?logo=huggingface
license $S/badge/license-GPL--2.0-blue.svg?logo=gnu&logoColor=white
python $S/badge/python-3.10-blue.svg?logo=python&logoColor=white
uv $S/badge/managed%20with-uv-261230.svg?logo=uv&logoColor=white
ruff $S/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
LIST
