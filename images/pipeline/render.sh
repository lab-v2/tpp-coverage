#!/usr/bin/env bash
# Render images/pipeline/pipeline.mmd to pipeline.png, the README's code-structure diagram, and record
# the source's sha256 beside it. GitHub's mobile app shows a Mermaid block as source, so the README
# embeds this PNG. Needs mermaid-cli (npm install -g @mermaid-js/mermaid-cli@11.4.2) and a Chromium.
set -euo pipefail
cd "$(dirname "$0")"
chrome=${CHROME:-$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome 2>/dev/null | tail -1)}
pptr=$(mktemp)
trap 'rm -f "$pptr"' EXIT
printf '{"executablePath":"%s","args":["--no-sandbox"]}' "$chrome" >"$pptr"
mmdc -p "$pptr" -c config.json -i pipeline.mmd -o pipeline.png -s 3 -b transparent
sha256sum pipeline.mmd | cut -d' ' -f1 >pipeline.png.sha256
