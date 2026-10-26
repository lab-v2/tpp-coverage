#!/usr/bin/env python3
"""Download the normalised run tables from HuggingFace (no account needed).

This is the fast alternative to ``fetch_data.py``: the same two tables that
``fetch_data.py`` builds from the upstream releases, as published with the paper.

    runs_normalised.csv   terminal observations, the five-cell corpus of CRITERIA.md
    runs_expanded.csv     the ten-suite corpus (E = terminal, X = adds mid-training)

Both land in ``<work>/artifacts/`` (see paths.py), where every other script reads them.
"""

from __future__ import annotations

import argparse
import shutil

from paths import ARTIFACTS, HF_REPO_ID

TABLES = ("runs_normalised.csv", "runs_expanded.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", nargs="+", default=list(TABLES), choices=TABLES)
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in args.tables:
        cached = hf_hub_download(HF_REPO_ID, name, repo_type="dataset")
        shutil.copyfile(cached, ARTIFACTS / name)
        print(f"{HF_REPO_ID}/{name} -> {ARTIFACTS / name}")


if __name__ == "__main__":
    main()
