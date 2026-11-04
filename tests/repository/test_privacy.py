"""No private hosts, users, paths, cluster defaults or review-process residue in the tree.

Machine, user, allocation and node names are checked by the SHA-256 of each word, so this
file, which is public too, names none of them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from conftest import REPO

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "work"}
SKIP_FILES = {"uv.lock", Path(__file__).name}

# SHA-256 of the lowercase private words: machine names, cluster user and allocation names,
# node names and the cluster's old default checkout name. A word also matches with its
# trailing digits removed (a numbered machine or user).
PRIVATE_WORD_HASHES = frozenset({
    "ce52b7da492a2713f8617e7ba30332e3a9ba20832a15b8aed3239972114c2b1e",
    "fb4e71f75b173583843c5dbb5604c53025493bc66bd5dbf2dd59968f37e01bfb",
    "3f780370dcefe1e745f2808b64c876778d5e606a67a8a2367e44f645f8307557",
    "cb16f4d3ac48d01e4aa6242ceac542b27904ba8990ac416fecf5e772b0d01627",
    "ca71c5e34fa20e4dd07c010ce58b95b4e87d3f6845bf81f66fbc11a4cf56bc3d",
    "f2f82eaf599508ac368b33805d93ef61c1fec3ea05fc36c558828df7f5492664",
    "ee6a2830c22dda4f4bc5da58b84db49481277dc1652565f2f0943d25bcd56bd9",
    "ceb040cebf796c3c94d85450bb6997de50bff4fb02ddffa427026744250c0c2b",
    "a21a966c4f4ae7911d4563f3becc490490a58ab0151fe883c37a4160a1e7e1b3",
    "de88dcc495055dd2291007586423a3df9cdd877ac928d25504bb9cecd9be8f1f",
    "8d9f4815b49616eb5ac47d4db431a38cea1970dacb668cad8f0000db5a225516",
    "008c3abc0705ba29f167482a9216aa201bee5bb424e41d0bea520de07a6dba08",
    "91e0b99d61526a6606b6d93e2da3eaaa38c2e9ca171bcc91b207cda893fa7080",
    "0c68874543a89024d29d8725243c69a5257163f28cdb5a5eacff5a1eaf3e6d32",
    "c026e8f1e08169e795f0284026b9acf6abcbc47396947e3a97dc46461c9707d6",
    "baaf810baa365cd1de4ddb9dfe3be403a8cd17550ef8a43394b842e41e936aa4",
    "4028853bcf484786252d8a87f224ea3b3442b19233fcfa5d7cd0a6658765ba01",
    "a24d4074589216b6d35e7074708ae9f89a3960dfa12e53f8db9d9f6a6f743cd5",
})

FORBIDDEN = {
    "home or allocation paths": r"(/home/[a-z]|/work/nvme|/scratch/[a-z])",
    "review process": r"(\breviewers?\b|\bktkD\b|\bnK5Y\b|\bPnLF\b|\bQNJP\b|oVM1ErNilo|openreview\.net|rebuttal)",
    "anonymization": r"(anonymous\.4open|4open\.science)",
    "em dash": "\u2014",
}


def _sha(word: str) -> str:
    return hashlib.sha256(word.encode()).hexdigest()


def private_words(line: str, hashes: frozenset[str] = PRIVATE_WORD_HASHES) -> list[str]:
    """Words of ``line`` whose hash, or whose hash without trailing digits, is in ``hashes``."""
    lower = line.lower()
    words = set(re.findall(r"[a-z0-9_]+", lower)) | set(re.findall(r"[a-z0-9]+", lower))
    return sorted(w for w in words if _sha(w) in hashes or _sha(w.rstrip("0123456789")) in hashes)


def _files() -> list[Path]:
    out = []
    for path in REPO.rglob("*"):
        if path.is_file() and not (set(path.relative_to(REPO).parts) & SKIP_DIRS) \
                and path.name not in SKIP_FILES:
            out.append(path)
    return out


def _text(path: Path) -> str | None:
    data = path.read_bytes()
    if path.suffix == ".gz" or b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


FILES = _files()


def test_tree_is_scanned():
    assert len(FILES) > 80


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_forbidden_strings(label):
    pattern = re.compile(FORBIDDEN[label], re.IGNORECASE)
    hits = []
    for path in FILES:
        text = _text(path)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:120]}")
    assert not hits, f"{label}:\n" + "\n".join(hits[:50])


def test_no_private_words():
    hits = []
    for path in FILES:
        rel = str(path.relative_to(REPO))
        if private_words(rel):
            hits.append(f"file name: {rel}")
        text = _text(path)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if private_words(line):
                hits.append(f"{rel}:{n}: {line.strip()[:120]}")
    assert not hits, "private words:\n" + "\n".join(hits[:50])


def test_pattern_catches_a_planted_leak():
    # The scanner must be able to fail: each pattern matches a known-bad sample.
    samples = {
        "home or allocation paths": "/home/someone/x",
        "review process": "Reviewer ktkD", "anonymization": "anonymous.4open.science/r/x",
        "em dash": "a \u2014 b",
    }
    for label, sample in samples.items():
        assert re.search(FORBIDDEN[label], sample, re.IGNORECASE), label


def test_private_word_matcher_catches_a_planted_word():
    planted = frozenset({_sha("plantedhost")})
    assert private_words("ssh PlantedHost03 -p 22", planted) == ["plantedhost03"]
    assert private_words("export DIR=/x/plantedhost/run_1", planted) == ["plantedhost"]
    assert private_words("a planted host, not the word", planted) == []
    assert len(PRIVATE_WORD_HASHES) == 18
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in PRIVATE_WORD_HASHES)
