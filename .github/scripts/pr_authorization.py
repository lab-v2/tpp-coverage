"""The `pr-authorization` check: who may merge a pull request into main.

A pull request may merge only when its author is trusted, or when a trusted
account approved its current head commit. Trusted means the repository owner's
login, or an `author_association` of OWNER or MEMBER (a member of the owning
organization). COLLABORATOR, CONTRIBUTOR and NONE are not trusted.

Branch protection cannot express "author OR approval", so
.github/workflows/pr-authorization.yml runs this, as a required check, on every
pull_request and pull_request_review event, reading the pull request and its
reviews from the API with the job's GITHUB_TOKEN. Exit 0 when it may merge.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: One object of GitHub's REST answers (a pull request or a review).
Record = dict[str, Any]

#: Always trusted, whatever association GitHub reports for it.
OWNER_LOGINS = frozenset({"krichelj"})
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER"})
#: Review states that replace an account's earlier verdict. COMMENTED and PENDING
#: do not: GitHub keeps an approval standing after a later plain comment.
VERDICT_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def is_trusted(login: str | None, association: str | None) -> bool:
    """Whether an account counts as the owner or a member of the organization."""
    if not login:
        return False
    return login.lower() in OWNER_LOGINS or (association or "").upper() in TRUSTED_ASSOCIATIONS


def standing_verdicts(reviews: list[Record]) -> dict[str, Record]:
    """Each account's latest review that carries a verdict, in the API's chronological order."""
    latest: dict[str, Record] = {}
    for review in reviews:
        login = (review.get("user") or {}).get("login")
        if login and review.get("state") in VERDICT_STATES:
            latest[login.lower()] = review
    return latest


def decide(pull: Record, reviews: list[Record]) -> Decision:
    """The verdict for one pull request, from its API record and its reviews.

    An approval counts only when it is the approving account's standing verdict,
    that account is trusted, and it was given on the pull request's head commit,
    so a push after the approval needs a new one.
    """
    author = (pull.get("user") or {}).get("login")
    association = pull.get("author_association")
    if is_trusted(author, association):
        return Decision(True, f"author {author} ({association}) is the owner or a lab-v2 member")
    head = (pull.get("head") or {}).get("sha")
    approvers = sorted(
        login
        for login, review in standing_verdicts(reviews).items()
        if review.get("state") == "APPROVED"
        and is_trusted(login, review.get("author_association"))
        and login != (author or "").lower()
        and head
        and review.get("commit_id") == head
    )
    if approvers:
        return Decision(
            True,
            f"author {author} ({association}) is not a member; head {head[:12]} approved by "
            + ", ".join(approvers),
        )
    return Decision(
        False,
        f"author {author} ({association}) is not the owner or a lab-v2 member, and no owner or "
        f"member approval of head {(head or '?')[:12]} stands",
    )


def api(path: str) -> object:
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    req = urllib.request.Request(f"{base}/{path}")
    req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


def all_reviews(repo: str, number: int) -> list[Record]:
    """Every review of the pull request, oldest first, across all pages."""
    out: list[Record] = []
    page = 1
    while True:
        batch = api(f"repos/{repo}/pulls/{number}/reviews?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise RuntimeError(f"reviews page {page} is not a list")
        out += [r for r in batch if isinstance(r, dict)]
        if len(batch) < 100:
            return out
        page += 1


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = (event.get("pull_request") or {}).get("number")
    if not number:
        print("pr-authorization: this event has no pull request")
        return 1
    repo = os.environ["GITHUB_REPOSITORY"]
    # The API's record, not the event's copy: a review event carries the pull
    # request as it was, and the head or the reviews may have moved since.
    pull = api(f"repos/{repo}/pulls/{number}")
    if not isinstance(pull, dict):
        raise RuntimeError("the pull request record is not an object")
    decision = decide(pull, all_reviews(repo, number))
    print(f"pr-authorization: {'allowed' if decision.allowed else 'refused'}: {decision.reason}")
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    sys.exit(main())
