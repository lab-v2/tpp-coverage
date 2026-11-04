"""The merge rule of .github/scripts/pr_authorization.py, against hostile cases.

Records follow the shapes of GitHub's REST API (GET /repos/{o}/{r}/pulls/{n} and
GET .../pulls/{n}/reviews): `user.login`, `author_association`, `head.sha`, and per
review `state` and `commit_id`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import REPO

_spec = importlib.util.spec_from_file_location(
    "pr_authorization", REPO / ".github" / "scripts" / "pr_authorization.py"
)
pra = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pra  # dataclasses look their module up there
_spec.loader.exec_module(pra)

HEAD = "a" * 40
OLD = "b" * 40


def pull(login: str, association: str, head: str = HEAD) -> pra.Record:
    return {"user": {"login": login}, "author_association": association, "head": {"sha": head}}


def review(login: str, association: str, state: str, commit: str = HEAD) -> pra.Record:
    return {
        "user": {"login": login},
        "author_association": association,
        "state": state,
        "commit_id": commit,
    }


def test_owner_author_passes_without_approval() -> None:
    assert pra.decide(pull("krichelj", "MEMBER"), []).allowed


def test_owner_login_passes_whatever_the_association() -> None:
    assert pra.decide(pull("KrichelJ", "CONTRIBUTOR"), []).allowed


def test_member_author_passes_without_approval() -> None:
    assert pra.decide(pull("labmate", "MEMBER"), []).allowed


def test_org_owner_association_passes() -> None:
    assert pra.decide(pull("admin", "OWNER"), []).allowed


def test_outside_contributor_without_approval_fails() -> None:
    d = pra.decide(pull("stranger", "CONTRIBUTOR"), [])
    assert not d.allowed and "stranger" in d.reason


def test_first_time_author_without_approval_fails() -> None:
    assert not pra.decide(pull("stranger", "FIRST_TIME_CONTRIBUTOR"), []).allowed


def test_outside_contributor_approved_by_a_member_passes() -> None:
    d = pra.decide(pull("stranger", "NONE"), [review("labmate", "MEMBER", "APPROVED")])
    assert d.allowed and "labmate" in d.reason


def test_outside_contributor_approved_by_the_owner_passes() -> None:
    assert pra.decide(pull("stranger", "NONE"), [review("krichelj", "MEMBER", "APPROVED")]).allowed


def test_approval_later_dismissed_fails() -> None:
    # GitHub rewrites a dismissed review's state to DISMISSED in the reviews list.
    assert not pra.decide(
        pull("stranger", "NONE"), [review("labmate", "MEMBER", "DISMISSED")]
    ).allowed


def test_approval_followed_by_a_dismissal_record_fails() -> None:
    reviews = [review("labmate", "MEMBER", "APPROVED"), review("labmate", "MEMBER", "DISMISSED")]
    assert not pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_collaborator_author_who_is_not_a_member_fails() -> None:
    assert not pra.decide(pull("helper", "COLLABORATOR"), []).allowed


def test_approval_by_a_non_member_does_not_count() -> None:
    for assoc in ("COLLABORATOR", "CONTRIBUTOR", "NONE", "FIRST_TIME_CONTRIBUTOR"):
        reviews = [review("friend", assoc, "APPROVED")]
        assert not pra.decide(pull("stranger", "NONE"), reviews).allowed, assoc


def test_changes_requested_by_a_member_does_not_count_as_approval() -> None:
    reviews = [review("labmate", "MEMBER", "CHANGES_REQUESTED")]
    assert not pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_approval_then_changes_requested_by_the_same_member_fails() -> None:
    reviews = [review("labmate", "MEMBER", "APPROVED"), review("labmate", "MEMBER", "CHANGES_REQUESTED")]
    assert not pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_comment_after_an_approval_keeps_it_standing() -> None:
    reviews = [review("labmate", "MEMBER", "APPROVED"), review("labmate", "MEMBER", "COMMENTED")]
    assert pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_comment_by_a_member_is_not_an_approval() -> None:
    reviews = [review("labmate", "MEMBER", "COMMENTED")]
    assert not pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_approval_of_an_older_commit_does_not_count() -> None:
    reviews = [review("labmate", "MEMBER", "APPROVED", commit=OLD)]
    assert not pra.decide(pull("stranger", "NONE", head=HEAD), reviews).allowed


def test_one_member_approval_suffices_beside_a_non_member_rejection() -> None:
    reviews = [review("friend", "NONE", "CHANGES_REQUESTED"), review("labmate", "MEMBER", "APPROVED")]
    assert pra.decide(pull("stranger", "NONE"), reviews).allowed


def test_deleted_account_records_are_ignored() -> None:
    reviews = [{"user": None, "author_association": "MEMBER", "state": "APPROVED", "commit_id": HEAD}]
    assert not pra.decide(pull("stranger", "NONE"), reviews).allowed
    assert not pra.decide({"user": None, "author_association": "MEMBER", "head": {"sha": HEAD}}, []).allowed


def test_main_reads_the_api_record_not_the_event_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The event says MEMBER, the API says NONE with no approval: the API wins.
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7, **pull("stranger", "MEMBER")}}))
    answers = {
        "repos/o/r/pulls/7": pull("stranger", "NONE"),
        "repos/o/r/pulls/7/reviews?per_page=100&page=1": [],
    }
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setattr(pra, "api", answers.__getitem__)
    assert pra.main() == 1
    assert "refused" in capsys.readouterr().out


def test_reviews_are_read_across_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    first = [review(f"x{i}", "NONE", "COMMENTED") for i in range(100)]
    second = [review("labmate", "MEMBER", "APPROVED")]
    pages = {
        "repos/o/r/pulls/7/reviews?per_page=100&page=1": first,
        "repos/o/r/pulls/7/reviews?per_page=100&page=2": second,
    }
    monkeypatch.setattr(pra, "api", pages.__getitem__)
    got = pra.all_reviews("o/r", 7)
    assert len(got) == 101
    assert pra.decide(pull("stranger", "NONE"), got).allowed


@pytest.mark.parametrize("state", ["APPROVED", "CHANGES_REQUESTED", "DISMISSED"])
def test_every_verdict_state_replaces_an_earlier_one(state: str) -> None:
    reviews = [review("labmate", "MEMBER", "APPROVED"), review("labmate", "MEMBER", state)]
    assert pra.decide(pull("stranger", "NONE"), reviews).allowed is (state == "APPROVED")
