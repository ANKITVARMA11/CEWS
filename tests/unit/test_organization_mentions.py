"""Unit tests for ``organization_mentions`` - which record fields become candidate
organizations for the resolver to match or create.

Regression focus: a publication's author affiliations must never be resolved as organizations.
Root-caused from a real database where "waiting for review" reached 146,206 items against 20
tracked competitors - traced to every author affiliation on every publication (up to
``MAX_AFFILIATIONS`` per paper, over three years of PubMed/Europe PMC records) being run through
the same organization resolver as real sponsors and assignees. Affiliation text is extremely
free-form (department, building, city, country folded into one string), so very little of it
ever matched an existing organization cleanly - each distinct-looking affiliation created its own
new ``Organization`` row and its own review-queue item. This was verified NOT to be a
deduplication bug first (see ``tests/unit/test_review_actions.py`` and a direct experiment
re-resolving the same and near-identical names across many repeated pipeline runs, which
correctly produced one organization and one review item every time) before concluding affiliation
volume itself was the cause.
"""

from __future__ import annotations

import pytest

from cews.constants import RelationshipType
from cews.database.models import (
    Announcement,
    ClinicalTrial,
    FundingAward,
    Patent,
    Publication,
    SourceRecord,
)
from cews.normalization.pipeline import Mention, organization_mentions

pytestmark = pytest.mark.unit


def _record(**raw_payload: object) -> SourceRecord:
    return SourceRecord(
        source="test",
        source_record_id="r1",
        record_type="irrelevant",
        content_hash="hash",
        raw_payload_json=dict(raw_payload),
    )


def test_a_publications_affiliations_never_become_organization_mentions() -> None:
    """The fix: a paper's author affiliations are no longer fed through organization matching
    at all - co-authorship is not sponsorship, and the affiliation text was never clean enough
    to resolve well."""
    detail = Publication(
        publication_identifier="PM1",
        affiliations=[
            "Dept. of Oncology, Stanford University School of Medicine, Stanford, CA, USA",
            "Fixture Oncology, Inc., Boston, MA, USA",
            "Zentavia Biosciences, Cambridge, MA, USA",
        ],
    )
    mentions = organization_mentions(_record(), detail)
    assert mentions == []
    assert not any(m.relationship == RelationshipType.AFFILIATION for m in mentions)


def test_a_clinical_trials_sponsor_and_collaborators_are_still_mentioned() -> None:
    """The fix is specific to affiliations - real sponsorship evidence is untouched."""
    detail = ClinicalTrial(trial_identifier="NCT1", sponsor_name="Zentavia Biosciences")
    record = _record(
        protocolSection={
            "sponsorCollaboratorsModule": {"collaborators": [{"name": "Partner Pharma"}]}
        }
    )
    mentions = organization_mentions(record, detail)
    names_and_relationships = {(m.raw_name, m.relationship) for m in mentions}
    assert ("Zentavia Biosciences", RelationshipType.SPONSOR) in names_and_relationships
    assert ("Partner Pharma", RelationshipType.PARTNER) in names_and_relationships


def test_a_patents_assignee_is_still_mentioned() -> None:
    detail = Patent(patent_identifier="US1", assignee_name="Zentavia Biosciences")
    mentions = organization_mentions(_record(), detail)
    assert mentions == [Mention("Zentavia Biosciences", RelationshipType.ASSIGNEE)]


def test_a_funding_awards_recipient_is_still_mentioned() -> None:
    detail = FundingAward(award_identifier="A1", recipient_name="Zentavia Biosciences")
    mentions = organization_mentions(_record(), detail)
    assert mentions == [Mention("Zentavia Biosciences", RelationshipType.RECIPIENT)]


def test_an_announcements_organization_and_partners_are_still_mentioned() -> None:
    detail = Announcement(
        organization_name="Zentavia Biosciences",
        partner_organizations=["Partner Pharma"],
    )
    mentions = organization_mentions(_record(), detail)
    names_and_relationships = {(m.raw_name, m.relationship) for m in mentions}
    assert ("Zentavia Biosciences", RelationshipType.ANNOUNCER) in names_and_relationships
    assert ("Partner Pharma", RelationshipType.PARTNER) in names_and_relationships
