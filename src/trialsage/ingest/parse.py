"""Map a raw ClinicalTrials.gov API v2 study into flat Python objects.

Every field path below was verified against live API responses rather than
recalled -- the JSON nests everything under ``protocolSection`` modules and the
exact paths are easy to get subtly wrong. See ``docs/api_fields.md`` for the
verification notes.

Two shapes deserve attention because they are easy to model incorrectly:

* ``designModule.phases`` is a **list** -- a trial can be PHASE1/PHASE2.
* ``contactsLocationsModule.locations`` is a list of sites, each with its own
  ``status``. A trial can be RECRUITING overall while a given site is closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .ages import parse_age_to_years
from .eligibility import Criterion, split_eligibility

_DATE_FULL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DATE_YEAR = re.compile(r"^(\d{4})$")


def parse_ctg_date(raw: Optional[str]) -> Tuple[Optional[date], Optional[str]]:
    """Parse a registry date, returning ``(date, precision)``.

    The API emits ``2024-03-15``, ``2024-03`` or occasionally ``2024``. We always
    return a real ``date`` (imputing the 1st where the day or month is missing)
    plus the precision, so downstream code can tell a reported day from an
    invented one. Year filtering -- which is what the questions actually need --
    is unaffected by the imputation.
    """
    if not raw:
        return None, None
    raw = raw.strip()

    if m := _DATE_FULL.match(raw):
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "day"
    if m := _DATE_MONTH.match(raw):
        return date(int(m.group(1)), int(m.group(2)), 1), "month"
    if m := _DATE_YEAR.match(raw):
        return date(int(m.group(1)), 1, 1), "year"
    return None, None


@dataclass
class Location:
    facility: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip: Optional[str]
    country: Optional[str]
    location_status: Optional[str]


@dataclass
class Intervention:
    intervention_type: Optional[str]
    name: Optional[str]


@dataclass
class ParsedTrial:
    nct_id: str
    brief_title: Optional[str] = None
    official_title: Optional[str] = None
    brief_summary: Optional[str] = None

    study_type: Optional[str] = None
    overall_status: Optional[str] = None
    why_stopped: Optional[str] = None

    phase_display: Optional[str] = None
    phases: List[str] = field(default_factory=list)

    enrollment_count: Optional[int] = None
    enrollment_type: Optional[str] = None

    start_date: Optional[date] = None
    start_date_precision: Optional[str] = None
    primary_completion_date: Optional[date] = None
    completion_date: Optional[date] = None
    last_update_posted: Optional[date] = None

    lead_sponsor: Optional[str] = None
    sponsor_class: Optional[str] = None

    sex: Optional[str] = None
    healthy_volunteers: Optional[bool] = None
    min_age_years: Optional[float] = None
    max_age_years: Optional[float] = None
    min_age_raw: Optional[str] = None
    max_age_raw: Optional[str] = None

    eligibility_raw: Optional[str] = None
    criteria: List[Criterion] = field(default_factory=list)

    conditions: List[str] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)


def parse_study(study: Dict[str, Any]) -> Optional[ParsedTrial]:
    """Convert one raw API study dict into a :class:`ParsedTrial`.

    Returns ``None`` if the record has no NCT ID, which would make it
    uncitable and therefore useless to us.
    """
    protocol = study.get("protocolSection") or {}

    ident = protocol.get("identificationModule") or {}
    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    status_mod = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    elig = protocol.get("eligibilityModule") or {}
    sponsor = (protocol.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}
    desc = protocol.get("descriptionModule") or {}
    conditions_mod = protocol.get("conditionsModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    contacts = protocol.get("contactsLocationsModule") or {}

    phases = list(design.get("phases") or [])
    enrollment = design.get("enrollmentInfo") or {}

    start_date, start_precision = parse_ctg_date(
        (status_mod.get("startDateStruct") or {}).get("date")
    )
    primary_completion, _ = parse_ctg_date(
        (status_mod.get("primaryCompletionDateStruct") or {}).get("date")
    )
    completion, _ = parse_ctg_date(
        (status_mod.get("completionDateStruct") or {}).get("date")
    )
    last_update, _ = parse_ctg_date(
        (status_mod.get("lastUpdatePostDateStruct") or {}).get("date")
    )

    min_age_raw = elig.get("minimumAge")
    max_age_raw = elig.get("maximumAge")
    eligibility_raw = elig.get("eligibilityCriteria")

    return ParsedTrial(
        nct_id=nct_id,
        brief_title=ident.get("briefTitle"),
        official_title=ident.get("officialTitle"),
        brief_summary=desc.get("briefSummary"),
        study_type=design.get("studyType"),
        overall_status=status_mod.get("overallStatus"),
        why_stopped=status_mod.get("whyStopped"),
        phase_display="/".join(phases) if phases else None,
        phases=phases,
        enrollment_count=enrollment.get("count"),
        enrollment_type=enrollment.get("type"),
        start_date=start_date,
        start_date_precision=start_precision,
        primary_completion_date=primary_completion,
        completion_date=completion,
        last_update_posted=last_update,
        lead_sponsor=sponsor.get("name"),
        sponsor_class=sponsor.get("class"),
        sex=elig.get("sex"),
        healthy_volunteers=elig.get("healthyVolunteers"),
        min_age_years=parse_age_to_years(min_age_raw),
        max_age_years=parse_age_to_years(max_age_raw),
        min_age_raw=min_age_raw,
        max_age_raw=max_age_raw,
        eligibility_raw=eligibility_raw,
        criteria=split_eligibility(eligibility_raw),
        conditions=list(conditions_mod.get("conditions") or []),
        interventions=[
            Intervention(intervention_type=i.get("type"), name=i.get("name"))
            for i in (arms.get("interventions") or [])
        ],
        locations=[
            Location(
                facility=loc.get("facility"),
                city=loc.get("city"),
                state=loc.get("state"),
                zip=loc.get("zip"),
                country=loc.get("country"),
                location_status=loc.get("status"),
            )
            for loc in (contacts.get("locations") or [])
        ],
    )
