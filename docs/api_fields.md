# ClinicalTrials.gov API v2 — verified field paths

Everything here was checked against live responses from
`https://clinicaltrials.gov/api/v2/studies` on **2026-07-31**, not recalled from
memory. If the parser starts producing nulls, re-run the profiling and compare.

## Response envelope

```
{ "studies": [ {...} ], "nextPageToken": "...", "totalCount": 19038 }
```

- `pageSize` maxes out at **1000**. Larger values are silently capped, not rejected.
- `countTotal=true` is required for `totalCount` to appear.
- Paginate by passing the previous response's `nextPageToken` as `pageToken`.
- A `fields=` projection works and cuts payload size, but keeps the same nesting.

Each study has three top-level keys: `protocolSection`, `derivedSection`,
`hasResults`. We only read `protocolSection`.

## Field paths used by the parser

| Column | Path under `protocolSection` |
|---|---|
| `nct_id` | `identificationModule.nctId` |
| `brief_title` | `identificationModule.briefTitle` |
| `official_title` | `identificationModule.officialTitle` |
| `brief_summary` | `descriptionModule.briefSummary` |
| `overall_status` | `statusModule.overallStatus` |
| `why_stopped` | `statusModule.whyStopped` |
| `start_date` | `statusModule.startDateStruct.date` |
| `primary_completion_date` | `statusModule.primaryCompletionDateStruct.date` |
| `completion_date` | `statusModule.completionDateStruct.date` |
| `last_update_posted` | `statusModule.lastUpdatePostDateStruct.date` |
| `study_type` | `designModule.studyType` |
| phases | `designModule.phases` — **list** |
| `enrollment_count` | `designModule.enrollmentInfo.count` |
| `enrollment_type` | `designModule.enrollmentInfo.type` |
| `lead_sponsor` | `sponsorCollaboratorsModule.leadSponsor.name` |
| `sponsor_class` | `sponsorCollaboratorsModule.leadSponsor.class` |
| conditions | `conditionsModule.conditions` — list |
| interventions | `armsInterventionsModule.interventions[].{type,name}` |
| `eligibility_raw` | `eligibilityModule.eligibilityCriteria` |
| `sex` | `eligibilityModule.sex` |
| `healthy_volunteers` | `eligibilityModule.healthyVolunteers` |
| `min_age_raw` | `eligibilityModule.minimumAge` |
| `max_age_raw` | `eligibilityModule.maximumAge` |
| locations | `contactsLocationsModule.locations[].{facility,status,city,state,zip,country}` |

## Things that are easy to model wrongly

**`phases` is a list.** A trial can be `["PHASE1","PHASE2"]`. Values seen:
`EARLY_PHASE1`, `PHASE1`, `PHASE2`, `PHASE3`, `PHASE4`, `NA`. About 40% of
interventional trials are `NA` — mostly device and behavioural studies. Our
corpus filter excludes them.

**`locations` is a list, and each site has its own `status`.** A trial can be
`RECRUITING` overall while a given site is closed. Both readings are supported:
`v_trials.states` (any site in that state) and `v_trials.recruiting_states`
(sites actively recruiting).

**Dates are variable-precision.** `2024-03-15`, `2024-03`, and occasionally
`2024`. `parse_ctg_date` returns the date plus a precision flag so an imputed
day is never mistaken for a reported one.

**Eligibility text is escaped markdown.** A real record reads:

```
* Patients must be \> 365 days and \< 18 years ... ABL-class \[COG\]
```

Unescape before chunking or the criteria embed badly.

## Corpus filter

```
AREA[StudyType]INTERVENTIONAL
  AND AREA[StartDate]RANGE[2018-01-01,MAX]
  AND (AREA[Phase]PHASE1 OR AREA[Phase]PHASE2 OR AREA[Phase]PHASE3 OR AREA[Phase]PHASE4)
```

Counts measured 2026-07-31:

| Area | Matching trials |
|---|---|
| oncology (`cancer OR neoplasm OR tumor OR carcinoma`) | 30,521 |
| diabetes (`diabetes`) | 2,696 |
| cardiovascular (`cardiovascular OR heart failure OR hypertension OR coronary`) | 7,921 |
| **total (pre-dedupe)** | **41,138** |

## Profiling notes (200 real oncology records)

Used to design the eligibility splitter:

- Bullet style: `* ` 82%, numbered 15%, none 2%.
- Headers: both present 91%, inclusion only 3.5%, neither 5%.
- `eligibilityCriteria` absent: 0 of 200.
- `minimumAge` present far more often than `maximumAge`; both are frequently absent.
