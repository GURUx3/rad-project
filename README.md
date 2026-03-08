# Radius Radiology NLP Platform

Professional radiology text classification and clinical signal extraction using:
- `FastAPI` backend (`server.py`)
- Static dashboard frontend (`index.html`, `styles.css`, `app.js`)
- `LinearSVC + TF-IDF` model artifacts (`linear_svc_radiology.pkl`, `tfidf_vectorizer.pkl`)

## What Is Upgraded

This project now provides:
- Chat-style radiology workspace with dark premium UI and patient thread sidebar
- Automatic patient identity extraction from report text (name/DOB/MRN labels when present)
- Automatic thread grouping: multiple reports for same patient are merged in one thread
- Zero mandatory manual patient form fill; text + upload driven workflow
- Expanded NLP output (not only normal/abnormal):
  - Class probability distribution (percentage per class)
  - Affected region detection + regional involvement percentages
  - Suspected condition candidates
  - Severity and triage level
  - Key findings extraction
  - Negated findings extraction
  - Measurement extraction (`mm`, `cm`, `%`, etc.)
  - Modality and laterality extraction
  - Coding suggestions (ICD-10/CPT style)
  - Quality score + quality flags
  - Patient identity fields: `patient_display_name`, `patient_thread_key`, `patient_identity_source`
- Improved health endpoint with engine metadata and compatibility warnings
- Production-oriented API structure while preserving original endpoint/function names

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run backend API:

```bash
python server.py
```

3. Open `index.html` in browser (or serve with any static server), then run analysis.

## API

### POST `/analyze`

Request:

```json
{
  "findings": "Clinical Indication: ... Findings: ... Impression: ...",
  "patient_name": "Optional",
  "patient_dob": "Optional",
  "patient_id": "Optional",
  "referring_physician": "Optional",
  "hospital_name": "Optional"
}
```

Returns a `ClinicalReport` including legacy fields plus extended signals:
- `diagnosis`, `probability_distribution`, `reliability_index`
- `affected_area`, `affected_regions`, `regional_involvement`
- `severity`, `triage_level`, `suspected_conditions`
- `key_findings`, `measurements`, `negated_findings`
- `modality`, `laterality`
- `quality_score`, `quality_flags`, `section_presence`
- `recommendations`, `coding`, `advisory`

### GET `/health`

Returns API health, model readiness, load timestamp, and model compatibility warnings.

## Production Notes

Current model artifacts show a scikit-learn version-compatibility warning in some environments.  
`requirements.txt` pins `scikit-learn==1.6.1` to match saved model versions.

## Recommended Hardening Roadmap

1. Security and Access
- Replace open CORS (`*`) with allowlisted origins in `RADIUS_ALLOWED_ORIGINS`.
- Add authentication/authorization for API calls.
- Add request rate-limiting and audit logging.

2. Clinical Interoperability
- Map structured outputs to HL7 FHIR `DiagnosticReport`/`Observation`.
- Standardize coding output with validated ICD/SNOMED mapping workflows.

3. Model Reliability
- Replace pseudo-probability calibration with trained probability calibration pipeline.
- Add dataset-level validation, drift monitoring, and confidence threshold policies.

4. Operations
- Containerize API and run with process manager/reverse proxy.
- Add metrics, tracing, and alerting (`/health`, latency, failure rate, model confidence drift).
- Implement CI for lint/test/security scans.

5. Clinical Governance
- Add strict human-in-the-loop signoff flow before downstream clinical use.
- Document intended-use, contraindications, and validation boundaries.

## Reference Standards and Docs

- FastAPI deployment concepts: https://fastapi.tiangolo.com/deployment/concepts/
- FastAPI CORS middleware: https://fastapi.tiangolo.com/tutorial/cors/
- Uvicorn settings: https://www.uvicorn.org/settings/
- HL7 FHIR DiagnosticReport: https://hl7.org/fhir/diagnosticreport.html
- HL7 FHIR Observation: https://hl7.org/fhir/observation.html
- scikit-learn model persistence limitations: https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
- OWASP API Security Top 10: https://owasp.org/API-Security/

## Disclaimer

This system is a clinical decision-support aid and not a standalone diagnostic device.  
All outputs must be reviewed and validated by licensed clinicians/radiologists.
