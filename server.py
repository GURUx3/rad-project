from __future__ import annotations

import logging
import os
import re
import time
import warnings
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = Warning


logging.basicConfig(
    level=os.getenv("RADIUS_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("radius_nlp_api")

APP_VERSION = "2.1.0"
MODEL_FILES = {
    "tfidf": "tfidf_vectorizer.pkl",
    "svc": "linear_svc_radiology.pkl",
}

allowed_origins_raw = os.getenv("RADIUS_ALLOWED_ORIGINS", "*").strip()
allow_origins = ["*"] if allowed_origins_raw == "*" else [
    origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()
]

app = FastAPI(
    title="Radiology Report Classfication",
    version=APP_VERSION,
    description="Radiology NLP inference and structured clinical signal extraction API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global models
models = {"tfidf": None, "svc": None}
engine_state = {
    "model_loaded": False,
    "loaded_at": None,
    "warnings": [],
}

SEVERITY_RANK = {"Low": 1, "Moderate": 2, "High": 3, "HIGH RISK - EMERGENCY": 4}
TRIAGE_RANK = {"ROUTINE": 1, "EXPEDITED": 2, "URGENT": 3, "STAT": 4}

ANATOMICAL_MAP: Dict[str, List[str]] = {
    "Head & Neck": [
        "brain", "head", "skull", "intracranial", "cerebral", "basal ganglia",
        "cerebellum", "brainstem", "cervical", "sinus", "orbit", "neck",
    ],
    "Chest": [
        "chest", "lung", "pulmonary", "pleural", "cardiac", "heart", "mediastinum",
        "hilar", "thorax",
    ],
    "Abdomen": [
        "abdomen", "abdominal", "liver", "hepatic", "renal", "kidney", "spleen",
        "pancreas", "gallbladder", "bowel", "colon", "stomach",
    ],
    "Pelvis": [
        "pelvis", "pelvic", "bladder", "uterus", "ovary", "prostate", "rectum",
    ],
    "Spine": [
        "spine", "spinal", "lumbar", "thoracic", "vertebral", "disc", "sacrum",
    ],
    "Musculoskeletal": [
        "fracture", "humerus", "femur", "tibia", "fibula", "joint", "shoulder",
        "knee", "hip", "ankle", "wrist", "elbow", "osseous",
    ],
    "Vascular": [
        "artery", "venous", "vein", "aneurysm", "aorta", "embol", "vascular",
    ],
}

MODALITY_MAP: Dict[str, List[str]] = {
    "CT": ["ct", "computed tomography"],
    "MRI": ["mri", "magnetic resonance"],
    "X-Ray": ["xray", "x-ray", "radiograph", "plain film"],
    "Ultrasound": ["ultrasound", "sonography", "doppler", "usg"],
    "PET": ["pet", "positron emission"],
}

CONDITION_RULES = [
    {
        "name": "Intracranial Hemorrhage",
        "keywords": ["hemorrhage", "haemorrhage", "intracranial bleed", "intraparenchymal bleed", "sah"],
        "severity": "HIGH RISK - EMERGENCY",
        "triage": "STAT",
        "recommendations": [
            "Activate emergency neurology or neurosurgery pathway.",
            "Urgent repeat imaging if clinical status changes.",
        ],
        "coding": [
            ("I61.9", "Nontraumatic intracerebral hemorrhage, unspecified", "ICD-10-CM"),
            ("70450", "CT Head/Brain without contrast", "CPT"),
        ],
    },
    {
        "name": "Acute Infarct or Ischemia",
        "keywords": ["acute infarct", "ischemia", "ischemic change", "restricted diffusion", "stroke"],
        "severity": "High",
        "triage": "URGENT",
        "recommendations": [
            "Correlate with stroke timeline and NIHSS if applicable.",
            "Consider vascular imaging and specialist review.",
        ],
        "coding": [
            ("I63.9", "Cerebral infarction, unspecified", "ICD-10-CM"),
        ],
    },
    {
        "name": "Mass Effect or Midline Shift",
        "keywords": ["mass effect", "midline shift", "herniation"],
        "severity": "HIGH RISK - EMERGENCY",
        "triage": "STAT",
        "recommendations": [
            "Immediate specialist escalation due to potential raised intracranial pressure.",
        ],
        "coding": [
            ("G93.5", "Compression of brain", "ICD-10-CM"),
        ],
    },
    {
        "name": "Pneumonia or Consolidation",
        "keywords": ["pneumonia", "consolidation", "airspace opacity", "infiltrate"],
        "severity": "Moderate",
        "triage": "EXPEDITED",
        "recommendations": [
            "Clinical and laboratory correlation for infection is advised.",
            "Follow-up chest imaging based on treatment response.",
        ],
        "coding": [
            ("J18.9", "Pneumonia, unspecified organism", "ICD-10-CM"),
        ],
    },
    {
        "name": "Pleural Effusion",
        "keywords": ["pleural effusion"],
        "severity": "Moderate",
        "triage": "EXPEDITED",
        "recommendations": [
            "Evaluate need for interval follow-up imaging.",
        ],
        "coding": [
            ("J90", "Pleural effusion, not elsewhere classified", "ICD-10-CM"),
        ],
    },
    {
        "name": "Fracture",
        "keywords": ["fracture", "fractured"],
        "severity": "High",
        "triage": "URGENT",
        "recommendations": [
            "Orthopedic or trauma pathway correlation recommended.",
        ],
        "coding": [
            ("S02.91XA", "Unspecified fracture of skull, initial encounter", "ICD-10-CM"),
        ],
    },
    {
        "name": "Pulmonary Embolism",
        "keywords": ["pulmonary embolism", "pe", "embolus"],
        "severity": "HIGH RISK - EMERGENCY",
        "triage": "STAT",
        "recommendations": [
            "Urgent cardiopulmonary risk stratification recommended.",
        ],
        "coding": [
            ("I26.99", "Other pulmonary embolism without acute cor pulmonale", "ICD-10-CM"),
        ],
    },
]

SECTION_ALIASES = {
    "clinical indication": "indication",
    "indication": "indication",
    "history": "indication",
    "examination technique": "technique",
    "technique": "technique",
    "method": "technique",
    "findings": "findings",
    "impression": "impression",
    "conclusion": "impression",
    "recommendation": "recommendations",
    "recommendations": "recommendations",
}

EXPECTED_SECTIONS = ["indication", "technique", "findings", "impression"]

PLACEHOLDER_VALUES = {
    "",
    "n/a",
    "na",
    "unknown",
    "ref-unknown",
    "none",
    "--",
    "not provided",
}


def _safe_lower(value: str) -> str:
    return value.lower() if value else ""


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().lower()
    return normalized in PLACEHOLDER_VALUES


def _normalize_text(text: str) -> str:
    compact = re.sub(r"\r\n?", "\n", text or "")
    compact = re.sub(r"[ \t]+", " ", compact)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


def _clean_extracted_value(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n:;-")
    return cleaned


def _extract_by_label(text: str, labels: List[str], max_len: int = 80) -> str:
    for label in labels:
        pattern = rf"(?im)^\s*{label}\s*[:\-]\s*(.+?)\s*$"
        match = re.search(pattern, text)
        if match:
            value = _clean_extracted_value(match.group(1))
            if value:
                return value[:max_len]
    return ""


def _extract_patient_context(text: str) -> Dict[str, str]:
    patient_name = _extract_by_label(
        text,
        [
            r"patient\s*name",
            r"name",
        ],
        max_len=100,
    )
    patient_id = _extract_by_label(
        text,
        [
            r"patient\s*id",
            r"mrn",
            r"uhid",
            r"accession\s*no",
        ],
        max_len=64,
    )
    patient_dob = _extract_by_label(
        text,
        [
            r"dob",
            r"date\s*of\s*birth",
            r"birth\s*date",
        ],
        max_len=32,
    )
    physician = _extract_by_label(
        text,
        [
            r"referring\s*physician",
            r"physician",
            r"doctor",
        ],
        max_len=100,
    )
    hospital_name = _extract_by_label(
        text,
        [
            r"hospital",
            r"facility",
            r"institution",
            r"center",
            r"centre",
        ],
        max_len=120,
    )

    if not patient_dob:
        match_dob = re.search(
            r"\b(?:dob|date of birth)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})",
            text,
            flags=re.IGNORECASE,
        )
        if match_dob:
            patient_dob = _clean_extracted_value(match_dob.group(1))

    return {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "patient_dob": patient_dob,
        "physician": physician,
        "hospital_name": hospital_name,
    }


def _merge_identity_value(request_value: Optional[str], extracted_value: str, fallback: str) -> str:
    if not _is_placeholder(request_value):
        return str(request_value).strip()
    if not _is_placeholder(extracted_value):
        return extracted_value.strip()
    return fallback


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def _build_patient_thread_key(patient_id: str, patient_name: str, patient_dob: str, findings_raw: str) -> str:
    if not _is_placeholder(patient_id):
        return f"id-{_slugify(patient_id)}"
    if not _is_placeholder(patient_name) and not _is_placeholder(patient_dob):
        return f"name-{_slugify(patient_name)}-dob-{_slugify(patient_dob)}"
    if not _is_placeholder(patient_name):
        return f"name-{_slugify(patient_name)}"

    fingerprint = hashlib.sha1(findings_raw[:300].encode("utf-8")).hexdigest()[:12]
    return f"unknown-{fingerprint}"


def _derive_identity_source(
    request_patient_name: Optional[str],
    request_patient_dob: Optional[str],
    request_patient_id: Optional[str],
    extracted_context: Dict[str, str],
) -> str:
    request_has_identity = (
        not _is_placeholder(request_patient_name)
        or not _is_placeholder(request_patient_dob)
        or not _is_placeholder(request_patient_id)
    )
    extracted_has_identity = (
        not _is_placeholder(extracted_context.get("patient_name"))
        or not _is_placeholder(extracted_context.get("patient_dob"))
        or not _is_placeholder(extracted_context.get("patient_id"))
    )

    if request_has_identity and extracted_has_identity:
        return "mixed"
    if request_has_identity:
        return "provided"
    if extracted_has_identity:
        return "extracted"
    return "unknown"


def _parse_structured_sections(text: str) -> Dict[str, str]:
    section_pattern = re.compile(
        r"(?im)^(clinical indication|indication|history|examination technique|technique|method|findings|impression|conclusion|recommendations?)\s*[:\-]?\s*$"
    )
    matches = list(section_pattern.finditer(text))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        raw_header = match.group(1).strip().lower()
        canonical_header = SECTION_ALIASES.get(raw_header, raw_header)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[start:end].strip(" \n:-")
        if value:
            sections[canonical_header] = value
    return sections


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    hits = 0
    for keyword in keywords:
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        hits += len(re.findall(pattern, text))
    return hits


def _is_negated_keyword(text: str, keyword: str) -> bool:
    pattern = rf"\b(?:no|without|absence of|negative for)\s+[^.{{,}};]{{0,35}}\b{re.escape(keyword.lower())}\b"
    return bool(re.search(pattern, text))


def _extract_measurements(text: str) -> List[str]:
    measurements = set()
    patterns = [
        r"\b\d+(?:\.\d+)?\s?(?:x|X)\s?\d+(?:\.\d+)?(?:\s?(?:x|X)\s?\d+(?:\.\d+)?)?\s?(?:mm|cm)\b",
        r"\b\d+(?:\.\d+)?\s?(?:mm|cm|ml|mL)\b",
        r"\b\d+(?:\.\d+)?\s?%\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            measurements.add(match)
    return sorted(measurements)


def _extract_negated_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for match in re.finditer(
        r"\b(?:no|without|absence of|negative for)\s+([a-z][a-z0-9\s\-]{2,40})",
        text,
        flags=re.IGNORECASE,
    ):
        phrase = re.split(r"[,.;\n]", match.group(1))[0].strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:8]


def _extract_key_sentences(text: str, trigger_terms: List[str], limit: int = 4) -> List[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return []

    selected: List[str] = []
    lower_terms = [term.lower() for term in trigger_terms if term]
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(term in s_lower for term in lower_terms):
            selected.append(sentence)
        if len(selected) >= limit:
            break

    if not selected:
        selected = sentences[: min(limit, len(sentences))]
    return selected


def _derive_recommendations(
    recommendation_text: str,
    condition_matches: List[Dict],
    triage_level: str,
    prediction: str,
) -> List[str]:
    results: List[str] = []

    if recommendation_text:
        raw_parts = re.split(r"[\n;]+", recommendation_text)
        for part in raw_parts:
            cleaned = part.strip(" -*\t")
            if cleaned and cleaned not in results:
                results.append(cleaned)

    for condition in condition_matches:
        for rec in condition.get("recommendations", []):
            if rec not in results:
                results.append(rec)

    if not results:
        if prediction.lower() == "normal":
            results.append("No acute radiological red flags detected; routine clinical correlation advised.")
        else:
            results.append("Correlate with patient history and previous imaging before management decisions.")

    if triage_level == "STAT":
        urgent_line = "Immediate clinical communication is recommended due to potential life-threatening features."
        if urgent_line not in results:
            results.insert(0, urgent_line)
    elif triage_level == "URGENT":
        urgent_line = "Prioritized clinical review is recommended."
        if urgent_line not in results:
            results.insert(0, urgent_line)

    return results[:8]


def _derive_coding(condition_matches: List[Dict], prediction: str) -> List["CodingInfo"]:
    code_rows = []
    seen = set()
    for condition in condition_matches:
        for code, description, system in condition.get("coding", []):
            key = (code, system)
            if key in seen:
                continue
            seen.add(key)
            code_rows.append(CodingInfo(code=code, description=description, system=system))

    if not code_rows and prediction.lower() == "normal":
        code_rows.append(
            CodingInfo(
                code="Z00.00",
                description="General adult medical examination without abnormal findings",
                system="ICD-10-CM",
            )
        )
    elif not code_rows and prediction.lower() != "normal":
        code_rows.append(
            CodingInfo(
                code="R93.89",
                description="Abnormal findings on diagnostic imaging of other specified body structures",
                system="ICD-10-CM",
            )
        )
    return code_rows[:8]


def _pick_highest_rank(items: List[str], rank_map: Dict[str, int], default_value: str) -> str:
    if not items:
        return default_value
    return sorted(items, key=lambda x: rank_map.get(x, 0), reverse=True)[0]


def load_models():
    try:
        model_warnings: List[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            models["tfidf"] = joblib.load(MODEL_FILES["tfidf"])
            models["svc"] = joblib.load(MODEL_FILES["svc"])
            for warning_item in caught:
                model_warnings.append(str(warning_item.message))

        engine_state["warnings"] = model_warnings
        engine_state["model_loaded"] = bool(models["tfidf"] and models["svc"])
        engine_state["loaded_at"] = datetime.now(timezone.utc).isoformat()

        if engine_state["model_loaded"]:
            logger.info("RadiusAI models loaded successfully")
            if model_warnings:
                logger.warning("Model load warnings detected: %s", model_warnings)
    except Exception as exc:
        engine_state["model_loaded"] = False
        engine_state["warnings"] = [str(exc)]
        logger.exception("Error loading models: %s", exc)


@app.on_event("startup")
async def startup_event():
    load_models()


# --- PRO-GRADE SCHEMAS ---
class AnalysisRequest(BaseModel):
    findings: str = Field(..., min_length=10, max_length=50000)
    patient_name: Optional[str] = "N/A"
    patient_dob: Optional[str] = "N/A"
    patient_id: Optional[str] = "REF-UNKNOWN"
    referring_physician: Optional[str] = "N/A"
    hospital_name: Optional[str] = "CityCare Medical Center"


class CodingInfo(BaseModel):
    code: str
    description: str
    system: str


class ClinicalReport(BaseModel):
    execution_id: str
    timestamp: str
    api_version: str
    hospital_name: str
    patient_name: str
    patient_dob: str
    patient_id: str
    physician: str
    patient_display_name: str
    patient_thread_key: str
    patient_identity_source: str
    indication: str
    technique: str
    diagnosis: str
    reliability_index: float
    severity: str
    triage_level: str
    affected_area: str
    affected_regions: List[str]
    regional_involvement: Dict[str, float]
    key_findings: List[str]
    suspected_conditions: List[str]
    measurements: List[str]
    laterality: List[str]
    modality: List[str]
    negated_findings: List[str]
    quality_score: float
    quality_flags: List[str]
    section_presence: Dict[str, bool]
    recommendations: List[str]
    latency_ms: float
    findings_tokens: int
    probability_distribution: Dict[str, float]
    coding: List[CodingInfo]
    advisory: str


# --- CORE LOGIC ---
def get_mock_proba(decision_scores, classes):
    class_labels = [str(c) for c in classes]
    scores = np.asarray(decision_scores)

    if len(class_labels) == 2:
        raw_score = float(scores[0]) if scores.ndim == 1 else float(scores[0][0])
        prob_pos = 1.0 / (1.0 + np.exp(-raw_score))
        prob_neg = 1.0 - prob_pos
        return {class_labels[0]: float(prob_neg), class_labels[1]: float(prob_pos)}

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return dict(zip(class_labels, [float(p) for p in probs[0]]))


def extract_section(text: str, header: str) -> str:
    normalized = _normalize_text(text)
    sections = _parse_structured_sections(normalized)

    canonical = SECTION_ALIASES.get(header.strip().lower(), header.strip().lower())
    if canonical in sections:
        return sections[canonical]

    fallback_pattern = rf"(?is){re.escape(header)}\s*[:\-]\s*(.*?)(?=\n[A-Za-z][A-Za-z ]{{2,30}}\s*[:\-]|$)"
    match = re.search(fallback_pattern, normalized)
    return match.group(1).strip() if match else ""


@app.post("/analyze", response_model=ClinicalReport)
async def analyze_findings(request: AnalysisRequest):
    if not models["tfidf"] or not models["svc"]:
        raise HTTPException(status_code=503, detail="Pro-Core Not Initialized")

    start_time = time.time()
    findings_raw = _normalize_text(request.findings)
    if not findings_raw:
        raise HTTPException(status_code=422, detail="Findings text is required")

    try:
        extracted_context = _extract_patient_context(findings_raw)

        resolved_patient_name = _merge_identity_value(
            request.patient_name,
            extracted_context.get("patient_name", ""),
            "N/A",
        )
        resolved_patient_dob = _merge_identity_value(
            request.patient_dob,
            extracted_context.get("patient_dob", ""),
            "N/A",
        )
        resolved_patient_id = _merge_identity_value(
            request.patient_id,
            extracted_context.get("patient_id", ""),
            "REF-UNKNOWN",
        )
        resolved_physician = _merge_identity_value(
            request.referring_physician,
            extracted_context.get("physician", ""),
            "N/A",
        )
        resolved_hospital_name = _merge_identity_value(
            request.hospital_name,
            extracted_context.get("hospital_name", ""),
            "CityCare Medical Center",
        )
        patient_display_name = (
            resolved_patient_name
            if not _is_placeholder(resolved_patient_name)
            else (
                resolved_patient_id
                if not _is_placeholder(resolved_patient_id)
                else "Unidentified Patient"
            )
        )
        patient_thread_key = _build_patient_thread_key(
            resolved_patient_id,
            resolved_patient_name,
            resolved_patient_dob,
            findings_raw,
        )
        patient_identity_source = _derive_identity_source(
            request.patient_name,
            request.patient_dob,
            request.patient_id,
            extracted_context,
        )

        sections = _parse_structured_sections(findings_raw)

        indication = sections.get("indication") or extract_section(findings_raw, "Clinical Indication") or "General radiological evaluation"
        technique = sections.get("technique") or extract_section(findings_raw, "Examination Technique") or "Standard radiology protocol"
        recommendations_raw = sections.get("recommendations") or extract_section(findings_raw, "Recommendations")

        # We prioritize section-specific evidence for inference if present.
        inference_text = sections.get("findings") or sections.get("impression") or findings_raw
        inference_lower = _safe_lower(inference_text)
        report_lower = _safe_lower(findings_raw)

        # ML Inference
        vec = models["tfidf"].transform([inference_text])
        prediction = str(models["svc"].predict(vec)[0])
        decision = models["svc"].decision_function(vec)
        proba_map = get_mock_proba(decision, models["svc"].classes_)

        # Anatomical extraction
        region_hits: Dict[str, int] = {}
        for region, keywords in ANATOMICAL_MAP.items():
            hit_count = _count_keyword_hits(inference_lower, keywords)
            if hit_count:
                region_hits[region] = hit_count

        sorted_regions = sorted(region_hits.keys(), key=lambda x: region_hits[x], reverse=True)
        affected_regions = sorted_regions[:4]
        affected_area = " & ".join(affected_regions) if affected_regions else "Systemic / Unspecified"

        regional_involvement: Dict[str, float] = {}
        total_region_hits = sum(region_hits.values())
        if total_region_hits > 0:
            for region in affected_regions:
                regional_involvement[region] = round((region_hits[region] / total_region_hits) * 100, 2)

        # Modality extraction
        modality: List[str] = []
        for mod_name, keywords in MODALITY_MAP.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", report_lower) for keyword in keywords):
                modality.append(mod_name)

        # Laterality extraction
        laterality = []
        for label in ["left", "right", "bilateral", "midline", "diffuse"]:
            if re.search(rf"\b{label}\b", inference_lower):
                laterality.append(label.title())

        # Condition extraction and risk scoring
        condition_matches = []
        severity_candidates = ["Low"]
        triage_candidates = ["ROUTINE"]
        for rule in CONDITION_RULES:
            hit = False
            for keyword in rule["keywords"]:
                if re.search(rf"\b{re.escape(keyword)}\b", inference_lower) and not _is_negated_keyword(inference_lower, keyword):
                    hit = True
                    break
            if hit:
                condition_matches.append(rule)
                severity_candidates.append(rule["severity"])
                triage_candidates.append(rule["triage"])

        if prediction.lower() != "normal":
            severity_candidates.append("Moderate")
            triage_candidates.append("EXPEDITED")

        severity = _pick_highest_rank(severity_candidates, SEVERITY_RANK, "Low")
        triage_level = _pick_highest_rank(triage_candidates, TRIAGE_RANK, "ROUTINE")

        suspected_conditions = [rule["name"] for rule in condition_matches]
        if not suspected_conditions and prediction.lower() != "normal":
            suspected_conditions = ["Non-specific abnormal radiology pattern"]

        measurements = _extract_measurements(inference_text)
        negated_findings = _extract_negated_phrases(inference_text)

        key_terms = []
        for region in affected_regions:
            key_terms.append(region.lower())
        for condition in suspected_conditions:
            key_terms.append(condition.lower())
        key_findings = _extract_key_sentences(inference_text, key_terms, limit=4)

        recommendations = _derive_recommendations(
            recommendation_text=recommendations_raw,
            condition_matches=condition_matches,
            triage_level=triage_level,
            prediction=prediction,
        )

        coding = _derive_coding(condition_matches, prediction)

        section_presence = {section: (section in sections) for section in EXPECTED_SECTIONS}
        section_score = sum(1 for available in section_presence.values() if available)
        quality_score = round((section_score / len(EXPECTED_SECTIONS)) * 100, 2)

        quality_flags = []
        token_count = len(inference_text.split())
        if token_count < 20:
            quality_flags.append("Very short report text may lower extraction reliability.")
        if section_score < 2:
            quality_flags.append("Report has limited structured headers (e.g., Findings/Impression).")
        if not affected_regions and prediction.lower() != "normal":
            quality_flags.append("No clear anatomical region detected in the provided text.")
        if engine_state["warnings"]:
            quality_flags.append("Model loaded with version compatibility warnings; validate output carefully.")

        latency = (time.time() - start_time) * 1000

        return ClinicalReport(
            execution_id=f"RAD-{datetime.now().strftime('%Y%j-%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            api_version=APP_VERSION,
            hospital_name=resolved_hospital_name,
            patient_name=resolved_patient_name,
            patient_dob=resolved_patient_dob,
            patient_id=resolved_patient_id,
            physician=resolved_physician,
            patient_display_name=patient_display_name,
            patient_thread_key=patient_thread_key,
            patient_identity_source=patient_identity_source,
            indication=indication,
            technique=technique,
            diagnosis=prediction,
            reliability_index=round(max(proba_map.values()), 4),
            severity=severity,
            triage_level=triage_level,
            affected_area=affected_area,
            affected_regions=affected_regions,
            regional_involvement=regional_involvement,
            key_findings=key_findings,
            suspected_conditions=suspected_conditions,
            measurements=measurements,
            laterality=laterality,
            modality=modality,
            negated_findings=negated_findings,
            quality_score=quality_score,
            quality_flags=quality_flags,
            section_presence=section_presence,
            recommendations=recommendations,
            latency_ms=round(latency, 2),
            findings_tokens=token_count,
            probability_distribution=proba_map,
            coding=coding,
            advisory=(
                "Automated clinical decision-support output. "
                "Use only with radiologist validation and full patient context."
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if engine_state["model_loaded"] else "degraded",
        "engine": "ML_READY" if models["svc"] else "ML_OFFLINE",
        "api_version": APP_VERSION,
        "loaded_at": engine_state["loaded_at"],
        "warnings": engine_state["warnings"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
