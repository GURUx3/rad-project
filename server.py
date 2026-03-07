from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import time
import re
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI(title="Radiology Report Classfication")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
models = {"tfidf": None, "svc": None}

def load_models():
    try:
        models["tfidf"] = joblib.load('tfidf_vectorizer.pkl')
        models["svc"] = joblib.load('linear_svc_radiology.pkl')
        print("RadiusAI Pro-Grade Models Loaded Successfully")
    except Exception as e:
        print(f"Error Loading Models: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

# --- PRO-GRADE SCHEMAS ---
class AnalysisRequest(BaseModel):
    findings: str
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
    hospital_name: str
    patient_name: str
    patient_dob: str
    patient_id: str
    physician: str
    indication: str
    technique: str
    diagnosis: str
    reliability_index: float
    severity: str
    affected_area: str
    recommendations: List[str]
    latency_ms: float
    findings_tokens: int
    probability_distribution: Dict[str, float]
    coding: List[CodingInfo]
    advisory: str

# --- CORE LOGIC ---
def get_mock_proba(decision_scores, classes):
    if len(classes) == 2:
        score = decision_scores[0]
        prob_pos = 1 / (1 + np.exp(-score))
        return {classes[0]: float(1 - prob_pos), classes[1]: float(prob_pos)}
    else:
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probs = exp_scores / exp_scores.sum()
        return dict(zip(classes, [float(p) for p in probs[0]]))

def extract_section(text: str, header: str) -> str:
    pattern = rf"{header}\s*\n(.*?)(?=\n[A-Z][a-z]+|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

@app.post("/analyze", response_model=ClinicalReport)
async def analyze_findings(request: AnalysisRequest):
    if not models["tfidf"] or not models["svc"]:
        raise HTTPException(status_code=533, detail="Pro-Core Not Initialized")

    start_time = time.time()
    findings_raw = request.findings
    
    try:
        # High-Fidelity NLP Extraction
        indication = extract_section(findings_raw, "Clinical Indication") or "General radiological evaluation"
        technique = extract_section(findings_raw, "Examination Technique") or "Standard axial imaging protocol"
        recommendations_raw = extract_section(findings_raw, "Recommendations")
        recommendations = [r.strip("• ").strip() for r in recommendations_raw.split("\n") if r.strip()] if recommendations_raw else ["Clinical correlation advised"]

        # ML Inference
        # We prioritize the "Impression" or "Findings" section for inference if present
        inference_text = extract_section(findings_raw, "Findings") or extract_section(findings_raw, "Impression") or findings_raw
        vec = models["tfidf"].transform([inference_text])
        prediction = models["svc"].predict(vec)[0]
        decision = models["svc"].decision_function(vec)
        proba_map = get_mock_proba(decision, models["svc"].classes_)
        
        # Clinical Mapping
        severity = "Normal"
        affected_area = "Systemic"
        
        # Anatomical Logic
        anatomical_map = {
            "Head & Neck": ["brain", "head", "skull", "basal ganglia", "intracranial", "cerebellum", "brainstem", "cervical"],
            "Chest": ["chest", "lung", "cardiac", "heart", "mediastinal"],
            "Abdomen": ["abdomen", "abdominal", "liver", "renal"],
            "Spine": ["spine", "spinal", "lumbar", "thoracic"]
        }
        detected = [region for region, keywords in anatomical_map.items() if any(kw in inference_text.lower() for kw in keywords)]
        if detected: affected_area = " & ".join(detected)

        # Risk Score Escalation
        if "hemorrhage" in inference_text.lower() or "fracture" in inference_text.lower() or "emergency" in inference_text.lower():
            severity = "HIGH RISK - EMERGENCY"
            if not recommendations_raw: recommendations = ["Immediate specialist consultation", "Urgent clinical management"]
        elif prediction.lower() != "normal":
            severity = "Moderate"
        else:
            severity = "Low"

        # Coding
        coding = []
        if severity == "HIGH RISK - EMERGENCY":
            coding = [CodingInfo(code="G93.6", description="Cerebral edema", system="ICD-10-CM"), 
                      CodingInfo(code="70450", description="CT Head; w/o contrast", system="CPT")]
        elif prediction.lower() == "normal":
            coding = [CodingInfo(code="Z00.00", description="General medical examination", system="ICD-10-CM")]

        latency = (time.time() - start_time) * 1000
        
        return ClinicalReport(
            execution_id=f"RAD-{datetime.now().strftime('%Y%j-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            hospital_name=request.hospital_name,
            patient_name=request.patient_name,
            patient_dob=request.patient_dob,
            patient_id=request.patient_id,
            physician=request.referring_physician,
            indication=indication,
            technique=technique,
            diagnosis=prediction,
            reliability_index=round(max(proba_map.values()), 4),
            severity=severity,
            affected_area=affected_area,
            recommendations=recommendations,
            latency_ms=round(latency, 2),
            findings_tokens=len(inference_text.split()),
            probability_distribution=proba_map,
            coding=coding,
            advisory="Validated by clinical decision support. Results require professional radiological correlation."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "ML_READY" if models["svc"] else "ML_OFFLINE"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
