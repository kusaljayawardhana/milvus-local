import os
import json
import sqlite3
import PyPDF2
import google.generativeai as genai
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from typing import Optional
import io

# ── 1. Configuration ──────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MILVUS_URI      = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "healthcare_candidates_v3"
SQLITE_DB_PATH  = "crm_database.db"

# ── 2. Section definitions ────────────────────────────────────────────────────
SECTIONS: list[dict] = [
    {
        "key":    "profile",
        "label":  "Profile Summary",
        "weight": 0.15,
        "hint":   (
            "Current job title, current employer, NHS band (if stated), a brief personal "
            "statement or career objective, and any headline facts the candidate leads with. "
            "Do NOT include qualifications, specialties, work history, or skills here."
        ),
    },
    {
        "key":    "qualifications",
        "label":  "Professional Qualifications",
        "weight": 0.20,
        "hint":   (
            "All formal academic degrees (BSc, MSc, PhD), post-registration awards, diplomas, "
            "professional registrations (NMC PIN, GMC number, HCPC registration, GPhC, etc.), "
            "mandatory training certificates (ILS, ALS, ATLS, safeguarding levels), and CPD courses. "
            "Include awarding body and year where stated."
        ),
    },
    {
        "key":    "specialties",
        "label":  "Specialties",
        "weight": 0.25,
        "hint":   (
            "Clinical or professional specialisms and sub-specialisms the candidate focuses on "
            "(e.g. critical care, theatres, oncology, CAMHS, community nursing, radiology, A&E, "
            "neonatal, mental health, palliative care). Include any named specialist roles, "
            "service areas, or patient populations they have specialist knowledge of. "
            "Do NOT include general skills or full job history here."
        ),
    },
    {
        "key":    "experience",
        "label":  "Working Experience",
        "weight": 0.25,
        "hint":   (
            "Complete employment history: each role's job title, employer/trust, NHS band or grade, "
            "start and end dates, total years of experience, key responsibilities, scope of practice, "
            "patient caseload, team size managed, and notable achievements or projects. "
            "Include locum, agency, bank, voluntary, and overseas experience."
        ),
    },
    {
        "key":    "skills",
        "label":  "Skills",
        "weight": 0.15,
        "hint":   (
            "Discrete technical and transferable skills: clinical procedures and interventions "
            "(IV cannulation, catheterisation, venepuncture, etc.), software/IT systems (EPR, SystmOne, "
            "EMIS, Lorenzo, etc.), leadership and management skills, research and audit skills, "
            "languages spoken, and any other concrete capabilities not captured in specialties. "
            "List as specific items rather than prose where possible."
        ),
    },
]

assert abs(sum(s["weight"] for s in SECTIONS) - 1.0) < 1e-6, "Section weights must sum to 1.0"
SECTION_KEYS = [s["key"] for s in SECTIONS]

# Within each section: inner dense/sparse weighting
DENSE_WEIGHT  = 0.7
SPARSE_WEIGHT = 0.3

# ── 3. UK Clinical Tag Taxonomy ───────────────────────────────────────────────
# Two-level taxonomy: profession (regulatory/role level) + specialty (clinical area).
# A candidate or JD can have multiple tags at both levels with confidence scores:
#   1.0 = primary / extensive / required
#   0.7 = secondary / significant / strongly desirable
#   0.3 = exposure / brief / desirable

PROFESSION_TAGS = [
    # Nursing — by register and advanced role
    "rn_adult",                     # RGN, adult branch
    "rn_mental_health",             # RMN
    "rn_child",                     # RSCN / children's nursing
    "rn_learning_disability",       # RNLD
    "nursing_associate",            # NMC Nursing Associate
    "advanced_nurse_practitioner",  # ANP / ACP
    "clinical_nurse_specialist",    # CNS
    "nurse_consultant",
    "nurse_prescriber",             # V100 / V300
    # Medical — by grade
    "foundation_doctor",            # FY1 / FY2
    "core_trainee",                 # CT1/CT2 or equivalent ST1/ST2
    "specialty_registrar",          # SpR / StR
    "consultant",
    "gp_principal",
    "gp_salaried",
    "gp_locum",
    "staff_grade",                  # SAS doctor — Staff Grade
    "associate_specialist",         # SAS doctor — Associate Specialist
    "specialty_doctor",             # SAS doctor — Specialty Doctor
    "clinical_fellow",
    # AHP — by profession
    "physiotherapist",
    "occupational_therapist",
    "radiographer_diagnostic",
    "radiographer_therapeutic",
    "speech_language_therapist",
    "dietitian",
    "podiatrist",
    "orthoptist",
    "prosthetist_orthotist",
    "art_music_drama_therapist",
    "paramedic",
    "emergency_care_practitioner",
    "operating_department_practitioner",  # ODP
    # Pharmacy
    "pharmacist_clinical",
    "pharmacist_community",
    "pharmacist_primary_care",
    "pharmacy_technician",
    # Midwifery
    "midwife",
    "maternity_support_worker",
    # Psychology / therapy
    "clinical_psychologist",
    "counselling_psychologist",
    "forensic_psychologist",
    "health_psychologist",
    "psychological_wellbeing_practitioner",  # PWP (IAPT Step 2)
    "high_intensity_therapist",              # IAPT Step 3 / CBT
    # Healthcare science
    "biomedical_scientist",
    "clinical_scientist",
    "cardiac_physiologist",
    "neurophysiologist",
    "audiologist",
    "medical_physicist",
    # Social / support
    "social_worker_adult",
    "social_worker_children",
    "approved_mental_health_professional",   # AMHP
    "healthcare_assistant",
    "senior_healthcare_assistant",
    "support_worker_mental_health",
    # Management / non-clinical leadership
    "ward_manager",
    "service_manager",
    "matron",
    "modern_matron",
    "nhs_manager",
    "practice_manager",
    "clinical_director",
    "head_of_nursing",
    # Specialist / cross-cutting roles
    "infection_prevention_control_nurse",
    "tissue_viability_nurse",
    "continence_advisor",
    "stoma_care_nurse",
    "IV_therapy_nurse",
    "clinical_educator",
    "practice_educator",
    "research_nurse",
    "trial_coordinator",
    "school_nurse",
    "health_visitor",
    "district_nurse",
    "community_psychiatric_nurse",           # CPN
    "practice_nurse",
    "occupational_health_nurse",
    # Veterinary — by role and registration
    "veterinary_surgeon",           # MRCVS
    "veterinary_nurse_registered",  # RVN
    "veterinary_nurse_student",     # SVN
    "veterinary_care_assistant",    # VCA / ANA
    "veterinary_specialist",        # Diplomate / Board Certified
    "veterinary_locum",
    "practice_manager_veterinary",
    "veterinary_receptionist",
]

SPECIALTY_TAGS = [
    # Critical care & acute
    "intensive_care_adult",           # ICU / ITU / AICU
    "intensive_care_neonatal",        # NICU
    "intensive_care_paediatric",      # PICU
    "high_dependency",                # HDU / level 2
    "emergency_department",           # A&E / ED / MIU / UTC
    "acute_medicine",
    "acute_surgical",
    "resuscitation",
    # Surgery & theatres
    "theatres_scrub",
    "theatres_anaesthetics",
    "theatres_recovery",              # PACU / recovery room
    "day_surgery",
    "trauma_orthopaedics",
    "spinal_surgery",
    "vascular_surgery",
    "cardiothoracic_surgery",
    "neurosurgery",
    "plastics_burns",
    "colorectal_surgery",
    "upper_gi_surgery",
    "urology_surgery",
    "maxillofacial",
    "ophthalmic_surgery",
    "transplant_surgery",
    "robotics_minimal_access",
    # Cardiology & respiratory
    "cardiology",
    "cardiac_catheterisation",        # cath lab / PCI
    "cardiothoracic",
    "electrophysiology",
    "heart_failure",
    "respiratory_medicine",
    "sleep_medicine",
    "pulmonary_hypertension",
    # Neurology & stroke
    "neurology",
    "stroke",
    "neurorehabilitation",
    "epilepsy",
    "multiple_sclerosis",
    "parkinsons",
    # Renal & urology
    "renal_medicine",
    "renal_dialysis",                 # haemodialysis / peritoneal
    "renal_transplant",
    "urology",
    # GI & hepatology
    "gastroenterology",
    "hepatology",
    "inflammatory_bowel_disease",
    "endoscopy",
    # Oncology & haematology
    "oncology_medical",
    "oncology_clinical",              # radiotherapy
    "haematology",
    "bone_marrow_transplant",
    "paediatric_oncology",
    "palliative_care",
    "lymphoedema",
    # Endocrine & diabetes
    "endocrinology",
    "diabetes",
    "obesity_bariatrics",
    # Rheumatology & musculoskeletal
    "rheumatology",
    "musculoskeletal",
    # Dermatology
    "dermatology",
    "wound_care",
    # Ophthalmology & ENT
    "ophthalmology",
    "ent",
    # Obstetrics, gynaecology & maternity
    "obstetrics",
    "gynaecology",
    "fetal_medicine",
    "colposcopy",
    "fertility",
    "neonatal",
    # Paediatrics
    "paediatrics_general",
    "paediatrics_community",
    "paediatric_diabetes",
    "paediatric_cardiology",
    "paediatric_neurology",
    "paediatric_surgery",
    # Community & primary care
    "district_nursing",
    "community_nursing",
    "primary_care_general_practice",
    "minor_injuries_urgent_care",
    "walk_in_centre",
    "111_ooh",
    "health_visiting",
    "school_nursing",
    "occupational_health",
    "prison_healthcare",
    "armed_forces_healthcare",
    # Mental health — specific services
    "inpatient_acute_mental_health",
    "psychiatric_intensive_care",     # PICU (psychiatric)
    "forensic_mental_health",
    "community_mental_health",        # CMHT
    "crisis_resolution",              # CRHTT / crisis team
    "early_intervention_psychosis",   # EIP
    "eating_disorders",
    "camhs",
    "perinatal_mental_health",
    "older_adult_mental_health",
    "dementia",
    "dual_diagnosis",
    "addictions_substance_misuse",
    "learning_disabilities_inpatient",
    "learning_disabilities_community",
    "iapt_talking_therapies",
    # Therapies & rehab
    "rehabilitation_inpatient",
    "rehabilitation_community",
    "neurological_rehabilitation",
    "cardiac_rehabilitation",
    "pulmonary_rehabilitation",
    "musculoskeletal_physiotherapy",
    "hand_therapy",
    "vestibular_rehabilitation",
    # Diagnostics & imaging
    "radiology_general",
    "radiology_interventional",
    "mri",
    "ct",
    "ultrasound",
    "nuclear_medicine",
    "pet_ct",
    "pathology_general",
    "histopathology",
    "microbiology",
    "haematology_laboratory",
    "biochemistry",
    "immunology",
    "cytology",
    "blood_transfusion",
    # Other specialist areas
    "infection_prevention_control",
    "tissue_viability",
    "pain_management",
    "pre_operative_assessment",
    "clinical_genetics",
    "immunology_allergy",
    "HIV_sexual_health",
    "elderly_care_geriatrics",
    "pharmacy_clinical_services",
    "pharmacy_medicines_management",
    "clinical_trials_research",
    "simulation_training",
    # Veterinary — specific settings & species
    "small_animal_practice",        # Cats, dogs, rabbits
    "equine_practice",              # Horses
    "large_animal_farm",            # Livestock, cattle, sheep
    "mixed_practice",               # Small and large animal mix
    "veterinary_referral_centre",   # Tertiary/Specialist care
    "veterinary_emergency_critical_care", # ECC / Out of hours
    "exotics_wildlife",             # Zoo, reptiles, avian
    "veterinary_surgery_orthopaedic",
    "veterinary_surgery_soft_tissue",
    "veterinary_internal_medicine",
    "veterinary_diagnostic_imaging", # Vet X-ray, Ultrasound, CT
    "veterinary_anaesthesia",
]

# ── 4. Tag scoring constants ───────────────────────────────────────────────────
# Tag overlap multiplier range:
#   Perfect tag match  → ×TAG_BOOST_MAX  (reward exact specialty alignment)
#   No tag overlap     → ×TAG_PENALTY_MIN (soft floor — candidate might still be useful)
#   Neutral / unknown  → ×1.0
TAG_BOOST_MAX   = 1.5
TAG_PENALTY_MIN = 0.2

# Profession vs specialty weight within tag scoring
TAG_PROF_WEIGHT = 0.35
TAG_SPEC_WEIGHT = 0.65

# ── 5. Specialty boost ────────────────────────────────────────────────────────
SPECIALTY_BOOST_FACTOR = 1.0
SPECIALTY_BM25_FLOOR   = 0.05

# ── Shared state ───────────────────────────────────────────────────────────────
ml_models  = {}
db_clients = {}


# ── 6. SQLite CRM helpers ─────────────────────────────────────────────────────
def init_crm_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            crm_id                TEXT UNIQUE NOT NULL,
            name                  TEXT NOT NULL,
            job_title             TEXT,
            nhs_band              TEXT,
            location              TEXT,
            registration          TEXT,
            profession_domain     TEXT,
            profession_tags_json  TEXT DEFAULT '{}',
            specialty_tags_json   TEXT DEFAULT '{}',
            tag_reasoning         TEXT,
            ai_summary            TEXT,
            sections_json         TEXT,
            milvus_ids_json       TEXT,
            created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Non-destructive migration for existing databases
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(candidates)")}
    new_cols = {
        "profession_tags_json": "TEXT DEFAULT '{}'",
        "specialty_tags_json":  "TEXT DEFAULT '{}'",
        "tag_reasoning":        "TEXT",
    }
    for col, definition in new_cols.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE candidates ADD COLUMN {col} {definition}")
            print(f"✅ DB migrated — added column: {col}")
    conn.commit()
    conn.close()


def insert_candidate_crm(
    crm_id: str,
    name: str,
    extracted: dict,
    ai_summary: str,
    sections: dict,
    milvus_ids: dict,
) -> int:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO candidates
        (crm_id, name, job_title, nhs_band, location, registration,
         profession_domain, profession_tags_json, specialty_tags_json,
         tag_reasoning, ai_summary, sections_json, milvus_ids_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        crm_id,
        name,
        extracted.get("job_title", ""),
        extracted.get("nhs_band", ""),
        extracted.get("location", ""),
        extracted.get("registration", ""),
        extracted.get("profession_domain", ""),
        json.dumps(extracted.get("profession_tags", {})),
        json.dumps(extracted.get("specialty_tags", {})),
        extracted.get("tag_reasoning", ""),
        ai_summary,
        json.dumps(sections),
        json.dumps(milvus_ids),
    ))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_all_candidates_crm() -> list:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM candidates ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["sections"]        = json.loads(d.get("sections_json") or "{}")
        d["milvus_ids"]      = json.loads(d.get("milvus_ids_json") or "{}")
        d["profession_tags"] = json.loads(d.get("profession_tags_json") or "{}")
        d["specialty_tags"]  = json.loads(d.get("specialty_tags_json") or "{}")
        result.append(d)
    return result


def delete_candidate_by_crm_id(crm_id: str) -> Optional[list[int]]:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT milvus_ids_json FROM candidates WHERE crm_id = ?", (crm_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    milvus_ids = list(json.loads(row[0]).values())
    c.execute("DELETE FROM candidates WHERE crm_id = ?", (crm_id,))
    conn.commit()
    conn.close()
    return milvus_ids


# ── 7. Sparse-vector helper ───────────────────────────────────────────────────
def sparse_matrix_to_dict(sparse_matrix) -> dict:
    cx = sparse_matrix.tocoo()
    return {int(j): float(v) for j, v in zip(cx.col, cx.data) if v > 0}


# ── 8. Tag scoring ────────────────────────────────────────────────────────────
def compute_tag_match_score(
    candidate_prof: dict,
    candidate_spec: dict,
    jd_prof: dict,
    jd_spec: dict,
) -> tuple[float, str]:
    """
    Compute a tag-overlap multiplier (TAG_PENALTY_MIN – TAG_BOOST_MAX) and
    explanation string.

    Weighted Jaccard-style overlap: both JD and candidate must carry a tag for
    it to contribute.  JD confidence acts as the importance weight; candidate
    confidence scales the contribution (primary match > exposure match).

    Returns (multiplier, explanation).
    """

    def weighted_overlap(cand: dict, jd: dict) -> float:
        if not jd:
            return 0.5          # JD didn't specify this axis → neutral contribution
        max_possible = sum(jd.values())
        if max_possible == 0:
            return 0.5
        score = sum(jd_conf * cand.get(tag, 0.0) for tag, jd_conf in jd.items())
        return score / max_possible

    prof_overlap = weighted_overlap(candidate_prof, jd_prof)
    spec_overlap = weighted_overlap(candidate_spec, jd_spec)

    combined = (TAG_PROF_WEIGHT * prof_overlap) + (TAG_SPEC_WEIGHT * spec_overlap)

    # Map 0–1 combined overlap onto [TAG_PENALTY_MIN, TAG_BOOST_MAX]
    # combined=0   → TAG_PENALTY_MIN
    # combined=0.5 → ~1.0 (neutral — JD didn't specify or partial match)
    # combined=1.0 → TAG_BOOST_MAX
    span       = TAG_BOOST_MAX - TAG_PENALTY_MIN
    multiplier = TAG_PENALTY_MIN + (combined * span)
    multiplier = round(min(max(multiplier, TAG_PENALTY_MIN), TAG_BOOST_MAX), 3)

    explanation = (
        f"prof_overlap={prof_overlap:.2f} "
        f"spec_overlap={spec_overlap:.2f} "
        f"combined={combined:.2f} "
        f"→ ×{multiplier}"
    )
    return multiplier, explanation


# ── 9. LLM helpers ───────────────────────────────────────────────────────────

_BLEED_RULES = """
STRICT SEPARATION RULES — read carefully before writing each section:
  • "profile"        → ONLY current title, current employer, band, personal statement / career objective.
                        No qualifications, no specialty names, no job history, no skill lists.
  • "qualifications" → ONLY formal degrees, diplomas, registrations (NMC/GMC/HCPC/GPhC/etc.),
                        mandatory training certs (ALS, ILS, safeguarding), CPD awards.
                        No job history, no specialty names, no skills.
  • "specialties"    → ONLY named clinical/professional specialisms and sub-specialisms
                        (e.g. critical care, CAMHS, oncology, theatres, radiology).
                        No procedures, no job history, no qualifications.
  • "experience"     → ONLY employment history: role titles, employers/trusts, bands/grades,
                        dates, responsibilities, scope, caseload, achievements.
                        No isolated skill bullets, no qualifications, no specialty labels.
  • "skills"         → ONLY discrete, nameable skills: clinical procedures (cannulation,
                        catheterisation, etc.), IT systems (SystmOne, EMIS, Lorenzo),
                        leadership, languages, research/audit skills.
                        No job history, no qualifications, no specialty names.

NEVER repeat the same fact in two sections.
If a section has genuinely no content, write a single sentence: "<Section>: Not stated."
"""


def _parse_llm_json(raw: str, context: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned invalid JSON ({context}): {e}\n\nRaw: {raw[:400]}",
        )


def classify_tags(text: str, text_type: str = "cv") -> tuple[dict, dict, str]:
    """
    Classify a CV summary or JD excerpt into profession + specialty tags.

    Returns:
        profession_tags  — {tag: confidence}
        specialty_tags   — {tag: confidence}
        reasoning        — one-sentence explanation of key decisions
    """
    source_label  = "candidate CV summary" if text_type == "cv" else "job description"
    jd_conf_note  = (
        "\nFor the JD: score 1.0 for required, 0.7 for strongly desirable, 0.3 for desirable."
        if text_type == "jd" else ""
    )

    prompt = f"""You are a specialist UK NHS recruiter with expert knowledge of NHS banding,
NMC/GMC/HCPC registers, and UK clinical specialties.

Given this {source_label}, identify ALL relevant profession tags and specialty tags from the
lists below.  Output ONLY valid JSON, no markdown, no preamble.

PROFESSION TAGS (pick all that genuinely apply):
{json.dumps(PROFESSION_TAGS)}

SPECIALTY TAGS (pick all that genuinely apply):
{json.dumps(SPECIALTY_TAGS)}

Confidence scoring:
  1.0 = PRIMARY — main profession/specialty, extensive documented experience
  0.7 = SECONDARY — significant experience, not the primary focus
  0.3 = EXPOSURE — brief stint, short course, mentioned but not a focus
{jd_conf_note}

Rules:
  • Only include tags with clear evidence in the text.
  • A candidate can have multiple profession tags (e.g. rn_adult + clinical_nurse_specialist).
  • A candidate can have multiple specialty tags reflecting their full career.
  • Maximum ~8 profession tags, ~10 specialty tags.
  • If a tag is not evidenced at all, omit it entirely.

OUTPUT FORMAT (valid JSON only):
{{
  "profession_tags": {{"tag_name": confidence_score}},
  "specialty_tags":  {{"tag_name": confidence_score}},
  "reasoning": "<one sentence: key classification decisions and any ambiguities>"
}}

TEXT TO CLASSIFY:
{text[:1500]}"""

    model  = genai.GenerativeModel(LLM_MODEL)
    resp   = model.generate_content(prompt, stream=False)
    parsed = _parse_llm_json(resp.text, "tag classification")

    prof_tags = {
        k: float(v)
        for k, v in parsed.get("profession_tags", {}).items()
        if k in PROFESSION_TAGS and isinstance(v, (int, float))
    }
    spec_tags = {
        k: float(v)
        for k, v in parsed.get("specialty_tags", {}).items()
        if k in SPECIALTY_TAGS and isinstance(v, (int, float))
    }
    reasoning = parsed.get("reasoning", "")

    return prof_tags, spec_tags, reasoning


def section_cv(name: str, cv_text: str) -> tuple[str, dict, dict]:
    """
    Send candidate name + raw CV text to the LLM.

    Returns:
        summary   — 200-300-word holistic paragraph
        sections  — {section_key: text} for all 5 sections
        extracted — metadata dict including profession_tags, specialty_tags
    """
    prompt = f"""You are a specialist UK healthcare recruiter with deep NHS knowledge.

Given only the candidate's name and raw CV text below, respond with a single JSON object
(no markdown fences, no preamble) with EXACTLY this structure:

{{
  "summary": "<200-300-word third-person paragraph covering the candidate's clinical value, current role, key specialisms, experience level, registrations, and standout achievements>",
  "extracted": {{
    "job_title":    "<most recent or current job title from the CV>",
    "nhs_band":     "<NHS band or grade if stated, else empty string>",
    "location":     "<city or region from the CV, else empty string>",
    "registration": "<primary professional registration e.g. NMC PIN, GMC number, HCPC — else empty string>"
  }},
  "sections": {{
    "profile":        "<Current job title · Current employer · NHS band if stated · Personal statement or career objective headline>",
    "qualifications": "<All degrees, diplomas, professional registrations (NMC/GMC/HCPC/GPhC/etc.), mandatory training certs, CPD courses — include awarding body and year where given>",
    "specialties":    "<Named clinical or professional specialisms and sub-specialisms only>",
    "experience":     "<Full employment history: role title, employer/trust, band/grade, dates, responsibilities, scope, caseload size, achievements — chronological, most recent first>",
    "skills":         "<Discrete nameable skills: clinical procedures, IT/EPR systems, leadership, languages, research/audit — listed as specific items>"
  }}
}}

{_BLEED_RULES}

CANDIDATE NAME: {name}

RAW CV TEXT:
{cv_text[:7000]}

Respond with valid JSON only. No markdown. No extra keys."""

    model    = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt, stream=False)
    parsed   = _parse_llm_json(response.text, "CV sectioning")

    summary   = parsed.get("summary", "").strip()
    sections  = parsed.get("sections", {})
    extracted = parsed.get("extracted", {})

    for s in SECTIONS:
        k = s["key"]
        if not sections.get(k, "").strip():
            sections[k] = f"{s['label']}: Not stated."

    # ── Tag classification ────────────────────────────────────────────────────
    prof_tags, spec_tags, reasoning = classify_tags(summary, text_type="cv")
    extracted["profession_tags"]  = prof_tags
    extracted["specialty_tags"]   = spec_tags
    extracted["tag_reasoning"]    = reasoning
    # Legacy display field — highest-confidence profession tag
    extracted["profession_domain"] = (
        max(prof_tags, key=prof_tags.get) if prof_tags else "unclassified"
    )

    return summary, sections, extracted


def section_jd(job_description: str) -> tuple[dict, str, bool, dict, dict]:
    """
    Section a JD and classify its tags.

    Returns:
        sections            — {section_key: text}
        jd_display_domain   — highest-confidence profession tag (for display)
        is_specialty_specific — bool
        jd_prof_tags        — {tag: confidence}
        jd_spec_tags        — {tag: confidence}
    """
    prompt = f"""You are a specialist UK healthcare recruiter with deep NHS knowledge.

Given the job description below, extract its requirements into the same 5 sections
used for candidate CVs. Respond with a single JSON object (no markdown, no preamble):

{{
  "sections": {{
    "profile":        "<The type of candidate being sought: target job title, employer/trust context, band/grade, and any role-level summary>",
    "qualifications": "<Required or desirable qualifications, registrations, certifications, mandatory training>",
    "specialties":    "<Required or desirable clinical/professional specialisms and sub-specialisms>",
    "experience":     "<Required or desirable experience: years, types of roles, scope, settings, caseloads>",
    "skills":         "<Required or desirable discrete skills: clinical procedures, IT systems, leadership, languages, etc.>"
  }}
}}

{_BLEED_RULES}

If the JD does not mention a section at all, write exactly: "Not specified."
Be dense and factual. Preserve the JD's wording where useful.

JOB DESCRIPTION:
{job_description[:5000]}

Respond with valid JSON only. No markdown. No extra keys."""

    model    = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt, stream=False)
    parsed   = _parse_llm_json(response.text, "JD sectioning")

    sections = parsed.get("sections", {})
    for s in SECTIONS:
        if not sections.get(s["key"], "").strip():
            sections[s["key"]] = "Not specified."

    # ── Tag classification ────────────────────────────────────────────────────
    jd_prof_tags, jd_spec_tags, _ = classify_tags(job_description[:1500], text_type="jd")
    jd_display_domain = (
        max(jd_prof_tags, key=jd_prof_tags.get) if jd_prof_tags else "unclassified"
    )

    specialty_text       = sections.get("specialties", "")
    is_specialty_specific = (
        specialty_text.strip() not in ("Not specified.", "")
        and len(specialty_text.split()) > 5
    )

    return sections, jd_display_domain, is_specialty_specific, jd_prof_tags, jd_spec_tags


# ── 10. Server startup ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🏥 Booting Healthcare CV Search Engine…")

    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini client ready")

    print("Loading embedding model (all-mpnet-base-v2)…")
    ml_models["embedder"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    dimension = ml_models["embedder"].get_sentence_embedding_dimension()
    print(f"✅ Embedder ready (dim={dimension})")

    analyzer = build_default_analyzer(language="en")
    ml_models["bm25"] = {key: BM25EmbeddingFunction(analyzer) for key in SECTION_KEYS}

    init_crm_db()
    print("✅ SQLite CRM ready")

    existing = get_all_candidates_crm()
    for s in SECTIONS:
        key    = s["key"]
        corpus = [
            c["sections"].get(key, "")
            for c in existing
            if c.get("sections", {}).get(key, "").strip()
        ]
        if corpus:
            ml_models["bm25"][key].fit(corpus)
            print(f"✅ BM25[{key}] fitted on {len(corpus)} docs")
        else:
            ml_models["bm25"][key].fit([f"placeholder {key} healthcare text"])
            print(f"⚠️  BM25[{key}] fitted on placeholder (no candidates yet)")

    print("Connecting to Milvus…")
    client = MilvusClient(uri=MILVUS_URI)
    db_clients["milvus"] = client

    if not client.has_collection(COLLECTION_NAME):
        from pymilvus import DataType, CollectionSchema, FieldSchema

        fields = [
            FieldSchema(name="id",               dtype=DataType.INT64,              is_primary=True, auto_id=True),
            FieldSchema(name="candidate_crm_id", dtype=DataType.VARCHAR,            max_length=64),
            FieldSchema(name="section_key",      dtype=DataType.VARCHAR,            max_length=64),
            FieldSchema(name="section_text",     dtype=DataType.VARCHAR,            max_length=4096),
            FieldSchema(name="dense_vector",     dtype=DataType.FLOAT_VECTOR,       dim=dimension),
            FieldSchema(name="sparse_vector",    dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="Healthcare candidates — sectioned hybrid search")

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        print(f"✅ Milvus collection '{COLLECTION_NAME}' created")
    else:
        try:
            client.load_collection(collection_name=COLLECTION_NAME)
            state = client.get_load_state(collection_name=COLLECTION_NAME)
            print(f"✅ Milvus collection '{COLLECTION_NAME}' loaded — state: {state}")
        except Exception as e:
            print(f"❌ Failed to load Milvus collection: {e}")

    print("🚀 System fully operational!")
    yield

    print("Shutting down…")
    ml_models.clear()
    db_clients.clear()


app = FastAPI(title="Healthcare CV Semantic Search", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 11. PDF extraction ────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")


# ── 12. Pydantic models ───────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    job_description: str
    top_k: int = 5


class SectionScore(BaseModel):
    key:    str
    label:  str
    weight: float
    score:  float


class CandidateResult(BaseModel):
    crm_id:            str
    name:              str
    job_title:         str
    nhs_band:          str
    location:          str
    registration:      str
    profession_domain: str
    profession_tags:   dict
    specialty_tags:    dict
    tag_multiplier:    float
    tag_explanation:   str
    match_percentage:  float
    section_scores:    list[SectionScore]
    ai_summary:        str
    sections:          dict = {}


class SearchResponse(BaseModel):
    jd_sections:    dict
    jd_domain:      str
    jd_prof_tags:   dict
    jd_spec_tags:   dict
    results:        list[CandidateResult]


# ── 13. Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "healthy",
        "service": "Healthcare CV Search",
        "version": "4.0.0 (tag-based matching)",
    }


@app.get("/candidates", summary="List all candidates in CRM")
async def list_candidates():
    candidates = get_all_candidates_crm()
    return {"count": len(candidates), "candidates": candidates}


@app.get("/tags", summary="List all available profession and specialty tags")
async def list_tags():
    return {
        "profession_tags": sorted(PROFESSION_TAGS),
        "specialty_tags":  sorted(SPECIALTY_TAGS),
        "total_profession": len(PROFESSION_TAGS),
        "total_specialty":  len(SPECIALTY_TAGS),
    }


@app.get("/domains", summary="See every candidate's tag classification at a glance")
async def list_domains():
    candidates = get_all_candidates_crm()
    return {
        "by_top_profession_tag": {
            tag: [
                {"name": c["name"], "crm_id": c["crm_id"]}
                for c in candidates
                if c.get("profession_tags") and max(c["profession_tags"], key=c["profession_tags"].get) == tag
            ]
            for tag in PROFESSION_TAGS
            if any(
                c.get("profession_tags") and max(c["profession_tags"], key=c["profession_tags"].get) == tag
                for c in candidates
            )
        },
        "detail": [
            {
                "name":             c["name"],
                "crm_id":           c["crm_id"],
                "job_title":        c["job_title"],
                "profession_tags":  c.get("profession_tags", {}),
                "specialty_tags":   c.get("specialty_tags", {}),
                "tag_reasoning":    c.get("tag_reasoning", ""),
            }
            for c in candidates
        ],
    }


@app.post(
    "/ingest",
    summary="Ingest a candidate CV — name + PDF; everything else is extracted automatically",
)
async def ingest_candidate(
    name:    str        = Form(...),
    cv_file: UploadFile = File(...),
):
    if not cv_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    cv_bytes = await cv_file.read()
    cv_text  = extract_pdf_text(cv_bytes)
    if not cv_text:
        raise HTTPException(status_code=422, detail="Could not extract any text from the PDF.")

    print(f"⏳ Processing CV for {name}…")
    ai_summary, cv_sections, extracted = section_cv(name, cv_text)
    print(
        f"✅ CV processed — {extracted.get('job_title', 'unknown')} | "
        f"prof_tags={list(extracted.get('profession_tags', {}).keys())} | "
        f"spec_tags={list(extracted.get('specialty_tags', {}).keys())}"
    )

    crm_id   = f"CRM-{np.random.randint(100000, 999999)}"
    existing = get_all_candidates_crm()
    milvus_ids: dict = {}

    for s in SECTIONS:
        key  = s["key"]
        text = cv_sections[key]

        bm25   = ml_models["bm25"][key]
        corpus = [c["sections"].get(key, "") for c in existing if c.get("sections", {}).get(key, "")]
        corpus.append(text)
        bm25.fit(corpus)

        dense_vec     = ml_models["embedder"].encode([text], normalize_embeddings=True)[0].tolist()
        sparse_matrix = bm25.encode_documents([text])
        sparse_vec    = sparse_matrix_to_dict(sparse_matrix)
        if not sparse_vec:
            sparse_vec = {0: 1e-6}

        insert_result = db_clients["milvus"].insert(
            collection_name=COLLECTION_NAME,
            data=[{
                "candidate_crm_id": crm_id,
                "section_key":      key,
                "section_text":     text[:4096],
                "dense_vector":     dense_vec,
                "sparse_vector":    sparse_vec,
            }],
        )
        milvus_ids[key] = insert_result["ids"][0]
        print(f"  ✅ [{key}] → Milvus {milvus_ids[key]}")

    crm_row_id = insert_candidate_crm(crm_id, name, extracted, ai_summary, cv_sections, milvus_ids)

    return {
        "status":          "success",
        "crm_id":          crm_id,
        "crm_row_id":      crm_row_id,
        "name":            name,
        "extracted":       extracted,
        "ai_summary":      ai_summary,
        "sections":        cv_sections,
        "profession_tags": extracted.get("profession_tags", {}),
        "specialty_tags":  extracted.get("specialty_tags", {}),
        "tag_reasoning":   extracted.get("tag_reasoning", ""),
    }


@app.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest):
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    # ── Section + tag the JD ──────────────────────────────────────────────────
    print("⏳ Sectioning & tagging JD…")
    jd_sections, jd_display_domain, is_specialty_specific, jd_prof_tags, jd_spec_tags = (
        section_jd(request.job_description)
    )
    print(
        f"✅ JD processed | display_domain={jd_display_domain} | "
        f"specialty_specific={is_specialty_specific} | "
        f"prof_tags={list(jd_prof_tags.keys())} | "
        f"spec_tags={list(jd_spec_tags.keys())}"
    )

    # ── Effective section weights ─────────────────────────────────────────────
    if is_specialty_specific:
        raw_weights = {
            s["key"]: s["weight"] * SPECIALTY_BOOST_FACTOR if s["key"] == "specialties" else s["weight"]
            for s in SECTIONS
        }
        total_w          = sum(raw_weights.values())
        effective_weights = {k: v / total_w for k, v in raw_weights.items()}
        print(f"  ⚡ Specialty boost → specialties weight: {effective_weights['specialties']:.3f}")
    else:
        effective_weights = {s["key"]: s["weight"] for s in SECTIONS}

    fetch_limit = max(request.top_k * 5, 20)   # generous fetch so tag scoring has candidates to re-rank

    # ── Per-section hybrid search ─────────────────────────────────────────────
    candidate_section_scores: dict[str, dict[str, float]] = {}

    for s in SECTIONS:
        key  = s["key"]
        text = jd_sections[key]

        if text.strip() in ("Not specified.", ""):
            print(f"  ⏭️  Section '{key}' not in JD — skipped")
            continue

        bm25 = ml_models["bm25"][key]

        dense_q        = ml_models["embedder"].encode([text], normalize_embeddings=True)[0].tolist()
        sparse_q       = sparse_matrix_to_dict(bm25.encode_queries([text]))
        section_filter = f'section_key == "{key}"'

        try:
            dense_req = AnnSearchRequest(
                data=[dense_q],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"ef": 100}},
                limit=fetch_limit,
                expr=section_filter,
            )
            reqs = [dense_req]

            if sparse_q:
                reqs.append(AnnSearchRequest(
                    data=[sparse_q],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                    limit=fetch_limit,
                    expr=section_filter,
                ))

            hybrid_hits = db_clients["milvus"].hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=reqs,
                ranker=RRFRanker(k=60),
                limit=fetch_limit,
                output_fields=["candidate_crm_id"],
                filter=section_filter,
            )[0]

            # Dense scores for raw cosine similarity
            dense_raw = db_clients["milvus"].search(
                collection_name=COLLECTION_NAME,
                data=[dense_q],
                anns_field="dense_vector",
                search_params={"metric_type": "IP", "params": {"ef": 100}},
                filter=section_filter,
                limit=fetch_limit,
                output_fields=["candidate_crm_id"],
            )[0]
            dense_score_map = {h["id"]: float(h["distance"]) for h in dense_raw}

            # Sparse scores (normalised BM25)
            sparse_score_map: dict[int, float] = {}
            if sparse_q:
                sparse_raw = db_clients["milvus"].search(
                    collection_name=COLLECTION_NAME,
                    data=[sparse_q],
                    anns_field="sparse_vector",
                    search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                    filter=section_filter,
                    limit=fetch_limit,
                    output_fields=["candidate_crm_id"],
                )[0]
                raw_vals   = [float(h["distance"]) for h in sparse_raw]
                sparse_max = max(raw_vals) if raw_vals else 1.0
                sparse_max = sparse_max if sparse_max > 0 else 1.0
                sparse_score_map = {
                    h["id"]: float(h["distance"]) / sparse_max for h in sparse_raw
                }

            for hit in hybrid_hits:
                mid     = hit["id"]
                crm_cid = hit["entity"].get("candidate_crm_id", f"MV-{mid}")

                d_score = dense_score_map.get(mid, 0.0)
                s_score = sparse_score_map.get(mid, 0.0)
                section_score = (DENSE_WEIGHT * d_score) + (SPARSE_WEIGHT * s_score)

                if key == "specialties" and s_score < SPECIALTY_BM25_FLOOR:
                    section_score *= 0.5
                    print(f"    ⚠️  BM25 floor hit for {crm_cid} on specialties (sparse={s_score:.3f})")

                section_score = min(section_score, 1.0)

                if crm_cid not in candidate_section_scores:
                    candidate_section_scores[crm_cid] = {}
                candidate_section_scores[crm_cid][key] = max(
                    candidate_section_scores[crm_cid].get(key, 0.0), section_score
                )

        except Exception as e:
            print(f"  ⚠️  Section '{key}' search error: {e}")
            continue

    # ── Load CRM rows once ────────────────────────────────────────────────────
    all_crm_rows = get_all_candidates_crm()
    all_crm      = {c["crm_id"]: c for c in all_crm_rows}

    active_sections     = [
        s for s in SECTIONS
        if jd_sections.get(s["key"], "").strip() not in ("Not specified.", "")
    ]
    active_weight_total = sum(effective_weights[s["key"]] for s in active_sections)

    # ── Score + tag-multiply every candidate ─────────────────────────────────
    scored_candidates = []
    for crm_cid, sec_scores in candidate_section_scores.items():
        # Drop stale Milvus hits with no CRM record
        if crm_cid not in all_crm:
            print(f"  ⚠️  Skipping stale vector: {crm_cid}")
            continue

        weighted_sum = sum(
            sec_scores.get(s["key"], 0.0) * effective_weights[s["key"]]
            for s in active_sections
        )
        overall = (weighted_sum / active_weight_total) if active_weight_total > 0 else 0.0
        overall = min(overall, 1.0)

        # ── Tag-based match multiplier ────────────────────────────────────────
        cand_data = all_crm[crm_cid]
        cand_prof = cand_data.get("profession_tags", {})
        cand_spec  = cand_data.get("specialty_tags", {})

        multiplier, tag_explanation = compute_tag_match_score(
            cand_prof, cand_spec, jd_prof_tags, jd_spec_tags
        )
        overall *= multiplier
        overall  = min(overall, 1.0)   # cap at 100%

        print(
            f"  🏷️  {crm_cid} | raw={weighted_sum/active_weight_total:.3f} "
            f"| tag_mult={multiplier} | final={overall:.3f} | {tag_explanation}"
        )

        scored_candidates.append((crm_cid, round(overall * 100, 1), sec_scores, multiplier, tag_explanation))

    # Sort descending, take top_k
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    scored_candidates = scored_candidates[: request.top_k]

    # ── Build response ────────────────────────────────────────────────────────
    results: list[CandidateResult] = []
    for crm_cid, overall_pct, sec_scores, multiplier, tag_explanation in scored_candidates:
        crm = all_crm.get(crm_cid)
        if not crm:
            continue

        section_score_list = [
            SectionScore(
                key=s["key"],
                label=s["label"],
                weight=effective_weights[s["key"]],
                score=round(sec_scores.get(s["key"], 0.0) * 100, 1),
            )
            for s in SECTIONS
        ]

        results.append(CandidateResult(
            crm_id=crm["crm_id"],
            name=crm["name"],
            job_title=crm.get("job_title", ""),
            nhs_band=crm.get("nhs_band", ""),
            location=crm.get("location", ""),
            registration=crm.get("registration", ""),
            profession_domain=crm.get("profession_domain", ""),
            profession_tags=crm.get("profession_tags", {}),
            specialty_tags=crm.get("specialty_tags", {}),
            tag_multiplier=multiplier,
            tag_explanation=tag_explanation,
            match_percentage=overall_pct,
            section_scores=section_score_list,
            ai_summary=crm.get("ai_summary", ""),
            sections=crm.get("sections", {}),
        ))

    return SearchResponse(
        jd_sections=jd_sections,
        jd_domain=jd_display_domain,
        jd_prof_tags=jd_prof_tags,
        jd_spec_tags=jd_spec_tags,
        results=results,
    )


# ── 14. Delete ────────────────────────────────────────────────────────────────
@app.delete("/candidate/{crm_id}")
async def delete_candidate(crm_id: str):
    milvus_ids = delete_candidate_by_crm_id(crm_id)
    if milvus_ids is None:
        raise HTTPException(status_code=404, detail=f"Candidate '{crm_id}' not found.")

    errors = []
    for mid in milvus_ids:
        try:
            db_clients["milvus"].delete(collection_name=COLLECTION_NAME, ids=[mid])
        except Exception as e:
            errors.append(str(e))

    if errors:
        return {"status": "partial", "message": f"Removed from CRM. Milvus errors: {errors}"}
    return {
        "status":  "success",
        "message": f"Candidate '{crm_id}' deleted ({len(milvus_ids)} vectors removed).",
    }


# ── 15. Utilities ─────────────────────────────────────────────────────────────
@app.post("/cleanup-orphaned-vectors", summary="Delete Milvus vectors with no matching CRM record")
async def cleanup_orphaned_vectors():
    all_crm_ids = {c["crm_id"] for c in get_all_candidates_crm()}
    rows = db_clients["milvus"].query(
        collection_name=COLLECTION_NAME,
        filter='candidate_crm_id != ""',
        output_fields=["id", "candidate_crm_id"],
        limit=10000,
    )
    orphan_ids = [r["id"] for r in rows if r["candidate_crm_id"] not in all_crm_ids]
    if orphan_ids:
        db_clients["milvus"].delete(collection_name=COLLECTION_NAME, ids=orphan_ids)
        print(f"🧹 Cleaned {len(orphan_ids)} orphaned vectors")
    return {"deleted": len(orphan_ids), "orphan_ids": orphan_ids}


@app.patch("/candidate/{crm_id}/retag", summary="Re-run tag classification for a candidate using their stored summary")
async def retag_candidate(crm_id: str):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("SELECT ai_summary, name FROM candidates WHERE crm_id = ?", (crm_id,))
    row  = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Candidate {crm_id} not found")

    prof_tags, spec_tags, reasoning = classify_tags(row["ai_summary"], text_type="cv")
    top_prof = max(prof_tags, key=prof_tags.get) if prof_tags else "unclassified"

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute(
        """UPDATE candidates
           SET profession_tags_json=?, specialty_tags_json=?, tag_reasoning=?, profession_domain=?
           WHERE crm_id=?""",
        (json.dumps(prof_tags), json.dumps(spec_tags), reasoning, top_prof, crm_id),
    )
    conn.commit()
    conn.close()

    return {
        "crm_id":          crm_id,
        "name":            row["name"],
        "profession_tags": prof_tags,
        "specialty_tags":  spec_tags,
        "tag_reasoning":   reasoning,
        "profession_domain": top_prof,
    }


@app.post("/debug-match", summary="Raw Milvus hits + tag scores for a JD — for diagnostics")
async def debug_match(request: SearchRequest):
    jd_sections, jd_display_domain, is_specialty_specific, jd_prof_tags, jd_spec_tags = (
        section_jd(request.job_description)
    )

    report: dict = {
        "jd_display_domain":    jd_display_domain,
        "is_specialty_specific": is_specialty_specific,
        "jd_prof_tags":         jd_prof_tags,
        "jd_spec_tags":         jd_spec_tags,
        "jd_sections":          jd_sections,
        "milvus_stats":         {},
        "per_section_hits":     {},
        "candidate_tag_scores": {},
    }

    # Milvus vector counts per section
    for s in SECTIONS:
        key = s["key"]
        try:
            rows = db_clients["milvus"].query(
                collection_name=COLLECTION_NAME,
                filter=f'section_key == "{key}"',
                output_fields=["candidate_crm_id"],
                limit=1000,
            )
            report["milvus_stats"][key] = {
                "vector_count":      len(rows),
                "candidate_crm_ids": list({r["candidate_crm_id"] for r in rows}),
            }
        except Exception as e:
            report["milvus_stats"][key] = {"error": str(e)}

    # Raw dense hits per section
    for s in SECTIONS:
        key  = s["key"]
        text = jd_sections[key]
        if text.strip() in ("Not specified.", ""):
            report["per_section_hits"][key] = "SKIPPED"
            continue

        dense_q = ml_models["embedder"].encode([text], normalize_embeddings=True)[0].tolist()
        try:
            hits = db_clients["milvus"].search(
                collection_name=COLLECTION_NAME,
                data=[dense_q],
                anns_field="dense_vector",
                search_params={"metric_type": "IP", "params": {"ef": 100}},
                filter=f'section_key == "{key}"',
                limit=10,
                output_fields=["candidate_crm_id", "section_text"],
            )[0]
            report["per_section_hits"][key] = [
                {
                    "milvus_id":    h["id"],
                    "crm_id":       h["entity"]["candidate_crm_id"],
                    "score":        round(h["distance"], 4),
                    "text_snippet": h["entity"]["section_text"][:150],
                }
                for h in hits
            ]
        except Exception as e:
            report["per_section_hits"][key] = {"error": str(e)}

    # Tag overlap scores for every candidate
    for cand in get_all_candidates_crm():
        mult, expl = compute_tag_match_score(
            cand.get("profession_tags", {}),
            cand.get("specialty_tags", {}),
            jd_prof_tags,
            jd_spec_tags,
        )
        report["candidate_tag_scores"][cand["crm_id"]] = {
            "name":            cand["name"],
            "profession_tags": cand.get("profession_tags", {}),
            "specialty_tags":  cand.get("specialty_tags", {}),
            "tag_multiplier":  mult,
            "explanation":     expl,
        }

    return report