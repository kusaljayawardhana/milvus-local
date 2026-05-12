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
COLLECTION_NAME = "healthcare_candidates_v3"   # 5-section schema: profile/qualifications/specialties/experience/skills
SQLITE_DB_PATH  = "crm_database.db"

# ── Section definitions ───────────────────────────────────────────────────────
# 5 sections matching the canonical CV chunking schema.
# Each section gets its own embedding — dense + sparse — in Milvus.
# Weights must sum to 1.0. Tune them to shift scoring emphasis.
#
#   profile        — who the candidate is right now (title, employer, summary)
#   qualifications — formal credentials and professional registrations
#   specialties    — clinical/domain specialisms and areas of focus
#   experience     — full work history with scope, years, and seniority
#   skills         — discrete technical and soft skills
#
SECTIONS: list[dict] = [
    {
        "key":    "profile",
        "label":  "Profile Summary",
        "weight": 0.1,
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
        "weight": 0.3,
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
DENSE_WEIGHT  = 1.0
SPARSE_WEIGHT = 0.0

ml_models  = {}
db_clients = {}


# ── 2. SQLite CRM Database Helpers ────────────────────────────────────────────
def init_crm_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            crm_id          TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            job_title       TEXT,
            nhs_band        TEXT,
            location        TEXT,
            registration    TEXT,
            ai_summary      TEXT,
            sections_json   TEXT,
            milvus_ids_json TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_candidate_crm(
    crm_id: str,
    name: str,
    extracted: dict,
    ai_summary: str,
    sections: dict[str, str],
    milvus_ids: dict[str, int],
) -> int:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO candidates
        (crm_id, name, job_title, nhs_band, location, registration,
         ai_summary, sections_json, milvus_ids_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        crm_id,
        name,
        extracted.get("job_title", ""),
        extracted.get("nhs_band", ""),
        extracted.get("location", ""),
        extracted.get("registration", ""),
        ai_summary,
        json.dumps(sections),
        json.dumps(milvus_ids),
    ))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_candidate_by_milvus_id(milvus_id: int) -> Optional[dict]:
    """Find a candidate whose milvus_ids_json contains this id."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # We store milvus_ids as JSON {"clinical_skills": 123, ...}; search all rows
    c.execute("SELECT * FROM candidates")
    rows = c.fetchall()
    conn.close()
    for row in rows:
        d = dict(row)
        mids = json.loads(d.get("milvus_ids_json") or "{}")
        if milvus_id in mids.values():
            d["sections"]   = json.loads(d.get("sections_json") or "{}")
            d["milvus_ids"] = mids
            return d
    return None


def get_candidate_by_any_section_id(milvus_id: int) -> Optional[tuple[dict, str]]:
    """Return (candidate_dict, section_key) for the section that owns this Milvus id."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM candidates")
    rows = c.fetchall()
    conn.close()
    for row in rows:
        d = dict(row)
        mids: dict = json.loads(d.get("milvus_ids_json") or "{}")
        for sec_key, mid in mids.items():
            if mid == milvus_id:
                d["sections"]   = json.loads(d.get("sections_json") or "{}")
                d["milvus_ids"] = mids
                return d, sec_key
    return None, None


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
        d["sections"]   = json.loads(d.get("sections_json") or "{}")
        d["milvus_ids"] = json.loads(d.get("milvus_ids_json") or "{}")
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


# ── 3. Sparse-vector helper ───────────────────────────────────────────────────
def sparse_matrix_to_dict(sparse_matrix) -> dict:
    cx = sparse_matrix.tocoo()
    return {int(j): float(v) for j, v in zip(cx.col, cx.data) if v > 0}


# ── 4. LLM helpers ───────────────────────────────────────────────────────────

# Shared anti-bleed rules injected into every LLM prompt.
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
    """Strip markdown fences and parse JSON, raising HTTP 502 on failure."""
    clean = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
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
            detail=f"LLM returned invalid JSON ({context}): {e}\n\nRaw snippet: {raw[:400]}",
        )


def section_cv(name: str, cv_text: str) -> tuple[str, dict[str, str], dict[str, str]]:
    """
    Send only the candidate name + raw CV text to the LLM.
    The LLM extracts everything else.

    Returns:
        summary   — 200-300-word holistic paragraph (stored as ai_summary)
        sections  — {section_key: text} for all 5 sections
        extracted — {job_title, nhs_band, location, registration} pulled from CV
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

    # Guarantee all 5 section keys are present and non-empty
    for s in SECTIONS:
        k = s["key"]
        if not sections.get(k, "").strip():
            sections[k] = f"{s['label']}: Not stated."

    return summary, sections, extracted


def section_jd(job_description: str) -> dict[str, str]:
    """
    Send the JD to the LLM and extract the same 5 sections so we can do
    section-to-section vector matching at search time.

    Returns {section_key: requirement_text}.
    Sections the JD doesn't mention are returned as "Not specified." and are
    excluded from scoring at search time (candidates are not penalised).
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

    return sections


# ── 5. Server Startup ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🏥 Booting Healthcare CV Search Engine (5-section: profile / qualifications / specialties / experience / skills)...")

    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini client ready")

    print("Loading embedding model (all-mpnet-base-v2)...")
    ml_models["embedder"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    dimension = ml_models["embedder"].get_sentence_embedding_dimension()
    print(f"✅ Embedder ready (dim={dimension})")

    # One BM25 model per section so IDF is learned from section-specific text
    analyzer = build_default_analyzer(language="en")
    ml_models["bm25"] = {
        key: BM25EmbeddingFunction(analyzer) for key in SECTION_KEYS
    }

    init_crm_db()
    print("✅ SQLite CRM database ready")

    # Refit BM25 per section from existing data
    existing = get_all_candidates_crm()
    for s in SECTIONS:
        key = s["key"]
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
            print(f"⚠️  BM25[{key}] fitted on placeholder")

    print("Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    db_clients["milvus"] = client

    if not client.has_collection(COLLECTION_NAME):
        from pymilvus import DataType, CollectionSchema, FieldSchema

        fields = [
            FieldSchema(name="id",            dtype=DataType.INT64,              is_primary=True, auto_id=True),
            FieldSchema(name="candidate_crm_id", dtype=DataType.VARCHAR,         max_length=64),
            FieldSchema(name="section_key",   dtype=DataType.VARCHAR,            max_length=64),
            FieldSchema(name="section_text",  dtype=DataType.VARCHAR,            max_length=4096),
            FieldSchema(name="dense_vector",  dtype=DataType.FLOAT_VECTOR,       dim=dimension),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
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
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Milvus collection '{COLLECTION_NAME}' loaded")

    print("🚀 System fully operational!")
    yield

    print("Shutting down...")
    ml_models.clear()
    db_clients.clear()


app = FastAPI(title="Healthcare CV Semantic Search", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 6. Helper: PDF Extraction ─────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")


# ── 7. Pydantic Models ────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    job_description: str
    top_k: int = 5


class SectionScore(BaseModel):
    key:    str
    label:  str
    weight: float
    score:  float   # 0–100


class CandidateResult(BaseModel):
    crm_id:           str
    name:             str
    job_title:        str
    nhs_band:         str
    location:         str
    registration:     str
    match_percentage: float
    section_scores:   list[SectionScore]
    ai_summary:       str
    sections:         dict[str, str] = {}


class SearchResponse(BaseModel):
    jd_sections: dict[str, str]
    results:     list[CandidateResult]


# ── 8. Ingest ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Healthcare CV Search", "version": "3.0.0 (5-section: profile/qualifications/specialties/experience/skills)"}


@app.get("/candidates", summary="List all candidates in CRM")
async def list_candidates():
    candidates = get_all_candidates_crm()
    return {"count": len(candidates), "candidates": candidates}


@app.post("/ingest", summary="Ingest a candidate CV — just name + PDF, everything else is extracted automatically")
async def ingest_candidate(
    name:    str        = Form(..., description="Candidate full name"),
    cv_file: UploadFile = File(...,  description="Candidate CV as a PDF"),
):
    # ── Extract PDF text ──────────────────────────────────────────────────────
    if not cv_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    cv_bytes = await cv_file.read()
    cv_text  = extract_pdf_text(cv_bytes)
    if not cv_text:
        raise HTTPException(status_code=422, detail="Could not extract any text from the PDF.")

    # ── LLM: section + extract ────────────────────────────────────────────────
    print(f"⏳ Processing CV for {name}...")
    ai_summary, cv_sections, extracted = section_cv(name, cv_text)
    print(f"✅ CV processed — {extracted.get('job_title', 'unknown title')}")

    # ── Auto-generate CRM ID ──────────────────────────────────────────────────
    crm_id = f"CRM-{np.random.randint(100000, 999999)}"

    # ── Re-fit BM25 + embed + insert into Milvus (one row per section) ────────
    existing   = get_all_candidates_crm()
    milvus_ids: dict[str, int] = {}

    for s in SECTIONS:
        key  = s["key"]
        text = cv_sections[key]

        bm25   = ml_models["bm25"][key]
        corpus = [c["sections"].get(key, "") for c in existing if c.get("sections", {}).get(key, "")]
        corpus.append(text)
        bm25.fit(corpus)

        dense_vec    = ml_models["embedder"].encode([text], normalize_embeddings=True)[0].tolist()
        sparse_matrix = bm25.encode_documents([text])
        sparse_vec   = sparse_matrix_to_dict(sparse_matrix)
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

    # ── Store in SQLite ───────────────────────────────────────────────────────
    crm_row_id = insert_candidate_crm(crm_id, name, extracted, ai_summary, cv_sections, milvus_ids)

    return {
        "status":      "success",
        "crm_id":      crm_id,
        "crm_row_id":  crm_row_id,
        "name":        name,
        "extracted":   extracted,
        "ai_summary":  ai_summary,
        "sections":    cv_sections,
    }


# ── 9. Search ─────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, summary="Search candidates by job description")
async def search_candidates(request: SearchRequest):
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    # ── LLM: section the JD ──────────────────────────────────────────────────
    print("⏳ Sectioning JD...")
    jd_sections = section_jd(request.job_description)
    print("✅ JD sectioned:", list(jd_sections.keys()))

    fetch_limit = request.top_k * 3

    # ── Per-section hybrid search ─────────────────────────────────────────────
    # candidate_crm_id → {section_key → weighted_score}
    candidate_section_scores: dict[str, dict[str, float]] = {}
    # Store which Milvus IDs belong to which crm_id (for CRM enrichment)
    mid_to_crm: dict[int, str] = {}

    for s in SECTIONS:
        key   = s["key"]
        text  = jd_sections[key]

        if text.strip() in ("Not specified.", ""):
            # Skip sections the JD didn't mention — don't penalise candidates
            print(f"  ⏭️  Section '{key}' not in JD — skipped")
            continue

        bm25 = ml_models["bm25"][key]

        dense_q = ml_models["embedder"].encode(
            [text], normalize_embeddings=True
        )[0].tolist()

        sparse_matrix_q = bm25.encode_queries([text])
        sparse_q = sparse_matrix_to_dict(sparse_matrix_q)

        try:
            # Filter to only hits for this section_key
            section_filter = f'section_key == "{key}"'

            dense_req = AnnSearchRequest(
                data=[dense_q],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"ef": 100}},
                limit=fetch_limit,
                expr=section_filter,
            )

            reqs = [dense_req]
            if sparse_q:
                sparse_req = AnnSearchRequest(
                    data=[sparse_q],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                    limit=fetch_limit,
                    expr=section_filter,
                )
                reqs.append(sparse_req)

            # Hybrid search (RRF) — gives us ranking order for this section
            hybrid_hits = db_clients["milvus"].hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=reqs,
                ranker=RRFRanker(k=60),
                limit=fetch_limit,
                output_fields=["candidate_crm_id"],
                filter=section_filter,
            )[0]

            # Dense-only search — raw cosine similarities for scoring
            dense_raw = db_clients["milvus"].search(
                collection_name=COLLECTION_NAME,
                data=[dense_q],
                anns_field="dense_vector",
                search_params={"metric_type": "IP", "params": {"ef": 100}},
                filter=section_filter,
                limit=fetch_limit,
                output_fields=["candidate_crm_id"],
            )[0]

            dense_score_map: dict[int, float] = {
                hit["id"]: float(hit["distance"]) for hit in dense_raw
            }

            # Sparse-only search — normalised BM25 scores
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
                raw_vals = [float(h["distance"]) for h in sparse_raw]
                sparse_max = max(raw_vals) if raw_vals else 1.0
                sparse_max = sparse_max if sparse_max > 0 else 1.0
                sparse_score_map = {
                    hit["id"]: float(hit["distance"]) / sparse_max
                    for hit in sparse_raw
                }

            # Map hits to crm_id and record per-section score
            for hit in hybrid_hits:
                mid     = hit["id"]
                crm_cid = hit["entity"].get("candidate_crm_id", f"MV-{mid}")
                mid_to_crm[mid] = crm_cid

                d_score = dense_score_map.get(mid, 0.0)
                s_score = sparse_score_map.get(mid, 0.0)
                section_score = (DENSE_WEIGHT * d_score) + (SPARSE_WEIGHT * s_score)
                section_score = min(section_score, 1.0)

                if crm_cid not in candidate_section_scores:
                    candidate_section_scores[crm_cid] = {}
                # Keep highest score if candidate appears in multiple section hits
                existing_score = candidate_section_scores[crm_cid].get(key, 0.0)
                candidate_section_scores[crm_cid][key] = max(existing_score, section_score)

        except Exception as e:
            print(f"  ⚠️  Section '{key}' search error: {e}")
            continue

    # ── Compute weighted overall match % per candidate ────────────────────────
    # Only average over sections that the JD actually specified.
    active_sections = [
        s for s in SECTIONS
        if jd_sections.get(s["key"], "").strip() not in ("Not specified.", "")
    ]
    active_weight_total = sum(s["weight"] for s in active_sections)

    scored_candidates: list[tuple[str, float, dict[str, float]]] = []
    for crm_cid, sec_scores in candidate_section_scores.items():
        weighted_sum = sum(
            sec_scores.get(s["key"], 0.0) * s["weight"]
            for s in active_sections
        )
        # Re-normalise so scores still reach 100 % even if some sections are absent in JD
        overall = (weighted_sum / active_weight_total) if active_weight_total > 0 else 0.0
        overall = round(min(overall, 1.0) * 100, 1)
        scored_candidates.append((crm_cid, overall, sec_scores))

    # Sort by overall descending, take top_k
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    scored_candidates = scored_candidates[: request.top_k]

    # ── Enrich with CRM data ──────────────────────────────────────────────────
    all_crm = {c["crm_id"]: c for c in get_all_candidates_crm()}

    results: list[CandidateResult] = []
    for crm_cid, overall_pct, sec_scores in scored_candidates:
        crm = all_crm.get(crm_cid)

        section_score_list = [
            SectionScore(
                key=s["key"],
                label=s["label"],
                weight=s["weight"],
                score=round(sec_scores.get(s["key"], 0.0) * 100, 1),
            )
            for s in SECTIONS
        ]

        if crm:
            results.append(CandidateResult(
                crm_id=crm["crm_id"],
                name=crm["name"],
                job_title=crm.get("job_title", ""),
                nhs_band=crm.get("nhs_band", ""),
                location=crm.get("location", ""),
                registration=crm.get("registration", ""),
                match_percentage=overall_pct,
                section_scores=section_score_list,
                ai_summary=crm.get("ai_summary", ""),
                sections=crm.get("sections", {}),
            ))
        else:
            results.append(CandidateResult(
                crm_id=crm_cid,
                name="Unknown",
                job_title="",
                nhs_band="",
                location="",
                registration="",
                match_percentage=overall_pct,
                section_scores=section_score_list,
                ai_summary="",
                sections={},
            ))

    return SearchResponse(jd_sections=jd_sections, results=results)


# ── 10. Delete ────────────────────────────────────────────────────────────────

@app.delete("/candidate/{crm_id}", summary="Remove a candidate from the system")
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
    return {"status": "success", "message": f"Candidate '{crm_id}' deleted (all {len(milvus_ids)} section vectors removed)."}