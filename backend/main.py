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
from pymilvus import MilvusClient
from typing import Optional
import io

# ── 1. Configuration ──────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MILVUS_URI        = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME   = "healthcare_candidates"
SQLITE_DB_PATH    = "crm_database.db"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

ml_models = {}
db_clients = {}

# ── 2. SQLite CRM Database Helpers ────────────────────────────────────────────
def init_crm_db():
    """Creates the SQLite CRM database and tables if they don't exist."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            crm_id      TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            email       TEXT,
            phone       TEXT,
            location    TEXT,
            job_title   TEXT,
            nhs_band    TEXT,
            years_exp   INTEGER,
            specialisms TEXT,
            availability TEXT,
            salary_exp  TEXT,
            registration TEXT,
            ai_summary  TEXT,
            milvus_id   INTEGER,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_candidate_crm(crm_data: dict, ai_summary: str, milvus_id: int) -> int:
    """Inserts a candidate record into the SQLite CRM."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO candidates 
        (crm_id, name, email, phone, location, job_title, nhs_band, years_exp,
         specialisms, availability, salary_exp, registration, ai_summary, milvus_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        crm_data.get("crm_id", f"CRM-{np.random.randint(10000,99999)}"),
        crm_data.get("name", "Unknown"),
        crm_data.get("email", ""),
        crm_data.get("phone", ""),
        crm_data.get("location", ""),
        crm_data.get("job_title", ""),
        crm_data.get("nhs_band", ""),
        crm_data.get("years_exp", 0),
        json.dumps(crm_data.get("specialisms", [])),
        crm_data.get("availability", ""),
        crm_data.get("salary_exp", ""),
        crm_data.get("registration", ""),
        ai_summary,
        milvus_id
    ))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_candidate_by_milvus_id(milvus_id: int) -> Optional[dict]:
    """Fetches a candidate from CRM by their Milvus vector ID."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM candidates WHERE milvus_id = ?", (milvus_id,))
    row = c.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["specialisms"] = json.loads(d.get("specialisms") or "[]")
        return d
    return None

def get_all_candidates_crm() -> list:
    """Returns all candidates from the CRM database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM candidates ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["specialisms"] = json.loads(d.get("specialisms") or "[]")
        result.append(d)
    return result

def delete_candidate_by_crm_id(crm_id: str) -> bool:
    """Deletes a candidate from the CRM and returns their milvus_id."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT milvus_id FROM candidates WHERE crm_id = ?", (crm_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    milvus_id = row[0]
    c.execute("DELETE FROM candidates WHERE crm_id = ?", (crm_id,))
    conn.commit()
    conn.close()
    return milvus_id

# ── 3. Server Startup ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🏥 Booting Healthcare CV Search Engine...")

    # Init Google Gemini client
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini client ready")

    # Init embedding model
    print("Loading embedding model (all-mpnet-base-v2)...")
    ml_models["embedder"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    dimension = ml_models["embedder"].get_sentence_embedding_dimension()
    print(f"✅ Embedder ready (dim={dimension})")

    # Init Milvus
    print("Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    db_clients["milvus"] = client
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dimension,
            auto_id=True,
            metric_type="IP"
        )
        print(f"✅ Milvus collection '{COLLECTION_NAME}' created")
    else:
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Milvus collection '{COLLECTION_NAME}' loaded")

    # Init SQLite CRM
    init_crm_db()
    print("✅ SQLite CRM database ready")

    print("🚀 System fully operational!")
    yield

    print("Shutting down...")
    ml_models.clear()
    db_clients.clear()

app = FastAPI(title="Healthcare CV Semantic Search", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 4. Helper: PDF Extraction ─────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

# ── 5. Helper: Gemini Summarisation ──────────────────────────────────────────
def summarise_candidate(profile_data: dict, cv_text: str) -> str:
    prompt = f"""You are a specialist UK healthcare recruiter with deep NHS knowledge.
Synthesise the candidate's CRM profile data and raw CV into a single dense paragraph (200-300 words).
Write in the third person. Focus on:
- Clinical specialisms and competencies
- NHS Band level and equivalent roles
- Years of experience in UK healthcare settings
- Professional registrations (NMC, GMC, HCPC, etc.)
- Notable achievements and specialist skills
- Geographic availability and flexibility

CRM PROFILE DATA:
{json.dumps(profile_data, indent=2)}

RAW CV TEXT:
{cv_text[:6000]}

Produce ONLY the summary paragraph, no preamble or labels."""

    try:
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt, stream=False)
        return response.text.strip()
    except Exception as e:
        print(f"LLM error ({LLM_MODEL}): {e}", flush=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

# ── 6. Pydantic Models ────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    job_description: str
    top_k: int = 5

class CandidateResult(BaseModel):
    crm_id: str
    milvus_id: int
    name: str
    job_title: str
    nhs_band: str
    location: str
    years_exp: int
    specialisms: list
    availability: str
    salary_exp: str
    registration: str
    match_percentage: float
    ai_summary: str

# ── 7. Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Healthcare CV Search"}

@app.get("/candidates", summary="List all candidates in CRM")
async def list_candidates():
    """Returns all candidates stored in the CRM database."""
    candidates = get_all_candidates_crm()
    return {"count": len(candidates), "candidates": candidates}

@app.post("/ingest", summary="Ingest a candidate CV + profile into the system")
async def ingest_candidate(
    profile_data: str = Form(..., description='JSON with name, email, crm_id, job_title, nhs_band, specialisms, etc.'),
    cv_file: UploadFile = File(None, description="Optional PDF CV file")
):
    """
    Ingests a candidate into the system:
    1. Extracts text from PDF CV (if provided)
    2. Gemini summarises CV + profile data
    3. Embeds the summary
    4. Stores vector in Milvus
    5. Stores full profile in SQLite CRM
    Returns the CRM record and generated summary.
    """
    try:
        profile_dict = json.loads(profile_data)
    except Exception:
        raise HTTPException(status_code=400, detail="profile_data must be valid JSON.")

    # Extract PDF text if provided
    cv_text = ""
    if cv_file and cv_file.filename:
        if not cv_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files accepted.")
        cv_bytes = await cv_file.read()
        cv_text = extract_pdf_text(cv_bytes)

    # If no CV text, build a text block from profile data
    if not cv_text:
        cv_text = f"""
Name: {profile_dict.get('name', '')}
Job Title: {profile_dict.get('job_title', '')}
NHS Band: {profile_dict.get('nhs_band', '')}
Years Experience: {profile_dict.get('years_exp', '')}
Specialisms: {', '.join(profile_dict.get('specialisms', []))}
Location: {profile_dict.get('location', '')}
Registration: {profile_dict.get('registration', '')}
Notes: {profile_dict.get('notes', '')}
"""

    # Summarise with Claude
    ai_summary = summarise_candidate(profile_dict, cv_text)

    # Embed the summary
    vector = ml_models["embedder"].encode(
        [ai_summary], normalize_embeddings=True
    )[0].tolist()

    # Store in Milvus
    insert_result = db_clients["milvus"].insert(
        collection_name=COLLECTION_NAME,
        data=[{
            "vector": vector,
            "name": profile_dict.get("name", "Unknown"),
            "ai_summary": ai_summary
        }]
    )
    milvus_id = insert_result["ids"][0]

    # Store in SQLite CRM
    crm_row_id = insert_candidate_crm(profile_dict, ai_summary, milvus_id)

    return {
        "status": "success",
        "crm_row_id": crm_row_id,
        "milvus_id": milvus_id,
        "crm_id": profile_dict.get("crm_id"),
        "name": profile_dict.get("name"),
        "ai_summary": ai_summary
    }

@app.post("/search", response_model=list[CandidateResult], summary="Search candidates by job description")
async def search_candidates(request: SearchRequest):
    """
    Semantic search: takes a UK healthcare job description, 
    embeds it, queries Milvus, enriches results from SQLite CRM,
    returns top candidates with match percentages.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    # Embed the JD
    jd_vector = ml_models["embedder"].encode(
        [request.job_description], normalize_embeddings=True
    )[0].tolist()

    # Search Milvus
    try:
        results = db_clients["milvus"].search(
            collection_name=COLLECTION_NAME,
            data=[jd_vector],
            limit=request.top_k,
            output_fields=["name", "ai_summary"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus search error: {e}")

    # Enrich with CRM data
    enriched = []
    if results and len(results[0]) > 0:
        for hit in results[0]:
            milvus_id = hit["id"]
            score = hit["distance"]
            crm_profile = get_candidate_by_milvus_id(milvus_id)

            if crm_profile:
                enriched.append(CandidateResult(
                    crm_id=crm_profile["crm_id"],
                    milvus_id=milvus_id,
                    name=crm_profile["name"],
                    job_title=crm_profile["job_title"],
                    nhs_band=crm_profile["nhs_band"],
                    location=crm_profile["location"],
                    years_exp=crm_profile["years_exp"],
                    specialisms=crm_profile["specialisms"],
                    availability=crm_profile["availability"],
                    salary_exp=crm_profile["salary_exp"],
                    registration=crm_profile["registration"],
                    match_percentage=round(score * 100, 1),
                    ai_summary=crm_profile["ai_summary"]
                ))
            else:
                # Fallback if not found in CRM
                enriched.append(CandidateResult(
                    crm_id=f"MV-{milvus_id}",
                    milvus_id=milvus_id,
                    name=hit["entity"].get("name", "Unknown"),
                    job_title="",
                    nhs_band="",
                    location="",
                    years_exp=0,
                    specialisms=[],
                    availability="",
                    salary_exp="",
                    registration="",
                    match_percentage=round(score * 100, 1),
                    ai_summary=hit["entity"].get("ai_summary", "")
                ))

    return enriched

@app.delete("/candidate/{crm_id}", summary="Remove a candidate from the system")
async def delete_candidate(crm_id: str):
    """Deletes a candidate from both Milvus and the CRM SQLite database."""
    milvus_id = delete_candidate_by_crm_id(crm_id)
    if milvus_id is None:
        raise HTTPException(status_code=404, detail=f"Candidate '{crm_id}' not found.")

    try:
        db_clients["milvus"].delete(
            collection_name=COLLECTION_NAME,
            ids=[milvus_id]
        )
    except Exception as e:
        return {"status": "partial", "message": f"Removed from CRM but Milvus error: {e}"}

    return {"status": "success", "message": f"Candidate '{crm_id}' deleted from CRM and vector store."}
