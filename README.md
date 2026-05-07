# Healthcare CV Semantic Search — Prototype

Semantic search over UK healthcare candidate CVs and profiles using **Milvus** (vector DB), **Claude** (AI summarisation), and a local **SQLite CRM** to mimic a real recruitment database.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│  Search by JD  │  Ingest Candidate  │  CRM Database     │
└────────────────────────┬────────────────────────────────┘
                         │ REST (JSON)
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                        │
│                                                          │
│  /ingest  ──► Claude (summarise CV + profile)           │
│           ──► all-mpnet-base-v2 (embed summary)         │
│           ──► Milvus (store vector)                      │
│           ──► SQLite CRM (store full profile)            │
│                                                          │
│  /search  ──► all-mpnet-base-v2 (embed JD)              │
│           ──► Milvus ANN search (top-K by cosine)       │
│           ──► SQLite CRM (enrich with profile data)     │
│           ──► Return: crm_id + match % + profile        │
└────────────────────────┬────────────────────────────────┘
         ┌───────────────┴──────────────────┐
┌────────▼────────┐               ┌─────────▼──────────┐
│  Milvus 2.4.9   │               │   SQLite CRM DB    │
│  (vector store) │               │  (profile records) │
└─────────────────┘               └────────────────────┘
```

---

## Quick Start (Docker Compose — recommended)

### 1. Prerequisites
- Docker Desktop (or Docker + Docker Compose v2)
- Anthropic API key

### 2. Clone and configure
```bash
git clone <repo>
cd healthcare-cv-search
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start everything
```bash
docker compose up --build
```

Wait ~2 minutes for Milvus to initialise, the embedding model to download, and all services to be healthy.

### 4. Seed sample data (12 UK healthcare candidates)
```bash
docker compose exec backend python seed_data.py
```

### 5. Open the UI
```
http://localhost:5173
```

---

## Manual Setup (without Docker)

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Ensure Milvus is running (see Milvus standalone install docs)
# Default: http://localhost:19530

# Start FastAPI
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## API Reference

### `POST /ingest`
Ingests a candidate into the system.

**Form fields:**
- `profile_data` (required): JSON string with candidate profile
- `cv_file` (optional): PDF file

**Profile JSON fields:**
```json
{
  "crm_id": "CRM-001",
  "name": "Sarah Mitchell",
  "email": "sarah@example.com",
  "phone": "07700 900001",
  "location": "Manchester",
  "job_title": "Senior Staff Nurse",
  "nhs_band": "Band 6",
  "years_exp": 9,
  "specialisms": ["Adult Critical Care", "ICU"],
  "availability": "Immediately available",
  "salary_exp": "£35,000–£42,000",
  "registration": "NMC PIN: 12A3456B",
  "notes": "Additional context..."
}
```

### `POST /search`
Semantic search by job description.

**Request body:**
```json
{
  "job_description": "We are seeking a Band 6 ICU nurse...",
  "top_k": 5
}
```

**Response:**
```json
[
  {
    "crm_id": "CRM-001",
    "milvus_id": 449812345,
    "name": "Sarah Mitchell",
    "job_title": "Senior Staff Nurse",
    "nhs_band": "Band 6",
    "location": "Manchester",
    "years_exp": 9,
    "specialisms": ["Adult Critical Care", "ICU"],
    "availability": "Immediately available",
    "salary_exp": "£35,000–£42,000",
    "registration": "NMC PIN: 12A3456B",
    "match_percentage": 91.4,
    "ai_summary": "Sarah Mitchell is an experienced Band 6..."
  }
]
```

### `GET /candidates`
Returns all candidates in the CRM database.

### `DELETE /candidate/{crm_id}`
Removes a candidate from both Milvus and the CRM.

---

## How match percentage works

1. On ingest, Claude generates a dense 200–300 word summary emphasising clinical skills, NHS band, specialisms, and registrations
2. The summary is embedded using `all-mpnet-base-v2` (768-dimensional vector, normalised)
3. On search, the job description is embedded with the same model
4. Milvus performs approximate nearest-neighbour search using **inner product** (= cosine similarity on normalised vectors)
5. Raw cosine scores (0–1) are multiplied by 100 to give a **match percentage**

> Note: Scores above ~75% indicate strong semantic alignment. Scores above 85% are rare and indicate exceptional match.

---

## Sample job description to test with

```
We are seeking an experienced Band 6 Staff Nurse to join our Adult Intensive Care Unit 
at Manchester University NHS Foundation Trust. 

The successful candidate must hold current NMC registration and have a minimum of 3 years 
post-registration experience in an adult critical care or ICU setting. 

Essential: Competency in mechanical ventilation management, central line care, 
arterial line monitoring, and sepsis recognition and management. Experience with 
ECMO or CVVHDF is highly desirable.

You will be expected to act as shift co-ordinator and provide mentorship to Band 5 nurses. 
Immediate availability preferred. Salary £35,000–£41,659 per annum (AfC Band 6).
```

---

## Project Structure

```
healthcare-cv-search/
├── backend/
│   ├── main.py            # FastAPI app (ingest, search, CRM endpoints)
│   ├── seed_data.py       # 12 sample UK healthcare candidates
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React UI (Search / Ingest / CRM tabs)
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml     # Milvus + backend + frontend
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection URI |

---

## Migrating to production

- Replace SQLite with PostgreSQL (swap `sqlite3` calls for `asyncpg`)
- Replace local Milvus with Zilliz Cloud managed service
- Add authentication (JWT) to the FastAPI endpoints
- Add batch ingestion endpoint for bulk CV uploads
- Store raw CV PDFs in S3/Azure Blob alongside the vector
