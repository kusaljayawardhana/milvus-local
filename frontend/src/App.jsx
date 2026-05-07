import { useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const NHS_BANDS = [
  "Band 2", "Band 3", "Band 4", "Band 5", "Band 6", "Band 7", "Band 8a",
  "Band 8b", "Band 8c", "Band 8d", "Band 9", "SAS Grade", "Specialty Registrar",
  "Consultant", "GP Principal", "Other"
];

const SPECIALISMS = [
  "Adult Critical Care", "ICU", "A&E / Emergency Medicine", "Paediatrics", "CAMHS",
  "District Nursing", "Community Nursing", "General Practice", "Mental Health",
  "Psychiatry", "Forensic Psychiatry", "Musculoskeletal", "Physiotherapy",
  "Occupational Therapy", "Radiology", "Interventional Radiology",
  "Operating Department", "Anaesthetics", "Cardiology", "Respiratory",
  "Orthopaedics", "Stroke Rehabilitation", "Palliative Care", "Oncology",
  "Diabetes Management", "Wound Care", "Prescribing", "Pre-hospital Care",
  "HEMS", "Internal Medicine", "Endocrinology", "Neurology", "Renal",
  "Gastroenterology", "Infectious Diseases", "Midwifery", "Neonatal",
];

function getInitials(name) {
  return name.split(" ").map(n => n[0]).slice(0, 2).join("").toUpperCase();
}

function getMatchColor(pct) {
  if (pct >= 80) return { bg: "#eaf3de", color: "#3b6d11", border: "#639922" };
  if (pct >= 60) return { bg: "#faeeda", color: "#854f0b", border: "#ba7517" };
  return { bg: "#fcebeb", color: "#a32d2d", border: "#e24b4a" };
}

function getAvatarColor(name) {
  const colors = [
    { bg: "#EEEDFE", color: "#534AB7" }, { bg: "#E1F5EE", color: "#0F6E56" },
    { bg: "#E6F1FB", color: "#185FA5" }, { bg: "#FBEAF0", color: "#993556" },
    { bg: "#FAEEDA", color: "#854F0B" }, { bg: "#EAF3DE", color: "#3B6D11" },
  ];
  return colors[name.charCodeAt(0) % colors.length];
}

// ── Components ────────────────────────────────────────────────────────────────

function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4, borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 24 }}>
      {tabs.map(t => (
        <button key={t.id} onClick={() => onChange(t.id)} style={{
          padding: "10px 20px", background: "none", border: "none",
          borderBottom: active === t.id ? "2px solid var(--color-text-primary)" : "2px solid transparent",
          color: active === t.id ? "var(--color-text-primary)" : "var(--color-text-secondary)",
          fontWeight: active === t.id ? 500 : 400, cursor: "pointer", fontSize: 14,
          transition: "all 0.15s", marginBottom: -1,
        }}>{t.label}</button>
      ))}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color?.bg || "var(--color-background-secondary)",
      color: color?.color || "var(--color-text-secondary)",
      border: `0.5px solid ${color?.border || "var(--color-border-tertiary)"}`,
      borderRadius: 6, padding: "2px 8px", fontSize: 11, fontWeight: 500,
    }}>{text}</span>
  );
}

function Alert({ type, message }) {
  const styles = {
    success: { bg: "#eaf3de", color: "#3b6d11", icon: "ti-check" },
    error: { bg: "#fcebeb", color: "#a32d2d", icon: "ti-alert-circle" },
    info: { bg: "#e6f1fb", color: "#185fa5", icon: "ti-info-circle" },
  };
  const s = styles[type] || styles.info;
  return (
    <div style={{
      background: s.bg, color: s.color, borderRadius: 8, padding: "10px 14px",
      fontSize: 13, display: "flex", alignItems: "center", gap: 8, marginTop: 12,
    }}>
      <i className={`ti ${s.icon}`} style={{ fontSize: 16 }} aria-hidden="true" />
      {message}
    </div>
  );
}

function CandidateCard({ result, rank }) {
  const [expanded, setExpanded] = useState(false);
  const mc = getMatchColor(result.match_percentage);
  const av = getAvatarColor(result.name);
  return (
    <div style={{
      background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: 12, padding: "16px 20px", transition: "border-color 0.15s",
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
        <div style={{ position: "relative", flexShrink: 0 }}>
          <div style={{
            width: 46, height: 46, borderRadius: "50%", background: av.bg,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: 500, fontSize: 14, color: av.color,
          }}>{getInitials(result.name)}</div>
          <div style={{
            position: "absolute", bottom: -2, right: -2, width: 18, height: 18,
            borderRadius: "50%", background: "var(--color-background-primary)",
            display: "flex", alignItems: "center", justifyContent: "center",
            border: "0.5px solid var(--color-border-tertiary)",
            fontSize: 10, fontWeight: 600, color: "var(--color-text-secondary)",
          }}>#{rank}</div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
            <span style={{ fontWeight: 500, fontSize: 15 }}>{result.name}</span>
            <Badge text={result.crm_id} />
            <div style={{
              marginLeft: "auto", background: mc.bg, color: mc.color,
              border: `0.5px solid ${mc.border}`, borderRadius: 8,
              padding: "3px 10px", fontSize: 13, fontWeight: 500,
            }}>
              {result.match_percentage}% match
            </div>
          </div>
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 8 }}>
            {result.job_title}{result.nhs_band ? ` · ${result.nhs_band}` : ""}
            {result.location ? ` · ` : ""}
            {result.location && <><i className="ti ti-map-pin" style={{ fontSize: 13, verticalAlign: -1 }} aria-hidden="true" /> {result.location}</>}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
            {result.years_exp > 0 && <Badge text={`${result.years_exp} yrs exp`} />}
            {result.availability && <Badge text={result.availability} color={{ bg: "#e1f5ee", color: "#0f6e56", border: "#1d9e75" }} />}
            {result.salary_exp && <Badge text={result.salary_exp} />}
            {(result.specialisms || []).slice(0, 3).map(s => <Badge key={s} text={s} />)}
            {(result.specialisms || []).length > 3 && <Badge text={`+${result.specialisms.length - 3} more`} />}
          </div>
          {result.registration && (
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8 }}>
              <i className="ti ti-id-badge" style={{ fontSize: 13, verticalAlign: -1 }} aria-hidden="true" /> {result.registration}
            </div>
          )}
          <button onClick={() => setExpanded(e => !e)} style={{
            fontSize: 12, color: "var(--color-text-secondary)", background: "none",
            border: "none", cursor: "pointer", padding: 0, display: "flex", alignItems: "center", gap: 4,
          }}>
            <i className={`ti ${expanded ? "ti-chevron-up" : "ti-chevron-down"}`} style={{ fontSize: 13 }} aria-hidden="true" />
            {expanded ? "Hide" : "Show"} AI summary
          </button>
          {expanded && (
            <div style={{
              marginTop: 10, padding: "10px 14px", background: "var(--color-background-secondary)",
              borderRadius: 8, fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6,
              borderLeft: "2px solid var(--color-border-secondary)",
            }}>
              {result.ai_summary}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Match % bar chart ─────────────────────────────────────────────────────────
function MatchChart({ results }) {
  if (!results.length) return null;
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8, marginTop: 0 }}>
        Semantic match scores
      </p>
      {results.map((r, i) => {
        const mc = getMatchColor(r.match_percentage);
        return (
          <div key={r.crm_id} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)", width: 120, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {r.name.split(" ")[0]}
            </span>
            <div style={{ flex: 1, background: "var(--color-background-secondary)", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{ width: `${r.match_percentage}%`, background: mc.border, height: "100%", borderRadius: 4, transition: "width 0.6s ease" }} />
            </div>
            <span style={{ fontSize: 12, fontWeight: 500, color: mc.color, width: 42, textAlign: "right", flexShrink: 0 }}>
              {r.match_percentage}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Search Tab ────────────────────────────────────────────────────────────────
function SearchTab() {
  const [jd, setJd] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  const handleSearch = async () => {
    if (!jd.trim()) { setError("Please enter a job description."); return; }
    setLoading(true); setError(""); setResults([]); setSearched(false);
    try {
      const resp = await fetch(`${API_BASE}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_description: jd, top_k: topK }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || "Search failed");
      setResults(await resp.json());
      setSearched(true);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <div style={{ display: "grid", gap: 12, marginBottom: 16 }}>
        <label style={{ fontSize: 13, fontWeight: 500 }}>Job description</label>
        <textarea
          value={jd} onChange={e => setJd(e.target.value)}
          placeholder="Paste a UK healthcare job description here…&#10;&#10;Example: We are seeking a Band 6 ICU Staff Nurse for a busy NHS teaching hospital in Manchester. The successful candidate must hold current NMC registration and have at least 3 years of adult critical care experience including ventilator management and sepsis protocols..."
          rows={8}
          style={{ width: "100%", resize: "vertical", boxSizing: "border-box", fontFamily: "var(--font-sans)", fontSize: 13, lineHeight: 1.6 }}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Top results:</label>
            <input type="number" min={1} max={20} value={topK} onChange={e => setTopK(Number(e.target.value))}
              style={{ width: 60 }} />
          </div>
          <button onClick={handleSearch} disabled={loading} style={{
            marginLeft: "auto", padding: "9px 22px", background: "var(--color-text-primary)",
            color: "var(--color-background-primary)", border: "none", borderRadius: 8,
            fontWeight: 500, fontSize: 14, cursor: loading ? "wait" : "pointer",
            opacity: loading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 8,
          }}>
            {loading ? <><i className="ti ti-loader-2" style={{ fontSize: 16 }} aria-hidden="true" /> Searching…</> : <><i className="ti ti-search" style={{ fontSize: 16 }} aria-hidden="true" /> Find candidates</>}
          </button>
        </div>
      </div>

      {error && <Alert type="error" message={error} />}

      {searched && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <p style={{ margin: 0, fontWeight: 500 }}>
              {results.length > 0 ? `${results.length} candidates found` : "No candidates matched"}
            </p>
          </div>
          {results.length > 0 && <MatchChart results={results} />}
          <div style={{ display: "grid", gap: 12 }}>
            {results.map((r, i) => <CandidateCard key={r.crm_id} result={r} rank={i + 1} />)}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Ingest Tab ────────────────────────────────────────────────────────────────
function IngestTab({ onSuccess }) {
  const [form, setForm] = useState({
    crm_id: "", name: "", email: "", phone: "", location: "",
    job_title: "", nhs_band: "Band 6", years_exp: "", specialisms: [],
    availability: "", salary_exp: "", registration: "", notes: "",
  });
  const [cvFile, setCvFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));
  const toggleSpec = (s) => set("specialisms", form.specialisms.includes(s) ? form.specialisms.filter(x => x !== s) : [...form.specialisms, s]);

  const handleSubmit = async () => {
    if (!form.name.trim() || !form.crm_id.trim()) {
      setStatus({ type: "error", message: "CRM ID and Name are required." }); return;
    }
    setLoading(true); setStatus(null);
    try {
      const fd = new FormData();
      fd.append("profile_data", JSON.stringify({ ...form, years_exp: Number(form.years_exp) || 0 }));
      if (cvFile) fd.append("cv_file", cvFile);
      const resp = await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
      if (!resp.ok) throw new Error((await resp.json()).detail || "Ingest failed");
      const data = await resp.json();
      setStatus({ type: "success", message: `${data.name} ingested successfully. Milvus ID: ${data.milvus_id}` });
      setForm({ crm_id: "", name: "", email: "", phone: "", location: "", job_title: "", nhs_band: "Band 6", years_exp: "", specialisms: [], availability: "", salary_exp: "", registration: "", notes: "" });
      setCvFile(null);
      onSuccess();
    } catch (e) { setStatus({ type: "error", message: e.message }); }
    finally { setLoading(false); }
  };

  const f = (label, key, placeholder, type = "text") => (
    <div>
      <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>{label}</label>
      <input type={type} value={form[key]} onChange={e => set(key, e.target.value)}
        placeholder={placeholder} style={{ width: "100%", boxSizing: "border-box" }} />
    </div>
  );

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {f("CRM ID *", "crm_id", "e.g. CRM-100")}
        {f("Full name *", "name", "e.g. Sarah Mitchell")}
        {f("Email", "email", "candidate@example.com", "email")}
        {f("Phone", "phone", "07700 000000")}
        {f("Location", "location", "e.g. Manchester, Greater Manchester")}
        {f("Job title", "job_title", "e.g. Senior Staff Nurse")}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        <div>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>NHS band</label>
          <select value={form.nhs_band} onChange={e => set("nhs_band", e.target.value)} style={{ width: "100%", boxSizing: "border-box" }}>
            {NHS_BANDS.map(b => <option key={b}>{b}</option>)}
          </select>
        </div>
        {f("Years exp.", "years_exp", "e.g. 8", "number")}
        {f("Salary expectation", "salary_exp", "e.g. £35,000–£42,000")}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {f("Registration", "registration", "e.g. NMC PIN: 12A3456B")}
        {f("Availability", "availability", "e.g. Immediately available")}
      </div>

      <div>
        <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 8 }}>Specialisms</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {SPECIALISMS.map(s => (
            <button key={s} onClick={() => toggleSpec(s)} style={{
              padding: "4px 10px", borderRadius: 6, fontSize: 12, cursor: "pointer",
              border: form.specialisms.includes(s) ? "1.5px solid #185fa5" : "0.5px solid var(--color-border-tertiary)",
              background: form.specialisms.includes(s) ? "#e6f1fb" : "var(--color-background-secondary)",
              color: form.specialisms.includes(s) ? "#185fa5" : "var(--color-text-secondary)",
              fontWeight: form.specialisms.includes(s) ? 500 : 400,
            }}>{s}</button>
          ))}
        </div>
      </div>

      <div>
        <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>Additional notes</label>
        <textarea value={form.notes} onChange={e => set("notes", e.target.value)}
          placeholder="Additional context, qualifications, achievements…" rows={3}
          style={{ width: "100%", boxSizing: "border-box", fontFamily: "var(--font-sans)", fontSize: 13 }} />
      </div>

      <div>
        <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>CV file (PDF, optional)</label>
        <div style={{
          border: "0.5px dashed var(--color-border-secondary)", borderRadius: 8, padding: "14px 16px",
          display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
          background: "var(--color-background-secondary)",
        }} onClick={() => document.getElementById("cv-upload").click()}>
          <i className="ti ti-file-text" style={{ fontSize: 20, color: "var(--color-text-tertiary)" }} aria-hidden="true" />
          <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
            {cvFile ? cvFile.name : "Click to upload PDF CV — Claude will summarise it automatically"}
          </span>
          <input id="cv-upload" type="file" accept=".pdf" style={{ display: "none" }}
            onChange={e => setCvFile(e.target.files[0] || null)} />
        </div>
      </div>

      {status && <Alert type={status.type} message={status.message} />}

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <button onClick={handleSubmit} disabled={loading} style={{
          padding: "9px 22px", background: "var(--color-text-primary)",
          color: "var(--color-background-primary)", border: "none", borderRadius: 8,
          fontWeight: 500, fontSize: 14, cursor: loading ? "wait" : "pointer",
          opacity: loading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 8,
        }}>
          {loading ? <><i className="ti ti-loader-2" style={{ fontSize: 16 }} aria-hidden="true" /> Processing…</> : <><i className="ti ti-cloud-upload" style={{ fontSize: 16 }} aria-hidden="true" /> Ingest candidate</>}
        </button>
      </div>
    </div>
  );
}

// ── CRM Tab ───────────────────────────────────────────────────────────────────
function CrmTab({ refresh }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [status, setStatus] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/candidates`);
      const data = await resp.json();
      setCandidates(data.candidates || []);
    } catch (e) { setStatus({ type: "error", message: e.message }); }
    finally { setLoading(false); }
  }, []);

  useState(() => { load(); }, []);
  // reload when refresh token changes
  useState(() => { if (refresh > 0) load(); }, [refresh]);

  const handleDelete = async (crm_id, name) => {
    if (!window.confirm(`Remove ${name} from the system?`)) return;
    setDeletingId(crm_id);
    try {
      const resp = await fetch(`${API_BASE}/candidate/${encodeURIComponent(crm_id)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error((await resp.json()).detail);
      setStatus({ type: "success", message: `${name} removed.` });
      load();
    } catch (e) { setStatus({ type: "error", message: e.message }); }
    finally { setDeletingId(null); }
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
          {loading ? "Loading…" : `${candidates.length} candidates in CRM`}
        </span>
        <button onClick={load} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
          <i className="ti ti-refresh" style={{ fontSize: 14 }} aria-hidden="true" /> Refresh
        </button>
      </div>
      {status && <Alert type={status.type} message={status.message} />}
      <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
        {candidates.map(c => {
          const av = getAvatarColor(c.name);
          return (
            <div key={c.crm_id} style={{
              background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: 10, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12,
            }}>
              <div style={{
                width: 38, height: 38, borderRadius: "50%", background: av.bg,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontWeight: 500, fontSize: 13, color: av.color, flexShrink: 0,
              }}>{getInitials(c.name)}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontWeight: 500, fontSize: 14 }}>{c.name}</span>
                  <Badge text={c.crm_id} />
                  {c.nhs_band && <Badge text={c.nhs_band} />}
                </div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                  {c.job_title}{c.location ? ` · ${c.location}` : ""}
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
                {c.availability && <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{c.availability}</span>}
                <button onClick={() => handleDelete(c.crm_id, c.name)} disabled={deletingId === c.crm_id}
                  style={{
                    background: "none", border: "0.5px solid var(--color-border-tertiary)",
                    borderRadius: 6, padding: "4px 8px", cursor: "pointer",
                    color: "#a32d2d", fontSize: 12, display: "flex", alignItems: "center", gap: 4,
                  }}>
                  <i className="ti ti-trash" style={{ fontSize: 13 }} aria-hidden="true" />
                  {deletingId === c.crm_id ? "Removing…" : "Remove"}
                </button>
              </div>
            </div>
          );
        })}
        {!loading && candidates.length === 0 && (
          <div style={{ textAlign: "center", padding: "40px 0", color: "var(--color-text-tertiary)" }}>
            <i className="ti ti-database-off" style={{ fontSize: 32, display: "block", marginBottom: 8 }} aria-hidden="true" />
            No candidates ingested yet. Use the Ingest tab to add candidates.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("search");
  const [refreshCrm, setRefreshCrm] = useState(0);

  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "24px 20px" }}>
      <h2 className="sr-only">Healthcare CV semantic search prototype</h2>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8, background: "#e6f1fb",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <i className="ti ti-stethoscope" style={{ fontSize: 18, color: "#185fa5" }} aria-hidden="true" />
          </div>
          <span style={{ fontWeight: 500, fontSize: 18 }}>Healthcare CV Search</span>
          <span style={{ fontSize: 12, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 6, padding: "2px 8px", color: "var(--color-text-tertiary)" }}>
            Prototype
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
          Semantic candidate matching for UK healthcare roles · Milvus vector DB · Claude AI summaries
        </p>
      </div>

      <Tabs
        tabs={[
          { id: "search", label: "🔍 Search by JD" },
          { id: "ingest", label: "➕ Ingest candidate" },
          { id: "crm", label: "🗂 CRM database" },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === "search" && <SearchTab />}
      {tab === "ingest" && <IngestTab onSuccess={() => setRefreshCrm(r => r + 1)} />}
      {tab === "crm" && <CrmTab refresh={refreshCrm} />}
    </div>
  );
}
