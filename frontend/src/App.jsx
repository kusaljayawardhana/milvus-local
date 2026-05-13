import { useState, useCallback, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// ── Helpers ───────────────────────────────────────────────────────────────────
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

const SECTION_META = [
  { key: "profile",        label: "Profile Summary" },
  { key: "qualifications", label: "Professional Qualifications" },
  { key: "specialties",    label: "Specialties" },
  { key: "experience",     label: "Working Experience" },
  { key: "skills",         label: "Skills" },
];

// ── Shared components ─────────────────────────────────────────────────────────

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
    error:   { bg: "#fcebeb", color: "#a32d2d", icon: "ti-alert-circle" },
    info:    { bg: "#e6f1fb", color: "#185fa5", icon: "ti-info-circle" },
  };
  const s = styles[type] || styles.info;
  return (
    <div style={{
      background: s.bg, color: s.color, borderRadius: 8, padding: "10px 14px",
      fontSize: 13, display: "flex", alignItems: "flex-start", gap: 8, marginTop: 12,
    }}>
      <i className={`ti ${s.icon}`} style={{ fontSize: 16, flexShrink: 0, marginTop: 1 }} />
      <span style={{ lineHeight: 1.5 }}>{message}</span>
    </div>
  );
}

// ── JD Sections panel ─────────────────────────────────────────────────────────
function JdSectionsPanel({ jdSections, jdDomain }) {
  const [open, setOpen] = useState(false);

  if (!jdSections || Object.keys(jdSections).length === 0) return null;

  return (
    <div style={{
      border: "0.5px solid #a8cfef",
      borderRadius: 10,
      background: "#f0f7fd",
      marginBottom: 20,
      overflow: "hidden",
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", background: "none", border: "none", cursor: "pointer",
          padding: "11px 16px", display: "flex", alignItems: "center", gap: 10,
          textAlign: "left",
        }}
      >
        <i className="ti ti-file-text" style={{ fontSize: 15, color: "#185fa5", flexShrink: 0 }} />
        <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#185fa5" }}>
          How the LLM read the job description
        </span>
        {jdDomain && (
          <span style={{
            fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 6,
            background: "#e6f1fb", color: "#185fa5", border: "0.5px solid #a8cfef",
          }}>
            {jdDomain.replace(/_/g, " ")}
          </span>
        )}
        <i
          className={`ti ${open ? "ti-chevron-up" : "ti-chevron-down"}`}
          style={{ fontSize: 14, color: "#185fa5", flexShrink: 0 }}
        />
      </button>

      {open && (
        <div style={{ borderTop: "0.5px solid #a8cfef" }}>
          {SECTION_META.map(({ key, label }) => {
            const text = jdSections[key] || "";
            const empty = text.trim() === "Not specified." || text.trim() === "";
            return (
              <div
                key={key}
                style={{
                  padding: "10px 16px",
                  borderBottom: "0.5px solid #d4e8f5",
                  background: empty ? "transparent" : "#fff",
                  opacity: empty ? 0.5 : 1,
                }}
              >
                <div style={{
                  fontSize: 11, fontWeight: 600, color: "#185fa5",
                  textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4,
                }}>
                  {label}
                  {empty && (
                    <span style={{ fontWeight: 400, textTransform: "none", marginLeft: 6, color: "#888" }}>
                      · not in JD (section skipped in scoring)
                    </span>
                  )}
                </div>
                <p style={{
                  margin: 0, fontSize: 12.5, color: empty ? "#aaa" : "#2c3e50",
                  lineHeight: 1.6, fontStyle: empty ? "italic" : "normal",
                }}>
                  {text || "Not specified."}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── CV Sections drawer (inside candidate card) ────────────────────────────────
function CvSectionsDrawer({ sections }) {
  const [openKey, setOpenKey] = useState(null);

  if (!sections || Object.keys(sections).length === 0) return (
    <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", fontStyle: "italic", marginTop: 6 }}>
      No CV sections stored.
    </div>
  );

  return (
    <div style={{
      marginTop: 10,
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: 8,
      overflow: "hidden",
    }}>
      {SECTION_META.map(({ key, label }) => {
        const text = sections[key] || "";
        const empty = !text.trim() || text.trim().endsWith("Not stated.");
        const isOpen = openKey === key;
        return (
          <div key={key} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <button
              onClick={() => setOpenKey(isOpen ? null : key)}
              style={{
                width: "100%", background: isOpen
                  ? "var(--color-background-secondary)"
                  : "var(--color-background-primary)",
                border: "none", cursor: empty ? "default" : "pointer",
                padding: "8px 12px",
                display: "flex", alignItems: "center", gap: 8, textAlign: "left",
              }}
              disabled={empty}
            >
              <span style={{
                width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
                background: empty ? "var(--color-border-secondary)" : "#639922",
              }} />
              <span style={{
                flex: 1, fontSize: 12, fontWeight: 500,
                color: empty ? "var(--color-text-tertiary)" : "var(--color-text-primary)",
              }}>
                {label}
              </span>
              {!empty && (
                <i
                  className={`ti ${isOpen ? "ti-chevron-up" : "ti-chevron-down"}`}
                  style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}
                />
              )}
              {empty && (
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", fontStyle: "italic" }}>
                  not stated
                </span>
              )}
            </button>
            {isOpen && !empty && (
              <div style={{
                padding: "10px 12px 12px 28px",
                fontSize: 12.5, color: "var(--color-text-secondary)",
                lineHeight: 1.65,
                background: "var(--color-background-secondary)",
                borderTop: "0.5px solid var(--color-border-tertiary)",
                whiteSpace: "pre-wrap",
              }}>
                {text}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Section score bar ─────────────────────────────────────────────────────────
function SectionScoreBar({ label, score, weight }) {
  const mc = getMatchColor(score);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
      <span style={{ width: 130, flexShrink: 0, color: "var(--color-text-secondary)" }}>
        {label}
        <span style={{ color: "var(--color-text-tertiary)", fontWeight: 400 }}> · {Math.round(weight * 100)}%</span>
      </span>
      <div style={{ flex: 1, background: "var(--color-background-secondary)", borderRadius: 3, height: 6, overflow: "hidden" }}>
        <div style={{ width: `${score}%`, background: mc.border, height: "100%", borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ width: 36, textAlign: "right", fontWeight: 500, color: mc.color, flexShrink: 0 }}>{score}%</span>
    </div>
  );
}

// ── Candidate result card ─────────────────────────────────────────────────────
function CandidateCard({ result, rank }) {
  const [showSections, setShowSections]   = useState(false);
  const [showSummary, setShowSummary]     = useState(false);
  const [showCvSections, setShowCvSections] = useState(false);
  const mc = getMatchColor(result.match_percentage);
  const av = getAvatarColor(result.name);

  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: 12, padding: "16px 20px",
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
        {/* Avatar */}
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

        {/* Main info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Name row */}
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

          {/* Job title / band / location */}
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 8 }}>
            {result.job_title}
            {result.nhs_band ? ` · ${result.nhs_band}` : ""}
            {result.location ? ` · ` : ""}
            {result.location && (
              <><i className="ti ti-map-pin" style={{ fontSize: 13, verticalAlign: -1 }} /> {result.location}</>
            )}
          </div>

          {/* Domain + registration */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 8, fontSize: 12 }}>
            {result.profession_domain && (
              <span style={{
                padding: "1px 7px", borderRadius: 5, fontWeight: 500,
                background: "#eeedfe", color: "#534ab7", border: "0.5px solid #c4c0f5",
              }}>
                {result.profession_domain.replace(/_/g, " ")}
              </span>
            )}
            {result.registration && (
              <span style={{ color: "var(--color-text-tertiary)" }}>
                <i className="ti ti-id-badge" style={{ fontSize: 12, verticalAlign: -1 }} /> {result.registration}
              </span>
            )}
          </div>

          {/* Toggle buttons row */}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            <ToggleBtn
              icon="ti-chart-bar"
              label="section scores"
              open={showSections}
              onClick={() => setShowSections(s => !s)}
            />
            <ToggleBtn
              icon="ti-file-description"
              label="AI summary"
              open={showSummary}
              onClick={() => setShowSummary(s => !s)}
            />
            <ToggleBtn
              icon="ti-layout-list"
              label="CV sections (how LLM read the CV)"
              open={showCvSections}
              onClick={() => setShowCvSections(s => !s)}
            />
          </div>

          {/* Section score breakdown */}
          {showSections && result.section_scores?.length > 0 && (
            <div style={{
              marginTop: 10, padding: "10px 14px",
              background: "var(--color-background-secondary)",
              borderRadius: 8, display: "grid", gap: 6,
              borderLeft: "2px solid var(--color-border-secondary)",
            }}>
              {result.section_scores.map(s => (
                <SectionScoreBar key={s.key} label={s.label} score={s.score} weight={s.weight} />
              ))}
            </div>
          )}

          {/* AI summary */}
          {showSummary && (
            <div style={{
              marginTop: 10, padding: "10px 14px",
              background: "var(--color-background-secondary)",
              borderRadius: 8, fontSize: 13, color: "var(--color-text-secondary)",
              lineHeight: 1.6, borderLeft: "2px solid var(--color-border-secondary)",
            }}>
              {result.ai_summary || <em style={{ color: "var(--color-text-tertiary)" }}>No summary stored.</em>}
            </div>
          )}

          {/* CV sections drawer */}
          {showCvSections && (
            <div style={{ marginTop: 10 }}>
              <CvSectionsDrawer sections={result.sections} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ToggleBtn({ icon, label, open, onClick }) {
  return (
    <button onClick={onClick} style={{
      fontSize: 12, color: "var(--color-text-secondary)", background: "none",
      border: "none", cursor: "pointer", padding: 0, display: "flex", alignItems: "center", gap: 4,
    }}>
      <i className={`ti ${open ? "ti-chevron-up" : icon}`} style={{ fontSize: 13 }} />
      {open ? `Hide ${label}` : `Show ${label}`}
    </button>
  );
}

// ── Match % overview bar chart ────────────────────────────────────────────────
function MatchChart({ results }) {
  if (!results.length) return null;
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8, marginTop: 0 }}>
        Overall match scores
      </p>
      {results.map(r => {
        const mc = getMatchColor(r.match_percentage);
        return (
          <div key={r.crm_id} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{
              fontSize: 12, color: "var(--color-text-secondary)",
              width: 120, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
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

// ── Search tab ────────────────────────────────────────────────────────────────
function SearchTab() {
  const [jd, setJd] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [jdSections, setJdSections] = useState({});
  const [jdDomain, setJdDomain] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  const handleSearch = async () => {
    if (!jd.trim()) { setError("Please enter a job description."); return; }
    setLoading(true); setError(""); setResults([]); setSearched(false);
    setJdSections({}); setJdDomain("");
    try {
      const resp = await fetch(`${API_BASE}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_description: jd, top_k: topK }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || "Search failed");
      const data = await resp.json();
      setResults(data.results || []);
      setJdSections(data.jd_sections || {});
      setJdDomain(data.jd_domain || "");
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
          placeholder={"Paste a UK healthcare job description here…\n\nExample: We are seeking a Band 6 ICU Staff Nurse for a busy NHS teaching hospital in Manchester. The successful candidate must hold current NMC registration and have at least 3 years of adult critical care experience including ventilator management and sepsis protocols…"}
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
            {loading
              ? <><i className="ti ti-loader-2" style={{ fontSize: 16 }} /> Searching…</>
              : <><i className="ti ti-search" style={{ fontSize: 16 }} /> Find candidates</>}
          </button>
        </div>
      </div>

      {error && <Alert type="error" message={error} />}

      {searched && (
        <div style={{ marginTop: 20 }}>
          {/* JD sections panel — always shown after a search */}
          <JdSectionsPanel jdSections={jdSections} jdDomain={jdDomain} />

          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
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

// ── Ingest tab ────────────────────────────────────────────────────────────────
function IngestTab({ onSuccess }) {
  const [name, setName] = useState("");
  const [cvFile, setCvFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [lastIngested, setLastIngested] = useState(null);

  const handleSubmit = async () => {
    if (!name.trim()) {
      setStatus({ type: "error", message: "Please enter the candidate's name." }); return;
    }
    if (!cvFile) {
      setStatus({ type: "error", message: "Please upload a CV PDF." }); return;
    }
    setLoading(true); setStatus(null); setLastIngested(null);
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("cv_file", cvFile);
      const resp = await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
      if (!resp.ok) throw new Error((await resp.json()).detail || "Ingest failed");
      const data = await resp.json();
      setStatus({
        type: "success",
        message: `${data.name} ingested successfully · ${data.crm_id} · ${data.extracted?.job_title || "role extracted"} · ${data.extracted?.nhs_band || ""}`,
      });
      setLastIngested(data);
      setName("");
      setCvFile(null);
      document.getElementById("cv-upload").value = "";
      onSuccess();
    } catch (e) {
      setStatus({ type: "error", message: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 20, maxWidth: 520 }}>
      {/* Info banner */}
      <div style={{
        background: "var(--color-background-secondary)", borderRadius: 8,
        padding: "10px 14px", fontSize: 13, color: "var(--color-text-secondary)",
        display: "flex", gap: 8, alignItems: "flex-start",
        border: "0.5px solid var(--color-border-tertiary)",
      }}>
        <i className="ti ti-sparkles" style={{ fontSize: 15, flexShrink: 0, marginTop: 1 }} />
        <span>Gemini will automatically extract job title, NHS band, location, registration, specialties, skills and experience directly from the CV.</span>
      </div>

      {/* Name field */}
      <div>
        <label style={{ fontSize: 13, fontWeight: 500, display: "block", marginBottom: 6 }}>
          Candidate name <span style={{ color: "#a32d2d" }}>*</span>
        </label>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="e.g. Sarah Mitchell"
          style={{ width: "100%", boxSizing: "border-box", fontSize: 14 }}
          onKeyDown={e => e.key === "Enter" && handleSubmit()}
        />
      </div>

      {/* CV upload */}
      <div>
        <label style={{ fontSize: 13, fontWeight: 500, display: "block", marginBottom: 6 }}>
          CV file (PDF) <span style={{ color: "#a32d2d" }}>*</span>
        </label>
        <div
          style={{
            border: cvFile ? "0.5px solid #639922" : "0.5px dashed var(--color-border-secondary)",
            borderRadius: 8, padding: "16px",
            display: "flex", alignItems: "center", gap: 12, cursor: "pointer",
            background: cvFile ? "#eaf3de" : "var(--color-background-secondary)",
            transition: "all 0.15s",
          }}
          onClick={() => document.getElementById("cv-upload").click()}
        >
          <i
            className={`ti ${cvFile ? "ti-file-check" : "ti-file-upload"}`}
            style={{ fontSize: 22, color: cvFile ? "#3b6d11" : "var(--color-text-tertiary)", flexShrink: 0 }}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            {cvFile ? (
              <>
                <div style={{ fontSize: 13, fontWeight: 500, color: "#3b6d11", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {cvFile.name}
                </div>
                <div style={{ fontSize: 11, color: "#639922", marginTop: 2 }}>
                  {(cvFile.size / 1024).toFixed(0)} KB · click to change
                </div>
              </>
            ) : (
              <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                Click to upload PDF
              </span>
            )}
          </div>
          {cvFile && (
            <button
              onClick={e => { e.stopPropagation(); setCvFile(null); document.getElementById("cv-upload").value = ""; }}
              style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "#639922", display: "flex" }}
            >
              <i className="ti ti-x" style={{ fontSize: 16 }} />
            </button>
          )}
        </div>
        <input
          id="cv-upload" type="file" accept=".pdf" style={{ display: "none" }}
          onChange={e => setCvFile(e.target.files[0] || null)}
        />
      </div>

      {status && <Alert type={status.type} message={status.message} />}

      {/* CV section preview after successful ingest */}
      {lastIngested?.sections && (
        <div>
          <p style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)", marginBottom: 6, marginTop: 0, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            How the LLM sectioned this CV
          </p>
          <CvSectionsDrawer sections={lastIngested.sections} />
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{
          padding: "10px 24px", background: "var(--color-text-primary)",
          color: "var(--color-background-primary)", border: "none", borderRadius: 8,
          fontWeight: 500, fontSize: 14, cursor: loading ? "wait" : "pointer",
          opacity: loading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 8,
          justifyContent: "center",
        }}
      >
        {loading
          ? <><i className="ti ti-loader-2" style={{ fontSize: 16 }} /> Extracting &amp; ingesting…</>
          : <><i className="ti ti-cloud-upload" style={{ fontSize: 16 }} /> Ingest candidate</>}
      </button>
    </div>
  );
}

// ── CRM tab ───────────────────────────────────────────────────────────────────
function CrmTab({ refresh }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [status, setStatus] = useState(null);
  const [expandedCrm, setExpandedCrm] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/candidates`);
      const data = await resp.json();
      setCandidates(data.candidates || []);
    } catch (e) {
      setStatus({ type: "error", message: e.message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (refresh > 0) load(); }, [refresh, load]);

  const handleDelete = async (crm_id, name) => {
    if (!window.confirm(`Remove ${name} from the system?`)) return;
    setDeletingId(crm_id);
    try {
      const resp = await fetch(`${API_BASE}/candidate/${encodeURIComponent(crm_id)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error((await resp.json()).detail);
      setStatus({ type: "success", message: `${name} removed.` });
      load();
    } catch (e) {
      setStatus({ type: "error", message: e.message });
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
          {loading ? "Loading…" : `${candidates.length} candidate${candidates.length !== 1 ? "s" : ""} in CRM`}
        </span>
        <button onClick={load} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, background: "none", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 6, padding: "5px 10px", cursor: "pointer", color: "var(--color-text-secondary)" }}>
          <i className="ti ti-refresh" style={{ fontSize: 14 }} /> Refresh
        </button>
      </div>

      {status && <Alert type={status.type} message={status.message} />}

      <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
        {candidates.map(c => {
          const av = getAvatarColor(c.name);
          const isExpanded = expandedCrm === c.crm_id;
          return (
            <div key={c.crm_id} style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: 10, overflow: "hidden",
            }}>
              <div style={{ padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{
                  width: 38, height: 38, borderRadius: "50%", background: av.bg,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontWeight: 500, fontSize: 13, color: av.color, flexShrink: 0,
                }}>{getInitials(c.name)}</div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{ fontWeight: 500, fontSize: 14 }}>{c.name}</span>
                    <Badge text={c.crm_id} />
                    {c.nhs_band && <Badge text={c.nhs_band} />}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                    {c.job_title}
                    {c.location ? ` · ` : ""}
                    {c.location && (
                      <><i className="ti ti-map-pin" style={{ fontSize: 12, verticalAlign: -1 }} /> {c.location}</>
                    )}
                    {c.registration && (
                      <span style={{ marginLeft: 8, color: "var(--color-text-tertiary)" }}>
                        <i className="ti ti-id-badge" style={{ fontSize: 12, verticalAlign: -1 }} /> {c.registration}
                      </span>
                    )}
                  </div>
                </div>

                <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
                  <button
                    onClick={() => setExpandedCrm(isExpanded ? null : c.crm_id)}
                    style={{
                      background: "none", border: "0.5px solid var(--color-border-tertiary)",
                      borderRadius: 6, padding: "4px 8px", cursor: "pointer",
                      color: "var(--color-text-secondary)", fontSize: 12,
                      display: "flex", alignItems: "center", gap: 4,
                    }}
                  >
                    <i className={`ti ${isExpanded ? "ti-chevron-up" : "ti-layout-list"}`} style={{ fontSize: 13 }} />
                    {isExpanded ? "Hide" : "CV"}
                  </button>
                  <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
                    {new Date(c.created_at).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}
                  </span>
                  <button
                    onClick={() => handleDelete(c.crm_id, c.name)}
                    disabled={deletingId === c.crm_id}
                    style={{
                      background: "none", border: "0.5px solid var(--color-border-tertiary)",
                      borderRadius: 6, padding: "4px 8px", cursor: "pointer",
                      color: "#a32d2d", fontSize: 12, display: "flex", alignItems: "center", gap: 4,
                    }}
                  >
                    <i className="ti ti-trash" style={{ fontSize: 13 }} />
                    {deletingId === c.crm_id ? "Removing…" : "Remove"}
                  </button>
                </div>
              </div>

              {/* CV sections accordion inside CRM */}
              {isExpanded && (
                <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 16px", background: "var(--color-background-secondary)" }}>
                  <p style={{ margin: "0 0 8px", fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    LLM CV Sections
                  </p>
                  <CvSectionsDrawer sections={c.sections} />
                </div>
              )}
            </div>
          );
        })}

        {!loading && candidates.length === 0 && (
          <div style={{ textAlign: "center", padding: "48px 0", color: "var(--color-text-tertiary)" }}>
            <i className="ti ti-database-off" style={{ fontSize: 32, display: "block", marginBottom: 8 }} />
            No candidates ingested yet. Use the Ingest tab to add candidates.
          </div>
        )}
      </div>
    </div>
  );
}

// ── App shell ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("search");
  const [refreshCrm, setRefreshCrm] = useState(0);

  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "24px 20px" }}>
      <h2 className="sr-only">Healthcare CV semantic search</h2>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8, background: "#e6f1fb",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <i className="ti ti-stethoscope" style={{ fontSize: 18, color: "#185fa5" }} />
          </div>
          <span style={{ fontWeight: 500, fontSize: 18 }}>Healthcare CV Search</span>
          <span style={{
            fontSize: 12, background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)", borderRadius: 6,
            padding: "2px 8px", color: "var(--color-text-tertiary)",
          }}>v3</span>
        </div>
        <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
          Semantic candidate matching for UK healthcare roles · Milvus vector DB · Gemini AI · 5-section hybrid search
        </p>
      </div>

      <Tabs
        tabs={[
          { id: "search", label: "🔍 Search by JD" },
          { id: "ingest", label: "➕ Ingest candidate" },
          { id: "crm",    label: "🗂 CRM database" },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === "search" && <SearchTab />}
      {tab === "ingest" && <IngestTab onSuccess={() => setRefreshCrm(r => r + 1)} />}
      {tab === "crm"    && <CrmTab refresh={refreshCrm} />}
    </div>
  );
}