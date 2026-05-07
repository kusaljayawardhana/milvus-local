"""
Realistic Milvus Semantic Search Load Test — Healthcare Recruitment
====================================================================
True asymmetric search: realistic nurse CV summaries (written to reflect
actual CV language) are embedded and stored as the candidate database.
Real job description text is embedded and used as queries.

No API key required — CV summaries are hardcoded realistic text.

Requirements:
    pip install pymilvus numpy sentence-transformers

Usage:
    python milvus_healthcare_load_test.py
"""

import time
import random
import statistics
import concurrent.futures

import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_RPS       = 100
DURATION_SECONDS = 60
DB_SIZE          = 50_000
TOP_K            = 10
COLLECTION_NAME  = "healthcare_recruitment_bench"
MILVUS_URI       = "http://localhost:19530"
EMBEDDING_MODEL  = "sentence-transformers/all-mpnet-base-v2"  # dim=768
QUERY_NOISE_STD  = 0.02   # busts Milvus exact-match cache; cosine sim ~0.9998 to base JD

# ── Realistic CV Summaries (Database Side) ─────────────────────────────────────
# Written in third-person recruiter summary style — the linguistic register of
# real CV profiles extracted by an ATS parser. Intentionally different from JD
# language: first/third person mix, employer names, dates, certification years,
# achievement framing. This is what makes JD->CV search genuinely asymmetric.

CV_SUMMARIES = {
    "ICU_CriticalCare": [
        "Experienced Band 6 critical care nurse with seven years in a 20-bed medical-surgical ICU at Manchester University NHS Foundation Trust. Holds CCRN certification and ACLS provider status. Proficient in invasive haemodynamic monitoring, vasopressor titration, and ventilator weaning protocols. Led the implementation of a prone positioning checklist that reduced pressure injury incidence by 30%. Mentors Band 5 staff and acts as shift coordinator.",
        "Senior ICU nurse with four years at Bupa Cromwell Hospital London following three years at Salford Royal. Specialises in post-cardiothoracic surgery care including CABG and valve replacement patients. Competent in IABP management, temporary pacing, and chest drain care. Completed CSC certification 2022. Regularly acts as link nurse for infection control and antimicrobial stewardship.",
        "Band 5 registered nurse with two years in a busy tertiary ICU at Leeds Teaching Hospitals. Gained competencies in arterial line insertion assistance, CVC care, and CRRT circuit management. NMC registered since 2021. Currently undertaking the Arden and Greater East Midlands Critical Care Network post-registration course. Keen interest in early mobilisation and ICU rehabilitation.",
        "Agency critical care nurse with twelve years across multiple Level 3 ICUs in the South East including King's College Hospital and St George's. Extensive experience with ECMO patient care, inhaled nitric oxide, and high-frequency oscillatory ventilation. CCRN and CMC dual certified. Comfortable in both medical and surgical ICU environments. Available for long-block and ad hoc shifts.",
        "Band 7 ICU Sister at Birmingham Queen Elizabeth Hospital with nine years critical care experience. Manages a 28-bed neurosurgical and trauma ICU. Leads nurse-led weaning and extubation protocol. MSc Critical Care Nursing (University of Birmingham, 2020). Active member of BACCN. Involved in Trust-wide sepsis bundle audit and quality improvement projects.",
        "Registered nurse with five years in a combined coronary care and cardiac ICU at Golden Jubilee National Hospital, Glasgow. Skilled in 12-lead ECG interpretation, haemodynamic optimisation post-PCI, and management of patients on milrinone and levosimendan infusions. ACLS and BLS current. Completed cardiac catheter lab rotation 2023.",
        "ICU nurse with three years at Derriford Hospital Plymouth. Competent in standard critical care including ventilator management, sedation and analgesia protocols, and end-of-life care in ICU. Band 5 working towards Band 6. Completed care of the deteriorating patient course and ALERT provider. Strong interest in palliative and comfort-focused ICU care.",
        "Highly experienced senior staff nurse with eight years at Royal Infirmary of Edinburgh ICU. Expertise in traumatic brain injury management, ICP monitoring, and targeted temperature management post-cardiac arrest. Certified in neurocritical care. Participated in Scottish Intensive Care Society audit programme. Trains junior staff on ICP waveform interpretation.",
    ],

    "AE_Emergency": [
        "Emergency Nurse Practitioner with six years in a major trauma centre at St Mary's Hospital London. Holds CEN and TNCC certifications and an independent prescribing qualification. Experienced in primary and secondary trauma surveys, FAST ultrasound, RSI assistance, and haemorrhage control. Manages a caseload of minors and majors autonomously. Led departmental triage redesign reducing four-hour breach by 18%.",
        "Band 6 emergency nurse with five years at Leeds General Infirmary ED seeing 160 patients daily. Triages across all acuity levels, manages resuscitation bay, and supports medical students on clinical placement. ACLS and ATLS provider. Completed non-medical prescribing 2022. Special interest in frailty and acute presentations in older adults.",
        "Newly qualified Band 5 RN with 12-month ED rotation at Sheffield Teaching Hospitals. Confident in IV cannulation, blood cultures, ECG recording, plaster casting, and wound closure. Completed clinical skills competency pack in year one. ACLS enrolled. Seeking substantive ED post with structured Band 5 development programme.",
        "Agency emergency nurse with ten years across multiple busy EDs including Royal London and Nottingham University Hospitals. Comfortable in triage, majors, minors, and resus. Paediatric emergency experience including febrile convulsions and anaphylaxis management. ENPC and TNCC certified. Reliable, adaptable, and accustomed to high-pressure environments.",
        "Band 7 Emergency Department Sister at University Hospital Coventry and Warwickshire. Oversees nursing operations across a 45-bay ED. Leads the department's mental health liaison pathway and flow improvement group. MSc Advanced Clinical Practice. Independent prescriber. Active RCEM nursing faculty member. Involved in national 111 clinical advice hub pilot.",
        "Experienced ED nurse with seven years at Addenbrooke's Hospital Cambridge. Specialises in toxicology and overdose presentations, sepsis identification, and rapid assessment of chest pain pathways. ACLS current. Completed BASICS pre-hospital care course and participates in HEMS observer shifts. Interest in wilderness and expedition medicine.",
        "Band 5 RN with three years emergency experience at Royal Victoria Hospital Belfast. Rotating across all ED areas. Competent in paediatric and adult triage, rapid assessment, and supporting patients in mental health crisis. BLS and paediatric first aid certified. Currently studying for CEN examination.",
        "Senior staff nurse with eight years in a combined ED and urgent treatment centre at Bristol Royal Infirmary. Experienced in streaming, see-and-treat, and managing ambulance handover delays. Qualified Nurse Prescriber. Involved in development of the department's sepsis six compliance audit. Speaks fluent Polish — valuable for Eastern European patient cohort.",
    ],

    "Perioperative_Theatre": [
        "Experienced scrub nurse with seven years in orthopaedic and spinal theatre at Robert Jones and Agnes Hunt Orthopaedic Hospital, Oswestry. Competent in THR, TKR, spinal fusion, and complex revision arthroplasty. Manages specialist implant sets and liaises directly with industry representatives for new instrumentation. CNOR certified. Leads new-starter scrub induction programme.",
        "Band 6 perioperative practitioner at Nuffield Health Oxford with five years in general, laparoscopic, and colorectal surgery. Experienced as both scrub and circulating nurse. Proficient in robotic-assisted surgery support (da Vinci). WHO surgical safety checklist champion. Completed perioperative care MSc module 2022.",
        "ODP with six years in cardiac and thoracic theatres at Papworth Hospital. Skilled in cardiopulmonary bypass preparation, aortic valve scrub, and thoracoscopic procedure support. HCPC registered. Experienced working alongside perfusionists and cardiac anaesthetists in complex cases. Completed advanced airway management study day.",
        "Band 5 theatre nurse with two years post-qualification experience at University College London Hospitals. Rotating across trauma, plastics, and urology lists. Competent in sterile field maintenance, instrument counts, and specimen handling. Currently undertaking 180-credit perioperative care programme at London South Bank University.",
        "Senior scrub practitioner with nine years in ophthalmic and ENT surgery at Moorfields Eye Hospital and secondment at Great Ormond Street. Meticulous attention to detail in microsurgical environments. Experienced in phacoemulsification, corneal transplant, and cochlear implant procedures. Mentors junior scrub staff and conducts competency sign-offs.",
        "Perioperative Band 6 at Sheffield Teaching Hospitals with four years in gynaecology and obstetric theatres. Experienced in LSCS, hysterectomy, and laparoscopic gynaecology including endometriosis excision. Emergency theatre experience including ruptured ectopic and postpartum haemorrhage. Completed advanced scrub practitioner course.",
        "Theatre practitioner with eight years across private sector hospitals including Spire Leeds and BMI The Priory. Highly flexible across general surgery, urology, and ENT. Comfortable with all aspects of perioperative care from pre-assessment through to recovery handover. NMC registered. Keen interest in day surgery and enhanced recovery after surgery (ERAS) pathways.",
        "CNOR-certified scrub nurse with eleven years at King's College Hospital in hepatobiliary and transplant surgery. Extensive experience with liver resection, Whipple's procedure, and renal transplant instrumentation. Accustomed to long and complex cases. Involved in theatre team WHO checklist compliance audit and debrief culture improvement.",
    ],

    "Pediatrics_NICU": [
        "Neonatal intensive care nurse with six years at Liverpool Women's Hospital Level 3 NICU. Provides 1:1 care for extremely preterm infants from 23 weeks gestation. Competent in HFOV, conventional ventilation, and inhaled nitric oxide. Manages PICC lines and umbilical arterial catheters. RNC-NIC certified 2021. NRP provider. Champions developmental care and family-integrated care model.",
        "Band 6 paediatric nurse with five years in a 28-bed general paediatric ward at Great North Children's Hospital Newcastle. Specialises in respiratory and infectious disease presentations in children 0–16. Experienced in NGT feeding, blood glucose monitoring, IV medication administration using smart pumps, and supporting children through procedural anxiety. PALS provider.",
        "NICU nurse with three years at Simpson Centre for Reproductive Health Edinburgh. Cares for surgical neonates post-cardiac and abdominal surgery. Experienced in post-operative pain assessment, drain management, and supporting families through complex diagnoses including CHD and TOF repair. Neonatal surgery pathway link nurse.",
        "Paediatric oncology nurse with four years at Royal Manchester Children's Hospital. Manages central venous access, chemotherapy administration, and bone marrow transplant nursing care. Experienced in neutropenic sepsis management and palliative symptom control in children. CPHON studying. Involved in bereavement support group facilitation.",
        "Band 5 children's nurse with two years at Bristol Royal Hospital for Children. Rotating across general surgery, neurology, and acute assessment unit. Competent in paediatric assessment, PEWS documentation, fluid balance, and family-centred care. PALS certified. Child Branch BSc (Hons) from University of the West of England.",
        "Senior NICU nurse with nine years at St Thomas' Hospital London, including three years as shift leader in a 40-cot Level 3 unit. Expertise in complex ventilation strategies, therapeutic hypothermia for HIE, and management of NEC. MSc Advanced Neonatal Practice. Teaches neonatal resuscitation on regional NRP instructor course.",
        "Community children's nurse with five years at Birmingham Children's Hospital community team. Manages technology-dependent children at home including those on home ventilation, enteral feeding, and subcutaneous medication infusions. Experienced in training parents and carers on complex clinical procedures. Strong liaison with GP, school nursing, and social care.",
        "Paediatric emergency nurse with four years at Alder Hey Children's Hospital ED. Experienced in triage of paediatric presentations including febrile illness, head injury, and mental health crisis in adolescents. PALS and EPALS provider. Supports junior nursing staff in paediatric assessment. Involved in sepsis pathway audit.",
    ],

    "Oncology_Haematology": [
        "Oncology CNS with seven years at The Christie NHS Foundation Trust Manchester. Key worker for lung cancer patients receiving immunotherapy and chemotherapy combinations. Administers SACT via peripheral and central access, monitors for immune-related adverse events, and coordinates urgent oncology reviews. OCN certified. Non-medical prescribing qualification 2021. Involved in clinical trial coordination.",
        "Haematology nurse specialist with five years at King's College Hospital. Manages patients through bone marrow transplant pathway including conditioning chemotherapy, stem cell infusion, and GVHD monitoring. BMTCN certified. Experienced in managing neutropenic sepsis, mucositis, and transfusion reactions. Leads nurse-led transplant follow-up clinic.",
        "Band 5 oncology staff nurse with three years at Clatterbridge Cancer Centre. Administers systemic anti-cancer therapy including targeted therapies and immunotherapy. Competent in port-a-cath access, CVAD care, and extravasation management. Completed SACT competency programme. Interest in oncology rehabilitation and survivorship care.",
        "Senior chemotherapy nurse with eight years across NHS and private oncology settings including BUPA Cromwell and Royal Marsden. Experienced in breast, colorectal, and haematological malignancy treatment pathways. PICC line insertion trained. Completed advanced communication skills training for breaking bad news. Mentors newly qualified oncology nurses.",
        "Oncology CNS – Haematology at Sheffield Teaching Hospitals with six years experience. Supports patients with AML, CLL, and lymphoma through active treatment and remission monitoring. Runs nurse-led hydroxycarbamide and thalidomide monitoring clinics. Prescribing qualification. Participates in departmental clinical audit on neutropenic sepsis bundle compliance.",
        "Macmillan palliative oncology nurse with four years at Velindre Cancer Centre Cardiff. Provides holistic needs assessments, symptom management advice, and advance care planning support across oncology wards and outpatient clinics. Experience with syringe driver management, lymphoedema assessment, and coordinating discharge to hospice.",
        "Band 6 chemotherapy nurse at Ipswich Hospital with five years oncology experience. Responsible for pre-treatment checks, patient education, SACT administration, and acute toxicity management. Involved in implementation of electronic prescribing system for SACT. Completed clinical leadership course and chairs monthly oncology nursing forum.",
        "Teenage and Young Adult (TYA) oncology nurse with three years at University College London Hospitals. Provides age-appropriate psychosocial support and clinical care to patients aged 16–24 during cancer treatment. Experienced in fertility preservation counselling referral and managing treatment side effects in younger adults. TYA specialist course completed.",
    ],

    "Cardiology_CathLab": [
        "Cardiac catheter lab nurse with six years at Royal Brompton Hospital London. Experienced in primary PCI for STEMI, elective coronary angiography, TAVI, and complex EP studies including AF ablation. Scrubs and circulates for all interventional procedures. Competent in sheath removal, closure device application, and post-procedure monitoring. RCIS certification in progress.",
        "Band 6 cardiology nurse with five years on a 24-bed coronary care unit at Northern General Hospital Sheffield. Expert in continuous telemetry monitoring, 12-lead ECG interpretation, management of arrhythmias, and post-cardiac catheterisation care. Completed CCRN-CMC 2022. Runs nurse-led cardiac rehabilitation education sessions.",
        "Electrophysiology lab nurse with four years at St Bartholomew's Hospital London. Supports complex ablation procedures including AF, VT, and SVT. Competent in managing conscious sedation, haemodynamic monitoring during EP mapping, and device check clinics for pacemakers and ICDs. Completed arrhythmia nursing course.",
        "Community cardiac nurse with seven years at Harrogate and District NHS Trust. Manages heart failure caseload of 150 patients, conducting home visits and telephone review clinics. Optimises GDMT including uptitration of ACE inhibitors, beta-blockers, and SGLT2 inhibitors. Non-medical prescribing qualification. Involved in national heart failure audit.",
        "Senior cath lab practitioner with nine years at Golden Jubilee National Hospital Glasgow. Experienced in structural heart interventions including MitraClip, LAAO, and TAVI. Manages IVUS, FFR, and rotational atherectomy support. Acts as lead practitioner for complex PCI lists. Trains junior staff in radiation safety and sterile technique.",
        "Band 5 cardiac nurse with two years at Wythenshawe Hospital Manchester cardiothoracic ward. Cares for post-CABG, valve replacement, and LVAD patients. Competent in chest drain management, epicardial pacing wire care, and telemetry monitoring. ACLS current. Keen interest in transitioning to cath lab nursing.",
        "Cardiac nurse specialist with eight years in heart failure and transplant at Freeman Hospital Newcastle. Manages advanced heart failure patients on IV inotropes, VAD outpatient follow-up, and bridge-to-transplant pathways. Prescribing qualified. Runs outpatient VAD clinic and contributes to INTERMACS registry data entry.",
        "Interventional cardiology scrub nurse with five years at Leeds General Infirmary. Routinely scrubs for primary PCI, rotablation, and bifurcation stenting. Competent in operating contrast injectors, pressure wire systems, and intravascular imaging. Involved in door-to-balloon time improvement project achieving sub-60 minute median.",
    ],

    "MentalHealth_Psych": [
        "Community mental health nurse with six years on an Early Intervention in Psychosis team in Manchester. Manages a caseload of 25 service users aged 14–35 with first-episode psychosis. Conducts mental state examinations, HoNOS assessments, and structured risk formulations. Administers paliperidone and aripiprazole depot injections. CBT-p trained. Non-medical prescribing qualification 2022.",
        "Inpatient psychiatric nurse with five years on an acute adult ward at Bethlem Royal Hospital. Experienced in de-escalation, PMVA, and managing patients under the Mental Health Act 1983. Skilled in rapid tranquillisation protocols and post-incident debriefing. Completed Cognitive Behavioural Therapy for Psychosis awareness course. Section 12 Approved.",
        "Forensic mental health nurse with seven years at a medium secure unit in the East Midlands. Manages patients with comorbid psychosis and personality disorder. Competent in dynamic risk assessment, therapeutic boundary maintenance, and multi-agency public protection arrangements. Completed Violence Risk Scale training. Involved in tribunal report preparation.",
        "CMHT band 5 RMN with three years on an older adults community team. Conducts memory assessments, carer support reviews, and crisis de-escalation visits. Experienced in coordinating dementia care packages and supporting families through behavioural and psychological symptoms of dementia. Completed dementia care mapping training.",
        "Perinatal mental health nurse with four years at a specialist mother and baby unit in Birmingham. Supports mothers with postpartum psychosis, severe depression, and anxiety during the perinatal period. Experienced in infant-mother bonding interventions, medication management in breastfeeding mothers, and safeguarding referrals. Completed SIGN perinatal mental health guidelines training.",
        "Crisis resolution and home treatment nurse with five years at Pennine Care NHS Foundation Trust. Provides intensive community support as alternative to admission. Conducts mental state and risk assessments. Experienced in supporting patients through suicidal crisis, self-harm, and acute psychosis episodes. DBT-informed practice. Drives own vehicle for home visits.",
        "Band 6 RMN with eight years in dual diagnosis services supporting patients with co-occurring mental health and substance use disorders. Experienced in motivational interviewing, structured relapse prevention, and naloxone distribution training. Works closely with drug and alcohol services, probation, and housing. Completed DANOS framework training.",
        "Learning disabilities nurse with six years in a community LD team in North Yorkshire. Supports adults with complex needs including autism, challenging behaviour, and epilepsy management. Experienced in positive behaviour support planning, capacity assessment under MCA, and health action plan facilitation. Completed PBS practitioner training.",
    ],

    "Community_District": [
        "District nurse team leader with eight years at Norfolk Community Health and Care NHS Trust. Manages a caseload of 90 housebound patients and supervises a team of five Band 5 nurses and two HCAs. Expert in complex wound management including Doppler ABPI, compression therapy, and VAC dressing application. Non-medical prescribing qualification. SystmOne superuser.",
        "Community nurse with five years at Leicestershire Partnership NHS Trust. Manages patients with long-term conditions including heart failure, COPD, and type 2 diabetes. Conducts structured medication reviews, self-management education, and coordinates care with GP practices and social services. Competent in subcutaneous fluid administration and syringe driver setup.",
        "Specialist community nurse – tissue viability with six years at Leeds Community Healthcare. Assesses and manages complex wounds across community settings including leg ulcers, pressure injuries, and surgical wounds. Provides specialist advice to district nursing teams and care homes. WCET certified. Involved in community pressure injury prevention campaign.",
        "End-of-life care community nurse with seven years at St Helena Hospice community team in Essex. Manages complex symptom control in the last weeks of life including syringe driver initiation, anticipatory prescribing coordination, and Liverpool Care Pathway documentation. Non-medical prescriber. Supports bereaved families and conducts after-death care in community settings.",
        "Band 5 district nurse with three years at Solent NHS Trust Southampton. Provides wound care, catheter management, insulin administration, and post-discharge monitoring to housebound patients. Competent in venepuncture, ECG recording at home, and falls risk assessment. Full UK driving licence. Enjoys the autonomy and variety of community nursing.",
        "Community matron with ten years experience at Salford Royal community services. Manages a high-intensity caseload of patients with multiple long-term conditions and frequent hospital admissions. Conducts comprehensive geriatric assessments and implements personalised care plans. Independent prescriber. Works with integrated neighbourhood teams to reduce unplanned admissions.",
        "Practice nurse with six years at a large urban GP surgery in Birmingham. Runs chronic disease management clinics for asthma, COPD, hypertension, and diabetes. Delivers travel health consultations, childhood immunisations, and cervical screening. Completed diabetes specialist nursing certificate. QOF lead for the practice. Experienced in minor illness management.",
        "School nurse with four years at a community trust covering secondary schools in Surrey. Provides health assessments, emotional wellbeing support, sexual health advice, and immunisation programmes to young people aged 11–19. Experienced in child protection referrals and working with CAMHS. Completed safeguarding children Level 3. Mentors student nurses on placement.",
    ],
}

# ── Real Job Descriptions (Query Side) ────────────────────────────────────────
JOB_DESCRIPTIONS = {
    "ICU_CriticalCare": [
        "Band 6 Senior Staff Nurse – Intensive Care Unit, NHS Foundation Trust Manchester. Manage ventilated patients in 16-bed medical/surgical ICU. Haemodynamic monitoring via arterial lines and CVCs. Initiate vasopressors per sepsis protocol. Supervise Band 5 nurses. CCRN preferred. Minimum 2 years ICU/HDU. ACLS and BLS required.",
        "Critical Care Nurse Specialist – Cardiac ICU, Private Hospital London. Post-cardiac surgery care: CABG, valve replacement, LVAD insertion. Haemodynamic optimisation, temporary pacemaker management, IABP support, early extubation protocols. CCRN-CMC preferred. 3 years cardiac critical care.",
    ],
    "AE_Emergency": [
        "Emergency Nurse Practitioner – Major Trauma Centre Birmingham. Autonomous ENP in ED seeing 180+ patients daily. ATLS trauma surveys, RSI, FAST ultrasound, haemorrhage control. CEN or TNCC. Independent prescribing. 4 years emergency nursing. ACLS/ATLS provider.",
        "Staff Nurse – Emergency Department Band 5, District General Hospital Leeds. Rotate triage, majors, minors, resus. IV cannulation, ECG, wound closure, mental health crisis support. ACLS desirable. Newly qualified welcome.",
    ],
    "Perioperative_Theatre": [
        "Scrub Practitioner – Orthopaedic and Spinal Theatre Bristol. THR, TKR, spinal fusion, revision arthroplasty. Sterile field, instrument and implant counts, WHO checklist. CNOR preferred. 2 years scrub experience.",
    ],
    "Pediatrics_NICU": [
        "Neonatal Intensive Care Nurse – Level 3 NICU Edinburgh. 1:1 care for ventilated neonates, HFOV, inhaled nitric oxide, PICC lines, umbilical catheters, phototherapy, developmental care. RNC-NIC. NRP certified.",
        "Paediatric Staff Nurse – General Ward Band 5, Cardiff NHS Trust. Children 0–16, medical and surgical. Paediatric dosing, NGT management, blood glucose monitoring, family-centred care. PALS desirable. Child Branch degree.",
    ],
    "Oncology_Haematology": [
        "Oncology CNS – Breast Cancer, Cancer Centre London. SACT via port-a-cath, neutropenic sepsis, CINV management, MDT coordination. OCN certified. Non-medical prescribing preferred. 3 years oncology.",
    ],
    "Cardiology_CathLab": [
        "Cardiac Catheterisation Laboratory Nurse, Tertiary Cardiac Centre Glasgow. Primary PCI, TAVI, EP studies. Scrub and circulate, haemodynamic monitoring, arrhythmia response, sheath and closure device management, IVUS. RCIS. ACLS.",
    ],
    "MentalHealth_Psych": [
        "Community Mental Health Nurse – Early Intervention in Psychosis, Sheffield CMHT. First-episode psychosis aged 14–35. Mental state examinations, HoNOS risk assessment, depot antipsychotic administration, EPSE monitoring, CBT-p support. RMN registered.",
    ],
    "Community_District": [
        "District Nurse Team Leader – Community Health, NHS Community Trust Norfolk. Complex home care: wound care, Doppler ABPI, compression therapy, catheter management, end-of-life care, COPD, heart failure, diabetes. Non-medical prescribing. Community nursing SPQ desirable. Full UK driving licence.",
    ],
}

SPECIALTIES = [
    {"name": "ICU_CriticalCare",      "weight": 0.20, "cv_noise": 0.06},
    {"name": "AE_Emergency",          "weight": 0.18, "cv_noise": 0.06},
    {"name": "Perioperative_Theatre", "weight": 0.15, "cv_noise": 0.05},
    {"name": "Pediatrics_NICU",       "weight": 0.12, "cv_noise": 0.05},
    {"name": "Oncology_Haematology",  "weight": 0.10, "cv_noise": 0.06},
    {"name": "Cardiology_CathLab",    "weight": 0.10, "cv_noise": 0.06},
    {"name": "MentalHealth_Psych",    "weight": 0.08, "cv_noise": 0.07},
    {"name": "Community_District",    "weight": 0.07, "cv_noise": 0.08},
]
assert abs(sum(s["weight"] for s in SPECIALTIES) - 1.0) < 1e-6

# ── Load Embedding Model ───────────────────────────────────────────────────────
print(f"Loading embedding model: {EMBEDDING_MODEL}")
print("(First run downloads ~420MB — cached afterwards)\n")
model = SentenceTransformer(EMBEDDING_MODEL)
DIMENSION = model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {DIMENSION}\n")

# ── Embed CV Summaries ─────────────────────────────────────────────────────────
print("=" * 72)
print("  PHASE 1: Embedding CV summaries")
print("=" * 72)

all_cv_texts = []
cv_spec_map  = []  # parallel list: which SPECIALTIES index each CV belongs to

for spec_idx, spec in enumerate(SPECIALTIES):
    for cv_text in CV_SUMMARIES[spec["name"]]:
        all_cv_texts.append(cv_text.strip())
        cv_spec_map.append(spec_idx)

print(f"Embedding {len(all_cv_texts)} CV summaries...")
cv_embeddings = model.encode(
    all_cv_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32,
).astype(np.float32)
print(f"CV embeddings shape: {cv_embeddings.shape}\n")

# Attach CV pool to each specialty
for spec_idx, spec in enumerate(SPECIALTIES):
    indices = [i for i, s in enumerate(cv_spec_map) if s == spec_idx]
    spec["cv_pool"] = cv_embeddings[indices]

# ── Embed Job Descriptions ─────────────────────────────────────────────────────
print("=" * 72)
print("  PHASE 2: Embedding job descriptions (query vectors)")
print("=" * 72)

flat_jd_texts  = []
jd_spec_labels = []
for spec in SPECIALTIES:
    for jd_text in JOB_DESCRIPTIONS[spec["name"]]:
        flat_jd_texts.append(jd_text.strip())
        jd_spec_labels.append(spec["name"])

print(f"Embedding {len(flat_jd_texts)} job descriptions...")
jd_embeddings = model.encode(
    flat_jd_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32,
).astype(np.float32)
print(f"JD embeddings shape: {jd_embeddings.shape}\n")

# Attach JD pool to each specialty
for spec in SPECIALTIES:
    spec["jd_pool"] = np.array([
        jd_embeddings[i]
        for i, l in enumerate(jd_spec_labels)
        if l == spec["name"]
    ], dtype=np.float32)

# ── Sanity Check ───────────────────────────────────────────────────────────────
print("JD->CV cosine similarity (within specialty should exceed cross-specialty):")
for spec in SPECIALTIES:
    jd_vec = spec["jd_pool"][0]
    within = float(np.mean([np.dot(jd_vec, cv) for cv in spec["cv_pool"]]))
    other  = random.choice([s for s in SPECIALTIES if s["name"] != spec["name"]])
    cross  = float(np.mean([np.dot(jd_vec, cv) for cv in other["cv_pool"]]))
    marker = "✓" if within > cross else "✗"
    print(f"  {marker} {spec['name']:30s}  within: {within:.4f}  "
          f"cross ({other['name'][:20]}): {cross:.4f}")
print()

# ── Connect & Populate Milvus ──────────────────────────────────────────────────
print("=" * 72)
print("  PHASE 3: Populating Milvus")
print("=" * 72)
print(f"Connecting to {MILVUS_URI}...")
client = MilvusClient(uri=MILVUS_URI)

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=DIMENSION,
    metric_type="IP",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    },
)
print("Collection created (HNSW M=16, efConstruction=200, IP/cosine).\n")

# ── Heterogeneous DB Vector Generation ────────────────────────────────────────
# Real CV databases are NOT 8 clean blobs. They contain:
#   ~65% pure specialists    — tight around one specialty anchor
#   ~20% dual-trained nurses — interpolated between two related specialties
#                              (e.g. ICU+Cardiology, AE+Perioperative)
#   ~10% generalists         — broad spread across all anchors
#   ~5%  outliers            — unusual career paths, wide noise
#
# This creates an irregular, lumpy vector space that forces HNSW to do
# genuinely varied graph traversal throughout the test — no warm-up bias.

# Related specialty pairs for dual-trained interpolation
DUAL_TRAINED_PAIRS = [
    ("ICU_CriticalCare",      "Cardiology_CathLab"),
    ("ICU_CriticalCare",      "AE_Emergency"),
    ("AE_Emergency",          "Perioperative_Theatre"),
    ("Pediatrics_NICU",       "ICU_CriticalCare"),
    ("Oncology_Haematology",  "Community_District"),
    ("MentalHealth_Psych",    "Community_District"),
]

spec_by_name = {s["name"]: s for s in SPECIALTIES}

def pick_cv_anchor(spec: dict) -> np.ndarray:
    """Pick a random real CV embedding from a specialty pool."""
    return spec["cv_pool"][random.randint(0, len(spec["cv_pool"]) - 1)]

def make_db_vector_heterogeneous(record_idx: int) -> list:
    """
    Generate one DB record using a realistic mix of candidate archetypes.

    Distribution (by record_idx bucket to guarantee proportions):
      0–64%  : pure specialist  — perturb single specialty CV anchor
      65–84% : dual-trained     — interpolate between two related specialty anchors
      85–94% : generalist       — weighted mix of 3–5 random specialty anchors
      95–99% : outlier          — pure specialist with 3× noise (unusual career path)
    """
    roll = record_idx % 100  # deterministic bucket so proportions are exact

    if roll < 65:
        # Pure specialist
        spec = random.choices(SPECIALTIES, weights=[s["weight"] for s in SPECIALTIES])[0]
        base = pick_cv_anchor(spec)
        noise_std = spec["cv_noise"]

    elif roll < 85:
        # Dual-trained: interpolate between two related specialty anchors
        pair_names = random.choice(DUAL_TRAINED_PAIRS)
        spec_a = spec_by_name[pair_names[0]]
        spec_b = spec_by_name[pair_names[1]]
        alpha  = random.uniform(0.3, 0.7)   # blend ratio
        base   = alpha * pick_cv_anchor(spec_a) + (1 - alpha) * pick_cv_anchor(spec_b)
        base  /= np.linalg.norm(base)
        noise_std = max(spec_a["cv_noise"], spec_b["cv_noise"]) * 1.2

    elif roll < 95:
        # Generalist: weighted centroid of 3–5 random specialties
        k      = random.randint(3, 5)
        chosen = random.choices(SPECIALTIES, k=k)
        base   = np.mean([pick_cv_anchor(s) for s in chosen], axis=0)
        base  /= np.linalg.norm(base)
        noise_std = 0.12  # wider spread — generalists sit between clusters

    else:
        # Outlier: unusual career path, much higher noise
        spec = random.choice(SPECIALTIES)
        base = pick_cv_anchor(spec)
        noise_std = spec["cv_noise"] * 3.0

    noise = np.random.standard_normal(base.shape).astype(np.float32) * noise_std
    v = base.astype(np.float32) + noise
    return (v / np.linalg.norm(v)).tolist()

print(f"Generating {DB_SIZE:,} candidate records with heterogeneous distribution:")
print(f"  65% pure specialists | 20% dual-trained | 10% generalists | 5% outliers")

BATCH_SIZE = 5_000
for batch_start in range(0, DB_SIZE, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, DB_SIZE)
    records = [
        {"id": batch_start + i, "vector": make_db_vector_heterogeneous(batch_start + i)}
        for i in range(batch_end - batch_start)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=records)
    print(f"  Inserted {batch_end:,} / {DB_SIZE:,}")

print("\nWaiting 3 seconds for HNSW index to flush...")
time.sleep(25)

# ── Load Test ──────────────────────────────────────────────────────────────────
def make_query_vector(jd_base: np.ndarray) -> list:
    """Unique query per call — tiny noise busts Milvus cache, semantics unchanged."""
    noise = np.random.standard_normal(jd_base.shape).astype(np.float32) * QUERY_NOISE_STD
    v = jd_base + noise
    return (v / np.linalg.norm(v)).tolist()

def perform_search(jd_base: np.ndarray) -> float:
    query_vector = make_query_vector(jd_base)
    t0 = time.perf_counter()
    client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=TOP_K,
        search_params={"metric_type": "IP", "params": {"ef": 64}},
    )
    return (time.perf_counter() - t0) * 1000  # ms

weights = [s["weight"] for s in SPECIALTIES]

def build_slot_bases() -> list:
    """Assign each RPS slot to a JD embedding, weighted by specialty workforce share."""
    return [
        spec["jd_pool"][random.randint(0, len(spec["jd_pool"]) - 1)]
        for spec in random.choices(SPECIALTIES, weights=weights, k=TARGET_RPS)
    ]

print(f"\n{'='*72}")
print(f"  PHASE 4: LOAD TEST  {TARGET_RPS} RPS | {DURATION_SECONDS}s | {DB_SIZE:,} candidates | top-{TOP_K}")
print(f"  DB  : perturbed real CV embeddings (8 specialties, authentic language)")
print(f"  Query: JD embeddings + unique noise per call (zero cache hits)")
print(f"{'='*72}\n")

slot_jd_bases = build_slot_bases()
all_latencies = []
successful    = 0

for current_second in range(DURATION_SECONDS):
    loop_start = time.time()

    if current_second % 10 == 0 and current_second > 0:
        slot_jd_bases = build_slot_bases()

    with concurrent.futures.ThreadPoolExecutor(max_workers=TARGET_RPS) as executor:
        latencies = list(executor.map(perform_search, slot_jd_bases))

    successful    += TARGET_RPS
    all_latencies.extend(latencies)
    elapsed = time.time() - loop_start

    if current_second % 5 == 0:
        lat_s = sorted(latencies)
        p50   = statistics.median(lat_s)
        p95   = lat_s[int(len(lat_s) * 0.95)]
        p99   = lat_s[int(len(lat_s) * 0.99)]
        print(
            f"[{current_second:>3}s]  RPS: {TARGET_RPS / elapsed:>6.1f}  |  "
            f"p50: {p50:>6.1f}ms  p95: {p95:>6.1f}ms  p99: {p99:>6.1f}ms  |  "
            f"wall: {elapsed:.3f}s"
        )

    time.sleep(max(0, 1.0 - elapsed))

# ── Summary ────────────────────────────────────────────────────────────────────
n = len(all_latencies)
s = sorted(all_latencies)
print(f"\n{'='*72}")
print(f"  FINAL RESULTS — {successful:,} searches over {DURATION_SECONDS}s")
print(f"{'='*72}")
print(f"  mean          : {statistics.mean(s):>8.2f} ms")
print(f"  p50  (median) : {statistics.median(s):>8.2f} ms")
print(f"  p95           : {s[int(n * 0.95)]:>8.2f} ms")
print(f"  p99           : {s[int(n * 0.99)]:>8.2f} ms")
print(f"  p99.9         : {s[int(n * 0.999)]:>8.2f} ms")
print(f"  max           : {max(s):>8.2f} ms")
print(f"{'='*72}\n")

print("Cleaning up...")
client.drop_collection(COLLECTION_NAME)
print("Done.")
