"""
seed_data.py — Populates the system with realistic UK healthcare candidate profiles.
Run this once after the server is up: python seed_data.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

CANDIDATES = [
    {
        "crm_id": "CRM-001",
        "name": "Sarah Mitchell",
        "email": "sarah.mitchell@example.com",
        "phone": "07700 900001",
        "location": "Manchester, Greater Manchester",
        "job_title": "Senior Staff Nurse",
        "nhs_band": "Band 6",
        "years_exp": 9,
        "specialisms": ["Adult Critical Care", "ICU", "Ventilator Management", "Sepsis Protocol"],
        "availability": "Immediately available",
        "salary_exp": "£35,000 - £42,000",
        "registration": "NMC PIN: 12A3456B",
        "notes": "CCRN qualified, experienced in ECMO support, mentored 12 Band 5 nurses. ITU charge nurse experience."
    },
    {
        "crm_id": "CRM-002",
        "name": "James Okonkwo",
        "email": "james.okonkwo@example.com",
        "phone": "07700 900002",
        "location": "Birmingham, West Midlands",
        "job_title": "Paramedic Specialist Practitioner",
        "nhs_band": "Band 7",
        "years_exp": 14,
        "specialisms": ["Emergency Pre-hospital Care", "Advanced Life Support", "Trauma", "HEMS"],
        "availability": "4 weeks notice",
        "salary_exp": "£43,000 - £50,000",
        "registration": "HCPC PA12345",
        "notes": "HEMS paramedic for 6 years, PHEM training, MERIT team experience, clinical educator."
    },
    {
        "crm_id": "CRM-003",
        "name": "Dr. Priya Sharma",
        "email": "priya.sharma@example.com",
        "phone": "07700 900003",
        "location": "London, South East",
        "job_title": "Specialty Doctor - Psychiatry",
        "nhs_band": "SAS Grade",
        "years_exp": 11,
        "specialisms": ["Adult Psychiatry", "CAMHS", "Forensic Psychiatry", "Clozapine Clinic"],
        "availability": "3 months notice",
        "salary_exp": "£52,000 - £68,000",
        "registration": "GMC: 7654321",
        "notes": "MRCPsych Part 1 & 2 passed, experienced in Section 12 assessments, CPA lead, dual diagnosis specialist."
    },
    {
        "crm_id": "CRM-004",
        "name": "Thomas Hughes",
        "email": "thomas.hughes@example.com",
        "phone": "07700 900004",
        "location": "Leeds, West Yorkshire",
        "job_title": "Physiotherapist",
        "nhs_band": "Band 6",
        "years_exp": 7,
        "specialisms": ["Musculoskeletal", "Sports Rehabilitation", "Orthopaedics", "Hydrotherapy"],
        "availability": "1 month notice",
        "salary_exp": "£33,000 - £40,000",
        "registration": "HCPC PH76543",
        "notes": "Extended scope practitioner in MSK, first contact practitioner in GP setting, MCSP, injection therapy trained."
    },
    {
        "crm_id": "CRM-005",
        "name": "Fatima Al-Rashid",
        "email": "fatima.alrashid@example.com",
        "phone": "07700 900005",
        "location": "Bristol, Avon",
        "job_title": "Community District Nurse",
        "nhs_band": "Band 6",
        "years_exp": 8,
        "specialisms": ["District Nursing", "Wound Care", "Palliative Care", "Diabetes Management"],
        "availability": "Immediately available",
        "salary_exp": "£34,000 - £41,000",
        "registration": "NMC PIN: 22B7654C",
        "notes": "V300 prescriber, tissue viability lead for community team, end-of-life care champion, EMIS Web proficient."
    },
    {
        "crm_id": "CRM-006",
        "name": "Dr. Mark Stevenson",
        "email": "mark.stevenson@example.com",
        "phone": "07700 900006",
        "location": "Edinburgh, Scotland",
        "job_title": "Consultant Radiologist",
        "nhs_band": "Consultant",
        "years_exp": 18,
        "specialisms": ["Interventional Radiology", "CT/MRI Reporting", "Breast Imaging", "Vascular IR"],
        "availability": "6 months notice",
        "salary_exp": "£93,000 - £126,000",
        "registration": "GMC: 4523678",
        "notes": "FRCR, subspecialty in interventional oncology, CIRSE faculty member, MDT lead for vascular radiology."
    },
    {
        "crm_id": "CRM-007",
        "name": "Angela Chen",
        "email": "angela.chen@example.com",
        "phone": "07700 900007",
        "location": "Newcastle upon Tyne, Tyne and Wear",
        "job_title": "Occupational Therapist",
        "nhs_band": "Band 5",
        "years_exp": 3,
        "specialisms": ["Acute Medicine", "Stroke Rehabilitation", "Cognitive Assessment", "Discharge Planning"],
        "availability": "Immediately available",
        "salary_exp": "£27,000 - £32,000",
        "registration": "HCPC OT98765",
        "notes": "Newly qualified with strong stroke rehab placement, trained in FIM/FAM assessments, keen to develop."
    },
    {
        "crm_id": "CRM-008",
        "name": "Robert Chambers",
        "email": "robert.chambers@example.com",
        "phone": "07700 900008",
        "location": "Cardiff, Wales",
        "job_title": "Operating Department Practitioner",
        "nhs_band": "Band 5",
        "years_exp": 5,
        "specialisms": ["Anaesthetics", "Scrub Practitioner", "Orthopaedic Surgery", "Laparoscopic"],
        "availability": "2 weeks notice",
        "salary_exp": "£29,000 - £36,000",
        "registration": "HCPC OD34567",
        "notes": "Experienced in high-dependency anaesthetic support, robotics-assisted surgery scrub, ACLS instructor."
    },
    {
        "crm_id": "CRM-009",
        "name": "Natasha Petrov",
        "email": "natasha.petrov@example.com",
        "phone": "07700 900009",
        "location": "Sheffield, South Yorkshire",
        "job_title": "Advanced Nurse Practitioner",
        "nhs_band": "Band 7",
        "years_exp": 15,
        "specialisms": ["Primary Care", "Minor Injuries", "Chronic Disease Management", "Prescribing"],
        "availability": "3 months notice",
        "salary_exp": "£43,000 - £52,000",
        "registration": "NMC PIN: 88D1234E, V300 Prescriber",
        "notes": "Independent prescriber, MSc Advanced Practice, clinical supervisor, QOF and DES delivery experience."
    },
    {
        "crm_id": "CRM-010",
        "name": "David Yeboah",
        "email": "david.yeboah@example.com",
        "phone": "07700 900010",
        "location": "Nottingham, Nottinghamshire",
        "job_title": "Radiographer - Diagnostic",
        "nhs_band": "Band 5",
        "years_exp": 4,
        "specialisms": ["Plain Film Radiography", "CT Reporting", "Fluoroscopy", "Mammography"],
        "availability": "1 month notice",
        "salary_exp": "£28,000 - £34,000",
        "registration": "HCPC RA23456",
        "notes": "Reporting radiographer qualification underway, experienced in emergency plain film, paediatric imaging."
    },
    {
        "crm_id": "CRM-011",
        "name": "Claire Donaldson",
        "email": "claire.donaldson@example.com",
        "phone": "07700 900011",
        "location": "Liverpool, Merseyside",
        "job_title": "Mental Health Nurse",
        "nhs_band": "Band 6",
        "years_exp": 10,
        "specialisms": ["Acute Inpatient Psychiatry", "Psychosis", "Crisis Assessment", "De-escalation"],
        "availability": "Immediately available",
        "salary_exp": "£35,000 - £43,000",
        "registration": "NMC PIN: 55F9876G",
        "notes": "PMVA instructor, CPA coordinator, Section 17 leave management, CBT level 1 trained, liaison psychiatry experience."
    },
    {
        "crm_id": "CRM-012",
        "name": "Dr. Amir Hassan",
        "email": "amir.hassan@example.com",
        "phone": "07700 900012",
        "location": "Leicester, Leicestershire",
        "job_title": "ST4 Registrar - General Medicine",
        "nhs_band": "Specialty Registrar",
        "years_exp": 8,
        "specialisms": ["Internal Medicine", "Cardiology", "Respiratory", "Endocrinology"],
        "availability": "End of rotation - 3 months",
        "salary_exp": "£51,000 - £58,000",
        "registration": "GMC: 8901234",
        "notes": "MRCP qualified, Cardiology subspecialty interest, echo trained, ALS/ILS instructor, research publications in AF management."
    }
]

def seed():
    print(f"Seeding {len(CANDIDATES)} UK healthcare candidates...")
    success = 0
    for c in CANDIDATES:
        try:
            resp = requests.post(
                f"{BASE_URL}/ingest",
                data={"profile_data": json.dumps(c)},
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✅ {c['name']} ({c['crm_id']}) → Milvus ID {data.get('milvus_id')}")
                success += 1
            else:
                print(f"  ❌ {c['name']}: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"  ❌ {c['name']}: {e}")

    print(f"\nSeeding complete: {success}/{len(CANDIDATES)} candidates ingested.")

if __name__ == "__main__":
    seed()
