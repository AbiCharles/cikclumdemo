"""
data/prior_auth_docs.py

Synthetic prior-authorization corpus used by the RetrievalAgent.

This module provides:
1) patient_plan_map(): deterministic patient_id -> plan
2) build_synthetic_corpus(): small list of policy "snippets" to index into Qdrant

Fields per snippet:
  - doc_id:     unique, stable string id (used in citations)
  - drug:       canonical drug name ("Humira", "Ozempic", "Eliquis")
  - plan:       display name of plan ("Acme Gold HMO", "CarePlus PPO", "Acme Silver EPO", or "Any" for general notes)
  - section:    one of {"eligibility","step_therapy","documentation","forms","notes"}
  - content:    short policy text fragment
"""

from __future__ import annotations
from typing import List, Dict


def patient_plan_map() -> Dict[str, str]:
    """Tiny deterministic mapping (non-PHI)."""
    return {
        "P001": "Acme Gold HMO",
        "P002": "Acme Gold HMO",
        "P003": "CarePlus PPO",
        "P004": "CarePlus PPO",
        "P005": "Acme Silver EPO",
    }


def build_synthetic_corpus() -> List[Dict]:
    """
    Returns a small list of 'documents' representing prior authorization policy snippets
    across a few drugs and health plans. Each item is payload-friendly for Qdrant.

    NOTE:
    - Keep doc_id unique & stable (used by citations).
    - Coverage is intentionally small but spans sections so reflection can find gaps.
    """
    docs: List[Dict] = []

    def add(doc_id: str, drug: str, plan: str, section: str, content: str):
        docs.append({
            "doc_id": doc_id,
            "drug": drug,
            "plan": plan,
            "section": section,  # "eligibility" | "step_therapy" | "documentation" | "forms" | "notes"
            "content": content,
        })

    # ---- Drug: Humira (adalimumab) ----
    add("HUM-ACME-ELIG-1", "Humira", "Acme Gold HMO", "eligibility",
        "For moderate-to-severe rheumatoid arthritis when prescribed by rheumatology; "
        "patient must have active disease and inadequate response to at least 1 DMARD.")
    add("HUM-ACME-STEP-1", "Humira", "Acme Gold HMO", "step_therapy",
        "Failure or intolerance to methotrexate for 12 weeks OR contraindication documented.")
    add("HUM-ACME-DOCS-1", "Humira", "Acme Gold HMO", "documentation",
        "Required: chart notes, disease activity score (e.g., DAS28), and prior DMARD history.")
    add("HUM-ACME-FORMS-1", "Humira", "Acme Gold HMO", "forms",
        "Use PA Request Form HUM-01; include TB screening within last 12 months.")
    add("HUM-CAREPLUS-ELIG-1", "Humira", "CarePlus PPO", "eligibility",
        "Covered for Crohn’s disease with moderate-to-severe activity; "
        "patient ≥ 18 years; failure of corticosteroids or immunomodulators.")
    add("HUM-CAREPLUS-DOCS-1", "Humira", "CarePlus PPO", "documentation",
        "Submit colonoscopy report or fecal calprotectin, and prior therapy rationale.")
    add("HUM-CAREPLUS-FORMS-1", "Humira", "CarePlus PPO", "forms",
        "Use Prior Auth Portal with form CP-IMM-03; baseline TB and hepatitis B screening required.")

    # ---- Drug: Ozempic (semaglutide) ----
    add("OZE-ACME-ELIG-1", "Ozempic", "Acme Gold HMO", "eligibility",
        "Type 2 diabetes mellitus; A1c ≥ 7.0% despite metformin; adjunct to diet/exercise; "
        "not for weight loss alone.")
    add("OZE-ACME-STEP-1", "Ozempic", "Acme Gold HMO", "step_therapy",
        "Trial of metformin at max tolerated dose for 3 months unless contraindicated.")
    add("OZE-ACME-DOCS-1", "Ozempic", "Acme Gold HMO", "documentation",
        "Submit A1c within last 60 days and prior antidiabetic medications with dates/doses.")
    add("OZE-CAREPLUS-ELIG-1", "Ozempic", "CarePlus PPO", "eligibility",
        "Type 2 diabetes; BMI not a criterion. Must be age ≥ 18; not covered for type 1 diabetes.")
    add("OZE-CAREPLUS-FORMS-1", "Ozempic", "CarePlus PPO", "forms",
        "Form CP-ENDO-02; include renal function and hypoglycemia history.")
    # NEW: add CarePlus documentation (fills a common missing section in Fast Mode)
    add("OZE-CAREPLUS-DOCS-1", "Ozempic", "CarePlus PPO", "documentation",
        "Provide recent A1c, prior therapies and responses, and renal function labs if available.")

    # ---- Drug: Eliquis (apixaban) ----
    add("ELI-ACME-ELIG-1", "Eliquis", "Acme Silver EPO", "eligibility",
        "Nonvalvular atrial fibrillation; CHA2DS2-VASc ≥ 2; no mechanical heart valve.")
    add("ELI-ACME-DOCS-1", "Eliquis", "Acme Silver EPO", "documentation",
        "Provide recent CBC, creatinine clearance, and concurrent meds review.")
    add("ELI-ACME-FORMS-1", "Eliquis", "Acme Silver EPO", "forms",
        "Acme EPO Anticoagulation form EPO-CARD-07; include bleeding risk assessment.")
    add("ELI-CAREPLUS-ELIG-1", "Eliquis", "CarePlus PPO", "eligibility",
        "Treatment of DVT/PE or risk reduction of recurrence after initial therapy.")
    add("ELI-CAREPLUS-STEP-1", "Eliquis", "CarePlus PPO", "step_therapy",
        "If for DVT/PE, must have completed at least 5 days of parenteral anticoagulation unless contraindicated.")
    # NEW: add CarePlus documentation (again, helps Fast Mode completeness)
    add("ELI-CAREPLUS-DOCS-1", "Eliquis", "CarePlus PPO", "documentation",
        "Provide imaging reports and anticoagulation history; include bleeding risk assessment when available.")

    # ---- Generic notes (plan='Any' works because we don't hard-filter by plan) ----
    add("GEN-NOTES-1", "Humira", "Any", "notes",
        "Renewal requires evidence of clinical response and adherence; reassess annually.")
    add("GEN-NOTES-2", "Ozempic", "Any", "notes",
        "Coverage excludes combination with GLP-1 receptor agonists unless specified by plan.")

    return docs
