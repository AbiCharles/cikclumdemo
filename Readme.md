# A2A Prior Authorization Demo

This project demonstrates **Agent-to-Agent (A2A)** communication between two autonomous agents:
- **RetrievalAgent** — retrieves prior-authorization policy snippets from a Qdrant vector DB.
- **SummarizationAgent** — orchestrates retrieval, optionally performs a reflection step to fetch missing sections, and produces a concise, actionable summary.

It includes:
- **Fast Mode toggle** (skip reflection for faster results)
- **Big “Retrieval in process…”** status banner
- **Reset Form** button in the UI
- **Docker Compose** setup with Qdrant and the app in separate services

---

## **Architecture Overview**

**Local stack (all running in Docker Compose)**:
- **Qdrant** (vector database) — stores synthetic prior authorization policy snippets.
- **FastAPI app** — hosts both agents under one service.
- **Gradio UI** — mounted at `/ui`, used to input Patient ID and Drug Name.
- **Together.ai API** — used for chat completions (meta-llama/Llama-3.3-70B-Instruct-Turbo-Free) and embeddings (togethercomputer/m2-bert-80M-32k-retrieval).

---

## **Agents**

### RetrievalAgent
- On first request:
  - Auto-detects embedding dimension from Together.ai.
  - Creates the Qdrant collection (`prior_auth_demo`) if missing.
  - Ingests a synthetic dataset of policy snippets.
- Retrieves `TOP_K` relevant policy snippets for a given `(patient_id, drug_name)` pair.

### SummarizationAgent
- Delegates to RetrievalAgent to get initial snippets.
- **Optional reflection step**:
  - Identifies missing sections (eligibility, step therapy, documentation, forms).
  - If not in Fast Mode, re-queries the retrieval agent for those sections.
- Produces a final summary with checklist and citations.

---

## **UI Features**

- **Fast Mode** (default ON):
  - Skips the reflection re-query.
  - Reduces retrieval + summarization to a single pass.
  - Much faster, but summaries may omit some sections.
- **Big Status Banner**:
  - Shows “⏳ Retrieval in process…” while the request is running.
  - Disables the **Run** button until processing completes.
- **Reset Form**:
  - Resets Patient ID, Drug Name, and Fast Mode to defaults.
  - Clears all outputs and re-enables the **Run** button.

---

## **Synthetic Dataset**

- **Patients → Plans**:
  - `P001`, `P002` → Acme Gold HMO
  - `P003`, `P004` → CarePlus PPO
  - `P005` → Acme Silver EPO

- **Drugs**:
  - Humira
  - Ozempic
  - Eliquis

---

## **Running the App**

### 1. **Prerequisites**
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.
- Together.ai API key ([Sign up here](https://api.together.xyz)).

### 2. **Setup**
Create a `.env` file in the repo root:
```bash
TOGETHER_API_KEY=sk-your-key-here
# Optional performance tuning
TOP_K=3
MAX_NEW_TOKENS=300
TEMPERATURE=0.1 

```
### 3. **Start Stack**
`docker compose up --build`

This will:
* Build the app image.
* Start Qdrant on port 6333.
* Start the FastAPI app on port 8080 (with Gradio UI mounted at /ui).

### Accessing the App

**UI:**
`http://localhost:8080/ui`
* Enter a Patient ID (e.g., P001) and Drug Name (e.g., Humira).
* Toggle Fast Mode ON/OFF.
* Click Run.

**Health check:**
`curl http://localhost:8080/health`

### Testing via CLI

`data/test.py` can send requests to the running summarization agent:
`python -m data.test --patient P003 --drug Ozempic`

### Example Test Cases

| Patient ID | Drug | Plan | Expected focus |
| :--- | :--- | :--- | :--- |
| P001 | Humira | Acme Gold HMO | RA criteria + TB screening form |
| P003 | Humira | CarePlus PPO | Crohn’s criteria + portal form CP-IMM-03 |
| P001 | Ozempic | Acme Gold HMO | A1c ≥ 7%, metformin trial |
| P003 | Ozempic | CarePlus PPO | Age ≥ 18, renal function, hypoglycemia history |
| P005 | Eliquis | Acme Silver EPO | NVAF + bleeding risk form EPO-CARD-07 |
| P003 | Eliquis | CarePlus PPO | DVT/PE, 5-day parenteral anticoag pre-req |

### Stopping the App
`docker compose down`

### Notes
* **Fast Mode** significantly reduces latency but may miss policy sections.
* **Reflection mode** (Fast Mode OFF) is slower but more complete.
* Qdrant dashboard is available at `http://localhost:6333/dashboard`.
* To change models, edit `.env`:
  * `TOGETHER_CHAT_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
  * `TOGETHER_EMBED_MODEL=togethercomputer/m2-bert-80M-32k-retrieval`
```
