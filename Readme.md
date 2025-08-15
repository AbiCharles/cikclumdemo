# A2A Prior Authorization Demo

## Architecture Overview

This demo implements a **co-located Agent-to-Agent (A2A) architecture** where two autonomous agents collaborate to retrieve and summarize prior authorization policy details for a given patient and drug.

**Local stack (all running in Docker Compose)**:
- **Qdrant** (vector database) — stores synthetic prior authorization policy snippets.
- **FastAPI app** — hosts both agents under one service.
- **Gradio UI** — mounted at `/ui`, used to input Patient ID and Drug Name.
- **Together.ai API** — used for chat completions (`meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`) and embeddings (`togethercomputer/m2-bert-80M-32k-retrieval`).

### **SummarizationAgent**
- **Role**: Orchestrator.
- **Purpose**:
  - Accepts the user’s goal (patient ID + drug name).
  - Determines the patient’s plan from a synthetic dataset.
  - Delegates retrieval tasks to the RetrievalAgent.
  - Optionally performs a *reflection step* to identify missing policy sections and re-query for them.
  - Produces the final summary, checklist, and citations.

### **RetrievalAgent**
- **Role**: Information retriever.
- **Purpose**:
  - Accepts retrieval tasks from the SummarizationAgent.
  - Converts the query into vector embeddings using Together.ai’s embedding model.
  - Searches a Qdrant vector database for relevant policy snippets.
  - Returns top-K results, optionally filtered to specific sections.

---

## Demo Behavior to Highlight A2A Concepts

### 1. Goal Decomposition
- **User input**: Patient ID and Drug Name are entered in the Gradio UI.
- **SummarizationAgent**:
  - Determines the patient's plan.
  - Decomposes the task into retrieval and summarization.
  - Delegates retrieval to the RetrievalAgent via an A2A HTTP call.

### 2. Task Delegation
- **SummarizationAgent → RetrievalAgent**:
  - RetrievalAgent embeds the query text.
  - Runs vector search in Qdrant.
  - Returns results (section, content, relevance score) to SummarizationAgent.

### 3. Optional Reflection & Iteration
- **Fast Mode OFF (Reflection Enabled)**:
  - SummarizationAgent asks Together.ai to identify **missing sections** (`eligibility`, `step_therapy`, `documentation`, `forms`).
  - If missing sections exist, it issues a **focused re-query** to RetrievalAgent.
  - Merges initial and focused results.
- **Fast Mode ON (Reflection Disabled)**:
  - Skips missing-section detection and re-query.
  - Produces the summary from the first retrieval only.

### 4. Final Summarization
- SummarizationAgent sends the retrieved content to Together.ai’s chat model.
- Generates:
  - **Concise summary**
  - **Checklist** of required documentation
  - **Citations** for referenced policy snippets.

### 5. User Feedback in the UI
- **Big Status Banner**: Shows “⏳ Retrieval in process…” immediately when **Run** is clicked.
- **Run Button**: Disabled while processing, re-enabled after results are ready.
- **Reset Form**: Resets defaults, clears outputs, re-enables **Run**.

### 6. Traceability
- Both agents return an **A2A-style JSON response** with:
  - `task_id`
  - `agent` name
  - `state` (`completed` / `failed`)
  - `output` (summary, checklist, citations)
  - `trace` (states, messages, nested traces)
  - `raw_intermediate` (e.g., missing sections, fast mode flag)

### 7. Observing the Difference
- **Fast Mode ON**:
  - One RetrievalAgent call.
  - Faster output, may omit sections.
- **Fast Mode OFF**:
  - Potentially multiple RetrievalAgent calls.
  - More complete summaries.

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

---
```
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