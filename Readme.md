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
```
```
## **Running the App**

### 1. **Prerequisites**
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.
- Together.ai API key ([Sign up here](https://api.together.xyz)).
- Git installed (for cloning the repo, if not already on your machine).
- Python 3.12+ (optional if running locally without Docker).


### 2. Clone the Repository
Clone the demo repository using Git and navigate into its directory.

`git clone https://github.com/AbiCharles/cikclumdemo.git`
cd <your-repo>

### 3. **Setup**
Create a `.env` file in the repo root:
```bash
TOGETHER_API_KEY=sk-your-key-here
# Optional performance tuning
TOP_K=3
MAX_NEW_TOKENS=300
TEMPERATURE=0.1 

```
### 4. **Start Stack**
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

These example test cases are designed to exercise **both Fast Mode and Reflection Mode**.  
- **Fast Mode** should return a complete summary if all key sections (eligibility, step therapy, documentation, forms) are already covered in the top-K results.  
- **Reflection Mode** will detect missing sections and re-query the RetrievalAgent to fill in gaps, demonstrating the iterative A2A workflow.  
They also ensure coverage across all available drugs and plans in the synthetic dataset.

| Patient ID | Drug     | Plan            | Expected focus (what the summary should surface) |
|:--|:--|:--|:--|
| **P001** | **Humira**  | **Acme Gold HMO**   | RA eligibility (active disease), **DMARD trial** (e.g., MTX), **TB screening required**, **Form HUM-01**, documentation: DAS28 + prior DMARDs |
| **P003** | **Humira**  | **CarePlus PPO**    | **Crohn’s eligibility**, GI documentation (colonoscopy or fecal calprotectin), **Portal form CP-IMM-03**, baseline TB/HBV screening |
| **P001** | **Ozempic** | **Acme Gold HMO**   | **A1c ≥ 7%**, **metformin trial** unless contraindicated, documentation: **A1c within 60 days** + prior meds |
| **P003** | **Ozempic** | **CarePlus PPO**    | **Age ≥ 18**, renal function & **hypoglycemia history**, **Form CP-ENDO-02**, documentation: **A1c + prior therapies** |
| **P005** | **Eliquis** | **Acme Silver EPO** | **NVAF eligibility**, documentation: **CBC + CrCl + med review**, **Form EPO-CARD-07** (bleeding risk) |
| **P003** | **Eliquis** | **CarePlus PPO**    | **DVT/PE eligibility**, **5-day parenteral anticoag pre-req** (unless contraindicated), documentation: **imaging + anticoag history** |

### Test Data Coverage

This table lists all drug/plan combinations in the synthetic dataset and the **sections** available for each.  
- A ✅ means the section exists in the dataset for that combination.  
- A ❌ means it’s missing, which means **Fast Mode** will not return it without Reflection Mode.

| Drug     | Plan            | Eligibility | Step Therapy | Documentation | Forms | Notes |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Humira   | Acme Gold HMO   | ✅ | ✅ | ✅ | ✅ | ❌ |
| Humira   | CarePlus PPO    | ✅ | ❌ | ✅ | ✅ | ❌ |
| Ozempic  | Acme Gold HMO   | ✅ | ✅ | ✅ | ❌ | ❌ |
| Ozempic  | CarePlus PPO    | ✅ | ❌ | ✅ *(added)* | ✅ | ❌ |
| Eliquis  | Acme Silver EPO | ✅ | ❌ | ✅ | ✅ | ❌ |
| Eliquis  | CarePlus PPO    | ✅ | ✅ | ✅ *(added)* | ❌ | ❌ |
| Humira   | Any             | ❌ | ❌ | ❌ | ❌ | ✅ |
| Ozempic  | Any             | ❌ | ❌ | ❌ | ❌ | ✅ |

**Notes:**
- Rows with **Any** plan are generic notes, not tied to a specific plan.
- The “*(added)*” markers show new documentation entries added in the dataset updates to improve Fast Mode completeness.
- Missing forms or step therapy entries are intentional in some cases to allow Reflection Mode to demonstrate filling those gaps.


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
```
## UI Tabs Description

When you open the Gradio UI at `http://localhost:8080/ui`, you’ll see the following tabs:

### **Summary**
- Displays the **final policy summary** generated by the SummarizationAgent.
- Includes:
  - Concise eligibility, step therapy, and documentation requirements.
  - A checklist of documents to include in the PA request.
  - Citations referencing the source documents retrieved.

### **Raw A2A Messages**
- Shows the **raw JSON** response from the SummarizationAgent’s A2A endpoint.
- Includes:
  - `task_id`, `agent` name, and `state`.
  - `output` with summary, checklist, and citations.
  - `trace` showing states, messages, and nested RetrievalAgent calls.
  - `raw_intermediate` with missing sections (if reflection used) and fast mode flag.

### **Human Trace**
- Provides a **human-readable timeline** of agent actions.
- Shows:
  - Which agent acted.
  - State changes (submitted → completed).
  - High-level messages sent between agents.
  - Nested traces for delegated tasks.

### **Benchmark**
- Compares **Fast Mode** vs **Reflection Mode** latency for a fixed patient/drug pair.
- Runs multiple trials for each mode and reports:
  - Number of runs (N)
  - Mean latency
  - Median latency
  - 95th percentile latency (p95)
  - Minimum and maximum times
- Also displays **per-trial logs** for both modes.
- Lets you set:
  - Patient ID and Drug Name.
  - Number of trials.
  - Warmup runs (excluded from results).
- **Fast Mode** skips the reflection re-query and is generally faster.
- **Reflection Mode** is slower but produces more complete summaries.

---

## Benchmarking in the App

You can benchmark the app in two ways:

### **1. In the UI**
- Go to the **Benchmark** tab in the Gradio interface.
- Set:
  - **Patient ID** (e.g., `P003`)
  - **Drug Name** (e.g., `Ozempic`)
  - **Trials** (number of timed runs per mode)
  - **Warmup** (runs to “warm up” the models and caches)
- Click **Run Benchmark**.
- The results table will compare Fast Mode and Reflection Mode latencies, and logs will show per-trial timings.

### **2. From the CLI**
Run the included benchmarking script:
```bash
python -m data.bench --patient P003 --drug Ozempic --trials 5 --warmup 1
```

## Benchmarking Results from CLI

The following table details the individual trial latencies for **Fast** and **Reflection** modes when processing the request for patient P003 and the drug Ozempic over 5 trials.

| Mode | Trial | Latency (s) |
| :--- | :--- | :--- |
| FAST | 1/5 | 21.298 |
| FAST | 2/5 | 12.320 |
| FAST | 3/5 | 16.755 |
| FAST | 4/5 | 18.461 |
| FAST | 5/5 | 12.835 |
| REFL | 1/5 | 76.169 |
| REFL | 2/5 | 65.257 |
| REFL | 3/5 | 19.025 |
| REFL | 4/5 | 29.478 |
| REFL | 5/5 | 31.674 |

***

## Latency Summary

This table provides a statistical summary of the latency data, highlighting the difference in performance between the two modes. The **Reflection (REFL)** mode, which performs an additional query, shows a significantly higher mean and maximum latency compared to the **Fast (FAST)** mode.

| Name | Count | Mean (s) | Median (s) | P95 (s) | Min (s) | Max (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FAST | 5 | 16.334 | 16.755 | 18.461 | 12.320 | 21.298 |
| REFL | 5 | 44.321 | 31.674 | 65.257 | 19.025 | 76.169 |

