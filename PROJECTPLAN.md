# Project Plan — **AI‑GENIE Streamlit Prototype**

## 1 · Purpose & Context
This project will deliver a **Streamlit‑based proof‑of‑concept** that re‑implements the *AI‑GENIE* pipeline (van Lissa et al., 2024) **entirely in Python**, wrapping it in an intuitive UI for non‑technical researchers, speech‑language therapists (SLTs) and parents.  
The prototype will specialise in **generating psychometric items that measure communicative participation outcomes of children aged 6‑11 with communication difficulties**.  
Ultimately, it will help a PhD student showcase the feasibility of AI‑assisted, network‑psychometric item development without relying on the original, R‑centric implementation.

### Success criteria
- One‑click generation of item banks from a user‑supplied prompt and examples.  
- Visual step‑through of the six AI‑GENIE phases with interactive plots.  
- Final export of the refined items **(.csv)** and visual report **(.pdf)**.  
- Deployable on **Streamlit Community Cloud** using only an **OpenAI API key** in `secrets.toml`.

## 2 · Constraints & Assumptions
| Theme | Decision |
|-------|----------|
| LLM provider | **OpenAI** (GPT‑4o & `text-embedding-3-small`) only, for now |
| Storage | Local filesystem (JSON/Parquet) — no external DB |
| Languages | **Python ≥3.11**; *no R dependencies* |
| Users | Primarily the supervising PhD student; low concurrency |
| Deployment | Streamlit Community Cloud |
| Budget | Minimal; stay within free‑tier where possible |

## 3 · AI‑GENIE Computational Pipeline (Python version)
1. **Item Generation**  
   – Invoke GPT‑4o with few‑shot prompt + live "duplicate list" guard.  
   – Temperature exposed as slider (0.5 – 1.5).
2. **Text Embedding**  
   – Call `text-embedding-3-small`; store dense & L2‑sparse matrices.  
   – Option to download `.npz` for offline reuse.
3. **Initial EGA (TMFG & EBICglasso variants)**  
   – Build partial‑correlation networks with `networkx` + `graphical‑lasso`.  
   – Community detection via Walktrap (*python‑igraph*).
4. **Unique Variable Analysis (UVA)**  
   – Compute weighted topological overlap (wTO).  
   – Iteratively remove items with wTO ≥ 0.20.
5. **Embedding‑choice Check**  
   – Compare NMI (sparse vs full); keep higher.
6. **bootEGA Stability**  
   – 100 bootstrap resamples; drop items with cluster‑stability < 0.75.  
   – Recompute final NMI and surface summary stats.

Each phase will emit interactive artefacts (network graph, heat‑maps, bar charts) and feed the next phase automatically or step‑wise, depending on UI mode.

## 4 · System Architecture
```mermaid
graph TD
    subgraph Front‑End
        A[Streamlit UI] -->|User inputs| B[Controller]
    end
    subgraph Core Engine (Python)
        B --> C[LLM ⇢ Item Generator]
        C --> D[Embedder]
        D --> E[EGA Builder]
        E --> F[UVA Filter]
        F --> G[bootEGA Validator]
    end
    subgraph Storage
        C ---|raw_items.json| S1[(Local)]
        G ---|final_items.csv| S1
        Plots --- S1
    end
    G --> H[Export Module]
    H -->|CSV/PDF| A
```

## 5 · Roadmap
### Phase 0 — Project Foundation
- [ ] Initialise GitHub repo with MIT license
- [ ] `requirements.txt` with Streamlit ≥1.33, `openai`, `numpy`, `pandas`, `igraph`, `scikit‑learn`, `networkx`, `matplotlib`, `reportlab` (PDF)
- [x] Create `LOG.md` file

### Phase 1 — LLM Item Generator
- [x] Draft base prompt template (system & few‑shot examples)
- [x] Implement `generate_items(prompt_focus, n, temperature, previous_items)`
- [x] UI: text‑area for prompt, good/bad examples, sliders (n items, temperature)
- [x] Live duplicate‑list guard to enforce uniqueness
- [x] Add Big Five test cases to UI options and prompts
- [x] UI: Add optional text areas for custom positive/negative examples
- [x] UI: Add optional text area for forbidden words
- [x] Update `generate_items` to handle custom examples & forbidden words

### Phase 2 — Embedding Service
- [x] Wrap OpenAI embeddings with local caching (`joblib`) (for dense embeddings)
- [x] Build dense matrices (OpenAI) & sparse matrices (TF-IDF); ~~L2‑norm & pruning~~ (TF-IDF handles sparsity)
- [ ] Unit‑test deterministic caching & shape consistency

### Phase 3 — Exploratory Graph Analysis (Initial)
- [x] Calculate correlation/partial correlation matrix from embeddings (Input for network construction)
- [x] Implement TMFG network constructor (likely using `networkx` and algorithms adapted for TMFG)
- [x] Implement EBICglasso variant (using `sklearn.covariance.graphical_lasso` or `graphical_lassoCV`)
- [x] Implement Walktrap community detection (using `python-igraph`)
- [x] Calculate network fit metrics: TEFI & NMI (TEFI implemented; NMI placeholder for Phase 5/6)
- [x] UI: Add radio buttons for selecting network type (TMFG/EBICglasso)
- [x] UI: Display real-time metrics (TEFI, NMI, number of communities) (TEFI displayed; NMI shows N/A)
- [x] Plot: Implement interactive network visualization (e.g., `pyvis` or `plotly`) showing items colored by detected community

### Phase 4 — Unique Variable Analysis
- [x] Compute wTO based on network structure from Phase 3 (Refactored: Implemented in `remove_redundant_items_uva` using graph adjacency)
- [x] Iteratively remove item with highest wTO ≥ threshold (default 0.20, adjustable via slider)
- [x] Display dropped items table; ~~allow user override (restore checkbox)~~ (Override not implemented yet)
- [x] **Refactor UVA implementation:** Modify `remove_redundant_items_uva` to accept the constructed network graph (TMFG or EBICglasso) from Phase 3.
- [x] Inside the UVA loop, calculate wTO based on an adjacency matrix derived from the *current subgraph's structure* (e.g., absolute partial correlations for EBICglasso, existing edge weights for TMFG), **not** the raw similarity matrix subset.
- [x] Iteratively remove item with highest wTO ≥ threshold (default 0.20, adjustable via slider), using sum of *graph-based* connection strengths for tie-breaking.
- [x] Display dropped items table; ~~allow user override (restore checkbox) [UI feature]~~ (Override not implemented yet)

### Phase 5 — bootEGA Stability
- [x] Implement bootstrap resampling engine:
    - [x] Takes the item list remaining after UVA (Phase 4).
    - [x] Performs N bootstrap iterations (e.g., 100, configurable).
    - [x] In each iteration: samples items *with replacement*, re-runs the selected EGA pipeline (network construction + community detection from Phase 3) on the sample.
    - [x] Utilize multiprocessing for efficiency.
- [x] Implement iterative stability analysis and item removal:
    - [x] Calculate item stability: Proportion of bootstrap samples where an item is assigned to its original (pre-UVA/bootEGA) community.
    - [x] UI: Display stability scores (e.g., histogram or table).
    - [x] UI: Add slider for stability threshold (default 0.75).
    - [x] Iteratively remove the item with the lowest stability below the threshold.
    - [x] Re-run the *entire* bootEGA process (resampling, analysis) on the reduced set until all items meet the threshold.
    - [x] UI: Display items removed due to instability at each iteration.
- [x] Final structure assessment:
    - [x] Run EGA one last time on the final set of stable items.
    - [x] Calculate final NMI based on this stable structure (comparing detected communities vs. original pre-filtering communities or a theoretical structure if applicable).
    - [x] UI: Display final NMI and compare it to the initial NMI from Phase 3.

### Phase 6 — Visualisation & Export
- [ ] Collate plots into single PDF (`reportlab`)
- [ ] Export CSV of final items (+ metadata: cluster, stability, wTO)
- [ ] "Download report" & "Download items" buttons in UI

### Phase 7 — Deployment & Polish
- [ ] Add Streamlit Cloud secrets instructions & safety checks
- [ ] README walkthrough with screenshots/GIF
- [ ] User‑testing session with PhD student; collect feedback
- [ ] Tag `v0.1.0` release

## 6 · Stretch Goals (Post‑v0.1)
- [ ] Add support for additional OpenAI models (function call to `/models` list)
- [ ] Plug‑in architecture for non‑Big‑Five constructs (JSON schema upload)
- [ ] Background job queue (Celery + Redis) for heavy bootEGA runs
- [ ] Persistent user projects via TinyDB or SQLite
- [ ] Theming & localisation for SLT/parent‑facing version

---

**Maintainer:** Zander Wessels | **Last updated:** 18 Apr 2025
