# Punta del Hedge ML Project — Project Charter (v1.0)

**Date created:** 2025-04-25  
**Authors:** ChatGPT (assistant) — reviewed by Santiago Merlano

---

## 1 Vision & Scope

Build a **nightly, laptop-friendly machine-learning pipeline** that:

* Ingests point-in-time market data each evening  
* Produces a **0-to-5 grade** for every covered security (Fundamentals 50 %, Technicals 15 %, Outlook 20 %, Macro 15 %)  
* Outputs a **12-month price-target range** with confidence bands  
* Exposes a monitoring dashboard (alpha, hit-rate, turnover, latency, data-drift)  
* Remains fully explainable via SHAP so Santiago can modify or extend every module himself

---

## 2 Key Performance Indicators (KPIs)

| Category | KPI | Success target |
|----------|-----|----------------|
| Alpha generation | 12-month excess return vs. sector ETFs | **≥ +2 %** |
| Signal quality | Hit-rate (≥ 2 % α) | **≥ 55 %** |
| Risk-adjusted return | Information Ratio | **≥ 0.60** |
| Trading efficiency | Annual turnover | **≤ 250 %** |
| Ops reliability | Grades ready by 06:00 ET | **< 06:00 ET** |
| Model health | PSI on top features | **< 0.25** |
| Learning outcome | Quiz / explain-back score | **≥ 80 %** |

---

## 3 High-Level Architecture (overview)
External Feeds ─► RAW ─► BRONZE ─► SILVER ─► GOLD ─► Nightly Scoring
│ │
└─────► Model Training (weekends)
│
MLflow Registry
│
Grafana Dashboard + Alerts

*(A PNG diagram will live at `docs/img/architecture_v1.png` when available.)*

---

## 4 90-Day Timeline

| Week | Milestone |
|------|-----------|
| –1   | **Stage 0 sign-off → Save State #0** |
| 1-2  | Stage 1 — Data architecture & Great Expectations |
| 3-4  | Stage 2 — Feature library & label table |
| 5-7  | Stage 3 — Model training (Elastic-Net → LightGBM) |
| 8-9  | Stage 4 — Walk-forward back-test |
| 10-11| Stage 5 — Nightly Prefect flow + MLflow + Grafana |
| 12   | Stage 6 — Governance & manual-override SOP |
| 13   | Stage 7 — Knowledge checks & project retrospective |

---

## 5 RACI Matrix

| Task | Responsible | Accountable | Consulted | Informed |
|------|-------------|-------------|-----------|----------|
| Charter & design docs | **Assistant** | **Santiago** | — | — |
| Data-access keys | Santiago | Santiago | Assistant | — |
| ETL & data tests | Assistant | Santiago | — | — |
| Feature engineering | Assistant | Santiago | — | — |
| Model tuning | Assistant | Santiago | — | — |
| Back-test review | Santiago | Santiago | Assistant | — |
| Nightly ops setup | Assistant | Santiago | — | — |
| Dashboard monitoring | Assistant | Santiago | — | — |
| Learning quizzes | Santiago | Santiago | Assistant | — |

---

## 6 Constraints & Assumptions

* **Compute:** MacBook Pro 14″, 16 GB RAM (heavy jobs run evenings / weekends)  
* **Schedule:** ≈ 15 h per week (daytime study, evening coding)  
* **Data sources:** free / open feeds (Yahoo Finance, Stooq, SEC EDGAR, FRED, ECB CSV)  
* **Stakeholders:** Santiago only (initially)  
* **Repository name:** `punta-ml` — single **`main`** branch with save-state tags

---

## 7 Top Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Free-API rate limits | Medium | Medium | Local caching; secondary feeds |
| Laptop unavailable | High | Low | Heavy jobs scheduled off-hours; Save-States allow pause/resume |
| Data-schema change | Medium | High | Great Expectations schema tests + alerts |
| Time crunch | Medium | Medium | 90-day timeline with buffer; save-state checkpoints |
| Concept overload | Medium | Medium | Weekly recaps; quizzes ensure retention |

---

## 8 Save-State #0 Outputs

* `docs/00_charter.md` — this file  
* `docs/img/architecture_v1.png` — annotated architecture diagram (to be added)

---

________________________________


