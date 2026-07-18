# TODO — Implementation Prompt: HDB Resale Price Predictor (Live Data + Full Engineering Pipeline)

> This file is a complete, copy-ready execution specification. Hand it to a coding assistant
> (or follow it manually) to implement the project. It is written to be executed without
> assumptions: anything not specified here must be raised as `[CONFIRM: ...]`, never invented.

---

# ROLE AND CONTEXT

You are a careful senior Python ML engineer working inside the GitHub repository `muhammadmirza97/Capstone-Project-SNAIC-Week-1`. You make no assumptions: everything you need is specified below, and anything not specified must be marked `[CONFIRM: ...]` and asked about — never invented.

**Project purpose:** Turn a notebook-based capstone (HDB resale price prediction) into a personal, continuously-updated price-prediction tool so the owner (Mirza, Windows user) can sanity-check whether they are buying/selling an HDB flat at a fair price.

**Technology stack (locked):** Python 3.11, pandas, scikit-learn, Streamlit, joblib, requests, Prefect, MLflow, pytest, pytest-cov, `responses` (HTTP mocking). No other runtime dependencies.

**Current repository state (verified — do not assume otherwise):** The repo contains ONLY this TODO plus two project files, no code:
- `data/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv` — 82,801 data rows + 1 header row; exactly 11 columns: `month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease, resale_price`. Singapore HDB resale transactions from Jan 2017 onward.
- `group4.pdf` — the capstone slide deck (report only; keep as documentation).

The original model code lives in an uncommitted Colab notebook and is NOT available; you are building fresh from this specification. The capstone report's champion model (RandomForest on 4 features) achieved MAE ≈ $57,403, RMSE ≈ $83,186, R² ≈ 0.708, MAPE ≈ 12.7% — use these as sanity baselines, not hard targets.

# STEP 0 — CODEBASE ANALYSIS (NO CODE YET)

Before writing any code:
1. Run `git status`, `git branch -a`, list the working tree, and print the CSV header plus first 5 rows to confirm the state described above.
2. Inspect the `remaining_lease` column: confirm it contains BOTH formats `"61 years 04 months"` AND `"61 years"` (no months part). Confirm `storey_range` format is `"X TO Y"` (e.g. `"10 TO 12"`). Confirm `month` format is `"YYYY-MM"`.
3. Make ONE small live request to verify the data API before building ingest:
   `GET https://data.gov.sg/api/action/datastore_search?resource_id=d_8b84c4ee58e3cfc0ece0d773c8ca6abc&limit=5`
   Confirm the JSON contains `result.records` (list of row dicts matching the CSV columns, plus an extra `_id` field) and `result.total` (integer). If the response shape differs from this, STOP and report `[CONFIRM: data.gov.sg API response shape changed — paste actual JSON]` before proceeding.
4. List your assumptions and which files you will create. Do not write implementation code during Step 0.

This is a one-shot autonomous implementation: resolve non-critical uncertainties from evidence (the CSV, the API probe) instead of stopping repeatedly. Only stop for the `[CONFIRM]` cases explicitly defined in this prompt.

# SCOPE — EXACT DELIVERABLES

Work on a new branch `feature/hdb-live-pipeline` (create from `main`). Build exactly this tree:

```
README.md
requirements.txt
.gitignore                      # must ignore: models/, mlruns/, data/cache/, __pycache__/, .coverage, .pytest_cache/, venv/
data/raw/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv   # git mv from data/
docs/group4.pdf                 # git mv from repo root
src/hdb/__init__.py
src/hdb/config.py
src/hdb/ingest.py
src/hdb/validate.py
src/hdb/cleaning.py
src/hdb/features.py
src/hdb/model.py
src/hdb/evaluate.py
src/hdb/drift.py
src/hdb/registry.py
src/hdb/predict.py
pipelines/__init__.py
pipelines/flow.py
app.py
tests/  (files listed in TESTING section, plus tests/fixtures/sample_200.csv)
.github/workflows/ci.yml
.github/workflows/retrain.yml
```

Module-by-module specification (implement exactly; do not add extra features):

1. **`src/hdb/config.py`** — constants only: `DATASET_RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"`, `API_URL = "https://data.gov.sg/api/action/datastore_search"`, `RAW_CSV_PATH`, `CACHE_DIR = "data/cache"`, `MODELS_DIR = "models"`, `RANDOM_SEED = 42`, `EXPECTED_COLUMNS` (the 11 column names above, in order), `MODEL_USED_COLUMNS = ["month", "town", "flat_type", "storey_range", "floor_area_sqm", "remaining_lease", "resale_price"]`, RF params (`N_ESTIMATORS = 200`, `MAX_DEPTH = 20`), `TEST_SIZE = 0.2`, `GATE_MAE_TOLERANCE = 1.10`, `PSI_THRESHOLD = 0.2`.

2. **`src/hdb/ingest.py`** — `fetch_latest(force_refresh: bool = False) -> tuple[pd.DataFrame, str, str]` returning `(df, source, fetched_at_iso)` where `source` is `"api"`, `"cache"`, or `"local_csv"`. Behaviour: if a cache parquet/csv in `data/cache/` is younger than 24h and not `force_refresh`, return it (`"cache"`). Otherwise page through the API with `limit=10000` and increasing `offset` until `len(all_records) >= result.total`, drop the `_id` column, cache the result, return `("api", ...)`. On ANY network/HTTP/JSON error: log a clear warning and fall back to `data/raw/...csv` (`"local_csv"`). Never crash on network failure.

3. **`src/hdb/validate.py`** — `validate(df) -> dict` report; raises `ValueError` listing every violation if: columns ≠ `EXPECTED_COLUMNS` (order-insensitive), zero rows, any `resale_price` ≤ 0 or non-numeric, any `floor_area_sqm` ≤ 0, any `month` not matching `^\d{4}-\d{2}$`.

4. **`src/hdb/cleaning.py`** — `clean(df) -> pd.DataFrame`: `drop_duplicates()`, then `dropna(subset=MODEL_USED_COLUMNS)`. Return a copy; log rows removed by each step.

5. **`src/hdb/features.py`** — three pure functions + a builder:
   - `parse_storey_range(s: str) -> int`: `"10 TO 12"` → `11` (midpoint, integer). Regex `^(\d+) TO (\d+)$` after strip/upper; anything else raises `ValueError`.
   - `parse_remaining_lease(s: str) -> int` returning TOTAL MONTHS: `"61 years 04 months"` → `736`; `"61 years"` → `732` (this second format EXISTS in the data — handle it, do not assume months is always present); `"61 years 0 months"` → `732`. Regex `^(\d+) years?(?: (\d+) months?)?$`; anything else raises `ValueError`.
   - `parse_month_to_year(s: str) -> int`: `"2019-07"` → `2019`.
   - `build_features(df) -> pd.DataFrame` adding columns `storey_id`, `remaining_months`, `year` and returning ONLY: `["floor_area_sqm", "storey_id", "year", "remaining_months", "town", "flat_type", "resale_price"]`.

6. **`src/hdb/model.py`** — `build_pipeline(include_location: bool) -> sklearn.pipeline.Pipeline` with named steps EXACTLY `"prep"` (ColumnTransformer: `OneHotEncoder(handle_unknown="ignore")` on `["town","flat_type"]` when `include_location=True`, numeric columns `["floor_area_sqm","storey_id","year","remaining_months"]` passed through) and `"rf"` (`RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)`). Also `split_data(features_df)` → X_train, X_test, y_train, y_test via `train_test_split(test_size=0.2, random_state=42)`. CRITICAL ORDER: split FIRST, then fit only on training data — nothing may be fit on the full dataset.

7. **`src/hdb/evaluate.py`** — `evaluate_model(y_true, y_pred) -> dict` with keys `"mae"`, `"rmse"`, `"r2"`, `"mape"` (MAPE as percentage, `mean(abs((y_true - y_pred)/y_true))*100`; safe because validate() guarantees price > 0).

8. **`src/hdb/drift.py`** — `psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float` using decile bin edges computed from `reference` (clip proportions to min 1e-4 to avoid log(0)); `drift_report(reference_df, current_df) -> dict` with PSI for `resale_price`, `floor_area_sqm`, `remaining_months`, per-town median price and row count for both frames, and `"drift_detected": True` if any PSI > 0.2.

9. **`src/hdb/registry.py`** — versioned artifact store. `save_version(model, metrics, drift, data_meta) -> str` writes `models/v<YYYYMMDD-HHMMSS>/{model.joblib, metrics.json, drift.json, data_meta.json}` and updates `models/latest.txt` containing the version dirname. `load_latest() -> (model, metrics) | None`. `quality_gate(new_metrics, previous_metrics) -> tuple[bool, str]`: pass if no previous model exists; otherwise pass only if `new_metrics["mae"] <= previous_metrics["mae"] * 1.10`; the returned string explains the decision.

10. **`src/hdb/predict.py`** — `predict_price(model_pipeline, inputs: dict) -> dict` where `inputs` has keys `town, flat_type, floor_area_sqm, storey_id, year, remaining_months`. Build a 1-row DataFrame, transform it with `model_pipeline.named_steps["prep"]`, then collect one prediction per tree from `model_pipeline.named_steps["rf"].estimators_`. Return `{"point": <pipeline.predict value>, "low": np.percentile(tree_preds, 10), "high": np.percentile(tree_preds, 90)}`.

11. **`pipelines/flow.py`** — Prefect `@flow` named `hdb-training-flow`, runnable via `python -m pipelines.flow`, composed of `@task`s (ingest task with `retries=3, retry_delay_seconds=10`): ingest → validate → clean → build_features → split → train BOTH a baseline pipeline (`include_location=False`, i.e. the capstone's 4-feature model) and the location model (`include_location=True`) → evaluate both on the SAME test split → drift_report (reference = previous version's `data_meta` distributions if present, else skip with a log line) → quality_gate on the location model → if passed, `save_version` + print comparison table; if failed, exit code 1 with the gate reason. Wrap the run in MLflow (`mlflow.set_tracking_uri("file:./mlruns")`, experiment `"hdb-resale"`): log params (rf params, n rows, data source), all 8 metrics (both models, prefixed `baseline_` / `location_`), and the model artifact.

12. **`app.py`** — Streamlit app: on start call `registry.load_latest()`; if none, run the training flow once (show spinner). Sidebar shows: data source + fetched-at timestamp, model version, location-model test MAE/R². Inputs: `town` selectbox and `flat_type` selectbox (unique sorted values from the cleaned data), `storey_range` selectbox (unique values from data, converted via `parse_storey_range`), `floor_area_sqm` number_input (20–300, default 93), remaining lease as two number_inputs years (1–99) + months (0–11) converted to `remaining_months`, `year` number_input defaulting to the current year. A "Predict" button showing predicted price and the 10th–90th percentile range formatted as SGD. A "Refresh data & retrain" button that calls `fetch_latest(force_refresh=True)` and re-runs the flow. A fixed caption: "Decision-support estimate, not a formal valuation."

13. **`requirements.txt`** — pinned major versions of exactly: pandas, scikit-learn, streamlit, joblib, requests, prefect, mlflow, pytest, pytest-cov, responses, pyarrow.

14. **`.github/workflows/ci.yml`** — on push + pull_request: Python 3.11, `pip install -r requirements.txt`, `pytest --cov=src --cov-report=term-missing`. Must pass fully offline (tests use mocks/fixtures only).

15. **`.github/workflows/retrain.yml`** — `schedule: cron "0 2 1 * *"` + `workflow_dispatch`: install deps, run `python -m pipelines.flow` (live API), upload `models/` and `mlruns/` as workflow artifacts, and update a GitHub release tagged `model-latest` with `model.joblib` + `metrics.json`. Job must fail (red) if the quality gate rejects.

16. **`README.md`** — Windows-first instructions (PowerShell: `py -3.11 -m venv venv`, `venv\Scripts\Activate.ps1`) plus bash equivalents: setup, `pytest --cov=src`, `python -m pipelines.flow`, `streamlit run app.py`, `mlflow ui`, and a "First live run" section telling Mirza to verify the log line says `source: api` on their machine.

# OUT OF SCOPE — DO NOT

- Do not modify, re-export, truncate, or re-encode the CSV's contents (moving it with `git mv` is required and allowed).
- Do not delete or alter `group4.pdf` (move to `docs/` only).
- Do not commit `models/`, `mlruns/`, `data/cache/`, or any trained artifact.
- Do not add dependencies beyond requirements.txt above (no Docker, no FastAPI, no database, no cloud services).
- Do not build authentication, deployment configs, or hosting — this runs locally only.
- Do not change the data.gov.sg resource ID or invent alternative data sources.

# CONSTRAINTS

- Never invent filenames, column names, API fields, or functions beyond this specification.
- `random_state=42` everywhere randomness exists; two consecutive runs on identical data must produce identical test predictions.
- The train/test split must occur BEFORE any fitting (encoder, model) — this is a locked methodological requirement.
- All parsers must raise `ValueError` on unexpected input rather than returning silently wrong values.
- Network failure must never crash ingest — CSV fallback is mandatory.
- If anything in this spec conflicts with reality you observe (e.g. API shape), stop and report `[CONFIRM: ...]` with evidence; do not improvise.

# SAFETY AND APPROVAL

- Inspect before editing; never commit to `main`; all work on `feature/hdb-live-pipeline`.
- After Step 0, list the exact batch of files you will create, then proceed with the full batch (pre-authorised — no further checkpoints needed unless a `[CONFIRM]` triggers).
- Do not push or open a pull request until the owner approves.
- No secrets exist in this project; do not add any.

# TESTING (ALL MUST ACTUALLY RUN — NEVER CLAIM UNRUN TESTS PASSED)

Create `tests/fixtures/sample_200.csv`: the real CSV's header + its first 200 data rows, but verify it contains at least one `remaining_lease` value WITHOUT a months part (format `"NN years"`); if the first 200 rows don't include one, append real rows from the CSV that do.

- `tests/test_features.py` — `parse_storey_range`: `"10 TO 12"`→11, `"01 TO 03"`→2, `"49 TO 51"`→50, `"10-12"` raises, `""` raises. `parse_remaining_lease`: `"61 years 04 months"`→736, `"61 years"`→732, `"99 years 0 months"`→1188, `"61 yrs"` raises. `parse_month_to_year`: `"2017-01"`→2017.
- `tests/test_validate.py` — fixture passes; a frame missing a column raises; a frame with `resale_price=0` raises; bad month format raises.
- `tests/test_cleaning.py` — duplicated row removed; row with NA in `storey_range` removed; row with NA only in `street_name` (unused column) KEPT.
- `tests/test_ingest.py` — using `responses`: (a) two-page pagination (total=15, limit=10) returns all 15 rows without `_id`; (b) `ConnectionError` on the API → falls back to local CSV with `source == "local_csv"`; (c) malformed JSON body → same fallback. Point config paths at tmp_path fixtures — no real network.
- `tests/test_evaluate.py` — hand-computed: `y_true=[100000, 200000]`, `y_pred=[110000, 190000]` → mae=10000, rmse=10000, mape=7.5; r2 checked against `sklearn.metrics.r2_score`.
- `tests/test_model.py` — on the 200-row fixture: same seed twice → identical predictions (exact array equality); train and test index sets are disjoint and cover all rows; pipeline has named steps `"prep"` and `"rf"`.
- `tests/test_drift.py` — `psi(x, x) < 0.01`; `psi(N(0,1) sample, N(3,1) sample) > 0.2`; `drift_report` sets `drift_detected` accordingly.
- `tests/test_registry.py` — save then `load_latest` round-trips (use tmp_path); gate passes with no previous model; gate passes at new_mae = old_mae*1.05; gate fails at new_mae = old_mae*1.20.
- `tests/test_predict.py` — train on fixture; prediction `point > 0` and `low <= point <= high` is NOT guaranteed for point-vs-percentiles edge cases, so assert `low <= high` and `low > 0` and `point > 0`.

End-to-end verification (run and paste real output):
1. `pytest --cov=src --cov-report=term-missing` — all green; report coverage %.
2. `python -m pipelines.flow` on the full CSV (offline fallback path is acceptable) — paste the baseline-vs-location metrics table. Sanity: baseline MAE within roughly $45k–$70k and R² near 0.65–0.75 (comparable to the report's MAE $57,403 / R² 0.708); location model MAE strictly lower than baseline. If not, investigate before proceeding.
3. `streamlit run app.py` headless; confirm HTTP 200 and one manual prediction (town=TAMPINES, flat_type=4 ROOM, storey 10 TO 12, 93 sqm, 70y lease, current year) returns a positive price with a range.
4. Confirm `mlflow ui` shows the run (or list `mlruns/` contents).

# REQUIRED OUTPUT (FINAL REPORT)

1. Complete list of files created/moved with one-line purpose each.
2. Full test command output (actual, not paraphrased) and coverage percentage.
3. The metrics comparison table from the flow run, with the data source used (`api` / `local_csv`).
4. Any `[CONFIRM: ...]` items or blockers encountered and how each was resolved.
5. Manual steps remaining for the owner: run the flow once on their Windows machine to verify `source: api`, then optionally trigger the `retrain.yml` workflow_dispatch.
6. The branch name and commit list — do NOT push or open a PR; wait for approval.
