# CLAUDE.md

## What this repository is

The **reference system** for BANA 7075, ML System Design for Business (University of Cincinnati). Students follow this one end-to-end ML system through all seven weeks of the course: they run it, inspect it, break it on purpose, and explain what happened. Many students do not code (MBA and AI Certificate students take the course alongside M.S. Business Analytics students).

It is a simplified copy of [rossmann-forecasting](https://github.com/bradleyboehmke/rossmann-forecasting) (imported at `9553f37` without history). The subject of the course is the system around the model, not the model. This repository is also a GitHub template: student teams create their project repositories from it.

**Status:** being rebuilt in Phase 1 of the course redesign. REF-04 is done: one LightGBM model with fixed hyperparameters and 12 features listed in `config/params.yaml`. The rest still reflects the original Rossmann project (Great Expectations 0.x, DVC with a local remote) until the remaining Phase 1 tasks land. Work is tracked in [issue #1](https://github.com/bradleyboehmke/uc-bana-7075-reference-system/issues/1).

## Where the plan lives

The course plan is in the sibling course repository, `../uc-bana-7075/` ([GitHub](https://github.com/bradleyboehmke/uc-bana-7075)). Read what the task needs, not everything:

| Need | Read |
|---|---|
| A task's description, dependencies, and acceptance criteria | `../uc-bana-7075/planning/DEVELOPMENT_BACKLOG.md` (search for the task ID, such as `REF-04`) |
| Why something was decided | `../uc-bana-7075/planning/DECISION_LOG.md` (decisions are `D-nn`) |
| What this system must contain when finished | `../uc-bana-7075/planning/REFERENCE_SYSTEM_OPTIONS.md`, section "What the simplified template must contain" |
| Which week uses which part of the system | `../uc-bana-7075/planning/CURRICULUM_MAP.md` |
| Course terms (Try It, lab check, evidence minimum, and others) | `../uc-bana-7075/GLOSSARY.md` |

Do not copy planning content into this repository. If a planning document is wrong or out of date, say so rather than working around it.

## Rules for every change

- **Simpler wins.** When choosing between a more sophisticated and a more explainable version, choose the explainable one. Model accuracy is secondary.
- **Colab speed budget.** The full pipeline must run in under 5 minutes on a free Colab runtime; training in under 2 minutes.
- **Students can start any week fresh.** Each pipeline stage must have committed, precomputed outputs (under `artifacts/` once REF-06 lands), so a student who missed a week can begin the next one.
- **No accounts or keys** beyond GitHub and Google (for Colab). No new hosted services.
- **Pin exact versions** of every dependency once REF-08 lands; never introduce `>=` pins.
- **Keep data small.** The data is 100 of Rossmann's 1,115 stores (stratified by StoreType × Assortment, seed 7075). Do not commit trained models or large binaries.
- **Never modify the original `rossmann-forecasting` repository.** Week 1 uses its three maturity branches as they are.
- Explainability (SHAP) stays in code only, with no UI pages or extra features built around it. The LLM "predict, then explain" feature is deferred.

## How to work

- One backlog task per session and per pull request. Title PRs with the task ID, for example `REF-04: Simplify to a single LightGBM model`, and reference issue #1.
- Phase 1 order: REF-04 → REF-05 → REF-06 → REF-08 → ENV-01 → QA-01.
- A task is done when its acceptance criteria in the backlog are met and `pytest` passes.
- When a change removes something (a model, a feature, a script), also remove its tests and docs rather than leaving dead references.

## Commands

These reflect the current code and will change as Phase 1 tasks land; update this section in the same PR as the change.

```bash
# Environment (uv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
pytest -v

# Data pipeline
python -m src.data.make_dataset
python src/data/validate_data.py --stage raw
python -m src.features.build_features

# Model: train and register (seconds), validate and promote, batch predict
python -m src.models.train
python src/models/validate_model.py
python src/models/validate_model.py --promote-to-production
python src/models/predict.py --stage Production

# Services: MLflow (5000), FastAPI (8000), Streamlit (8501)
bash scripts/launch_dashboard.sh

# Monitoring
python src/monitoring/prepare_reference_data.py
python src/monitoring/generate_reports.py --days 7
```

## Known issues

- The docs workflow (`.github/workflows/docs.yml`) fails because unpinned mkdocstrings versions conflict. REF-08 fixes it; students do not use the MkDocs site.
- Unpinned `mlflow>=2.8.0` currently installs MLflow 3.x, which refuses the `./mlruns` file store unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Set it locally until REF-08 pins versions (or moves tracking to SQLite). Tests use a temporary SQLite store and are unaffected.
- This file is written for developing the system. Before launch, REF-09 adds student-facing documentation, and this file should be revisited so student teams that create repositories from the template get guidance suited to them.
