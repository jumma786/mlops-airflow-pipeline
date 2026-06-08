# 🌊 Airflow ML Pipeline Orchestration

![CI](https://github.com/jumma786/mlops-airflow-pipeline/actions/workflows/pipeline.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.9.1-red)
![XGBoost](https://img.shields.io/badge/XGBoost-champion-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 9 of 10  
> Orchestrates the full ML training pipeline using Apache Airflow — data ingestion, feature engineering, model training, and validation gating — with XCom state passing between tasks.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 🌊 Airflow DAG | [dags/training_pipeline.py](dags/training_pipeline.py) |
| 📥 Ingest Task | [src/tasks/ingest.py](src/tasks/ingest.py) |
| ⚙️ Feature Task | [src/tasks/features.py](src/tasks/features.py) |
| 🏋️ Train Task | [src/tasks/train.py](src/tasks/train.py) |
| ✅ Validate Task | [src/tasks/validate.py](src/tasks/validate.py) |
| 🚀 Standalone Runner | [run_pipeline.py](run_pipeline.py) |
| 🧪 Tests | [tests/test_pipeline.py](tests/test_pipeline.py) |

---

## 🎯 What This Project Does

Orchestrates the bank marketing ML pipeline as a proper Airflow DAG:

1. **ingest_data** — loads UCI Bank Marketing, validates, saves as Parquet
2. **engineer_features** — encodes categoricals, saves feature matrix
3. **train_model** — trains XGBoost, cross-validates, saves artifacts
4. **validate_model** — gates promotion on AUC threshold (≥ 0.75)
5. **notify** — logs final summary

State is passed between tasks via **Airflow XCom**.

---

## 🔗 DAG Structure

```
start → ingest_data → engineer_features → train_model → validate_model → notify → end
```

**Schedule:** Weekly (Monday 6am UTC)  
**Retries:** 1 (5 min delay)

---

## 📊 Real Data Results

| Step | Metric | Value |
|---|---|---|
| Ingest | Rows | 41,188 |
| Ingest | Positive rate | 11.3% |
| Features | Features | 19 |
| Train | AUC | ~0.81 |
| Train | CV AUC | ~0.80 ± 0.01 |
| Validate | Threshold | 0.75 |
| Validate | Result | ✅ PASSED |

---

## 🚀 Quick Start

### Option 1 — Standalone (no Airflow needed)
```bash
git clone https://github.com/jumma786/mlops-airflow-pipeline.git
cd mlops-airflow-pipeline
pip install pandas numpy scikit-learn xgboost scipy pyarrow joblib
python run_pipeline.py --data-path data/bank-additional-full.csv
```

### Option 2 — Full Airflow
```bash
pip install -r requirements.txt
make airflow-init
make airflow-up
# Open http://localhost:8080 (admin/admin)
# Trigger: mlops_training_pipeline
```

### Tests
```bash
make install-ci
make test
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| **9** | **Airflow Pipeline** | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ This repo |
| 10 | Kubernetes Platform | mlops-k8s-platform | 🔜 |

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 9 of 10 — MLOps Portfolio Series.*
