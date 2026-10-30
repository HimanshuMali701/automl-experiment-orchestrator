# AutoML Experiment Orchestrator

A production-minded automated machine learning experimentation platform with stateful workflow orchestration and experiment tracking.

## Core Capabilities
- **Automated Data Profiling**: Statistical summary, missingness, cardinality, data type classification, and target column validation.
- **Controlled Experiment Planning**: Deterministic rule-based plans and schema-constrained LLM planners with strict allowlists (no arbitrary code execution).
- **Leakage-Free ML Pipelines**: Strict separation of training/validation data prior to preprocessing fitting, using Scikit-learn `Pipeline` and `ColumnTransformer`.
- **Reproducible Experiment Tracking**: Complete metric, parameter, and model artifact logging using MLflow backed by SQLite.
- **Stateful Orchestration**: LangGraph workflow engine with deterministic fallbacks and approval gates.
- **REST API & Dashboard**: FastAPI backend and Streamlit interactive frontend.

## Project Structure
```
automl-experiment-orchestrator/
├── src/automl/
│   ├── core/           # Configuration, logging, domain exceptions
│   ├── profiler/       # Data profiling, schema analysis, target validation
│   ├── planner/        # Heuristic and LLM-guided experiment planner
│   ├── trainer/        # Leakage-free training pipelines, metrics, CV
│   ├── tracking/       # MLflow integration and experiment catalog
│   ├── workflow/       # LangGraph state machine and workflow transitions
│   ├── api/            # FastAPI routes and schemas
│   └── ui/             # Streamlit application components
├── tests/              # Pytest test suite with synthetic fixtures
├── requirements.txt    # Project dependencies
├── pytest.ini          # Pytest configuration
└── README.md
```

## Setup & Testing
```bash
# Activate existing virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest
```
