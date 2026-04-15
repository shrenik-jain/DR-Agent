# DRAgent Project Structure

```
DR-Agent/
│
├── README.md                 # Documentation
├── PROJECT_STRUCTURE.md      # This file
├── pyproject.toml            # Package metadata + dependencies (pip install -e .)
├── requirements.txt          # Same deps as pyproject (for pip install -r)
├── setup.sh                  # Automated setup (venv, deps, editable install)
│
├── dragent/                  # Installable Python package
│   ├── __init__.py           # Public exports (import from `dragent`)
│   ├── agent.py              # LangChain agent, tools, CVXPY optimizer
│   ├── config.py             # Model, tariff, appliance defaults
│   └── input_validation.py   # Appliance spec validation
│
├── apps/
│   └── app.py                # Gradio chat UI
│
├── examples/
│   └── demo.py               # Usage demonstrations
│
├── scripts/
│   ├── evaluation.py         # Agent vs baseline evaluation suite
│   └── ablations.py          # Ablation study CLI
│
├── tests/
│   ├── test_ablation_appliance_count_1.py
│   ├── test_ablation_appliance_count_3.py
│   ├── test_ablation_appliance_count_7.py
│   └── test_ablation_infeasible_single_appliance.py
│
└── notebooks/
    └── dragent_interactive.ipynb
```

## Layout conventions

- **Library code** lives under `dragent/` and is meant to be imported after `pip install -e .`.
- **Runnable entry points** are grouped under `apps/`, `examples/`, and `scripts/` so the repository root stays small.
- **Regression / ablation drivers** live in `tests/` (run from repo root, e.g. `python tests/test_ablation_appliance_count_1.py`).

## Key components

### 1. Core package (`dragent/`)

- **`agent.py`**: Tools (`fetch_sdge_prices`, `fetch_caiso_carbon`, `fetch_weather_forecast`, `solve_dr_optimization`, …), `create_dr_agent`, baseline helpers.
- **`config.py`**: Model name, horizons, tariff constants, `APPLIANCE_DEFAULTS`.
- **`input_validation.py`**: Validates and normalizes appliance JSON before optimization.

### 2. Applications

- **`apps/app.py`**: Gradio UI; run with `python apps/app.py` (after editable install).

### 3. Examples & evaluation

- **`examples/demo.py`**: Basic usage and comparisons.
- **`scripts/evaluation.py`**: Structured scenarios and metrics.
- **`scripts/ablations.py`**: Multi-architecture ablation CLI.

### 4. Notebooks

- **`notebooks/dragent_interactive.ipynb`**: Exploratory plots and tool calls.

## Usage patterns

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Quick start

```bash
python examples/demo.py
python scripts/evaluation.py
python apps/app.py
jupyter notebook notebooks/dragent_interactive.ipynb
```

### Import in code

```python
from dragent import create_dr_agent

agent = create_dr_agent()
result = agent.invoke({"input": "your query here"})
print(result["output"])
```

## Configuration

Settings are centralized in `dragent/config.py` (model, solver, tariffs, appliance defaults, evaluation scenario names).

## Extending the agent

Add tools and wire them in `dragent/agent.py`; add defaults in `dragent/config.py`. See README.md for solver and evaluation notes.
