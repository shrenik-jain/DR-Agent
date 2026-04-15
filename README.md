# DRAgent: Agentic AI for Residential Demand Response

An intelligent agent that helps homeowners optimize electricity usage to save money and reduce carbon emissions through automated demand response.

## 🎯 Project Overview

DRAgent implements a "Reason-Act-Summarize" pipeline using LangChain to automate residential demand response. The system:

1. **Retrieves** real-time electricity prices (SDG&E) and carbon intensity data (CAISO)
2. **Optimizes** appliance schedules using constrained Linear Programming
3. **Explains** recommendations in natural, user-friendly language

### Key Features

- ✅ **Real Data Integration**: Fetches actual TOU tariffs and carbon forecasts
- ✅ **Guaranteed Feasibility**: Uses CVXPY to ensure all constraints are satisfied
- ✅ **Multi-Objective**: Optimize for cost, carbon, or both simultaneously
- ✅ **Natural Language**: Conversational interface for non-expert users
- ✅ **Transparent**: Clear explanations of why recommendations work

## 🏗️ Architecture

```
User Query → LangChain Agent (Orchestrator)
                ↓
    ┌───────────┼───────────────┐
    ↓           ↓               ↓
Data Tools   Optimization    Explanation
(SDG&E/CAISO)  (CVXPY)      (OpenAI LLM)
```

### Three-Stage Pipeline

1. **Retrieval & Grounding**: Tool-calling to fetch price/carbon data
2. **Optimization Engine**: Constrained LP solving for appliance scheduling
3. **Explanation Synthesis**: LLM generates natural language reports

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

```bash
# Clone or download the project files
cd DR-Agent

# Install dependencies and register the package (needed for `import dragent`)
pip install -r requirements.txt
pip install -e .

# Set your API key
export OPENAI_API_KEY='your-api-key-here'
```

## 🚀 Quick Start

### Basic Usage

```python
from dragent import create_dr_agent

# Create the agent
agent = create_dr_agent()

# Ask for help
query = """Help me schedule my EV charging for tomorrow.
- Needs 16 kWh
- Available 10 PM to 7 AM
- Max charging rate: 11 kW
I want to minimize my electricity bill."""

result = agent.invoke({"input": query})
print(result["output"])
```

**Example Output:**
```
I'll help you optimize your EV charging schedule for tomorrow.

First, let me get the latest electricity rates and grid data...

[Agent fetches SDG&E prices and CAISO carbon data]

Recommended Schedule:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Charge your EV from 1:00 AM to 2:30 AM

Expected Savings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Cost: $1.92 saved vs charging immediately ($4.32 → $2.40)
• Carbon: 3.2 lbs CO₂ avoided
• Monthly: ~$58 in savings if you charge daily

Why This Works:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your EV charging is scheduled during SDG&E's super off-peak period
(12-6 AM) when electricity costs just $0.27/kWh compared to $0.52/kWh
during evening peak hours. The grid is also 44% cleaner at this time
with more renewable energy available.
```

### Run Demo

```bash
pip install -e .
python examples/demo.py
```

This will show:
- Basic usage examples
- Agent vs Baseline comparison
- Failure case handling

### Run Evaluation

```bash
python scripts/evaluation.py
```

This compares Agent vs Baseline on:
- **Objective Improvement**: Cost/carbon savings
- **Feasibility**: Constraint adherence
- **Faithfulness**: Explanation accuracy

## 📊 Evaluation Framework

### Test Scenarios

1. **Sufficient Information** (Golden Path)
   - All required parameters provided
   - Expected: Optimal schedule with accurate savings

2. **Insufficient Information**
   - Missing constraints (time windows, power limits)
   - Expected: Agent requests missing information

3. **Redundant Information**
   - Noisy context with irrelevant data
   - Expected: Agent filters and focuses on relevant info

4. **Carbon Optimization**
   - User prioritizes emissions over cost
   - Expected: Different schedule optimizing carbon

### Metrics

| Metric | Description | Agent | Baseline |
|--------|-------------|-------|----------|
| **Objective** | Actual cost/carbon savings | ✅ Optimal | ⚠️ Approximate |
| **Feasibility** | Respects all constraints | ✅ Guaranteed | ❌ May violate |
| **Faithfulness** | Explanation matches computation | ✅ Verifiable | ⚠️ Unverifiable |

## 🔧 Optimization Formulation

The agent solves the following constrained optimization problem:

**Decision Variables:**
- `x[a][h]`: Power consumption (kW) of appliance `a` at hour `h`

**Objective:**
```
minimize: Σ(h=1 to 24) price[h] × Σ(a∈A) x[a][h]
```

**Constraints:**
1. Energy requirement: `Σ(h=α to β) x[a][h] = E[a]` for each appliance
2. Time windows: `x[a][h] = 0` outside `[α[a], β[a]]`
3. Power limits: `P_min[a] ≤ x[a][h] ≤ P_max[a]`
4. Household peak: `Σ(a∈A) x[a][h] ≤ P_max_house`

**Solver:** CVXPY with ECOS (convex optimization)

## 🛠️ Tools

The agent has access to three tools:

### 1. `fetch_sdge_prices()`
Retrieves Time-of-Use electricity rates from SDG&E.

**Returns:**
```json
{
  "utility": "SDG&E",
  "tariff": "EV-TOU-5",
  "prices": [
    {"hour": 0, "price_per_kwh": 0.27, "period": "super_off_peak"},
    {"hour": 16, "price_per_kwh": 0.52, "period": "on_peak"},
    ...
  ]
}
```

### 2. `fetch_caiso_carbon()`
Retrieves grid carbon intensity from CAISO.

**Returns:**
```json
{
  "source": "CAISO",
  "carbon_data": [
    {"hour": 0, "carbon_intensity_lbs_per_mwh": 250, "intensity_level": "low"},
    {"hour": 18, "carbon_intensity_lbs_per_mwh": 550, "intensity_level": "high"},
    ...
  ]
}
```

### 3. `solve_dr_optimization()`
Solves the constrained optimization problem.

**Inputs:**
- Appliance specifications (energy, time windows, power limits)
- Price data
- Carbon data
- Optimization goal ("cost", "carbon", or "both")

**Returns:**
```json
{
  "status": "success",
  "schedule": {
    "EV": {
      "hourly_consumption_kwh": [...],
      "operating_hours": [1, 2, 3]
    }
  },
  "metrics": {
    "cost_savings_dollars": 1.92,
    "carbon_reduction_lbs": 3.2,
    ...
  }
}
```

## 📝 Appliance Specification Format

When describing appliances to the agent, use this format:

```json
{
  "name": "EV",
  "energy_required_kwh": 16.0,
  "start_hour": 22,  // 10 PM (use 24-hour format)
  "end_hour": 7,     // 7 AM next day
  "min_power_kw": 0.0,
  "max_power_kw": 11.0,
  "household_peak_limit": 15.0
}
```

**Common Appliances:**
- **EV (Tesla Model 3)**: 16 kWh, 11 kW max
- **Dishwasher**: 3.6 kWh, 2 kW max
- **Dryer**: 4.5 kWh, 4 kW max
- **Water Heater**: 3-4 kWh, 4.5 kW max

## 🎓 Comparison: Agent vs Baseline

### Baseline LLM (No Tools)

```python
from dragent import create_baseline_llm, run_baseline_recommendation

llm = create_baseline_llm()
response = run_baseline_recommendation(llm, "Help me charge my EV...")
```

**Characteristics:**
- ❌ No access to real price/carbon data
- ❌ No optimization solver
- ❌ Cannot guarantee constraint satisfaction
- ✅ Can provide general guidance
- ⚠️ Savings estimates are approximate

### Agentic Framework (With Tools)

```python
from dragent import create_dr_agent

agent = create_dr_agent()
response = agent.invoke({"input": "Help me charge my EV..."})
```

**Characteristics:**
- ✅ Retrieves real SDG&E and CAISO data
- ✅ Runs constrained optimization (CVXPY)
- ✅ Guarantees feasibility
- ✅ Provides accurate savings calculations
- ✅ Explains reasoning clearly

## 🔬 Ablation Studies

### 1. Redundant Information Test
**Setup:** Inject noisy data (other utilities, outdated prices)
**Expected:** Agent filters irrelevant context, focuses on SDG&E

### 2. Sufficient Information Test
**Setup:** Provide complete, clean specifications
**Expected:** Agent produces optimal schedule with accurate savings

### 3. Insufficient Information Test
**Setup:** Omit critical data (time windows, power limits)
**Expected:** Agent identifies gaps, requests clarification

## 🚨 Failure Cases

The agent gracefully handles:

1. **Missing Information**
   - User: "Charge my EV to save money"
   - Agent: "I need more details: how much energy, what time window, max charging rate?"

2. **Infeasible Constraints**
   - User requests 50 kWh in 1 hour with 11 kW max (impossible)
   - Agent: "This isn't feasible because 11 kW × 1 hour = 11 kWh < 50 kWh needed"

3. **Ambiguous Goals**
   - User has conflicting objectives
   - Agent: "Would you like to prioritize cost savings or carbon reduction?"

## 📈 Expected Results

Based on typical SDG&E TOU rates and CAISO carbon patterns:

| Appliance | Baseline Cost | Optimized Cost | Savings | Carbon Reduction |
|-----------|---------------|----------------|---------|------------------|
| EV (16 kWh) | $4.32 | $2.40 | $1.92 (44%) | 3.2 lbs (18%) |
| Dishwasher | $1.87 | $1.30 | $0.57 (30%) | 0.8 lbs (15%) |
| Dryer (4.5 kWh) | $2.34 | $1.62 | $0.72 (31%) | 1.1 lbs (16%) |

**Monthly savings for typical household:** $50-80 in electricity costs, 40-60 lbs CO₂

## 🔮 Future Extensions (Stretch Goals)

### HVAC Flexibility Model
Add pre-cooling/pre-heating capabilities:
```python
{
  "name": "HVAC",
  "temperature_setpoint": 72,
  "flexibility_degrees": 2,  // Can pre-cool to 70°F
  "thermal_mass": "medium"
}
```

### Battery Storage Integration
Optimize home battery charging/discharging:
```python
{
  "name": "Battery",
  "capacity_kwh": 13.5,  // Tesla Powerwall
  "max_charge_rate": 5.0,
  "max_discharge_rate": 5.0,
  "initial_soc": 0.5
}
```

### Multi-Day Optimization
Extend horizon beyond 24 hours for better EV planning:
```python
optimize_schedule(
  appliances=appliances,
  horizon_hours=72,  // 3 days
  trip_schedule={"Monday": 40, "Wednesday": 60}  // miles
)
```

## 📚 References

- SDG&E TOU Rates: https://www.sdge.com/residential/pricing-plans
- CAISO Emissions: https://www.caiso.com/todays-outlook/emissions
- Related Paper: [BuildingAgent: Towards Autonomous and Adaptive Smart Building Management](https://dl.acm.org/doi/abs/10.1145/3538637.3538871)

## 🐛 Troubleshooting

### "API key not found"
```bash
export OPENAI_API_KEY='your-key-here'
# or add to ~/.bashrc or ~/.zshrc
```

### "CVXPY solver failed"
```bash
pip install --upgrade cvxpy
# Try alternative solver
problem.solve(solver=cp.SCS)
```

### "Agent not calling tools"
- Check that API key is valid
- Verify LangChain version (>=0.1.0)
- Enable verbose mode: `AgentExecutor(verbose=True)`

## 📄 License

MIT License - feel free to use for your project!

---

**Questions?** See `examples/demo.py` for working examples or run `scripts/evaluation.py` for comprehensive test cases.
