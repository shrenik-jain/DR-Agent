"""
ablations.py — Ablation Study Suite for DRAgent
================================================

Every scenario is evaluated under THREE architectures:

  ┌─────────────────────────────┬────────────────────────────────────────────────────────┐
  │ Architecture                │ Description                                            │
  ├─────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Baseline (static optim.)    │ No LLM. Solver only with hardcoded heuristic inputs.   │
  │                             │ Always charges at start of window, no real data fetch. │
  ├─────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Single-pass (no loop)       │ LLM agent fetches real data, calls solver once, done.  │
  │                             │ No self-checking. One linear pass per query.           │
  ├─────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Agentic (verif.-gated)      │ Like Single-pass, but adds a verification loop.        │
  │                             │ After solving, a second LLM call checks feasibility    │
  │                             │ and faithfulness. If check fails, it retries (max 2x). │
  └─────────────────────────────┴────────────────────────────────────────────────────────┘

Five scenario groups (28 scenarios total):
  1. PROMPT SPARSITY   — minimal / partial / exact / redundant / adversarial
  2. APPLIANCE COUNT   — scale 1 → N until solver or agent breaks
  3. APPLIANCE TYPE    — HVAC, EV (SoC), V2G, pool pump, water heater, mixed
  4. OPTIM. GOAL       — cost vs carbon vs both
  5. EDGE CASES        — infeasible, zero-energy, midnight-crossing, tight boundary

Metrics (per scenario × architecture):
  feasibility_pct   — % of hard constraints satisfied in the produced schedule
  cost_reduced_pct  — % cost reduction vs always-charge-immediately baseline
  iters_to_pass     — agent loop iterations used (0 for Baseline)
  total_tool_calls  — total tool invocations across the run
  safe_fail_pct     — % of expected-failure cases caught gracefully (group-level)
  faithfulness_pct  — % of 6 faithfulness sub-checks passed in LLM response

Usage
-----
    python scripts/ablations.py                 # all groups, all architectures
    python scripts/ablations.py --group sparsity
    python scripts/ablations.py --arch baseline
    python scripts/ablations.py --dry-run
    python scripts/ablations.py --output my_run.json

Outputs
-------
  ablation_results.json   — full raw results (JSON)
  ablation_summary.txt    — side-by-side metrics table
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import re
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from dragent import (
    create_dr_agent,
    create_baseline_llm,
    fetch_caiso_carbon,
    fetch_sdge_prices,
    fetch_weather_forecast,
    run_baseline_recommendation,
    solve_dr_optimization,
)


# ==============================================================================
# ARCHITECTURE NAMES — used as dict keys throughout
# ==============================================================================

ARCH_BASELINE    = "baseline"
ARCH_SINGLEPASS  = "single_pass"
ARCH_AGENTIC     = "agentic"
ALL_ARCHS        = [ARCH_BASELINE, ARCH_SINGLEPASS, ARCH_AGENTIC]

ARCH_LABELS = {
    ARCH_BASELINE:   "Baseline (static optim.)",
    ARCH_SINGLEPASS: "Single-pass (no loop)",
    ARCH_AGENTIC:    "Agentic (verif.-gated)",
}


# ==============================================================================
# SHARED DATA FIXTURES
# ==============================================================================

def _prices() -> str:
    return fetch_sdge_prices.invoke({})

def _carbon() -> str:
    return fetch_caiso_carbon.invoke({})

def _weather() -> str:
    return fetch_weather_forecast.invoke({})


# ==============================================================================
# ARCHITECTURE 1 — BASELINE (static optimization)
#
# No LLM, no real data fetch.
# Uses hardcoded SDG&E TOU prices and CAISO carbon patterns.
# Heuristic: schedule each appliance starting at the beginning of its window
# (i.e., no time-shifting — this is the "dumb" reference point).
# Runs solve_dr_optimization with static price/carbon data so feasibility
# and cost-reduction metrics are computed consistently.
# ==============================================================================

# Static price array matching dragent.agent.fetch_sdge_prices output
_STATIC_PRICES_JSON = json.dumps({
    "utility": "SDG&E", "tariff": "EV-TOU-5", "date": "static",
    "prices": [
        {"hour": h,
         "price_per_kwh": 0.238 if h < 6 else (0.52 if 16 <= h < 21 else 0.36),
         "period": "super_off_peak" if h < 6 else ("on_peak" if 16 <= h < 21 else "off_peak")}
        for h in range(24)
    ],
    "currency": "USD"
})

# Static carbon array matching dragent.agent.fetch_caiso_carbon output
def _static_carbon_intensity(h: int) -> float:
    if h < 6:    return 250 + h * 5
    if h < 10:   return 300 + (h - 6) * 40
    if h < 16:   return 200 + abs(h - 13) * 10
    if h < 22:   return 400 + (h - 16) * 25
    return 400 - (h - 22) * 25

_STATIC_CARBON_JSON = json.dumps({
    "source": "CAISO", "region": "California", "date": "static",
    "carbon_data": [
        {"hour": h,
         "carbon_intensity_lbs_per_mwh": _static_carbon_intensity(h),
         "intensity_level": "low" if _static_carbon_intensity(h) < 300
                            else ("medium" if _static_carbon_intensity(h) < 400 else "high")}
        for h in range(24)
    ],
    "unit": "lbs_co2_per_mwh"
})

# Static weather (typical San Diego summer day)
_STATIC_WEATHER_JSON = json.dumps({
    "source": "weather_forecast", "location": "San Diego, CA", "date": "static",
    "hourly_temperatures": [
        {"hour": h,
         "temperature_f": round(
             65 + h * 0.5 if h < 6
             else 67.5 + (h - 6) * 2.0 if h < 14
             else 83.5 + (h - 14) * 0.5 if h < 18
             else 85.5 - (h - 18) * 1.5,
             1),
         "temperature_c": 0.0}
        for h in range(24)
    ],
    "units": "fahrenheit"
})


def run_baseline_architecture(
    appliances: List[Dict],
    goal: str = "cost",
    requires_weather: bool = False,
) -> Dict:
    """
    Baseline: call solve_dr_optimization with STATIC (hardcoded) price/carbon data.
    No LLM, no real-time fetch.

    This reflects 'no demand-response intelligence' — the solver still finds the
    mathematical optimum given the static inputs, but the inputs are fixed
    approximations, not live data. A real static baseline would just run each
    appliance at window-start; we use the solver here so feasibility metrics
    are comparable across architectures.

    Returns a record dict with all six metric fields populated.
    """
    t0 = time.perf_counter()
    tool_calls = 0
    try:
        raw = solve_dr_optimization.invoke({
            "appliances_json":   json.dumps(appliances),
            "prices_json":       _STATIC_PRICES_JSON,
            "carbon_json":       _STATIC_CARBON_JSON,
            "weather_json":      _STATIC_WEATHER_JSON if requires_weather else "{}",
            "optimization_goal": goal,
        })
        tool_calls += 1
        elapsed = time.perf_counter() - t0
        result  = json.loads(raw)
        status  = result.get("status", "unknown")
        success = status in ("success", "optimal_inaccurate")

        return {
            "arch":             ARCH_BASELINE,
            "solver_result":    result,
            "response":         "",          # no LLM response
            "status":           status,
            "success":          success,
            "iters_to_pass":    0,
            "total_tool_calls": tool_calls,
            "elapsed_s":        round(elapsed, 4),
            "error":            result.get("error"),
        }

    except Exception as exc:
        return {
            "arch":             ARCH_BASELINE,
            "solver_result":    {},
            "response":         "",
            "status":           "exception",
            "success":          False,
            "iters_to_pass":    0,
            "total_tool_calls": tool_calls,
            "elapsed_s":        round(time.perf_counter() - t0, 4),
            "error":            str(exc),
            "traceback":        traceback.format_exc(),
        }


# ==============================================================================
# ARCHITECTURE 2 — SINGLE-PASS (no loop)
#
# Full LangChain agent with real data fetch + CVXPY solver.
# One linear pass: understand → fetch → solve → respond. No re-checking.
# This is the current DRAgent implementation.
# ==============================================================================

def run_single_pass_architecture(
    query: str,
    appliances: Optional[List[Dict]],
    goal: str = "cost",
    requires_weather: bool = False,
    agent=None,
) -> Dict:
    """
    Single-pass: invoke the LangChain agent once and return its output.
    Captures intermediate_steps to count tool calls and iterations.
    """
    if agent is None:
        agent = create_dr_agent()

    t0 = time.perf_counter()
    try:
        result_raw         = agent.invoke({"input": query})
        elapsed            = time.perf_counter() - t0
        response           = result_raw.get("output", "")
        intermediate_steps = result_raw.get("intermediate_steps", [])
        iters              = len(intermediate_steps)

        # Get ground-truth solver result for metric computation
        solver_result = _get_solver_result(appliances, goal, requires_weather)

        return {
            "arch":             ARCH_SINGLEPASS,
            "solver_result":    solver_result,
            "response":         response,
            "status":           "success" if response else "empty",
            "success":          bool(response),
            "iters_to_pass":    iters,
            "total_tool_calls": iters,
            "elapsed_s":        round(elapsed, 2),
            "error":            None,
        }

    except Exception as exc:
        return {
            "arch":             ARCH_SINGLEPASS,
            "solver_result":    {},
            "response":         "",
            "status":           "exception",
            "success":          False,
            "iters_to_pass":    0,
            "total_tool_calls": 0,
            "elapsed_s":        round(time.perf_counter() - t0, 2),
            "error":            str(exc),
            "traceback":        traceback.format_exc(),
        }


# ==============================================================================
# ARCHITECTURE 3 — AGENTIC (verification-gated)
#
# Same as Single-pass, but after the first response the agent runs a
# VERIFICATION step:
#   • A second LLM call checks the response for feasibility and faithfulness.
#   • If the verifier flags a problem, the agent retries the original query
#     (max MAX_RETRIES times).
#   • Metrics reflect the FINAL accepted response.
#   • iters_to_pass = total loop iterations across all attempts.
#   • total_tool_calls accumulates across retries.
# ==============================================================================

MAX_RETRIES = 2   # max number of re-attempts after a failed verification

_VERIFICATION_PROMPT = """\
You are a strict quality-control reviewer for a demand-response scheduling assistant.

Given the USER QUERY and the AGENT RESPONSE below, answer ONLY with a JSON object
(no markdown, no extra text) with these fields:

  "feasibility_ok":  true/false  — does the schedule respect all stated constraints?
                                   (time windows, power limits, energy requirements,
                                    household peak limit)
  "faithfulness_ok": true/false  — do all claimed savings/cost figures appear consistent
                                   with the price data mentioned (SDG&E TOU rates)?
  "issues": [list of short strings describing any problems found, or empty list]
  "verdict": "pass" or "fail"   — fail if EITHER feasibility_ok or faithfulness_ok is false

USER QUERY:
{query}

AGENT RESPONSE:
{response}
"""


def _verify_response(llm, query: str, response: str) -> Dict:
    """
    Call the verifier LLM and parse its JSON verdict.
    Returns a dict with keys: feasibility_ok, faithfulness_ok, issues, verdict.
    Falls back to {"verdict": "pass"} on parse failure so we don't infinite-loop.
    """
    prompt = _VERIFICATION_PROMPT.format(query=query, response=response)
    try:
        raw_verdict = llm.invoke(prompt).content
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw_verdict).strip()
        return json.loads(clean)
    except Exception:
        return {"verdict": "pass", "issues": [], "feasibility_ok": True, "faithfulness_ok": True}


def run_agentic_architecture(
    query: str,
    appliances: Optional[List[Dict]],
    goal: str = "cost",
    requires_weather: bool = False,
    agent=None,
    verifier_llm=None,
) -> Dict:
    """
    Agentic (verification-gated):
      1. Run single-pass agent.
      2. Run verifier LLM on the response.
      3. If verifier says 'fail', retry the agent (up to MAX_RETRIES times).
      4. Return the last accepted (or best available) response.
    """
    if agent is None:
        agent = create_dr_agent()
    if verifier_llm is None:
        verifier_llm = create_baseline_llm()

    total_iters       = 0
    total_tool_calls  = 0
    total_elapsed     = 0.0
    verification_log  = []
    final_response    = ""
    final_solver_result = {}
    all_issues        = []

    for attempt in range(1 + MAX_RETRIES):
        # ── Step 1: agent pass ────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            result_raw         = agent.invoke({"input": query})
            elapsed_agent      = time.perf_counter() - t0
            response           = result_raw.get("output", "")
            intermediate_steps = result_raw.get("intermediate_steps", [])
            iters_this_pass    = len(intermediate_steps)
        except Exception as exc:
            return {
                "arch":              ARCH_AGENTIC,
                "solver_result":     {},
                "response":          "",
                "status":            "exception",
                "success":           False,
                "iters_to_pass":     total_iters,
                "total_tool_calls":  total_tool_calls,
                "elapsed_s":         round(time.perf_counter() - t0, 2),
                "verification_log":  verification_log,
                "error":             str(exc),
                "traceback":         traceback.format_exc(),
            }

        total_iters      += iters_this_pass
        total_tool_calls += iters_this_pass
        total_elapsed    += elapsed_agent
        final_response    = response

        # ── Step 2: verification LLM call ────────────────────────────────────
        t1 = time.perf_counter()
        verdict = _verify_response(verifier_llm, query, response)
        total_elapsed    += time.perf_counter() - t1
        total_tool_calls += 1   # count the verifier call as a tool call

        verification_log.append({
            "attempt":      attempt + 1,
            "verdict":      verdict.get("verdict", "pass"),
            "issues":       verdict.get("issues", []),
            "feasibility":  verdict.get("feasibility_ok", True),
            "faithfulness": verdict.get("faithfulness_ok", True),
        })
        all_issues.extend(verdict.get("issues", []))

        if verdict.get("verdict", "pass") == "pass":
            break   # accepted — stop retrying

        # Retry: prepend context about what was wrong
        if attempt < MAX_RETRIES:
            issues_str = "; ".join(verdict.get("issues", ["unspecified issues"]))
            query = (
                f"[Previous answer had issues: {issues_str}. "
                f"Please correct and try again.]\n\n" + query
            )

    # ── Ground-truth solver result for metrics ────────────────────────────────
    final_solver_result = _get_solver_result(appliances, goal, requires_weather)

    return {
        "arch":              ARCH_AGENTIC,
        "solver_result":     final_solver_result,
        "response":          final_response,
        "status":            "success" if final_response else "empty",
        "success":           bool(final_response),
        "iters_to_pass":     total_iters,
        "total_tool_calls":  total_tool_calls,
        "elapsed_s":         round(total_elapsed, 2),
        "verification_log":  verification_log,
        "verification_issues": all_issues,
        "error":             None,
    }


# ==============================================================================
# SHARED HELPER — get ground-truth solver result with LIVE data
# ==============================================================================

def _get_solver_result(
    appliances: Optional[List[Dict]],
    goal: str = "cost",
    requires_weather: bool = False,
) -> Dict:
    """Run the solver with live-fetched data to get ground-truth metrics."""
    if not appliances:
        return {}
    try:
        raw = solve_dr_optimization.invoke({
            "appliances_json":   json.dumps(appliances),
            "prices_json":       _prices(),
            "carbon_json":       _carbon(),
            "weather_json":      _weather() if requires_weather else "{}",
            "optimization_goal": goal,
        })
        return json.loads(raw)
    except Exception:
        return {}


# ==============================================================================
# METRICS ENGINE
# ==============================================================================

def compute_feasibility_pct(solver_result: Dict, appliances: List[Dict]) -> float:
    """
    % of hard constraints satisfied:
      (a) energy delivered ≈ required (within 1%)
      (b) active hours inside allowed window
      (c) per-hour power ≤ max_power_kw
      (d) household peak not exceeded (global)
    """
    if solver_result.get("status") not in ("success", "optimal_inaccurate"):
        return 0.0
    schedule = solver_result.get("schedule", {})
    if not schedule:
        return 0.0

    H = 24
    cp = ct = 0
    peak_limit   = float(appliances[0].get("household_peak_limit", 15.0))
    hourly_total = np.zeros(H)

    for app in appliances:
        entry = schedule.get(app["name"])
        if entry is None:
            ct += 3; continue
        atype = entry.get("type", "flexible")

        if atype == "flexible":
            kwh   = entry.get("hourly_consumption_kwh", [0.0]*H)
            E_req = float(app["energy_required_kwh"])
            P_max = float(app["max_power_kw"])
            a, b  = app["start_hour"], app["end_hour"]
            win   = set(range(a, b+1)) if b >= a else set(range(a, H))|set(range(0, b+1))
            ct += 1; cp += int(E_req == 0 or abs(sum(kwh)-E_req)/max(E_req,1e-6) < 0.01)
            ct += 1; cp += int(not any(kwh[h]>0.01 and h not in win for h in range(H)))
            ct += 1; cp += int(all(v <= P_max+0.01 for v in kwh))
            hourly_total += np.array(kwh)

        elif atype == "ev":
            ch    = entry.get("hourly_charge_kw", [0.0]*H)
            P_max = float(app.get("max_charge_power_kw", 11.0))
            S_tgt = float(app.get("target_soc_kwh", 0.0))
            f_soc = float(entry.get("final_soc_kwh", 0.0))
            a, b  = app["start_hour"], app["end_hour"]
            win   = set(range(a, b+1)) if b >= a else set(range(a, H))|set(range(0, b+1))
            ct += 1; cp += int(f_soc >= S_tgt - 0.1)
            ct += 1; cp += int(not any(ch[h]>0.01 and h not in win for h in range(H)))
            ct += 1; cp += int(all(v <= P_max+0.01 for v in ch))
            hourly_total += np.array(ch)

        elif atype == "hvac":
            pwr  = entry.get("hourly_power_kw", [0.0]*H)
            T_in = entry.get("hourly_indoor_temp_f", [72.0]*H)
            tmin = entry.get("comfort_band_min_f", [70.0]*H)
            tmax = entry.get("comfort_band_max_f", [78.0]*H)
            P_max = float(app.get("max_power_kw", 3.5))
            ct += 1; cp += int(all(tmin[h]-0.1 <= T_in[h] <= tmax[h]+0.1 for h in range(H)))
            ct += 1; cp += int(all(v <= P_max+0.01 for v in pwr))
            ct += 1; cp += int(all(v >= -0.01 for v in pwr))
            hourly_total += np.array(pwr)

    ct += 1; cp += int(np.all(hourly_total <= peak_limit + 0.05))
    return round(100.0 * cp / max(ct, 1), 1)


def compute_cost_reduced_pct(solver_result: Dict) -> float:
    if solver_result.get("status") not in ("success", "optimal_inaccurate"):
        return 0.0
    m = solver_result.get("metrics", {})
    base = m.get("baseline_cost_dollars", 0.0)
    return round(100.0 * m.get("cost_savings_dollars", 0.0) / base, 1) if base > 0 else 0.0


def compute_faithfulness_pct(response: str, solver_result: Optional[Dict]) -> float:
    """
    Six sub-checks:
      F1  Mentions SDG&E or CAISO
      F2  Dollar figure present ($X.XX)
      F3  Carbon unit present (lbs/kg CO2)
      F4  Names a price period (off-peak etc.)
      F5  Claimed savings within 20% of solver value
      F6  Mentions hours consistent with optimal schedule
    """
    if not response:
        return 0.0
    checks: Dict[str, Optional[bool]] = {}
    checks["data_source"]  = bool(re.search(r"sdg&?e|caiso|real.?time|live (price|data)", response, re.I))
    checks["dollar"]       = bool(re.search(r"\$\s*\d+\.\d{2}", response))
    checks["carbon_unit"]  = bool(re.search(r"\d+\.?\d*\s*(lbs?|kg|pounds?)\s*(CO2|carbon|emissions)", response, re.I))
    checks["price_period"] = bool(re.search(r"(super.?off.?peak|off.?peak|on.?peak)", response, re.I))

    if solver_result and solver_result.get("status") in ("success", "optimal_inaccurate"):
        actual = solver_result.get("metrics", {}).get("cost_savings_dollars", 0.0)
        hits   = re.findall(r"\$\s*(\d+\.\d{2})", response)
        if hits and actual > 0:
            closest = min([float(v) for v in hits], key=lambda v: abs(v - actual))
            checks["savings_accuracy"] = abs(closest - actual) / actual <= 0.20
        else:
            checks["savings_accuracy"] = False

        opt_hours: List[int] = []
        for s in solver_result.get("schedule", {}).values():
            opt_hours += s.get("operating_hours", []) + s.get("charging_hours", []) + s.get("active_hours", [])
        found = False
        for h in set(opt_hours):
            pats = (["12:00 AM","midnight","12 AM"] if h == 0
                    else [f"{h}:00 AM", f"{h} AM"] if h < 12
                    else ["12:00 PM","noon","12 PM"] if h == 12
                    else [f"{h-12}:00 PM", f"{h-12} PM"])
            if any(p.lower() in response.lower() for p in pats):
                found = True; break
        checks["schedule_hours"] = found
    else:
        checks["savings_accuracy"] = None
        checks["schedule_hours"]   = None

    vals = [v for v in checks.values() if v is not None]
    return round(100.0 * sum(vals) / len(vals), 1) if vals else 0.0


def compute_safe_fail_pct(records: List[Dict]) -> float:
    """Among expect_failure=True records, % where graceful_fail=True."""
    fails = [r for r in records if r.get("expect_failure")]
    if not fails:
        return 100.0
    return round(100.0 * sum(1 for r in fails if r.get("graceful_fail")) / len(fails), 1)


# ==============================================================================
# RESULT BUILDER — turns a raw arch output into a metrics record
# ==============================================================================

def _build_record(
    key: str,
    group: str,
    label: str,
    arch: str,
    arch_output: Dict,
    appliances: Optional[List[Dict]],
    expect_failure: bool,
    expect_clarification: bool,
    n_appliances: int,
    goal: str,
) -> Dict:
    solver_result  = arch_output.get("solver_result", {})
    response       = arch_output.get("response", "")
    success        = arch_output.get("success", False)

    # Feasibility + cost from solver result
    feasibility_pct  = compute_feasibility_pct(solver_result, appliances) if appliances else None
    cost_reduced_pct = compute_cost_reduced_pct(solver_result)

    # Faithfulness only meaningful if there is a response
    faithfulness_pct = compute_faithfulness_pct(response, solver_result) if response else None

    # Graceful fail: did we correctly handle an expected failure?
    if expect_failure:
        solver_failed   = solver_result.get("status") not in ("success", "optimal_inaccurate")
        text_caught     = _check_infeasibility_mentioned(response)
        graceful_fail   = solver_failed or text_caught
    else:
        graceful_fail = None

    # Did agent ask for clarification when expected?
    asks_clarif = _check_clarification(response) if response else False

    # Overall pass heuristic
    if expect_clarification:
        passed = asks_clarif
    elif expect_failure:
        passed = bool(graceful_fail)
    else:
        passed = (feasibility_pct or 0) > 50 or bool(
            re.search(r"\b(AM|PM|midnight|schedule|\$\d)\b", response, re.I)
        )

    return {
        # identity
        "key":               key,
        "group":             group,
        "label":             label,
        "arch":              arch,
        "arch_label":        ARCH_LABELS[arch],
        "n_appliances":      n_appliances,
        "goal":              goal,
        # pass/fail
        "passed":            passed,
        "expect_failure":    expect_failure,
        "expect_clarif":     expect_clarification,
        "graceful_fail":     graceful_fail,
        "asks_clarif":       asks_clarif,
        # the six metrics
        "feasibility_pct":   feasibility_pct,
        "cost_reduced_pct":  cost_reduced_pct,
        "iters_to_pass":     arch_output.get("iters_to_pass"),
        "total_tool_calls":  arch_output.get("total_tool_calls"),
        "safe_fail_pct":     None,   # filled in after group finishes
        "faithfulness_pct":  faithfulness_pct,
        # provenance
        "elapsed_s":         arch_output.get("elapsed_s"),
        "status":            arch_output.get("status"),
        "error":             arch_output.get("error"),
        "response_preview":  response[:300] if response else "",
        "verification_log":  arch_output.get("verification_log", []),
    }


def _check_clarification(response: str) -> bool:
    pats = [r"\?", r"could you (provide|share|tell|give)",
            r"(need|require|missing).{0,30}(information|detail|spec)",
            r"what (is|are) (the|your)",
            r"(please|can you) (clarify|specify|provide)",
            r"(energy|kwh|kw).{0,20}(need|required|available|missing)"]
    low = response.lower()
    return any(re.search(p, low) for p in pats)


def _check_infeasibility_mentioned(response: str) -> bool:
    pats = [r"infeasib", r"not (feasible|possible|achievable)",
            r"cannot be (met|satisfied|achiev)", r"exceeds",
            r"insufficient", r"too (little|short|small)",
            r"impossible", r"no (valid|feasible) solution"]
    low = response.lower()
    return any(re.search(p, low) for p in pats)


# ==============================================================================
# SCENARIO DEFINITIONS
# ==============================================================================

# ── GROUP 1: SPARSITY ─────────────────────────────────────────────────────────

SPARSITY_SCENARIOS: Dict[str, Dict] = {
    "sparse_minimal": {
        "label": "Minimal — almost no info",
        "query": "I want to charge my EV and run my dishwasher tonight to save money.",
        "expect_clarification": True, "expect_failure": False,
        "ground_truth_appliances": None,
    },
    "sparse_partial": {
        "label": "Partial — time window only",
        "query": ("I want to charge my EV between 10 PM and 7 AM, "
                  "and run the dishwasher after 8 PM. Help me save money."),
        "expect_clarification": True, "expect_failure": False,
        "ground_truth_appliances": None,
    },
    "sparse_exact": {
        "label": "Exact — all info, nothing extra",
        "query": ("Schedule these appliances to minimise cost.\n\n"
                  "- EV: needs 16 kWh, available 10 PM–7 AM, max 11 kW\n"
                  "- Dishwasher: needs 3.6 kWh, available 8 PM–11 PM, max 2 kW\n\n"
                  "Household peak limit: 15 kW."),
        "expect_clarification": False, "expect_failure": False,
        "ground_truth_appliances": [
            {"name": "EV", "type": "flexible", "energy_required_kwh": 16.0,
             "start_hour": 22, "end_hour": 7, "min_power_kw": 0.0,
             "max_power_kw": 11.0, "household_peak_limit": 15.0},
            {"name": "Dishwasher", "type": "flexible", "energy_required_kwh": 3.6,
             "start_hour": 20, "end_hour": 23, "min_power_kw": 0.0,
             "max_power_kw": 2.0, "household_peak_limit": 15.0},
        ],
    },
    "sparse_redundant": {
        "label": "Redundant — specs buried in noise",
        "query": ("My neighbour uses PG&E ($0.40/kWh peak) in San Francisco. "
                  "Gas prices are up. Reddit says solar changes everything. "
                  "My actual appliances:\n"
                  "- EV: needs 16 kWh, available 10 PM–7 AM, max 11 kW\n"
                  "- Dishwasher: needs 3.6 kWh, available 8 PM–11 PM, max 2 kW\n\n"
                  "SDG&E customer, San Diego. Household peak: 15 kW. Minimise cost."),
        "expect_clarification": False, "expect_failure": False,
        "ground_truth_appliances": [
            {"name": "EV", "type": "flexible", "energy_required_kwh": 16.0,
             "start_hour": 22, "end_hour": 7, "min_power_kw": 0.0,
             "max_power_kw": 11.0, "household_peak_limit": 15.0},
            {"name": "Dishwasher", "type": "flexible", "energy_required_kwh": 3.6,
             "start_hour": 20, "end_hour": 23, "min_power_kw": 0.0,
             "max_power_kw": 2.0, "household_peak_limit": 15.0},
        ],
    },
    "sparse_adversarial": {
        "label": "Adversarial — user states wrong flat rate",
        "query": ("I read SDG&E charges $0.10/kWh flat — no TOU. Timing doesn't matter. "
                  "But optimise anyway:\n"
                  "- EV: needs 16 kWh, available 10 PM–7 AM, max 11 kW\n"
                  "- Dishwasher: needs 3.6 kWh, available 8 PM–11 PM, max 2 kW\n\n"
                  "Household peak: 15 kW."),
        "expect_clarification": False, "expect_failure": False,
        "ground_truth_appliances": [
            {"name": "EV", "type": "flexible", "energy_required_kwh": 16.0,
             "start_hour": 22, "end_hour": 7, "min_power_kw": 0.0,
             "max_power_kw": 11.0, "household_peak_limit": 15.0},
            {"name": "Dishwasher", "type": "flexible", "energy_required_kwh": 3.6,
             "start_hour": 20, "end_hour": 23, "min_power_kw": 0.0,
             "max_power_kw": 2.0, "household_peak_limit": 15.0},
        ],
    },
}


# ── GROUP 2: COUNT SCALING ────────────────────────────────────────────────────

_CATALOGUE = [
    ("Dishwasher",           3.6,  20, 23,  2.0),
    ("Clothes Dryer",        4.5,  21, 24,  4.0),
    ("Washing Machine",      1.8,  20, 23,  2.0),
    ("Pool Pump",            8.0,   1, 10,  2.0),
    ("Water Heater",         3.5,  22,  6,  4.5),
    ("EV Charger A",        16.0,  22,  7, 11.0),
    ("EV Charger B",        16.0,  22,  7, 11.0),
    ("Space Heater",         4.0,  18, 22,  2.0),
    ("Dehumidifier",         1.5,   0, 24,  0.8),
    ("Chest Freezer",        2.0,   0, 24,  0.5),
    ("Air Purifier",         1.2,   0, 24,  0.3),
    ("Battery Charger",      5.0,  22,  6,  3.0),
    ("Hot Tub Heater",      10.0,  22,  6,  5.0),
    ("Grow Lights",          6.0,  20,  8,  1.5),
    ("Workshop Compressor",  3.0,   6, 18,  3.0),
    ("Irrigation Pump",      2.5,   5, 10,  1.5),
    ("EV Charger C",        12.0,  23,  7,  7.2),
    ("Sauna Heater",         6.0,  19, 23,  4.0),
    ("Heat Pump WH",         2.8,  22,  6,  1.5),
    ("Greenhouse Heating",   4.0,   0, 10,  2.0),
]

def _make_flex(idx: int, peak: float) -> Dict:
    name, energy, start, end, max_kw = _CATALOGUE[idx % len(_CATALOGUE)]
    suffix = f" #{idx // len(_CATALOGUE) + 1}" if idx >= len(_CATALOGUE) else ""
    return {"name": name+suffix, "type": "flexible",
            "energy_required_kwh": energy, "start_hour": start,
            "end_hour": end % 24, "min_power_kw": 0.0,
            "max_power_kw": max_kw, "household_peak_limit": peak}

def _count_scenario(n: int) -> Dict:
    peak = max(60.0, n * 6.0)
    apps = [_make_flex(i, peak) for i in range(n)]
    lines = "\n".join(
        f"- {a['name']}: {a['energy_required_kwh']} kWh, "
        f"{a['start_hour']}:00–{a['end_hour']}:00, max {a['max_power_kw']} kW"
        for a in apps)
    return {
        "label": f"Count={n} appliances",
        "n_appliances": n, "appliances": apps,
        "expect_failure": False, "expect_clarification": False,
        "query": (f"Minimise cost for all {n} appliance(s) below.\n"
                  f"Household peak: {peak:.0f} kW.\n\n{lines}"),
    }

COUNT_SCENARIOS: Dict[str, Dict] = {
    f"count_{n:03d}": _count_scenario(n) for n in [1, 2, 3, 5, 8, 10, 15, 20]
}


# ── GROUP 3: DIVERSITY ────────────────────────────────────────────────────────

DIVERSITY_SCENARIOS: Dict[str, Dict] = {
    "diversity_hvac_only": {
        "label": "HVAC only — thermal scheduling",
        "requires_weather": True, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Central AC", "type": "hvac",
                        "initial_temp_f": 76.0, "temp_min_f": 70.0, "temp_max_f": 78.0,
                        "max_power_kw": 3.5, "cop": 3.0,
                        "thermal_resistance": 4.0, "thermal_capacitance": 2.0,
                        "household_peak_limit": 15.0}],
        "query": ("My central AC is at 76°F. Keep it 70–78°F all day. "
                  "Household peak: 15 kW. Minimise cost."),
    },
    "diversity_hvac_setback": {
        "label": "HVAC with occupancy setback",
        "requires_weather": True, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Central AC", "type": "hvac",
                        "initial_temp_f": 74.0,
                        "temp_min_f": [68.0]*6 + [70.0]*16 + [68.0]*2,
                        "temp_max_f": [80.0]*6 + [76.0]*16 + [80.0]*2,
                        "max_power_kw": 3.5, "cop": 3.0,
                        "household_peak_limit": 15.0}],
        "query": ("AC at 74°F. Keep 70–76°F when home (6 AM–10 PM), "
                  "68–80°F overnight. Peak: 15 kW. Minimise cost."),
    },
    "diversity_ev_battery": {
        "label": "EV — SoC battery model (25→85%)",
        "requires_weather": False, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Tesla Model 3", "type": "ev",
                        "start_hour": 22, "end_hour": 7,
                        "battery_capacity_kwh": 75.0,
                        "initial_soc_kwh": 18.75, "target_soc_kwh": 63.75,
                        "max_charge_power_kw": 11.0, "charge_efficiency": 0.95,
                        "min_soc_kwh": 5.0, "max_discharge_power_kw": 0.0,
                        "household_peak_limit": 15.0}],
        "query": ("Tesla Model 3 (75 kWh, 25% charge). Plug in 10 PM, "
                  "need 85% by 7 AM. Max 11 kW, 95% eff. Peak: 15 kW. Minimise cost."),
    },
    "diversity_ev_v2g": {
        "label": "EV with V2G discharge",
        "requires_weather": False, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "V2G EV", "type": "ev",
                        "start_hour": 16, "end_hour": 7,
                        "battery_capacity_kwh": 75.0,
                        "initial_soc_kwh": 30.0, "target_soc_kwh": 60.0,
                        "max_charge_power_kw": 11.0, "charge_efficiency": 0.95,
                        "discharge_efficiency": 0.95, "max_discharge_power_kw": 7.2,
                        "min_soc_kwh": 10.0, "household_peak_limit": 15.0}],
        "query": ("EV (75 kWh, 40% charge) plugs in at 4 PM. Need 80% by 7 AM. "
                  "Can discharge at up to 7.2 kW. Charge up to 11 kW. Peak 15 kW. Minimise cost."),
    },
    "diversity_pool_pump": {
        "label": "Pool pump — wide all-day window",
        "requires_weather": False, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Pool Pump", "type": "flexible",
                        "energy_required_kwh": 8.0, "start_hour": 0, "end_hour": 23,
                        "min_power_kw": 0.0, "max_power_kw": 2.0,
                        "household_peak_limit": 15.0}],
        "query": "Pool pump needs 8 kWh. Runs midnight–11 PM, max 2 kW. Peak: 15 kW. Minimise cost.",
    },
    "diversity_water_heater": {
        "label": "Water heater — overnight flexible",
        "requires_weather": False, "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Water Heater", "type": "flexible",
                        "energy_required_kwh": 3.5, "start_hour": 22, "end_hour": 6,
                        "min_power_kw": 0.0, "max_power_kw": 4.5,
                        "household_peak_limit": 15.0}],
        "query": "Water heater: 3.5 kWh, 10 PM–6 AM, up to 4.5 kW. Peak: 15 kW. Minimise cost.",
    },
    "diversity_mixed_full": {
        "label": "Mixed — EV (SoC) + HVAC + Dishwasher",
        "requires_weather": True, "expect_failure": False, "expect_clarification": False,
        "appliances": [
            {"name": "Central AC", "type": "hvac",
             "initial_temp_f": 76.0, "temp_min_f": 70.0, "temp_max_f": 78.0,
             "max_power_kw": 3.5, "cop": 3.0, "household_peak_limit": 15.0},
            {"name": "Tesla Model 3", "type": "ev",
             "start_hour": 22, "end_hour": 7, "battery_capacity_kwh": 75.0,
             "initial_soc_kwh": 18.75, "target_soc_kwh": 63.75,
             "max_charge_power_kw": 11.0, "charge_efficiency": 0.95,
             "min_soc_kwh": 5.0, "max_discharge_power_kw": 0.0,
             "household_peak_limit": 15.0},
            {"name": "Dishwasher", "type": "flexible",
             "energy_required_kwh": 3.6, "start_hour": 20, "end_hour": 23,
             "min_power_kw": 0.0, "max_power_kw": 2.0, "household_peak_limit": 15.0},
        ],
        "query": ("Schedule everything to minimise cost.\n"
                  "- Central AC: keep house 70–78°F (currently 76°F)\n"
                  "- Tesla Model 3: 25%→85%, plug in 10 PM, need by 7 AM, max 11 kW\n"
                  "- Dishwasher: 3.6 kWh, 8 PM–11 PM, max 2 kW\n"
                  "Household peak: 15 kW."),
    },
}


# ── GROUP 4: GOAL ─────────────────────────────────────────────────────────────

_GOAL_APPS = [
    {"name": "EV", "type": "flexible", "energy_required_kwh": 16.0,
     "start_hour": 22, "end_hour": 7, "min_power_kw": 0.0,
     "max_power_kw": 11.0, "household_peak_limit": 15.0},
    {"name": "Dryer", "type": "flexible", "energy_required_kwh": 4.5,
     "start_hour": 21, "end_hour": 24, "min_power_kw": 0.0,
     "max_power_kw": 4.0, "household_peak_limit": 15.0},
]

GOAL_SCENARIOS: Dict[str, Dict] = {
    "goal_cost": {
        "label": "Goal: cost only", "goal": "cost",
        "expect_failure": False, "expect_clarification": False, "appliances": _GOAL_APPS,
        "query": ("Schedule EV (16 kWh, 10 PM–7 AM, 11 kW) and dryer "
                  "(4.5 kWh, 9 PM–midnight, 4 kW) to minimise electricity bill. Peak: 15 kW."),
    },
    "goal_carbon": {
        "label": "Goal: carbon only", "goal": "carbon",
        "expect_failure": False, "expect_clarification": False, "appliances": _GOAL_APPS,
        "query": ("Schedule EV (16 kWh, 10 PM–7 AM, 11 kW) and dryer "
                  "(4.5 kWh, 9 PM–midnight, 4 kW) to minimise carbon emissions. Peak: 15 kW."),
    },
    "goal_both": {
        "label": "Goal: cost + carbon equally", "goal": "both",
        "expect_failure": False, "expect_clarification": False, "appliances": _GOAL_APPS,
        "query": ("Schedule EV (16 kWh, 10 PM–7 AM, 11 kW) and dryer "
                  "(4.5 kWh, 9 PM–midnight, 4 kW) balancing cost and carbon equally. Peak: 15 kW."),
    },
}


# ── GROUP 5: EDGE CASES ───────────────────────────────────────────────────────

EDGE_SCENARIOS: Dict[str, Dict] = {
    "edge_infeasible_energy": {
        "label": "Infeasible — 50 kWh in 1 h window",
        "expect_failure": True, "expect_clarification": False,
        "appliances": [{"name": "EV", "type": "flexible", "energy_required_kwh": 50.0,
                        "start_hour": 22, "end_hour": 23, "min_power_kw": 0.0,
                        "max_power_kw": 11.0, "household_peak_limit": 15.0}],
        "query": "Schedule EV: needs 50 kWh, available 10 PM–11 PM only, max 11 kW. Peak: 15 kW.",
    },
    "edge_infeasible_peak": {
        "label": "Infeasible — combined load > peak limit",
        "expect_failure": True, "expect_clarification": False,
        "appliances": [
            {"name": "EV", "type": "flexible", "energy_required_kwh": 11.0,
             "start_hour": 22, "end_hour": 23, "min_power_kw": 0.0,
             "max_power_kw": 11.0, "household_peak_limit": 10.0},
            {"name": "Dryer", "type": "flexible", "energy_required_kwh": 4.0,
             "start_hour": 22, "end_hour": 23, "min_power_kw": 0.0,
             "max_power_kw": 4.0, "household_peak_limit": 10.0},
        ],
        "query": ("Schedule EV (11 kWh, 10 PM–11 PM, 11 kW) and dryer "
                  "(4 kWh, 10 PM–11 PM, 4 kW). Household peak: 10 kW."),
    },
    "edge_zero_energy": {
        "label": "Edge — zero energy requirement",
        "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "Dishwasher", "type": "flexible", "energy_required_kwh": 0.0,
                        "start_hour": 20, "end_hour": 23, "min_power_kw": 0.0,
                        "max_power_kw": 2.0, "household_peak_limit": 15.0}],
        "query": "Dishwasher: 0 kWh needed, 8 PM–11 PM, max 2 kW. Peak: 15 kW. Minimise cost.",
    },
    "edge_midnight_crossing": {
        "label": "Edge — window crosses midnight",
        "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "EV", "type": "flexible", "energy_required_kwh": 16.0,
                        "start_hour": 22, "end_hour": 7, "min_power_kw": 0.0,
                        "max_power_kw": 11.0, "household_peak_limit": 15.0}],
        "query": "Schedule EV: 16 kWh, 10 PM–7 AM, max 11 kW. Peak: 15 kW. Minimise cost.",
    },
    "edge_tight_feasible": {
        "label": "Edge — boundary feasible (11 kWh in 1 h)",
        "expect_failure": False, "expect_clarification": False,
        "appliances": [{"name": "EV", "type": "flexible", "energy_required_kwh": 11.0,
                        "start_hour": 22, "end_hour": 23, "min_power_kw": 0.0,
                        "max_power_kw": 11.0, "household_peak_limit": 15.0}],
        "query": "Schedule EV: 11 kWh, 10 PM–11 PM only, max 11 kW. Peak: 15 kW.",
    },
}


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

ALL_SCENARIO_GROUPS = {
    "sparsity":  SPARSITY_SCENARIOS,
    "count":     COUNT_SCENARIOS,
    "diversity": DIVERSITY_SCENARIOS,
    "goal":      GOAL_SCENARIOS,
    "edge":      EDGE_SCENARIOS,
}


def run_all(
    groups: List[str],
    archs: List[str],
    verbose: bool = True,
) -> List[Dict]:
    """
    Run every (scenario, architecture) combination and return a flat list of records.
    Agents and verifier LLM are created once and reused across scenarios.
    """
    # Lazy-init expensive objects
    agent        = create_dr_agent()        if any(a != ARCH_BASELINE for a in archs) else None
    verifier_llm = create_baseline_llm()   if ARCH_AGENTIC in archs else None

    all_records: List[Dict] = []

    for group_name in groups:
        scenarios = ALL_SCENARIO_GROUPS.get(group_name, {})
        group_records: List[Dict] = []

        for key, sc in scenarios.items():
            apps          = sc.get("ground_truth_appliances") or sc.get("appliances")
            goal          = sc.get("goal", "cost")
            needs_weather = sc.get("requires_weather", False)
            n_apps        = sc.get("n_appliances", len(apps) if apps else 0)
            expect_fail   = sc.get("expect_failure", False)
            expect_clarif = sc.get("expect_clarification", False)
            label         = sc.get("label", key)

            if verbose:
                print(f"\n  {'─'*70}")
                print(f"  [{group_name.upper()}] {key}  —  {label}")

            for arch in archs:
                if verbose:
                    print(f"    ▷ {ARCH_LABELS[arch]}")

                # ── Run architecture ──────────────────────────────────────────
                if arch == ARCH_BASELINE:
                    raw = run_baseline_architecture(
                        appliances=apps or [],
                        goal=goal,
                        requires_weather=needs_weather,
                    )
                elif arch == ARCH_SINGLEPASS:
                    raw = run_single_pass_architecture(
                        query=sc["query"],
                        appliances=apps,
                        goal=goal,
                        requires_weather=needs_weather,
                        agent=agent,
                    )
                else:  # ARCH_AGENTIC
                    raw = run_agentic_architecture(
                        query=sc["query"],
                        appliances=apps,
                        goal=goal,
                        requires_weather=needs_weather,
                        agent=agent,
                        verifier_llm=verifier_llm,
                    )

                # ── Build metrics record ──────────────────────────────────────
                rec = _build_record(
                    key=key, group=group_name, label=label,
                    arch=arch, arch_output=raw,
                    appliances=apps or [],
                    expect_failure=expect_fail,
                    expect_clarification=expect_clarif,
                    n_appliances=n_apps,
                    goal=goal,
                )
                group_records.append(rec)

                if verbose:
                    icon = "✓" if rec["passed"] else "✗"
                    f = _fmt(rec["feasibility_pct"])
                    c = _fmt(rec["cost_reduced_pct"])
                    i = _fmt(rec["iters_to_pass"], suffix="", decimals=0)
                    t = _fmt(rec["total_tool_calls"], suffix="", decimals=0)
                    fa = _fmt(rec["faithfulness_pct"])
                    print(f"      {icon} feasible={f}  cost↓={c}  "
                          f"iters={i}  calls={t}  faithful={fa}  "
                          f"({rec['elapsed_s']:.1f}s)")
                    if rec.get("error"):
                        print(f"      ⚠  {rec['error'][:100]}")

        # Fill in group-level safe_fail_pct for every architecture
        for arch in archs:
            arch_recs = [r for r in group_records if r["arch"] == arch]
            sf = compute_safe_fail_pct(arch_recs)
            for r in arch_recs:
                r["safe_fail_pct"] = sf

        all_records.extend(group_records)

    return all_records


# ==============================================================================
# SUMMARY PRINTER
# ==============================================================================

def _fmt(val, suffix="%", decimals=1) -> str:
    if val is None:
        return "—"
    if decimals == 0:
        return f"{int(val)}{suffix}"
    return f"{val:.{decimals}f}{suffix}"


def _print_summary(records: List[Dict]) -> str:
    lines = []
    W  = 120
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [f"\n{'═'*W}",
              f"  DRAGENT ABLATION RESULTS  ·  {ts}",
              f"  Architectures: {' | '.join(ARCH_LABELS[a] for a in ALL_ARCHS)}",
              "═"*W]

    HDR = (f"  {'Scenario':<40} {'Group':<10} {'Architecture':<26} "
           f"{'Feasible%':>9} {'Cost↓%':>7} {'Iters':>6} "
           f"{'Calls':>6} {'SafeFail%':>10} {'Faithful%':>10}  Pass")
    SEP = "  " + "─" * (W - 2)

    groups_seen = []
    for rec in records:
        g = rec["group"]
        if g not in groups_seen:
            lines += [f"\n── {g.upper()} {'─'*(W-6-len(g))}", HDR, SEP]
            groups_seen.append(g)
        icon = "✓" if rec["passed"] else "✗"
        lines.append(
            f"  {rec['label']:<40} {rec['group']:<10} {rec['arch_label']:<26} "
            f"{_fmt(rec['feasibility_pct']):>9} "
            f"{_fmt(rec['cost_reduced_pct']):>7} "
            f"{_fmt(rec['iters_to_pass'], suffix='', decimals=0):>6} "
            f"{_fmt(rec['total_tool_calls'], suffix='', decimals=0):>6} "
            f"{_fmt(rec['safe_fail_pct']):>10} "
            f"{_fmt(rec['faithfulness_pct']):>10}  {icon}"
        )

    # Per-architecture aggregate
    lines += [f"\n{'═'*W}", "  AGGREGATES BY ARCHITECTURE", "═"*W,
              f"  {'Architecture':<26} {'Feasible%':>9} {'Cost↓%':>7} "
              f"{'Iters':>6} {'Calls':>6} {'SafeFail%':>10} {'Faithful%':>10}  {'Pass':>6}"]
    lines.append(SEP)

    def avg(recs, key):
        v = [r[key] for r in recs if r.get(key) is not None]
        return sum(v)/len(v) if v else None

    for arch in ALL_ARCHS:
        ar = [r for r in records if r["arch"] == arch]
        if not ar:
            continue
        pass_rate = f"{sum(r['passed'] for r in ar)}/{len(ar)}"
        lines.append(
            f"  {ARCH_LABELS[arch]:<26} "
            f"{_fmt(avg(ar,'feasibility_pct')):>9} "
            f"{_fmt(avg(ar,'cost_reduced_pct')):>7} "
            f"{_fmt(avg(ar,'iters_to_pass'), suffix='', decimals=1):>6} "
            f"{_fmt(avg(ar,'total_tool_calls'), suffix='', decimals=1):>6} "
            f"{_fmt(avg(ar,'safe_fail_pct')):>10} "
            f"{_fmt(avg(ar,'faithfulness_pct')):>10}  {pass_rate:>6}"
        )

    # Count scaling latency subplot
    count_recs = [r for r in records if r["group"] == "count"]
    if count_recs:
        lines += [f"\n── COUNT SCALING LATENCY (seconds per architecture) {'─'*50}"]
        ns = sorted(set(r["n_appliances"] for r in count_recs))
        lines.append(f"  {'n':>4}  " + "  ".join(f"{ARCH_LABELS[a][:18]:>18}" for a in ALL_ARCHS))
        for n in ns:
            row = f"  {n:>4}  "
            for arch in ALL_ARCHS:
                hits = [r for r in count_recs if r["n_appliances"]==n and r["arch"]==arch]
                t = hits[0]["elapsed_s"] if hits and hits[0]["elapsed_s"] is not None else None
                row += f"{_fmt(t, suffix='s', decimals=3):>20}"
            lines.append(row)

    lines.append(f"\n{'═'*W}\n")
    out = "\n".join(lines)
    print(out)
    return out


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="DRAgent Ablation Study — 3 architectures")
    parser.add_argument("--group",
                        choices=list(ALL_SCENARIO_GROUPS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--arch",
                        choices=ALL_ARCHS + ["all"],
                        default="all",
                        help="Which architecture(s) to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Baseline only (no LLM calls)")
    parser.add_argument("--output", default="ablation_results.json")
    args = parser.parse_args()

    groups = (list(ALL_SCENARIO_GROUPS.keys()) if args.group == "all"
              else [args.group])
    archs  = ([ARCH_BASELINE] if args.dry_run
              else ALL_ARCHS if args.arch == "all"
              else [args.arch])

    print("=" * 80)
    print("  DRAGENT ABLATION STUDY — THREE ARCHITECTURES")
    print(f"  Groups : {groups}")
    print(f"  Archs  : {[ARCH_LABELS[a] for a in archs]}")
    print("  Metrics: Feasibility%  Cost↓%  Iters  Calls  SafeFail%  Faithfulness%")
    print("=" * 80)

    records = run_all(groups=groups, archs=archs, verbose=True)
    summary = _print_summary(records)

    payload = {
        "timestamp":  datetime.now().isoformat(),
        "groups":     groups,
        "archs":      archs,
        "arch_labels": ARCH_LABELS,
        "records":    records,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open("ablation_summary.txt", "w") as f:
        f.write(summary)

    print(f"\nResults → {args.output}")
    print("Summary → ablation_summary.txt")


if __name__ == "__main__":
    main()
