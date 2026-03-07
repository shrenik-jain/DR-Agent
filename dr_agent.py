"""
DRAgent: Agentic AI for Residential Demand Response
Supports three appliance classes:
  1. Flexible loads  — dishwasher, dryer, washer (energy-window model)
  2. EVs             — SoC-dynamics battery model with optional V2G
  3. HVAC            — RC thermal circuit model with comfort-band constraints

References
----------
[1] Boyd & Vandenberghe, Convex Optimization, Cambridge, 2004.
[2] Sortomme & El-Sharkawi, IEEE Trans. Smart Grid 2(1):131-138, 2011.
[3] Rotering & Ilic, IEEE Trans. Power Syst. 26(3):1021-1029, 2011.
[4] Kempton & Tomic, Journal of Power Sources 144(1):268-279, 2005.
[5] Koch, Mathieu & Callaway, Proc. PSCC, 2011.
[6] Callaway & Hiskens, Proc. IEEE 99(1):184-199, 2011.
[7] ASHRAE, Handbook — Fundamentals, 2021.
[8] Wetter, J. Building Performance Simulation 4(3):185-203, 2011.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import cvxpy as cp
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

from input_validation import validate_appliance_specs


# ============================================================================
# TOOLS: Data Retrieval
# ============================================================================

@tool
def fetch_sdge_prices(date: Optional[str] = None) -> str:
    """
    Fetch SDG&E Time-of-Use electricity prices for residential customers.

    Args:
        date: Date string in YYYY-MM-DD format (defaults to tomorrow)

    Returns:
        JSON string with hourly prices for 24 hours
    """
    prices = []
    for hour in range(24):
        if 0 <= hour < 6:
            price = 0.238    # Super Off-Peak
        elif 16 <= hour < 21:
            price = 0.52     # On-Peak
        else:
            price = 0.36     # Off-Peak

        prices.append({
            "hour": hour,
            "price_per_kwh": price,
            "period": (
                "super_off_peak" if hour < 6
                else ("on_peak" if 16 <= hour < 21 else "off_peak")
            )
        })

    return json.dumps({
        "utility": "SDG&E",
        "tariff": "EV-TOU-5",
        "date": date or "tomorrow",
        "prices": prices,
        "currency": "USD"
    }, indent=2)


@tool
def fetch_caiso_carbon(date: Optional[str] = None) -> str:
    """
    Fetch CAISO grid carbon intensity forecast (lbs CO2 per MWh).

    Args:
        date: Date string in YYYY-MM-DD format (defaults to tomorrow)

    Returns:
        JSON string with hourly carbon intensity for 24 hours
    """
    carbon_intensity = []
    for hour in range(24):
        if 0 <= hour < 6:
            intensity = 250 + hour * 5
        elif 6 <= hour < 10:
            intensity = 300 + (hour - 6) * 40
        elif 10 <= hour < 16:
            intensity = 200 + abs(hour - 13) * 10
        elif 16 <= hour < 22:
            intensity = 400 + (hour - 16) * 25
        else:
            intensity = 400 - (hour - 22) * 25

        carbon_intensity.append({
            "hour": hour,
            "carbon_intensity_lbs_per_mwh": intensity,
            "intensity_level": (
                "low" if intensity < 300
                else ("medium" if intensity < 400 else "high")
            )
        })

    return json.dumps({
        "source": "CAISO",
        "region": "California",
        "date": date or "tomorrow",
        "carbon_data": carbon_intensity,
        "unit": "lbs_co2_per_mwh"
    }, indent=2)


@tool
def fetch_weather_forecast(date: Optional[str] = None) -> str:
    """
    Fetch hourly outdoor temperature forecast for San Diego, CA.
    Used by the HVAC thermal model to compute heat infiltration.

    Args:
        date: Date string in YYYY-MM-DD format (defaults to tomorrow)

    Returns:
        JSON string with hourly outdoor temperatures (°F) for 24 hours
    """
    # Typical San Diego summer day profile (°F)
    # Cool nights (~65°F), warm afternoons (~85°F)
    temps_f = []
    for hour in range(24):
        if 0 <= hour < 6:
            temp = 65.0 + hour * 0.5            # Cool early morning
        elif 6 <= hour < 14:
            temp = 67.5 + (hour - 6) * 2.0     # Morning warm-up
        elif 14 <= hour < 18:
            temp = 83.5 + (hour - 14) * 0.5    # Afternoon peak ~85.5°F
        else:
            temp = 85.5 - (hour - 18) * 1.5    # Evening cool-down

        temps_f.append({
            "hour": hour,
            "temperature_f": round(temp, 1),
            "temperature_c": round((temp - 32) * 5 / 9, 2)
        })

    return json.dumps({
        "source": "weather_forecast",
        "location": "San Diego, CA",
        "date": date or "tomorrow",
        "hourly_temperatures": temps_f,
        "units": "fahrenheit"
    }, indent=2)


@tool
def check_required_inputs(appliances_json: str) -> str:
    """
    Check whether appliance specs have all required inputs for optimization.
    Call this BEFORE solve_dr_optimization. Pass the JSON list (or single object)
    of appliance specs you have inferred from the user.

    Returns JSON with:
      - ready: true if all required fields are present (or filled by defaults)
      - specs_with_defaults: use this as appliances_json when calling solve_dr_optimization if ready is true
      - follow_up_questions: list of questions to ask the user if ready is false
      - missing_by_appliance: which fields are missing per appliance
      - defaults_applied: which fields were filled from defaults
      - error: message if JSON was invalid

    If ready is false, respond to the user with the follow_up_questions and do NOT
    call solve_dr_optimization until the user provides the missing information.
    If ready is true, use specs_with_defaults and proceed to fetch prices/carbon/weather
    and then call solve_dr_optimization.
    """
    result = validate_appliance_specs(appliances_json)
    return json.dumps({
        "ready": result["ready"],
        "specs_with_defaults": result.get("specs_with_defaults", []),
        "follow_up_questions": result.get("follow_up_questions", []),
        "missing_by_appliance": result.get("missing_by_appliance", {}),
        "defaults_applied": result.get("defaults_applied", {}),
        "error": result.get("error"),
    }, indent=2)


# ============================================================================
# CORE HELPER: Flexible Load Model
# ============================================================================

def _build_flex_variables_and_constraints(
    app: Dict, H: int
) -> Tuple[cp.Variable, List]:
    """
    Build CVXPY variable and constraints for a standard flexible load.

    Model: power x[h] can be distributed freely within [alpha, beta],
    subject to total energy == E_a and per-hour power bounds [1].

    Required fields: name, energy_required_kwh, start_hour, end_hour,
                     max_power_kw, min_power_kw (optional, default 0)
    """
    alpha = app["start_hour"]
    beta  = app["end_hour"]
    E_a   = float(app["energy_required_kwh"])
    P_min = float(app.get("min_power_kw", 0.0))
    P_max = float(app["max_power_kw"])

    window = (
        list(range(alpha, beta + 1)) if beta >= alpha
        else list(range(alpha, H)) + list(range(0, beta + 1))
    )

    x = cp.Variable(H, nonneg=True, name=app["name"])
    constraints = []

    # Total energy completeness
    constraints.append(cp.sum([x[h] for h in window]) == E_a)

    # Per-hour power bounds inside window
    for h in window:
        constraints.append(x[h] >= P_min)
        constraints.append(x[h] <= P_max)

    # Zero outside window
    for h in range(H):
        if h not in window:
            constraints.append(x[h] == 0)

    return x, constraints


# ============================================================================
# CORE HELPER: EV Battery Model
# ============================================================================

def _build_ev_variables_and_constraints(
    ev: Dict, H: int
) -> Tuple[cp.Variable, cp.Variable, cp.Variable, List]:
    """
    Build CVXPY variables and constraints for an EV using the
    SoC-dynamics battery model [2, 3].

    SoC recurrence (per hour h in plug-in window [alpha_v, beta_v]):
        s[h] = s[h-1] + eta_ch * p_ch[h] - (1/eta_dis) * p_dis[h]

    Required EV fields:
        name, start_hour, end_hour,
        initial_soc_kwh, target_soc_kwh, battery_capacity_kwh,
        max_charge_power_kw,
        charge_efficiency     (default 0.95)
        min_soc_kwh           (default 0)
        max_discharge_power_kw(default 0 — charge-only; >0 enables V2G [4])
        discharge_efficiency  (default 0.95)
    """
    alpha     = ev["start_hour"]
    beta      = ev["end_hour"]
    S_init    = float(ev["initial_soc_kwh"])
    S_target  = float(ev["target_soc_kwh"])
    S_max     = float(ev["battery_capacity_kwh"])
    S_min     = float(ev.get("min_soc_kwh", 0.0))
    P_ch_max  = float(ev["max_charge_power_kw"])
    P_dis_max = float(ev.get("max_discharge_power_kw", 0.0))
    eta_ch    = float(ev.get("charge_efficiency", 0.95))
    eta_dis   = float(ev.get("discharge_efficiency", 0.95))

    assert 0 < eta_ch  <= 1, "charge_efficiency must be in (0,1]"
    assert 0 < eta_dis <= 1, "discharge_efficiency must be in (0,1]"
    assert S_target <= S_max, "target_soc_kwh cannot exceed battery_capacity_kwh"
    assert S_init   <= S_max, "initial_soc_kwh cannot exceed battery_capacity_kwh"

    window = (
        list(range(alpha, beta + 1)) if beta >= alpha
        else list(range(alpha, H)) + list(range(0, beta + 1))
    )

    p_ch = cp.Variable(H, nonneg=True, name=f"{ev['name']}_p_ch")
    p_dis= cp.Variable(H, nonneg=True, name=f"{ev['name']}_p_dis")
    s    = cp.Variable(H, nonneg=True, name=f"{ev['name']}_soc")

    constraints = []

    for idx, h in enumerate(window):
        soc_prev = S_init if idx == 0 else s[window[idx - 1]]

        # SoC recurrence [3]
        constraints.append(
            s[h] == soc_prev + eta_ch * p_ch[h] - (1.0 / eta_dis) * p_dis[h]
        )
        constraints.append(s[h] >= S_min)
        constraints.append(s[h] <= S_max)
        constraints.append(p_ch[h]  <= P_ch_max)
        constraints.append(p_dis[h] <= P_dis_max)

    # Must reach target SoC by departure (inequality gives solver freedom [2])
    constraints.append(s[beta] >= S_target)

    # Zero and frozen outside window
    last_h = window[-1]
    for h in range(H):
        if h not in window:
            constraints.append(p_ch[h]  == 0)
            constraints.append(p_dis[h] == 0)
            constraints.append(s[h]     == s[last_h])

    return p_ch, p_dis, s, constraints


# ============================================================================
# CORE HELPER: HVAC Thermal Model  ← NEW
# ============================================================================

def _build_hvac_variables_and_constraints(
    hvac: Dict,
    H: int,
    outdoor_temps_f: np.ndarray
) -> Tuple[cp.Variable, cp.Variable, List]:
    """
    Build CVXPY variables and constraints for an HVAC unit using the
    first-order RC (Equivalent Thermal Parameter) building model [5, 6].

    Thermal recurrence (per hour h):
        T[h] = T[h-1] + (T_out[h] - T[h-1]) / (R*C)
                       - (COP / C) * p_hvac[h]

    where:
        T[h]       — indoor temperature at end of hour h  (°F)
        T_out[h]   — outdoor temperature at hour h        (°F)
        R          — thermal resistance  (°F / kW)        [7]
        C          — thermal capacitance (kWh / °F)       [6]
        COP        — cooling coefficient of performance   [7]
        p_hvac[h]  — electrical power draw of HVAC (kW)

    The comfort band T_min[h] <= T[h] <= T_max[h] enforces occupant
    preferences and can encode setback schedules (tighter when home,
    relaxed when away or sleeping) [8].

    Required HVAC fields:
        name                  — display name
        initial_temp_f        — T_init: indoor temperature at hour 0 (°F)
                                default 72.0
        thermal_resistance    — R (°F/kW); typical US home ~2–8
                                default 4.0  (average US home)
        thermal_capacitance   — C (kWh/°F); typical US home ~1–3
                                default 2.0  (average US home)
        cop                   — COP of AC/heat-pump unit; typical 2.5–4.0 [7]
                                default 3.0  (modern split-system AC)
        max_power_kw          — P_hvac_max: rated electrical draw of unit (kW)
                                default 3.5  (typical 2-ton residential unit)
        min_power_kw          — P_hvac_min: 0 for on/off, >0 for inverter units
                                default 0.0
        temp_min_f            — scalar or 24-element list: lower comfort bound (°F)
                                default 70.0
        temp_max_f            — scalar or 24-element list: upper comfort bound (°F)
                                default 78.0
        cooling_only          — bool, default True (set False for heat-pump heating)

    Returns:
        p_hvac      : cp.Variable(H)  — HVAC electrical power (kW)
        T_in        : cp.Variable(H)  — indoor temperature (°F)
        constraints : list of CVXPY constraints
    """
    name         = hvac["name"]
    T_init       = float(hvac.get("initial_temp_f",       72.0))
    R            = float(hvac.get("thermal_resistance",    4.0))   # °F/kW
    C            = float(hvac.get("thermal_capacitance",   2.0))   # kWh/°F
    COP          = float(hvac.get("cop",                   3.0))
    P_max        = float(hvac.get("max_power_kw",          3.5))
    P_min        = float(hvac.get("min_power_kw",          0.0))
    cooling_only = bool(hvac.get("cooling_only",           True))

    # Comfort bounds — accept scalar or per-hour list [8]
    raw_min = hvac.get("temp_min_f", 70.0)
    raw_max = hvac.get("temp_max_f", 78.0)
    T_min = np.full(H, float(raw_min)) if np.isscalar(raw_min) else np.array(raw_min, dtype=float)
    T_max = np.full(H, float(raw_max)) if np.isscalar(raw_max) else np.array(raw_max, dtype=float)

    assert len(T_min) == H and len(T_max) == H, "temp_min_f / temp_max_f must be scalar or 24-element list"
    assert np.all(T_min <= T_max), "temp_min_f must be <= temp_max_f for all hours"
    assert R > 0 and C > 0 and COP > 0

    # ── Decision variables ────────────────────────────────────────────────────
    p_hvac = cp.Variable(H, nonneg=True, name=f"{name}_power")
    T_in   = cp.Variable(H,              name=f"{name}_temp")

    constraints = []

    # ── Thermal dynamics — RC model [5, 6] ───────────────────────────────────
    # Discretised ETP equation (Δt = 1 hour):
    #   T[h] = T[h-1] + (T_out[h] - T[h-1])/(R*C) - (COP/C)*p_hvac[h]
    for h in range(H):
        T_prev = T_init if h == 0 else T_in[h - 1]

        constraints.append(
            T_in[h] == T_prev
                       + (outdoor_temps_f[h] - T_prev) / (R * C)
                       - (COP / C) * p_hvac[h]
        )

    # ── Comfort band [8] ─────────────────────────────────────────────────────
    for h in range(H):
        constraints.append(T_in[h] >= T_min[h])
        constraints.append(T_in[h] <= T_max[h])

    # ── Power bounds ──────────────────────────────────────────────────────────
    for h in range(H):
        constraints.append(p_hvac[h] >= P_min)
        constraints.append(p_hvac[h] <= P_max)

    # ── Cooling-only constraint (standard residential AC) ─────────────────────
    # The HVAC can only remove heat (p_hvac >= 0 already enforced by nonneg=True).
    # For a heat-pump in heating mode, the COP sign would be reversed and this
    # constraint relaxed — left as a future extension.
    if cooling_only:
        # nonneg=True already guarantees p_hvac[h] >= 0; nothing extra needed.
        pass

    return p_hvac, T_in, constraints


# ============================================================================
# TOOL: Unified Convex Optimizer
# ============================================================================

@tool
def solve_dr_optimization(
    appliances_json: str,
    prices_json: str,
    carbon_json: str,
    weather_json: str = "{}",
    optimization_goal: str = "cost"
) -> str:
    """
    Solve the demand response optimization problem to schedule:
      - Flexible loads (dishwasher, dryer, washer): energy-window model
      - EVs: SoC-dynamics battery model with optional V2G
      - HVAC: RC thermal circuit model with comfort-band constraints

    Appliance type field:
        "flexible"  — standard shiftable load  (default if omitted)
        "ev"        — electric vehicle
        "hvac"      — heating/cooling unit

    Args:
        appliances_json  : JSON list (or single object) of appliance specs
        prices_json      : JSON from fetch_sdge_prices
        carbon_json      : JSON from fetch_caiso_carbon
        weather_json     : JSON from fetch_weather_forecast (required for HVAC)
        optimization_goal: "cost" | "carbon" | "both"

    Returns:
        JSON with optimal schedule, temperature trajectories (HVAC),
        SoC trajectories (EV), and savings metrics.
    """
    try:
        # ── Parse inputs ──────────────────────────────────────────────────────
        appliances_raw = json.loads(appliances_json)
        price_data     = json.loads(prices_json)
        carbon_data    = json.loads(carbon_json)
        weather_data   = json.loads(weather_json) if weather_json.strip() not in ("{}", "") else {}

        appliances = [appliances_raw] if isinstance(appliances_raw, dict) else appliances_raw

        price_list  = price_data.get("prices",       price_data)  if isinstance(price_data,   dict) else price_data
        carbon_list = carbon_data.get("carbon_data",  carbon_data) if isinstance(carbon_data,  dict) else carbon_data

        H = 24
        prices = np.zeros(H)
        carbon = np.zeros(H)

        for p in price_list:
            prices[p["hour"]] = p["price_per_kwh"]
        for c in carbon_list:
            carbon[c["hour"]] = c["carbon_intensity_lbs_per_mwh"]

        prices[prices == 0] = np.mean(prices[prices > 0]) if np.any(prices > 0) else 0.36
        carbon[carbon == 0] = np.mean(carbon[carbon > 0]) if np.any(carbon > 0) else 350.0

        # Outdoor temperature array for HVAC (°F)
        outdoor_temps = np.full(H, 75.0)   # default if no weather data
        if weather_data:
            temp_list = weather_data.get("hourly_temperatures", [])
            for t in temp_list:
                outdoor_temps[t["hour"]] = t["temperature_f"]

        P_max_house = float(appliances[0].get("household_peak_limit", 15.0))

        # ── Classify appliances ───────────────────────────────────────────────
        flex_apps = [a for a in appliances if a.get("type", "flexible") == "flexible"]
        ev_apps   = [a for a in appliances if a.get("type", "").lower() == "ev"]
        hvac_apps = [a for a in appliances if a.get("type", "").lower() == "hvac"]

        # ── Build variables and constraints per class ─────────────────────────
        flex_vars        = []   # list of (x_var, app)
        flex_constraints = []
        for app in flex_apps:
            xv, cons = _build_flex_variables_and_constraints(app, H)
            flex_vars.append((xv, app))
            flex_constraints.extend(cons)

        ev_vars        = []     # list of (p_ch, p_dis, s_var, app)
        ev_constraints = []
        for ev in ev_apps:
            p_ch, p_dis, s_var, cons = _build_ev_variables_and_constraints(ev, H)
            ev_vars.append((p_ch, p_dis, s_var, ev))
            ev_constraints.extend(cons)

        hvac_vars        = []   # list of (p_hvac, T_in_var, app)
        hvac_constraints = []
        for hvac in hvac_apps:
            p_hvac, T_in_var, cons = _build_hvac_variables_and_constraints(
                hvac, H, outdoor_temps
            )
            hvac_vars.append((p_hvac, T_in_var, hvac))
            hvac_constraints.extend(cons)

        # ── Total grid draw per hour ──────────────────────────────────────────
        def total_draw_expr(h):
            terms = []
            for xv, _ in flex_vars:
                terms.append(xv[h])
            for p_ch, p_dis, _, _ in ev_vars:
                terms.append(p_ch[h] - p_dis[h])   # net draw; V2G earns revenue [4]
            for p_hvac, _, _ in hvac_vars:
                terms.append(p_hvac[h])
            return cp.sum(terms) if terms else cp.Constant(0)

        # ── Objective [1] ─────────────────────────────────────────────────────
        total_cost   = cp.sum([prices[h] * total_draw_expr(h) for h in range(H)])
        total_carbon = cp.sum([carbon[h] * total_draw_expr(h) for h in range(H)]) / 1000.0

        if optimization_goal == "cost":
            objective = cp.Minimize(total_cost)
        elif optimization_goal == "carbon":
            objective = cp.Minimize(total_carbon)
        else:
            objective = cp.Minimize(total_cost / 10.0 + total_carbon / 30.0)

        # ── Household peak constraint (charge-side draw only) ─────────────────
        peak_constraints = []
        for h in range(H):
            charge_terms = []
            for xv, _ in flex_vars:
                charge_terms.append(xv[h])
            for p_ch, _, _, _ in ev_vars:
                charge_terms.append(p_ch[h])
            for p_hvac, _, _ in hvac_vars:
                charge_terms.append(p_hvac[h])
            if charge_terms:
                peak_constraints.append(cp.sum(charge_terms) <= P_max_house)

        # ── Solve ─────────────────────────────────────────────────────────────
        all_constraints = (
            flex_constraints + ev_constraints
            + hvac_constraints + peak_constraints
        )
        problem = cp.Problem(objective, all_constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            return json.dumps({
                "status": "failed",
                "error": (
                    f"Optimization status: {problem.status}. "
                    "Check that HVAC comfort bounds are physically achievable "
                    "given the thermal parameters and outdoor temperatures, "
                    "and that EV energy targets fit within the plug-in window."
                )
            })

        # ── Extract: flexible loads ───────────────────────────────────────────
        schedule = {}
        for xv, app in flex_vars:
            vals = [float(xv.value[h]) if xv.value[h] > 0.01 else 0.0 for h in range(H)]
            schedule[app["name"]] = {
                "type": "flexible",
                "hourly_consumption_kwh": [round(v, 3) for v in vals],
                "total_energy_kwh":       round(sum(vals), 3),
                "operating_hours":        [h for h in range(H) if vals[h] > 0.05]
            }

        # ── Extract: EVs ─────────────────────────────────────────────────────
        for p_ch, p_dis, s_var, ev in ev_vars:
            ch_vals  = [max(0.0, float(p_ch.value[h]))  for h in range(H)]
            dis_vals = [max(0.0, float(p_dis.value[h])) for h in range(H)]
            soc_vals = [round(float(s_var.value[h]), 3)  for h in range(H)]

            schedule[ev["name"]] = {
                "type":                        "ev",
                "hourly_charge_kw":            [round(v, 3) for v in ch_vals],
                "hourly_discharge_kw":         [round(v, 3) for v in dis_vals],
                "hourly_net_draw_kw":          [round(ch_vals[h] - dis_vals[h], 3) for h in range(H)],
                "soc_trajectory_kwh":          soc_vals,
                "final_soc_kwh":               round(soc_vals[ev["end_hour"]], 3),
                "target_soc_kwh":              ev["target_soc_kwh"],
                "total_energy_charged_kwh":    round(sum(ch_vals), 3),
                "total_energy_discharged_kwh": round(sum(dis_vals), 3),
                "charging_hours":              [h for h in range(H) if ch_vals[h]  > 0.05],
                "v2g_hours":                   [h for h in range(H) if dis_vals[h] > 0.05]
            }

        # ── Extract: HVAC ─────────────────────────────────────────────────────
        for p_hvac, T_in_var, hvac in hvac_vars:
            pwr_vals  = [max(0.0, float(p_hvac.value[h]))   for h in range(H)]
            temp_vals = [round(float(T_in_var.value[h]), 2) for h in range(H)]

            # Reconstruct comfort bounds for reporting
            raw_min = hvac.get("temp_min_f", 70.0)
            raw_max = hvac.get("temp_max_f", 78.0)
            t_min_arr = (np.full(H, float(raw_min)) if np.isscalar(raw_min)
                         else np.array(raw_min, dtype=float))
            t_max_arr = (np.full(H, float(raw_max)) if np.isscalar(raw_max)
                         else np.array(raw_max, dtype=float))

            schedule[hvac["name"]] = {
                "type":                     "hvac",
                "hourly_power_kw":          [round(v, 3) for v in pwr_vals],
                "hourly_indoor_temp_f":     temp_vals,
                "hourly_outdoor_temp_f":    [round(outdoor_temps[h], 1) for h in range(H)],
                "comfort_band_min_f":       [round(t_min_arr[h], 1) for h in range(H)],
                "comfort_band_max_f":       [round(t_max_arr[h], 1) for h in range(H)],
                "total_energy_kwh":         round(sum(pwr_vals), 3),
                "active_hours":             [h for h in range(H) if pwr_vals[h] > 0.05],
                "pre_cooling_hours":        [
                    h for h in range(H)
                    if pwr_vals[h] > 0.05 and temp_vals[h] <= t_min_arr[h] + 0.5
                ],
                "peak_avoidance_hours":     [
                    h for h in range(H)
                    if pwr_vals[h] < 0.1 and 16 <= h < 21
                ]
            }

        # ── Baseline cost/carbon (no DR) ──────────────────────────────────────
        # Flex: run at full power starting at alpha
        # EV:   charge at full rate starting at alpha
        # HVAC: run at fixed thermostat (average between T_min and T_max)
        baseline_cost   = 0.0
        baseline_carbon = 0.0

        for _, app in flex_vars:
            alpha    = app["start_hour"]
            E_a      = float(app["energy_required_kwh"])
            P_max_fl = float(app["max_power_kw"])
            h_needed = int(np.ceil(E_a / P_max_fl))
            for i in range(h_needed):
                h     = (alpha + i) % H
                power = min(P_max_fl, E_a - i * P_max_fl)
                if power > 0:
                    baseline_cost   += prices[h] * power
                    baseline_carbon += (carbon[h] / 1000.0) * power

        for _, _, _, ev in ev_vars:
            alpha        = ev["start_hour"]
            S_init_ev    = float(ev["initial_soc_kwh"])
            S_target_ev  = float(ev["target_soc_kwh"])
            eta_ch_ev    = float(ev.get("charge_efficiency", 0.95))
            P_ch_max_ev  = float(ev["max_charge_power_kw"])
            energy_needed = (S_target_ev - S_init_ev) / eta_ch_ev
            h_needed      = int(np.ceil(energy_needed / P_ch_max_ev))
            for i in range(h_needed):
                h     = (alpha + i) % H
                power = min(P_ch_max_ev, energy_needed - i * P_ch_max_ev)
                if power > 0:
                    baseline_cost   += prices[h] * power
                    baseline_carbon += (carbon[h] / 1000.0) * power

        for _, _, hvac in hvac_vars:
            # Baseline HVAC: thermostat fixed at midpoint of comfort band [8]
            # Uses same RC model to compute actual power needed each hour
            raw_min_b = hvac.get("temp_min_f", 70.0)
            raw_max_b = hvac.get("temp_max_f", 78.0)
            t_min_b = (np.full(H, float(raw_min_b)) if np.isscalar(raw_min_b)
                       else np.array(raw_min_b, dtype=float))
            t_max_b = (np.full(H, float(raw_max_b)) if np.isscalar(raw_max_b)
                       else np.array(raw_max_b, dtype=float))

            T_setpoint = (t_min_b + t_max_b) / 2.0   # naive thermostat midpoint
            R_b     = float(hvac.get("thermal_resistance",  4.0))
            C_b     = float(hvac.get("thermal_capacitance", 2.0))
            COP_b   = float(hvac.get("cop",                 3.0))
            P_max_b = float(hvac.get("max_power_kw",        3.5))
            T_cur   = float(hvac.get("initial_temp_f",      72.0))

            for h in range(H):
                # Power needed to hit setpoint, clamped to [0, P_max]
                heat_in    = (outdoor_temps[h] - T_cur) / (R_b * C_b)
                p_needed   = max(0.0, min(P_max_b, C_b * (T_cur - T_setpoint[h] + heat_in) / COP_b))
                baseline_cost   += prices[h] * p_needed
                baseline_carbon += (carbon[h] / 1000.0) * p_needed
                T_cur = T_cur + heat_in - (COP_b / C_b) * p_needed

        # ── Optimised totals ──────────────────────────────────────────────────
        opt_cost = float(sum(
            prices[h] * (
                sum(float(xv.value[h]) for xv, _ in flex_vars)
                + sum(float(p_ch.value[h]) - float(p_dis.value[h]) for p_ch, p_dis, _, _ in ev_vars)
                + sum(float(p_hvac.value[h]) for p_hvac, _, _ in hvac_vars)
            )
            for h in range(H)
        ))
        opt_carbon = float(sum(
            carbon[h] * (
                sum(float(xv.value[h]) for xv, _ in flex_vars)
                + sum(float(p_ch.value[h]) - float(p_dis.value[h]) for p_ch, p_dis, _, _ in ev_vars)
                + sum(float(p_hvac.value[h]) for p_hvac, _, _ in hvac_vars)
            )
            for h in range(H)
        )) / 1000.0

        def safe_pct(saved, base):
            return round(100 * saved / base, 1) if base > 0 else 0.0

        return json.dumps({
            "status":             "success",
            "optimization_goal":  optimization_goal,
            "schedule":           schedule,
            "metrics": {
                "optimized_cost_dollars":   round(opt_cost, 2),
                "baseline_cost_dollars":    round(baseline_cost, 2),
                "cost_savings_dollars":     round(baseline_cost - opt_cost, 2),
                "cost_savings_percent":     safe_pct(baseline_cost - opt_cost, baseline_cost),
                "optimized_carbon_lbs":     round(opt_carbon, 2),
                "baseline_carbon_lbs":      round(baseline_carbon, 2),
                "carbon_reduction_lbs":     round(baseline_carbon - opt_carbon, 2),
                "carbon_reduction_percent": safe_pct(baseline_carbon - opt_carbon, baseline_carbon)
            }
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "status":    "error",
            "error":     str(e),
            "traceback": traceback.format_exc()
        })


# ============================================================================
# AGENT SETUP
# ============================================================================

def create_dr_agent(model_name: str = "gpt-4-turbo"):
    """Create the DRAgent using LangChain's agent framework."""

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    tools = [check_required_inputs, fetch_sdge_prices, fetch_caiso_carbon,
             fetch_weather_forecast, solve_dr_optimization]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are DRAgent, an expert residential demand response assistant that helps
homeowners optimise their electricity usage to save money and reduce carbon emissions.

Your capabilities:
1. Retrieve real-time electricity prices from SDG&E
2. Fetch grid carbon intensity data from CAISO
3. Fetch hourly outdoor temperature forecasts (required when HVAC is present)
4. Solve constrained optimisation problems for flexible loads, EV charging, and HVAC
5. Explain recommendations in clear, user-friendly language

Process:
1. UNDERSTAND  — Identify each appliance and its type: flexible | ev | hvac. Build a JSON list of appliance specs from the user message (partial specs are OK).
2. CHECK INPUTS — Call check_required_inputs with that JSON. If ready is false: respond to the user with the follow_up_questions only; do NOT call fetch_* or solve_dr_optimization. If ready is true: use the returned specs_with_defaults for the next steps.
3. RETRIEVE    — Fetch prices, carbon, and (if HVAC present) weather forecast (only when inputs are ready).
4. OPTIMISE    — Call solve_dr_optimization with specs_with_defaults and the fetched data.
5. EXPLAIN     — Report schedule, savings, indoor temperature trajectory for HVAC,
                 SoC trajectory for EV, and the reasoning behind each choice.

When the user message is vague (e.g. "help me charge my EV" with no numbers), still build the best partial spec you can (e.g. one EV with type "ev" and name "EV"), then call check_required_inputs. Use the returned follow_up_questions in your reply. Do not guess required numeric values; ask for them.

─── Appliance spec: flexible load ───────────────────────────────────────────
{{
  "name": "Dishwasher",
  "type": "flexible",
  "energy_required_kwh": 3.6,
  "start_hour": 20,
  "end_hour": 23,
  "min_power_kw": 0.0,
  "max_power_kw": 2.0,
  "household_peak_limit": 15.0
}}

─── Appliance spec: EV ──────────────────────────────────────────────────────
{{
  "name": "Tesla Model 3",
  "type": "ev",
  "start_hour": 22,
  "end_hour": 7,
  "initial_soc_kwh": 20.0,
  "target_soc_kwh": 63.75,
  "battery_capacity_kwh": 75.0,
  "max_charge_power_kw": 11.0,
  "charge_efficiency": 0.95,
  "min_soc_kwh": 5.0,
  "max_discharge_power_kw": 0.0,
  "discharge_efficiency": 0.95,
  "household_peak_limit": 15.0
}}

─── Appliance spec: HVAC ────────────────────────────────────────────────────
Only "name", "type": "hvac", and the comfort bounds are typically needed.
All physical parameters have sensible defaults — only include them if the user
explicitly provides different values.

{{
  "name": "Central AC",
  "type": "hvac",

  // Comfort bounds — the only fields users usually specify
  "temp_min_f": 70.0,              // lower comfort bound (°F); default 70.0
  "temp_max_f": 78.0,              // upper comfort bound (°F); default 78.0

  // Physical parameters — USE DEFAULTS unless user provides explicit values
  "initial_temp_f": 72.0,          // current indoor temp (°F);   default 72.0
  "max_power_kw": 3.5,             // AC unit size (kW);           default 3.5
  "cop": 3.0,                      // cooling efficiency;          default 3.0
  "thermal_resistance": 4.0,       // building insulation (°F/kW); default 4.0
  "thermal_capacitance": 2.0,      // building thermal mass;       default 2.0

  // Rarely needed
  "min_power_kw": 0.0,             // 0 for on/off; >0 for inverter; default 0.0
  "cooling_only": true,            // default true
  "household_peak_limit": 15.0
}}

HVAC parameter guidance:
- Only override a default when the user explicitly states a value (e.g. "my AC is
  4 kW", "the house is at 74°F right now", "COP is 2.8").
- temp_min_f / temp_max_f: if user says "keep it between X and Y°F", use those.
  Can be a 24-element list for a setback schedule (relaxed when away/sleeping).
- If the user mentions a well-insulated or older home, you may adjust
  thermal_resistance (higher = better insulated) — but only if stated.

For EV SoC as percentage: kwh = (pct/100) * battery_capacity_kwh
Always quantify savings and explain WHY the schedule is optimal."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=12,
        handle_parsing_errors=True
    )


# ============================================================================
# BASELINE: Zero-Shot LLM (No Tools)
# ============================================================================

def create_baseline_llm(model_name: str = "gpt-4-turbo"):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )


def run_baseline_recommendation(llm, user_query: str) -> str:
    prompt = f"""You are an expert residential energy advisor specialising in demand response
for SDG&E customers in San Diego, CA. Your job is to schedule the user's appliances
to minimise electricity cost and carbon emissions using only your general knowledge —
you do NOT have access to live price or weather data.

═══════════════════════════════════════════════════════════
SDG&E EV-TOU-5 RATE STRUCTURE (use these exact figures)
═══════════════════════════════════════════════════════════
  Super Off-Peak   12:00 AM – 6:00 AM    $0.238 / kWh   ← cheapest
  Off-Peak          6:00 AM – 4:00 PM    $0.360 / kWh
  Off-Peak          9:00 PM – 12:00 AM   $0.360 / kWh
  On-Peak           4:00 PM – 9:00 PM    $0.520 / kWh   ← most expensive

═══════════════════════════════════════════════════════════
CAISO GRID CARBON INTENSITY — typical San Diego day
═══════════════════════════════════════════════════════════
  12 AM – 6 AM   ~250–280 lbs CO₂/MWh   (low — night baseload)
  6 AM – 10 AM   ~300–460 lbs CO₂/MWh   (rising — morning gas ramp)
  10 AM – 4 PM   ~200–250 lbs CO₂/MWh   (low — solar peak)
  4 PM – 10 PM   ~400–550 lbs CO₂/MWh   (high — evening peak, solar gone)
  10 PM – 12 AM  ~350–400 lbs CO₂/MWh   (declining)

═══════════════════════════════════════════════════════════
SAN DIEGO SUMMER OUTDOOR TEMPERATURE PROFILE (typical)
═══════════════════════════════════════════════════════════
  12 AM – 6 AM   ~65–68°F   (cool)
  6 AM – 2 PM    ~68–83°F   (warming)
  2 PM – 6 PM    ~83–86°F   (hottest — peak cooling load)
  6 PM – 12 AM   ~86–70°F   (cooling down)

═══════════════════════════════════════════════════════════
SCHEDULING HEURISTICS — apply these in order
═══════════════════════════════════════════════════════════
FLEXIBLE LOADS (dishwasher, dryer, washer):
  • Shift entirely into Super Off-Peak (12 AM – 6 AM) when possible.
  • If the user's allowed window does not reach midnight, use the latest
    Off-Peak hours just before the window closes to stay out of On-Peak.
  • Never schedule inside On-Peak (4–9 PM) unless there is no alternative.

EV CHARGING:
  • Charge exclusively during Super Off-Peak (12 AM – 6 AM).
  • Calculate energy needed = (target_soc_kwh − initial_soc_kwh) / efficiency.
  • Calculate hours needed = ceil(energy_needed / max_charge_power_kw).
  • Schedule those hours starting from 12 AM; confirm they fit before 6 AM.
  • If energy needed exceeds Super Off-Peak capacity, extend into early
    Off-Peak (6 AM onward) — still far cheaper than On-Peak.

HVAC (pre-cooling strategy):
  • Pre-cool the house to near the lower comfort bound during cheap morning
    hours (6 AM – 2 PM, Off-Peak, low carbon) before the outdoor heat peaks.
  • Reduce or stop AC during On-Peak (4–9 PM) and let the building's thermal
    mass absorb the load, as long as indoor temp stays within comfort bounds.
  • Resume normal operation after 9 PM (Off-Peak rate resumes).
  • Rule of thumb: every 1 hour of pre-cooling at 3.5 kW / COP 3.0 stores
    ~10.5 kWh of cooling capacity, enough to offset ~3 hours of passive heat
    gain in a typical San Diego home.

HOUSEHOLD PEAK CONSTRAINT:
  • Check that simultaneous loads never exceed the stated peak limit (kW).
  • If they do, stagger start times by 1 hour to flatten the load profile.

═══════════════════════════════════════════════════════════
USER QUERY
═══════════════════════════════════════════════════════════
{user_query}

═══════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT
═══════════════════════════════════════════════════════════
Produce your answer in the following sections:

1. SCHEDULE — for each appliance state:
   • Exact start and end time (e.g. "12:00 AM – 3:30 AM")
   • Rate period it falls in and price per kWh
   • Energy consumed (kWh) and cost ($) for that run
   For HVAC also state: pre-cooling window, expected indoor temp range,
   and which On-Peak hours it idles.
   For EV also state: starting SoC, final SoC, kWh charged.

2. COST SUMMARY — table with columns:
   Appliance | Baseline cost ($) | Optimised cost ($) | Savings ($)
   Baseline = running each appliance immediately at the start of its allowed
   window at full power, regardless of rate period.
   Show a TOTAL row.

3. CARBON SUMMARY — same table structure for lbs CO₂.
   Use the carbon intensity figures above; interpolate if needed.

4. REASONING — 2–4 sentences explaining the key trade-offs and why
   the chosen schedule is close to optimal given the rate structure.

5. CAVEATS — one sentence noting that these are estimates based on
   typical patterns and that a real optimiser with live data may differ."""

    return llm.invoke(prompt).content


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DRAgent: Residential Demand Response (Flexible + EV + HVAC)")
    print("=" * 80)

    user_query = """I need help scheduling my home for tomorrow to save money.

I have:
- Central AC. The house is currently at 76°F. I want to keep it between
  70°F and 78°F at all times.

- A Tesla Model 3. I'll plug in at 10 PM. It's at 25% charge (75 kWh battery).
  I need 85% charge by 7 AM. Charges at up to 11 kW, 95% efficiency.

- A dishwasher (3.6 kWh, between 8 PM–11 PM, max 2 kW).
- A dryer (4.5 kWh, between 9 PM–midnight, max 4 kW).

House peak limit: 15 kW. Goal: minimise electricity cost."""

    print("\n" + "=" * 80)
    print("AGENTIC AI (with Tools + Optimiser)")
    print("=" * 80)
    agent  = create_dr_agent()
    result = agent.invoke({"input": user_query})
    print("\n" + "-" * 80)
    print("AGENT OUTPUT:")
    print("-" * 80)
    print(result["output"])

    print("\n" + "=" * 80)
    print("BASELINE LLM (No Tools)")
    print("=" * 80)
    baseline_llm    = create_baseline_llm()
    baseline_result = run_baseline_recommendation(baseline_llm, user_query)
    print("\n" + "-" * 80)
    print("BASELINE OUTPUT:")
    print("-" * 80)
    print(baseline_result)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)