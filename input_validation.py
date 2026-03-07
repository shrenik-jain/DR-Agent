"""
Input validation for DRAgent.
Checks appliance specs for required fields, applies defaults when possible,
and returns follow-up questions for missing user-specific inputs.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import APPLIANCE_DEFAULTS, DEFAULT_HOUSEHOLD_PEAK_KW
except ImportError:
    APPLIANCE_DEFAULTS = {}
    DEFAULT_HOUSEHOLD_PEAK_KW = 15.0


# Required fields per appliance type (must come from user or be defaultable)
REQUIRED_FLEXIBLE = {"name", "energy_required_kwh", "start_hour", "end_hour", "max_power_kw"}
REQUIRED_EV = {
    "name", "start_hour", "end_hour",
    "initial_soc_kwh", "target_soc_kwh", "battery_capacity_kwh",
    "max_charge_power_kw"
}
REQUIRED_HVAC = {"name"}  # All other HVAC fields have built-in defaults in dr_agent

# Optional fields with defaults we can apply without asking
OPTIONAL_DEFAULTS_FLEXIBLE = {
    "min_power_kw": 0.0,
    "household_peak_limit": DEFAULT_HOUSEHOLD_PEAK_KW,
}
OPTIONAL_DEFAULTS_EV = {
    "charge_efficiency": 0.95,
    "min_soc_kwh": 0.0,
    "max_discharge_power_kw": 0.0,
    "discharge_efficiency": 0.95,
    "household_peak_limit": DEFAULT_HOUSEHOLD_PEAK_KW,
}
OPTIONAL_DEFAULTS_HVAC = {
    "initial_temp_f": 72.0,
    "thermal_resistance": 4.0,
    "thermal_capacitance": 2.0,
    "cop": 3.0,
    "max_power_kw": 3.5,
    "min_power_kw": 0.0,
    "temp_min_f": 70.0,
    "temp_max_f": 78.0,
    "cooling_only": True,
    "household_peak_limit": DEFAULT_HOUSEHOLD_PEAK_KW,
}

# Human-readable labels and follow-up question templates
FIELD_LABELS = {
    "name": "appliance name",
    "energy_required_kwh": "energy needed (kWh)",
    "start_hour": "plug-in or start hour (0-23)",
    "end_hour": "end or departure hour (0-23)",
    "max_power_kw": "max power (kW)",
    "min_power_kw": "min power (kW)",
    "initial_soc_kwh": "current battery level (kWh) or state of charge",
    "target_soc_kwh": "target battery level (kWh) or state of charge by departure",
    "battery_capacity_kwh": "battery capacity (kWh)",
    "max_charge_power_kw": "max charging rate (kW)",
    "household_peak_limit": "household peak limit (kW)",
    "temp_min_f": "minimum comfortable temperature (°F)",
    "temp_max_f": "maximum comfortable temperature (°F)",
}


def _get_type(spec: Dict[str, Any]) -> str:
    """Return normalized appliance type: flexible, ev, or hvac."""
    t = (spec.get("type") or "flexible").strip().lower()
    if t == "ev":
        return "ev"
    if t == "hvac":
        return "hvac"
    return "flexible"


def _apply_defaults(spec: Dict[str, Any], optional_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of spec with missing optional fields filled from optional_defaults."""
    out = dict(spec)
    for k, v in optional_defaults.items():
        if k not in out or out[k] is None:
            out[k] = v
    return out


# Keys the agent may send that we can use to infer required fields
EV_ENERGY_KEYS = ("energy_needed_kwh", "energy_kwh", "energy_required_kwh")
EV_PCT_KEYS = ("initial_soc_pct", "initial_soc_percent", "target_soc_pct", "target_soc_percent")
DEFAULT_EV_BATTERY_KWH = 75.0
DEFAULT_EV_WINDOW = (22, 7)  # 10 PM to 7 AM
DEFAULT_EV_MAX_CHARGE_KW = 11.0


def _get_config_default(app_name: str, field: str, app_type: str) -> Optional[Any]:
    """Get default from config.APPLIANCE_DEFAULTS if name matches a known appliance."""
    name_lower = (app_name or "").lower()
    # Broader matching: ev/tesla/car -> EV; dishwasher/dryer/washer -> flexible
    if app_type == "ev":
        for key in ("EV_MODEL3", "ev", "electric vehicle", "tesla", "car"):
            if key in name_lower or name_lower in key:
                defaults = APPLIANCE_DEFAULTS.get("EV_MODEL3", {})
                if field == "max_charge_power_kw" and "max_power_kw" in defaults:
                    return defaults["max_power_kw"]
                if field in ("start_hour", "end_hour") and "typical_window" in defaults:
                    s, e = defaults["typical_window"]
                    return s if field == "start_hour" else e
                if field == "battery_capacity_kwh":
                    return DEFAULT_EV_BATTERY_KWH
                break
    for key, defaults in APPLIANCE_DEFAULTS.items():
        key_lower = key.lower().replace("_", " ")
        if key_lower in name_lower or name_lower in key_lower:
            if field == "energy_required_kwh" and "energy_kwh" in defaults:
                return defaults["energy_kwh"]
            if field == "max_power_kw" and "max_power_kw" in defaults:
                return defaults["max_power_kw"]
            if field == "max_charge_power_kw" and "max_power_kw" in defaults:
                return defaults["max_power_kw"]
            if field in ("start_hour", "end_hour") and "typical_window" in defaults:
                s, e = defaults["typical_window"]
                return s if field == "start_hour" else e
    return None


def _infer_ev_fields(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Infer EV initial_soc_kwh, target_soc_kwh, battery_capacity_kwh from
    energy_needed_kwh, initial_soc_pct, target_soc_pct, etc.
    Returns (updated_spec, dict of what was inferred for defaults_applied).
    """
    out = dict(spec)
    applied = {}

    # Battery capacity: use default if missing (needed for % and for capping)
    cap = out.get("battery_capacity_kwh")
    if cap is None or (isinstance(cap, (int, float)) and cap <= 0):
        out["battery_capacity_kwh"] = DEFAULT_EV_BATTERY_KWH
        applied["battery_capacity_kwh"] = DEFAULT_EV_BATTERY_KWH
    cap = out.get("battery_capacity_kwh")
    if cap is None:
        return out, applied
    cap = float(cap)

    # SoC from percentages (agent may send initial_soc_pct, target_soc_pct)
    initial_pct = out.get("initial_soc_pct") or out.get("initial_soc_percent")
    target_pct = out.get("target_soc_pct") or out.get("target_soc_percent")
    if initial_pct is not None and out.get("initial_soc_kwh") is None:
        out["initial_soc_kwh"] = round(cap * float(initial_pct) / 100.0, 2)
        applied["initial_soc_kwh"] = out["initial_soc_kwh"]
    if target_pct is not None and out.get("target_soc_kwh") is None:
        out["target_soc_kwh"] = round(cap * float(target_pct) / 100.0, 2)
        applied["target_soc_kwh"] = out["target_soc_kwh"]

    # Energy needed (kWh to add): infer initial or target
    energy_needed = None
    for k in EV_ENERGY_KEYS:
        if out.get(k) is not None:
            try:
                energy_needed = float(out[k])
                break
            except (TypeError, ValueError):
                pass
    if energy_needed is not None:
        initial = out.get("initial_soc_kwh")
        target = out.get("target_soc_kwh")
        if initial is not None and target is None:
            out["target_soc_kwh"] = round(min(initial + energy_needed, cap), 2)
            applied["target_soc_kwh"] = out["target_soc_kwh"]
        elif target is not None and initial is None:
            out["initial_soc_kwh"] = round(max(0.0, target - energy_needed), 2)
            applied["initial_soc_kwh"] = out["initial_soc_kwh"]
        elif initial is None and target is None:
            # Only "need X kWh": assume start from 0, target = min(X, capacity)
            out["initial_soc_kwh"] = 0.0
            out["target_soc_kwh"] = round(min(energy_needed, cap), 2)
            applied["initial_soc_kwh"] = 0.0
            applied["target_soc_kwh"] = out["target_soc_kwh"]

    # Defaults for window and max charge if still missing
    if out.get("start_hour") is None:
        out["start_hour"] = DEFAULT_EV_WINDOW[0]
        applied["start_hour"] = DEFAULT_EV_WINDOW[0]
    if out.get("end_hour") is None:
        out["end_hour"] = DEFAULT_EV_WINDOW[1]
        applied["end_hour"] = DEFAULT_EV_WINDOW[1]
    if out.get("max_charge_power_kw") is None:
        out["max_charge_power_kw"] = DEFAULT_EV_MAX_CHARGE_KW
        applied["max_charge_power_kw"] = DEFAULT_EV_MAX_CHARGE_KW

    return out, applied


def _missing_for_spec(spec: Dict[str, Any], required: set) -> List[str]:
    """Return list of required field names that are missing (None or empty). Numeric 0 is valid."""
    missing = []
    for k in required:
        val = spec.get(k)
        if val is None:
            missing.append(k)
        elif isinstance(val, str) and not str(val).strip():
            missing.append(k)
    return missing


def _one_follow_up(name: str, missing: List[str], app_type: str) -> str:
    """Return a single consolidated follow-up question per appliance."""
    if app_type == "ev":
        need_soc = any(
            f in missing
            for f in ("initial_soc_kwh", "target_soc_kwh", "battery_capacity_kwh")
        )
        need_window = "start_hour" in missing or "end_hour" in missing
        need_power = "max_charge_power_kw" in missing
        parts = []
        if need_soc:
            parts.append("current and target charge (e.g. 25% to 85%, or “need 16 kWh”)")
        if need_window:
            parts.append("when you can plug in and by when you need it (e.g. 10 PM to 7 AM)")
        if need_power:
            parts.append("max charging rate in kW (e.g. 11)")
        if parts:
            return f"For {name}: " + ", ".join(parts) + "."
    if app_type == "flexible":
        return f"For {name}: energy needed (kWh), time window (e.g. 8 PM–11 PM), and max power (kW)."
    return f"For {name}: any missing details (name is already set)."


def validate_appliance_specs(appliances_json: str) -> Dict[str, Any]:
    """
    Validate appliance specs. Apply defaults where possible; return follow-up
    questions for missing required user-specific inputs.

    Args:
        appliances_json: JSON string — list of appliance objects or single object

    Returns:
        Dict with: ready (bool), specs_with_defaults (list), follow_up_questions (list),
        missing_by_appliance (dict), defaults_applied (dict), error (str or None)
    """
    result = {
        "ready": False,
        "specs_with_defaults": [],
        "follow_up_questions": [],
        "missing_by_appliance": {},
        "defaults_applied": {},
        "error": None,
    }

    try:
        raw = json.loads(appliances_json)
        specs = [raw] if isinstance(raw, dict) else raw
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {e}"
        return result

    if not specs:
        result["follow_up_questions"] = [
            "Which appliances would you like to schedule? (e.g. EV, dishwasher, dryer, HVAC)"
        ]
        return result

    all_ready = True
    specs_out = []
    defaults_applied = {}
    missing_by_appliance = {}
    follow_ups = []

    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            continue
        spec = dict(spec)
        app_type = _get_type(spec)
        name = spec.get("name") or f"Appliance {i+1}"

        if app_type == "flexible":
            required = REQUIRED_FLEXIBLE
            optional_defaults = OPTIONAL_DEFAULTS_FLEXIBLE
        elif app_type == "ev":
            required = REQUIRED_EV
            optional_defaults = OPTIONAL_DEFAULTS_EV
        else:
            required = REQUIRED_HVAC
            optional_defaults = OPTIONAL_DEFAULTS_HVAC

        spec = _apply_defaults(spec, optional_defaults)

        # EV: infer initial/target/capacity from energy_needed, %, and apply window/charge defaults
        if app_type == "ev":
            spec, ev_applied = _infer_ev_fields(spec)
            for k, v in ev_applied.items():
                defaults_applied[f"{name}.{k}"] = v

        missing = _missing_for_spec(spec, required)

        for m in list(missing):
            default_val = _get_config_default(name, m, app_type)
            if default_val is not None:
                spec[m] = default_val
                missing.remove(m)
                defaults_applied[f"{name}.{m}"] = default_val

        if missing:
            all_ready = False
            missing_by_appliance[name] = missing
            # One short question per appliance instead of one per field
            follow_ups.append(_one_follow_up(name, missing, app_type))

        spec["type"] = app_type
        if i == 0 and spec.get("household_peak_limit") is None:
            spec["household_peak_limit"] = DEFAULT_HOUSEHOLD_PEAK_KW
        specs_out.append(spec)

    result["specs_with_defaults"] = specs_out
    result["missing_by_appliance"] = missing_by_appliance
    result["defaults_applied"] = defaults_applied
    result["follow_up_questions"] = follow_ups
    result["ready"] = all_ready

    return result


def get_follow_up_response(validation_result: Dict[str, Any]) -> str:
    """Build a user-friendly follow-up message from validation result."""
    if validation_result.get("ready"):
        return ""
    qs = validation_result.get("follow_up_questions", [])
    if not qs:
        return "I need a bit more information to create an optimal schedule. Could you share which appliances you'd like to schedule and their details (energy needed, time window, max power)?"
    return "To create an optimal schedule, I need a few more details:\n\n" + "\n".join(f"• {q}" for q in qs)
