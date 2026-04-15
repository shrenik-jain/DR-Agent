"""
DRAgent: Agentic AI for Residential Demand Response.

Public API — import from ``dragent`` (e.g. ``from dragent import create_dr_agent``).
"""

from dragent.agent import (
    check_required_inputs,
    create_baseline_llm,
    create_dr_agent,
    fetch_caiso_carbon,
    fetch_sdge_prices,
    fetch_weather_forecast,
    run_baseline_recommendation,
    solve_dr_optimization,
)

__all__ = [
    "check_required_inputs",
    "create_baseline_llm",
    "create_dr_agent",
    "fetch_caiso_carbon",
    "fetch_sdge_prices",
    "fetch_weather_forecast",
    "run_baseline_recommendation",
    "solve_dr_optimization",
]
