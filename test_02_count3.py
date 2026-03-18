"""
Ablation Test 02 of 10 — Appliance Count Scaling
==================================================
Count  : 3 appliances (Dishwasher, Clothes dryer, EV)
Goal   : cost minimisation

Baseline  — LLM answers the query directly from general knowledge.
            No tools, no solver. Savings figures are self-reported
            and unverifiable.

Agentic   — Agent receives the same query, calls fetch_sdge_prices,
            fetch_caiso_carbon, and solve_dr_optimization internally,
            then writes a response. Savings figures come from the
            actual solver computation.

Run with:
    python test_02_count3.py
"""

from dotenv import load_dotenv
load_dotenv()

import re
from dr_agent import create_dr_agent, create_baseline_llm, run_baseline_recommendation

QUERY = """I need to schedule three appliances tonight to save money.

1. Dishwasher: needs 3.6 kWh, available 8 PM to 11 PM, draws at most 2 kW
2. Clothes Dryer: needs 4.5 kWh, available 9 PM to midnight, draws at most 4 kW
3. EV: needs 16 kWh, available 3 PM to 10 PM, draws at most 11 kW

My household peak limit is 15 kW."""

print("=" * 60)
print("TEST 02 — 3 Appliances (Dishwasher, Clothes dryer, EV)")
print("=" * 60)
print(f"\nQuery:\n{QUERY}\n")

# ==============================================================================
# BASELINE
# ==============================================================================

print("=" * 60)
print("BASELINE")
print("=" * 60)

llm         = create_baseline_llm()
bl_response = run_baseline_recommendation(llm, QUERY)

print(bl_response)

# ==============================================================================
# AGENTIC
# ==============================================================================

print("\n" + "=" * 60)
print("AGENTIC")
print("=" * 60)

# return_intermediate_steps=True is required to get tool call details back
agent     = create_dr_agent()
agent.return_intermediate_steps = True

ag_result   = agent.invoke({"input": QUERY})
ag_response = ag_result["output"]
ag_steps    = ag_result.get("intermediate_steps", [])

print(ag_response)

# ==============================================================================
# METRICS
# ==============================================================================

print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)

# ── Tool Calls ────────────────────────────────────────────────────────────────
ag_tools = [step[0].tool for step in ag_steps]

print(f"\n  Tool Calls")
print(f"    Baseline : none (LLM answers directly, no tools available)")
print(f"    Agentic  : {ag_tools}  (total {len(ag_tools)})")

# ── Iter. to Pass ─────────────────────────────────────────────────────────────
print(f"\n  Iter. to Pass")
print(f"    Baseline : 0")
print(f"    Agentic  : {len(ag_steps)}")

# ── Cost Red. (%) ─────────────────────────────────────────────────────────────
# Baseline: scrape savings and baseline figures from the LLM response.
#           Tries table format (| TOTAL | $X | $X |) first, then inline text.
#           Self-reported and unverifiable — labelled as such.
# Agentic:  read from the LAST solve_dr_optimization call in intermediate_steps.
#           The agent sometimes calls the solver twice (e.g. retries), so we
#           take the last result which reflects the final accepted schedule.

import json

print(f"\n  Cost Red. (%)")

# Baseline — try table TOTAL row first, then inline text
bl_cost_red = None
table_match = re.search(
    r'\*{0,2}TOTAL\*{0,2}\s*\|[^|]*\|\s*\*{0,2}\$?([\d.]+)\*{0,2}\s*\|'
    r'\s*\*{0,2}\$?([\d.]+)\*{0,2}',
    bl_response, re.I
)
if table_match:
    bl_base    = float(table_match.group(1))
    bl_opt     = float(table_match.group(2))
    bl_savings = bl_base - bl_opt
    bl_cost_red = round(100.0 * bl_savings / bl_base, 1) if bl_base > 0 else None
    print(f"    Baseline : {bl_cost_red}%  "
          f"(self-reported — baseline ${bl_base:.2f}, "
          f"optimised ${bl_opt:.2f}, savings ${bl_savings:.2f})")
else:
    # fallback: look for any dollar savings figure mentioned inline
    all_dollars = re.findall(r'\$\s*([\d.]+)', bl_response)
    if len(all_dollars) >= 2:
        bl_base    = float(all_dollars[0])
        bl_opt     = float(all_dollars[1])
        bl_savings = bl_base - bl_opt
        bl_cost_red = round(100.0 * bl_savings / bl_base, 1) if bl_base > 0 else None
        print(f"    Baseline : {bl_cost_red}%  (self-reported, approximate parse)")
    else:
        print(f"    Baseline : N/A  (could not parse savings from response)")

# Agentic — use the LAST solve_dr_optimization result
ag_cost_red   = None
last_solve_out = None
for step in ag_steps:
    if step[0].tool == "solve_dr_optimization":
        try:
            last_solve_out = json.loads(step[1])
        except Exception:
            pass

if last_solve_out:
    m           = last_solve_out.get("metrics", {})
    base        = m.get("baseline_cost_dollars", 0.0)
    saved       = m.get("cost_savings_dollars", 0.0)
    ag_cost_red = round(100.0 * saved / base, 1) if base > 0 else None
    print(f"    Agentic  : {ag_cost_red}%  "
          f"(solver computed — savings ${saved:.2f}, baseline ${base:.2f})")
else:
    print(f"    Agentic  : N/A  (agent did not call the solver)")

# ── Feasibility (%) ───────────────────────────────────────────────────────────
# Only meaningful for agentic — baseline never called the solver so there is
# no schedule to check. We extract the appliance specs the agent passed to
# the solver from step[0].tool_input, then verify the returned schedule.

def fmt(v):
    return "—" if v is None else f"{v:.1f}%"

import numpy as np

print(f"\n  Feas. (%)")

ag_feas          = None
last_solve_step  = None
for step in ag_steps:
    if step[0].tool == "solve_dr_optimization":
        last_solve_step = step

if last_solve_out and last_solve_out.get("status") in ("success", "optimal_inaccurate") \
        and last_solve_step is not None:
    try:
        appliances_used = json.loads(last_solve_step[0].tool_input["appliances_json"])
        if isinstance(appliances_used, dict):
            appliances_used = [appliances_used]

        schedule     = last_solve_out["schedule"]
        H            = 24
        cp = ct      = 0
        peak_limit   = float(appliances_used[0].get("household_peak_limit", 15.0))
        hourly_total = np.zeros(H)

        for app in appliances_used:
            name  = app["name"]
            entry = schedule.get(name, {})
            kwh   = entry.get("hourly_consumption_kwh", [0.0] * H)
            E_req = float(app["energy_required_kwh"])
            P_max = float(app["max_power_kw"])
            a, b  = int(app["start_hour"]), int(app["end_hour"])
            win   = set(range(a, b + 1)) if b >= a \
                    else set(range(a, H)) | set(range(0, b + 1))

            e_ok = abs(sum(kwh) - E_req) / max(E_req, 1e-6) < 0.01
            ct += 1; cp += int(e_ok)
            print(f"    [{name}] (a) energy : {'OK' if e_ok else 'FAIL'}"
                  f"  ({sum(kwh):.3f} kWh delivered, {E_req} required)")

            w_ok = not any(kwh[h] > 0.01 and h not in win for h in range(H))
            ct += 1; cp += int(w_ok)
            print(f"    [{name}] (b) window : {'OK' if w_ok else 'FAIL'}"
                  f"  (allowed hours {sorted(win)})")

            p_ok = all(v <= P_max + 0.01 for v in kwh)
            ct += 1; cp += int(p_ok)
            print(f"    [{name}] (c) power  : {'OK' if p_ok else 'FAIL'}"
                  f"  (max seen {max(kwh):.3f} kW, limit {P_max} kW)")

            hourly_total += np.array(kwh)

        pk_ok = bool(np.all(hourly_total <= peak_limit + 0.05))
        ct += 1; cp += int(pk_ok)
        print(f"    [global]  (d) peak   : {'OK' if pk_ok else 'FAIL'}"
              f"  (peak seen {hourly_total.max():.3f} kW, limit {peak_limit} kW)")

        ag_feas = round(100.0 * cp / ct, 1)
    except Exception as e:
        print(f"    could not compute: {e}")

print(f"    Baseline : N/A  (no solver called)")
print(f"    Agentic  : {fmt(ag_feas)}")

# ── Safe Fail (%) ─────────────────────────────────────────────────────────────
# Set EXPECT_FAILURE = True in tests where the query is deliberately infeasible.
# When True, we check whether the agent correctly abstained rather than
# fabricating a schedule. For golden-path tests this is always N/A.

EXPECT_FAILURE = False

print(f"\n  Safe Fail (%)")
if EXPECT_FAILURE:
    fail_keywords = [
        r"infeasib", r"not (feasible|possible)", r"cannot be (met|satisfied)",
        r"impossible", r"no (valid|feasible) solution",
        r"insufficient", r"too (little|short|small)", r"exceeds",
    ]
    graceful  = any(re.search(p, ag_response, re.I) for p in fail_keywords)
    safe_fail = 100.0 if graceful else 0.0
    print(f"    agent abstained gracefully : {graceful}")
    print(f"    Agentic  : {fmt(safe_fail)}")
else:
    safe_fail = None
    print(f"    N/A — golden-path test, no failure expected")

# ── Faithful (%) ──────────────────────────────────────────────────────────────
# F1 mentions SDG&E or CAISO
# F2 contains a $X.XX dollar figure
# F3 names a price period (off-peak / on-peak)
# F4 mentions a specific hour recommendation

def faithful_score(response):
    f1 = bool(re.search(r"sdg&?e|caiso", response, re.I))
    f2 = bool(re.search(r'\$\s*\d+\.\d{2}', response))
    f3 = bool(re.search(r"super.?off.?peak|off.?peak|on.?peak", response, re.I))
    f4 = bool(re.search(r'\d+\s*(?:AM|PM)', response, re.I))
    return f1, f2, f3, f4, round(100.0 * sum([f1, f2, f3, f4]) / 4, 1)

print(f"\n  Faithful (%)")
for arch, response in [("Baseline", bl_response), ("Agentic", ag_response)]:
    f1, f2, f3, f4, score = faithful_score(response)
    print(f"\n    {arch}")
    print(f"      F1 mentions SDG&E/CAISO : {f1}")
    print(f"      F2 dollar figure        : {f2}")
    print(f"      F3 price period         : {f3}")
    print(f"      F4 specific hour        : {f4}")
    print(f"      → Faithful = {score}%")

bl_faithful = faithful_score(bl_response)[4]
ag_faithful = faithful_score(ag_response)[4]

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Model':<12} {'Feas.':>7} {'CostRed.':>9} {'Iters':>6} {'Calls':>6} {'SafeFail':>9} {'Faithful':>9}")
print(f"  {'─'*12} {'─'*7} {'─'*9} {'─'*6} {'─'*6} {'─'*9} {'─'*9}")
print(f"  {'Baseline':<12} {'—':>7} {fmt(bl_cost_red):>9} {'0':>6} {'0':>6} {'—':>9} {fmt(bl_faithful):>9}")
print(f"  {'Agentic':<12} {fmt(ag_feas):>7} {fmt(ag_cost_red):>9} {len(ag_steps):>6} {len(ag_tools):>6} {fmt(safe_fail) if EXPECT_FAILURE else '—':>9} {fmt(ag_faithful):>9}")
print("=" * 60)
print("\nNote: Baseline Cost Red. is self-reported by the LLM and unverifiable.")
print("      Agentic Cost Red. and Feas. are computed from the solver.")

# ==============================================================================
# SAVE TO FILE
# ==============================================================================

lines = []
lines.append("=" * 60)
lines.append("TEST 02 — 3 Appliances (Dishwasher, Clothes Dryer, EV)")
lines.append("=" * 60)
lines.append("\nQUERY")
lines.append("-" * 60)
lines.append(QUERY)

lines.append("\n" + "=" * 60)
lines.append("BASELINE RESPONSE")
lines.append("=" * 60)
lines.append(bl_response)

lines.append("\n" + "=" * 60)
lines.append("AGENTIC RESPONSE")
lines.append("=" * 60)
lines.append(ag_response)

lines.append("\n" + "=" * 60)
lines.append("METRICS")
lines.append("=" * 60)
lines.append(f"\n  Tool Calls")
lines.append(f"    Baseline : none")
lines.append(f"    Agentic  : {ag_tools}  (total {len(ag_tools)})")
lines.append(f"\n  Iter. to Pass")
lines.append(f"    Baseline : 0")
lines.append(f"    Agentic  : {len(ag_steps)}")
lines.append(f"\n  Feas. (%)")
lines.append(f"    Baseline : N/A")
lines.append(f"    Agentic  : {fmt(ag_feas)}")
lines.append(f"\n  Cost Red. (%)")
lines.append(f"    Baseline : {fmt(bl_cost_red)}  (self-reported)")
lines.append(f"    Agentic  : {fmt(ag_cost_red)}  (solver computed)")
lines.append(f"\n  Safe Fail (%)")
lines.append(f"    Baseline : N/A")
lines.append(f"    Agentic  : {fmt(safe_fail) if EXPECT_FAILURE else 'N/A'}")
lines.append(f"\n  Faithful (%)")
lines.append(f"    Baseline : {fmt(bl_faithful)}")
lines.append(f"    Agentic  : {fmt(ag_faithful)}")
lines.append("\n" + "=" * 60)
lines.append("SUMMARY TABLE")
lines.append("=" * 60)
lines.append(f"  {'Model':<12} {'Feas.':>7} {'CostRed.':>9} {'Iters':>6} {'Calls':>6} {'SafeFail':>9} {'Faithful':>9}")
lines.append(f"  {'─'*12} {'─'*7} {'─'*9} {'─'*6} {'─'*6} {'─'*9} {'─'*9}")
lines.append(f"  {'Baseline':<12} {'—':>7} {fmt(bl_cost_red):>9} {'0':>6} {'0':>6} {'—':>9} {fmt(bl_faithful):>9}")
lines.append(f"  {'Agentic':<12} {fmt(ag_feas):>7} {fmt(ag_cost_red):>9} {len(ag_steps):>6} {len(ag_tools):>6} {fmt(safe_fail) if EXPECT_FAILURE else '—':>9} {fmt(ag_faithful):>9}")
lines.append("=" * 60)
lines.append("\nNote: Baseline Cost Red. is self-reported by the LLM and unverifiable.")
lines.append("      Agentic Cost Red. and Feas. are computed from the solver.")

with open("test_02_results.txt", "w") as f:
    f.write("\n".join(lines))

print("\nResults saved to test_02_results.txt")