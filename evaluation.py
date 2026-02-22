"""
Evaluation Script for DRAgent
Compares Agentic AI vs Baseline LLM on:
1. Objective Improvement (cost/carbon savings)
2. Feasibility (constraint adherence)
3. Faithfulness (explanation matches computation)
"""

import json
import re
from typing import Dict, List, Tuple
import numpy as np
from dr_agent import (
    create_dr_agent,
    create_baseline_llm,
    run_baseline_recommendation,
    solve_dr_optimization,
    fetch_sdge_prices,
    fetch_caiso_carbon
)


# ============================================================================
# TEST SCENARIOS
# ============================================================================

SCENARIOS = {
    "sufficient_info": {
        "name": "Sufficient Information (Golden Path)",
        "query": """Help me schedule my appliances for tomorrow to minimize cost.

Appliances:
- EV: needs 16 kWh, available 10 PM to 7 AM, max 11 kW
- Dishwasher: needs 3.6 kWh, available 8 PM to 11 PM, max 2 kW

My household peak limit is 15 kW.""",
        "appliances": [
            {
                "name": "EV",
                "energy_required_kwh": 16.0,
                "start_hour": 22,
                "end_hour": 7,
                "min_power_kw": 0.0,
                "max_power_kw": 11.0,
                "household_peak_limit": 15.0
            },
            {
                "name": "Dishwasher",
                "energy_required_kwh": 3.6,
                "start_hour": 20,
                "end_hour": 23,
                "min_power_kw": 0.0,
                "max_power_kw": 2.0,
                "household_peak_limit": 15.0
            }
        ],
        "expected_behavior": "Should fetch data, optimize, and provide clear schedule with savings"
    },
    
    "insufficient_info": {
        "name": "Insufficient Information (Missing Constraints)",
        "query": """I want to charge my EV overnight. It needs 16 kWh. 
What's the best schedule to save money?""",
        "appliances": None,  # Missing time windows and power limits
        "expected_behavior": "Should identify missing information and ask for it"
    },
    
    "redundant_info": {
        "name": "Redundant Information (Noisy Context)",
        "query": """Help me schedule my appliances for tomorrow.

My appliances:
- EV: needs 16 kWh, available 10 PM to 7 AM, max 11 kW
- Dishwasher: needs 3.6 kWh, available 8 PM to 11 PM, max 2 kW

Household peak: 15 kW

By the way, I also read that PG&E (a different utility in Northern California) has rates of $0.40/kWh during peak. 
Last year in 2023, my neighbor said their rates were different. Also, someone told me that natural gas prices might affect electricity.
I'm in San Diego and use SDG&E. My friend in LA uses different rates.
""",
        "appliances": [
            {
                "name": "EV",
                "energy_required_kwh": 16.0,
                "start_hour": 22,
                "end_hour": 7,
                "min_power_kw": 0.0,
                "max_power_kw": 11.0,
                "household_peak_limit": 15.0
            },
            {
                "name": "Dishwasher",
                "energy_required_kwh": 3.6,
                "start_hour": 20,
                "end_hour": 23,
                "min_power_kw": 0.0,
                "max_power_kw": 2.0,
                "household_peak_limit": 15.0
            }
        ],
        "expected_behavior": "Should filter out irrelevant information and focus on SDG&E rates"
    },
    
    "carbon_optimization": {
        "name": "Carbon Optimization Goal",
        "query": """I care more about reducing my carbon footprint than saving money.

Appliances:
- EV: needs 16 kWh, available 10 PM to 7 AM, max 11 kW
- Dryer: needs 4.5 kWh, available 9 PM to midnight, max 4 kW

Household peak: 15 kW. Help me minimize carbon emissions.""",
        "appliances": [
            {
                "name": "EV",
                "energy_required_kwh": 16.0,
                "start_hour": 22,
                "end_hour": 7,
                "min_power_kw": 0.0,
                "max_power_kw": 11.0,
                "household_peak_limit": 15.0
            },
            {
                "name": "Dryer",
                "energy_required_kwh": 4.5,
                "start_hour": 21,
                "end_hour": 24,
                "min_power_kw": 0.0,
                "max_power_kw": 4.0,
                "household_peak_limit": 15.0
            }
        ],
        "expected_behavior": "Should optimize for carbon instead of cost"
    }
}


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_objective_improvement(response: str, scenario: Dict) -> Dict:
    """
    Metric 1: Objective Improvement
    Extract cost/carbon savings from response and verify against ground truth.
    """
    
    metrics = {
        "cost_savings_claimed": None,
        "carbon_reduction_claimed": None,
        "cost_savings_actual": None,
        "carbon_reduction_actual": None,
        "objective_score": 0.0
    }
    
    # Extract claimed savings from text
    cost_match = re.search(r'\$?(\d+\.?\d*)\s*(?:dollars?|\/day|per day|in savings)', response, re.IGNORECASE)
    if cost_match:
        metrics["cost_savings_claimed"] = float(cost_match.group(1))
    
    carbon_match = re.search(r'(\d+\.?\d*)\s*(?:lbs?|pounds?)\s*(?:CO2|carbon|emissions)', response, re.IGNORECASE)
    if carbon_match:
        metrics["carbon_reduction_claimed"] = float(carbon_match.group(1))
    
    # Calculate actual savings if we have appliance data
    if scenario["appliances"]:
        try:
            prices = fetch_sdge_prices.invoke({})
            carbon = fetch_caiso_carbon.invoke({})
            
            optimization_result = solve_dr_optimization.invoke({
                "appliances_json": json.dumps(scenario["appliances"]),
                "prices_json": prices,
                "carbon_json": carbon,
                "optimization_goal": "cost"
            })
            
            result = json.loads(optimization_result)
            if result["status"] == "success":
                metrics["cost_savings_actual"] = result["metrics"]["cost_savings_dollars"]
                metrics["carbon_reduction_actual"] = result["metrics"]["carbon_reduction_lbs"]
                
                # Score: how close are claimed savings to actual?
                if metrics["cost_savings_claimed"]:
                    error_pct = abs(metrics["cost_savings_claimed"] - metrics["cost_savings_actual"]) / max(metrics["cost_savings_actual"], 0.01)
                    metrics["objective_score"] = max(0, 1 - error_pct)
        except Exception as e:
            print(f"Warning: Could not compute ground truth - {e}")
    
    return metrics


def evaluate_feasibility(response: str, scenario: Dict) -> Dict:
    """
    Metric 2: Feasibility
    Check if recommended schedule respects constraints.
    """
    
    feasibility = {
        "time_windows_respected": None,
        "power_limits_respected": None,
        "energy_requirements_met": None,
        "feasibility_score": 0.0,
        "violations": []
    }
    
    if not scenario["appliances"]:
        return feasibility
    
    # Extract recommended times from response
    ev_time_match = re.search(r'EV.*?(\d+)\s*(?:AM|PM|:00)', response, re.IGNORECASE)
    dish_time_match = re.search(r'(?:dishwasher|dish).*?(\d+)\s*(?:AM|PM|:00)', response, re.IGNORECASE)
    
    violations = []
    
    # Check EV timing
    if ev_time_match:
        ev_hour = int(ev_time_match.group(1))
        if 'PM' in ev_time_match.group(0):
            ev_hour = ev_hour % 12 + 12
        elif 'AM' in ev_time_match.group(0) and ev_hour == 12:
            ev_hour = 0
            
        # EV window: 22 (10 PM) to 7 (7 AM)
        if not (ev_hour >= 22 or ev_hour <= 7):
            violations.append(f"EV scheduled at {ev_hour}:00, outside window 22-7")
    
    # Check dishwasher timing
    if dish_time_match:
        dish_hour = int(dish_time_match.group(1))
        if 'PM' in dish_time_match.group(0):
            dish_hour = dish_hour % 12 + 12
        
        # Dishwasher window: 20 (8 PM) to 23 (11 PM)
        if not (20 <= dish_hour <= 23):
            violations.append(f"Dishwasher scheduled at {dish_hour}:00, outside window 20-23")
    
    feasibility["violations"] = violations
    feasibility["time_windows_respected"] = len(violations) == 0
    feasibility["feasibility_score"] = 1.0 if len(violations) == 0 else 0.0
    
    return feasibility


def evaluate_faithfulness(response: str, scenario: Dict) -> Dict:
    """
    Metric 3: Faithfulness
    Check if explanation matches computed optimization results.
    """
    
    faithfulness = {
        "mentions_data_source": None,
        "quantifies_savings": None,
        "explains_reasoning": None,
        "matches_optimization": None,
        "faithfulness_score": 0.0
    }
    
    # Check if mentions data sources
    faithfulness["mentions_data_source"] = any([
        "SDG&E" in response or "sdge" in response.lower(),
        "CAISO" in response or "caiso" in response.lower(),
        "price" in response.lower() and "data" in response.lower()
    ])
    
    # Check if quantifies savings
    faithfulness["quantifies_savings"] = bool(
        re.search(r'\$\d+\.?\d*', response) or 
        re.search(r'\d+\.?\d*\s*(?:lbs?|pounds?)', response)
    )
    
    # Check if explains WHY (mentions off-peak, cheap hours, etc.)
    reasoning_keywords = ["off-peak", "cheap", "expensive", "peak", "renewable", "solar", "carbon"]
    faithfulness["explains_reasoning"] = any(kw in response.lower() for kw in reasoning_keywords)
    
    # Check if matches optimization
    if scenario["appliances"]:
        try:
            prices = fetch_sdge_prices.invoke({})
            carbon = fetch_caiso_carbon.invoke({})
            
            optimization_result = solve_dr_optimization.invoke({
                "appliances_json": json.dumps(scenario["appliances"]),
                "prices_json": prices,
                "carbon_json": carbon,
                "optimization_goal": "cost"
            })
            
            result = json.loads(optimization_result)
            if result["status"] == "success":
                # Check if recommended times align with optimization
                schedule = result["schedule"]
                
                # For EV, check if recommended time is in optimal operating hours
                if "EV" in schedule:
                    optimal_hours = schedule["EV"]["operating_hours"]
                    # Check if response mentions times in optimal range
                    for hour in optimal_hours[:3]:  # Check first few hours
                        if hour < 12:
                            if f"{hour} AM" in response or f"{hour}:00" in response:
                                faithfulness["matches_optimization"] = True
                                break
                        else:
                            pm_hour = hour - 12 if hour > 12 else 12
                            if f"{pm_hour} PM" in response:
                                faithfulness["matches_optimization"] = True
                                break
        except Exception as e:
            print(f"Warning: Could not verify optimization match - {e}")
    
    # Calculate score
    score = 0
    if faithfulness["mentions_data_source"]:
        score += 0.25
    if faithfulness["quantifies_savings"]:
        score += 0.25
    if faithfulness["explains_reasoning"]:
        score += 0.25
    if faithfulness["matches_optimization"]:
        score += 0.25
    
    faithfulness["faithfulness_score"] = score
    
    return faithfulness


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation():
    """
    Run comprehensive evaluation of Agent vs Baseline.
    """
    
    print("=" * 80)
    print("DRAGENT EVALUATION")
    print("=" * 80)
    
    agent = create_dr_agent()
    baseline_llm = create_baseline_llm()
    
    results = {
        "agent": {},
        "baseline": {}
    }
    
    for scenario_key, scenario in SCENARIOS.items():
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'=' * 80}")
        print(f"Query: {scenario['query'][:100]}...")
        print(f"Expected: {scenario['expected_behavior']}")
        
        # Test Agent
        print(f"\n{'-' * 80}")
        print("TESTING AGENT...")
        print(f"{'-' * 80}")
        
        try:
            agent_result = agent.invoke({"input": scenario["query"]})
            agent_response = agent_result["output"]
            print(agent_response[:500] + "..." if len(agent_response) > 500 else agent_response)
            
            # Evaluate agent
            obj_metrics = evaluate_objective_improvement(agent_response, scenario)
            feas_metrics = evaluate_feasibility(agent_response, scenario)
            faith_metrics = evaluate_faithfulness(agent_response, scenario)
            
            results["agent"][scenario_key] = {
                "response": agent_response,
                "objective": obj_metrics,
                "feasibility": feas_metrics,
                "faithfulness": faith_metrics
            }
            
        except Exception as e:
            print(f"Agent failed: {e}")
            results["agent"][scenario_key] = {"error": str(e)}
        
        # Test Baseline
        print(f"\n{'-' * 80}")
        print("TESTING BASELINE...")
        print(f"{'-' * 80}")
        
        try:
            baseline_response = run_baseline_recommendation(baseline_llm, scenario["query"])
            print(baseline_response[:500] + "..." if len(baseline_response) > 500 else baseline_response)
            
            # Evaluate baseline
            obj_metrics = evaluate_objective_improvement(baseline_response, scenario)
            feas_metrics = evaluate_feasibility(baseline_response, scenario)
            faith_metrics = evaluate_faithfulness(baseline_response, scenario)
            
            results["baseline"][scenario_key] = {
                "response": baseline_response,
                "objective": obj_metrics,
                "feasibility": feas_metrics,
                "faithfulness": faith_metrics
            }
            
        except Exception as e:
            print(f"Baseline failed: {e}")
            results["baseline"][scenario_key] = {"error": str(e)}
    
    # Generate summary report
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    
    for approach in ["agent", "baseline"]:
        print(f"\n{approach.upper()} RESULTS:")
        print("-" * 40)
        
        total_obj = 0
        total_feas = 0
        total_faith = 0
        count = 0
        
        for scenario_key, result in results[approach].items():
            if "error" not in result:
                obj_score = result["objective"].get("objective_score", 0)
                feas_score = result["feasibility"].get("feasibility_score", 0)
                faith_score = result["faithfulness"].get("faithfulness_score", 0)
                
                print(f"\n{scenario_key}:")
                print(f"  Objective Score: {obj_score:.2f}")
                print(f"  Feasibility Score: {feas_score:.2f}")
                print(f"  Faithfulness Score: {faith_score:.2f}")
                
                if result["objective"].get("cost_savings_claimed"):
                    print(f"  Cost Savings (claimed): ${result['objective']['cost_savings_claimed']:.2f}")
                if result["objective"].get("cost_savings_actual"):
                    print(f"  Cost Savings (actual): ${result['objective']['cost_savings_actual']:.2f}")
                
                if result["feasibility"]["violations"]:
                    print(f"  Violations: {result['feasibility']['violations']}")
                
                total_obj += obj_score
                total_feas += feas_score
                total_faith += faith_score
                count += 1
        
        if count > 0:
            print(f"\nAVERAGE SCORES:")
            print(f"  Objective: {total_obj/count:.2f}")
            print(f"  Feasibility: {total_feas/count:.2f}")
            print(f"  Faithfulness: {total_faith/count:.2f}")
            print(f"  Overall: {(total_obj + total_feas + total_faith)/(3*count):.2f}")
    
    # Save detailed results
    with open("/home/claude/evaluation_results.json", "w") as f:
        json.dumps(results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print("Detailed results saved to evaluation_results.json")
    print(f"{'=' * 80}")
    
    return results


if __name__ == "__main__":
    run_evaluation()
