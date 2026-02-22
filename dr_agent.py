"""
DRAgent: Agentic AI for Residential Demand Response
Using LangChain for orchestration and tool calling
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import cvxpy as cp
from datetime import datetime, timedelta
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json


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
    # For demo purposes, using SDG&E EV-TOU-5 rates (typical TOU structure)
    # In production, this would scrape or API call to SDG&E
    
    # SDG&E typical TOU structure:
    # Super Off-Peak (12am-6am): ~$0.27/kWh
    # Off-Peak (6am-4pm weekdays): ~$0.36/kWh  
    # On-Peak (4pm-9pm weekdays): ~$0.52/kWh
    # Off-Peak (9pm-12am): ~$0.36/kWh
    
    prices = []
    for hour in range(24):
        if 0 <= hour < 6:  # Super Off-Peak
            price = 0.27
        elif 16 <= hour < 21:  # On-Peak
            price = 0.52
        else:  # Off-Peak
            price = 0.36
        
        prices.append({
            "hour": hour,
            "price_per_kwh": price,
            "period": "super_off_peak" if hour < 6 else ("on_peak" if 16 <= hour < 21 else "off_peak")
        })
    
    result = {
        "utility": "SDG&E",
        "tariff": "EV-TOU-5",
        "date": date or "tomorrow",
        "prices": prices,
        "currency": "USD"
    }
    
    return json.dumps(result, indent=2)


@tool
def fetch_caiso_carbon(date: Optional[str] = None) -> str:
    """
    Fetch CAISO grid carbon intensity forecast (lbs CO2 per MWh).
    Uses previous day as proxy for tomorrow as suggested in project spec.
    
    Args:
        date: Date string in YYYY-MM-DD format (defaults to tomorrow)
    
    Returns:
        JSON string with hourly carbon intensity for 24 hours
    """
    # Realistic CAISO carbon intensity pattern:
    # Night (low demand, more renewables): ~250-300 lbs/MWh
    # Morning ramp (gas plants starting): ~400-450 lbs/MWh
    # Midday (solar peak): ~200-250 lbs/MWh
    # Evening peak (high gas, solar ramping down): ~500-550 lbs/MWh
    # Late evening: ~350-400 lbs/MWh
    
    carbon_intensity = []
    for hour in range(24):
        if 0 <= hour < 6:  # Night - clean
            intensity = 250 + hour * 5
        elif 6 <= hour < 10:  # Morning ramp - getting dirtier
            intensity = 300 + (hour - 6) * 40
        elif 10 <= hour < 16:  # Solar peak - cleanest
            intensity = 200 + abs(hour - 13) * 10
        elif 16 <= hour < 22:  # Evening peak - dirtiest
            intensity = 400 + (hour - 16) * 25
        else:  # Late evening
            intensity = 400 - (hour - 22) * 25
        
        carbon_intensity.append({
            "hour": hour,
            "carbon_intensity_lbs_per_mwh": intensity,
            "intensity_level": "low" if intensity < 300 else ("medium" if intensity < 400 else "high")
        })
    
    result = {
        "source": "CAISO",
        "region": "California",
        "date": date or "tomorrow",
        "carbon_data": carbon_intensity,
        "unit": "lbs_co2_per_mwh"
    }
    
    return json.dumps(result, indent=2)


@tool
def solve_dr_optimization(
    appliances_json: str,
    prices_json: str,
    carbon_json: str,
    optimization_goal: str = "cost"
) -> str:
    """
    Solve the demand response optimization problem to schedule appliances.
    
    Args:
        appliances_json: JSON string with appliance specifications
        prices_json: JSON string with hourly electricity prices
        carbon_json: JSON string with hourly carbon intensity
        optimization_goal: Either "cost", "carbon", or "both" (weighted combination)
    
    Returns:
        JSON string with optimal schedule and savings metrics
    """
    try:
        # Parse inputs
        appliances = json.loads(appliances_json)
        price_data = json.loads(prices_json)
        carbon_data = json.loads(carbon_json)
        
        # Extract price and carbon arrays
        prices = np.array([p["price_per_kwh"] for p in price_data["prices"]])
        carbon = np.array([c["carbon_intensity_lbs_per_mwh"] for c in carbon_data["carbon_data"]])
        
        H = 24  # 24-hour horizon
        A = len(appliances)
        P_max_house = appliances[0].get("household_peak_limit", 15.0)  # kW
        
        # Create decision variables
        x = {}
        for a in range(A):
            x[a] = cp.Variable(H)
        
        # Build objective based on goal
        total_cost = sum(prices[h] * sum(x[a][h] for a in range(A)) for h in range(H))
        total_carbon = sum(carbon[h] * sum(x[a][h] for a in range(A)) for h in range(H)) / 1000  # MWh conversion
        
        if optimization_goal == "cost":
            objective = cp.Minimize(total_cost)
        elif optimization_goal == "carbon":
            objective = cp.Minimize(total_carbon)
        else:  # both - weighted combination (normalize first)
            # Normalize by typical values
            normalized_cost = total_cost / 10.0  # Typical daily cost ~$10
            normalized_carbon = total_carbon / 30.0  # Typical daily carbon ~30 lbs
            objective = cp.Minimize(normalized_cost + normalized_carbon)
        
        # Build constraints
        constraints = []
        
        for a, app in enumerate(appliances):
            alpha = app["start_hour"]
            beta = app["end_hour"]
            E_a = app["energy_required_kwh"]
            P_min = app.get("min_power_kw", 0.0)
            P_max = app["max_power_kw"]
            
            # Energy requirement constraint
            constraints.append(cp.sum(x[a][alpha:beta+1]) == E_a)
            
            # Power limits within operation window
            for h in range(alpha, beta + 1):
                constraints.append(x[a][h] >= P_min)
                constraints.append(x[a][h] <= P_max)
            
            # Zero consumption outside window
            for h in range(alpha):
                constraints.append(x[a][h] == 0)
            for h in range(beta + 1, H):
                constraints.append(x[a][h] == 0)
        
        # Household peak constraint
        for h in range(H):
            constraints.append(sum(x[a][h] for a in range(A)) <= P_max_house)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status != "optimal":
            return json.dumps({
                "status": "failed",
                "error": f"Optimization failed with status: {problem.status}"
            })
        
        # Extract solution
        schedule = {}
        for a, app in enumerate(appliances):
            schedule[app["name"]] = {
                "hourly_consumption_kwh": [float(x[a].value[h]) if x[a].value[h] > 0.01 else 0.0 
                                           for h in range(H)],
                "total_energy_kwh": float(sum(x[a].value)),
                "operating_hours": [h for h in range(H) if x[a].value[h] > 0.1]
            }
        
        # Calculate baseline (no DR - use appliances at earliest available time)
        baseline_cost = 0
        baseline_carbon = 0
        for a, app in enumerate(appliances):
            alpha = app["start_hour"]
            E_a = app["energy_required_kwh"]
            P_max = app["max_power_kw"]
            hours_needed = int(np.ceil(E_a / P_max))
            
            for h in range(alpha, alpha + hours_needed):
                power = min(P_max, E_a - (h - alpha) * P_max)
                if power > 0:
                    baseline_cost += prices[h] * power
                    baseline_carbon += (carbon[h] / 1000) * power
        
        # Optimized metrics
        optimized_cost = float(sum(prices[h] * sum(x[a].value[h] for a in range(A)) for h in range(H)))
        optimized_carbon = float(sum(carbon[h] * sum(x[a].value[h] for a in range(A)) for h in range(H)) / 1000)
        
        result = {
            "status": "success",
            "optimization_goal": optimization_goal,
            "schedule": schedule,
            "metrics": {
                "optimized_cost_dollars": round(optimized_cost, 2),
                "baseline_cost_dollars": round(baseline_cost, 2),
                "cost_savings_dollars": round(baseline_cost - optimized_cost, 2),
                "cost_savings_percent": round(100 * (baseline_cost - optimized_cost) / baseline_cost, 1),
                "optimized_carbon_lbs": round(optimized_carbon, 2),
                "baseline_carbon_lbs": round(baseline_carbon, 2),
                "carbon_reduction_lbs": round(baseline_carbon - optimized_carbon, 2),
                "carbon_reduction_percent": round(100 * (baseline_carbon - optimized_carbon) / baseline_carbon, 1)
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


# ============================================================================
# AGENT SETUP
# ============================================================================

def create_dr_agent(model_name: str = "gpt-4-turbo"):
    """
    Create the DRAgent using LangChain's agent framework.
    
    Args:
        model_name: OpenAI model to use
    
    Returns:
        AgentExecutor instance
    """
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Define tools
    tools = [fetch_sdge_prices, fetch_caiso_carbon, solve_dr_optimization]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are DRAgent, an expert residential demand response assistant that helps homeowners optimize their electricity usage to save money and reduce carbon emissions.

Your capabilities:
1. Retrieve real-time electricity prices from SDG&E
2. Fetch grid carbon intensity data from CAISO
3. Solve constrained optimization problems to schedule flexible appliances
4. Explain recommendations in clear, user-friendly language

Process for handling user requests:
1. UNDERSTAND: Parse what appliances the user wants to schedule and their constraints
2. RETRIEVE: Fetch tomorrow's electricity prices and carbon intensity data
3. OPTIMIZE: Use the optimization tool with the appliance specifications and data
4. EXPLAIN: Generate a clear, actionable report with:
   - Recommended schedule for each appliance (what time to run them)
   - Expected savings (both cost in dollars and carbon in lbs)
   - Clear explanation of WHY these times are optimal
   - Comparison to what would happen without optimization

Important guidelines:
- Always fetch BOTH price and carbon data before optimizing
- Be specific about times (e.g., "Charge your EV from 1:00 AM to 3:00 AM")
- Quantify savings clearly
- Explain the reasoning (e.g., "during super off-peak hours when electricity is cheapest")
- If information is missing, ask the user for clarification
- If optimization fails, explain what went wrong and what's needed

Appliance specification format:
{
  "name": "EV",
  "energy_required_kwh": 16.0,
  "start_hour": 22,  // 10 PM
  "end_hour": 7,     // 7 AM next day (use 7, not 31)
  "min_power_kw": 0.0,
  "max_power_kw": 11.0
}

Remember: Your goal is to make demand response accessible and trustworthy for everyday homeowners."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ============================================================================
# BASELINE: Zero-Shot LLM (No Tools)
# ============================================================================

def create_baseline_llm(model_name: str = "gpt-4-turbo"):
    """
    Create a baseline LLM that does NOT have access to tools.
    Uses only its general knowledge to make recommendations.
    
    Returns:
        ChatOpenAI instance configured for zero-shot prompting
    """
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    return llm


def run_baseline_recommendation(llm, user_query: str) -> str:
    """
    Get a recommendation from the baseline LLM without tools.
    
    Args:
        llm: ChatAnthropic instance
        user_query: User's demand response query
    
    Returns:
        LLM's text response
    """
    
    baseline_prompt = f"""You are a helpful assistant for residential demand response. The user wants advice on scheduling their appliances to save money and reduce carbon emissions.

Based on your general knowledge of typical electricity pricing patterns and grid operations:
- Electricity is usually cheaper late at night and more expensive during evening peak hours (4-9 PM)
- Grid carbon intensity is typically lower during midday (when solar is abundant) and higher during evening peaks
- Time-of-Use rates commonly have super off-peak (~$0.27/kWh), off-peak (~$0.36/kWh), and on-peak (~$0.52/kWh) periods

User query: {user_query}

Provide a specific recommendation including:
1. When to run each appliance
2. Estimated cost savings
3. Estimated carbon reduction
4. Brief explanation of your reasoning

Be specific with times and dollar amounts based on typical patterns."""

    response = llm.invoke(baseline_prompt)
    return response.content


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DRAgent: Residential Demand Response Assistant")
    print("=" * 80)
    
    # Example user query
    user_query = """I need help scheduling my appliances for tomorrow to save money.

I have:
- An electric vehicle that needs 16 kWh of charging. I'll plug it in at 10 PM and need it ready by 7 AM. It can charge at up to 11 kW.
- A dishwasher that needs 3.6 kWh. I can run it anytime between 8 PM and 11 PM. Max power is 2 kW.
- A clothes dryer that needs 4.5 kWh. I can run it between 9 PM and midnight. Max power is 4 kW.

My house has a 15 kW peak limit. I want to minimize my electricity cost. What should I do?"""

    print("\n" + "=" * 80)
    print("TESTING AGENTIC AI APPROACH")
    print("=" * 80)
    
    # Create and run agent
    agent = create_dr_agent()
    result = agent.invoke({"input": user_query})
    
    print("\n" + "-" * 80)
    print("AGENT OUTPUT:")
    print("-" * 80)
    print(result["output"])
    
    print("\n" + "=" * 80)
    print("TESTING BASELINE LLM (No Tools)")
    print("=" * 80)
    
    # Create and run baseline
    baseline_llm = create_baseline_llm()
    baseline_result = run_baseline_recommendation(baseline_llm, user_query)
    
    print("\n" + "-" * 80)
    print("BASELINE OUTPUT:")
    print("-" * 80)
    print(baseline_result)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
