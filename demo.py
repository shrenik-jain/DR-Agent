"""
Simple demo script for DRAgent
Shows basic usage and example outputs
"""

import os
from dr_agent import create_dr_agent, create_baseline_llm, run_baseline_recommendation


def demo_basic_usage():
    """
    Basic demonstration of DRAgent functionality.
    """
    
    print("=" * 80)
    print("DRAgent Demo - Basic Usage")
    print("=" * 80)
    
    # Example 1: Simple EV charging optimization
    print("\n--- Example 1: EV Charging Optimization ---\n")
    
    query1 = """I need to charge my Tesla Model 3 tonight. It needs 16 kWh of charging.
I'll plug it in at 10 PM and need it ready by 7 AM tomorrow morning.
The car can charge at up to 11 kW.

Can you help me figure out the best time to charge it to minimize my electricity bill?"""

    agent = create_dr_agent()
    result = agent.invoke({"input": query1})
    print("AGENT RESPONSE:")
    print(result["output"])
    
    print("\n" + "=" * 80)
    
    # Example 2: Multiple appliances with carbon goal
    print("\n--- Example 2: Multiple Appliances + Carbon Goal ---\n")
    
    query2 = """I want to reduce my carbon footprint. Help me schedule these appliances:

1. Electric Vehicle: needs 16 kWh, available 10 PM to 7 AM, max 11 kW
2. Dishwasher: needs 3.6 kWh, available 8 PM to 11 PM, max 2 kW  
3. Clothes Dryer: needs 4.5 kWh, available 9 PM to midnight, max 4 kW

My house has a 15 kW peak limit. I care more about reducing emissions than saving money."""

    result = agent.invoke({"input": query2})
    print("AGENT RESPONSE:")
    print(result["output"])
    
    print("\n" + "=" * 80)


def demo_comparison():
    """
    Compare agent vs baseline on the same query.
    """
    
    print("\n" + "=" * 80)
    print("DRAgent vs Baseline Comparison")
    print("=" * 80)
    
    query = """Help me schedule my EV and dishwasher for tomorrow to save money.

- EV: needs 16 kWh, available 10 PM to 7 AM, max 11 kW
- Dishwasher: needs 3.6 kWh, available 8 PM to 11 PM, max 2 kW

Household peak: 15 kW"""
    
    # Agent with tools
    print("\n--- AGENT (With Tool Access) ---\n")
    agent = create_dr_agent()
    agent_result = agent.invoke({"input": query})
    print(agent_result["output"])
    
    # Baseline without tools
    print("\n" + "=" * 80)
    print("--- BASELINE (No Tools) ---\n")
    baseline_llm = create_baseline_llm()
    baseline_result = run_baseline_recommendation(baseline_llm, query)
    print(baseline_result)
    
    print("\n" + "=" * 80)
    print("\nKEY DIFFERENCES:")
    print("- Agent uses real SDG&E price data and CAISO carbon data")
    print("- Agent runs actual optimization to guarantee feasibility")
    print("- Agent provides precise, verifiable savings numbers")
    print("- Baseline uses only general knowledge and approximations")
    print("=" * 80)


def demo_failure_cases():
    """
    Demonstrate how the agent handles failure cases.
    """
    
    print("\n" + "=" * 80)
    print("DRAgent - Handling Failure Cases")
    print("=" * 80)
    
    agent = create_dr_agent()
    
    # Failure Case 1: Missing information
    print("\n--- Failure Case 1: Missing Critical Information ---\n")
    
    query1 = """I want to charge my EV to save money. Help me out!"""
    
    result = agent.invoke({"input": query1})
    print("AGENT RESPONSE:")
    print(result["output"])
    print("\nExpected: Agent should ask for missing information (energy needed, time window, etc.)")
    
    # Failure Case 2: Infeasible constraints
    print("\n" + "=" * 80)
    print("--- Failure Case 2: Infeasible Constraints ---\n")
    
    query2 = """Schedule my appliances:
- EV: needs 50 kWh, available 10 PM to 11 PM (only 1 hour!), max 11 kW

This is impossible since 11 kW * 1 hour = 11 kWh < 50 kWh needed."""
    
    result = agent.invoke({"input": query2})
    print("AGENT RESPONSE:")
    print(result["output"])
    print("\nExpected: Agent should identify infeasibility and explain why")
    
    # Failure Case 3: Conflicting goals
    print("\n" + "=" * 80)
    print("--- Failure Case 3: Ambiguous/Conflicting Goals ---\n")
    
    query3 = """I want to save money but also reduce carbon and also I heard my neighbor 
runs everything during peak hours which seems wrong but maybe they have solar panels?
Should I copy them or do something different? I have an EV that needs charging."""
    
    result = agent.invoke({"input": query3})
    print("AGENT RESPONSE:")
    print(result["output"])
    print("\nExpected: Agent should clarify the user's actual goal and request missing specs")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    # Run demos
    demo_basic_usage()
    demo_comparison()
    demo_failure_cases()
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
