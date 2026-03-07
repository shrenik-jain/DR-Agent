"""
app.py - Simple Gradio UI for DRAgent
Run with: python app.py
"""

import os
import gradio as gr
from dr_agent import create_dr_agent

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set. Set it before running.")

# Initialize the agent once at startup
agent = create_dr_agent()

def _build_context(message: str, history: list) -> str:
    """
    Build input with conversation history so the agent remembers prior answers.
    Limits to last 10 exchanges to avoid token overflow.
    Handles both tuple format [(user, bot), ...] and messages format [{"role":..., "content":...}, ...].
    """
    if not history:
        return message
    lines = []
    for turn in history[-10:]:
        user_msg, bot_msg = None, None
        if isinstance(turn, (list, tuple)):
            user_msg = turn[0] if len(turn) > 0 else None
            bot_msg = turn[1] if len(turn) > 1 else None
        elif isinstance(turn, dict):
            if turn.get("role") == "user":
                user_msg = turn.get("content", "")
            else:
                bot_msg = turn.get("content", "")
        if user_msg:
            lines.append(f"User: {user_msg}")
        if bot_msg:
            lines.append(f"Assistant: {bot_msg}")
    lines.append(f"User: {message}")
    return "\n\n".join(lines)


def chat_with_agent(message, history):
    """
    Handle chat messages and return agent response.
    Passes conversation history so the agent uses prior user answers.
    
    Args:
        message: User's input message
        history: Chat history (list of [user_msg, bot_msg] pairs)
    
    Returns:
        Agent's response string
    """
    try:
        context = _build_context(message, history)
        result = agent.invoke({"input": context})
        response = result["output"]
        return response
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your API key and try again."


# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="🔌 DRAgent - Residential Demand Response Assistant",
    description="""
    **Welcome to DRAgent!** I help you optimize your home appliance schedules to save money and reduce carbon emissions.
    
    **What I can do:**
    - 📊 Fetch real-time electricity prices from SDG&E
    - 🌱 Get grid carbon intensity data from CAISO  
    - ⚡ Optimize appliance schedules with real mathematical optimization
    - 💰 Show you exact cost savings and carbon reductions
    
    **Try asking me:**
    - "Help me schedule my EV charging tonight to save money"
    - "I want to run my dishwasher, dryer, and charge my EV - what's the best schedule?"
    - "How can I reduce my carbon footprint with my appliances?"
    
    **Pro tip:** Include appliance details like energy needed (kWh), time window, and max power (kW) for best results. If you leave details out, I'll ask follow-up questions and use sensible defaults where possible.
    """,
    examples=[
        "I need to charge my Tesla Model 3 tonight. It needs 16 kWh, I'll plug it in at 10 PM and need it by 7 AM. Max charging rate is 11 kW. Help me minimize cost!",
        "Schedule my appliances for tomorrow: EV needs 16 kWh (10 PM-7 AM, max 11 kW), Dishwasher needs 3.6 kWh (8 PM-11 PM, max 2 kW). My house limit is 15 kW.",
        "I want to reduce carbon emissions. Help schedule: EV 16 kWh (10 PM-7 AM, 11 kW max), Dryer 4.5 kWh (9 PM-midnight, 4 kW max). House limit 15 kW."
    ],
)

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        share=True,  # Set to True for public Colab link
        debug=True
    )