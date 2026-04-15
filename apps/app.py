import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

from dragent import create_dr_agent

# Repo-root .env (OPENAI_API_KEY, etc.)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set. Set it before running.")

# Token streaming for the chat UI (LangChain forwards streamed chunks to callbacks).
agent = create_dr_agent(streaming=True)

_DONE = object()


class _TokenQueueCallback(BaseCallbackHandler):
    """Pushes each streamed LLM token to a queue for the Gradio generator thread."""

    def __init__(self, token_queue: "queue.Queue[Any]") -> None:
        self._q = token_queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self._q.put(token)


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


def chat_with_agent(message: str, history: list) -> Generator[str, None, None]:
    """
    Stream assistant text as the LLM produces tokens. Tool calls still run between
    streamed segments; the UI updates whenever new tokens arrive.
    """
    context = _build_context(message, history)
    token_queue: "queue.Queue[Any]" = queue.Queue()
    state: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            handler = _TokenQueueCallback(token_queue)
            result = agent.invoke(
                {"input": context},
                config={"callbacks": [handler]},
            )
            state["result"] = result
        except Exception as exc:
            state["error"] = exc
        finally:
            token_queue.put(_DONE)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    accumulated = ""
    while True:
        try:
            item = token_queue.get(timeout=0.2)
        except queue.Empty:
            # Keep Gradio refreshing while tools / model work
            yield accumulated
            continue

        if item is _DONE:
            break

        accumulated += str(item)
        yield accumulated

    thread.join(timeout=600)

    if state.get("error") is not None:
        err = state["error"]
        yield f"❌ Error: {err}\n\nPlease check your API key and try again."
        return

    result: Optional[Dict[str, Any]] = state.get("result")
    final_text = (result or {}).get("output", accumulated) if result else accumulated
    if final_text != accumulated:
        yield final_text


# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="DRAgent - Residential Demand Response Assistant",
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
        "I want to reduce carbon emissions. Help schedule: EV 16 kWh (10 PM-7 AM, 11 kW max), Dryer 4.5 kWh (9 PM-midnight, 4 kW max). House limit 15 kW.",
    ],
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
    )
