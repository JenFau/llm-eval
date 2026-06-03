# =============================================================================
# IMPORTS
# =============================================================================
# Standard library modules
import os
import time
from datetime import datetime

# Streamlit turns this Python script into an interactive web app.
# Every widget (dropdowns, text boxes, buttons) is created by calling st.*
import streamlit as st

# ThreadPoolExecutor lets us call multiple models simultaneously rather than
# one after another, so both responses arrive at roughly the same time.
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reads API keys from the .env file into the environment
from dotenv import load_dotenv

load_dotenv()

# Our Groq client — handles communication with the Groq API
from clients.groq_client import GroqClient


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Be concise and direct."

# Must be the first Streamlit call — sets the browser tab title and layout
st.set_page_config(
    page_title="LLM Eval Project",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE
# =============================================================================
# Streamlit reruns the entire script on every user interaction, so normal
# variables reset each time. st.session_state persists values across reruns.
# The pattern below initialises each key only on the very first run.

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "responses" not in st.session_state:
    # Stores the last set of model responses: { model_name: { response, timing } }
    st.session_state.responses = {}

if "current_prompt" not in st.session_state:
    # The prompt that produced the currently displayed responses
    st.session_state.current_prompt = ""

if "timeout" not in st.session_state:
    st.session_state.timeout = 120


# =============================================================================
# API CLIENT
# =============================================================================
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

client = GroqClient(
    api_key=GROQ_KEY,
    timeout=st.session_state.timeout,
)


# =============================================================================
# SIDEBAR
# =============================================================================
# Everything inside this block renders in the left sidebar panel.
with st.sidebar:
    st.title("🧪 LLM Eval Project")
    st.caption("Model comparison tool")

    # Halt the app if no API key is set
    if not GROQ_KEY:
        st.error("Set GROQ_API_KEY in .env")
        st.stop()

    # Fetch the live list of available models from Groq
    available_models = client.list_models()

    # Preferred defaults — picks the first two that Groq currently offers,
    # falling back to whatever is at the top of the list if none match
    PREFERRED_DEFAULTS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
    ]
    default_models = ([m for m in PREFERRED_DEFAULTS if m in available_models] or available_models)[:2]

    selected_models = st.multiselect(
        "Models to compare",
        options=available_models,
        default=default_models,
    )

    st.divider()

    # System prompt sent to every model — user can edit or clear it
    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=150,
    )

    # How long to wait for a model response before giving up
    st.number_input(
        "Response timeout (s)",
        min_value=30,
        max_value=600,
        step=30,
        key="timeout",
    )


# =============================================================================
# GUARD: AT LEAST ONE MODEL MUST BE SELECTED
# =============================================================================
if not selected_models:
    st.info("Select at least one model in the sidebar to start.")
    st.stop()


# =============================================================================
# DISPLAY PREVIOUS RESULTS
# =============================================================================
# On every rerun, redraw the last prompt and responses if they exist.
# Streamlit doesn't remember what it rendered before — the page is rebuilt
# from scratch each time — so we pull the saved values from session_state.
st.markdown("## Prompt")

if st.session_state.current_prompt and st.session_state.responses:
    st.markdown(f"**Prompt:** {st.session_state.current_prompt}")
    cols = st.columns(len(selected_models))
    for col, model in zip(cols, selected_models):
        with col:
            st.markdown(f"**{model}**")
            data = st.session_state.responses.get(model, {})
            if data:
                st.markdown(data.get("response", ""))
                st.caption(f"⏱ {data.get('timing', '?')}s")


# =============================================================================
# PROMPT INPUT & MODEL CALLS
# =============================================================================
# st.chat_input renders the text box pinned to the bottom of the page.
# Returns the submitted string, or None if nothing has been typed yet.
user_input = st.chat_input("Enter your prompt...")

if user_input:
    history = [{"role": "user", "content": user_input}]

    def stream_model(model: str) -> tuple[str, str, float]:
        """Calls one model and collects its full response. Runs in a background thread."""
        t0 = time.time()
        full = ""
        for chunk in client.stream_chat(model, history, system=system_prompt):
            full += chunk
        return model, full, round(time.time() - t0, 1)

    # Show the prompt and create a placeholder column for each model.
    # Placeholders display a waiting message until the response arrives.
    st.markdown(f"**Prompt:** {user_input}")
    cols = st.columns(len(selected_models))
    placeholders = {}
    for i, m in enumerate(selected_models):
        cols[i].markdown(f"**{m}**")
        ph = cols[i].empty()
        ph.markdown(f"*⏳ Waiting for {m.split(':')[0]}...*")
        placeholders[m] = ph

    response_texts = {}
    elapsed_times = {}

    # Call all selected models in parallel — each runs in its own thread.
    # as_completed() yields each result as soon as that model finishes,
    # so faster models appear immediately without waiting for slower ones.
    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = {executor.submit(stream_model, m): m for m in selected_models}
        for future in as_completed(futures):
            model = futures[future]
            try:
                _, response, elapsed = future.result()
                response_texts[model] = response
                elapsed_times[model] = elapsed
                placeholders[model].markdown(response)
            except Exception as e:
                # Show a readable error in the column rather than crashing the app
                error_msg = str(e)
                response_texts[model] = error_msg
                elapsed_times[model] = 0.0
                placeholders[model].error(error_msg)

    # Save results to session_state so they survive the rerun below
    st.session_state.current_prompt = user_input
    st.session_state.responses = {
        m: {"response": response_texts[m], "timing": elapsed_times[m]}
        for m in selected_models
    }
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rerun so the page redraws cleanly with results in the display section above
    st.rerun()
