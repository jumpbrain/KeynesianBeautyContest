"""
Entry point for the LLM Keynes Beauty Contest app
Initialize logging, env vars, and styling as needed
Ensure an Arena exists in session via Arena.default()
Delegate rendering duties to Display for all UI components

To see it in action, run:
python -m streamlit run app.py
"""

from dotenv import load_dotenv
import logging
from game.arenas import Arena
import streamlit as st
from util.setup import setup_logger, STYLE
from views.displays import Display
from models.storage import MoveLogger

root = logging.getLogger()
if "root" not in st.session_state:
    st.session_state.root = root
    setup_logger(root)

load_dotenv(override=True)

st.set_page_config(
    layout="wide",
        page_title="Keynesian Beauty Contest",
    menu_items={
        "About": "Beauty is an LLM arena running repeated Keynes Beauty Contests."
    },
    page_icon="🦾",
    initial_sidebar_state="expanded",
)
st.markdown(STYLE, unsafe_allow_html=True)

if "arena" not in st.session_state:
    st.session_state.arena = Arena.default()

# Load persistent move log into session state for analytics and quick access
try:
    st.session_state.move_log = MoveLogger.load_df()
except Exception:
    st.session_state.move_log = None

if "auto_move" not in st.session_state:
    st.session_state.auto_move = False
    st.session_state.do_move = False
arena = st.session_state.arena

Display(arena).display_page()
