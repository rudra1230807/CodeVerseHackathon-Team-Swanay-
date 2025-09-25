# Home.py

import streamlit as st

st.set_page_config(
    page_title="Mssrine-Foul-Detect",
    page_icon="⚓",
    layout="wide"
)

# --- FUNCTION TO LOAD CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- LOAD THE CUSTOM CSS ---
local_css("style.css")


# --- HEADER ---
with st.container():
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image("Swanay.png", width=150)
    with col2:
        st.title("Marine-Foul-Detect")
        st.write("""
        A visually enhanced, dynamic application for automated biofouling analysis, 
        built purely in Streamlit.
        """)

st.divider()

# --- FEATURE CARDS (These will now have the frosted glass effect) ---
st.header("✨ Key Features")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("⚡ Dynamic Analysis")
        st.write("Upload an image and receive live progress updates during analysis.")
        st.info("Powered by a fine-tuned YOLOv8 model for high accuracy.")

with col2:
    with st.container(border=True):
        st.subheader("📊 Interactive Dashboard")
        st.write("Explore analysis results with dynamic charts and metrics.")
        st.info("Automatically quantify fouling coverage and classify severity.")

with col3:
    with st.container(border=True):
        st.subheader("📋 Instant Reports")
        st.write("Generate and view a history of professional PDF reports from your session.")
        st.info("Download detailed analysis data with a single click.")

st.sidebar.success("Select a page above to begin.")