import streamlit as st

st.set_page_config(page_title="Report History", page_icon="🗂️", layout="wide")
st.title("🗂️ Session Report History")
st.write("View and download all reports generated during this session.")
st.markdown("---")

if 'report_history' not in st.session_state: st.session_state['report_history'] = []

if not st.session_state['report_history']:
    st.info("No reports generated yet. Analyze an image on the 'Image Analysis' page to create one.")
else:
    st.subheader("Available Reports")
    for i, report in enumerate(st.session_state['report_history']):
        with st.container(border=True):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"**{report['name']}**")
                st.caption(f"Generated on: {report['date']}")
            with col2:
                st.download_button("⬇️ Download PDF", data=report['data'], file_name=report['file_name'], mime="application/pdf", key=f"dl_{i}")