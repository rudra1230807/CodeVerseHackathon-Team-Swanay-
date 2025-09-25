import streamlit as st
import plotly.express as px
from streamlit_image_comparison import image_comparison
import pandas as pd
from utils import create_pdf_report
from datetime import datetime
import os

st.set_page_config(page_title="Analysis Dashboard", page_icon="📊", layout="wide")

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

st.title("📊 Analysis Dashboard")
st.write("Explore the results of your most recent ROI analysis.")
st.divider()

if 'analysis_results' not in st.session_state:
    st.warning("⚠️ No analysis data found. Please analyze an image first.")
else:
    results = st.session_state['analysis_results']
    
    st.subheader("Key Performance Indicators for Selected ROI")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fouling Coverage", f"{results['coverage']:.2f}%")
        col2.metric("Species Detected", results['details_df']['Object'].nunique())
        col3.metric("Fouling Area Density", f"{results.get('real_fouling_area_m2', 0):.3f} m²")
        col4.metric("Severity Level", results['severity'])
    
    st.info(f"**Recommendation:** {results['recommendation']}")
    st.divider()

    fig_bar = px.bar(results['summary_counts'], x='Count', y='Object', orientation='h', title="Detection Counts per Species", color='Object')
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "🖼️ ROI Comparison", "🔥 Heatmap", "📋 Detailed Data"])
    with tab1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with tab2:
        image_comparison(img1=results['original_image'], label1="Original ROI", img2=results['annotated_image'], label2="Annotated ROI")
    with tab3:
        st.image(results['heatmap_image'], caption="Red areas indicate higher density of detections.", use_column_width=True)
    with tab4:
        st.dataframe(results['details_df'], use_container_width=True)
        csv = results['details_df'].to_csv(index=False).encode('utf-8')
        st.download_button("Download ROI Data (CSV)", data=csv, file_name="roi_detection_details.csv", mime="text/csv")

    st.divider()
    st.subheader("Download Full Report")
    with st.container(border=True):
        graph_bytes = fig_bar.to_image(format="png")
        pdf_buffer = create_pdf_report(
            results['original_image'], results['annotated_image'], results['summary_counts'], results['coverage'], 
            results['severity'], results['recommendation'], results['heatmap_image'], graph_bytes, 
            results['real_fouling_area_m2'], results['real_roi_area_m2']
        )
        if pdf_buffer:
            st.download_button(label="📥 Download PDF Report", data=pdf_buffer, 
                               file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                               mime="application/pdf", use_container_width=True)