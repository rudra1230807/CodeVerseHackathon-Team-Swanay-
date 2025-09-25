import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time
import os
from datetime import datetime
from streamlit_cropper import st_cropper
import plotly.express as px
from utils import load_yolo_model, preprocess_frame, create_pdf_report, create_detection_heatmap

st.set_page_config(page_title="Image Analysis", layout="wide")

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

st.title("🔬 Image Analysis Workflow")
st.write("Upload an image, calibrate its real-world size, select the ROI, and run the analysis.")
st.divider()

if 'report_history' not in st.session_state: st.session_state['report_history'] = []

uploaded_file = st.sidebar.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    orig_image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Step 1: Calibrate Image Area")
    st.info("Enter the real-world dimensions of the area visible in the entire image.")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        real_width_m = col1.number_input("Real-world width (meters)", min_value=0.1, value=1.0, step=0.1, key="real_width")
        real_height_m = col2.number_input("Real-world height (meters)", min_value=0.1, value=1.0, step=0.1, key="real_height")
    
    st.subheader("Step 2: Select Region of Interest (ROI)")
    with st.container(border=True):
        cropped_image = st_cropper(orig_image, realtime_update=True, box_color='lime', aspect_ratio=None, return_type='image')
    st.image(cropped_image, caption="Preview of Selected Region", use_column_width=True)

    if st.button("🚀 Analyze Selected Region", use_container_width=True):
        model = load_yolo_model("/Users/rudra/SY/CodeVerse/p2/runs/detect/yolov8_biofouling_fast3/weights/best.pt")
        if not model: st.error("Model not found."); st.stop()
        
        with st.spinner("Analyzing selected region..."):
            frame = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
            enhanced = preprocess_frame(frame)
            results = model(enhanced)
            res = results[0]
            
            rows, fouling_area_pixels = [], 0
            for box in res.boxes:
                name = model.names[int(box.cls[0])]
                rows.append([name, f"{float(box.conf[0]):.2f}"])
                x1, y1, x2, y2 = box.xyxy[0]; fouling_area_pixels += (x2 - x1) * (y2 - y1)
            
            details_df = pd.DataFrame(rows, columns=["Object", "Confidence"])
            summary_counts = details_df['Object'].value_counts().reset_index() if not details_df.empty else pd.DataFrame(columns=['Object', 'Count'])
            if 'Object' in summary_counts: summary_counts.columns = ['Object', 'Count']

            pixel_total_roi_area = cropped_image.width * cropped_image.height
            coverage = (fouling_area_pixels / pixel_total_roi_area) * 100 if pixel_total_roi_area > 0 else 0
            coverage = min(coverage, 100.0)

            real_total_image_area = real_width_m * real_height_m
            real_roi_area_m2 = real_total_image_area * (pixel_total_roi_area / (orig_image.width * orig_image.height))
            real_fouling_area_m2 = real_roi_area_m2 * (coverage / 100)
            
            if coverage < 10: severity, recommendation = "Light", "Monitor condition."
            elif coverage < 30: severity, recommendation = "Moderate", "Cleaning recommended."
            else: severity, recommendation = "Heavy", "Immediate cleaning is critical."

            annotated_cropped_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            cropped_image_np_bgr = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
            heatmap_image = create_detection_heatmap(cropped_image_np_bgr, res.boxes)
            
            st.session_state['analysis_results'] = {
                "details_df": details_df, "summary_counts": summary_counts, "coverage": coverage, 
                "severity": severity, "recommendation": recommendation, "original_image": cropped_image, 
                "annotated_image": annotated_cropped_rgb, "heatmap_image": heatmap_image,
                "fouling_area_pixels": fouling_area_pixels, "real_fouling_area_m2": real_fouling_area_m2, 
                "real_roi_area_m2": real_roi_area_m2
            }
            
            fig_bar = px.bar(summary_counts, x='Count', y='Object', orientation='h')
            graph_bytes = fig_bar.to_image(format="png")

            pdf_buffer = create_pdf_report(
                cropped_image, annotated_cropped_rgb, summary_counts, coverage, severity, recommendation, 
                heatmap_image, graph_bytes, real_fouling_area_m2, real_roi_area_m2
            )
            if pdf_buffer:
                report_name = f"ROI Analysis of '{uploaded_file.name}'"
                new_report = {"name": report_name, "date": datetime.now().strftime('%Y-%m-%d %H:%M'), "data": pdf_buffer, "file_name": f"report_roi_{datetime.now().strftime('%H%M%S')}.pdf"}
                st.session_state['report_history'].insert(0, new_report)
        
        st.success("✅ Analysis Complete!")
        st.info("Navigate to the Dashboard or Report History pages.")
else:
    st.info("Please upload an image using the sidebar to begin.")