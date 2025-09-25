import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as PlatypusImage, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from streamlit_image_comparison import image_comparison

# --- Page config ---
st.set_page_config(page_title="Marine-Foul-Detect", layout="wide")

# --- Paths ---
MODEL_PATH = '/Users/rudra/SY/CodeVerse/p2 copy/runs/detect/yolov8_biofouling_fast3/weights/best.pt'
LOGO_PATH = "Swanay.png"

# --- Load YOLOv8 model ---
@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

# --- Image preprocessing ---
def preprocess_frame(frame, use_denoise, use_clahe, use_sharpen):
    processed = frame.copy()
    if use_denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    if use_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    if use_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)
    return processed

# --- PDF report generation ---
def create_pdf_report(orig_img, annotated_img, summary_df, coverage, severity, recommendation):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Marine Fouling Detection Report", styles['h1']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Overall Fouling Coverage:</b> {coverage:.2f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Severity Level:</b> {severity}", styles['Normal']))
    story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles['Normal']))

    orig_buffer = io.BytesIO()
    annotated_buffer = io.BytesIO()
    orig_img.save(orig_buffer, format="PNG")
    Image.fromarray(annotated_img).save(annotated_buffer, format="PNG")
    orig_buffer.seek(0)
    annotated_buffer.seek(0)

    im1 = PlatypusImage(orig_buffer, width=200, height=150)
    im2 = PlatypusImage(annotated_buffer, width=200, height=150)
    story.append(Table([[im1, im2]], colWidths=[220, 220]))

    story.append(Paragraph("Detection Summary by Class", styles['h2']))
    table_data = [['Object Type', 'Count']] + summary_df.values.tolist()
    summary_table = Table(table_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- UI Layout ---
col_logo, col_title = st.columns([0.1, 0.9])
if os.path.exists(LOGO_PATH):
    with col_logo:
        st.image(LOGO_PATH, width=120)
with col_title:
    st.title("Marine-Foul-Detect")
    st.write("An advanced tool for detecting and quantifying marine fouling on ship hulls using YOLOv8.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Analysis Controls")
uploaded_file = st.sidebar.file_uploader("Upload an underwater image", type=["jpg","jpeg","png"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
with st.sidebar.expander("Advanced Preprocessing Options"):
    use_denoise = st.checkbox("Denoise", value=True)
    use_clahe = st.checkbox("Improve Contrast (CLAHE)", value=True)
    use_sharpen = st.checkbox("Sharpen", value=True)

# --- Main App Body ---
if uploaded_file is None:
    st.info("⬅️ Upload an image using the sidebar to start the analysis.")
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    orig_image = Image.open(tmp_in_path).convert("RGB")
    img_np = np.array(orig_image)

    # --- Step 1: ROI Selection ---
    st.subheader("Step 1: Select Ship Region")
    
    # Display the original image
    st.image(orig_image, caption="Original Image")
    
    # Create columns for coordinate inputs
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.slider("X1", 0, orig_image.width, int(orig_image.width * 0.25))
        y1 = st.slider("Y1", 0, orig_image.height, int(orig_image.height * 0.25))
    
    with col2:
        x2 = st.slider("X2", 0, orig_image.width, int(orig_image.width * 0.75))
        y2 = st.slider("Y2", 0, orig_image.height, int(orig_image.height * 0.75))
    
    # --- Create ROI mask ---
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    ship_roi = cv2.bitwise_and(img_np, img_np, mask=mask)

    # Draw rectangle on the original image to visualize selection
    img_with_rect = img_np.copy()
    cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(img_with_rect, caption="Selected Region")
    st.subheader("Step 2: Selected Ship ROI")
    st.image(ship_roi, caption="Ship Region")

    # --- Preprocess ROI ---
    enhanced = preprocess_frame(ship_roi, use_denoise, use_clahe, use_sharpen)

    # --- Step 3: Run Detection ---
    if st.button("🚀 Analyze Ship ROI"):
        model = load_yolo_model(MODEL_PATH)
        if model is None:
            st.stop()
        with st.spinner("Processing..."):
            results = model(enhanced, imgsz=640)
            res = results[0]

            filtered_boxes = [box for box in res.boxes if box.conf[0] >= conf_threshold]
            fouling_area = 0
            total_area = enhanced.shape[0]*enhanced.shape[1]
            rows = []

            for box in filtered_boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                rows.append([name, f"{conf:.2f}"])
                x1_box, y1_box, x2_box, y2_box = box.xyxy[0]
                fouling_area += (x2_box - x1_box)*(y2_box - y1_box)

            details_df = pd.DataFrame(rows, columns=["Object Type","Confidence"])
            coverage = (fouling_area/total_area)*100 if total_area>0 else 0
            if coverage <10:
                severity = "Light"
                recommendation = "Monitor condition. Next cleaning in 2–3 months."
            elif coverage <30:
                severity = "Moderate"
                recommendation = "Cleaning recommended to prevent efficiency loss."
            else:
                severity = "Heavy"
                recommendation = "Immediate cleaning required."

            annotated = res.plot()
            annotated_rgb = cv2.cvtColor(annotated.astype(np.uint8), cv2.COLOR_BGR2RGB)

            # --- Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Summary & Report", "🖼️ Image Comparison", "📋 Detailed Detections"])

            with tab1:
                st.subheader("Analysis Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Detections", len(details_df))
                col2.metric("Fouling Coverage", f"{coverage:.2f}%")
                col3.metric("Severity Level", severity)
                st.info(f"**Recommendation:** {recommendation}")
                st.markdown("---")
                if not details_df.empty:
                    summary_counts = details_df['Object Type'].value_counts().reset_index()
                    summary_counts.columns = ['Object Type','Count']
                    st.bar_chart(summary_counts.set_index('Object Type')['Count'])
                    pdf_buffer = create_pdf_report(orig_image, annotated_rgb, summary_counts, coverage, severity, recommendation)
                    st.download_button("📥 Download PDF Report", pdf_buffer, file_name=f"fouling_report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
                    csv_buffer = details_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv_buffer, file_name=f"detailed_detections_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
                else:
                    st.success("✅ No fouling detected above the confidence threshold!")

            with tab2:
                st.subheader("Visual Comparison")
                image_comparison(img1=orig_image, label1="Original", img2=annotated_rgb, label2="Annotated")

            with tab3:
                st.subheader("Detailed Detection Data")
                st.dataframe(details_df)

    # --- Clean temp file ---
    if os.path.exists(tmp_in_path):
        os.remove(tmp_in_path)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import tempfile
# import os
# from PIL import Image
# import io
# from datetime import datetime
# from reportlab.lib.pagesizes import A4
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as PlatypusImage, Paragraph
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors
# from streamlit_image_comparison import image_comparison
# from streamlit_drawable_canvas import st_canvas

# # --- Page config ---
# st.set_page_config(page_title="Marine-Foul-Detect", layout="wide")

# # --- Paths ---
# MODEL_PATH = "/Users/rudra/SY/CodeVerse/p2 copy/runs/detect/yolov8_biofouling_fast3/weights/best.pt"
# LOGO_PATH = "Swanay.png"

# # --- Load YOLOv8 model ---
# @st.cache_resource
# def load_yolo_model(path):
#     try:
#         return YOLO(path)
#     except Exception as e:
#         st.error(f"Failed to load YOLO model: {e}")
#         return None

# # --- Image preprocessing ---
# def preprocess_frame(frame, use_denoise, use_clahe, use_sharpen):
#     processed = frame.copy()
#     if use_denoise:
#         processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
#     if use_clahe:
#         lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         cl = clahe.apply(l)
#         processed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
#     if use_sharpen:
#         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         processed = cv2.filter2D(processed, -1, kernel)
#     return processed

# # --- PDF report generation ---
# def create_pdf_report(orig_img, annotated_img, summary_df, coverage, severity, recommendation):
#     buffer = io.BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4)
#     styles = getSampleStyleSheet()
#     story = []

#     story.append(Paragraph("Marine Fouling Detection Report", styles['h1']))
#     story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
#     story.append(Paragraph(f"<b>Overall Fouling Coverage:</b> {coverage:.2f}%", styles['Normal']))
#     story.append(Paragraph(f"<b>Severity Level:</b> {severity}", styles['Normal']))
#     story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles['Normal']))

#     orig_buffer = io.BytesIO()
#     annotated_buffer = io.BytesIO()
#     orig_img.save(orig_buffer, format="PNG")
#     Image.fromarray(annotated_img).save(annotated_buffer, format="PNG")
#     orig_buffer.seek(0)
#     annotated_buffer.seek(0)

#     im1 = PlatypusImage(orig_buffer, width=200, height=150)
#     im2 = PlatypusImage(annotated_buffer, width=200, height=150)
#     story.append(Table([[im1, im2]], colWidths=[220, 220]))

#     story.append(Paragraph("Detection Summary by Class", styles['h2']))
#     table_data = [['Object Type', 'Count']] + summary_df.values.tolist()
#     summary_table = Table(table_data)
#     summary_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
#     story.append(summary_table)

#     doc.build(story)
#     buffer.seek(0)
#     return buffer

# # --- UI Layout ---
# col_logo, col_title = st.columns([0.1, 0.9])
# if os.path.exists(LOGO_PATH):
#     with col_logo:
#         st.image(LOGO_PATH, width=120)
# with col_title:
#     st.title("Marine-Foul-Detect")
#     st.write("An advanced tool for detecting and quantifying marine fouling on ship hulls using YOLOv8.")

# # --- Sidebar Controls ---
# st.sidebar.header("⚙️ Analysis Controls")
# uploaded_file = st.sidebar.file_uploader("Upload an underwater image", type=["jpg","jpeg","png"])
# conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
# with st.sidebar.expander("Advanced Preprocessing Options"):
#     use_denoise = st.checkbox("Denoise", value=True)
#     use_clahe = st.checkbox("Improve Contrast (CLAHE)", value=True)
#     use_sharpen = st.checkbox("Sharpen", value=True)

# # --- Main App Body ---
# if uploaded_file is None:
#     st.info("⬅️ Upload an image using the sidebar to start the analysis.")
# else:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_in:
#         tmp_in.write(uploaded_file.read())
#         tmp_in_path = tmp_in.name

#     orig_image = Image.open(tmp_in_path).convert("RGB")
#     img_np = np.array(orig_image)

#     # --- Step 1: Polygon ROI Selection ---
#     st.subheader("Step 1: Draw Polygon around the Ship")
#     st.write("👉 Use the polygon tool to select the ship region")

#     canvas_result = st_canvas(
#         fill_color="rgba(0, 0, 0, 0)",
#         stroke_width=2,
#         stroke_color="red",
#         background_image=orig_image,
#         update_streamlit=True,
#         height=orig_image.height,
#         width=orig_image.width,
#         drawing_mode="polygon",
#         key="canvas",
#     )

#     ship_roi = None
#     if canvas_result.json_data and "objects" in canvas_result.json_data:
#         objects = canvas_result.json_data["objects"]
#         if len(objects) > 0:
#             obj = objects[0]
#             if "path" in obj:
#                 pts = np.array([[p[1], p[2]] for p in obj["path"] if len(p) == 3], np.int32)
#             elif "points" in obj:
#                 pts = np.array(obj["points"], np.int32)
#             else:
#                 pts = None

#             if pts is not None:
#                 pts = pts.reshape((-1, 1, 2))
#                 mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
#                 cv2.fillPoly(mask, [pts], 255)
#                 ship_roi = cv2.bitwise_and(img_np, img_np, mask=mask)

#     if ship_roi is not None:
#         st.subheader("Step 2: Selected Ship ROI")
#         st.image(ship_roi, caption="Ship Region")

#         # --- Preprocess ROI ---
#         enhanced = preprocess_frame(ship_roi, use_denoise, use_clahe, use_sharpen)

#         # --- Step 3: Run Detection Automatically ---
#         model = load_yolo_model(MODEL_PATH)
#         if model is not None:
#             with st.spinner("Analyzing ship ROI..."):
#                 results = model(enhanced, imgsz=640)
#                 res = results[0]

#                 filtered_boxes = [box for box in res.boxes if box.conf[0] >= conf_threshold]
#                 fouling_area = 0
#                 total_area = enhanced.shape[0]*enhanced.shape[1]
#                 rows = []

#                 for box in filtered_boxes:
#                     cls_id = int(box.cls[0])
#                     conf = float(box.conf[0])
#                     name = model.names[cls_id]
#                     rows.append([name, f"{conf:.2f}"])
#                     x1_box, y1_box, x2_box, y2_box = box.xyxy[0]
#                     fouling_area += (x2_box - x1_box)*(y2_box - y1_box)

#                 details_df = pd.DataFrame(rows, columns=["Object Type","Confidence"])
#                 coverage = (fouling_area/total_area)*100 if total_area>0 else 0
#                 if coverage <10:
#                     severity = "Light"
#                     recommendation = "Monitor condition. Next cleaning in 2–3 months."
#                 elif coverage <30:
#                     severity = "Moderate"
#                     recommendation = "Cleaning recommended to prevent efficiency loss."
#                 else:
#                     severity = "Heavy"
#                     recommendation = "Immediate cleaning required."

#                 annotated = res.plot()
#                 annotated_rgb = cv2.cvtColor(annotated.astype(np.uint8), cv2.COLOR_BGR2RGB)

#                 # --- Tabs ---
#                 tab1, tab2, tab3 = st.tabs(["📊 Summary & Report", "🖼️ Image Comparison", "📋 Detailed Detections"])

#                 with tab1:
#                     st.subheader("Analysis Summary")
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric("Total Detections", len(details_df))
#                     col2.metric("Fouling Coverage", f"{coverage:.2f}%")
#                     col3.metric("Severity Level", severity)
#                     st.info(f"**Recommendation:** {recommendation}")
#                     st.markdown("---")
#                     if not details_df.empty:
#                         summary_counts = details_df['Object Type'].value_counts().reset_index()
#                         summary_counts.columns = ['Object Type','Count']
#                         st.bar_chart(summary_counts.set_index('Object Type')['Count'])
#                         pdf_buffer = create_pdf_report(orig_image, annotated_rgb, summary_counts, coverage, severity, recommendation)
#                         st.download_button("📥 Download PDF Report", pdf_buffer, file_name=f"fouling_report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
#                         csv_buffer = details_df.to_csv(index=False).encode('utf-8')
#                         st.download_button("📥 Download CSV", csv_buffer, file_name=f"detailed_detections_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
#                     else:
#                         st.success("✅ No fouling detected above the confidence threshold!")

#                 with tab2:
#                     st.subheader("Visual Comparison")
#                     image_comparison(img1=orig_image, label1="Original", img2=annotated_rgb, label2="Annotated")

#                 with tab3:
#                     st.subheader("Detailed Detection Data")
#                     st.dataframe(details_df)

#     # --- Clean temp file ---
#     if os.path.exists(tmp_in_path):
#         os.remove(tmp_in_path)
