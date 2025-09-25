import streamlit as st
import cv2
import io
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as PlatypusImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

@st.cache_resource
def load_yolo_model(path):
    try: return YOLO(path)
    except Exception as e: st.error(f"Failed to load YOLO model: {e}"); return None

def preprocess_frame(frame):
    return frame.astype(np.uint8)

def create_detection_heatmap(image_np, boxes):
    heatmap = np.zeros_like(image_np[:, :, 0], dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), 50, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    overlayed_image = cv2.addWeighted(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), 0.6, heatmap_colored, 0.4, 0)
    return overlayed_image

def create_pdf_report(orig_img, annotated_img, summary_df, coverage, severity, recommendation, 
                      heatmap_img, graph_bytes, real_fouling_area_m2, real_roi_area_m2):
    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        styles, story = getSampleStyleSheet(), []

        story.append(Paragraph("Marine Fouling Detection Report", styles['h1']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Severity Analysis", styles['h2']))
        story.append(Paragraph(f"<b>Fouling Coverage:</b> {coverage:.2f}%", styles['Normal']))
        story.append(Paragraph(f"<b>Fouling Area Density:</b> {real_fouling_area_m2:.3f} m² (out of {real_roi_area_m2:.2f} m² total ROI area)", styles['Normal']))
        story.append(Paragraph(f"<b>Severity Level:</b> {severity}", styles['Normal']))
        story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Visual Analysis", styles['h2']))
        annotated_buffer = io.BytesIO(); Image.fromarray(annotated_img).save(annotated_buffer, format="PNG"); annotated_buffer.seek(0)
        heatmap_buffer = io.BytesIO(); Image.fromarray(heatmap_img).save(heatmap_buffer, format="PNG"); heatmap_buffer.seek(0)
        im1 = PlatypusImage(annotated_buffer, width=3*inch, height=2.25*inch, kind='proportional')
        im2 = PlatypusImage(heatmap_buffer, width=3*inch, height=2.25*inch, kind='proportional')
        story.append(Table([[im1, im2]], colWidths=[3.2*inch, 3.2*inch]))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Species Distribution Graph", styles['h2']))
        graph_buffer = io.BytesIO(graph_bytes)
        graph_img = PlatypusImage(graph_buffer, width=6*inch, height=3*inch, kind='proportional')
        story.append(graph_img)
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Detection Summary Table", styles['h2']))
        if not summary_df.empty:
            table_data = [['Object Type', 'Count']] + summary_df.values.tolist()
            summary_table = Table(table_data, colWidths=[4*inch, 2*inch])
            summary_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.beige),('GRID', (0, 0), (-1, -1), 1, colors.black)]))
            story.append(summary_table)

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}"); return None