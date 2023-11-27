# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import easyocr  
reader = easyocr.Reader(['en'], gpu=False) 
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import sqlite3
import sys


# SETTINGS #


# Get the absolute path of the current file (only works in .py files) - path to this file ./settings.py
file_path = Path(__file__).resolve()

# Get the parent directory of the current file (main file: /yolov8-streamlit)
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there : allows for things like helper.process_license_plate()
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the main folder (basically IMAGES_DIR = ../yolov8-streamlit/'images')
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / '1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / '2.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8s.pt'

# HELPER #

def load_model(model_path):

    model = YOLO(model_path)
    return model


# Connection to SQL Database
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Create a table to store detection results
cursor.execute('''CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY,
                num_detected_objects INTEGER,
                class_name TEXT,
                count INTEGER)''')

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar header
st.sidebar.header("ML Model Config")

model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)

    try:
        model = load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image Config")
    source_radio = st.sidebar.radio(
        "Select Source", SOURCES_LIST)

    source_img = None

    if source_radio == IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                             use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                             use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                         use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image , max_det = 2000 , conf=confidence)
                    boxes = res[0].boxes
                    num_boxes = len(boxes)  # Total count of detected boxes

                    if num_boxes == 0:
                        st.error("No objects detected in the image.")
                    else:
                        st.write(f"Number of detected objects: {num_boxes}")  # Display the total count

                        # Create a dictionary to count each class separately
                        class_counts = {}
                        class_name_map = {
                            0: 'C 32 2.5' ,
                            1: 'C 38 2.9' ,
                            2: 'C 48 2.9' ,
                            3: 'R 20 40 1.9' ,
                            4: 'R 25 75 1.9' ,
                            5: 'R 48 96 2.0' ,
                            6: 'R 48 96 2.9' ,
                            7: 'R 60 40 1.9' ,
                            8: 'R 80 40 1.2' ,
                            9: 'R 96 48 2.0' ,
                            10: 'R 96 48 2.9' ,
                            11: 'S 20 20 1.2' ,
                            12: 'S 20 20 1.5' ,
                            13: 'S 20 20 1.9' ,
                            14: 'S 25 25 1.9' ,
                            15: 'S 25 25 2.5' ,
                            16: 'S 38 38 1.9' ,
                            17: 'S 40 40 2.5' ,
                            18: 'S 40 40 3.2' ,
                            19: 'S 50 50 1.5' ,
                            20: 'S 50 50 1.9' ,
                            21: 'S 50 50 4.0' ,
                            22: 'S 60 60 2.0' ,
                            23: 'S 72 72 4.0' ,
                            24: 'S 72 72 4.8' ,
                            25: 'S 80 40 1.2' ,
                        }
                        for box in boxes:
                            # Access the last element of the tensor to get the class label
                            class_label = int(box.cls)  # Convert to integer
                            class_name = class_name_map.get(class_label , "Unknown")
                            if class_name not in class_counts:
                                class_counts[class_name] = 1
                            else:
                                class_counts[class_name] += 1

                            # Store the number of detected objects and class counts in the database
                            cursor.execute("INSERT INTO detection_results (num_detected_objects, class_name, count) VALUES (?, ?, ?)",
                                        (num_boxes, class_name, class_counts[class_name]))
                            conn.commit()

                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image', use_column_width=True)

                        st.write("Count of each class:")
                        for class_name, count in class_counts.items():
                            st.write(f"{class_name}: {count}")

                        try:
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)
                        except Exception as ex:
                            st.write("No image is uploaded yet!")

    else:
        st.error("Please select a valid source type!")