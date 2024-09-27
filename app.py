import streamlit as st
from streamlit_drawable_canvas import st_canvas
from detector import load_yolo_model, get_images_in_folder, detect_objects_batch
from database import create_database, insert_object_detection, get_unique_labels, query_images_by_label, query_images_by_bounding_box, insert_no_detection, image_has_no_detection, insert_color_detection
from filter_model import load_bert, extract_objects_from_text_bert, load_clip, extract_objects_from_text_clip
from color_detection import colors, get_dominant_color
from constants import ROOT_FOLDER
from tqdm import tqdm
import contextlib
import sys
import math
import cv2

import os
from PIL import Image
import sqlite3

# Load models
yolo_model = load_yolo_model()
# bert_model = load_bert()
# clip_model = load_clip()

# Prepare database
create_database()

# Check if image already exists in the database
def image_exists_in_db(image_name):
    conn = sqlite3.connect("image_detection.db")
    c = conn.cursor()
    c.execute('SELECT 1 FROM images WHERE image_name = ?', (image_name,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


# Function to suppress YOLO output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Object Filter", "Drawable Canvas"])

# Settings: Batch size
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=20, step=1)

# Page 1: Object Filter
if page == "Object Filter":
    st.title("Object Detection and Image Search")
    
    # Overwrite option
    overwrite_option = st.radio("If an image is already in the database, do you want to overwrite the results?", ('Overwrite', 'Ignore'))
    
    if st.button("Run Object Detection on All Images"):
        images = get_images_in_folder() # Get all images in the root folder
        total_images = len(images)  # Total number of images
        num_batches = math.ceil(total_images / batch_size)  # Number of batches to process

        progress_bar = st.progress(0)  # Streamlit progress bar
        # pbar = tqdm(total=num_batches, desc="Processing Images", unit="batches")  # tqdm progress bar

        # Process images in batches
        for batch_num in tqdm(range(num_batches), desc="Processing Images", unit="batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_images)
            image_batch = images[start_idx:end_idx]

            # Join image paths with root folder
            image_batch = [os.path.join(ROOT_FOLDER, img) for img in image_batch]

            # Filter out images already in database or no_detections if "Ignore" is selected
            if overwrite_option == 'Ignore':
                image_batch = [
                    img for img in image_batch 
                    if not image_exists_in_db(img) and not image_has_no_detection(img)
                ]

            if image_batch:
                detections = detect_objects_batch(yolo_model, image_batch)

                # Insert detection results into the database
                for img_path, detection_list in detections.items():
                    if detection_list:
                        for label, bbox, conf in detection_list:
                            insert_object_detection(img_path, label, bbox, conf)
                    else:
                        # No objects detected, insert into no_detections table
                        insert_no_detection(img_path)


                # st.write(f"Processed batch {batch_num + 1}/{num_batches}.")

            # Update progress bars
            progress_bar.progress((batch_num + 1) / num_batches)
            # pbar.update((batch_num + 1))

        # pbar.close()
        st.write("Object detection complete!")

    if st.button("Run Object Color Detection"):
        # Run color detection on all objects in the database
        conn = sqlite3.connect("image_detection.db")
        c = conn.cursor()
        if overwrite_option == "Ignore":
            c.execute("SELECT image_name, object_label, x_min, y_min, x_max, y_max FROM images WHERE object_color IS NULL")
        else:
            c.execute("SELECT image_name, object_label, x_min, y_min, x_max, y_max FROM images")
        results = c.fetchall()
        conn.close()

        for image_name, object_label, x_min, y_min, x_max, y_max in tqdm(results, desc="Detecting Colors", unit="objects"):
            img_path = os.path.join(ROOT_FOLDER, image_name)
            img = cv2.imread(img_path)
            # Convert x and y to int
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            # Get region
            object_region = img[y_min:y_max, x_min:x_max]
            # Get dominant color
            object_color = get_dominant_color(object_region)
            # Insert color detection into the database
            insert_color_detection(image_name, object_label, x_min, y_min, x_max, y_max, object_color)

    
    # Search images using object labels
    object_labels = get_unique_labels()
    selected_labels = st.multiselect("Select Object Labels", object_labels)

    colors = list(colors.keys())
    selected_colors = st.multiselect("Filter by Object Color", colors)

    

    if st.button("Search Images"):
        matching_images = query_images_by_label(selected_labels)
        
        # Modify query to filter by color
        conn = sqlite3.connect("image_detection.db")
        c = conn.cursor()
        query_conditions = []
        query_values = []

        # Add selected labels and colors to the query
        if selected_labels:
            query_conditions.append("object_label IN ({})".format(','.join(['?'] * len(selected_labels))))
            query_values.extend(selected_labels)

        if selected_colors:
            query_conditions.append("object_color IN ({})".format(','.join(['?'] * len(selected_colors))))
            query_values.extend(selected_colors)

        # Construct the query based on filters
        query = f"SELECT image_name, MAX(confidence) as max_conf, frame_idx FROM images JOIN image_mapping ON images.image_name = image_mapping.image_path"
        if query_conditions:
            query += " WHERE " + " AND ".join(query_conditions)
        query += " GROUP BY image_name HAVING max_conf>0.75 ORDER BY max_conf DESC"

        c.execute(query, query_values)
        matching_images = c.fetchall()
        conn.close()


        # Display images in 4 columns
        cols = st.columns(4)
        for i, (image, confidence, frame_idx) in enumerate(matching_images):
            img_path = os.path.join(ROOT_FOLDER, image)
            with cols[i % 4]:
                st.image(Image.open(img_path), caption=f"{image} - Frame {frame_idx} - Confidence: {confidence:.2f}", use_column_width=True)

# Page 2: Drawable Canvas
elif page == "Drawable Canvas":
    st.title("Search by Drawing Bounding Boxes")

    # Drawable canvas for user to draw bounding boxes
    st.subheader("Draw Bounding Boxes to Search")
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color='#FF0000',
        background_color='#FFFFFF',
        height=400,
        width=400,
        drawing_mode="rect",  # Rectangular mode for bounding boxes
        key="canvas"
    )

    # If there are drawn objects, capture the bounding boxes
    if canvas_result.json_data is not None:
        drawn_boxes = canvas_result.json_data["objects"]
        bounding_boxes = []

        for box in drawn_boxes:
            if box['type'] == 'rect':  # Ensure it's a rectangle
                x_min = box['left']
                y_min = box['top']
                x_max = x_min + box['width']
                y_max = y_min + box['height']
                bounding_boxes.append({
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })

        # If bounding boxes exist, query the database
        if bounding_boxes:
            st.write(f"Detected {len(bounding_boxes)} bounding boxes. Searching images...")
            
            # Query the database for matching images
            matching_images = query_images_by_bounding_box(bounding_boxes)
            
            if matching_images:
                st.write("Matching images:")
                cols = st.columns(4)
                for i, image_name in enumerate(matching_images):
                    img_path = os.path.join(ROOT_FOLDER, image_name)
                    with cols[i % 4]:
                        st.image(Image.open(img_path), caption=image_name, use_column_width=True)
            else:
                st.write("No matching images found.")
        else:
            st.write("No bounding boxes detected.")
