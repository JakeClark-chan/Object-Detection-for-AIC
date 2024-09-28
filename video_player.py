import cv2
import streamlit as st
import numpy as np
import tempfile
import os
import re
import csv
import math
from constants import VIDEO_ROOT_FOLDER
import subprocess

# Function to load video and seek to a specific frame
def load_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

# Function to convert a frame (OpenCV image) to bytes for display
def frame_to_bytes(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Function to update session state
def update_session_state(key, value = None):
    if key == "frame_number":
        return st.session_state[key]
    elif key == "button":
        st.session_state['frame_number'] = value
    else:
        st.session_state['frame_number'] = st.session_state[key]

# Function to update button state
def update_button_state(cmd, total_frames, value = None):
    if cmd == "left_1":
        if st.session_state['frame_number'] > 0:
            update_session_state("button", st.session_state['frame_number'] - 1)
    elif cmd == "right_1":
        if st.session_state['frame_number'] < total_frames - 1:
            update_session_state("button", st.session_state['frame_number'] + 1)
    elif cmd == "left_10":
        if st.session_state['frame_number'] >= 10:
            update_session_state("button", st.session_state['frame_number'] - 10)
    elif cmd == "right_10":
        if st.session_state['frame_number'] <= total_frames - 11:
            update_session_state("button", st.session_state['frame_number'] + 10)
    else:
        return st.session_state['frame_number']

# Streamlit app structure
def main():
    st.title("Video Frame Seeker with Responsive Slider")

    # Input video path
    # video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    video_file = st.text_input("Enter the video name, with .mp4", 
                               "L01_V001.mp4" if "video_file" not in st.session_state 
                               else st.session_state["video_file"]
                               )
    
    if video_file is not None:
        # Construct the video path
        # Use re to capture L??_V???.mp4
        video_list = re.search(r"\bL\d+", video_file)
        video_path = os.path.join(VIDEO_ROOT_FOLDER, f"Videos_{video_list.group()}_a/{video_file}")
        st.write(f"Searching in path {video_path}")

        # Add more button to execute "djv video_path" command
        def execute_command(video_path):
            subprocess.run(["djv", video_path])
        st.button("Open video in DJV", on_click=execute_command, args=(video_path,))
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Initialize session state for frame number if not already set
        if 'frame_number' not in st.session_state:
            st.session_state['frame_number'] = 0
        if "number_input" not in st.session_state:
            st.session_state["number_input"] = 0
        if "slider" not in st.session_state:
            st.session_state["slider"] = 0

        # Input frame number
        new_frame = st.number_input(
            "Go to frame:", 
            min_value=0, 
            max_value=total_frames-1, 
            value=st.session_state['frame_number'], 
            step=1, 
            key="number_input",
            on_change=update_session_state,
            args=("number_input",)
        )

        # Responsive slider for seeking frames (step size of 1 frame)
        new_frame = st.slider(
            "Seek video frame",
            min_value=0,
            max_value=total_frames-1,
            value=st.session_state['frame_number'],
            step=1,
            key="slider",
            on_change=update_session_state,
            args=("slider",)
        )

        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.button("⇐ Left (10 Frames)", on_click=update_button_state, args=("left_10", total_frames))
        
        with col2:
            st.button("← Left (1 Frame)", on_click=update_button_state, args=("left_1", total_frames))
        
        with col3:
            st.button("→ Right (1 Frame)", on_click=update_button_state, args=("right_1", total_frames))
        
        with col4:
            st.button("⇒ Right (10 Frames)", on_click=update_button_state, args=("right_10", total_frames))

        # Load and display the current frame
        frame = load_video(video_path, st.session_state['frame_number'])
        if frame is not None:
            st.image(frame_to_bytes(frame), channels="BGR", caption=f"Current frame: {st.session_state['frame_number']}")

        """
        Output CSV with 100 frames from start to end
        """
        # Output CSV with 100 frames from start to end
        # Input frame number start
        start_frame = st.number_input(
            "Start frame for CSV:", 
            min_value=0, 
            max_value=total_frames-1, 
            value=st.session_state['frame_number'] - 100, 
            step=1
        )
        # Input frame number end
        end_frame = st.number_input(
            "End frame for CSV:", 
            min_value=0, 
            max_value=total_frames-1, 
            value=st.session_state['frame_number'] + 100, 
            step=1
        )
        # Input additional data that will append at end of each line
        additional_data = st.text_input("Enter additional data to append at end of each line", "Nope")
        # Input CSV file name
        csv_file = st.text_input("Enter the CSV file name", "query_output.csv")
        # Logic to generate CSV content
        if start_frame <= end_frame:
            # Generate CSV content with "L??_V???, range(start_frame, end_frame+1, (end_frame-start_frame)//100)"
            csv_content = []    
            for i in range(start_frame, end_frame, math.ceil((end_frame-start_frame)/100)):
                video_name = re.search(r"\bL\d+_V\d+", video_file)
                csv_content.append(f"{video_name.group()},{i}{',' + additional_data if additional_data != 'Nope' else ''}")
            csv_content.append(f"{video_name.group()},{end_frame}")
            csv_content = "\n".join(csv_content)
        else:
            st.error("Start frame must be less than or equal to end frame")
        # Download button for CSV
        st.download_button("Download CSV", data=csv_content, file_name=csv_file, mime="text/csv")
        st.write("Note: Ensure that you press Enter after inputting above values to generate CSV content")
            

if __name__ == "__main__":
    main()