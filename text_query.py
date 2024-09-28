from qdrant_client import QdrantClient
import streamlit as st
import clip
import torch
import numpy as np
import pandas as pd
import os
import re
from PIL import Image, ImageDraw, ImageFont
# from googletrans import Translator
from translate import Translator
from constants import QDRANT_CLIENT


collection_name = 'image_embeddings'

if 'text_record' not in st.session_state:
    st.session_state.text_record = None
if 'show_nearest_frames' not in st.session_state:
    st.session_state.show_nearest_frames = False
if 'result_images' not in st.session_state:
    st.session_state.result_images = None
if 'qdrant_client' not in st.session_state:
    # st.session_state.qdrant_client = QdrantClient(
    #     # url="https://0ce8a723-d55a-4a50-a83a-5f89ed6e2514.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    #     # api_key="YmNJDym2vZ2pqRb8WwVJ0EbBUG5bLF_zZ-zUlTdLM8QqRMw1EUvOqQ",
    #     # timeout=60

    #     url="https://e1739bbf-81ec-43df-9757-352070ca5845.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    #     api_key="sXOzR8G8PS3R9O2TO1mfQhwA2Iz3OX-VFr_RSHHiecUU0e0XK7oYjw",
    #     timeout = 60
    # )
    st.session_state.qdrant_client = QDRANT_CLIENT

if 'clip_model' not in st.session_state or 'clip_preprocess' not in st.session_state or 'device' not in st.session_state:
    st.session_state.device ="cpu"
    st.session_state.clip_model, st.session_state.clip_preprocess = clip.load("ViT-L/14", device=st.session_state.device)

def set_text_record(new_record):
    translator = Translator(from_lang='vi', to_lang='en')
    translated = translator.translate(new_record)
    return translated

def get_image_from_text(text):
    client = st.session_state.qdrant_client
    device = st.session_state.device
    model = st.session_state.clip_model

    text = set_text_record(text)
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = np.squeeze(text_features)

    return client.search(
        collection_name=collection_name,
        query_vector=text_features,
        limit=100
    )
    
    return records

def display_image_with_text(image_path, name_vid, frame_idx):
    # New: Replace image_path "C:\Users\pc\Desktop\HCM AIC\CLIP\keyframe" with "keyframe" - inside Home partition
    image_path = image_path.replace("D:\\UIT\\aic\\frames", "/media/jc/Home/extracted_frames")
    image_path = image_path.replace("\\", "/")
    if os.path.exists(image_path):
        # Load the image and prepare text
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Define font size and load font
        font_size = max(15, image.width // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()  # Use default font if custom font is not available

        # Define text and its bounding box
        text = f"Video: {name_vid}, Frame: {frame_idx}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = (image.width // 2 - text_width // 2, image.height - text_height - 10)

        # Create an image for text
        text_image = Image.new("RGB", (image.width, text_height + 30), (0, 0, 0))
        draw_text = ImageDraw.Draw(text_image)
        draw_text.text((image.width // 2 - text_width // 2, 10), text, fill=(255, 255, 255), font=font)

        # Combine the original image with text image
        combined = Image.new("RGB", (image.width, image.height + text_image.height))
        combined.paste(image, (0, 0))
        combined.paste(text_image, (0, image.height))

        st.image(combined, use_column_width=True)
    else:
        st.write(f"Không tìm thấy ảnh: {image_path}")

def get_n_nearest_frame(record_id, n=10):
    record_id = int(record_id)
    start = record_id - n if record_id - n > 0 else 0
    end = record_id + n if record_id + n < 551794 else 551794

    client = st.session_state.qdrant_client

    return client.retrieve(
        collection_name=collection_name,
        ids=range(start, end),
    )

def custom_page():
    st.title("Text Query")
    text_input = st.text_input('Input query', st.session_state.text_record if st.session_state.text_record else "")
    csv_selector = st.empty()
    if text_input:
        # Save the text record
        st.session_state.text_record = text_input
        # Get the image records
        records = get_image_from_text(text_input)
        answer_images = []

        col = st.columns(3)
        for idx, record in enumerate(records):
            image_path = record.payload['Image_Path']
            name_vid = record.payload['Name_Vid']
            frame_idx = record.payload['frame_idx']
            
            answer_images.append([name_vid, frame_idx])
            with col[idx % 3]:
                display_image_with_text(image_path, name_vid, frame_idx)
                def send_button_callback(image, frame_idx):
                    # Just take video name from image name
                    video_list = re.search(r"\bL\d+_V\d+", image)
                    video_list = video_list.group()
                    st.session_state['video_file'] = video_list + ".mp4"
                    st.session_state['frame_number'] = frame_idx
                    st.write(f"Sent to Video Frame {video_list} at frame {frame_idx}")
                st.button("Send to Video Frame", on_click=send_button_callback, args=(image_path, frame_idx), key=f"button_{idx}")
        # If there are images to display, show the CSV input
        if answer_images:
            with csv_selector.container():
                # Input additional data that will append at end of each line
                additional_data = st.text_input("Enter additional data to append at end of each line, keep Nope for no additional data", "Nope")
                # Input CSV file name
                csv_file = st.text_input("Enter the CSV file name", "query_output.csv")
                # Logic to generate CSV content
                if not csv_file:
                    csv_file = "query_output.csv"
                if additional_data != "Nope":
                    answer_images = [row + [additional_data] for row in answer_images]
                    df = pd.DataFrame(answer_images, columns=["Name_Vid", "Frame_Idx", "Additional_Info"])
                else:
                    df = pd.DataFrame(answer_images, columns=["Name_Vid", "Frame_Idx"])
                
                # Chuyển đổi DataFrame thành định dạng CSV
                csv = df.to_csv(index=False, header=False)

                # Show CSV content
                toggle = st.toggle("Show CSV content", False)
                if toggle:
                    # Show code box with CSV content
                    st.code(csv, language="csv")
                # Cho phép người dùng tải xuống file CSV
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=csv_file if csv_file.endswith('.csv') else f"{csv_file}.csv",
                    mime="text/csv"
                )
                

def main():
    st.title("SEARCH")
    text_input = st.text_input('Nhập câu truy vấn')

    button_placeholder = st.empty()


    if text_input:
        if not st.session_state.show_nearest_frames:
            records = get_image_from_text(text_input)
            answer_images = []

            col = st.columns(3)
            for idx, record in enumerate(records):
                image_path = record.payload['Image_Path']
                name_vid = record.payload['Name_Vid']
                frame_idx = record.payload['frame_idx']
                
                answer_images.append([name_vid, frame_idx])
                with col[idx % 3]:
                    display_image_with_text(image_path, name_vid, frame_idx)
                    if st.button(label='Find nearest frame', key=idx):
                        st.session_state.selected_id = record.id
                        st.session_state.show_nearest_frames = True
                        st.rerun()

            

        if st.session_state.show_nearest_frames:
            selected_id = st.session_state.selected_id
            records = get_n_nearest_frame(selected_id)

            st.title(f"Nearest frames to frame:")
            client = st.session_state.qdrant_client
            selected_frame = client.retrieve(
                collection_name=collection_name,
                ids=[selected_id]
            )[0]
            display_image_with_text(selected_frame.payload['Image_Path'], selected_frame.payload['Name_Vid'], selected_frame.payload['frame_idx'])
            if st.button(label='Quay lại'):
                st.session_state.show_nearest_frames = False
                st.rerun()
            st.divider()

            col= st.columns(3)
            answer_images = []
            for idx, record in enumerate(records):
                image_path = record.payload['Image_Path']
                name_vid = record.payload['Name_Vid']
                frame_idx = record.payload['frame_idx']

                # Append to answer_images
                answer_images.append([name_vid, frame_idx])

                with col[idx % 3]:
                    display_image_with_text(image_path, name_vid, frame_idx)

        if answer_images:
            with button_placeholder.container(): 
                combined_input = st.text_input('Nhập thông tin bổ sung (tùy chọn) và tên file CSV (cách nhau bởi dấu phẩy)', value='')
                    # Kiểm tra nếu người dùng đã nhập gì đó
                if combined_input:
                    # Tách thông tin bổ sung và tên file CSV
                    parts = combined_input.split(',', 1)  # Tách chuỗi dựa trên dấu phẩy

                    if len(parts) == 2:
                        # Trường hợp có cả thông tin bổ sung và tên file CSV
                        additional_info = parts[0].strip()  # Thông tin bổ sung
                        csv_filename = parts[1].strip()  # Tên file CSV
                    else:
                        # Trường hợp chỉ có tên file CSV, không có thông tin bổ sung
                        additional_info = ''
                        csv_filename = parts[0].strip()  # Tên file CSV

                    # Nếu không có tên file CSV, dùng mặc định
                    if not csv_filename:
                        csv_filename = 'image_search_results.csv'

                    # Cập nhật các hàng với thông tin bổ sung nếu có
                    if additional_info:
                        answer_images = [row + [additional_info] for row in answer_images]
                        df = pd.DataFrame(answer_images, columns=["Name_Vid", "Frame_Idx", "Additional_Info"])
                    else:
                        df = pd.DataFrame(answer_images, columns=["Name_Vid", "Frame_Idx"])

                    # Chuyển đổi DataFrame thành định dạng CSV
                    csv = df.to_csv(index=False, header=False)

                    # Cho phép người dùng tải xuống file CSV
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=csv_filename if csv_filename.endswith('.csv') else f"{csv_filename}.csv",
                        mime="text/csv"
                    )

                else:
                    st.error("Hãy nhập ít nhất tên file CSV.")

if __name__ == "__main__":
    main()