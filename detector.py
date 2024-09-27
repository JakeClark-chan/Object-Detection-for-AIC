import os
from ultralytics import YOLO
import torch
import cv2
from constants import ROOT_FOLDER

# Initialize YOLOv8 Model
def load_yolo_model():
    return YOLO('yolov10x.pt')

# Recursively find images in the root folder
def get_images_in_folder():
    images = []
    for root, _, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.relpath(os.path.join(root, file), ROOT_FOLDER))
    return images

# Run YOLO detection on multiple images
def detect_objects(model, image_path):
    results = model(image_path)
    detected_objects = []
    for result in results:
        for obj in result.boxes:
            label = result.names[obj.cls.item()]  # Get object label
            conf = obj.conf.item()  # Confidence score
            bbox = obj.xyxy[0].tolist()  # Bounding box as [x_min, y_min, x_max, y_max]
            detected_objects.append((label, bbox, conf))
    return detected_objects

def detect_objects_batch(yolo_model, image_paths):
    """
    Run YOLO object detection on a batch of images using the Ultralytics YOLO model.
    
    Args:
        yolo_model: Loaded YOLO model from ultralytics.
        image_paths: List of file paths of images to process in the batch.
    
    Returns:
        dict: Dictionary with image paths as keys and detection results (labels, bounding boxes, confidences, colors) as values.
    """
    results = yolo_model(image_paths)
    detections = {}

    for result, img_path in zip(results, image_paths):
        detections_list = []
        img = cv2.imread(img_path)  # Load image with OpenCV
        
        for box in result.boxes:
            label = result.names[int(box.cls)]
            bbox = box.xyxy.tolist()[0]
            confidence = box.conf.item()
            
            # Extract object region from the image using the bounding box
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Round coordinates to 4 decimal places
            x_min, y_min, x_max, y_max = round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)
            
            detections_list.append((label, bbox, confidence))
        
        detections[img_path] = detections_list

    return detections
