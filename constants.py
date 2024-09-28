from qdrant_client import QdrantClient

ROOT_FOLDER = '/media/jc/Home/extracted_frames/'  # Constant root folder for images

DB_NAME = "image_detection.db"  # Database name

VIDEO_ROOT_FOLDER = "/media/jc/CRIT Data/archives/"  # Constant root folder for videos

QDRANT_CLIENT = QdrantClient(
        host="localhost",
        port=6333,
    )