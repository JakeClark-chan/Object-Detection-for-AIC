import pandas as pd
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import ast

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host="localhost",
    port=6333,
)

# # Delete the collection if it already exists and recreate it
# if qdrant_client.collection_exists("image_embeddings"):
#     qdrant_client.delete_collection("image_embeddings")

# Create a collection for image embeddings (if not already created)
if not qdrant_client.collection_exists("image_embeddings"):
    qdrant_client.create_collection(
        collection_name="image_embeddings",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE, on_disk = True),
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        # quantization_config=models.ScalarQuantization(
        #     scalar=models.ScalarQuantizationConfig(
        #         type=models.ScalarType.INT8,
        #         always_ram=True,
        #     ),
        # ),
    )

# Load large CSV in chunks
csv_file = "output.csv"  # Replace with your file path
chunksize = 70  # Number of rows per chunk

# Create a CSV reader with chunksize
csv_iterator = pd.read_csv(csv_file, chunksize=chunksize)

import hashlib

# Function to generate a hash based on row content
def generate_hash(row):
    # Combine relevant fields to create a unique identifier
    hash_input = f"{row['image_feature']}_{row['Name_Vid']}_{row['frame_idx']}"
    
    # Generate a SHA-256 hash
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

# Process each chunk and upload to Qdrant
for chunk in tqdm(csv_iterator):
    points = []

    # Process each row in the chunk
    for id, row in chunk.iterrows():
        # Convert string to numpy array
        row["image_feature"] = ast.literal_eval(row["image_feature"])

        point = {
            "id": id,  # Use unique identifier for each point
            "vector": row["image_feature"],  # Convert numpy array to list
            "payload": {
                "n": row["n"],
                "frame_idx": row["frame_idx"],
                "Name_Vid": row["Name_Vid"],
                "Image_Path": row["Image_Path"],
            },
        }
        points.append(point)

    # Upload the batch to Qdrant
    qdrant_client.upsert(
        collection_name="image_embeddings",
        points=points
    )

# ------------------------------

# Step 4: Use CLIP ViT-L/14 to embed text query and search for similar images
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy
import matplotlib
import matplotlib.pyplot as plt