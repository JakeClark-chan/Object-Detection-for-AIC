import sqlite3
import pandas as pd
from tqdm import tqdm

# Connect to SQLite database (or create it)
conn = sqlite3.connect('image_detection.db')
cursor = conn.cursor()

# Create table 'image_mapping' with two columns: 'image_path' and 'frame_idx'
cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_mapping (
        image_path TEXT PRIMARY KEY,
        frame_idx INTEGER
    )
''')

# Define the CSV file and chunksize
csv_file = "output.csv"  # Replace with your file path
chunksize = 1000  # Number of rows per chunk

# CSV reader with chunksize
csv_iterator = pd.read_csv(csv_file, chunksize=chunksize)

# Define the prefix to be removed
prefix_to_remove = "D:\\UIT\\aic\\frames\\"

# Process each chunk
for chunk in tqdm(csv_iterator):
    # Iterate through each row in the chunk
    for _, row in chunk.iterrows():
        # Remove the prefix from the image_path
        cleaned_image_path = row["Image_Path"].replace(prefix_to_remove, "")
        
        # Insert into SQLite database
        cursor.execute('''
            INSERT OR IGNORE INTO image_mapping (image_path, frame_idx)
            VALUES (?, ?)
        ''', (cleaned_image_path, row["frame_idx"]))
    
    # Commit after each chunk
    conn.commit()

# Close the connection
conn.close()