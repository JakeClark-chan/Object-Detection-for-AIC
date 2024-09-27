import sqlite3
from constants import DB_NAME

def column_exists(conn, table_name, column_name):
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in c.fetchall()]
    return column_name in columns

# Create the SQLite database and table with separate bounding box columns
def create_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            object_label TEXT,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            confidence REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS no_detections (
            image_name TEXT PRIMARY KEY
        )
    ''')
    # Check if 'object_color' column exists and add it if it doesn't
    if not column_exists(conn, 'images', 'object_color'):
        c.execute("ALTER TABLE images ADD COLUMN object_color TEXT;")
    conn.commit()
    conn.close()

# Insert detection results into the database with separate bounding box columns
def insert_object_detection(image_name, label, bbox, confidence):
    """Insert the detection result into the database."""
    conn = sqlite3.connect('image_detection.db')
    c = conn.cursor()
    
    x_min, y_min, x_max, y_max = bbox
    c.execute("""
        INSERT INTO images (image_name, object_label, x_min, y_min, x_max, y_max, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_name, label, x_min, y_min, x_max, y_max, confidence))
    
    conn.commit()
    conn.close()

# Insert color information into the database
# Arguments: image_name, object_label, x_min, y_min, x_max, y_max, object_color
# With coordinates, accept error of 0.1
def insert_color_detection(image_name, object_label, x_min, y_min, x_max, y_max, object_color):
    error = 0.1
    conn = sqlite3.connect('image_detection.db')
    c = conn.cursor()
    c.execute("""
              UPDATE images 
              SET object_color = ? 
              WHERE image_name = ? AND object_label = ? AND x_min >= ? AND y_min >= ? AND x_max <= ? AND y_max <= ?
              """, 
              (object_color, image_name, object_label, x_min - error, y_min - error, x_max + error, y_max + error))
    conn.commit()
    conn.close()


def insert_no_detection(image_name):
    """Insert an image into the no_detections table."""
    conn = sqlite3.connect('image_detection.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO no_detections (image_name) VALUES (?)", (image_name,))
    conn.commit()
    conn.close()

def image_has_no_detection(image_name):
    """Check if an image is in the no_detections table."""
    conn = sqlite3.connect('image_detection.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM no_detections WHERE image_name = ?", (image_name,))
    no_detection = c.fetchone() is not None
    conn.close()
    return no_detection

# Get unique object labels from the database
def get_unique_labels():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT DISTINCT object_label FROM images')
    labels = [row[0] for row in c.fetchall()]
    conn.close()
    return labels

# Query images by bounding boxes that cover user-drawn points
def query_images_by_bounding_box(bounding_boxes):
    """
    Query images that have bounding boxes overlapping with any of the user-drawn boxes.
    
    Args:
        bounding_boxes (list of dicts): List of dictionaries, each containing 'x_min', 'y_min', 'x_max', 'y_max'.
    
    Returns:
        list: List of image names that match the bounding box query.
    """
    conn = sqlite3.connect("image_detection.db")
    c = conn.cursor()

    # Prepare query for multiple bounding boxes
    conditions = []
    params = []
    
    for box in bounding_boxes:
        THRESHOLD = 50  # Threshold for bounding box overlap
        x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        condition = f"""
        (x_min - {THRESHOLD} <= ? AND x_max + {THRESHOLD} >= ? AND y_min - {THRESHOLD} <= ? AND y_max + {THRESHOLD} >= ?)
        """
        conditions.append(condition)
        params.extend([x_max, x_min, y_max, y_min])

    # Combine all conditions with OR (since any bounding box match is enough)
    query = f"""
    SELECT DISTINCT image_name 
    FROM images 
    WHERE {' AND '.join(conditions)}
    LIMIT 100;
    """
    
    c.execute(query, params)
    results = c.fetchall()
    conn.close()

    return [result[0] for result in results]

# Query images based on object filters
def query_images_by_label(object_labels):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    query = f"SELECT image_name FROM images WHERE object_label IN ({','.join(['?']*len(object_labels))}) GROUP BY image_name LIMIT 200"
    c.execute(query, object_labels)
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results
