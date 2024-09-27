import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
}

def get_dominant_color(image, k=3):
    """
    Detect the dominant color in an image using K-Means clustering.
    
    Args:
        image (numpy array): The image to analyze (portion inside bounding box).
        k (int): The number of clusters for K-Means. Defaults to 3.
    
    Returns:
        str: The name of the dominant color.
    """
    # Resize the image to reduce processing time (optional)
    # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Reshape the image into a 2D array of pixels and 3 color values (R, G, B)
    pixels = image.reshape((-1, 3))
    
    # Remove black pixels (optional, to avoid shadows being dominant)
    pixels = pixels[np.sum(pixels, axis=1) != 0]
    
    # Apply K-Means clustering to find the dominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the most common cluster center
    counts = Counter(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    
    # Convert dominant color to RGB
    dominant_color = dominant_color.astype(int)
    
    # Map the RGB values to a color name
    return get_color_name(dominant_color)

def get_color_name(rgb):
    """
    Map the RGB values to a predefined color name.
    
    Args:
        rgb (array): Array of 3 values representing the RGB color.
    
    Returns:
        str: The name of the closest color.
    """
    
    # Compute the Euclidean distance between the RGB value and known colors
    closest_color = min(colors, key=lambda color: np.linalg.norm(np.array(colors[color]) - rgb))
    
    return closest_color
