import cv2 as cv
import os
import face_recognition
import csv
from datetime import datetime, timedelta


# Function to load images and labels from a specified directory
def load_images_and_labels(image_path):
    """
    Loads all images from the specified directory, converts them to RGB, and stores them in a list.
    It also creates a list of labels based on the image filenames.

    Args:
        image_path (str): The directory path where the images are stored.

    Returns:
        images (list): A list of images loaded and converted to RGB.
        labels (list): A list of labels (file names) corresponding to the loaded images.
    """
    images = []
    labels = []

    # Loop through all files in the directory
    for file in os.listdir(image_path):
        print(f"Loading: {file}")
        full_path = os.path.join(image_path, file)
        
        # Load and convert the image to RGB format
        image = cv.imread(full_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        images.append(image)
        labels.append(file)

    return images, labels


