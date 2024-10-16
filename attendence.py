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


# Function to compute face encodings for each image
def encode_faces(images):
    """
    Given a list of images, this function computes the face encodings for each image.
    The encodings are unique representations of faces and are used for comparison during recognition.

    Args:
        images (list): List of images for which face encodings will be generated.

    Returns:
        face_encodings (list): List of face encodings corresponding to each image.
    """
    face_encodings = []
    for image in images:
        face_encoding = face_recognition.face_encodings(image)[0]  # Only take the first face encoding
        face_encodings.append(face_encoding)
    
    return face_encodings


# Function to initialize the attendance CSV file with headers
def initialize_csv(attendance_file):
    """
    Creates a CSV file with headers ("Name", "Date", "Time") if it does not already exist.

    Args:
        attendance_file (str): The filename of the CSV file to create or open.
    """
    if not os.path.isfile(attendance_file):
        with open(attendance_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])  # CSV Headers

