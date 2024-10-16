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


# Function to update attendance records and write to CSV
def record_attendance(detected_person, attendance_file, attendance_records):
    """
    Records the attendance of a detected person. If the person has not been recorded within the past second,
    it updates the CSV file with the person's name, date, and time of detection.

    Args:
        detected_person (str): Name of the detected person.
        attendance_file (str): The CSV file where attendance records are stored.
        attendance_records (dict): A dictionary that tracks the last recorded time of each person.
    """
    now = datetime.now()

    # Check if the person has been recorded in the last second
    if (detected_person not in attendance_records) or (now - attendance_records[detected_person] >= timedelta(seconds=1)):
        attendance_records[detected_person] = now  # Update the last recorded time

        # Format the current date and time for CSV
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Write the attendance record to the CSV file
        with open(attendance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([detected_person, date_str, time_str])


