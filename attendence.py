import cv2 as cv
import os
import face_recognition
import csv
from datetime import datetime, timedelta
from threading import Thread


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
        face_encoding = face_recognition.face_encodings(image, num_jitters=100)[0]  # Only take the first face encoding
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


# Function to process each frame from the webcam, detect faces, and recognize them
def process_frame(image_read, face_encodings, labels, attendance_file, attendance_records, scale=0.80, face_score=0.5):
    """
    Processes the current frame from the webcam, detects faces, and compares them with known face encodings.
    If a match is found, the person's attendance is recorded.

    Args:
        image_read (ndarray): The current frame captured by the webcam.
        face_encodings (list): Pre-computed face encodings of known faces.
        labels (list): Corresponding labels (names) for the face encodings.
        attendance_file (str): The CSV file where attendance records are stored.
        attendance_records (dict): A dictionary that tracks the last recorded time of each person.
        scale (float): Scale factor to resize the input image for faster processing. Default is 0.25.
        face_score (float): Threshold to consider a face match. Default is 0.6.

    Returns:
        image_read (ndarray): The processed image with rectangles and labels drawn on detected faces.
    """
    # Resize and convert the image for faster processing
    image_rgb = cv.cvtColor(image_read, cv.COLOR_BGR2RGB)
    image_rgb = cv.resize(image_rgb, (0, 0), None, scale, scale)

    try:
        # Detect face locations and encodings in the frame
        face_locations = face_recognition.face_locations(image_rgb, model="cnn")
        face_encodings_in_frame = face_recognition.face_encodings(image_rgb, face_locations)

        # Loop through each detected face
        for face_location, face_encoding in zip(face_locations, face_encodings_in_frame):
            face_match = face_recognition.compare_faces(face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_index = face_match.index(True) if True in face_match else None

            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = int(y1 / scale), int(x2 / scale), int(y2 / scale), int(x1 / scale)

            if best_match_index is not None and face_distance[best_match_index] < face_score:
                detected_person = labels[best_match_index].replace('.jpg', '')
                record_attendance(detected_person, attendance_file, attendance_records)
            else:
                detected_person = "Unknown"

            # Draw rectangles and labels on the image
            cv.rectangle(image_read, (x1, y1), (x2, y2), (50, 200, 50), 2)
            cv.rectangle(image_read, (x1, y1 - 25), (x2, y1 - 4), (50, 200, 50), cv.FILLED)
            cv.putText(image_read, detected_person, (x1, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    except IndexError as e:
        print('No Face Detected', e)

    return image_read

# Main function to start the video capture and recognition process
def main():
    """
    The main function initializes everything and starts the webcam feed for facial recognition.
    It continuously captures frames, processes them, and displays the results. Attendance is also recorded.
    """
    image_path = "images"
    attendance_file = "attendance.csv"
    attendance_records = {}  # Dictionary to track the last attendance time of each person

    # Load images and labels from the directory
    images, labels = load_images_and_labels(image_path)

    # Compute face encodings for the loaded images
    face_encodings = encode_faces(images)

    # Initialize the CSV file for attendance records
    initialize_csv(attendance_file)

    # Start the webcam
    capture = cv.VideoCapture(1)  # 0 for default webcam


    while True:
        result, image_read = capture.read()
        if result:
            flipped = cv.flip(image_read, 1)
            # Process each frame for face detection and recognition
            processed_frame = process_frame(flipped, face_encodings, labels, attendance_file, attendance_records)
            cv.imshow("Window", processed_frame)

        if cv.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    capture.release()
    cv.destroyAllWindows()


# Call the main function to start the program
if __name__ == "__main__":
    main()
