import face_recognition
import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image


# Load known faces
known_face_encodings = []
known_face_names = []
path = "person_images"

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image with Pillow
        pil_image = Image.open(f"{path}/{filename}")
        
        # Convert the image to RGB format (face_recognition requires RGB format)
        rgb_image = pil_image.convert("RGB")
        
        # Convert to a numpy array
        numpy_image = np.array(rgb_image)
        
        # Encode the face
        encoding = face_recognition.face_encodings(numpy_image)[0]
        known_face_encodings.append(encoding)
        
        # Extract the person's name from the image filename
        name = os.path.splitext(filename)[0]
        known_face_names.append(name)

attendance_file = 'attendance.csv'

# Check if file exists; if not, create it
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write("Name,Date,Time\n")  # CSV header


# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = video_capture.read()
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare detected faces with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the closest match
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            # Get current date and time
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Record attendance only once per person per session
            attendance_data = pd.read_csv(attendance_file)
            if not ((attendance_data['Name'] == name) & (attendance_data['Date'] == date)).any():
                with open(attendance_file, 'a') as f:
                    f.write(f"{name},{date},{time}\n")

        # Draw a rectangle around the face
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale up since we resized
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the video feed with annotations
    cv2.imshow('Attendance System', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

