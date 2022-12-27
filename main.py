import numpy as np
import pickle
import cv2
import face_recognition
import cvzone
from pymongo import MongoClient
from config import db_faces, db_logs, FRAME_FREQUENCY
from utils import str_to_arr, arr_to_str, load_database

def run(known_encodings, known_names):
    # Set up the video capture, and define the width and height
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)


    frame_iterable = 0

    while True:
        success, img = cap.read()

        # Process every FRAME_FREQUENCY frame(s)
        if (frame_iterable % FRAME_FREQUENCY == 0):

            # Resize frame and convert to RGB
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Check if the face is a match for known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                # Use the face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                try:
                    best_match_index = np.argmin(face_distances)
                except:
                    print(f"ERROR: Length of known_encodings: {len(known_encodings)} and face_distances: {len(face_distances)}")
                # And check if the match value is True
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                face_names.append(name)

        # Update the value
        frame_iterable = (frame_iterable + 1) % FRAME_FREQUENCY

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Face Attendance", img)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_encodings, known_names = load_database()
    print(f"Loaded {len(known_encodings)} entries from Database...")
    run(known_encodings, known_names)