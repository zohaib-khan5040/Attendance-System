import numpy as np
import pickle
import cv2
import face_recognition
import cvzone

# Set up the video capture, and define the width and height
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load encodings
print("Loading encodings...")
with open("encodings.pickle", "rb") as f:
    encodings, names = pickle.load(f)
print("Loaded!")



while True:
    success, img = cap.read()

    # Take a scaled down version of the image to find the encodings
    small_img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    # Find the encodings in the small image
    face_locations = face_recognition.face_locations(small_img)
    face_encodings = face_recognition.face_encodings(small_img, face_locations)


    # Loop through encodings and compare with the ones we have
    for face_encoding, face_location in zip(face_encodings, face_locations):
        
        # Get the matches and the distances
        matches = face_recognition.compare_faces(encodings, face_encoding)
        face_distances = face_recognition.face_distance(encodings, face_encoding)

        # Find the index of the smallest distance
        best_match_idx = np.argmin(face_distances)

        # Check if the corresponding match value is True (meaning it's a match)
        if matches[best_match_idx]:
            name = names[best_match_idx]
            print(f"Match found: {name}")

            # Parse the face locations logic and scale the values
            y1, x2, y2, x1 = face_location
            x1, y1 = x1*4, y1*4
            x2, y2 = x2*4, y2*4

            # Generate the bbox variable
            bbox = [x1, y1, x2-x1, y2-y1]

            # Draw the bbox for matching faces
            cvzone.cornerRect(img, bbox, rt=0)



    cv2.imshow("Face Attendance", img)
    cv2.waitKey(1) 